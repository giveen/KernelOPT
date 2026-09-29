//! Backend abstraction: which kind of CUDA kernel tree we optimize, and the
//! uniform `Target` that every discovery path produces.
//!
//! A `Target` is everything the pipeline needs to optimize one kernel file:
//! the editable files, the CMake build targets, the correctness filter, and the
//! timing authority. `ninfer` and `llamacpp` differ only in how those fields are
//! filled and which runner commands interpret them.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// ninfer: CMake + ctest + `ninfer_<op>_bench`.
    Ninfer,
    /// llama.cpp: CMake + `test-backend-ops test|perf`.
    Llamacpp,
    /// Any CUDA repo: gates declared in `kernelopt.toml`.
    Custom,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Ninfer => "ninfer",
            Backend::Llamacpp => "llamacpp",
            Backend::Custom => "custom",
        }
    }

    pub fn parse(s: &str) -> Result<Backend> {
        match s.to_ascii_lowercase().as_str() {
            "ninfer" => Ok(Backend::Ninfer),
            "llamacpp" | "llama.cpp" | "llama" => Ok(Backend::Llamacpp),
            "custom" | "generic" | "cuda" => Ok(Backend::Custom),
            other => anyhow::bail!("unknown backend {other:?} (use ninfer|llamacpp|custom|auto)"),
        }
    }

    /// Detect the backend from repo markers.
    pub fn detect(repo: &Path) -> Option<Backend> {
        if repo.join("kernelopt.toml").is_file() {
            Some(Backend::Custom)
        } else if repo.join("src/ops").is_dir() && repo.join("include/ninfer/ops").is_dir() {
            Some(Backend::Ninfer)
        } else if repo.join("ggml/src/ggml-cuda").is_dir() {
            Some(Backend::Llamacpp)
        } else {
            None
        }
    }
}

/// Resolve `--mode auto|ninfer|llamacpp` against a repo path.
pub fn resolve_backend(repo: &Path, requested: Option<&str>) -> Result<Backend> {
    match requested {
        None => detect_or_err(repo),
        Some(s) if s.eq_ignore_ascii_case("auto") => detect_or_err(repo),
        Some(s) => Backend::parse(s),
    }
}

fn detect_or_err(repo: &Path) -> Result<Backend> {
    Backend::detect(repo).with_context(|| {
        format!(
            "could not auto-detect backend at {} (expected ninfer src/ops or llama.cpp ggml/src/ggml-cuda; pass --mode)",
            repo.display()
        )
    })
}

/// One optimizable kernel target, backend-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub backend: Backend,
    /// Token used to address the target (`add_bias`, `fp8_linear_add`, `SOFT_MAX`).
    pub op: String,
    /// Grouping key (`add_bias`, `linear_add`, `SOFT_MAX`).
    pub family: String,
    /// Quant/branch variant for ninfer (`fp8`, `bf16`); None otherwise.
    pub variant: Option<String>,
    /// Editable kernel files (relative to repo).
    pub kernel_files: Vec<String>,
    /// Read-only context files (plan/dispatch/launcher/wrapper).
    pub context_files: Vec<String>,
    /// Contract headers (semantic authority).
    pub contract_files: Vec<String>,
    /// The single file the Executor edits (relative to repo).
    pub target_file: String,
    /// CMake targets to build for Gate 1.
    pub build_targets: Vec<String>,
    /// ninfer: ctest names; llamacpp: `-o` op filters.
    pub test_filters: Vec<String>,
    /// ninfer: the test translation units (for the measured-shape gate).
    #[serde(default)]
    pub test_sources: Vec<String>,
    /// ninfer: bench binary target; llamacpp: None (uses `test-backend-ops perf`).
    pub bench_binary: Option<String>,
    /// Extra argv forwarded to the bench invocation.
    pub bench_args: Vec<String>,
    /// Whether a timing authority exists (Gate 4 possible).
    pub timing: bool,
    /// Non-fatal problems worth surfacing.
    pub warnings: Vec<String>,
    /// CMake configure flags for this target's project (custom backends set
    /// their own; built-ins default to `Backend::configure_args`).
    #[serde(default)]
    pub configure_args: Vec<String>,
    /// Explicit Gate-2 command (custom). `{repo}`/`{build}` are substituted.
    #[serde(default)]
    pub test_cmd: Option<Vec<String>>,
    /// Explicit Gate-4 command (custom). `{repo}`/`{build}`/`{csv}` substituted.
    #[serde(default)]
    pub bench_cmd: Option<Vec<String>>,
    /// Bench output format for custom targets: `csv` | `stdout` | `llama`.
    #[serde(default)]
    pub bench_format: Option<String>,
}

impl Target {
    /// Simple targets (fewer editable files) first — a sane campaign order.
    pub fn complexity(&self) -> usize {
        self.kernel_files.len().max(1)
    }
}

/// Discover every runnable target in a repo.
pub fn discover_targets(repo: &Path, backend: Backend) -> Result<Vec<Target>> {
    let mut targets = match backend {
        Backend::Ninfer => crate::ninfer::discover_targets(repo)?,
        Backend::Llamacpp => crate::llamacpp::discover_targets(repo)?,
        Backend::Custom => crate::custom::discover_targets(repo)?,
    };
    targets.sort_by(|a, b| {
        a.complexity()
            .cmp(&b.complexity())
            .then_with(|| a.op.cmp(&b.op))
    });
    Ok(targets)
}

/// Discover a single target by op token.
pub fn discover_target(repo: &Path, backend: Backend, op: &str) -> Result<Target> {
    match backend {
        Backend::Ninfer => crate::ninfer::discover_target(repo, op),
        Backend::Llamacpp => crate::llamacpp::discover_target(repo, op),
        Backend::Custom => crate::custom::discover_target(repo, op),
    }
}

/// ninfer configure flags (cmake `-D...`).
pub fn ninfer_configure_args() -> Vec<String> {
    vec![
        "-DCMAKE_BUILD_TYPE=Release".into(),
        "-DBUILD_TESTING=ON".into(),
        "-DNINFER_BUILD_BENCHMARKS=ON".into(),
    ]
}

/// llama.cpp configure flags.
pub fn llamacpp_configure_args() -> Vec<String> {
    vec![
        "-DCMAKE_BUILD_TYPE=Release".into(),
        "-DGGML_CUDA=ON".into(),
        "-DLLAMA_BUILD_TESTS=ON".into(),
        "-DLLAMA_BUILD_EXAMPLES=ON".into(),
    ]
}

impl Backend {
    pub fn configure_args(&self) -> Vec<String> {
        match self {
            Backend::Ninfer => ninfer_configure_args(),
            Backend::Llamacpp => llamacpp_configure_args(),
            // Custom projects declare their own in `kernelopt.toml`.
            Backend::Custom => vec!["-DCMAKE_BUILD_TYPE=Release".into()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_parse() {
        assert_eq!(Backend::parse("ninfer").unwrap(), Backend::Ninfer);
        assert_eq!(Backend::parse("llama.cpp").unwrap(), Backend::Llamacpp);
        assert!(Backend::parse("nope").is_err());
    }
}
