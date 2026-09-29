//! Generic CUDA backend: optimize **any** repo that declares its gates in
//! `kernelopt.toml`. No code changes per project — ninfer and llama.cpp are
//! just built-in presets of the same idea.
//!
//! ```toml
//! [project]
//! name = "my-engine"
//! configure_args = ["-DCMAKE_BUILD_TYPE=Release", "-DMY_BUILD_TESTS=ON"]
//!
//! [[target]]
//! op = "add_bias"
//! file = "src/ops/add_bias.cu"
//! context_files  = ["src/ops/launcher.cu"]      # read-only (dispatch/launcher)
//! contract_files = ["include/add_bias.h"]       # semantic authority
//! build_targets  = ["my_add_bias_test", "my_add_bias_bench"]
//! test_cmd    = ["ctest", "--test-dir", "{build}", "-R", "my_add_bias_test"]
//! bench_cmd   = ["{build}/bench/my_add_bias_bench", "--csv-out", "{csv}"]
//! bench_format = "csv"                          # csv | stdout | llama
//! ```
//!
//! Placeholders: `{repo}`, `{build}`, `{csv}`.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Default)]
struct Descriptor {
    #[serde(default)]
    project: Project,
    #[serde(default)]
    target: Vec<TargetSpec>,
}

#[derive(Debug, Deserialize, Default)]
struct Project {
    #[allow(dead_code)]
    name: Option<String>,
    #[serde(default)]
    configure_args: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct TargetSpec {
    op: String,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    kernel_files: Vec<String>,
    #[serde(default)]
    context_files: Vec<String>,
    #[serde(default)]
    contract_files: Vec<String>,
    #[serde(default)]
    build_targets: Vec<String>,
    #[serde(default)]
    test_cmd: Option<Vec<String>>,
    #[serde(default)]
    bench_cmd: Option<Vec<String>>,
    #[serde(default)]
    bench_format: Option<String>,
    #[serde(default = "yes")]
    timing: bool,
}

fn yes() -> bool {
    true
}

fn load(repo: &Path) -> Result<Descriptor> {
    let path = repo.join("kernelopt.toml");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {} (is this a custom-backend repo?)", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn discover_targets(repo: &Path) -> Result<Vec<crate::backend::Target>> {
    let d = load(repo)?;
    if d.target.is_empty() {
        bail!("kernelopt.toml has no [[target]] entries");
    }
    d.target
        .iter()
        .map(|t| to_target(repo, &d.project, t))
        .collect()
}

pub fn discover_target(repo: &Path, op: &str) -> Result<crate::backend::Target> {
    discover_targets(repo)?
        .into_iter()
        .find(|t| t.op == op)
        .with_context(|| format!("op {op:?} not in kernelopt.toml"))
}

fn to_target(repo: &Path, project: &Project, t: &TargetSpec) -> Result<crate::backend::Target> {
    let kernel_files = if t.kernel_files.is_empty() {
        t.file.clone().map(|f| vec![f]).unwrap_or_default()
    } else {
        t.kernel_files.clone()
    };
    let target_file = t
        .file
        .clone()
        .or_else(|| kernel_files.first().cloned())
        .with_context(|| format!("target {:?} needs `file` or `kernel_files`", t.op))?;
    if !repo.join(&target_file).exists() {
        bail!("target {:?}: file not found: {}", t.op, target_file);
    }
    let configure_args = if project.configure_args.is_empty() {
        crate::backend::Backend::Custom.configure_args()
    } else {
        project.configure_args.clone()
    };
    Ok(crate::backend::Target {
        backend: crate::backend::Backend::Custom,
        op: t.op.clone(),
        family: t.op.clone(),
        variant: None,
        kernel_files,
        context_files: t.context_files.clone(),
        contract_files: t.contract_files.clone(),
        target_file,
        build_targets: t.build_targets.clone(),
        test_filters: vec![],
        test_sources: vec![],
        bench_binary: None,
        bench_args: vec![],
        timing: t.timing,
        warnings: vec![],
        configure_args,
        test_cmd: t.test_cmd.clone(),
        bench_cmd: t.bench_cmd.clone(),
        bench_format: t.bench_format.clone(),
    })
}

/// Expand `{repo}` / `{build}` / `{csv}` placeholders in a declared command.
pub fn expand(argv: &[String], repo: &Path, build: &Path, csv: Option<&Path>) -> Vec<String> {
    let repo_s = repo.to_string_lossy().to_string();
    let build_s = build.to_string_lossy().to_string();
    let csv_s = csv.map(|c| c.to_string_lossy().to_string()).unwrap_or_default();
    argv.iter()
        .map(|a| {
            a.replace("{repo}", &repo_s)
                .replace("{build}", &build_s)
                .replace("{csv}", &csv_s)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_expands_a_descriptor() {
        let root = std::env::temp_dir().join(format!("kopt-custom-{}", uuid::Uuid::new_v4()));
        let src = root.join("src/ops");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("add_bias.cu"), b"// k\n").unwrap();
        std::fs::write(
            root.join("kernelopt.toml"),
            r#"
[project]
name = "my-engine"
configure_args = ["-DMY_TESTS=ON"]

[[target]]
op = "add_bias"
file = "src/ops/add_bias.cu"
context_files = ["src/ops/launcher.cu"]
build_targets = ["my_add_bias_test", "my_add_bias_bench"]
test_cmd = ["ctest", "--test-dir", "{build}", "-R", "my_add_bias_test"]
bench_cmd = ["{build}/bench/my_add_bias_bench", "--csv-out", "{csv}"]
bench_format = "csv"
"#,
        )
        .unwrap();

        let targets = discover_targets(&root).unwrap();
        assert_eq!(targets.len(), 1);
        let t = &targets[0];
        assert_eq!(t.backend, crate::backend::Backend::Custom);
        assert_eq!(t.op, "add_bias");
        assert_eq!(t.target_file, "src/ops/add_bias.cu");
        assert_eq!(t.context_files, vec!["src/ops/launcher.cu"]);
        assert_eq!(t.configure_args, vec!["-DMY_TESTS=ON"]);
        assert!(t.test_cmd.is_some() && t.bench_cmd.is_some());

        let argv = expand(
            t.bench_cmd.as_ref().unwrap(),
            &root,
            &root.join("build"),
            Some(Path::new("/tmp/x.csv")),
        );
        assert_eq!(argv[0], format!("{}/build/bench/my_add_bias_bench", root.display()));
        assert_eq!(argv[2], "/tmp/x.csv");

        let _ = std::fs::remove_dir_all(&root);
    }
}
