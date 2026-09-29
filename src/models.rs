//! Local model discovery for the engine-E2E (Gate 3) check.
//!
//! Gate 3 runs a *local* model on the baseline and candidate engine builds, so
//! you must point it at an artifact. This module finds the artifacts actually
//! on the machine — so you can pick one by name instead of a full path, or say
//! `auto`.
//!
//! Search order (no machine-specific paths): `KERNELOPT_MODELS_DIR`
//! (colon-separated), `<repo>/models`, `./models`, `~/models`.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, PartialEq)]
pub struct LocalModel {
    pub path: PathBuf,
    pub name: String,
    /// `ninfer` | `llamacpp` | `hf`
    pub engine: String,
    pub size_bytes: u64,
    pub modified: Option<SystemTime>,
}

/// Directories to scan, in order.
pub fn search_dirs(repo: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(spec) = std::env::var("KERNELOPT_MODELS_DIR") {
        for p in spec.split(':').filter(|s| !s.trim().is_empty()) {
            dirs.push(PathBuf::from(crate::dotenv::expand_tilde(p.trim())));
        }
    }
    dirs.push(repo.join("models"));
    dirs.push(PathBuf::from("models"));
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join("models"));
    }
    dirs
}

/// Engine inferred from the artifact (mirrors the runner's `detect_engine`).
pub fn engine_of(path: &Path) -> Option<&'static str> {
    let s = path.to_string_lossy();
    if s.ends_with(".ninfer") {
        return Some("ninfer");
    }
    if s.ends_with(".gguf") {
        return Some("llamacpp");
    }
    if s.ends_with(".safetensors") {
        return Some("hf");
    }
    if path.is_dir() {
        let hf = std::fs::read_dir(path)
            .ok()
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    n.ends_with(".safetensors") || n == "config.json"
                })
            })
            .unwrap_or(false);
        if hf {
            return Some("hf");
        }
    }
    None
}

fn size_of(path: &Path) -> u64 {
    if path.is_file() {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    std::fs::read_dir(path)
        .ok()
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Discover local model artifacts, newest first. `backend` filters by engine
/// (`ninfer`/`llamacpp`); `hf` models are matched for both.
pub fn discover(repo: &Path, backend: Option<&str>) -> Vec<LocalModel> {
    let mut out: Vec<LocalModel> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for dir in search_dirs(repo) {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.filter_map(|e| e.ok()) {
            let path = entry.path();
            let Some(engine) = engine_of(&path) else {
                continue;
            };
            if let Some(b) = backend {
                if b != "auto" && engine != b && engine != "hf" {
                    continue;
                }
            }
            let canon = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if !seen.insert(canon.clone()) {
                continue;
            }
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let modified = entry.metadata().ok().and_then(|m| m.modified().ok());
            out.push(LocalModel {
                path: canon,
                name,
                engine: engine.to_string(),
                size_bytes: size_of(&path),
                modified,
            });
        }
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

/// Resolve a `--e2e-weights` spec: an existing path, `auto`, or a discovered
/// model name (exact stem, then substring).
pub fn resolve(spec: &str, repo: &Path, backend: Option<&str>) -> Result<PathBuf> {
    let expanded = PathBuf::from(crate::dotenv::expand_tilde(spec));
    if spec.eq_ignore_ascii_case("auto") {
        return discover(repo, backend)
            .into_iter()
            .next()
            .map(|m| m.path)
            .context(
                "no local models found for `--e2e-weights auto` — set KERNELOPT_MODELS_DIR \
                 or pass a path/name",
            );
    }
    if expanded.exists() {
        return Ok(expanded);
    }
    let models = discover(repo, backend);
    if let Some(m) = models.iter().find(|m| m.name == spec) {
        return Ok(m.path.clone());
    }
    if let Some(m) = models.iter().find(|m| m.name.contains(spec)) {
        return Ok(m.path.clone());
    }
    let hint = models
        .iter()
        .take(8)
        .map(|m| m.name.clone())
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::bail!(
        "model {spec:?} not found — pass a path, `auto`, or a discovered name{}",
        if hint.is_empty() {
            " (none found; set KERNELOPT_MODELS_DIR)".to_string()
        } else {
            format!(": {hint}{}", if models.len() > 8 { ", …" } else { "" })
        }
    )
}

/// Human-readable size (e.g. `21.0 GiB`).
pub fn human_size(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.0} MiB", b / MIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_detection_by_extension() {
        assert_eq!(engine_of(Path::new("m.ninfer")), Some("ninfer"));
        assert_eq!(engine_of(Path::new("m.gguf")), Some("llamacpp"));
        assert_eq!(engine_of(Path::new("m.safetensors")), Some("hf"));
        assert_eq!(engine_of(Path::new("m.bin")), None);
    }

    #[test]
    fn discovers_and_resolves_by_name() {
        let root = std::env::temp_dir().join(format!("kopt-models-{}", uuid::Uuid::new_v4()));
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("qwen3_8_27b.ninfer"), b"x").unwrap();
        std::fs::write(models.join("qwen3_8_27b_nvfp4.ninfer"), b"xx").unwrap();
        std::fs::write(models.join("notes.txt"), b"nope").unwrap();
        std::env::set_var("KERNELOPT_MODELS_DIR", &models);
        // Isolate from the developer's real repo/models and ~/models.
        let repo = root.join("empty-repo");
        std::fs::create_dir_all(&repo).unwrap();

        let found = discover(&repo, Some("ninfer"));
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found.iter().all(|m| m.engine == "ninfer"));

        // Exact stem wins over substring.
        let p = resolve("qwen3_8_27b_nvfp4", &repo, Some("ninfer")).unwrap();
        assert!(p.ends_with("qwen3_8_27b_nvfp4.ninfer"), "{}", p.display());
        // Substring fallback.
        let p = resolve("nvfp4", &repo, Some("ninfer")).unwrap();
        assert!(p.ends_with("qwen3_8_27b_nvfp4.ninfer"), "{}", p.display());
        // An existing path is used verbatim.
        let explicit = models.join("qwen3_8_27b.ninfer");
        assert_eq!(resolve(explicit.to_str().unwrap(), &repo, Some("ninfer")).unwrap(), explicit);
        // `auto` returns something.
        assert!(resolve("auto", &repo, Some("ninfer")).is_ok());
        // Unknown name errors with a hint.
        let err = resolve("does-not-exist", &repo, Some("ninfer")).unwrap_err().to_string();
        assert!(err.contains("qwen3_8_27b"), "{err}");

        std::env::remove_var("KERNELOPT_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn human_size_formats() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(1024 * 1024), "1 MiB");
        assert_eq!(human_size(21 * 1024 * 1024 * 1024), "21.0 GiB");
    }
}
