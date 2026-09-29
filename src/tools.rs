//! Retrieval helpers for the Planner: ripgrep-backed search and bounded file
//! reads over the repo/worktree, so a large kernel file does not have to be
//! dumped wholesale into the prompt.
//!
//! Uses `rg` when present (fast, respects `.gitignore`); otherwise a simple
//! line scan. All reads are confined to the given root.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_SEARCH_CHARS: usize = 6000;
const MAX_READ_LINES: usize = 240;

fn rg_available() -> bool {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|d| d.join("rg").is_file()))
        .unwrap_or(false)
}

/// Search a repo/worktree for `query`. `glob` filters filenames (rg syntax).
pub fn search(root: &Path, query: &str, glob: Option<&str>, max: usize) -> Result<String> {
    let max = max.clamp(1, 200);
    if rg_available() {
        let mut cmd = Command::new("rg");
        cmd.args(["--line-number", "--no-heading", "--color=never", "--max-columns", "200"])
            .arg("--max-count")
            .arg(max.to_string());
        if let Some(g) = glob {
            cmd.arg("--glob").arg(g);
        }
        cmd.arg("--").arg(query).arg(root);
        let out = cmd.output().context("running rg")?;
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        if text.trim().is_empty() {
            return Ok(format!("(no matches for {query:?})"));
        }
        return Ok(cap(&strip_prefix(&text, root), MAX_SEARCH_CHARS));
    }
    naive_search(root, query, max)
}

/// Read lines `[start, end]` (1-based, inclusive) of `rel` under `root`.
pub fn read_region(root: &Path, rel: &str, start: usize, end: usize) -> Result<String> {
    let path = safe_join(root, rel)?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let lines: Vec<&str> = text.lines().collect();
    let start = start.max(1);
    let end = end.min(lines.len()).min(start + MAX_READ_LINES);
    if start > lines.len() {
        return Ok(format!("(file has {} lines)", lines.len()));
    }
    let mut out = format!("{rel}:{start}-{end}\n");
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        out.push_str(&format!("{:>5}  {}\n", start + i, line));
    }
    Ok(cap(&out, MAX_SEARCH_CHARS))
}

/// A compact outline of a source file: kernel/device/template/struct lines.
pub fn outline(root: &Path, rel: &str) -> Result<String> {
    let path = safe_join(root, rel)?;
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let total = text.lines().count();
    let mut out = format!("{rel} — {total} lines; outline:\n");
    for (i, line) in text.lines().enumerate() {
        let t = line.trim_start();
        if t.starts_with("__global__")
            || t.starts_with("__device__")
            || t.starts_with("__forceinline__")
            || t.starts_with("template")
            || t.contains("launch_bounds")
            || t.starts_with("struct ")
            || t.starts_with("using ")
            || t.starts_with("namespace ")
        {
            out.push_str(&format!("{:>5}  {}\n", i + 1, cap(line.trim(), 160)));
        }
    }
    Ok(cap(&out, MAX_SEARCH_CHARS))
}

fn safe_join(root: &Path, rel: &str) -> Result<PathBuf> {
    let candidate = root.join(rel);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("no such file: {}", candidate.display()))?;
    let root_canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if !canonical.starts_with(&root_canon) {
        anyhow::bail!("path escapes the repo: {rel}");
    }
    Ok(canonical)
}

fn strip_prefix(text: &str, root: &Path) -> String {
    let prefix = format!("{}/", root.to_string_lossy());
    text.replace(&prefix, "")
}

fn cap(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…[truncated]", s.chars().take(n).collect::<String>())
    }
}

fn naive_search(root: &Path, query: &str, max: usize) -> Result<String> {
    let q = query.to_ascii_lowercase();
    let mut hits = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if hits.len() >= max {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name == ".git" || name == "build" || name == ".kernelopt" {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !matches!(ext, "cu" | "cuh" | "cpp" | "h" | "hpp" | "py") {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&p) {
                for (i, line) in text.lines().enumerate() {
                    if line.to_ascii_lowercase().contains(&q) {
                        hits.push(format!("{}:{}: {}", p.display(), i + 1, line.trim()));
                        if hits.len() >= max {
                            break;
                        }
                    }
                }
            }
        }
    }
    if hits.is_empty() {
        Ok(format!("(no matches for {query:?})"))
    } else {
        Ok(cap(&hits.join("\n"), MAX_SEARCH_CHARS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_region_is_bounded_and_numbered() {
        let dir = std::env::temp_dir().join(format!("ko_tools_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("k.cu");
        std::fs::write(&file, "__global__ void a(){}\nint b;\n__device__ int c(){}\n").unwrap();
        let out = read_region(&dir, "k.cu", 2, 3).unwrap();
        assert!(out.contains("int b;"));
        assert!(out.contains("__device__ int c(){}"));
        let outline = outline(&dir, "k.cu").unwrap();
        assert!(outline.contains("__global__ void a()"));
        assert!(!outline.contains("int b;"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_path_escape() {
        let dir = std::env::temp_dir();
        assert!(read_region(&dir, "../../etc/passwd", 1, 5).is_err());
    }

    #[test]
    fn search_finds_matching_lines() {
        let dir = std::env::temp_dir().join(format!("ko_search_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.cuh"), "int keep;\n__device__ int target_fn(){}\n").unwrap();
        let out = search(&dir, "target_fn", None, 10).unwrap();
        assert!(out.contains("target_fn"), "got: {out}");
        let none = search(&dir, "definitely_absent_xyz", None, 10).unwrap();
        assert!(none.contains("no matches"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
