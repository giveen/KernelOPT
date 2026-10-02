//! Best-effort CUDA/CUB API docs lookup, to explain compile failures.
//!
//! The dominant failure mode on new arch/toolchains is *API availability and
//! calling convention* (`no instance of overloaded function "__reduce_max_sync"`,
//! `cub::WarpMergeSort` has a deleted constructor, …). When a candidate fails to
//! compile, we look the offending symbol up and attach a short excerpt to the
//! executor's retry so the model can fix it instead of re-guessing.
//!
//! Sources, in order:
//!   1. An MCP docs server (opt-in: `KERNELOPT_CUDA_DOCS_TOKEN`, URL via
//!      `KERNELOPT_CUDA_DOCS_URL`). Best-effort; any failure falls through.
//!   2. The **local CUDA/CCCL headers** — authoritative for "does this symbol
//!      exist and how is it declared", offline, and always available.

use std::path::{Path, PathBuf};
use std::process::Command;

const EXCERPT_CHARS: usize = 1400;

/// Explain a compiler error: look up its offending symbol and return a short
/// docs excerpt, or `None` when nothing useful is found.
pub fn explain(error: &str) -> Option<String> {
    let sym = symbol_from_error(error)?;
    lookup(&sym).map(|text| {
        let head = format!("CUDA DOCS for `{sym}` (verify availability/arguments for THIS arch/toolchain):");
        format!("{head}\n{text}")
    })
}

/// Look a symbol up in the configured docs sources.
pub fn lookup(symbol: &str) -> Option<String> {
    let token = std::env::var("KERNELOPT_CUDA_DOCS_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty());
    if let Some(token) = token {
        if let Some(text) = mcp_lookup(symbol, &token) {
            return Some(text);
        }
    }
    local_lookup(symbol)
}

/// The most likely offending identifier in a compiler diagnostic.
pub fn symbol_from_error(msg: &str) -> Option<String> {
    let mut preferred: Option<String> = None;
    let mut fallback: Option<String> = None;
    for quoted in quoted_strings(msg) {
        let ident = base_ident(&quoted);
        if ident.len() < 3 {
            continue;
        }
        if ident.contains("::") || ident.starts_with("__") || ident.starts_with("cub") {
            preferred.get_or_insert(ident);
        } else {
            fallback.get_or_insert(ident);
        }
    }
    preferred.or(fallback)
}

/// Identifiers inside double quotes, e.g. `"__reduce_max_sync"` or
/// `"cub::WarpMergeSort<…>"`.
fn quoted_strings(msg: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = msg;
    while let Some(i) = rest.find('"') {
        let after = &rest[i + 1..];
        let Some(j) = after.find('"') else { break };
        out.push(after[..j].to_string());
        rest = &after[j + 1..];
    }
    out
}

/// Trim a quoted blob down to a bare identifier (drop template args, pointers,
/// argument lists, and any trailing prose).
fn base_ident(s: &str) -> String {
    let s = s.trim();
    let cut = s
        .find(|c: char| c == '<' || c == '(' || c == '[' || c == ' ' || c == '*' || c == '&')
        .unwrap_or(s.len());
    s[..cut].trim().to_string()
}

// --------------------------------------------------------------------------- //
// local CUDA/CCCL headers
// --------------------------------------------------------------------------- //

fn local_lookup(symbol: &str) -> Option<String> {
    let base = base_ident(symbol.rsplit("::").next().unwrap_or(symbol));
    let base = base.trim_start_matches('~');
    if base.len() < 3 {
        return None;
    }
    for dir in cuda_include_dirs() {
        let files = grep_files(&dir, base);
        if files.is_empty() {
            continue;
        }
        // Prefer a file named for the symbol (declaration over a passing mention).
        let snake = camel_to_snake(base.rsplit("::").next().unwrap_or(base));
        let chosen = files
            .iter()
            .find(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().contains(&snake))
                    .unwrap_or(false)
            })
            .unwrap_or(&files[0]);
        if let Some(ctx) = header_context(chosen, base) {
            return Some(ctx);
        }
    }
    None
}

/// `grep -rl` for `base` under `dir` (bounded), returning matching files.
fn grep_files(dir: &Path, base: &str) -> Vec<PathBuf> {
    let mut cmd = Command::new("grep");
    cmd.arg("-rl")
        .arg("--include=*.h")
        .arg("--include=*.hpp")
        .arg("--include=*.cuh")
        .arg("--include=*.inl")
        .arg("-m1")
        .arg(base)
        .arg(dir);
    let Ok(o) = crate::exec::run_capture(&mut cmd, 20) else {
        return Vec::new();
    };
    o.stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(64)
        .map(PathBuf::from)
        .collect()
}

/// `WarpMergeSort` → `warp_merge_sort` (header filenames are snake_case).
fn camel_to_snake(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// A short excerpt around the first mention of `base`: the declaration plus its
/// contiguous doc comment.
fn header_context(path: &Path, base: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    let idx = lines.iter().position(|l| l.contains(base))?;
    let start = comment_start(&lines, idx);
    let end = (idx + 5).min(lines.len());
    let excerpt: String = lines[start..end]
        .join("\n")
        .chars()
        .take(EXCERPT_CHARS)
        .collect();
    Some(format!("// {}:{}\n{}", path.display(), start + 1, excerpt))
}

fn comment_start(lines: &[&str], idx: usize) -> usize {
    let mut s = idx;
    while s > 0 {
        let prev = lines[s - 1].trim();
        let is_comment = prev.starts_with("///")
            || prev.starts_with("//!")
            || prev.starts_with("//")
            || prev.starts_with('*')
            || prev.starts_with("/*")
            || prev.ends_with("*/");
        if is_comment {
            s -= 1;
        } else {
            break;
        }
    }
    s
}

fn cuda_include_dirs() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for env in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(v) = std::env::var(env) {
            if !v.trim().is_empty() {
                roots.push(PathBuf::from(v));
            }
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    if let Ok(rd) = std::fs::read_dir("/usr/local") {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("cuda") {
                roots.push(e.path());
            }
        }
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    for r in roots {
        dirs.push(r.join("include"));
        if let Ok(rd) = std::fs::read_dir(r.join("targets")) {
            for e in rd.flatten() {
                dirs.push(e.path().join("include"));
            }
        }
    }
    dirs.retain(|d| d.is_dir());
    dirs.sort();
    dirs.dedup();
    dirs
}

// --------------------------------------------------------------------------- //
// optional MCP docs server (best-effort, opt-in)
// --------------------------------------------------------------------------- //

fn mcp_lookup(symbol: &str, token: &str) -> Option<String> {
    let url = std::env::var("KERNELOPT_CUDA_DOCS_URL").unwrap_or_else(|_| {
        "https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs".to_string()
    });
    let http = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .ok()?;
    let call = |body: serde_json::Value| -> Option<serde_json::Value> {
        let resp = http
            .post(&url)
            .bearer_auth(token)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&body)
            .send()
            .ok()?;
        let text = resp.text().ok()?;
        // Streamable HTTP may answer as SSE (`data: {json}`) or plain JSON.
        let json = text
            .lines()
            .find_map(|l| l.strip_prefix("data:").map(str::trim))
            .unwrap_or(text.trim());
        serde_json::from_str(json).ok()
    };
    call(serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "kernelopt", "version": "0"}}
    }))?;
    let tools = call(serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))?;
    let tool = tools["result"]["tools"]
        .as_array()?
        .iter()
        .find(|t| {
            let n = t["name"].as_str().unwrap_or("").to_ascii_lowercase();
            n.contains("doc") || n.contains("search") || n.contains("cuda") || n.contains("lookup")
        })
        .and_then(|t| t["name"].as_str())?
        .to_string();
    // The argument name is unknown ahead of time; try the common ones.
    for key in ["query", "q", "text", "symbol", "prompt"] {
        let res = call(serde_json::json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": tool, "arguments": {key: symbol}}
        }))?;
        if let Some(text) = mcp_text(&res) {
            return Some(text.chars().take(EXCERPT_CHARS).collect());
        }
    }
    None
}

fn mcp_text(res: &serde_json::Value) -> Option<String> {
    let content = res["result"]["content"].as_array()?;
    let joined: String = content
        .iter()
        .filter_map(|c| c["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    (!joined.trim().is_empty()).then_some(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_symbols_from_diagnostics() {
        assert_eq!(
            symbol_from_error("error: no instance of overloaded function \"__reduce_max_sync\" matches the argument list"),
            Some("__reduce_max_sync".into())
        );
        assert_eq!(
            symbol_from_error("error: the default constructor of \"cub::_V_3::WarpMergeSort<unsigned long long, 2, 32>\" cannot be referenced"),
            Some("cub::_V_3::WarpMergeSort".into())
        );
        assert_eq!(symbol_from_error("error: expression must be a modifiable lvalue"), None);
    }

    #[test]
    fn base_ident_strips_templates_and_qualifiers() {
        assert_eq!(base_ident("cub::WarpMergeSort<unsigned long long, 2>"), "cub::WarpMergeSort");
        assert_eq!(base_ident("unsigned long long *"), "unsigned");
    }

    #[test]
    fn comment_start_walks_back_over_doc_lines() {
        let lines = vec!["/// docs", "/// more", "struct Foo {", "};"];
        assert_eq!(comment_start(&lines, 2), 0);
        assert_eq!(comment_start(&lines, 3), 3);
    }

    #[test]
    fn camel_to_snake_matches_header_names() {
        assert_eq!(camel_to_snake("WarpMergeSort"), "warp_merge_sort");
        assert_eq!(camel_to_snake("BlockScan"), "block_scan");
        assert_eq!(camel_to_snake("cub"), "cub");
    }
}
