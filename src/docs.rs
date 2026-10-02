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

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

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
    // 1) stdio MCP server (auto: AMD's hip-docs-mcp when ROCm is present).
    if let Some(cmd) = mcp_cmd() {
        if let Some(text) = stdio_mcp_lookup(symbol, &cmd) {
            return Some(text);
        }
    }
    // 2) HTTP MCP docs server (NVIDIA cuda-docs, or any configured endpoint).
    let token = std::env::var("KERNELOPT_DOCS_TOKEN")
        .ok()
        .or_else(|| std::env::var("KERNELOPT_CUDA_DOCS_TOKEN").ok())
        .filter(|s| !s.trim().is_empty());
    if let Some(token) = token {
        if let Some(text) = mcp_lookup(symbol, &token) {
            return Some(text);
        }
    }
    // 3) Local CUDA/CCCL (and ROCm, when present) headers.
    local_lookup(symbol)
}

/// stdio MCP command: explicit `KERNELOPT_DOCS_MCP_CMD`, else AMD's
/// `hip-docs-mcp` when ROCm is detected and it is installed.
fn mcp_cmd() -> Option<String> {
    if let Ok(v) = std::env::var("KERNELOPT_DOCS_MCP_CMD") {
        if !v.trim().is_empty() {
            return Some(v);
        }
    }
    if rocm_present() && command_on_path("hip-docs-mcp") {
        return Some("hip-docs-mcp".to_string());
    }
    None
}

/// Drive a stdio MCP server: initialize → tools/list → tools/call. Best-effort;
/// the process is killed (whole group) after a timeout so a hung server can't
/// stall a run.
fn stdio_mcp_lookup(symbol: &str, cmd: &str) -> Option<String> {
    let mut command = Command::new("sh");
    command.arg("-c").arg(cmd);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let pid = child.id() as i32;
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(30));
        #[cfg(unix)]
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    });
    let mut stdin = child.stdin.take()?;
    let stdout = child.stdout.take()?;
    let mut reader = BufReader::new(stdout);

    let read_id = |reader: &mut BufReader<std::process::ChildStdout>, id: i64| -> Option<serde_json::Value> {
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).ok()? == 0 {
                return None;
            }
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(t) {
                if v.get("id").and_then(|i| i.as_i64()) == Some(id) {
                    return Some(v);
                }
            }
        }
    };

    let out = (|| {
        send_json(&mut stdin, &serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "kernelopt", "version": "0"}}
        }))?;
        read_id(&mut reader, 1)?;
        let _ = writeln!(stdin, "{}", serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        let _ = stdin.flush();
        send_json(&mut stdin, &serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))?;
        let tools = read_id(&mut reader, 2)?;
        let tool = tools["result"]["tools"]
            .as_array()?
            .iter()
            .find(|t| {
                let n = t["name"].as_str().unwrap_or("").to_ascii_lowercase();
                n.contains("doc") || n.contains("search") || n.contains("hip") || n.contains("lookup")
            })
            .and_then(|t| t["name"].as_str())?
            .to_string();
        for key in ["query", "q", "text", "symbol", "prompt"] {
            send_json(&mut stdin, &serde_json::json!({
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": tool, "arguments": {key: symbol}}
            }))?;
            if let Some(text) = read_id(&mut reader, 3).and_then(|r| mcp_text(&r)) {
                return Some(text.chars().take(EXCERPT_CHARS).collect());
            }
        }
        None
    })();

    let _ = child.kill();
    let _ = child.wait();
    out
}

/// The most likely offending identifier in a compiler diagnostic — only when it
/// is plausibly a CUDA/CUB API symbol. Generic tokens (`lambda`, `unsigned`,
/// user types) yield `None` so we never inject a noise lookup.
pub fn symbol_from_error(msg: &str) -> Option<String> {
    for quoted in quoted_strings(msg) {
        let ident = base_ident(&quoted);
        if confident_symbol(&ident) {
            return Some(ident);
        }
    }
    None
}

/// A namespaced name, a `__` intrinsic, or a known CUDA builtin prefix.
fn confident_symbol(s: &str) -> bool {
    if s.len() < 4 {
        return false;
    }
    if s.contains("::") {
        return true;
    }
    const PREFIXES: &[&str] = &[
        "__",
        "atomic",
        "make_",
        "tex",
        "surface",
        "wgmma",
        "tcgen05",
        "cuda",
        "cooperative_groups",
        // ROCm / HIP (used only when the ROCm docs source is enabled).
        "hip",
        "rocblas",
        "hipblas",
        "hipblaslt",
        "rocwmma",
        "wmma::",
        "amdgcn",
        "gfx",
    ];
    PREFIXES.iter().any(|p| s.starts_with(p))
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
    for dir in include_dirs() {
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

fn include_dirs() -> Vec<PathBuf> {
    let mut dirs = cuda_include_dirs();
    if rocm_enabled() {
        dirs.extend(rocm_include_dirs());
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

/// ROCm/HIP docs are **auto-enabled when ROCm is detected** (hipcc, rocm-smi,
/// or a ROCm install); force with `KERNELOPT_DOCS_ROCM=1`, or disable
/// explicitly with `=0`/`off`.
fn rocm_enabled() -> bool {
    match std::env::var("KERNELOPT_DOCS_ROCM").ok().as_deref() {
        Some("0") | Some("false") | Some("off") | Some("no") => false,
        Some(_) => true,
        None => rocm_present(),
    }
}

/// Is a ROCm/HIP toolchain present on this host?
fn rocm_present() -> bool {
    for env in ["ROCM_PATH", "HIP_PATH"] {
        if std::env::var(env).map(|v| !v.trim().is_empty()).unwrap_or(false) {
            return true;
        }
    }
    ["hipcc", "rocm-smi", "rocminfo", "amdgpu-arch"]
        .iter()
        .any(|t| command_on_path(t))
        || Path::new("/opt/rocm").is_dir()
}

fn command_on_path(tool: &str) -> bool {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|d| d.join(tool).exists()))
        .unwrap_or(false)
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

fn rocm_include_dirs() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for env in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(v) = std::env::var(env) {
            if !v.trim().is_empty() {
                roots.push(PathBuf::from(v));
            }
        }
    }
    roots.push(PathBuf::from("/opt/rocm"));
    if let Ok(rd) = std::fs::read_dir("/opt") {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("rocm") {
                roots.push(e.path());
            }
        }
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    for r in roots {
        dirs.push(r.join("include"));
        dirs.push(r.join("include").join("hip"));
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
    let url = std::env::var("KERNELOPT_DOCS_URL")
        .or_else(|_| std::env::var("KERNELOPT_CUDA_DOCS_URL"))
        .unwrap_or_else(|_| {
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

fn send_json(stdin: &mut impl Write, v: &serde_json::Value) -> Option<()> {
    writeln!(stdin, "{v}").ok()?;
    stdin.flush().ok()
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
        assert_eq!(
            symbol_from_error("error: no instance of overloaded function \"atomicAdd\" matches"),
            Some("atomicAdd".into())
        );
        // Non-symbols must not trigger a lookup.
        assert_eq!(symbol_from_error("error: expression must be a modifiable lvalue"), None);
        assert_eq!(
            symbol_from_error("error: function \"lambda [](int, unsigned int)\" cannot be referenced"),
            None
        );
        assert_eq!(symbol_from_error("error: identifier \"kFooBar\" is undefined"), None);
    }

    #[test]
    fn extracts_hip_symbols() {
        assert_eq!(
            symbol_from_error("error: no matching function for call to \"hipMalloc\" ..."),
            Some("hipMalloc".into())
        );
        assert_eq!(
            symbol_from_error("error: \"rocwmma::load_matrix_sync\" was not declared"),
            Some("rocwmma::load_matrix_sync".into())
        );
        assert_eq!(
            symbol_from_error("error: use of undeclared identifier \"__builtin_amdgcn_sched_group_barrier\""),
            Some("__builtin_amdgcn_sched_group_barrier".into())
        );
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

    const FAKE_MCP: &str = r#"
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        m = json.loads(line)
    except Exception:
        continue
    method = m.get("method")
    if method == "initialize":
        print(json.dumps({"jsonrpc":"2.0","id":m["id"],"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}}), flush=True)
    elif method == "tools/list":
        print(json.dumps({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":"search_docs","description":"","inputSchema":{"type":"object"}}]}}), flush=True)
    elif method == "tools/call":
        q = m.get("params", {}).get("arguments", {}).get("query", "")
        print(json.dumps({"jsonrpc":"2.0","id":m["id"],"result":{"content":[{"type":"text","text":"DOCS:" + q}]}}), flush=True)
"#;

    #[test]
    fn stdio_mcp_client_reads_docs() {
        if Command::new("python3").arg("--version").output().is_err() {
            return; // no python3; skip
        }
        let dir = std::env::temp_dir().join(format!("kopt-mcp-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake.py");
        std::fs::write(&script, FAKE_MCP).unwrap();
        let cmd = format!("python3 {}", script.display());
        let out = stdio_mcp_lookup("hipMalloc", &cmd);
        assert!(out.as_deref().unwrap_or("").contains("hipMalloc"), "{out:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
