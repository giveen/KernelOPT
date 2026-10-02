//! Optional structural code map via codebase-memory-mcp.
//!
//! codebase-memory-mcp is a single static binary (MIT, zero deps) that indexes
//! a repo into a knowledge graph (tree-sitter, 66 languages) and answers
//! structural queries over stdio MCP: who calls this function, what does it
//! call, impact of a diff. KernelOPT uses it to show the Planner/Executor the
//! *callers* of the kernel being edited — so a signature change that would
//! break an out-of-scope caller is visible before it is proposed.
//!
//! Absent binary (or `KERNELOPT_CODEMAP=0`) = disabled; every caller falls back
//! to the existing grep-based context.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

const TIMEOUT_INDEX_SECS: u64 = 300;
const TIMEOUT_QUERY_SECS: u64 = 60;

/// A live handle: command to spawn + indexed project name.
pub struct CodeMap {
    cmd: String,
    pub project: String,
}

/// Command to spawn, or `None` when the map is disabled.
pub fn detect_cmd() -> Option<String> {
    match std::env::var("KERNELOPT_CODEMAP").ok().as_deref() {
        Some("0") | Some("false") | Some("off") | Some("no") => return None,
        _ => {}
    }
    if let Ok(v) = std::env::var("KERNELOPT_CODEMAP_CMD") {
        if !v.trim().is_empty() {
            return Some(v);
        }
    }
    command_on_path("codebase-memory-mcp").then(|| "codebase-memory-mcp".to_string())
}

fn command_on_path(tool: &str) -> bool {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|d| d.join(tool).exists()))
        .unwrap_or(false)
}

/// Index `repo` (fast mode) and return a handle, or `None` when unavailable.
pub fn open(repo: &Path) -> Option<CodeMap> {
    let cmd = detect_cmd()?;
    let project = index_repo(
        &cmd,
        &serde_json::json!({"repo_path": repo.to_string_lossy(), "mode": "fast"}),
    )?;
    Some(CodeMap { cmd, project })
}

fn index_repo(cmd: &str, args: &serde_json::Value) -> Option<String> {
    let res = mcp_call(cmd, "index_repository", args, TIMEOUT_INDEX_SECS)?;
    let text = mcp_text(&res)?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()?
        .get("project")?
        .as_str()
        .map(String::from)
}

impl CodeMap {
    fn call(&self, tool: &str, args: &serde_json::Value) -> Option<serde_json::Value> {
        mcp_call(&self.cmd, tool, args, TIMEOUT_QUERY_SECS)
    }

    /// `(qualified_name, file, lines)` for top symbols defined in `file`.
    pub fn symbols_in_file(&self, file: &str, limit: usize) -> Vec<(String, String, String)> {
        let stem = Path::new(file)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| file.to_string());
        let res = match self.call(
            "search_graph",
            &serde_json::json!({
                "project": self.project, "file_pattern": stem,
                "format": "json", "limit": limit.max(10),
            }),
        ) {
            Some(r) => r,
            None => return Vec::new(),
        };
        graph_rows(&res)
            .into_iter()
            .filter_map(|r| {
                let qn = r.get("qn")?.as_str()?.to_string();
                let f = r.get("file")?.as_str()?.to_string();
                let lines = r.get("lines")?.as_str().unwrap_or("").to_string();
                // Only symbols actually defined in this file.
                (f == file || f.ends_with(file)).then_some((qn, f, lines))
            })
            .collect()
    }

    /// Short qualified names of direct callers of `symbol` (inbound, depth 1).
    pub fn callers(&self, symbol: &str, limit: usize) -> Vec<String> {
        let res = match self.call(
            "trace_path",
            &serde_json::json!({
                "project": self.project, "function_name": symbol,
                "direction": "inbound", "depth": 1, "limit": limit.max(20),
            }),
        ) {
            Some(r) => r,
            None => return Vec::new(),
        };
        let Some(text) = mcp_text(&res) else {
            return Vec::new();
        };
        parse_caller_lines(&text, &self.project)
    }

    /// "fn ← caller" summary for the main symbols of `file`, for the planner.
    pub fn caller_context(&self, file: &str) -> Option<String> {
        let syms = self.symbols_in_file(file, 12);
        if syms.is_empty() {
            return None;
        }
        let mut lines: Vec<String> = Vec::new();
        for (qn, _f, _l) in syms.iter().take(4) {
            let short = short_name(qn);
            for caller in self.callers(&short, 8).into_iter().take(5) {
                lines.push(format!("{short} <- {caller}"));
                if lines.len() >= 10 {
                    break;
                }
            }
            if lines.len() >= 10 {
                break;
            }
        }
        (!lines.is_empty()).then_some(lines.join("\n"))
    }
}

/// Last `::` segment, without template args: `a::b::Fn<T>` → `Fn`.
pub fn short_name(qn: &str) -> String {
    let base = qn.rsplit("::").next().unwrap_or(qn);
    let cut = base
        .find(|c: char| c == '<' || c == '(' || c == ' ')
        .unwrap_or(base.len());
    base[..cut].to_string()
}

/// Caller lines from a `trace_path` tree response, minus the project prefix.
fn parse_caller_lines(text: &str, project: &str) -> Vec<String> {
    let mut in_callers = false;
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("callers:") {
            in_callers = true;
            continue;
        }
        if !in_callers || t.is_empty() {
            continue;
        }
        // Rows look like `<qn> <hop>`; skip headers/footers.
        let mut parts = t.rsplitn(2, ' ');
        let hop = parts.next().unwrap_or("");
        let qn = parts.next().unwrap_or("");
        if qn.is_empty() || !hop.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            if t.starts_with("callers_total") || t.starts_with("callers:") || t.contains("cols:") {
                continue;
            }
            // Unknown line inside the section: stop rather than misread.
            if out.is_empty() {
                continue;
            }
            break;
        }
        let qn = qn.strip_prefix(project).unwrap_or(qn);
        let qn = qn.strip_prefix('.').unwrap_or(qn);
        out.push(qn.to_string());
    }
    out
}

/// Rows of a `search_graph` JSON response as maps.
fn graph_rows(res: &serde_json::Value) -> Vec<std::collections::HashMap<String, serde_json::Value>> {
    let text = match mcp_text(res) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let (Some(cols), Some(rows)) = (
        v.get("cols").and_then(|c| c.as_array()),
        v.get("rows").and_then(|r| r.as_array()),
    ) else {
        return Vec::new();
    };
    let names: Vec<String> = cols
        .iter()
        .filter_map(|c| c.as_str().map(String::from))
        .collect();
    rows.iter()
        .filter_map(|r| r.as_array())
        .map(|r| {
            names
                .iter()
                .zip(r.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .collect()
}

// --------------------------------------------------------------------------- //
// minimal stdio MCP client (initialize -> call, process-group kill on timeout)
// --------------------------------------------------------------------------- //

fn mcp_call(cmd: &str, tool: &str, args: &serde_json::Value, timeout_secs: u64) -> Option<serde_json::Value> {
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
        std::thread::sleep(Duration::from_secs(timeout_secs));
        #[cfg(unix)]
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    });
    let mut stdin = child.stdin.take()?;
    let stdout = child.stdout.take()?;
    let mut reader = BufReader::new(stdout);

    let out = (|| {
        send_json(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "kernelopt", "version": "0"}}
            }),
        )?;
        read_id(&mut reader, 1)?;
        let _ = writeln!(
            stdin,
            "{}",
            serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
        );
        let _ = stdin.flush();
        send_json(
            &mut stdin,
            &serde_json::json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool, "arguments": args}
            }),
        )?;
        read_id(&mut reader, 2)
    })();

    let _ = child.kill();
    let _ = child.wait();
    out
}

fn send_json(stdin: &mut impl Write, v: &serde_json::Value) -> Option<()> {
    writeln!(stdin, "{v}").ok()?;
    stdin.flush().ok()
}

fn read_id(reader: &mut BufReader<std::process::ChildStdout>, id: i64) -> Option<serde_json::Value> {
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
    fn short_name_strips_templates() {
        assert_eq!(short_name("a::b::Fn<T>"), "Fn");
        assert_eq!(short_name("add_bias_bf16x8_kernel"), "add_bias_bf16x8_kernel");
    }

    #[test]
    fn parses_trace_caller_lines() {
        let text = "function: add_bias_bf16x8_kernel\ndirection: inbound\ncallers_total: 2\ncallers_total_relation: eq\ncallers: 2  (cols: qn hop)\n  mnt-storage-Projects-ninfer-ext.src.ops.launcher.add_bias.ninfer::ops::detail.add_bias_launch 1\n  mnt-storage-Projects-ninfer-ext.src.ops.wrapper.add_bias.ninfer::ops.add_bias 2\n";
        let out = parse_caller_lines(text, "mnt-storage-Projects-ninfer-ext");
        assert_eq!(out.len(), 2);
        assert!(out[0].contains("add_bias_launch"));
        assert!(out[1].ends_with("ninfer::ops.add_bias"));
    }

    #[test]
    fn empty_trace_yields_no_callers() {
        assert!(parse_caller_lines("function: x\ndirection: inbound\ncallers_total: 0\n", "p").is_empty());
    }
}
