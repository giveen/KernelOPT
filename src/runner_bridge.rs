//! Bridge to the Python GPU runner: one stateless subprocess per command,
//! JSON request on stdin, JSON response on stdout (protocol v1).
//!
//! Optional wrapper command (e.g. `graphsignal-run --listen-port N`) launches
//! the runner under the Graphsignal sidecar so bench commands can read their
//! own /signals payload before exit.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerError {
    pub kind: String, // compile | runtime | correctness | timeout | protocol
    pub message: String,
    pub traceback: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerResponse {
    pub ok: bool,
    #[serde(default)]
    pub error: Option<RunnerError>,
    #[serde(flatten)]
    pub payload: serde_json::Value,
    pub protocol: Option<u32>,
}

pub struct RunnerBridge {
    pub runner_dir: std::path::PathBuf,
    pub python: String,
    /// Prefix command wrapping the runner process (graphsignal-run …).
    pub wrapper: Vec<String>,
}

impl RunnerBridge {
    pub fn new(runner_dir: std::path::PathBuf) -> Self {
        Self {
            runner_dir,
            python: "python3".into(),
            wrapper: Vec::new(),
        }
    }

    /// A bridge that launches the runner under `graphsignal-run` (CUPTI
    /// sidecar) so `bench` can embed a /signals attribution payload.
    pub fn wrapped(
        runner_dir: std::path::PathBuf,
        listen_port: u16,
        cuda_graph_trace: &str,
    ) -> Self {
        Self {
            wrapper: vec![
                "graphsignal-run".into(),
                "--listen-port".into(),
                listen_port.to_string(),
                "--cuda-graph-trace".into(),
                cuda_graph_trace.to_string(),
            ],
            runner_dir,
            python: "python3".into(),
        }
    }

    /// Execute a runner command. Returns the full response payload.
    /// `ok=false` responses are NOT errors here — gate failures are data.
    /// Transport failures (spawn/timeout/malformed JSON) return Err.
    pub fn call(&self, request: &serde_json::Value) -> Result<serde_json::Value> {
        self.call_with_timeout(request, 900)
    }

    pub fn call_with_timeout(
        &self,
        request: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<serde_json::Value> {
        let cmd_name = request
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();

        // File-based response channel: robust against pipe truncation when
        // the runner dies during CUDA/CUPTI teardown. stdout is the fallback.
        let resp_path = std::env::current_dir()
            .context("cwd")?
            .join(".kernelopt/tmp")
            .join(format!(
                "runner_resp_{}.json",
                uuid::Uuid::new_v4().simple()
            ));
        if let Some(parent) = resp_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut request = request.clone();
        request["response_file"] = serde_json::json!(resp_path.to_string_lossy().to_string());
        let input = serde_json::to_string(&request).context("serializing runner request")?;        // Wrapper (e.g. graphsignal-run …) prefixes the python invocation.
        let (exe, args): (&str, Vec<&str>) = if self.wrapper.is_empty() {
            (&self.python, vec!["-m", "kernelopt_runner"])
        } else {
            let mut a: Vec<&str> = self.wrapper[1..].iter().map(|s| s.as_str()).collect();
            a.push(&self.python);
            a.push("-m");
            a.push("kernelopt_runner");
            (self.wrapper[0].as_str(), a)
        };
        let mut command = Command::new(exe);
        command.args(&args);
        // Put the runner in its own process group so a terminal Ctrl-C reaches
        // only KernelOPT (which pauses gracefully) and not the build mid-flight.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command
            .current_dir(&self.runner_dir)
            .env("PYTHONPATH", &self.runner_dir)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning runner for command {cmd_name:?}"))?;

        // Feed stdin on a detached handle; drop closes the pipe.
        {
            let mut stdin = child.stdin.take().context("runner stdin")?;
            stdin.write_all(input.as_bytes())?;
        }

        let output = wait_with_timeout(child, timeout_secs, &cmd_name)?;

        // 1) Authoritative: the response file (atomic write by the runner).
        if let Ok(text) = std::fs::read_to_string(&resp_path) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                let _ = std::fs::remove_file(&resp_path);
                if !output.status.success() {
                    eprintln!(
                        "[runner] {cmd_name} exited {:?} but produced a valid response (stderr: {})",
                        output.status.code(),
                        truncate(&String::from_utf8_lossy(&output.stderr), 500)
                    );
                }
                return Ok(parsed);
            }
        }

        // 2) Fallback: stdout parse.
        if !output.status.success() && output.stdout.is_empty() {
            anyhow::bail!(
                "runner crashed for {cmd_name:?} (exit {:?}): {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let json_start = stdout.find('{').context(format!(
            "no JSON in runner stdout for {cmd_name:?}: {}",
            truncate(&stdout, 400)
        ))?;
        let parsed: serde_json::Value =
            serde_json::from_str(stdout[json_start..].trim()).with_context(|| {
                format!(
                    "malformed runner response for {cmd_name:?}: {}",
                    truncate(&stdout, 400)
                )
            })?;
        Ok(parsed)
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

fn wait_with_timeout(
    mut child: std::process::Child,
    timeout_secs: u64,
    cmd: &str,
) -> Result<std::process::Output> {
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait()? {
            Some(_status) => {
                return Ok(child.wait_with_output()?);
            }
            None => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    anyhow::bail!("runner command {cmd:?} timed out after {timeout_secs}s");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Ensure the runner dir exists relative to the project root.
pub fn locate_runner(start: &Path) -> std::path::PathBuf {
    let candidate = start.join("runner");
    if candidate.exists() {
        return candidate;
    }
    start.join("runner")
}
