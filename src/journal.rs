//! Append-only JSONL journal: every pipeline event is durable, enabling
//! `kernelopt resume` to skip completed stages.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    RunStarted {
        run_id: String,
        model_file: String,
        provider: String,
        model: String,
    },
    StageCompleted {
        stage: String,
        #[serde(flatten)]
        data: Value,
    },
    /// A long phase began — emitted so `--watch`/`status` show progress during
    /// otherwise-silent work (cold builds, NCU profiling).
    StageStarted {
        stage: String,
        note: String,
    },
    CandidateEvaluated {
        iteration: u32,
        chain: u32,
        plan: String,
        passed: bool,
        latency_ms: Option<f64>,
        error: Option<String>,
        /// Git commit of the candidate in the worktree (if committed).
        #[serde(default)]
        commit: Option<String>,
        /// Executor's summary of the change.
        #[serde(default)]
        change_summary: Option<String>,
        /// Planner's implementation hints.
        #[serde(default)]
        hints: Option<String>,
        /// Planner's evidence for the expected speedup.
        #[serde(default)]
        evidence: Option<String>,
    },
    LlmCall {
        agent: String,
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    /// A single Executor attempt that failed a gate (before a retry).
    AttemptFailed {
        iteration: u32,
        chain: u32,
        attempt: u32,
        /// compile | correctness | bench | no_tool_call | other
        category: String,
        error: String,
    },
    MemoryUpdated {
        action: String,
        direction: Option<String>,
    },
    GatesVerdict {
        stage: String, // "e2e"
        passed: bool,
        detail: Value,
    },
    RunFinished {
        outcome: String, // optimized | matched | fallback
        speedup: Option<f64>,
        root_cause: Option<String>,
        /// Why the loop stopped (patience/target_reached/max_iterations/budget/
        /// interrupted/baseline/no_candidate/e2e_rejected/…). `#[serde(default)]`
        /// keeps older journals without the field parseable.
        #[serde(default)]
        stop_reason: Option<String>,
    },
}

pub struct Journal {
    path: PathBuf,
    file: File,
}

impl Journal {
    pub fn create(runs_dir: &Path, run_id: &str) -> Result<Self> {
        let dir = runs_dir.join(run_id);
        std::fs::create_dir_all(&dir).context("creating run dir")?;
        let path = dir.join("journal.jsonl");
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self { path, file })
    }

    pub fn record(&mut self, event: &Event) -> Result<()> {
        // Add a timestamp without changing the event enum (replay ignores extras).
        let mut value = serde_json::to_value(event)?;
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "ts".into(),
                serde_json::json!(chrono::Utc::now().to_rfc3339()),
            );
        }
        writeln!(self.file, "{}", serde_json::to_string(&value)?)?;
        self.file.flush()?;
        Ok(())
    }

    /// Replay all events (for resume + reports).
    pub fn replay(runs_dir: &Path, run_id: &str) -> Result<Vec<Event>> {
        let path = runs_dir.join(run_id).join("journal.jsonl");
        if !path.exists() {
            anyhow::bail!("no journal at {}", path.display());
        }
        let f = File::open(&path)?;
        let mut events = Vec::new();
        for line in BufReader::new(f).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            events.push(serde_json::from_str(&line).with_context(|| {
                format!("bad journal line in {}: {}", path.display(), truncate(&line, 120))
            })?);
        }
        Ok(events)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

/// Collapse whitespace to one line, preserving the full text.
fn flatten(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One-line rendering of a journal event (full text, no truncation of plans/errors).
pub fn format_event(e: &Event) -> String {
    match e {
        Event::RunStarted { model_file, provider, model, .. } => {
            format!("▶ start {model_file} via {provider}/{model}")
        }
        Event::StageCompleted { stage, data } => {
            format!("· stage {stage}: {}", truncate(&flatten(&data.to_string()), 200))
        }
        Event::StageStarted { stage, note } => {
            format!("… {stage}: {note}")
        }
        Event::AttemptFailed { iteration, chain, attempt, category, error } => {
            format!("  i{iteration}/c{chain} a{attempt} ✗ {category}: {}", flatten(error))
        }
        Event::CandidateEvaluated { iteration, chain, passed, latency_ms, plan, error, commit, .. } => {
            let mut s = format!(
                "  i{iteration}/c{chain} {} {}",
                if *passed { "PASS" } else { "FAIL" },
                latency_ms.map(|m| format!("{m:.4} ms")).unwrap_or_default()
            );
            if let Some(c) = commit {
                s.push_str(&format!(" [{}]", &c[..c.len().min(10)]));
            }
            s.push_str(&format!(" — {}", flatten(plan)));
            if let Some(err) = error {
                s.push_str(&format!("\n      error: {}", flatten(err)));
            }
            s
        }
        Event::LlmCall { agent, prompt_tokens, completion_tokens } => {
            format!("  llm {agent}: {prompt_tokens}+{completion_tokens} tok")
        }
        Event::MemoryUpdated { action, direction } => {
            format!("  memory {action}: {}", direction.clone().unwrap_or_default())
        }
        Event::GatesVerdict { stage, passed, .. } => {
            format!("· gates {stage}: {}", if *passed { "PASS" } else { "REJECT" })
        }
        Event::RunFinished { outcome, speedup, root_cause, stop_reason, .. } => format!(
            "■ finished {outcome} {} {}{}",
            speedup.map(|s| format!("{s:.3}x")).unwrap_or_default(),
            root_cause.clone().unwrap_or_default(),
            stop_reason
                .as_deref()
                .map(|s| format!(" [{s}]"))
                .unwrap_or_default()
        ),
    }
}

/// Follow a journal file, printing new events to stderr, until `stop` is set.
/// Used by `--watch` so the run renders its own live view in one terminal.
pub fn tail_journal(path: &Path, stop: &std::sync::atomic::AtomicBool) {
    let mut offset = 0u64;
    loop {
        if let Ok(text) = std::fs::read_to_string(path) {
            let bytes = text.len() as u64;
            if bytes > offset {
                let tail = &text[offset as usize..];
                if let Some(last_nl) = tail.rfind('\n') {
                    for line in tail[..last_nl].lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(e) = serde_json::from_str::<Event>(line) {
                            eprintln!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), format_event(&e));
                            if let Event::RunFinished { .. } = e {
                                if let Some(dir) = path.parent() {
                                    eprintln!("  report:  {}", dir.join("report.md").display());
                                    eprintln!("  diff:    {}", dir.join("report.diff").display());
                                }
                            }
                        }
                    }
                    offset += (last_nl + 1) as u64;
                }
            }
        }
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("ko_journal_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut j = Journal::create(&dir, "r1").unwrap();
        j.record(&Event::StageCompleted {
            stage: "trace".into(),
            data: serde_json::json!({"classification": "optimizable"}),
        })
        .unwrap();
        j.record(&Event::RunFinished {
            outcome: "optimized".into(),
            speedup: Some(1.42),
            root_cause: None,
            stop_reason: Some("max_iterations".into()),
        })
        .unwrap();
        let events = Journal::replay(&dir, "r1").unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Event::StageCompleted { ref stage, .. } if stage == "trace"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
