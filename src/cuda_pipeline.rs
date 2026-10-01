//! CUDA kernel optimization pipeline, backend-agnostic.
//!
//! discover → isolated worktree + CMake baseline → loop
//! (plan → edit one kernel file → compile → correctness → bench) → perf gate →
//! diff. The same driver runs ninfer (ctest + `ninfer_<op>_bench`) and llama.cpp
//! (`test-backend-ops test|perf`) targets.
//!
//! Loop termination is configurable beyond the paper's fixed T:
//!   * `patience`     — stop after P consecutive non-improving iterations
//!   * `target_speedup` — stop once baseline/best ≥ X
//!   * `max_iterations` — hard safety cap
//!   * budget         — wall-clock deadline and/or LLM-call cap
//!
//! The user's checkout is never modified: edits land in a linked git worktree and
//! the surviving change is surfaced as a unified diff.

use crate::analyst;
use crate::backend::{Backend, Target};
use crate::config::Config;
use crate::journal::{Event, Journal};
use crate::llm::{LlmClient, ToolCall, ToolDef};
use crate::memory::{
    parse_summarizer_output, ExperienceItem, ExperienceMemory, MemoryUpdate, StrategyTracker,
};
use crate::runner_bridge::RunnerBridge;
use crate::search::{allocate_expansions, diverse_select_nodes, meltdown_detected, BeamNode, Candidate};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Inline the full target file only up to this many chars; larger files get an
/// outline and the Planner uses `read_file`/`search_repo` for detail.
const PLANNER_FULL_CHARS: usize = 20_000;
/// Max retrieval rounds before the Planner must submit a plan.
const MAX_TOOL_ROUNDS: usize = 6;
/// Warn when the Executor must reproduce a file larger than this (full-file
/// submission can exceed model output limits; consider a smaller `--kernel-file`).
const EXECUTOR_WARN_CHARS: usize = 40_000;

/// Correctness/verify timeout before the baseline calibrates it (seconds).
pub const DEFAULT_VERIFY_TIMEOUT_S: u64 = 1800;
/// A candidate's verify timeout = baseline verify time × this…
const VERIFY_TIMEOUT_MULT: u64 = 10;
/// …plus slack, clamped to [floor, ceil] seconds.
const VERIFY_TIMEOUT_FLOOR_S: u64 = 60;
const VERIFY_TIMEOUT_CEIL_S: u64 = 1800;

/// A candidate's correctness test should take roughly as long as the baseline's;
/// give generous slack, but never the flat default once the baseline is measured.
/// Guards against a runaway/hung candidate kernel burning the full timeout.
fn adaptive_verify_timeout(baseline: std::time::Duration) -> u64 {
    (baseline.as_secs() * VERIFY_TIMEOUT_MULT + 30)
        .clamp(VERIFY_TIMEOUT_FLOOR_S, VERIFY_TIMEOUT_CEIL_S)
}

/// First non-empty, truncated line of a failure — a stable-ish signature used
/// both to dedup the Planner's failure memory and to detect a repeated
/// (non-progressing) executor attempt.
fn error_signature(error: &str) -> String {
    let first = error.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    truncate_chars(first, 160)
}

/// `<category>: <signature>` as stored in the Planner's failure memory.
fn failure_signature(category: &str, error: &str) -> String {
    format!("{category}: {}", error_signature(error))
}

/// Last `n` chars of a field from the final bench run — used to surface the
/// real cause (crash text, CUDA error) when a bench yields no measurements.
fn last_run_tail(resp: &serde_json::Value, field: &str, n: usize) -> String {
    let s = resp["runs"]
        .as_array()
        .and_then(|r| r.last())
        .and_then(|r| r[field].as_str())
        .unwrap_or("")
        .trim();
    let chars: Vec<char> = s.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

pub struct CudaPipeline<'a> {
    pub cfg: &'a Config,
    pub llm: &'a dyn LlmClient,
    pub runner: &'a RunnerBridge,
    pub journal: &'a mut Journal,
    pub session_id: String,
    pub repo: PathBuf,
    pub target: Target,
    pub worktree: PathBuf,
    pub build_dir: PathBuf,
    /// Cross-process lockfile that serializes GPU work (bench/verify/ncu/e2e).
    pub gpu_lock_path: PathBuf,
    pub run_dir: PathBuf,
    pub ncu_set: String,
    /// How the Executor edits the file (full-file or unified diff).
    pub edit_mode: EditMode,
    /// Model-level (engine) E2E config; None disables Gate 3.
    pub e2e: Option<E2eConfig>,
    /// Baseline engine result captured at baseline for comparison.
    pub baseline_e2e: Option<E2eResult>,
    /// Base commit of the worktree (candidates are reset to this).
    pub base_sha: Option<String>,
    /// Pin the representative bench shape (substring of the bench row label).
    pub bench_shape: Option<String>,
    /// Interleaved baseline/candidate re-bench rounds at finalize.
    pub final_rounds: u32,
    /// Measurement noise (relative spread) from the most recent bench.
    pub last_bench_noise_pct: Option<f64>,
    /// Label of the shape used for the representative measurement.
    pub last_bench_label: Option<String>,
    /// Correctness/verify timeout in seconds, calibrated from the baseline on
    /// the first verify so a hung candidate can't burn the flat default.
    pub verify_timeout_s: std::cell::Cell<u64>,
    /// Compact signatures of recent failed attempts, fed to the Planner so it
    /// stops repeating dead ends (e.g. the same CUB API that won't compile).
    pub recent_failures: Vec<String>,
    /// Runtime-detected toolchain summary (compiler/arch) shown to the Planner.
    /// `None` until the baseline build has configured the tree.
    pub toolchain: Option<String>,
    /// Effective bandwidth (GB/s) of the most recent bench's representative shape.
    pub last_bench_gbs: Option<f64>,
    /// Device memory roofline (GB/s) reported by the most recent bench.
    pub last_bench_roofline_gbs: Option<f64>,
    pub memory: &'a mut ExperienceMemory,
    pub tracker: &'a mut StrategyTracker,
    // ---- loop policy ----
    pub patience: u32,
    pub target_speedup: f64,
    pub min_improvement: f64,
    pub max_iterations: u32,
    pub deadline: Option<Instant>,
    pub llm_call_budget: Option<u64>,
    // ---- stats (accumulated across the run) ----
    pub llm_calls: u64,
    pub tokens: u64,
    /// Per-run checkpoint path; written after each iteration so an interrupted
    /// target can resume without repeating completed work.
    pub checkpoint_path: Option<PathBuf>,
    /// Set by a Ctrl-C handler; the loop pauses at the next safe point.
    pub interrupted: Option<Arc<AtomicBool>>,
    /// Print live progress to stderr.
    pub verbose: bool,
    /// Tail the journal live (renders the `watch` view in this terminal).
    pub watch: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineResult {
    pub outcome: String,
    pub speedup: Option<f64>,
    pub baseline_ms: Option<f64>,
    pub final_ms: Option<f64>,
    pub iterations: u32,
    pub stop_reason: String,
    pub llm_calls: u64,
    pub tokens: u64,
    pub diff_path: Option<String>,
    pub root_cause: Option<String>,
    /// True when a budget pause left the target resumable (not finalized).
    pub paused: bool,
    /// Winner "what changed / why faster" summary (when a candidate won).
    pub summary: Option<serde_json::Value>,
}

/// How the Executor edits the file: full-file replacement or a unified diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditMode {
    /// Executor returns the complete file (reproduction drift possible).
    Full,
    /// Executor returns a unified diff applied with `git apply` (no drift).
    Patch,
}

impl EditMode {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "full" | "file" => Ok(EditMode::Full),
            "patch" | "diff" => Ok(EditMode::Patch),
            other => anyhow::bail!("unknown edit mode {other:?} (use full|patch)"),
        }
    }
}

/// A planner plan: what to change, how, and the evidence it cites.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlannedChange {
    pub change: String,
    #[serde(default)]
    pub hints: String,
    #[serde(default)]
    pub evidence: String,
}

/// Model-level (engine) E2E configuration: run the whole engine on the baseline
/// build and on the candidate build, then compare output and latency.
#[derive(Debug, Clone)]
pub struct E2eConfig {
    /// Model path: `.ninfer`, `.gguf`, or an HF safetensors directory.
    pub model: String,
    /// Engine override: ninfer | llamacpp | hf | vllm | sglang (default: detect).
    pub engine: Option<String>,
    /// Fully custom command (argv); bypasses engine dispatch.
    pub cmd: Option<Vec<String>>,
    pub prompt: String,
    pub max_new: u32,
    pub seed: u32,
}

/// A captured engine run (deterministic greedy generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2eResult {
    pub digest: String,
    pub elapsed_s: f64,
    pub text: String,
}

/// Per-run checkpoint (`.kernelopt/runs/<run_id>/checkpoint.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    op: String,
    backend: String,
    /// Next iteration index to run.
    iteration: u32,
    iterations_run: u32,
    best: Option<Candidate>,
    best_ms: f64,
    baseline_ms: f64,
    recent_directions: Vec<String>,
    patience_used: u32,
    llm_calls: u64,
    tokens: u64,
}

impl<'a> CudaPipeline<'a> {
    /// Run a GPU-touching runner command under the cross-process GPU lock, so
    /// only one bench/verify/ncu/e2e runs on the device at a time (also across
    /// separate `kernelopt` processes).
    fn gpu_call_timeout(
        &self,
        command: &str,
        mut req: serde_json::Value,
        secs: u64,
    ) -> Result<serde_json::Value> {
        req["command"] = json!(command);
        let _guard = crate::gpu_lock::lock(&self.gpu_lock_path)?;
        self.runner.call_with_timeout(&req, secs)
    }

    fn record_llm_usage(&mut self, agent: &str, c: &crate::llm::Completion) {
        self.llm_calls += 1;
        if let Some(u) = &c.usage {
            self.tokens += u.prompt_tokens + u.completion_tokens;
            let _ = self.journal.record(&Event::LlmCall {
                agent: agent.into(),
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
            });
        }
    }

    /// Remember a failed attempt (deduped, last few kept) so the next Planner
    /// call sees concrete dead ends instead of restating them. Returns `true`
    /// when this exactly repeats the previous *compile* failure — no progress,
    /// so the caller should stop retrying and save the rebuilds.
    fn record_failure(&mut self, category: &str, error: &str, prev: &mut Option<String>) -> bool {
        let sig = failure_signature(category, error);
        let repeated_compile = category == "compile" && prev.as_deref() == Some(sig.as_str());
        *prev = Some(sig.clone());
        if !self.recent_failures.contains(&sig) {
            self.recent_failures.push(sig);
            let n = self.recent_failures.len();
            if n > 6 {
                self.recent_failures.drain(0..n - 6);
            }
        }
        repeated_compile
    }

    fn budget_ok(&self) -> bool {
        if let Some(d) = self.deadline {
            if Instant::now() >= d {
                return false;
            }
        }
        if let Some(b) = self.llm_call_budget {
            if self.llm_calls >= b {
                return false;
            }
        }
        true
    }

    fn interrupted(&self) -> bool {
        self.interrupted
            .as_ref()
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Live progress line (stderr, so stdout stays machine-readable JSON).
    fn progress(&self, msg: impl AsRef<str>) {
        if self.verbose {
            eprintln!(
                "[{}] {}",
                chrono::Local::now().format("%H:%M:%S"),
                msg.as_ref()
            );
        }
    }

    /// Announce a long phase to both the live stderr log and the journal, so
    /// `--watch` (which suppresses `progress`) still shows what's happening
    /// during otherwise-silent work like cold builds and NCU profiling.
    fn stage_start(&mut self, stage: &str, note: impl AsRef<str>) {
        self.progress(format!("· {stage}: {}", note.as_ref()));
        let _ = self.journal.record(&Event::StageStarted {
            stage: stage.into(),
            note: note.as_ref().to_string(),
        });
    }

    fn target_path(&self) -> PathBuf {
        self.worktree.join(&self.target.target_file)
    }

    /// Contract/authority text (read from the worktree) the Executor must not
    /// violate. Empty for backends without a per-op contract (llama.cpp).
    fn authority_block(&self) -> String {
        let mut out = String::new();
        for rel in &self.target.contract_files {
            let path = self.worktree.join(rel);
            if let Ok(text) = std::fs::read_to_string(&path) {
                out.push_str(&format!("\n// {rel}\n{}\n", truncate_chars(&text, 2000)));
            }
            if out.chars().count() > 3000 {
                break;
            }
        }
        truncate_chars(&out, 3000)
    }

    fn render_prompt(&self, template: &str, vars: &serde_json::Value) -> Result<String> {
        let path = self.cfg.prompts_dir.join(template);
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading prompt template {}", path.display()))?;
        Ok(crate::prompts::render(&raw, vars))
    }

    // ---------- runner request builders (backend-specific) ----------

    fn compile_targets(&self) -> Vec<String> {
        let mut t = self.target.build_targets.clone();
        // The engine E2E gate runs app binaries built from the worktree.
        if self.e2e.is_some() {
            let engines: &[&str] = match self.target.backend {
                Backend::Ninfer => &["ninfer", "ninfer-perplexity", "ninfer_bench"],
                Backend::Llamacpp => &["llama-cli", "llama-perplexity", "llama-bench"],
                Backend::Custom => &[],
            };
            for b in engines {
                if !t.iter().any(|x| x == b) {
                    t.push(b.to_string());
                }
            }
        }
        t
    }

    /// Run the engine (deterministic greedy generation) and capture its result.
    fn run_engine(&self) -> Result<E2eResult> {
        let e2e = self.e2e.as_ref().context("no engine E2E config")?;
        let resp = self.gpu_call_timeout(
            "engine_generate",
            json!({
                "build_dir": self.build_dir,
                "model": e2e.model,
                "engine": e2e.engine,
                "cmd": e2e.cmd,
                "prompt": e2e.prompt,
                "max_new": e2e.max_new,
                "seed": e2e.seed,
                "timeout_s": 900,
            }),
            1000,
        )?;
        if resp["ok"] != json!(true) {
            anyhow::bail!(
                "engine_generate failed: {}",
                resp["error"]["message"].as_str().unwrap_or("unknown")
            );
        }
        Ok(E2eResult {
            digest: resp["digest"].as_str().unwrap_or_default().to_string(),
            elapsed_s: resp["elapsed_s"].as_f64().unwrap_or(0.0),
            text: resp["text"].as_str().unwrap_or_default().to_string(),
        })
    }

    // ---------- baseline ----------

    pub fn stage_baseline(&mut self) -> Result<f64> {
        let wt = crate::git::create(&self.repo, &self.worktree, &self.branch(), "HEAD")
            .context("worktree create")?;
        self.base_sha = wt["head"].as_str().map(|s| s.to_string());
        self.journal.record(&Event::StageCompleted {
            stage: "worktree".into(),
            data: json!({"worktree": self.worktree, "head": wt["head"], "reused": wt["reused"]}),
        })?;

        self.stage_start(
            "compile_baseline",
            "building baseline (cold builds can take minutes)",
        );
        let cfg = if let Some(bc) = &self.target.build_cmd {
            let argv = crate::custom::expand(bc, &self.worktree, &self.build_dir, None);
            crate::exec::run_argv(&argv, 3600).context("custom build")?
        } else {
            crate::exec::build(
                &self.worktree,
                &self.build_dir,
                &self.compile_targets(),
                true,
                true,
                &self.target.configure_args,
                num_cpus(),
                3600,
            )
            .context("cuda_compile (configure+build)")?
        };
        let cfg = crate::parse::compile_view(&cfg);
        self.journal.record(&Event::StageCompleted {
            stage: "compile_baseline".into(),
            data: json!({"passed": cfg["passed"], "errors": cfg["compiler_errors"]}),
        })?;
        if cfg["passed"] != json!(true) {
            anyhow::bail!("baseline build failed: {}", first_error(&cfg));
        }
        // Detect the host toolchain/arch for the Planner (never hardcoded).
        self.toolchain = crate::exec::detect_toolchain(&self.build_dir);

        let verify = {
            let t0 = Instant::now();
            let v = self.verify_now().context("verify (baseline)")?;
            if v["passed"] == json!(true) {
                // Calibrate candidate verify timeouts against the baseline.
                self.verify_timeout_s.set(adaptive_verify_timeout(t0.elapsed()));
            }
            v
        };
        self.journal.record(&Event::StageCompleted {
            stage: "verify_baseline".into(),
            data: json!({"passed": verify["passed"], "failing": verify["failing_cases"]}),
        })?;
        if verify["passed"] != json!(true) {
            anyhow::bail!("baseline correctness fails; refusing to optimize a red tree");
        }

        let baseline_ms = if self.target.timing {
            let ms = self.bench(None).context("baseline bench")?;
            self.journal.record(&Event::StageCompleted {
                stage: "bench_baseline".into(),
                data: json!({"median_ms": ms}),
            })?;
            ms
        } else {
            let _ = self.journal.record(&Event::StageCompleted {
                stage: "bench_baseline".into(),
                data: json!({"skipped": "no timing authority (correctness-only target)"}),
            });
            0.0
        };
        if self.e2e.is_some() {
            let res = self.run_engine().context("engine baseline (Gate 3)")?;
            self.journal.record(&Event::StageCompleted {
                stage: "engine_baseline".into(),
                data: json!({"digest": res.digest, "elapsed_s": res.elapsed_s, "text": res.text}),
            })?;
            self.baseline_e2e = Some(res);
        }
        Ok(baseline_ms)
    }

    /// Resolve the op-bench binary (absolute, else `<build>/bench/<name>`).
    fn bench_binary_path(&self) -> Result<PathBuf> {
        let b = self
            .target
            .bench_binary
            .as_deref()
            .context("no bench binary for this target")?;
        Ok(if Path::new(b).is_absolute() {
            PathBuf::from(b)
        } else {
            self.build_dir.join("bench").join(b)
        })
    }

    fn bench(&mut self, source: Option<&str>) -> Result<f64> {
        if let Some(src) = source {
            std::fs::write(self.target_path(), src).context("writing kernel into worktree")?;
        }
        let csv = self
            .run_dir
            .join("bench")
            .join(format!("{}.csv", uuid::Uuid::new_v4().simple()));
        // Direct invocation (GPU-serialized); parsing + aggregation in Rust.
        let resp = {
            let _guard = crate::gpu_lock::lock(&self.gpu_lock_path)?;
            match self.target.backend {
                Backend::Ninfer => {
                    let bin = self.bench_binary_path()?;
                    crate::exec::bench(
                        &bin,
                        &self.target.bench_args,
                        Some(&csv),
                        Some(20),
                        Some(100),
                        3,
                        1800,
                    )?
                }
                Backend::Llamacpp => {
                    let bin = self.build_dir.join("bin").join("test-backend-ops");
                    crate::exec::llama_perf(
                        &bin,
                        "CUDA0",
                        &self.target.test_filters,
                        &self.target.bench_args,
                        3,
                        1800,
                    )?
                }
                Backend::Custom => {
                    let bc = self
                        .target
                        .bench_cmd
                        .as_ref()
                        .context("custom target has no `bench_cmd` in kernelopt.toml")?;
                    let mut runs = Vec::new();
                    let mut last = None;
                    for _ in 0..3 {
                        let argv =
                            crate::custom::expand(bc, &self.worktree, &self.build_dir, Some(&csv));
                        let r = crate::exec::run_argv(&argv, 1800)?;
                        last = r["exit_code"].as_i64();
                        let csv_text = std::fs::read_to_string(&csv)
                            .ok()
                            .filter(|s| !s.trim().is_empty());
                        runs.push(json!({"stdout": r["raw_stdout"], "stderr": r["raw_stderr"], "csv": csv_text}));
                    }
                    json!({"ok": true, "exit_code": last, "runs": runs})
                }
            }
        };
        // Parse + aggregate in Rust (representative shape, noise, bandwidth).
        let parsed: Vec<crate::parse::ParsedBench> = resp["runs"]
            .as_array()
            .map(|runs| {
                runs.iter()
                    .map(|r| {
                        let stdout = r["stdout"].as_str().unwrap_or("");
                        let csv = r["csv"].as_str();
                        match self.target.bench_format.as_deref() {
                            Some("llama") => crate::parse::parse_bench_llama(stdout),
                            Some("csv") => {
                                crate::parse::parse_bench_csv(csv.unwrap_or(stdout))
                            }
                            Some("stdout") => crate::parse::parse_bench_stdout(stdout),
                            _ => match self.target.backend {
                                Backend::Llamacpp => crate::parse::parse_bench_llama(stdout),
                                _ => crate::parse::parse_bench_ninfer(stdout, csv),
                            },
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        let merged = crate::parse::merge_bench_runs(&parsed, self.bench_shape.as_deref());
        if resp["exit_code"].as_i64().unwrap_or(1) != 0 || merged.representative_us.is_none() {
            let rows: usize = parsed.iter().map(|p| p.rows.len()).sum();
            let code = resp["exit_code"].as_i64();
            let how = match code {
                Some(0) => "bench exited 0 but produced no parsable rows".to_string(),
                Some(c) => format!("bench exited {c}"),
                None => "bench terminated without an exit code (signal/crash?)".to_string(),
            };
            let mut msg = format!(
                "bench produced no usable measurements: {how}; {} run(s), {rows} row(s)",
                parsed.len()
            );
            if let Some(c) = resp["command"].as_str() {
                msg.push_str(&format!("\n  command: {c}"));
            }
            let out = last_run_tail(&resp, "stdout", 500);
            if !out.is_empty() {
                msg.push_str(&format!("\n  stdout tail: {out}"));
            }
            let err = last_run_tail(&resp, "stderr", 500);
            if !err.is_empty() {
                msg.push_str(&format!("\n  stderr tail: {err}"));
            }
            anyhow::bail!(msg);
        }
        self.last_bench_noise_pct = merged.noise_pct;
        self.last_bench_label = merged.representative_label.clone();
        self.last_bench_gbs = merged.representative_gbs;
        self.last_bench_roofline_gbs = merged.representative_roofline_gbs;
        let median_us = merged
            .representative_us
            .context("bench returned no representative_us")?;
        Ok(median_us / 1000.0)
    }

    pub fn stage_profile(&mut self) -> Option<serde_json::Value> {
        self.stage_start("profile", "profiling with NCU (this can take ~a minute)");
        let report = self
            .run_dir
            .join("ncu")
            .join(format!("{}.csv", uuid::Uuid::new_v4().simple()));
        let (target_argv, launch_skip) = match self.target.backend {
            Backend::Ninfer => {
                let bin = match self.bench_binary_path() {
                    Ok(b) => b,
                    Err(e) => return self.note_profile(&e.to_string()),
                };
                let mut v = vec![bin.to_string_lossy().to_string()];
                v.extend(self.target.bench_args.iter().cloned());
                (v, 5u32)
            }
            Backend::Llamacpp => {
                let bin = self.build_dir.join("bin").join("test-backend-ops");
                let mut v = vec![
                    bin.to_string_lossy().to_string(),
                    "perf".into(),
                    "-b".into(),
                    "CUDA0".into(),
                ];
                if !self.target.test_filters.is_empty() {
                    v.push("-o".into());
                    v.push(self.target.test_filters.join(","));
                }
                (v, 3u32)
            }
            Backend::Custom => {
                let Some(bc) = &self.target.bench_cmd else {
                    return self.note_profile("custom target has no `bench_cmd` to profile");
                };
                (
                    crate::custom::expand(bc, &self.worktree, &self.build_dir, None),
                    5u32,
                )
            }
        };
        let resp = {
            let _guard = match crate::gpu_lock::lock(&self.gpu_lock_path) {
                Ok(g) => g,
                Err(e) => return self.note_profile(&e.to_string()),
            };
            crate::exec::ncu(
                &target_argv,
                &self.ncu_set,
                Some(launch_skip),
                Some(1),
                None,
                1200,
            )
        };
        let resp = match resp {
            Ok(r) => r,
            Err(e) => return self.note_profile(&format!("{e:#}")),
        };
        {
            {
                // ncu CSV parsing lives in Rust now.
                let csv = crate::parse::extract_ncu_csv(resp["raw_stdout"].as_str().unwrap_or(""));
                let mut context = crate::parse::parse_ncu_csv(&csv);
                if context["kernels"]
                    .as_array()
                    .map(|a| a.is_empty())
                    .unwrap_or(true)
                {
                    let combined = format!(
                        "{}{}",
                        resp["raw_stdout"].as_str().unwrap_or(""),
                        resp["raw_stderr"].as_str().unwrap_or("")
                    );
                    let note = if combined.contains("ERR_NVGPUCTRPERM") {
                        "ERR_NVGPUCTRPERM: GPU performance counters are admin-only \
                         (RmProfilingAdminOnly=1). Set NVreg_RmProfilingAdminOnly=0 and reload."
                            .to_string()
                    } else {
                        let tail: String = combined
                            .chars()
                            .rev()
                            .take(600)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect();
                        format!(
                            "ncu produced no kernel data (exit {}): {tail}",
                            resp["exit_code"]
                        )
                    };
                    let _ = self.journal.record(&Event::StageCompleted {
                        stage: "profile".into(),
                        data: json!({"available": false, "note": note}),
                    });
                    return None;
                }
                if !csv.trim().is_empty() {
                    let _ = std::fs::write(&report, &csv);
                    context["raw_csv_path"] = json!(report.to_string_lossy());
                }
                context["ncu_command"] = json!(target_argv.join(" "));
                let ctx = analyst::planning_context(&context, 3);
                let _ = self.journal.record(&Event::StageCompleted {
                    stage: "profile".into(),
                    data: json!({"kernels": ctx["kernels"], "raw_csv": context["raw_csv_path"]}),
                });
                Some(ctx)
            }
        }
    }

    /// Journal a `profile` note (ncu unavailable) and return None.
    fn note_profile(&mut self, note: &str) -> Option<serde_json::Value> {
        let _ = self.journal.record(&Event::StageCompleted {
            stage: "profile".into(),
            data: json!({"available": false, "note": note}),
        });
        None
    }

    // ---------- plan ----------

    fn stage_plan(
        &mut self,
        kernel_source: &str,
        profiling_ctx: &serde_json::Value,
        diversity_hint: bool,
        recent_directions: &[String],
    ) -> Result<PlannedChange> {
        // Static system prompt so the request prefix is cacheable by the provider.
        let system = self.render_prompt("cuda-planner.md", &json!({}))?;
        let (context_label, context) = self.planner_context(kernel_source);
        // Bound the profiling context so it doesn't dominate the prompt.
        let ctx_str = truncate_chars(&profiling_ctx.to_string(), 1500);
        // Stable prefix first (target, contract, source/outline, profiling)…
        let mut user = format!(
            "BACKEND: {backend}\nOP: {op}  (family: {family}{variant})\n\
             TARGET KERNEL FILE (editable — the ONLY file you may change): {file}\n\
             ALL KERNEL FILES: {files:?}\n\
             READ-ONLY CONTEXT (launcher/dispatch/wrapper/plan): {context:?}\n\
             CONTRACT HEADER (read-only semantic authority — plan changes to the kernel only):\n{authority}\n\n\
             {label}:\n{context_block}\n\n\
             PROFILING CONTEXT:\n{ctx}\n",
            backend = self.target.backend.as_str(),
            op = self.target.op,
            family = self.target.family,
            variant = self.target.variant.as_deref().map(|v| format!(" / {v}")).unwrap_or_default(),
            file = self.target.target_file,
            files = self.target.kernel_files,
            context = self.target.context_files,
            authority = self.authority_block(),
            label = context_label,
            context_block = context,
            ctx = ctx_str,
        );
        // …variable context last, so it does not break prefix caching.
        user.push_str(&format!(
            "\n\nMEMORY (past attempts):\n{mem}\n\nKNOWN-BAD DIRECTIONS: {avoid}\n\n\
             RECENT DIRECTIONS (avoid repeats): {recent}\n\n{diversity}",
            mem = truncate_chars(&self.memory.context(), 1200),
            avoid = self.tracker.avoid_flags(&self.target.family).join(", "),
            recent = if recent_directions.is_empty() {
                "(none)".to_string()
            } else {
                recent_directions.join("; ")
            },
            diversity = if diversity_hint {
                "DIVERSITY ENFORCEMENT: recent plans collapsed to few unique approaches. \
                 Choose a DIFFERENT strategy family this time."
            } else {
                ""
            },
        ));
        if !self.recent_failures.is_empty() {
            user.push_str(&format!(
                "\n\nRECENT FAILURES (do NOT repeat these; if a direction is still right, \
                 fix the specific error rather than restating it):\n{}",
                self.recent_failures.join("\n")
            ));
        }
        if let Some(tc) = &self.toolchain {
            user.push_str(&format!(
                "\n\nTOOLCHAIN (detected at runtime on THIS host — target your edits to it; \
                 prefer primitives known to exist here over ones you assume are available):\n{tc}"
            ));
        }

        let tools = vec![
            ToolDef {
                name: "submit_plan".into(),
                description: "Submit one optimization plan for the Executor".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"plan": {"type": "object", "properties": {
                        "change": {"type": "string"},
                        "implementation_hints": {"type": "string"},
                        "evidence": {"type": "string"}
                    }, "required": ["change"]}},
                    "required": ["plan"]
                }),
            },
            ToolDef {
                name: "search_repo".into(),
                description: "ripgrep the repo for a symbol/pattern before planning".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "glob": {"type": "string", "description": "filename filter, e.g. *.cuh"},
                        "max": {"type": "integer"}
                    },
                    "required": ["query"]
                }),
            },
            ToolDef {
                name: "read_file".into(),
                description: "Read a line range of a repo file (1-based, inclusive)".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"}
                    },
                    "required": ["path"]
                }),
            },
        ];

        for _round in 0..MAX_TOOL_ROUNDS {
            let completion = self
                .llm
                .complete(&system, &user, &tools, &self.session_id, Some("submit_plan"))
                .context("planner LLM call")?;
            self.record_llm_usage("planner", &completion);

            if let Some(tc) = completion.tool_calls.iter().find(|tc| tc.name == "submit_plan") {
                let plan = tc.arguments.pointer("/plan");
                let get = |k: &str| {
                    plan.and_then(|p| p.get(k))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                };
                let pc = PlannedChange {
                    change: get("change"),
                    hints: get("implementation_hints"),
                    evidence: get("evidence"),
                };
                if is_no_plan(&pc.change) {
                    self.progress("  planner returned no concrete plan; re-prompting".to_string());
                    user.push_str(
                        "\n\nYour previous answer was not a concrete plan (a refusal or \
                         \"nothing to do\"). Propose ONE specific, DIFFERENT code change with a \
                         mechanism, in English, and call submit_plan.",
                    );
                    continue;
                }
                return Ok(pc);
            }

            let mut results = String::new();
            for tc in &completion.tool_calls {
                match tc.name.as_str() {
                    "search_repo" => {
                        let query = tc.arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
                        let glob = tc.arguments.get("glob").and_then(|v| v.as_str());
                        let max = tc.arguments.get("max").and_then(|v| v.as_u64()).unwrap_or(40) as usize;
                        let res = crate::tools::search(&self.worktree, query, glob, max)
                            .unwrap_or_else(|e| format!("search failed: {e}"));
                        self.progress(format!("  planner search_repo({query:?}) -> {} chars", res.len()));
                        results.push_str(&format!("\n\nTOOL search_repo(query={query:?}):\n{res}"));
                    }
                    "read_file" => {
                        let path = tc.arguments.get("path").and_then(|v| v.as_str()).unwrap_or("");
                        let start = tc.arguments.get("start").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                        let end = tc.arguments.get("end").and_then(|v| v.as_u64()).unwrap_or((start + 200) as u64) as usize;
                        let res = crate::tools::read_region(&self.worktree, path, start, end)
                            .unwrap_or_else(|e| format!("read failed: {e}"));
                        self.progress(format!("  planner read_file({path}:{start}-{end})"));
                        results.push_str(&format!("\n\nTOOL read_file({path}, {start}-{end}):\n{res}"));
                    }
                    _ => {}
                }
            }
            if results.is_empty() {
                let text = completion.content.unwrap_or_default();
                if is_no_plan(&text) {
                    self.progress("  planner returned no concrete plan; re-prompting".to_string());
                    user.push_str(
                        "\n\nYour previous answer was not a concrete plan. Propose ONE specific, \
                         DIFFERENT code change with a mechanism, in English, and call submit_plan.",
                    );
                    continue;
                }
                return Ok(PlannedChange {
                    change: text,
                    ..Default::default()
                });
            }
            user.push_str(&results);
        }
        // Exhausted rounds without a usable plan: empty change => caller skips.
        Ok(PlannedChange::default())
    }

    /// Planner context for the target file: the full source when small, an
    /// outline otherwise (the Planner uses `read_file`/`search_repo` for detail).
    fn planner_context(&self, kernel_source: &str) -> (String, String) {
        if kernel_source.chars().count() <= PLANNER_FULL_CHARS {
            (
                "KERNEL SOURCE".to_string(),
                format!("```cuda\n{kernel_source}\n```"),
            )
        } else {
            let outline = crate::tools::outline(&self.worktree, &self.target.target_file)
                .unwrap_or_else(|_| format!("({} chars)", kernel_source.chars().count()));
            (
                format!(
                    "KERNEL OUTLINE (file is {} lines/{} chars — too large to inline; use read_file/search_repo for the parts you need)",
                    kernel_source.lines().count(),
                    kernel_source.chars().count()
                ),
                outline,
            )
        }
    }

    // ---------- execute + gates 1/2/4 ----------

    fn stage_execute_and_verify(
        &mut self,
        kernel_source: &str,
        plan: &PlannedChange,
        chain_idx: u32,
        iteration: u32,
    ) -> Result<(Candidate, Option<String>)> {
        let prompt_file = match self.edit_mode {
            EditMode::Patch => "cuda-executor-patch.md",
            EditMode::Full => "cuda-executor.md",
        };
        let system = self.render_prompt(prompt_file, &json!({}))?;
        if self.edit_mode == EditMode::Full && kernel_source.chars().count() > EXECUTOR_WARN_CHARS {
            self.progress(format!(
                "  warning: {} is {} chars — full-file submission may exceed model output limits; consider a smaller --kernel-file",
                self.target.target_file,
                kernel_source.chars().count()
            ));
        }
        let base_user = format!(
            "BACKEND: {backend}\nTARGET FILE: {file}\n\n\
             CONTRACT HEADER (read-only semantic authority — do not change its semantics):\n{authority}\n\n\
             CURRENT CONTENT:\n```cuda\n{src}\n```\n\n\
             OPTIMIZATION PLAN:\n{plan}",
            backend = self.target.backend.as_str(),
            file = self.target.target_file,
            authority = self.authority_block(),
            src = kernel_source,
            plan = plan.change,
        );
        let tools = vec![match self.edit_mode {
            EditMode::Full => ToolDef {
                name: "submit_kernel".into(),
                description: "Submit the complete new content of the target kernel file".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "kernel_source": {"type": "string"},
                        "change_summary": {"type": "string"}
                    },
                    "required": ["kernel_source", "change_summary"]
                }),
            },
            EditMode::Patch => ToolDef {
                name: "submit_patch".into(),
                description: "Submit a unified diff for the target file (git-style, a/ b/ prefixes)".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string"},
                        "change_summary": {"type": "string"}
                    },
                    "required": ["patch", "change_summary"]
                }),
            },
        }];

        let mut last_error: Option<String> = None;
        // Signature of the previous failure, for detecting a repeated compile
        // error (no progress → stop paying for rebuilds).
        let mut last_fail_sig: Option<String> = None;

        for attempt in 0..self.cfg.hyper.k_retries {
            let user = match &last_error {
                Some(err) => format!(
                    "{base_user}\n\nPREVIOUS ATTEMPT FAILED (attempt {attempt}):\n{err}\n\n\
                     Diagnose the root cause and resubmit."
                ),
                None => base_user.clone(),
            };
            let tool_name = match self.edit_mode {
                EditMode::Patch => "submit_patch",
                EditMode::Full => "submit_kernel",
            };
            let completion = self
                .llm
                .complete(&system, &user, &tools, &self.session_id, Some(tool_name))
                .context("executor LLM call")?;
            self.record_llm_usage("executor", &completion);

            let submit_call = completion.tool_calls.iter().find(|tc| tc.name == tool_name);
            let summary_raw = submit_call
                .and_then(|tc: &ToolCall| tc.arguments.get("change_summary"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let submitted = submit_call
                .and_then(|tc: &ToolCall| match self.edit_mode {
                    EditMode::Full => tc.arguments.get("kernel_source"),
                    EditMode::Patch => tc.arguments.get("patch"),
                })
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let candidate_source = match submitted {
                Some(s) if !s.is_empty() => match self.edit_mode {
                    EditMode::Full => ensure_trailing_newline(strip_code_fences(&s)),
                    EditMode::Patch => {
                        let patch = strip_code_fences(&s);
                        // Apply the diff to the (baseline) worktree file.
                        if let Err(e) = crate::git::apply_patch(&self.worktree, &patch) {
                            let err = format!("patch did not apply: {e:#}");
                            self.journal.record(&Event::AttemptFailed {
                                iteration,
                                chain: chain_idx,
                                attempt,
                                category: "patch_apply".into(),
                                error: err.clone(),
                            })?;
                            let _ = self.record_failure("patch_apply", &err, &mut last_fail_sig);
                            last_error = Some(err);
                            continue;
                        }
                        let _ = std::fs::write(
                            self.run_dir
                                .join("candidates")
                                .join(format!("i{iteration}_c{chain_idx}_a{attempt}.patch")),
                            &patch,
                        );
                        std::fs::read_to_string(self.target_path())
                            .context("reading patched file")
                            .map(ensure_trailing_newline)?
                    }
                },
                _ => {
                    let err = format!("no {tool_name} tool call; call {tool_name}");
                    self.journal.record(&Event::AttemptFailed {
                        iteration,
                        chain: chain_idx,
                        attempt,
                        category: "no_tool_call".into(),
                        error: err.clone(),
                    })?;
                    let _ = self.record_failure("no_tool_call", &err, &mut last_fail_sig);
                    last_error = Some(err);
                    continue;
                }
            };
            // Fall back to a mechanical diff summary when the model omits one.
            let summary = Some(
                summary_raw
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| diff_summary(kernel_source, &candidate_source)),
            );
            // A byte-identical submission is a no-op: reject it and force a real
            // change rather than letting it become the "winner".
            if candidate_source == kernel_source {
                let err = "submission is byte-identical to the current file — \
                           the optimization appears already present; propose a DIFFERENT change."
                    .to_string();
                self.progress(format!("  c{chain_idx} ✗ no-op: {}", flatten(&err)));
                self.journal.record(&Event::AttemptFailed {
                    iteration,
                    chain: chain_idx,
                    attempt,
                    category: "no_change".into(),
                    error: err.clone(),
                })?;
                let _ = self.record_failure("no_change", &err, &mut last_fail_sig);
                last_error = Some(err);
                continue;
            }
            self.progress(format!(
                "  c{chain_idx} exec a{attempt}: {}",
                flatten(summary.as_deref().unwrap_or("(no summary)"))
            ));

            let sub_path = self
                .run_dir
                .join("candidates")
                .join(format!("i{iteration}_c{chain_idx}_a{attempt}.cu"));
            let _ = std::fs::write(&sub_path, &candidate_source);
            std::fs::write(self.target_path(), &candidate_source)
                .context("writing candidate into worktree")?;

            let compile = self.compile_now()?;
            if compile["passed"] != json!(true) {
                let err = format_compiler_errors(&compile);
                self.progress(format!("  c{chain_idx} ✗ compile: {}", flatten(&err)));
                self.journal.record(&Event::AttemptFailed {
                    iteration,
                    chain: chain_idx,
                    attempt,
                    category: "compile".into(),
                    error: err.clone(),
                })?;
                let repeated = self.record_failure("compile", &err, &mut last_fail_sig);
                last_error = Some(err);
                self.tracker.record(
                    &self.target.family.clone(),
                    &crate::memory::strategy_tag(&plan.change),
                    false,
                );
                if repeated {
                    self.progress(format!(
                        "  c{chain_idx} ✗ repeated compile error — stopping retries for this plan"
                    ));
                    break;
                }
                continue;
            }

            let verify = self.verify_now()?;
            if verify["passed"] != json!(true) {
                let err = format_verify_failure(&verify);
                self.progress(format!("  c{chain_idx} ✗ correctness: {}", flatten(&err)));
                self.journal.record(&Event::AttemptFailed {
                    iteration,
                    chain: chain_idx,
                    attempt,
                    category: "correctness".into(),
                    error: err.clone(),
                })?;
                let _ = self.record_failure("correctness", &err, &mut last_fail_sig);
                last_error = Some(err);
                self.tracker.record(
                    &self.target.family.clone(),
                    &crate::memory::strategy_tag(&plan.change),
                    false,
                );
                continue;
            }

            let latency_ms = if self.target.timing {
                match self.bench(Some(&candidate_source)) {
                    Ok(ms) => Some(ms),
                    Err(e) => {
                        let err = format!("bench failed: {e:#}");
                        self.journal.record(&Event::AttemptFailed {
                            iteration,
                            chain: chain_idx,
                            attempt,
                            category: "bench".into(),
                            error: err.clone(),
                        })?;
                        let _ = self.record_failure("bench", &err, &mut last_fail_sig);
                        last_error = Some(err);
                        continue;
                    }
                }
            } else {
                None
            };
            self.tracker.record(
                &self.target.family.clone(),
                &crate::memory::strategy_tag(&plan.change),
                true,
            );
            let id = format!("i{iteration}_c{chain_idx}_a{attempt}");
            // Commit the accepted candidate so it can be reverted to / read later.
            let commit = self.commit_candidate(&id, &plan.change, latency_ms);
            return Ok((
                Candidate {
                    id,
                    chain: chain_idx,
                    iteration,
                    source: candidate_source,
                    plan: plan.change.clone(),
                    latency_ms,
                    passed: true,
                    commit,
                    change_summary: summary.clone(),
                    hints: Some(plan.hints.clone()),
                    evidence: Some(plan.evidence.clone()),
                },
                None,
            ));
        }

        Ok((
            Candidate {
                id: format!("i{iteration}_c{chain_idx}_failed"),
                chain: chain_idx,
                iteration,
                source: String::new(),
                plan: plan.change.clone(),
                latency_ms: None,
                passed: false,
                commit: None,
                change_summary: None,
                hints: Some(plan.hints.clone()),
                evidence: Some(plan.evidence.clone()),
            },
            last_error,
        ))
    }

    // ---------- summarize → memory ----------

    fn stage_summarize(
        &mut self,
        slow_source: &str,
        fast_source: &str,
        plan: &str,
        speedup: f64,
        iteration: u32,
    ) -> Result<()> {
        let store = self.memory.should_store(speedup);
        let system = self.render_prompt("summarizer.md", &json!({}))?;
        let user = format!(
            "SLOW KERNEL:\n```cuda\n{slow_source}\n```\n\nFAST KERNEL:\n```cuda\n{fast_source}\n```\n\n\
             PLAN APPLIED: {plan}\n\nSPEEDUP: {speedup:.3}"
        );
        let completion = self
            .llm
            .complete(&system, &user, &[], &self.session_id, None)
            .context("summarizer LLM call")?;
        self.record_llm_usage("summarizer", &completion);
        let text = completion.content.unwrap_or_default();

        if let Ok((Some(item), update)) = parse_summarizer_output(&text) {
            let update = update.unwrap_or(MemoryUpdate {
                action: "append".into(),
                replace_item_id: None,
                reason: None,
            });
            let action = self.memory.apply(item, update);
            self.journal.record(&Event::MemoryUpdated {
                action: action.into(),
                direction: Some(plan.chars().take(80).collect()),
            })?;
        } else if store {
            let fallback = ExperienceItem {
                item_id: format!("fb_{}", uuid::Uuid::new_v4().simple()),
                iteration,
                speedup,
                rewrite_type: "unknown".into(),
                framework: self.target.backend.as_str().into(),
                direction: plan.chars().take(80).collect(),
                profiling_signal: String::new(),
                strategy_title: format!("(unparsed summarizer) {:.40}", plan),
                strategy_description: String::new(),
                slow_pseudocode: String::new(),
                fast_pseudocode: String::new(),
                applicable_when: String::new(),
                do_not_apply_when: String::new(),
                framework_notes: String::new(),
            };
            self.memory.apply(
                fallback,
                MemoryUpdate { action: "append".into(), replace_item_id: None, reason: None },
            );
            self.journal.record(&Event::MemoryUpdated {
                action: "append_fallback".into(),
                direction: Some(plan.chars().take(80).collect()),
            })?;
        }
        Ok(())
    }

    // ---------- full run ----------

    pub fn run(&mut self) -> Result<PipelineResult> {
        // `--watch`: render the journal live (stderr) in this same terminal.
        let stop = Arc::new(AtomicBool::new(false));
        let handle = if self.watch {
            let path = self.journal.path().to_path_buf();
            let stop2 = stop.clone();
            Some(std::thread::spawn(move || {
                crate::journal::tail_journal(&path, &stop2)
            }))
        } else {
            None
        };
        let result = self.run_inner();
        stop.store(true, Ordering::Relaxed);
        if let Some(h) = handle {
            let _ = h.join();
        }
        self.write_report();
        result
    }

    fn run_inner(&mut self) -> Result<PipelineResult> {
        self.journal.record(&Event::RunStarted {
            run_id: self.session_id.clone(),
            model_file: format!("{}:{}", self.target.backend.as_str(), self.target.op),
            provider: self.cfg.provider.clone(),
            model: self.cfg.model.clone(),
        })?;

        if self.target.kernel_files.is_empty() {
            return self.finish_fallback("discover found no kernel files for this op");
        }
        if self.target.test_filters.is_empty() {
            return self.finish_fallback("target has no correctness authority (Gate 2 missing)");
        }

        let baseline_ms = match self.stage_baseline() {
            Ok(ms) => ms,
            Err(e) => return self.finish_fallback(&format!("baseline failed: {e:#}")),
        };

        let mut profiling_ctx = self.stage_profile().unwrap_or(json!({"ncu": null}));
        // Tell the planner which shape the Gate-4 bench actually measures — the
        // NCU profile above may be a different launch/shape.
        profiling_ctx["measurement"] = json!({
            "gate_shape": self.last_bench_label,
            "gate_baseline_ms": baseline_ms,
            "note": "Gate 4 measures THIS shape (median of repeats). The NCU profile above may be a different launch; plan for the measured shape's regime.",
        });
        let baseline_source = std::fs::read_to_string(self.target_path())
            .context("reading baseline kernel from worktree")?;

        let mut best: Option<Candidate> = None;
        let mut best_ms = if baseline_ms > 0.0 { baseline_ms } else { f64::INFINITY };
        let mut recent_directions: Vec<String> = Vec::new();
        let mut patience_used = 0u32;
        let mut iteration = 0u32;
        let mut iterations_run = 0u32;
        let mut stop_reason = "max_iterations";

        // Beam frontier: start with the baseline as the root node. Chains expand
        // frontier nodes so improvements can compose (paper Algorithm 1).
        let mut frontier: Vec<BeamNode> = vec![BeamNode {
            expansions: 0,
            candidate: Candidate {
                id: "baseline".into(),
                chain: 0,
                iteration: 0,
                source: baseline_source.clone(),
                plan: "(baseline)".into(),
                latency_ms: if baseline_ms > 0.0 { Some(baseline_ms) } else { None },
                passed: true,
                commit: None,
                change_summary: None,
                hints: None,
                evidence: None,
            },
        }];

        // Resume an interrupted target: restore the search state saved after the
        // last completed iteration (never repeats finished LLM work).
        if let Some(cp) = self.load_checkpoint() {
            best = cp.best;
            if cp.best_ms.is_finite() && cp.best_ms > 0.0 {
                best_ms = cp.best_ms;
            }
            recent_directions = cp.recent_directions;
            patience_used = cp.patience_used;
            iteration = cp.iteration;
            iterations_run = cp.iterations_run;
            self.llm_calls = cp.llm_calls;
            self.tokens = cp.tokens;
            println!(
                "[pipeline] resuming {} from checkpoint: iteration {} (best {:?} ms)",
                self.target.op,
                iteration,
                best.as_ref().and_then(|b| b.latency_ms)
            );
        }

        'outer: while iteration < self.max_iterations {
            if self.interrupted() {
                stop_reason = "interrupted";
                self.save_checkpoint(iteration, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
                break;
            }
            if !self.budget_ok() {
                stop_reason = "budget";
                self.save_checkpoint(iteration, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
                break;
            }
            iterations_run += 1;
            self.progress(format!(
                "[{}] iteration {}/{} (best {})",
                self.target.op,
                iteration + 1,
                self.max_iterations,
                best.as_ref().and_then(|b| b.latency_ms).map(|m| format!("{m:.4} ms")).unwrap_or_else(|| "—".into()),
            ));
            let prev_best = best_ms;
            let mut children: Vec<BeamNode> = Vec::new();
            let diversity_hint = meltdown_detected(&recent_directions, 6, 2);

            // Allocate this iteration's N plans across the beam by UCB(c): the
            // best-latency arm draws more plans while underexpanded arms keep a
            // bonus (paper §4.3). `pending[arm]` = plans assigned to each arm.
            let slots = allocate_expansions(
                &frontier,
                self.cfg.hyper.n_plans,
                self.cfg.hyper.ucb_c,
            );
            let mut pending = vec![0u32; frontier.len()];
            for &a in &slots {
                pending[a] += 1;
            }

            for (arm, &count) in pending.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let parent = frontier[arm].candidate.clone();
                let parent_expansions = frontier[arm].expansions;
                let chain_u = arm as u32;
                for _ in 0..count {
                    if self.interrupted() {
                        stop_reason = "interrupted";
                        self.save_checkpoint(iteration, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
                        break 'outer;
                    }
                    if !self.budget_ok() {
                        stop_reason = "budget";
                        self.save_checkpoint(iteration, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
                        break 'outer;
                    }
                    // Start from the parent node's source (its commit, or its content).
                    self.reset_worktree()?;
                    std::fs::write(self.target_path(), &parent.source)?;
                    let plan = self.stage_plan(
                        &parent.source,
                        &profiling_ctx,
                        diversity_hint,
                        &recent_directions,
                    )?;
                    if plan.change.is_empty() {
                        self.progress(format!("  c{arm} no plan; skipping"));
                        self.journal.record(&Event::CandidateEvaluated {
                            iteration,
                            chain: chain_u,
                            plan: "(no plan)".into(),
                            passed: false,
                            latency_ms: None,
                            error: Some("planner produced no concrete plan".into()),
                            commit: None,
                            change_summary: None,
                            hints: None,
                            evidence: None,
                        })?;
                        continue;
                    }
                    recent_directions.push(plan.change.clone());
                    self.progress(format!(
                        "  c{arm} (from {}) plan: {}",
                        parent.id,
                        flatten(&plan.change)
                    ));

                    let (cand, last_error) =
                        self.stage_execute_and_verify(&parent.source, &plan, chain_u, iteration)?;
                    self.journal.record(&Event::CandidateEvaluated {
                        iteration,
                        chain: chain_u,
                        plan: plan.change.clone(),
                        passed: cand.passed,
                        latency_ms: cand.latency_ms,
                        error: last_error.clone(),
                        commit: cand.commit.clone(),
                        change_summary: cand.change_summary.clone(),
                        hints: cand.hints.clone(),
                        evidence: cand.evidence.clone(),
                    })?;

                    if cand.passed {
                        let speedup = cand
                            .latency_ms
                            .map(|m| if baseline_ms > 0.0 { baseline_ms / m } else { 1.0 })
                            .unwrap_or(1.0);
                        self.progress(format!(
                            "  c{arm} ✓ {}{}",
                            cand.latency_ms
                                .map(|m| format!("{m:.4} ms"))
                                .unwrap_or_else(|| "correct (no timing)".into()),
                            if baseline_ms > 0.0 {
                                format!(" · {speedup:.2}x")
                            } else {
                                String::new()
                            }
                        ));
                        self.stage_summarize(&parent.source, &cand.source, &plan.change, speedup, iteration)?;
                        if let Some(ms) = cand.latency_ms {
                            if ms < best_ms {
                                best_ms = ms;
                            }
                        }
                        if best
                            .as_ref()
                            .and_then(|b| b.latency_ms)
                            .map(|b| cand.latency_ms.map(|c| c < b).unwrap_or(true))
                            .unwrap_or(true)
                        {
                            best = Some(cand.clone());
                        }
                        children.push(BeamNode {
                            expansions: parent_expansions + 1,
                            candidate: cand,
                        });
                    } else if let Some(err) = &last_error {
                        self.progress(format!("  c{arm} ✗ {}", flatten(err)));
                    }
                }
            }

            let selected = diverse_select_nodes(&children, self.cfg.hyper.b_beam as usize);
            if !selected.is_empty() {
                frontier = selected;
            }

            // Improvement accounting drives patience termination.
            if best_ms < prev_best * (1.0 - self.min_improvement) {
                patience_used = 0;
            } else {
                patience_used += 1;
            }

            if self.target_speedup > 0.0
                && baseline_ms > 0.0
                && baseline_ms / best_ms >= self.target_speedup
            {
                stop_reason = "target_reached";
                self.save_checkpoint(iteration + 1, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
                break 'outer;
            }
            if patience_used >= self.patience {
                stop_reason = "patience";
                self.save_checkpoint(iteration + 1, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
                break 'outer;
            }
            iteration += 1;
            self.save_checkpoint(iteration, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
        }

        let iterations = iterations_run;

        // A budget pause or Ctrl-C leaves the target resumable rather than
        // finalizing it: keep the checkpoint, mark "paused", and let the campaign
        // retry later.
        if stop_reason == "budget" || stop_reason == "interrupted" {
            self.save_checkpoint(iteration, iterations_run, &best, best_ms, baseline_ms, &recent_directions, patience_used);
            let cause = if stop_reason == "interrupted" {
                "interrupted; resumable"
            } else {
                "budget paused; resumable"
            };
            self.journal.record(&Event::RunFinished {
                outcome: "paused".into(),
                speedup: None,
                root_cause: Some(cause.into()),
                stop_reason: Some(stop_reason.into()),
            })?;
            return Ok(PipelineResult {
                outcome: "paused".into(),
                speedup: None,
                baseline_ms: Some(baseline_ms),
                final_ms: None,
                iterations,
                stop_reason: stop_reason.into(),
                llm_calls: self.llm_calls,
                tokens: self.tokens,
                diff_path: None,
                root_cause: Some(cause.into()),
                paused: true,
                summary: None,
            });
        }

        match best {
            Some(cand) => self.finalize(cand, baseline_ms, iterations, stop_reason),
            None => {
                let cause = "no candidate passed compile+correctness";
                self.journal.record(&Event::RunFinished {
                    outcome: "fallback".into(),
                    speedup: None,
                    root_cause: Some(cause.into()),
                    stop_reason: Some(stop_reason.into()),
                })?;
                Ok(PipelineResult {
                    outcome: "fallback".into(),
                    speedup: None,
                    baseline_ms: Some(baseline_ms),
                    final_ms: None,
                    iterations,
                    stop_reason: stop_reason.into(),
                    llm_calls: self.llm_calls,
                    tokens: self.tokens,
                    diff_path: None,
                    root_cause: Some(cause.into()),
                    paused: false,
                    summary: None,
                })
            }
        }
    }

    fn finalize(
        &mut self,
        best: Candidate,
        baseline_ms: f64,
        iterations: u32,
        stop_reason: &str,
    ) -> Result<PipelineResult> {
        self.stage_start("finalize", "rebuilding the winner and running the gates");
        // Materialize the winner, rebuild, re-test (Gates 1–2).
        self.materialize_winner(&best)?;
        let compile = self.compile_now()?;
        let verify = self.verify_now()?;
        let gate1_2 = compile["passed"] == json!(true) && verify["passed"] == json!(true);

        // Gate 4: interleaved fresh baseline/candidate benches so drift cancels
        // and we can require the candidate to win in every round (sign test).
        let mut baseline_final_ms = baseline_ms;
        let mut wins = 0u32;
        let mut rounds = 0u32;
        let mut cand_gbs: Option<f64> = None;
        let mut cand_roofline_gbs: Option<f64> = None;
        let final_ms = if gate1_2 && self.target.timing {
            let base_ref = self.base_sha.clone().unwrap_or_else(|| "HEAD".to_string());
            let mut bases: Vec<f64> = Vec::new();
            let mut cands: Vec<f64> = Vec::new();
            for _ in 0..self.final_rounds.max(1) {
                self.revert_ref(&base_ref)?;
                let _ = self.compile_now()?;
                let b = self.bench(None)?;
                self.materialize_winner(&best)?;
                let _ = self.compile_now()?;
                let c = self.bench(None)?;
                cand_gbs = self.last_bench_gbs;
                cand_roofline_gbs = self.last_bench_roofline_gbs;
                if c < b {
                    wins += 1;
                }
                bases.push(b);
                cands.push(c);
            }
            rounds = bases.len() as u32;
            baseline_final_ms = median(&bases).unwrap_or(baseline_ms);
            median(&cands)
        } else {
            None
        };
        // Leave the worktree at the winner for the diff below.
        self.materialize_winner(&best)?;
        let _ = self.compile_now()?;

        // Model-level Gate 3: the whole engine must produce the SAME tokens and
        // not be slower. Compares the candidate build against the baseline build.
        let gate3 = if self.e2e.is_some() {
            let baseline = self.baseline_e2e.clone();
            match (baseline, self.run_engine()) {
                (Some(base), Ok(cand)) => {
                    let (passed, detail) = engine_gate(&base, &cand, self.cfg.hyper.gamma);
                    self.journal.record(&Event::GatesVerdict {
                        stage: "engine_e2e".into(),
                        passed,
                        detail,
                    })?;
                    passed
                }
                (None, _) => true, // baseline capture skipped (shouldn't happen)
                (_, Err(e)) => {
                    self.journal.record(&Event::GatesVerdict {
                        stage: "engine_e2e".into(),
                        passed: false,
                        detail: json!({"error": format!("{e:#}")}),
                    })?;
                    false
                }
            }
        } else {
            true
        };

        // Gate 4 verdict (pure, unit-tested below): a candidate counts as a real
        // improvement only if it beats the fresh baseline beyond BOTH the noise
        // floor and the minimum effect, AND wins every interleaved round.
        let noise = self.last_bench_noise_pct.unwrap_or(0.0);
        let verdict = performance_verdict(
            baseline_final_ms,
            final_ms,
            noise,
            self.min_improvement,
            rounds,
            wins,
            self.cfg.hyper.gamma,
            cand_gbs,
            cand_roofline_gbs,
            DEFAULT_MAX_ROOFLINE_RATIO,
        );
        let gate4 = !self.target.timing || verdict.gate4;
        let speedup = final_ms.map(|_| verdict.speedup);
        let within_noise = verdict.within_noise;

        // Gate 5: correctness at the *measured* shape. The op's canned test suite
        // may not cover the launch path the bench shape exercises (a real false
        // positive skipped 3/4 of the rows, passed the suite, and looked 22x
        // faster). Runs for any repeatable perf win; self-validates against the
        // baseline, so an unsupported test simply degrades to "skipped".
        let perf_win = verdict.gate4
            && verdict.improvement > self.min_improvement
            && !verdict.within_noise
            && (rounds == 0 || wins == rounds);
        let shape_gate = if perf_win {
            self.shape_consistency_gate(&best)?
        } else {
            ShapeGate::Skipped("no repeatable perf win to verify".into())
        };
        let shape_ok = !matches!(shape_gate, ShapeGate::Failed(_));
        self.journal.record(&Event::GatesVerdict {
            stage: "shape".into(),
            passed: shape_ok,
            detail: shape_gate.detail(),
        })?;

        let base_ref = self.base_sha.clone().unwrap_or_else(|| "HEAD".into());
        let diff = crate::git::diff(&self.worktree, &base_ref, &[self.target.target_file.clone()])?;
        let diff_stat = crate::parse::numstat_view(&diff);
        let diff_text = diff["diff"].as_str().unwrap_or("");
        let diff_path = self.run_dir.join("report.diff");
        let _ = std::fs::write(&diff_path, diff_text);

        let passed = gate1_2 && gate3 && gate4 && shape_ok;
        self.journal.record(&Event::GatesVerdict {
            stage: "gates".into(),
            passed,
            detail: json!({
                "gate1_2": gate1_2,
                "gate3": gate3,
                "gate5_shape": shape_gate.detail(),
                "gate4": {"passed": gate4, "final_ms": final_ms,
                          "baseline_ms": baseline_final_ms,
                          "baseline_ms_initial": baseline_ms,
                          "gamma": self.cfg.hyper.gamma,
                          "noise_pct": self.last_bench_noise_pct,
                          "shape": self.last_bench_label,
                          "cand_gbs": cand_gbs,
                          "roofline_gbs": cand_roofline_gbs,
                          "roofline_ratio": verdict.roofline_ratio,
                          "plausible": verdict.plausible,
                          "runs": 3,
                          "rounds": rounds,
                          "wins": wins},
                "diff_files": diff_stat["file_count"],
                "diff_insertions": diff_stat["insertions"],
                "diff_deletions": diff_stat["deletions"],
                "diff_path": diff_path,
                "stop_reason": stop_reason,
                "iterations": iterations,
            }),
        })?;

        let (outcome, root_cause): (&str, Option<String>) = if !verdict.plausible {
            (
                "fallback",
                Some(format!(
                    "implausibly fast: candidate moves {:.0}% of the memory roofline (>{}x) — it likely \
                     performs less work than the baseline, and the correctness suite may not cover the \
                     measured shape (baseline preserved)",
                    100.0 * verdict.roofline_ratio,
                    DEFAULT_MAX_ROOFLINE_RATIO,
                )),
            )
        } else if passed {
            if !self.target.timing || verdict.optimized {
                ("optimized", None)
            } else {
                ("matched", None)
            }
        } else if !shape_ok {
            (
                "fallback",
                Some(format!(
                    "candidate is correct on the op's suite but fails the measured shape {} (baseline preserved)",
                    self.last_bench_label.as_deref().unwrap_or("?")
                )),
            )
        } else if !gate1_2 {
            ("fallback", Some("winner failed to rebuild/re-test (baseline preserved)".into()))
        } else if !gate4 {
            ("fallback", Some("winner did not beat the perf gate γ (baseline preserved)".into()))
        } else {
            ("fallback", Some("whole-engine gate rejected the winner (baseline preserved)".into()))
        };

        self.journal.record(&Event::RunFinished {
            outcome: outcome.into(),
            speedup,
            root_cause: root_cause.clone(),
            stop_reason: Some(stop_reason.into()),
        })?;
        self.clear_checkpoint();
        let summary = json!({
            "candidate": best.id,
            "commit": best.commit,
            "what_changed": best.change_summary,
            "plan": best.plan,
            "why": best.evidence,
            "how": best.hints,
            "final_ms": final_ms,
            "baseline_ms": baseline_final_ms,
            "speedup": speedup,
            "noise_pct": noise,
            "within_noise": within_noise,
            "rounds": rounds,
            "wins": wins,
            "measurement": json!({
                "scope": "op",
                "metric": "median latency",
                "unit": "ms",
                "shape": self.last_bench_label,
                "runs": 3,
                "rounds": rounds,
                "wins": wins,
                "baseline_ms": baseline_final_ms,
                "final_ms": final_ms,
                "speedup": speedup,
                "noise_pct": noise,
                "within_noise": within_noise,
                "cand_gbs": cand_gbs,
                "roofline_gbs": cand_roofline_gbs,
                "roofline_ratio": verdict.roofline_ratio,
                "plausible": verdict.plausible,
                "note": "op-level bench; interleaved fresh baseline/candidate, 3 repeats each; not an end-to-end model speedup",
            }),
        });
        Ok(PipelineResult {
            outcome: outcome.into(),
            speedup,
            baseline_ms: Some(baseline_ms),
            final_ms,
            iterations,
            stop_reason: stop_reason.into(),
            llm_calls: self.llm_calls,
            tokens: self.tokens,
            diff_path: Some(diff_path.to_string_lossy().to_string()),
            root_cause,
            paused: false,
            summary: Some(summary),
        })
    }

    fn finish_fallback(&mut self, cause: &str) -> Result<PipelineResult> {
        self.journal.record(&Event::RunFinished {
            outcome: "fallback".into(),
            speedup: None,
            root_cause: Some(cause.into()),
            stop_reason: Some("baseline".into()),
        })?;
        self.clear_checkpoint();
        Ok(PipelineResult {
            outcome: "fallback".into(),
            speedup: None,
            baseline_ms: None,
            final_ms: None,
            iterations: 0,
            stop_reason: "baseline".into(),
            llm_calls: self.llm_calls,
            tokens: self.tokens,
            diff_path: None,
            root_cause: Some(cause.into()),
            paused: false,
            summary: None,
        })
    }

    /// Human-readable summary next to the journal.
    fn write_report(&self) {
        let Ok(events) = Journal::replay(&self.cfg.runs_dir, &self.session_id) else {
            return;
        };
        let mut s = format!(
            "# KernelOpt report — {}\n\nBackend: `{}`\nOp: `{}` (family `{}`{})\nRepo: `{}`\nKernel file: `{}`\n\n## Events\n",
            self.session_id,
            self.target.backend.as_str(),
            self.target.op,
            self.target.family,
            self.target.variant.as_deref().map(|v| format!(" / {v}")).unwrap_or_default(),
            self.repo.display(),
            self.target.target_file,
        );
        for e in &events {
            match e {
                Event::StageCompleted { stage, data } => {
                    s.push_str(&format!("- stage `{stage}`: {}\n", compact(data)));
                }
                Event::CandidateEvaluated { iteration, chain, passed, latency_ms, plan, error, .. } => {
                    s.push_str(&format!(
                        "- candidate i{iteration}/c{chain}: {} {} — {}\n",
                        if *passed { "PASS" } else { "FAIL" },
                        latency_ms.map(|m| format!("@ {m:.4} ms")).unwrap_or_default(),
                        plan.chars().take(120).collect::<String>(),
                    ));
                    if let Some(err) = error {
                        s.push_str(&format!("  - error: {}\n", err.chars().take(200).collect::<String>()));
                    }
                }
                Event::GatesVerdict { stage, passed, detail } => {
                    s.push_str(&format!(
                        "- gates `{stage}`: {} — {}\n",
                        if *passed { "PASS" } else { "REJECT" },
                        compact(detail)
                    ));
                }
                Event::RunFinished { outcome, speedup, root_cause, stop_reason, .. } => {
                    s.push_str(&format!("\n## outcome: {outcome}\n"));
                    if let Some(sr) = stop_reason {
                        s.push_str(&format!("stop reason: {sr}\n"));
                    }
                    if let Some(sp) = speedup {
                        s.push_str(&format!("speedup: {sp:.3}x\n"));
                    }
                    if let Some(c) = root_cause {
                        s.push_str(&format!("root cause: {c}\n"));
                    }
                }
                _ => {}
            }
        }
        if let Some(w) = winner_summary(&events) {
            s.push_str("\n## Winner — what changed and why\n\n");
            s.push_str(&format!("- candidate: `{}`\n", w["candidate"].as_str().unwrap_or("")));
            if let Some(c) = w["commit"].as_str() {
                s.push_str(&format!("- commit: `{c}`\n"));
            }
            if let Some(x) = w["what_changed"].as_str().filter(|x| !x.is_empty()) {
                s.push_str(&format!("- **what changed**: {x}\n"));
            }
            if let Some(x) = w["why"].as_str().filter(|x| !x.is_empty()) {
                s.push_str(&format!("- **why it's faster**: {x}\n"));
            }
            if let Some(x) = w["how"].as_str().filter(|x| !x.is_empty()) {
                s.push_str(&format!("- how: {x}\n"));
            }
            if let Some(m) = w.get("measurement") {
                let shape = m["shape"].as_str().unwrap_or("(all shapes)");
                let runs = m["runs"].as_u64().unwrap_or(1);
                if let (Some(base), Some(fin)) = (m["baseline_ms"].as_f64(), m["final_ms"].as_f64()) {
                    let sp = m["speedup"].as_f64().unwrap_or(1.0);
                    s.push_str(&format!(
                        "- **measured (op-level)**: median latency on `{shape}`, {runs} runs — {base:.4} ms → {fin:.4} ms = **{sp:.2}x** \
                         ({sp:.2}x = baseline/final: latency fell to {:.0}% of baseline)\n",
                        100.0 / sp
                    ));
                    if let Some(n) = m["noise_pct"].as_f64() {
                        s.push_str(&format!("- measurement noise: {:.2}%\n", n * 100.0));
                    }
                    if let (Some(g), Some(r)) = (m["cand_gbs"].as_f64(), m["roofline_gbs"].as_f64()) {
                        if r > 0.0 {
                            s.push_str(&format!(
                                "- bandwidth: {g:.0} GB/s = {:.0}% of the {r:.0} GB/s memory roofline\n",
                                g / r * 100.0
                            ));
                        }
                    }
                }
            }
            s.push_str(&format!("\n**Plan:** {}\n", w["plan"].as_str().unwrap_or("")));
        }
        let _ = std::fs::write(self.run_dir.join("report.md"), s);
    }

    fn branch(&self) -> String {
        format!("kernelopt/{}-{}", self.target.backend.as_str(), self.target.op)
    }

    // ---------- finalize helpers ----------

    fn compile_now(&self) -> Result<serde_json::Value> {
        let resp = if let Some(bc) = &self.target.build_cmd {
            // Non-CMake projects declare their own build command.
            let argv = crate::custom::expand(bc, &self.worktree, &self.build_dir, None);
            crate::exec::run_argv(&argv, 3600)?
        } else {
            crate::exec::build(
                &self.worktree,
                &self.build_dir,
                &self.compile_targets(),
                false,
                false,
                &[],
                num_cpus(),
                3600,
            )?
        };
        Ok(crate::parse::compile_view(&resp))
    }

    fn verify_now(&self) -> Result<serde_json::Value> {
        let _guard = crate::gpu_lock::lock(&self.gpu_lock_path)?;
        let timeout = self.verify_timeout_s.get();
        let resp = if let Some(tc) = &self.target.test_cmd {
            let argv = crate::custom::expand(tc, &self.worktree, &self.build_dir, None);
            crate::exec::run_argv(&argv, timeout)?
        } else {
            match self.target.backend {
                Backend::Ninfer => {
                    crate::exec::ctest(&self.build_dir, &self.target.test_filters, timeout)?
                }
                Backend::Llamacpp => {
                    let bin = self.build_dir.join("bin").join("test-backend-ops");
                    crate::exec::llama_test(&bin, "CUDA0", &self.target.test_filters, timeout)?
                }
                Backend::Custom => {
                    anyhow::bail!("custom target has no `test_cmd` in kernelopt.toml")
                }
            }
        };
        Ok(crate::parse::verify_view(&resp, self.target.backend))
    }

    fn revert_ref(&self, r: &str) -> Result<()> {
        crate::git::revert(&self.worktree, r)
            .map(|_| ())
            .with_context(|| format!("revert to {r}"))
    }

    /// Put the worktree exactly at the winning candidate (git revert, or reset+write).
    fn materialize_winner(&self, best: &Candidate) -> Result<()> {
        match &best.commit {
            Some(sha) => self.revert_ref(sha),
            None => {
                self.reset_worktree()?;
                std::fs::write(self.target_path(), &best.source)?;
                Ok(())
            }
        }
    }

    /// Gate 5: does the candidate compute the right answer at the *measured*
    /// shape? Extends the op's test with that shape, then runs it against the
    /// baseline (validating the generated case) and the candidate. Restores the
    /// test and leaves the worktree at the winner.
    fn shape_consistency_gate(&mut self, best: &Candidate) -> Result<ShapeGate> {
        if self.target.backend != Backend::Ninfer || !self.target.timing {
            return Ok(ShapeGate::Skipped("not a ninfer timing target".into()));
        }
        let Some(shape) = parse_bench_shape(self.last_bench_label.as_deref()) else {
            return Ok(ShapeGate::Skipped("no parseable bench shape".into()));
        };
        let mut chosen: Option<(String, String, String)> = None;
        for rel in &self.target.test_sources {
            let path = self.worktree.join(rel);
            let Ok(original) = std::fs::read_to_string(&path) else {
                continue;
            };
            if let Some(patched) = append_shape_case(&original, &self.target.op, &shape) {
                chosen = Some((rel.clone(), original, patched));
                break;
            }
        }
        let Some((rel, original, patched)) = chosen else {
            return Ok(ShapeGate::Skipped("no extendable run_case test found".into()));
        };
        let path = self.worktree.join(&rel);
        let base_ref = self.base_sha.clone().unwrap_or_else(|| "HEAD".into());

        // 1) Validate the generated case against the baseline (trusted-correct).
        self.revert_ref(&base_ref)?;
        std::fs::write(&path, &patched)?;
        let _ = self.compile_now()?;
        let base = self.verify_now()?;

        // 2) The same case against the candidate.
        self.materialize_winner(best)?;
        std::fs::write(&path, &patched)?;
        let _ = self.compile_now()?;
        let cand = self.verify_now()?;

        // Restore the test and leave the worktree at the winner for the diff.
        std::fs::write(&path, &original)?;
        self.materialize_winner(best)?;
        let _ = self.compile_now()?;

        let base_passed = base["passed"] == json!(true);
        let cand_passed = cand["passed"] == json!(true);
        let detail = json!({
            "shape": self.last_bench_label,
            "case_dims": shape,
            "source": rel,
            "baseline_passed": base_passed,
            "candidate_passed": cand_passed,
            "baseline_failures": base["failing_cases"],
            "candidate_failures": cand["failing_cases"],
        });
        if !base_passed {
            return Ok(ShapeGate::Skipped(format!(
                "baseline failed the generated case (gate not used): {detail}"
            )));
        }
        if !cand_passed {
            return Ok(ShapeGate::Failed(detail));
        }
        Ok(ShapeGate::Passed(detail))
    }

    fn reset_worktree(&self) -> Result<()> {
        let base = self.base_sha.clone().unwrap_or_else(|| "HEAD".to_string());
        crate::git::reset(&self.worktree, &base)
            .map(|_| ())
            .context("worktree reset")
    }

    /// Commit the current worktree state as a candidate and tag it so it stays
    /// reachable after the branch resets to base. Returns the commit sha.
    fn commit_candidate(&self, id: &str, plan: &str, latency_ms: Option<f64>) -> Option<String> {
        let message = format!(
            "kernelopt {id}: {} (latency {})",
            one_line(plan, 80),
            latency_ms
                .map(|m| format!("{m:.4} ms"))
                .unwrap_or_else(|| "-".to_string())
        );
        let tag = format!("kernelopt/{}/{}", self.session_id, id);
        crate::git::commit(&self.worktree, &message, Some(&tag))
            .ok()
            .and_then(|v| v["sha"].as_str().map(|s| s.to_string()))
    }

    // ---------- per-run checkpoint (resume without repeating work) ----------

    fn load_checkpoint(&self) -> Option<Checkpoint> {
        let path = self.checkpoint_path.as_ref()?;
        let text = std::fs::read_to_string(path).ok()?;
        let cp: Checkpoint = serde_json::from_str(&text).ok()?;
        (cp.op == self.target.op && cp.backend == self.target.backend.as_str()).then_some(cp)
    }

    #[allow(clippy::too_many_arguments)]
    fn save_checkpoint(
        &self,
        iteration: u32,
        iterations_run: u32,
        best: &Option<Candidate>,
        best_ms: f64,
        baseline_ms: f64,
        recent_directions: &[String],
        patience_used: u32,
    ) {
        let Some(path) = &self.checkpoint_path else {
            return;
        };
        let cp = Checkpoint {
            op: self.target.op.clone(),
            backend: self.target.backend.as_str().to_string(),
            iteration,
            iterations_run,
            best: best.clone(),
            best_ms,
            baseline_ms,
            recent_directions: recent_directions.to_vec(),
            patience_used,
            llm_calls: self.llm_calls,
            tokens: self.tokens,
        };
        if let Ok(text) = serde_json::to_string(&cp) {
            let _ = std::fs::write(path, text);
        }
    }

    fn clear_checkpoint(&self) {
        if let Some(path) = &self.checkpoint_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Detect a planner "non-plan" (refusal / "nothing to do") so we don't spend an
/// Executor round-trip on it.
fn is_no_plan(change: &str) -> bool {
    let t = change.trim();
    if t.chars().count() < 12 {
        return true;
    }
    let l = t.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "no plan", "no additional", "no meaningful", "rejected", "no further",
        "nothing to", "cannot propose", "no optimization", "already optimal",
        "no concrete",
    ];
    if MARKERS.iter().any(|m| l.contains(m)) {
        return true;
    }
    t.contains("拒绝") || t.contains("无需") || t.contains("没有")
}

/// Preserve the file's trailing newline (models often drop it).
fn ensure_trailing_newline(mut s: String) -> String {
    if !s.ends_with('\n') {
        s.push('\n');
    }
    s
}

/// A cheap "what changed" fallback when the Executor omits `change_summary`.
fn diff_summary(before: &str, after: &str) -> String {
    let b: Vec<&str> = before.lines().collect();
    let a: Vec<&str> = after.lines().collect();
    let changed = b.iter().zip(a.iter()).filter(|(x, y)| x != y).count();
    let delta = a.len() as isize - b.len() as isize;
    format!(
        "{changed} lines changed, {delta:+} lines net vs baseline (executor gave no summary)"
    )
}

/// Extract the winning candidate's "what changed / why faster" from a run's
/// journal (used by `report.md`, `report`, `analyze`).
pub fn winner_summary(events: &[Event]) -> Option<serde_json::Value> {
    let baseline_ms = events.iter().find_map(|e| match e {
        Event::StageCompleted { stage, data } if stage == "bench_baseline" => {
            data.get("median_ms").and_then(|v| v.as_f64())
        }
        _ => None,
    });
    let mut best: Option<(f64, &Event)> = None;
    for e in events {
        if let Event::CandidateEvaluated { passed: true, latency_ms: Some(ms), .. } = e {
            let ms = *ms;
            if best.map(|(b, _)| ms < b).unwrap_or(true) {
                best = Some((ms, e));
            }
        }
    }
    let (ms, e) = best?;
    let Event::CandidateEvaluated {
        iteration, chain, plan, change_summary, hints, evidence, commit, ..
    } = e
    else {
        return None;
    };
    let gate4 = events.iter().find_map(|ev| match ev {
        Event::GatesVerdict { stage, detail, .. } if stage == "gates" => detail.get("gate4").cloned(),
        _ => None,
    });
    let final_ms = gate4.as_ref().and_then(|g| g.get("final_ms")).and_then(|v| v.as_f64());
    let speedup = match (baseline_ms, final_ms) {
        (Some(b), Some(f)) if f > 0.0 => Some(b / f),
        _ => baseline_ms.map(|b| if ms > 0.0 { b / ms } else { 1.0 }),
    };
    Some(json!({
        "candidate": format!("i{iteration}/c{chain}"),
        "commit": commit,
        "what_changed": change_summary,
        "plan": plan,
        "why": evidence,
        "how": hints,
        "latency_ms": ms,
        "baseline_ms": baseline_ms,
        "speedup": speedup,
        "measurement": json!({
            "scope": "op",
            "metric": "median latency",
            "unit": "ms",
            "shape": gate4.as_ref().and_then(|g| g.get("shape")).cloned().unwrap_or(serde_json::Value::Null),
            "runs": gate4.as_ref().and_then(|g| g.get("runs")).cloned().unwrap_or(serde_json::Value::Null),
            "baseline_ms": baseline_ms,
            "final_ms": final_ms,
            "speedup": speedup,
            "noise_pct": gate4.as_ref().and_then(|g| g.get("noise_pct")).cloned().unwrap_or(serde_json::Value::Null),
            "note": "op-level bench; not an end-to-end model speedup",
        }),
    }))
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Median of a slice of floats (None when empty).
fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(s[s.len() / 2])
}

fn compact(v: &serde_json::Value) -> String {
    v.to_string().replace('\n', " ").chars().take(300).collect()
}

/// Collapse whitespace to a single line, preserving the full text (no truncation).
fn flatten(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Collapse to a single, length-bounded line for progress output.
fn one_line(s: &str, n: usize) -> String {
    let one = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= n {
        one
    } else {
        format!("{}…", one.chars().take(n).collect::<String>())
    }
}

/// Bounded multi-line text (for embedding a contract header in a prompt).
fn truncate_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

/// Model-level Gate 3: the engine must emit the same tokens (correctness) and not
/// be slower than the baseline by more than γ (perf).
fn engine_gate(base: &E2eResult, cand: &E2eResult, gamma: f64) -> (bool, serde_json::Value) {
    let same_tokens = base.digest == cand.digest;
    let not_slower = base.elapsed_s <= 0.0 || cand.elapsed_s <= base.elapsed_s * gamma;
    (
        same_tokens && not_slower,
        json!({
            "same_tokens": same_tokens,
            "not_slower": not_slower,
            "baseline_elapsed_s": base.elapsed_s,
            "candidate_elapsed_s": cand.elapsed_s,
            "gamma": gamma,
            "baseline_text": base.text,
            "candidate_text": cand.text,
        }),
    )
}

/// A memory-bound kernel cannot move data faster than the device's memory
/// roofline — but an L2-resident working set legitimately can (the bench reuses
/// the same buffers). This ceiling is deliberately generous so a real cache-fed
/// win is not rejected; a candidate that *skips work* shows a far larger ratio
/// (the false positive that motivated this check was ~7.8x).
const DEFAULT_MAX_ROOFLINE_RATIO: f64 = 4.0;

/// Gate-4 performance verdict. Pure so it can be unit-tested as a
/// positive/negative control: it must report `optimized` only for a real,
/// repeatable improvement that clears both the noise floor and the configured
/// minimum effect, wins every interleaved round, and is physically plausible.
#[derive(Debug, Clone, PartialEq)]
pub struct PerfVerdict {
    /// Candidate is not slower than `gamma * baseline`.
    pub gate4: bool,
    /// Candidate is a real, repeatable win (safe to call "optimized").
    pub optimized: bool,
    /// Measured improvement is within the measurement noise floor.
    pub within_noise: bool,
    /// Candidate's effective bandwidth is within the plausible roofline ceiling.
    pub plausible: bool,
    /// `baseline/final - 1` (0 when there is no timing).
    pub improvement: f64,
    /// `baseline/final` (1.0 when there is no timing).
    pub speedup: f64,
    /// Candidate effective GB/s / roofline GB/s (1.0 when unknown).
    pub roofline_ratio: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn performance_verdict(
    baseline_ms: f64,
    final_ms: Option<f64>,
    noise_pct: f64,
    min_improvement: f64,
    rounds: u32,
    wins: u32,
    gamma: f64,
    cand_gbs: Option<f64>,
    roofline_gbs: Option<f64>,
    max_roofline_ratio: f64,
) -> PerfVerdict {
    let speedup = final_ms
        .map(|m| if m > 0.0 { baseline_ms / m } else { 1.0 })
        .unwrap_or(1.0);
    let improvement = final_ms
        .map(|m| if m > 0.0 { baseline_ms / m - 1.0 } else { 0.0 })
        .unwrap_or(0.0);
    let gate4 = final_ms.map(|m| m <= baseline_ms * gamma).unwrap_or(false);
    let within_noise = improvement <= noise_pct;
    let won_all = rounds == 0 || wins == rounds;
    let roofline_ratio = match (cand_gbs, roofline_gbs) {
        (Some(c), Some(r)) if r > 0.0 => c / r,
        _ => 1.0,
    };
    let plausible = roofline_ratio <= max_roofline_ratio;
    let optimized = gate4 && improvement > min_improvement && !within_noise && won_all && plausible;
    PerfVerdict {
        gate4,
        optimized,
        within_noise,
        plausible,
        improvement,
        speedup,
        roofline_ratio,
    }
}

/// Outcome of Gate 5 (correctness at the *measured* shape).
#[derive(Debug, Clone, PartialEq)]
enum ShapeGate {
    /// Not applicable or not trustworthy, with a reason.
    Skipped(String),
    Passed(serde_json::Value),
    Failed(serde_json::Value),
}

impl ShapeGate {
    fn detail(&self) -> serde_json::Value {
        match self {
            ShapeGate::Skipped(why) => json!({"status": "skipped", "reason": why}),
            ShapeGate::Passed(d) => json!({"status": "passed", "detail": d}),
            ShapeGate::Failed(d) => json!({"status": "failed", "detail": d}),
        }
    }
}

/// Parse the shape tuple out of a bench label, e.g. `add_bias [4304,4096 ]`.
fn parse_bench_shape(label: Option<&str>) -> Option<Vec<i64>> {
    let label = label?;
    let open = label.rfind('[')?;
    let close = open + label[open..].find(']')?;
    let dims: Vec<i64> = label[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    (dims.len() >= 2).then_some(dims)
}

/// Split on top-level commas (ignoring commas inside quotes or brackets).
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut cur = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '\\' if in_str => {
                cur.push(c);
                if let Some(n) = it.next() {
                    cur.push(n);
                }
            }
            '"' => {
                in_str = !in_str;
                cur.push(c);
            }
            '(' | '{' | '[' if !in_str => {
                depth += 1;
                cur.push(c);
            }
            ')' | '}' | ']' if !in_str => {
                depth -= 1;
                cur.push(c);
            }
            ',' if !in_str && depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Byte offset of the `)` matching the `(` at byte offset `open_byte`.
fn matching_paren(s: &str, open_byte: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut it = s.get(open_byte..)?.char_indices();
    while let Some((off, c)) = it.next() {
        if in_str {
            match c {
                '\\' => {
                    it.next();
                }
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_byte + off);
                }
            }
            _ => {}
        }
    }
    None
}

/// Append an extra `run_case(...)` for the measured shape to a ninfer op test.
///
/// Reuses the *last* existing call so argument positions are preserved (the
/// op tests have varied signatures), and only when the signature is the common
/// `run_case(<label>, <int dims...>, ...)`. Returns `None` otherwise, and the
/// caller then skips the gate. The result is self-validated against the
/// baseline before it is trusted, so a wrong guess cannot reject a good candidate.
fn append_shape_case(source: &str, op: &str, shape: &[i64]) -> Option<String> {
    // 1) Signature: the leading run of integer params after the label param.
    let sig_at = source.find("int run_case(")?;
    let after = &source[sig_at + "int run_case".len()..];
    let sig_open = after.find('(')?;
    let sig_close = matching_paren(after, sig_open)?;
    let params = split_top_level(&after[sig_open + 1..sig_close]);
    if params.is_empty()
        || !(params[0].contains("char*") || params[0].contains("string") || params[0].contains("string_view"))
    {
        return None;
    }
    let is_int = |p: &str| {
        let ty = p.split('=').next().unwrap_or("").trim();
        let first = ty.split_whitespace().next().unwrap_or(ty);
        let first = first.trim_end_matches(['*', '&']);
        matches!(
            first,
            "std::int32_t" | "int32_t" | "std::int64_t" | "int64_t" | "std::int16_t" | "int16_t" | "int"
        )
    };
    let n_shape = params[1..].iter().take_while(|p| is_int(p)).count();
    if n_shape == 0 || n_shape != shape.len() {
        return None;
    }

    // 2) Reuse the last `run_case(...)` call line, substituting the shape args.
    let call_line = source
        .lines()
        .rev()
        .find(|l| l.contains("run_case(") && l.contains('"') && !l.trim_start().starts_with("int run_case"))?;
    let call_at = call_line.find("run_case(")?;
    let args_open = call_at + "run_case(".len() - 1;
    let args_close = matching_paren(call_line, args_open)?;
    let mut args = split_top_level(&call_line[args_open + 1..args_close]);
    if args.len() < 1 + n_shape {
        return None;
    }
    for (i, v) in shape.iter().enumerate() {
        args[1 + i] = v.to_string();
    }
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    args[0] = format!("\"{op} [{dims}] measured-shape\"");
    let prefix = &call_line[..call_at];
    let tail = &call_line[args_close + 1..];
    let new_line = format!("{prefix}run_case({}){}", args.join(", "), tail);

    // 3) Insert right after the reused call.
    let pos = source.rfind(call_line)?;
    let line_end = source[pos..].find('\n').map(|i| pos + i + 1).unwrap_or(source.len());
    let mut out = String::with_capacity(source.len() + new_line.len() + 1);
    out.push_str(&source[..line_end]);
    out.push_str(&new_line);
    out.push('\n');
    out.push_str(&source[line_end..]);
    Some(out)
}

fn first_error(compile: &serde_json::Value) -> String {
    compile["compiler_errors"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|e| e["message"].as_str())
        .unwrap_or("unknown")
        .to_string()
}

fn format_compiler_errors(compile: &serde_json::Value) -> String {
    let errs = compile["compiler_errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if errs.is_empty() {
        let raw = compile["raw_tail"].as_str().unwrap_or("");
        let tail: String = raw.chars().rev().take(1200).collect::<Vec<_>>().into_iter().rev().collect();
        return format!("compile failed (exit {:?}): {tail}", compile["exit_code"]);
    }
    errs.iter()
        .take(8)
        .map(|e| {
            format!(
                "{}:{}: {}",
                e["file"].as_str().unwrap_or("?"),
                e["line"].as_u64().unwrap_or(0),
                e["message"].as_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_verify_failure(verify: &serde_json::Value) -> String {
    let failing = verify["failing_cases"].as_array().cloned().unwrap_or_default();
    if !failing.is_empty() {
        let names: Vec<String> = failing
            .iter()
            .filter_map(|c| c["name"].as_str().map(String::from))
            .take(8)
            .collect();
        return format!("correctness failed: {}", names.join("; "));
    }
    format!(
        "correctness failed (exit {:?}): {}",
        verify["exit_code"],
        verify["suite_output"].as_str().unwrap_or("").chars().take(1500).collect::<String>()
    )
}

fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let after_first_line = rest.splitn(2, '\n').nth(1).unwrap_or(rest);
        let body = after_first_line.strip_suffix("```").unwrap_or(after_first_line);
        return body.trim().to_string();
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compiler_error_formatting() {
        let compile = json!({"compiler_errors": [
            {"file": "a.cuh", "line": 3, "message": "boom"}
        ]});
        assert!(format_compiler_errors(&compile).contains("a.cuh:3: boom"));
    }

    #[test]
    fn verify_failure_lists_names() {
        let v = json!({"failing_cases": [{"name": "ninfer_add_bias_test", "status": "failed"}]});
        assert!(format_verify_failure(&v).contains("ninfer_add_bias_test"));
    }

    #[test]
    fn fences_stripped() {
        assert_eq!(strip_code_fences("```cuda\n__global__ void k(){}\n```"), "__global__ void k(){}");
    }

    #[test]
    fn no_plan_detection() {
        assert!(is_no_plan(""));
        assert!(is_no_plan("no additional modification is meaningful"));
        assert!(is_no_plan("提案已拒绝：未找到任何额外修改"));
        assert!(!is_no_plan(
            "restructure add_bias_bf16x8_kernel to reuse the bias pack across rows"
        ));
    }

    #[test]
    fn diff_summary_counts_changed_lines() {
        let s = diff_summary("a\nb\nc\n", "a\nB\nc\n");
        assert!(s.contains("1 lines changed"), "{s}");
    }

    #[test]
    fn engine_gate_checks_tokens_and_speed() {
        let base = E2eResult { digest: "a".into(), elapsed_s: 5.0, text: "x".into() };
        let same = E2eResult { digest: "a".into(), elapsed_s: 5.0, text: "x".into() };
        assert!(engine_gate(&base, &same, 1.03).0);
        // different tokens => correctness fail
        let diff = E2eResult { digest: "b".into(), elapsed_s: 4.0, text: "y".into() };
        assert!(!engine_gate(&base, &diff, 1.03).0);
        // >3% slower => perf fail
        let slow = E2eResult { digest: "a".into(), elapsed_s: 6.0, text: "x".into() };
        assert!(!engine_gate(&base, &slow, 1.03).0);
        // within the noise margin => pass
        let ok = E2eResult { digest: "a".into(), elapsed_s: 5.1, text: "x".into() };
        assert!(engine_gate(&base, &ok, 1.03).0);
    }

    // ---- Positive/negative controls for the performance gate ---------------- //
    // These prove the gate reports a real improvement (not merely a correct
    // change) and refuses to report one for noise or an unstable win.

    #[test]
    fn perf_verdict_reports_a_real_win() {
        // 30% faster, above the 1% noise floor, wins both interleaved rounds.
        let v = performance_verdict(1.0, Some(0.70), 0.01, 0.01, 2, 2, 1.05, None, None, 2.5);
        assert!(v.optimized, "expected optimized: {v:?}");
        assert!(v.gate4 && !v.within_noise && v.plausible);
        assert!((v.speedup - 1.4286).abs() < 1e-3, "{v:?}");
        assert!((v.improvement - 0.4286).abs() < 1e-3, "{v:?}");
    }

    #[test]
    fn perf_verdict_rejects_correct_but_not_faster() {
        // A correct change that does not beat the baseline => matched, not optimized.
        let v = performance_verdict(1.0, Some(1.0), 0.01, 0.01, 2, 0, 1.05, None, None, 2.5);
        assert!(!v.optimized && v.within_noise);
        assert!(v.gate4);
        // 0.5% faster, but inside the 1% noise floor => matched.
        let v = performance_verdict(1.0, Some(0.995), 0.01, 0.01, 2, 2, 1.05, None, None, 2.5);
        assert!(!v.optimized && v.within_noise);
    }

    #[test]
    fn perf_verdict_rejects_unstable_or_slow_wins() {
        // 20% faster on the median but only won 1 of 2 rounds => matched.
        let v = performance_verdict(1.0, Some(0.80), 0.01, 0.01, 2, 1, 1.05, None, None, 2.5);
        assert!(!v.optimized, "unstable win must not count: {v:?}");
        // Slower than gamma*baseline => gate4 fails (=> fallback).
        let v = performance_verdict(1.0, Some(1.10), 0.01, 0.01, 2, 0, 1.05, None, None, 2.5);
        assert!(!v.gate4 && !v.optimized);
    }

    #[test]
    fn perf_verdict_without_timing_is_not_optimized() {
        // correctness-only targets: gate4 is overridden by !timing in finalize.
        let v = performance_verdict(1.0, None, 0.0, 0.01, 0, 0, 1.05, None, None, 2.5);
        assert!(!v.optimized && !v.gate4);
        assert_eq!(v.speedup, 1.0);
    }

    #[test]
    fn perf_verdict_rejects_implausibly_fast_candidate() {
        // Reproduces a real false positive: a candidate that skipped part of the
        // tensor looked 22x faster and reported 14 TB/s against a 1.8 TB/s
        // roofline (~780%). That is physically impossible, so it must be rejected
        // even though it cleared every other check.
        let v = performance_verdict(
            0.1125, Some(0.00497), 0.012, 0.01, 2, 2, 1.05,
            Some(14_000.0), Some(1_792.0), 4.0,
        );
        assert!(!v.plausible, "{v:?}");
        assert!(!v.optimized, "implausible win must not be optimized: {v:?}");
        assert!(v.roofline_ratio > 7.0, "{v:?}");
        // A genuine vectorization that reaches the roofline stays plausible.
        let ok = performance_verdict(
            0.1125, Some(0.070), 0.012, 0.01, 2, 2, 1.05,
            Some(1_500.0), Some(1_792.0), 4.0,
        );
        assert!(ok.plausible && ok.optimized, "{ok:?}");
        // A real L2-fed win measured at ~2.9x the DRAM roofline must stay
        // plausible (it was verified correct at the measured shape).
        let l2 = performance_verdict(
            0.1127, Some(0.0134), 0.007, 0.01, 2, 2, 1.05,
            Some(5_256.0), Some(1_792.0), 4.0,
        );
        assert!(l2.plausible && l2.optimized, "{l2:?}");
    }

    // ---- Gate 5: measured-shape correctness -------------------------------- //

    #[test]
    fn bench_shape_parses_from_label() {
        assert_eq!(parse_bench_shape(Some("add_bias [4304,4096 ]")), Some(vec![4304, 4096]));
        assert_eq!(parse_bench_shape(Some("gelu [4608,16]")), Some(vec![4608, 16]));
        assert_eq!(parse_bench_shape(Some("no shape here")), None);
        assert_eq!(parse_bench_shape(None), None);
    }

    #[test]
    fn shape_case_is_appended_for_supported_test() {
        let src = "\
int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    return 0;
}
int main() {
    int failures = 0;
    failures += run_case(\"add_bias [1152,1]\", 1152, 1, 101u);
    failures += run_case(\"add_bias [4304,257]\", 4304, 257, 301u);
    return failures;
}
";
        let patched = append_shape_case(src, "add_bias", &[4304, 4096]).unwrap();
        assert!(
            patched.contains("run_case(\"add_bias [4304,4096] measured-shape\", 4304, 4096, 301u);"),
            "{patched}"
        );
        // The original calls are preserved.
        assert!(patched.contains("run_case(\"add_bias [4304,257]\", 4304, 257, 301u);"));
        // The generated case comes after the reused call.
        let gen_at = patched.find("measured-shape").unwrap();
        let orig_at = patched.find("[4304,257]").unwrap();
        assert!(gen_at > orig_at);
    }

    #[test]
    fn shape_case_skips_unsupported_signatures() {
        // An enum param before the dims => unsupported (the gate is skipped).
        let gelu = "int run_case(const char* label, ops::GeluMode mode, std::int32_t rows, std::int32_t columns, std::uint32_t seed) { return 0; }\nint main(){ run_case(\"g\", ops::GeluMode::Tanh, 1, 2, 3u); }";
        assert!(append_shape_case(gelu, "gelu", &[4, 5]).is_none());
        // A shape-vector signature => unsupported.
        let shp = "int run_case(const char* label, const Shape& shape, std::uint32_t seed) { return 0; }\nint main(){ run_case(\"s\", {1,2,3}, 9u); }";
        assert!(append_shape_case(shp, "l2norm", &[4, 5]).is_none());
        // A 3-dim shape against a 2-int signature => unsupported.
        let ab = "int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) { return 0; }\nint main(){ run_case(\"a\", 1, 2, 3u); }";
        assert!(append_shape_case(ab, "add_bias", &[1, 2, 3]).is_none());
    }
}

#[cfg(test)]
mod verify_timeout_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn scales_with_baseline_and_clamps() {
        // 2s baseline → 2*10+30 = 50 → floored to 60.
        assert_eq!(adaptive_verify_timeout(Duration::from_secs(0)), VERIFY_TIMEOUT_FLOOR_S);
        assert_eq!(adaptive_verify_timeout(Duration::from_secs(2)), 60);
        assert_eq!(adaptive_verify_timeout(Duration::from_secs(120)), 1230);
        // A very slow baseline cannot run unbounded.
        assert_eq!(adaptive_verify_timeout(Duration::from_secs(1000)), VERIFY_TIMEOUT_CEIL_S);
    }

    #[test]
    fn failure_signature_uses_first_nonempty_line() {
        assert_eq!(error_signature("line one\nline two"), "line one");
        assert_eq!(error_signature("\n\n   spaced   \nmore"), "spaced");
        assert_eq!(failure_signature("compile", "boom\ndetail"), "compile: boom");
    }

    #[test]
    fn last_run_tail_takes_the_end() {
        let v = serde_json::json!({
            "runs": [{"stdout": "early"}, {"stdout": "abcdefghij", "stderr": "boom"}]
        });
        assert_eq!(last_run_tail(&v, "stdout", 4), "ghij");
        assert_eq!(last_run_tail(&v, "stderr", 10), "boom");
        assert_eq!(last_run_tail(&v, "missing", 4), "");
    }
}
