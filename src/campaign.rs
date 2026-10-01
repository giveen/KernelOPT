//! Campaign driver: point at a directory, discover every kernel target, and
//! keep optimizing each until the per-target policy or the global budget stops.
//!
//! State is persisted to `.kernelopt/campaigns/<id>/state.json` after every
//! target, so a long campaign is resumable (`--resume <id>`). Experience memory
//! and the strategy tracker are shared across targets (paper §4.5 cross-run).

use crate::attribution;
use crate::backend::{self, Backend};
use crate::config::Config;
use crate::cuda_pipeline::{CudaPipeline, E2eConfig, EditMode, PipelineResult, DEFAULT_VERIFY_TIMEOUT_S};
use crate::journal::Journal;
use crate::llm::LlmClient;
use crate::memory::{ExperienceMemory, StrategyTracker};
use crate::runner_bridge::RunnerBridge;
use crate::signals;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct CampaignOptions {
    pub repo: PathBuf,
    pub backend: Backend,
    /// Optional op/family filter (empty = all discovered targets).
    pub only: Vec<String>,
    /// 0 = no cap.
    pub max_targets: usize,
    pub max_iterations: u32,
    pub patience: u32,
    pub target_speedup: f64,
    pub min_improvement: f64,
    /// 0 = no time budget.
    pub budget_seconds: Option<u64>,
    pub llm_call_budget: Option<u64>,
    pub ncu_set: String,
    pub e2e: Option<E2eConfig>,
    /// Editing mode (full-file or unified diff).
    pub edit_mode: EditMode,
    /// "complexity" (default) or "engine" (order targets by measured GPU share).
    pub order: String,
    /// Workload argv for engine-share ordering (run under Graphsignal).
    pub profile_cmd: Vec<String>,
    pub profile_port: u16,
    pub profile_trace: String,
    /// Ctrl-C pause flag, shared with the pipeline.
    pub interrupted: Arc<AtomicBool>,
    /// Live per-iteration progress output.
    pub verbose: bool,
    /// Tail the journal live in this terminal.
    pub watch: bool,
    /// Pin the representative bench shape (substring match on the row label).
    pub bench_shape: Option<String>,
    /// Interleaved baseline/candidate re-bench rounds at finalize.
    pub final_rounds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetState {
    pub op: String,
    pub family: String,
    /// pending | running | optimized | matched | fallback | failed
    pub status: String,
    pub best_speedup: Option<f64>,
    pub iterations: u32,
    pub stop_reason: Option<String>,
    pub run_id: Option<String>,
    pub llm_calls: u64,
    /// Cumulative tokens for this target (delta-accounted across resumes).
    #[serde(default)]
    pub tokens: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignState {
    pub id: String,
    pub repo: String,
    pub backend: String,
    pub created: String,
    pub targets: Vec<TargetState>,
    pub cursor: usize,
    pub llm_calls: u64,
    pub tokens: u64,
    pub elapsed_s: f64,
}

impl CampaignState {
    pub fn summary(&self) -> serde_json::Value {
        let optimized = self.targets.iter().filter(|t| t.status == "optimized").count();
        let matched = self.targets.iter().filter(|t| t.status == "matched").count();
        let failed = self.targets.iter().filter(|t| t.status == "failed").count();
        let best = self
            .targets
            .iter()
            .filter_map(|t| t.best_speedup.map(|s| (s, t.op.clone())))
            .fold(None, |acc: Option<(f64, String)>, (s, op)| match acc {
                Some((bs, _)) if bs >= s => acc,
                _ => Some((s, op)),
            });
        serde_json::json!({
            "campaign": self.id,
            "backend": self.backend,
            "targets": self.targets.len(),
            "processed": self.cursor,
            "optimized": optimized,
            "matched": matched,
            "failed": failed,
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "elapsed_s": self.elapsed_s,
            "best_speedup": best.map(|(s, op)| serde_json::json!({"op": op, "speedup": s})),
        })
    }
}

pub fn campaign_dir(campaign_id: &str) -> Result<PathBuf> {
    Ok(std::env::current_dir()
        .context("cwd")?
        .join(".kernelopt/campaigns")
        .join(campaign_id))
}

/// Load a campaign's persisted state (for `status --campaign`).
pub fn load(campaign_id: &str) -> Result<CampaignState> {
    let path = campaign_dir(campaign_id)?.join("state.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("no campaign state at {}", path.display()))?;
    serde_json::from_str(&text).context("parsing campaign state")
}

/// Run (or resume) a campaign. Returns the final state.
pub fn run_campaign(
    cfg: &Config,
    llm: &dyn LlmClient,
    runner: &RunnerBridge,
    opts: &CampaignOptions,
    campaign_id: &str,
    resume: bool,
) -> Result<CampaignState> {
    let dir = campaign_dir(campaign_id)?;
    std::fs::create_dir_all(&dir)?;
    let state_path = dir.join("state.json");
    let started = Instant::now();

    let mut state = if resume && state_path.exists() {
        let s: CampaignState = serde_json::from_str(&std::fs::read_to_string(&state_path)?)
            .context("parsing campaign state")?;
        println!(
            "resuming campaign {} at target {}/{}",
            s.id,
            s.cursor,
            s.targets.len()
        );
        s
    } else {
        let mut targets = backend::discover_targets(&opts.repo, opts.backend)?;
        if !opts.only.is_empty() {
            targets.retain(|t| {
                opts.only
                    .iter()
                    .any(|o| o.eq_ignore_ascii_case(&t.op) || o.eq_ignore_ascii_case(&t.family))
            });
        }
        if opts.order.eq_ignore_ascii_case("engine") {
            if opts.profile_cmd.is_empty() {
                anyhow::bail!("--order engine requires --profile-cmd <workload …>");
            }
            let kernels = engine_kernel_times(runner, opts)?;
            let (ranking, unattributed) = attribution::rank_targets(&kernels, &targets);
            let share: HashMap<String, f64> =
                ranking.iter().map(|r| (r.op.clone(), r.ns)).collect();
            attribution::order_by_share(&mut targets, &share);
            println!(
                "campaign order: engine-share — {} op(s) attributed, {} kernel(s) unattributed",
                share.len(),
                unattributed.len()
            );
        }
        if targets.is_empty() {
            anyhow::bail!("no targets matched the campaign filter");
        }
        let states: Vec<TargetState> = targets
            .iter()
            .map(|t| TargetState {
                op: t.op.clone(),
                family: t.family.clone(),
                status: "pending".into(),
                best_speedup: None,
                iterations: 0,
                stop_reason: None,
                run_id: None,
                llm_calls: 0,
                tokens: 0,
                error: None,
            })
            .collect();
        CampaignState {
            id: campaign_id.to_string(),
            repo: opts.repo.to_string_lossy().to_string(),
            backend: opts.backend.as_str().to_string(),
            created: chrono::Utc::now().to_rfc3339(),
            targets: states,
            cursor: 0,
            llm_calls: 0,
            tokens: 0,
            elapsed_s: 0.0,
        }
    };

    save_state(&state_path, &state)?;
    println!(
        "campaign {}: {} targets (backend {}), patience={} target_speedup={} max_iterations={}",
        state.id,
        state.targets.len(),
        state.backend,
        opts.patience,
        opts.target_speedup,
        opts.max_iterations
    );

    let deadline = opts
        .budget_seconds
        .filter(|s| *s > 0)
        .map(|s| Instant::now() + Duration::from_secs(s));
    // Experience memory + strategy tracker persist across resume so the Planner
    // does not repeat directions it already learned (paper §4.5 cross-run).
    let memory_path = dir.join("memory.json");
    let tracker_path = dir.join("tracker.json");
    let mut memory = if resume && memory_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&memory_path)?)
            .context("parsing campaign memory")?
    } else {
        ExperienceMemory::new(cfg.hyper.q_memory, cfg.hyper.s_plus, cfg.hyper.s_minus)
    };
    let mut tracker = if resume && tracker_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&tracker_path)?)
            .context("parsing campaign tracker")?
    } else {
        StrategyTracker::default()
    };

    let mut processed = 0usize;
    while state.cursor < state.targets.len() {
        if opts.max_targets > 0 && processed >= opts.max_targets {
            println!("campaign: reached --max-targets ({})", opts.max_targets);
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                println!("campaign: time budget exhausted");
                break;
            }
        }
        if let Some(b) = opts.llm_call_budget {
            if state.llm_calls >= b {
                println!("campaign: LLM-call budget exhausted");
                break;
            }
        }

        let idx = state.cursor;
        let op = state.targets[idx].op.clone();
        // Reuse the run id (and its checkpoint) when resuming a target that was
        // interrupted mid-run, so it continues instead of restarting.
        let was_running = resume && state.targets[idx].status == "running";
        let run_id = if was_running {
            state.targets[idx].run_id.clone().unwrap_or_else(|| new_run_id(&op))
        } else {
            new_run_id(&op)
        };
        state.targets[idx].status = "running".into();
        state.targets[idx].run_id = Some(run_id.clone());
        save_state(&state_path, &state)?;

        let remaining_llm = opts
            .llm_call_budget
            .map(|b| b.saturating_sub(state.llm_calls))
            .filter(|r| *r > 0);
        println!(
            "\n=== [{}/{}] {} (run {run_id}){} ===",
            idx + 1,
            state.targets.len(),
            op,
            if was_running { " [resuming]" } else { "" }
        );

        let result = run_one_target(
            cfg, llm, runner, opts, &op, &run_id, &mut memory, &mut tracker, deadline, remaining_llm,
        );

        match result {
            Ok(res) if res.paused => {
                println!(
                    "  -> paused ({}); resume with:\n     kernelopt campaign --repo {} --mode {} --resume {}",
                    res.stop_reason, state.repo, state.backend, state.id
                );
                state.targets[idx].status = "running".into();
                state.targets[idx].iterations = res.iterations;
                state.targets[idx].stop_reason = Some(res.stop_reason.clone());
                account(&mut state, idx, &res);
                state.elapsed_s = started.elapsed().as_secs_f64();
                save_state(&state_path, &state)?;
                save_campaign_memory(&dir, &memory, &tracker)?;
                break;
            }
            Ok(res) => {
                println!(
                    "  -> {} speedup={:?} iterations={} stop={} llm_calls={}",
                    res.outcome, res.speedup, res.iterations, res.stop_reason, res.llm_calls
                );
                state.targets[idx].status = res.outcome.clone();
                state.targets[idx].best_speedup = res.speedup;
                state.targets[idx].iterations = res.iterations;
                state.targets[idx].stop_reason = Some(res.stop_reason.clone());
                account(&mut state, idx, &res);
                state.cursor += 1;
                processed += 1;
            }
            Err(e) => {
                eprintln!("  -> failed: {e:#}");
                state.targets[idx].status = "failed".into();
                state.targets[idx].error = Some(format!("{e:#}"));
                state.cursor += 1;
                processed += 1;
            }
        }

        state.elapsed_s = started.elapsed().as_secs_f64();
        save_state(&state_path, &state)?;
        save_campaign_memory(&dir, &memory, &tracker)?;
        append_event(&dir, &state.targets[idx])?;
    }

    state.elapsed_s = started.elapsed().as_secs_f64();
    save_state(&state_path, &state)?;
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn run_one_target(
    cfg: &Config,
    llm: &dyn LlmClient,
    runner: &RunnerBridge,
    opts: &CampaignOptions,
    op: &str,
    run_id: &str,
    memory: &mut ExperienceMemory,
    tracker: &mut StrategyTracker,
    deadline: Option<Instant>,
    llm_call_budget: Option<u64>,
) -> Result<PipelineResult> {
    let target = backend::discover_target(&opts.repo, opts.backend, op)?;
    let project_root = std::env::current_dir().context("cwd")?;
    let base = project_root.join(".kernelopt").join(opts.backend.as_str());
    let worktree = base.join("worktree");
    let build_dir = base.join("build");
    let run_dir = project_root.join(&cfg.runs_dir).join(run_id);
    for sub in ["candidates", "bench", "ncu"] {
        std::fs::create_dir_all(run_dir.join(sub))?;
    }
    let run_dir = run_dir.canonicalize().unwrap_or(run_dir);
    let mut journal = Journal::create(&cfg.runs_dir, run_id)?;
    let checkpoint_path = run_dir.join("checkpoint.json");

    let mut pipe = CudaPipeline {
        cfg,
        llm,
        runner,
        journal: &mut journal,
        session_id: run_id.to_string(),
        repo: opts.repo.clone(),
        target,
        worktree,
        build_dir,
        gpu_lock_path: crate::gpu_lock::default_path(&project_root),
        run_dir,
        ncu_set: opts.ncu_set.clone(),
        e2e: opts.e2e.clone(),
        edit_mode: opts.edit_mode,
        baseline_e2e: None,
        base_sha: None,
        last_bench_noise_pct: None,
        last_bench_label: None,
        last_bench_gbs: None,
        last_bench_roofline_gbs: None,
        memory,
        tracker,
        patience: opts.patience,
        target_speedup: opts.target_speedup,
        min_improvement: opts.min_improvement,
        max_iterations: opts.max_iterations,
        deadline,
        llm_call_budget,
        llm_calls: 0,
        tokens: 0,
        checkpoint_path: Some(checkpoint_path),
        interrupted: Some(opts.interrupted.clone()),
        verbose: opts.verbose,
        watch: opts.watch,
        bench_shape: opts.bench_shape.clone(),
        final_rounds: opts.final_rounds,
        verify_timeout_s: std::cell::Cell::new(DEFAULT_VERIFY_TIMEOUT_S),
        recent_failures: Vec::new(),
    };
    pipe.run()
}

fn save_state(path: &Path, state: &CampaignState) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn new_run_id(op: &str) -> String {
    format!(
        "{}_camp_{}",
        chrono::Utc::now().format("%Y%m%d_%H%M%S"),
        sanitize(op)
    )
}

/// Delta-account a target's LLM usage so a resumed target is not double-counted.
fn account(state: &mut CampaignState, idx: usize, res: &PipelineResult) {
    state.llm_calls += res.llm_calls.saturating_sub(state.targets[idx].llm_calls);
    state.tokens += res.tokens.saturating_sub(state.targets[idx].tokens);
    state.targets[idx].llm_calls = res.llm_calls;
    state.targets[idx].tokens = res.tokens;
}

/// Persist cross-target learning so `--resume` does not relearn it.
fn save_campaign_memory(
    dir: &Path,
    memory: &ExperienceMemory,
    tracker: &StrategyTracker,
) -> Result<()> {
    std::fs::write(dir.join("memory.json"), serde_json::to_string_pretty(memory)?)?;
    std::fs::write(dir.join("tracker.json"), serde_json::to_string_pretty(tracker)?)?;
    Ok(())
}

/// Rank kernels by GPU time in the campaign's workload. Uses the NVIDIA suite
/// (nsys → ncu) by default, falling back to Graphsignal when neither is present.
fn engine_kernel_times(runner: &RunnerBridge, opts: &CampaignOptions) -> Result<Vec<(String, f64)>> {
    match crate::engine_share::auto_engine() {
        "nsys" => return crate::engine_share::nsys_kernel_times(&opts.profile_cmd, 1800),
        "ncu" => return crate::engine_share::ncu_kernel_times(&opts.profile_cmd, 1800),
        _ => {}
    }
    let managed = std::env::current_dir()
        .context("cwd")?
        .join(".kernelopt/graphsignal");
    let resp = runner.call_with_timeout(
        &json!({
            "command": "graphsignal_profile",
            "cmd": opts.profile_cmd,
            "listen_port": opts.profile_port,
            "cuda_graph_trace": opts.profile_trace,
            "auto_setup": true,
            "managed_dir": managed,
            "timeout_s": 1800,
        }),
        1900,
    )?;
    if resp["ok"] != json!(true) {
        anyhow::bail!(
            "engine-share profiling failed: {}",
            resp["error"]["message"].as_str().unwrap_or("unknown")
        );
    }
    Ok(signals::kernel_times(&resp["signals"], 100))
}

fn append_event(dir: &Path, t: &TargetState) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("campaign.jsonl"))?;
    writeln!(f, "{}", serde_json::to_string(t)?)?;
    Ok(())
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_op_names() {
        assert_eq!(sanitize("SOFT_MAX"), "SOFT_MAX");
        assert_eq!(sanitize("fp8/linear.add"), "fp8_linear_add");
    }

    #[test]
    fn summary_counts() {
        let state = CampaignState {
            id: "c".into(),
            repo: "/r".into(),
            backend: "ninfer".into(),
            created: "now".into(),
            targets: vec![
                TargetState { op: "a".into(), family: "a".into(), status: "optimized".into(), best_speedup: Some(1.5), iterations: 2, stop_reason: None, run_id: None, llm_calls: 1, tokens: 0, error: None },
                TargetState { op: "b".into(), family: "b".into(), status: "failed".into(), best_speedup: None, iterations: 0, stop_reason: None, run_id: None, llm_calls: 0, tokens: 0, error: Some("x".into()) },
            ],
            cursor: 2,
            llm_calls: 1,
            tokens: 10,
            elapsed_s: 1.0,
        };
        let s = state.summary();
        assert_eq!(s["optimized"], 1);
        assert_eq!(s["failed"], 1);
        assert_eq!(s["best_speedup"]["op"], "a");
    }

    #[test]
    fn account_is_delta_based_across_resumes() {
        let mut state = CampaignState {
            id: "c".into(),
            repo: "/r".into(),
            backend: "ninfer".into(),
            created: "now".into(),
            targets: vec![TargetState {
                op: "a".into(),
                family: "a".into(),
                status: "running".into(),
                best_speedup: None,
                iterations: 1,
                stop_reason: None,
                run_id: Some("r".into()),
                llm_calls: 5,
                tokens: 50,
                error: None,
            }],
            cursor: 0,
            llm_calls: 5,
            tokens: 50,
            elapsed_s: 0.0,
        };
        let res = PipelineResult {
            outcome: "paused".into(),
            speedup: None,
            baseline_ms: None,
            final_ms: None,
            iterations: 2,
            stop_reason: "budget".into(),
            llm_calls: 7, // includes the 5 already spent before the pause
            tokens: 70,
            diff_path: None,
            root_cause: None,
            paused: true,
            summary: None,
        };
        account(&mut state, 0, &res);
        assert_eq!(state.llm_calls, 7); // 5 + (7 - 5), not 5 + 7
        assert_eq!(state.tokens, 70);
        assert_eq!(state.targets[0].llm_calls, 7);
    }
}
