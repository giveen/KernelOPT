//! kernelopt — dispatch-aware agentic GPU kernel optimization.

use kernelopt::attribution;
use kernelopt::backend::{self, Backend, Target};
use kernelopt::campaign::{self, CampaignOptions};
use kernelopt::config::{self, Config, Hyper, ProfilerMode};
use kernelopt::cuda_pipeline::{winner_summary, CudaPipeline, E2eConfig, EditMode, PipelineResult};
use kernelopt::journal::{format_event, Event, Journal};
use kernelopt::llm::{self, LlmClient};
use kernelopt::memory;
use kernelopt::pipeline;
use kernelopt::runner_bridge::RunnerBridge;
use kernelopt::signals;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "kernelopt", version, about = "LLM-driven GPU kernel optimization for compiled PyTorch models, ninfer CUDA Ops, and llama.cpp ggml-cuda kernels")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Optimize a compiled PyTorch model file (get_model/get_inputs convention).
    Run {
        /// Path to the model file (Python; must expose `get_model()` and `get_inputs()`).
        model_file: String,
        /// Provider preset (env: KERNELOPT_PROVIDER).
        #[arg(long)]
        provider: Option<String>,
        /// Model id (env: KERNELOPT_MODEL).
        #[arg(long)]
        model: Option<String>,
        /// Custom OpenAI-compatible base URL (env: KERNELOPT_BASE_URL).
        #[arg(long)]
        base_url: Option<String>,
        /// API key (env: KERNELOPT_API_KEY, else the preset's key var).
        #[arg(long)]
        api_key: Option<String>,
        /// Profiling tier.
        #[arg(long, value_enum, default_value = "both")]
        profiler: ProfilerMode,
        /// NCU section set (full = paper; basic for small-context models).
        #[arg(long, default_value = "full")]
        ncu_set: String,
        /// Inputs function name in the model file.
        #[arg(long, default_value = "get_inputs")]
        inputs_fn: String,
        /// Thinking level: none|minimal|low|medium|high (alias: --thinking).
        #[arg(long, visible_alias = "thinking")]
        reasoning_effort: Option<String>,
        /// Beam iterations T (default 5).
        #[arg(long)]
        iterations: Option<u32>,
        /// Plans per iteration N (default 4).
        #[arg(long)]
        plans: Option<u32>,
        /// Beam width B (default 4).
        #[arg(long)]
        beam: Option<u32>,
        /// Executor retries K (default 4).
        #[arg(long)]
        retries: Option<u32>,
    },
    /// Optimize one ninfer Op's CUDA kernel in an isolated git worktree.
    RunNinfer {
        /// ninfer op token, e.g. `add_bias`, `bf16_linear_add`, `fp8_linear_swiglu`.
        #[arg(long)]
        op: String,
        /// ninfer checkout path (env: NINFER_REPO).
        #[arg(long)]
        repo: Option<String>,
        #[command(flatten)]
        llm: LlmArgs,
        /// Editable kernel file (repo-relative); default: first from `discover`.
        #[arg(long)]
        kernel_file: Option<String>,
        /// CMake build dir (default: .kernelopt/ninfer/build, configured from the worktree).
        #[arg(long)]
        build_dir: Option<String>,
        /// Extra argv forwarded to the op bench binary (repeatable).
        #[arg(long = "bench-arg")]
        bench_args: Vec<String>,
        /// Model for the engine E2E check: `.ninfer`, `.gguf`, or an HF
        /// safetensors directory. Alias: --e2e-model.
        #[arg(long, visible_alias = "e2e-model")]
        e2e_weights: Option<String>,
        /// Engine override for E2E: ninfer | llamacpp | hf | vllm | sglang.
        #[arg(long)]
        e2e_engine: Option<String>,
        /// Custom E2E command (argv after --e2e-cmd; place last).
        #[arg(long = "e2e-cmd", num_args = 1.., allow_hyphen_values = true)]
        e2e_cmd: Vec<String>,
        /// Prompt for the engine E2E check.
        #[arg(long, default_value = "The capital of France is")]
        e2e_prompt: String,
        /// Generated tokens for the engine E2E check.
        #[arg(long, default_value_t = 16)]
        e2e_max_new: u32,
        /// Reuse a run id (loads its checkpoint to resume a paused run).
        #[arg(long)]
        run_id: Option<String>,
        /// NCU section set for planner context (full|basic).
        #[arg(long, default_value = "full")]
        ncu_set: String,
        /// Suppress live progress (stderr).
        #[arg(long)]
        quiet: bool,
        /// Tail the journal live in this terminal (no second terminal needed).
        #[arg(long)]
        watch: bool,
        /// Pin the representative bench shape (substring of the bench row label).
        #[arg(long)]
        bench_shape: Option<String>,
        /// Interleaved baseline/candidate re-bench rounds at finalize (default 2).
        #[arg(long, default_value_t = 2)]
        final_rounds: u32,
        /// Executor edit format: `full` (whole file, default) | `patch` (unified diff).
        /// This is internal only — the run always outputs a reviewable unified diff.
        #[arg(long, default_value = "full")]
        edit_mode: String,
        #[command(flatten)]
        loop_: LoopArgs,
    },
    /// Optimize one llama.cpp ggml-cuda kernel in an isolated git worktree.
    RunLlamacpp {
        /// ggml op name (e.g. `SOFT_MAX`) or kernel stem (e.g. `softmax`).
        #[arg(long)]
        op: String,
        /// llama.cpp checkout path (env: LLAMACPP_REPO).
        #[arg(long)]
        repo: Option<String>,
        #[command(flatten)]
        llm: LlmArgs,
        /// Editable kernel file (repo-relative); default: first from `discover`.
        #[arg(long)]
        kernel_file: Option<String>,
        /// CMake build dir (default: .kernelopt/llamacpp/build, configured from the worktree).
        #[arg(long)]
        build_dir: Option<String>,
        /// Extra argv forwarded to `test-backend-ops perf` (repeatable).
        #[arg(long = "bench-arg")]
        bench_args: Vec<String>,
        /// Model for the engine E2E check: `.gguf`, `.ninfer`, or HF safetensors dir.
        #[arg(long, visible_alias = "e2e-model")]
        e2e_weights: Option<String>,
        /// Engine override for E2E: llamacpp | ninfer | hf | vllm | sglang.
        #[arg(long)]
        e2e_engine: Option<String>,
        /// Custom E2E command (argv after --e2e-cmd; place last).
        #[arg(long = "e2e-cmd", num_args = 1.., allow_hyphen_values = true)]
        e2e_cmd: Vec<String>,
        #[arg(long, default_value = "The capital of France is")]
        e2e_prompt: String,
        #[arg(long, default_value_t = 16)]
        e2e_max_new: u32,
        /// Reuse a run id (loads its checkpoint to resume a paused run).
        #[arg(long)]
        run_id: Option<String>,
        /// NCU section set for planner context (full|basic).
        #[arg(long, default_value = "basic")]
        ncu_set: String,
        /// Suppress live progress (stderr).
        #[arg(long)]
        quiet: bool,
        /// Tail the journal live in this terminal (no second terminal needed).
        #[arg(long)]
        watch: bool,
        /// Pin the representative bench shape (substring of the bench row label).
        #[arg(long)]
        bench_shape: Option<String>,
        /// Interleaved baseline/candidate re-bench rounds at finalize (default 2).
        #[arg(long, default_value_t = 2)]
        final_rounds: u32,
        /// Executor edit format: `full` (whole file, default) | `patch` (unified diff).
        /// This is internal only — the run always outputs a reviewable unified diff.
        #[arg(long, default_value = "full")]
        edit_mode: String,
        #[command(flatten)]
        loop_: LoopArgs,
    },
    /// Point at a directory: discover every kernel and optimize them in a loop.
    Campaign {
        /// Target checkout (env: NINFER_REPO/LLAMACPP_REPO).
        #[arg(long)]
        repo: Option<String>,
        /// Backend: auto | ninfer | llamacpp.
        #[arg(long, default_value = "auto")]
        mode: String,
        /// Restrict to specific ops/families (repeatable).
        #[arg(long = "op")]
        ops: Vec<String>,
        /// Stop after this many targets (0 = all).
        #[arg(long, default_value_t = 0)]
        max_targets: usize,
        /// Wall-clock budget in hours (0 = none).
        #[arg(long, default_value_t = 0.0)]
        budget_hours: f64,
        /// Stop the campaign after this many LLM calls (0 = none).
        #[arg(long, default_value_t = 0)]
        budget_llm_calls: u64,
        /// No-improvement iterations before moving to the next target.
        #[arg(long, default_value_t = 2)]
        patience: u32,
        /// Stop a target once baseline/best reaches this speedup (0 = off).
        #[arg(long, default_value_t = 0.0)]
        target_speedup: f64,
        /// Minimum fractional improvement that resets patience (0.01 = 1%).
        #[arg(long, default_value_t = 0.01)]
        min_improvement: f64,
        /// Hard cap on iterations per target.
        #[arg(long, default_value_t = 25)]
        max_iterations: u32,
        /// Model for the engine E2E check: `.ninfer`, `.gguf`, or HF safetensors dir.
        #[arg(long, visible_alias = "e2e-model")]
        e2e_weights: Option<String>,
        /// Engine override for E2E: ninfer | llamacpp | hf | vllm | sglang.
        #[arg(long)]
        e2e_engine: Option<String>,
        /// Custom E2E command (argv after --e2e-cmd; place last).
        #[arg(long = "e2e-cmd", num_args = 1.., allow_hyphen_values = true)]
        e2e_cmd: Vec<String>,
        /// NCU section set (full|basic).
        #[arg(long, default_value = "full")]
        ncu_set: String,
        /// Resume a previous campaign by id.
        #[arg(long)]
        resume: Option<String>,
        /// Target order: `complexity` (default) or `engine` (by measured GPU share).
        #[arg(long, default_value = "complexity")]
        order: String,
        /// Workload for engine-share ordering (argv after `--profile-cmd`; place last).
        #[arg(long = "profile-cmd", num_args = 1.., allow_hyphen_values = true)]
        profile_cmd: Vec<String>,
        /// Graphsignal `/signals` port for engine-share ordering.
        #[arg(long, default_value_t = 18259)]
        profile_port: u16,
        /// CUDA graph trace for engine-share ordering: node | graph.
        #[arg(long, default_value = "node")]
        profile_trace: String,
        /// Suppress live progress (stderr).
        #[arg(long)]
        quiet: bool,
        /// Tail the journal live in this terminal (no second terminal needed).
        #[arg(long)]
        watch: bool,
        /// Pin the representative bench shape (substring of the bench row label).
        #[arg(long)]
        bench_shape: Option<String>,
        /// Interleaved baseline/candidate re-bench rounds at finalize (default 2).
        #[arg(long, default_value_t = 2)]
        final_rounds: u32,
        /// Executor edit format: `full` (whole file, default) | `patch` (unified diff).
        /// This is internal only — the run always outputs a reviewable unified diff.
        #[arg(long, default_value = "full")]
        edit_mode: String,
        #[command(flatten)]
        llm: LlmArgs,
        #[command(flatten)]
        loop_: LoopArgs,
    },
    /// Rank kernels by real engine share using Graphsignal (attribution only).
    Profile {
        /// Target checkout (env: NINFER_REPO/LLAMACPP_REPO).
        #[arg(long)]
        repo: Option<String>,
        /// Backend: auto | ninfer | llamacpp.
        #[arg(long, default_value = "auto")]
        mode: String,
        /// Workload to run under `graphsignal-run` (argv after --cmd).
        #[arg(long = "cmd", required = true, num_args = 1.., allow_hyphen_values = true)]
        cmd: Vec<String>,
        #[arg(long, default_value_t = 18259)]
        listen_port: u16,
        /// CUDA graph trace granularity: node | graph.
        #[arg(long, default_value = "node")]
        cuda_graph_trace: String,
        /// Max kernels to rank.
        #[arg(long, default_value_t = 30)]
        top: usize,
        /// Working directory for the workload.
        #[arg(long)]
        cwd: Option<String>,
        /// Skip auto-provisioning if `graphsignal-run` is missing.
        #[arg(long)]
        no_setup: bool,
        /// Auto-provision install source (path, git URL, or "pypi").
        #[arg(long)]
        source: Option<String>,
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Provision Graphsignal into KernelOPT's managed venv (no separate install).
    SetupGraphsignal {
        /// CUDA major version (12|13); default detected from nvcc/CUDA_HOME.
        #[arg(long)]
        cuda: Option<String>,
        /// Install from a local checkout or git URL (default: the llama.cpp/NInfer fork).
        #[arg(long)]
        source: Option<String>,
        /// Pin a PyPI version (default: latest).
        #[arg(long)]
        version: Option<String>,
        /// Install upstream PyPI graphsignal (no llama.cpp/NInfer launchers).
        #[arg(long)]
        upstream: bool,
        /// Reinstall even if already present.
        #[arg(long)]
        force: bool,
    },
    /// Inventory kernel targets in a directory (one op, or every target).
    Discover {
        /// Show one target (omit to list all with --list).
        #[arg(long)]
        op: Option<String>,
        /// Target checkout (env: NINFER_REPO/LLAMACPP_REPO).
        #[arg(long)]
        repo: Option<String>,
        /// Backend: auto | ninfer | llamacpp.
        #[arg(long, default_value = "auto")]
        mode: String,
        /// List every runnable target instead of one op.
        #[arg(long)]
        list: bool,
    },
    /// Resume an interrupted run by id (replays its journal).
    Resume {
        /// Run id, e.g. `20260928_203052_opencode-go`.
        run_id: String,
    },
    /// Show status of a run (or a campaign with --campaign).
    Status {
        /// Run id or campaign id.
        run_id: String,
        /// Treat the id as a campaign id.
        #[arg(long)]
        campaign: bool,
    },
    /// Tail a run's journal live (progress monitor).
    Watch {
        /// Run id (omit to watch the most recent run).
        run_id: Option<String>,
        /// Watch the most recent run.
        #[arg(long)]
        latest: bool,
        /// Print the existing journal and exit (don't follow).
        #[arg(long)]
        once: bool,
        /// Keep following after the run finishes (e.g. to watch a campaign's
        /// next target); by default `watch` stops and prints artifacts.
        #[arg(long)]
        follow: bool,
    },
    /// List a run's candidate commits (sha, status, latency, plan).
    History {
        /// Run id.
        run_id: String,
    },
    /// Revert the run's worktree to a candidate commit (sha or tag).
    Revert {
        /// Run id.
        run_id: String,
        /// Commit sha or tag from `history`.
        #[arg(long)]
        to: String,
    },
    /// Diagnose a run from its journal (failures, retries, tokens, plan diversity).
    Analyze {
        /// Run id.
        run_id: String,
        /// Emit JSON instead of tables.
        #[arg(long)]
        json: bool,
    },
    /// Aggregate a campaign into a self-benchmark (win rate, speedups, cost).
    Eval {
        /// Campaign id.
        campaign_id: String,
    },
    /// Render the final markdown report for a run.
    Report {
        /// Run id.
        run_id: String,
    },
    /// List provider presets.
    Providers {
        /// Provider preset to probe (default: opencode-go).
        #[arg(long)]
        provider: Option<String>,
        /// Model id to look for in the provider's model list.
        #[arg(long)]
        model: Option<String>,
        /// Custom OpenAI-compatible base URL.
        #[arg(long)]
        base_url: Option<String>,
        /// API key (else the provider's key env var).
        #[arg(long)]
        api_key: Option<String>,
        /// Skip the 1-token completion probe (only GET /models).
        #[arg(long)]
        no_ping: bool,
    },
    /// List local models usable for the engine-E2E (Gate 3) check.
    Models {
        /// Filter by backend engine (ninfer|llamacpp|hf).
        #[arg(long)]
        mode: Option<String>,
        /// Repo whose `models/` dir is searched (env: NINFER_REPO/LLAMACPP_REPO).
        #[arg(long)]
        repo: Option<String>,
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
}

#[derive(clap::Args, Clone)]
struct LlmArgs {
    /// Provider preset (opencode-go|openai|openrouter|ollama|vllm|lmstudio|mock).
    /// Env fallback: KERNELOPT_PROVIDER.
    #[arg(long)]
    provider: Option<String>,
    /// Model id at the provider. Env fallback: KERNELOPT_MODEL.
    #[arg(long)]
    model: Option<String>,
    /// Custom OpenAI-compatible base URL. Env fallback: KERNELOPT_BASE_URL.
    #[arg(long)]
    base_url: Option<String>,
    /// API key. Env fallback: KERNELOPT_API_KEY, else the preset's key env var.
    #[arg(long)]
    api_key: Option<String>,
    /// Thinking level sent as OpenAI `reasoning_effort`
    /// (none|minimal|low|medium|high). Defaults to "low"; overridable in
    /// config.toml or KERNELOPT_REASONING_EFFORT. Alias: --thinking.
    #[arg(long, visible_alias = "thinking")]
    reasoning_effort: Option<String>,
}

impl LlmArgs {
    fn resolve(&self) -> ResolvedLlm {
        resolve_llm(
            self.provider.clone(),
            self.model.clone(),
            self.base_url.clone(),
            self.api_key.clone(),
            self.reasoning_effort.clone(),
        )
    }
}

/// LLM settings after applying env fallbacks (CLI > env > built-in default).
struct ResolvedLlm {
    provider: String,
    model: String,
    base_url: Option<String>,
    api_key: Option<String>,
    reasoning_effort: Option<String>,
}

fn env_opt(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn resolve_llm(
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    reasoning_effort: Option<String>,
) -> ResolvedLlm {
    ResolvedLlm {
        provider: provider
            .or_else(|| env_opt("KERNELOPT_PROVIDER"))
            .unwrap_or_else(|| "opencode-go".to_string()),
        model: model
            .or_else(|| env_opt("KERNELOPT_MODEL"))
            .unwrap_or_else(|| "deepseek-v4-pro".to_string()),
        base_url: base_url.or_else(|| env_opt("KERNELOPT_BASE_URL")),
        api_key: api_key.or_else(|| env_opt("KERNELOPT_API_KEY")),
        reasoning_effort: reasoning_effort.or_else(|| env_opt("KERNELOPT_REASONING_EFFORT")),
    }
}

/// Build the engine-E2E (Gate 3) config. Enabled by `--e2e-weights` **or**
/// `--e2e-cmd`. `--e2e-weights` also reads `KERNELOPT_E2E_WEIGHTS`, and accepts
/// `auto` or a bare local-model name (see `kernelopt models`).
#[allow(clippy::too_many_arguments)]
fn build_e2e(
    weights: Option<String>,
    engine: Option<String>,
    cmd: Vec<String>,
    prompt: String,
    max_new: u32,
    repo: &std::path::Path,
    backend: Backend,
) -> Result<Option<E2eConfig>> {
    let has_cmd = !cmd.is_empty();
    let weights = weights.or_else(|| env_opt("KERNELOPT_E2E_WEIGHTS"));
    if weights.is_none() && !has_cmd {
        return Ok(None);
    }
    let model = match weights {
        // A custom command carries its own model path; keep the spec verbatim.
        Some(spec) if has_cmd => kernelopt::dotenv::expand_tilde(&spec),
        Some(spec) => kernelopt::models::resolve(&spec, repo, Some(backend.as_str()))?
            .to_string_lossy()
            .to_string(),
        None => String::new(),
    };
    Ok(Some(E2eConfig {
        model,
        engine,
        cmd: has_cmd.then_some(cmd),
        prompt,
        max_new,
        seed: 0,
    }))
}

#[derive(clap::Args, Clone)]
struct LoopArgs {
    /// Beam iterations T (single-target runs; campaign uses --max-iterations).
    #[arg(long)]
    iterations: Option<u32>,
    /// Plans per iteration N.
    #[arg(long)]
    plans: Option<u32>,
    /// Beam width B.
    #[arg(long)]
    beam: Option<u32>,
    /// Executor retries K.
    #[arg(long)]
    retries: Option<u32>,
}

fn hyper_from(llm: &LlmArgs, lp: &LoopArgs, default_t: u32) -> Hyper {
    let _ = llm;
    Hyper {
        t_iterations: lp.iterations.unwrap_or(default_t),
        n_plans: lp.plans.unwrap_or(4),
        k_retries: lp.retries.unwrap_or(4),
        b_beam: lp.beam.unwrap_or(4),
        ..Default::default()
    }
}

fn main() -> Result<()> {
    // `.env` (or $KERNELOPT_ENV_FILE) supplies paths, provider, model, key, and
    // effort; the real process environment always takes precedence.
    if let Some((path, n)) = kernelopt::dotenv::load_default() {
        if n > 0 {
            eprintln!("[env] loaded {n} variable(s) from {}", path.display());
        }
    }
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run {
            model_file,
            provider,
            model,
            base_url,
            api_key,
            profiler,
            ncu_set,
            inputs_fn,
            reasoning_effort,
            iterations,
            plans,
            beam,
            retries,
        } => {
            let hyper = Hyper {
                t_iterations: iterations.unwrap_or(5),
                n_plans: plans.unwrap_or(4),
                k_retries: retries.unwrap_or(4),
                b_beam: beam.unwrap_or(4),
                ..Default::default()
            };
            let llm = resolve_llm(provider, model, base_url, api_key, reasoning_effort);
            let cfg = Config::load(
                llm.provider, llm.model, llm.base_url, llm.api_key, hyper, profiler, ncu_set,
                inputs_fn, llm.reasoning_effort,
            )?;
            let run_id = format!("{}_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"), &cfg.provider);
            let mut journal = Journal::create(&cfg.runs_dir, &run_id)?;
            println!("run id: {run_id}");
            let client = build_llm(&cfg);
            let runner = RunnerBridge::new(cfg.runner_dir.clone());
            let wrapped_runner = RunnerBridge::wrapped(cfg.runner_dir.clone(), 18299, "node");
            let model_file = kernelopt::dotenv::expand_tilde(&model_file);
            let model_abs = std::fs::canonicalize(&model_file)
                .with_context(|| format!("model file not found: {model_file}"))?
                .to_string_lossy()
                .to_string();
            let mut pipe = pipeline::Pipeline {
                cfg: &cfg,
                llm: client.as_ref(),
                runner: &runner,
                wrapped_runner: &wrapped_runner,
                journal: &mut journal,
                session_id: run_id.clone(),
                model_file: model_abs.clone(),
                baseline_weights: None,
                memory: memory::ExperienceMemory::new(cfg.hyper.q_memory, cfg.hyper.s_plus, cfg.hyper.s_minus),
                tracker: memory::StrategyTracker::default(),
                next_signals_port: 18299,
                last_bench_signals: None,
            };
            let result = pipe.run(&model_abs)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }

        Cmd::RunNinfer { op, repo, llm: llm_args, kernel_file, build_dir, bench_args, e2e_weights, e2e_engine, e2e_cmd, e2e_prompt, e2e_max_new, run_id, ncu_set, quiet, watch, bench_shape, final_rounds, edit_mode, loop_ } => {
            let repo = resolve_repo(repo, &["NINFER_REPO"], "NINFER_REPO")?;
            let backend = Backend::Ninfer;
            let cfg = load_cuda_config(&llm_args, &loop_, ProfilerMode::Ncu, ncu_set)?;
            let target = backend::discover_target(&repo, backend, &op)?;
            let client = build_llm(&cfg);
            let runner = RunnerBridge::new(cfg.runner_dir.clone());
            let session_id = run_id.unwrap_or_else(|| format!("{}_{}_ninfer", chrono::Utc::now().format("%Y%m%d_%H%M%S"), cfg.provider));
            let e2e = build_e2e(e2e_weights, e2e_engine, e2e_cmd, e2e_prompt, e2e_max_new, &repo, backend)?;
            let result = run_single(&cfg, client.as_ref(), &runner, target, SingleRunOpts {
                repo,
                session_id: session_id.clone(),
                backend,
                ncu_set: cfg.ncu_set.clone(),
                e2e,
                kernel_file,
                bench_args,
                build_dir: build_dir.map(|p| PathBuf::from(kernelopt::dotenv::expand_tilde(&p))),
                patience: 2,
                target_speedup: 0.0,
                min_improvement: 0.01,
                max_iterations: loop_.iterations.unwrap_or(5),
                interrupted: install_interrupt(),
                edit_mode: EditMode::parse(&edit_mode)?,
                verbose: !quiet && !watch,
                watch,
                bench_shape,
                final_rounds,
            })?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if result.paused {
                eprintln!("paused; re-run with --run-id {session_id} to continue");
            }
            Ok(())
        }

        Cmd::RunLlamacpp { op, repo, llm: llm_args, kernel_file, build_dir, bench_args, e2e_weights, e2e_engine, e2e_cmd, e2e_prompt, e2e_max_new, run_id, ncu_set, quiet, watch, bench_shape, final_rounds, edit_mode, loop_ } => {
            let repo = resolve_repo(repo, &["LLAMACPP_REPO"], "LLAMACPP_REPO")?;
            let backend = Backend::Llamacpp;
            let cfg = load_cuda_config(&llm_args, &loop_, ProfilerMode::Ncu, ncu_set)?;
            let target = backend::discover_target(&repo, backend, &op)?;
            let client = build_llm(&cfg);
            let runner = RunnerBridge::new(cfg.runner_dir.clone());
            let session_id = run_id.unwrap_or_else(|| format!("{}_{}_llamacpp", chrono::Utc::now().format("%Y%m%d_%H%M%S"), cfg.provider));
            let e2e = build_e2e(e2e_weights, e2e_engine, e2e_cmd, e2e_prompt, e2e_max_new, &repo, backend)?;
            let result = run_single(&cfg, client.as_ref(), &runner, target, SingleRunOpts {
                repo,
                session_id: session_id.clone(),
                backend,
                ncu_set: cfg.ncu_set.clone(),
                e2e,
                kernel_file,
                bench_args,
                build_dir: build_dir.map(|p| PathBuf::from(kernelopt::dotenv::expand_tilde(&p))),
                patience: 2,
                target_speedup: 0.0,
                min_improvement: 0.01,
                max_iterations: loop_.iterations.unwrap_or(5),
                interrupted: install_interrupt(),
                edit_mode: EditMode::parse(&edit_mode)?,
                verbose: !quiet && !watch,
                watch,
                bench_shape,
                final_rounds,
            })?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if result.paused {
                eprintln!("paused; re-run with --run-id {session_id} to continue");
            }
            Ok(())
        }

        Cmd::Campaign {
            repo,
            mode,
            ops,
            max_targets,
            budget_hours,
            budget_llm_calls,
            patience,
            target_speedup,
            min_improvement,
            max_iterations,
            e2e_weights,
            e2e_engine,
            e2e_cmd,
            ncu_set,
            resume,
            order,
            profile_cmd,
            profile_port,
            profile_trace,
            quiet,
            watch,
            bench_shape,
            final_rounds,
            edit_mode,
            llm: llm_args,
            loop_,
        } => {
            let repo = resolve_repo(
                repo,
                &["NINFER_REPO", "LLAMACPP_REPO"],
                "NINFER_REPO or LLAMACPP_REPO",
            )?;
            let backend = backend::resolve_backend(&repo, Some(&mode))?;
            let cfg = load_cuda_config(&llm_args, &loop_, ProfilerMode::Ncu, ncu_set)?;
            let client = build_llm(&cfg);
            let runner = RunnerBridge::new(cfg.runner_dir.clone());

            let campaign_id = resume.clone().unwrap_or_else(|| {
                format!("{}_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"), backend.as_str())
            });
            let e2e = build_e2e(
                e2e_weights,
                e2e_engine,
                e2e_cmd,
                "The capital of France is".into(),
                16,
                &repo,
                backend,
            )?;
            let opts = CampaignOptions {
                repo,
                backend,
                only: ops,
                max_targets,
                max_iterations,
                patience,
                target_speedup,
                min_improvement,
                budget_seconds: if budget_hours > 0.0 {
                    Some((budget_hours * 3600.0) as u64)
                } else {
                    None
                },
                llm_call_budget: if budget_llm_calls > 0 { Some(budget_llm_calls) } else { None },
                ncu_set: cfg.ncu_set.clone(),
                e2e,
                order,
                profile_cmd,
                profile_port,
                profile_trace,
                interrupted: install_interrupt(),
                edit_mode: EditMode::parse(&edit_mode)?,
                verbose: !quiet && !watch,
                watch,
                bench_shape,
                final_rounds,
            };
            let state = campaign::run_campaign(
                &cfg,
                client.as_ref(),
                &runner,
                &opts,
                &campaign_id,
                resume.is_some(),
            )?;
            println!("{}", serde_json::to_string_pretty(&state.summary())?);
            Ok(())
        }

        Cmd::SetupGraphsignal { cuda, source, version, upstream, force } => {
            let runner = RunnerBridge::new(PathBuf::from("runner"));
            let managed = std::env::current_dir()?.join(".kernelopt/graphsignal");
            let source = if upstream { Some("pypi".to_string()) } else { source };
            let resp = runner.call_with_timeout(
                &serde_json::json!({
                    "command": "graphsignal_setup",
                    "cuda": cuda,
                    "source": source,
                    "version": version,
                    "force": force,
                    "managed_dir": managed,
                    "timeout_s": 1800,
                }),
                1900,
            )?;
            if resp["ok"] != serde_json::json!(true) {
                anyhow::bail!(
                    "graphsignal setup failed: {}",
                    resp["error"]["message"].as_str().unwrap_or("unknown")
                );
            }
            println!("{}", serde_json::to_string_pretty(&resp)?);
            Ok(())
        }

        Cmd::Profile { repo, mode, cmd, listen_port, cuda_graph_trace, top, cwd, no_setup, source, json } => {
            let envs: &[&str] = match mode.to_ascii_lowercase().as_str() {
                "llamacpp" | "llama.cpp" | "llama" => &["LLAMACPP_REPO"],
                "ninfer" => &["NINFER_REPO"],
                _ => &["NINFER_REPO", "LLAMACPP_REPO"],
            };
            let repo = resolve_repo(repo, envs, "NINFER_REPO or LLAMACPP_REPO")?;
            let backend = backend::resolve_backend(&repo, Some(&mode))?;
            let targets = backend::discover_targets(&repo, backend)?;
            let runner = RunnerBridge::new(PathBuf::from("runner"));
            let managed = std::env::current_dir()?.join(".kernelopt/graphsignal");
            let resp = runner.call(&serde_json::json!({
                "command": "graphsignal_profile",
                "cmd": cmd,
                "listen_port": listen_port,
                "cuda_graph_trace": cuda_graph_trace,
                "cwd": cwd,
                "auto_setup": !no_setup,
                "source": source,
                "managed_dir": managed,
                "timeout_s": 1800,
            }))?;
            if resp["ok"] != serde_json::json!(true) {
                anyhow::bail!(
                    "graphsignal_profile failed: {}",
                    resp["error"]["message"].as_str().unwrap_or("unknown")
                );
            }
            let payload = &resp["signals"];
            let kernels = signals::kernel_times(payload, top);
            let (ranking, unattributed) = attribution::rank_targets(&kernels, &targets);
            let summary = signals::summarize(payload, top);

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "backend": backend.as_str(),
                        "targets": targets.len(),
                        "kernels": kernels.len(),
                        "ranking": ranking,
                        "unattributed": unattributed,
                        "summary": summary,
                    }))?
                );
            } else if kernels.is_empty() {
                println!(
                    "no CUDA kernels captured — is the workload long enough? try --cuda-graph-trace node"
                );
                println!("summary: {summary}");
            } else {
                println!(
                    "engine share — graphsignal trace={} ({} kernels, {} targets)",
                    cuda_graph_trace,
                    kernels.len(),
                    targets.len()
                );
                println!("{:<24} {:>8} {:>14}  kernels", "op", "share", "time");
                for r in &ranking {
                    println!(
                        "{:<24} {:>7.1}% {:>11.2} ms  {}",
                        r.op,
                        r.share_pct,
                        r.ns / 1e6,
                        r.kernels.len()
                    );
                }
                if !unattributed.is_empty() {
                    let ns: f64 = unattributed.iter().map(|(_, n)| n).sum();
                    println!(
                        "unattributed: {} kernel(s), {:.2} ms",
                        unattributed.len(),
                        ns / 1e6
                    );
                }
                if let Some(errs) = summary["errors"].as_array() {
                    if !errs.is_empty() {
                        println!("errors: {} (see --json)", errs.len());
                    }
                }
            }
            Ok(())
        }

        Cmd::Discover { op, repo, mode, list } => {
            let envs: &[&str] = match mode.to_ascii_lowercase().as_str() {
                "llamacpp" | "llama.cpp" | "llama" => &["LLAMACPP_REPO"],
                "ninfer" => &["NINFER_REPO"],
                _ => &["NINFER_REPO", "LLAMACPP_REPO"],
            };
            let repo = resolve_repo(repo, envs, "NINFER_REPO or LLAMACPP_REPO")?;
            let backend = backend::resolve_backend(&repo, Some(&mode))?;
            if list || op.is_none() {
                let targets = backend::discover_targets(&repo, backend)?;
                println!("{}", serde_json::to_string_pretty(&targets)?);
                if backend == Backend::Llamacpp {
                    let unmapped = kernelopt::llamacpp::unmapped_kernel_files(&repo);
                    if !unmapped.is_empty() {
                        eprintln!("[discover] {} ggml-cuda kernel files are not in the curated map:", unmapped.len());
                        for f in unmapped {
                            eprintln!("  {f}");
                        }
                    }
                }
            } else {
                let target = backend::discover_target(&repo, backend, op.as_deref().unwrap())?;
                println!("{}", serde_json::to_string_pretty(&target)?);
            }
            Ok(())
        }

        Cmd::Resume { run_id } => {
            let events = Journal::replay(&mock_runs_dir()?, &run_id)?;
            let last = events
                .iter()
                .filter_map(|e| match e {
                    Event::RunFinished { outcome, speedup, root_cause } => Some(format!(
                        "already finished: outcome={outcome} speedup={speedup:?} cause={root_cause:?}"
                    )),
                    Event::StageCompleted { stage, .. } => Some(format!("completed stage: {stage}")),
                    _ => None,
                })
                .last()
                .unwrap_or_else(|| "no progress recorded".into());
            println!("resume {run_id}: {last}");
            Ok(())
        }

        Cmd::Status { run_id, campaign } => {
            if campaign {
                let state = campaign::load(&run_id)?;
                println!("{}", serde_json::to_string_pretty(&state.summary())?);
                println!("\n{:<24} {:>10} {:>6} {:>8}  {}", "op", "status", "iters", "speedup", "stop");
                for t in &state.targets {
                    println!(
                        "{:<24} {:>10} {:>6} {:>8}  {}",
                        t.op,
                        t.status,
                        t.iterations,
                        t.best_speedup.map(|s| format!("{s:.3}x")).unwrap_or_else(|| "-".into()),
                        t.stop_reason.clone().unwrap_or_default()
                    );
                }
                return Ok(());
            }
            let events = Journal::replay(&mock_runs_dir()?, &run_id)?;
            let mut candidates = 0usize;
            let mut tokens = 0u64;
            for e in &events {
                match e {
                    Event::CandidateEvaluated { passed, .. } if *passed => candidates += 1,
                    Event::LlmCall { prompt_tokens, completion_tokens, .. } => {
                        tokens += prompt_tokens + completion_tokens;
                    }
                    _ => {}
                }
            }
            println!("run {run_id}: {} events, {candidates} passing candidates, {tokens} tokens", events.len());
            Ok(())
        }

        Cmd::Watch { run_id, latest, once, follow } => {
            let run_id = match run_id {
                Some(id) if !latest => id,
                _ => latest_run()?,
            };
            watch_run(&run_id, once, follow)
        }

        Cmd::History { run_id } => {
            let events = Journal::replay(&mock_runs_dir()?, &run_id)?;
            println!("{:<14} {:>6} {:>10}  {:<12} plan", "candidate", "status", "latency", "commit");
            let mut n = 0;
            for e in &events {
                if let Event::CandidateEvaluated { iteration, chain, passed, latency_ms, commit, plan, .. } = e {
                    n += 1;
                    println!(
                        "{:<14} {:>6} {:>10}  {:<12} {}",
                        format!("i{iteration}/c{chain}"),
                        if *passed { "PASS" } else { "FAIL" },
                        latency_ms.map(|m| format!("{m:.4}")).unwrap_or_else(|| "-".into()),
                        commit.as_deref().map(|c| &c[..c.len().min(10)]).unwrap_or("-"),
                        truncate_line(plan, 60)
                    );
                }
            }
            if n == 0 {
                println!("(no candidates recorded)");
            }
            Ok(())
        }

        Cmd::Revert { run_id, to } => {
            let events = Journal::replay(&mock_runs_dir()?, &run_id)?;
            let backend = events
                .iter()
                .find_map(|e| match e {
                    Event::RunStarted { model_file, .. } => {
                        model_file.split(':').next().map(|s| s.to_string())
                    }
                    _ => None,
                })
                .unwrap_or_else(|| "ninfer".to_string());
            let worktree = std::env::current_dir()?
                .join(".kernelopt")
                .join(&backend)
                .join("worktree");
            if !worktree.exists() {
                anyhow::bail!("no worktree at {} (run the target first)", worktree.display());
            }
            let runner = RunnerBridge::new(PathBuf::from("runner"));
            let resp = runner.call(&serde_json::json!({
                "command": "cuda_worktree",
                "worktree_dir": worktree,
                "action": "revert",
                "ref": to,
            }))?;
            if resp["ok"] != serde_json::json!(true) {
                anyhow::bail!(
                    "revert failed: {}",
                    resp["error"]["message"].as_str().unwrap_or("unknown")
                );
            }
            println!("reverted {} to {}", worktree.display(), to);
            Ok(())
        }

        Cmd::Analyze { run_id, json } => analyze_run(&run_id, json),

        Cmd::Eval { campaign_id } => eval_campaign(&campaign_id),

        Cmd::Report { run_id } => {
            let events = Journal::replay(&mock_runs_dir()?, &run_id)?;
            print_report(&run_id, &events);
            Ok(())
        }

        Cmd::Providers { provider, model, base_url, api_key, no_ping } => {
            providers(provider, model, base_url, api_key, no_ping)
        }

        Cmd::Models { mode, repo, json } => list_models(mode, repo, json),
    }
}

// ---------------- helpers ----------------

/// Install a Ctrl-C handler that pauses gracefully at the next safe point.
fn install_interrupt() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let f = flag.clone();
    let _ = ctrlc::set_handler(move || {
        if f.swap(true, Ordering::SeqCst) {
            eprintln!("\n[interrupt] already pausing…");
        } else {
            eprintln!("\n[interrupt] finishing the current step, then pausing…");
        }
    });
    flag
}

fn resolve_repo(repo: Option<String>, envs: &[&str], hint: &str) -> Result<PathBuf> {
    let path = repo
        .filter(|s| !s.is_empty())
        .or_else(|| {
            envs.iter()
                .find_map(|e| std::env::var(e).ok().filter(|s| !s.is_empty()))
        })
        .with_context(|| format!("no repo given: pass --repo <dir> or set {hint}"))?;
    let path = kernelopt::dotenv::expand_tilde(&path);
    std::fs::canonicalize(&path).with_context(|| format!("repo not found: {path}"))
}

fn mock_runs_dir() -> Result<PathBuf> {
    Ok(std::env::current_dir()?.join(".kernelopt/runs"))
}

fn load_cuda_config(
    llm: &LlmArgs,
    loop_: &LoopArgs,
    profiler: ProfilerMode,
    ncu_set: String,
) -> Result<Config> {
    let hyper = hyper_from(llm, loop_, 5);
    let r = llm.resolve();
    Config::load(
        r.provider,
        r.model,
        r.base_url,
        r.api_key,
        hyper,
        profiler,
        ncu_set,
        "get_inputs".into(),
        r.reasoning_effort,
    )
}

fn build_llm(cfg: &Config) -> Box<dyn LlmClient> {
    if cfg.provider == "mock" {
        Box::new(llm::MockClient::new(vec![]))
    } else {
        let inner = llm::OpenAiCompatClient::new(
            cfg.base_url.clone().unwrap_or_default(),
            cfg.model.clone(),
            cfg.api_key.clone(),
            format!("kernelopt/{}", env!("CARGO_PKG_VERSION")),
            preset_session_header(&cfg.provider),
            Duration::from_secs(cfg.hyper.llm_timeout_s),
        )
        .with_reasoning_effort(cfg.reasoning_effort.clone());
        Box::new(llm::RetryClient::new(inner, 3))
    }
}

struct SingleRunOpts {
    repo: PathBuf,
    session_id: String,
    backend: Backend,
    ncu_set: String,
    e2e: Option<E2eConfig>,
    kernel_file: Option<String>,
    bench_args: Vec<String>,
    build_dir: Option<PathBuf>,
    patience: u32,
    target_speedup: f64,
    min_improvement: f64,
    max_iterations: u32,
    interrupted: Arc<AtomicBool>,
    verbose: bool,
    watch: bool,
    bench_shape: Option<String>,
    final_rounds: u32,
    edit_mode: EditMode,
}

#[allow(clippy::too_many_arguments)]
fn run_single(
    cfg: &Config,
    llm: &dyn LlmClient,
    runner: &RunnerBridge,
    mut target: Target,
    opts: SingleRunOpts,
) -> Result<PipelineResult> {
    if let Some(kf) = &opts.kernel_file {
        target.target_file = kf.clone();
    }
    if !opts.bench_args.is_empty() {
        target.bench_args = opts.bench_args.clone();
    }
    let project_root = std::env::current_dir().context("cwd")?;
    let base = project_root.join(".kernelopt").join(opts.backend.as_str());
    let worktree = base.join("worktree");
    let build_dir = opts
        .build_dir
        .map(|p| if p.is_absolute() { p } else { project_root.join(p) })
        .unwrap_or_else(|| base.join("build"));
    let run_dir = project_root.join(&cfg.runs_dir).join(&opts.session_id);
    for sub in ["candidates", "bench", "ncu"] {
        std::fs::create_dir_all(run_dir.join(sub))?;
    }
    let run_dir = run_dir.canonicalize().unwrap_or(run_dir);
    let mut journal = Journal::create(&cfg.runs_dir, &opts.session_id)?;
    println!("run id: {}", opts.session_id);
    let checkpoint_path = run_dir.join("checkpoint.json");
    let mut mem = memory::ExperienceMemory::new(cfg.hyper.q_memory, cfg.hyper.s_plus, cfg.hyper.s_minus);
    let mut tracker = memory::StrategyTracker::default();
    let mut pipe = CudaPipeline {
        cfg,
        llm,
        runner,
        journal: &mut journal,
        session_id: opts.session_id.clone(),
        repo: opts.repo,
        target,
        worktree,
        build_dir,
        gpu_lock_path: kernelopt::gpu_lock::default_path(&project_root),
        run_dir,
        ncu_set: opts.ncu_set,
        edit_mode: opts.edit_mode,
        e2e: opts.e2e,
        baseline_e2e: None,
        base_sha: None,
        last_bench_noise_pct: None,
        last_bench_label: None,
        last_bench_gbs: None,
        last_bench_roofline_gbs: None,
        memory: &mut mem,
        tracker: &mut tracker,
        patience: opts.patience,
        target_speedup: opts.target_speedup,
        min_improvement: opts.min_improvement,
        max_iterations: opts.max_iterations,
        deadline: None,
        llm_call_budget: None,
        llm_calls: 0,
        tokens: 0,
        checkpoint_path: Some(checkpoint_path),
        interrupted: Some(opts.interrupted),
        verbose: opts.verbose,
        watch: opts.watch,
        bench_shape: opts.bench_shape,
        final_rounds: opts.final_rounds,
    };
    pipe.run()
}

fn preset_session_header(provider: &str) -> Option<String> {
    if provider == "opencode-go" {
        Some("x-opencode-session".into())
    } else {
        None
    }
}

fn providers(
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    no_ping: bool,
) -> Result<()> {
    let r = resolve_llm(provider, model, base_url, api_key, None);
    let (provider, model, base_url, api_key) = (r.provider, r.model, r.base_url, r.api_key);
    for name in ["opencode-go", "openai", "openrouter", "ollama", "vllm", "lmstudio"] {
        if let Some(p) = config::provider_preset(name) {
            println!(
                "{:<14} {:<35} key-env: {}",
                p.name,
                p.base_url,
                p.api_key_env.unwrap_or_else(|| "(none)".into())
            );
        }
    }
    println!("{:<14} scripted completions, zero tokens (for tests)", "mock");
    println!();

    let preset = config::provider_preset(&provider);
    let base = base_url
        .or_else(|| preset.as_ref().map(|p| p.base_url.clone()))
        .with_context(|| format!("no base URL for provider {provider:?}"))?;
    let key = api_key.or_else(|| {
        preset
            .as_ref()
            .and_then(|p| p.api_key_env.clone())
            .and_then(|env| std::env::var(env).ok())
    });
    if preset.as_ref().map(|p| p.needs_key).unwrap_or(false) && key.is_none() {
        anyhow::bail!(
            "provider {provider:?} needs an API key: set {} or pass --api-key",
            preset.as_ref().and_then(|p| p.api_key_env.clone()).unwrap_or_default()
        );
    }
    let client = llm::OpenAiCompatClient::new(
        base.clone(),
        model.clone(),
        key,
        format!("kernelopt/{}", env!("CARGO_PKG_VERSION")),
        preset_session_header(&provider),
        Duration::from_secs(120),
    );
    println!("testing {provider} @ {base} (model: {model})");
    match client.list_models() {
        Ok(models) => {
            println!("  auth OK — {} models visible", models.len());
            if !models.iter().any(|m| m == &model) {
                println!("  WARNING: {model:?} not in model list; sample: {:?}", &models[..models.len().min(6)]);
            } else {
                println!("  model {model:?} available");
            }
        }
        Err(e) => {
            println!("  models check failed: {e:#}");
            return Ok(());
        }
    }
    if !no_ping {
        print!("  completion probe… ");
        std::io::Write::flush(&mut std::io::stdout())?;
        match client.ping() {
            Ok(reply) => println!("OK — reply: {}", reply.trim().chars().take(40).collect::<String>()),
            Err(e) => println!("FAILED: {e:#}"),
        }
        // The pipeline depends on forced tool calls — probe one explicitly, the
        // same way the Planner/Executor call the model.
        print!("  tool-call probe… ");
        std::io::Write::flush(&mut std::io::stdout())?;
        let tools = vec![llm::ToolDef {
            name: "probe".into(),
            description: "Echo a value back to the caller".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }),
        }];
        match client.complete(
            "You are a connectivity probe.",
            "Call the probe tool with value \"ok\".",
            &tools,
            "kernelopt-providers-test",
            Some("probe"),
        ) {
            Ok(c) => match c.tool_calls.iter().find(|t| t.name == "probe") {
                Some(tc) => println!(
                    "OK — probe(value={:?})",
                    tc.arguments.get("value").and_then(|v| v.as_str()).unwrap_or("?")
                ),
                None => println!(
                    "NO TOOL CALL — content only; forced tool calls may be unsupported here \
                     (the client falls back to \"auto\"/no tool_choice)"
                ),
            },
            Err(e) => println!("FAILED: {e:#}"),
        }
    }
    Ok(())
}

/// List local model artifacts usable for the engine-E2E (Gate 3) check.
fn list_models(mode: Option<String>, repo: Option<String>, json: bool) -> Result<()> {
    let repo = repo
        .map(|p| kernelopt::dotenv::expand_tilde(&p))
        .or_else(|| env_opt("NINFER_REPO").map(|p| kernelopt::dotenv::expand_tilde(&p)))
        .or_else(|| env_opt("LLAMACPP_REPO").map(|p| kernelopt::dotenv::expand_tilde(&p)))
        .unwrap_or_else(|| ".".to_string());
    let repo = std::path::PathBuf::from(repo);
    let backend = mode.as_deref().filter(|m| *m != "auto");
    let models = kernelopt::models::discover(&repo, backend);

    if json {
        let arr: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name, "engine": m.engine, "size_bytes": m.size_bytes,
                    "path": m.path.to_string_lossy(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }

    println!("search dirs (KERNELOPT_MODELS_DIR, <repo>/models, ./models, ~/models):");
    for d in kernelopt::models::search_dirs(&repo) {
        println!("  {}", d.display());
    }
    if models.is_empty() {
        println!(
            "\nno local models found — set KERNELOPT_MODELS_DIR, or place artifacts under <repo>/models"
        );
        return Ok(());
    }
    println!("\n{:<46} {:<9} {:>10}  path", "name", "engine", "size");
    for m in &models {
        println!(
            "{:<46} {:<9} {:>10}  {}",
            m.name,
            m.engine,
            kernelopt::models::human_size(m.size_bytes),
            m.path.display()
        );
    }
    println!("\nuse one with `--e2e-weights <name|auto>` — a bare name resolves to the path above.");
    Ok(())
}

/// Diagnose a run from its journal: failure taxonomy, retries, tokens, plans.
fn analyze_run(run_id: &str, json: bool) -> Result<()> {
    let events = Journal::replay(&mock_runs_dir()?, run_id)?;
    let mut categories: BTreeMap<String, u32> = BTreeMap::new();
    let mut attempts: Vec<serde_json::Value> = Vec::new();
    let mut llm: BTreeMap<String, (u32, u64)> = BTreeMap::new();
    let mut plans: BTreeSet<String> = BTreeSet::new();
    let mut outcome: Option<(String, Option<f64>, Option<String>)> = None;

    for e in &events {
        match e {
            Event::AttemptFailed { iteration, chain, attempt, category, error } => {
                *categories.entry(category.clone()).or_default() += 1;
                attempts.push(serde_json::json!({
                    "iteration": iteration, "chain": chain, "attempt": attempt,
                    "passed": false, "category": category,
                    "error": truncate_line(error, 140),
                }));
            }
            Event::CandidateEvaluated { iteration, chain, passed, latency_ms, plan, error, commit, .. } => {
                if *passed {
                    *categories.entry("pass".to_string()).or_default() += 1;
                }
                plans.insert(plan.chars().take(60).collect());
                attempts.push(serde_json::json!({
                    "iteration": iteration, "chain": chain, "passed": passed,
                    "latency_ms": latency_ms, "category": if *passed { "pass" } else { "failed" },
                    "commit": commit,
                    "error": error.as_deref().map(|s| truncate_line(s, 140)),
                }));
            }
            Event::LlmCall { agent, prompt_tokens, completion_tokens } => {
                let e = llm.entry(agent.clone()).or_default();
                e.0 += 1;
                e.1 += prompt_tokens + completion_tokens;
            }
            Event::RunFinished { outcome: o, speedup, root_cause } => {
                outcome = Some((o.clone(), *speedup, root_cause.clone()));
            }
            _ => {}
        }
    }

    let (total_calls, total_tokens) = llm
        .values()
        .fold((0u32, 0u64), |(c, t), (c2, t2)| (c + c2, t + t2));
    let passes = categories.get("pass").copied().unwrap_or(0);
    let fails: u32 = categories.iter().filter(|(k, _)| *k != "pass").map(|(_, v)| v).sum();

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "run_id": run_id,
                "outcome": outcome.as_ref().map(|(o, _, _)| o),
                "speedup": outcome.as_ref().and_then(|(_, s, _)| *s),
                "root_cause": outcome.as_ref().and_then(|(_, _, c)| c.clone()),
                "attempts": attempts,
                "categories": categories,
                "passes": passes,
                "fails": fails,
                "unique_plans": plans.len(),
                "llm_calls": total_calls,
                "tokens": total_tokens,
                "llm_by_agent": llm.iter().map(|(k, (c, t))| (k.clone(), serde_json::json!({"calls": c, "tokens": t}))).collect::<serde_json::Map<_, _>>(),
            }))?
        );
        return Ok(());
    }

    match &outcome {
        Some((o, sp, cause)) => {
            println!(
                "run {run_id}: outcome={o}{}{}",
                sp.map(|s| format!(" speedup={s:.3}x")).unwrap_or_default(),
                cause.as_ref().map(|c| format!(" — {c}")).unwrap_or_default()
            );
        }
        None => println!("run {run_id}: still running (no RunFinished event yet)"),
    }
    println!(
        "attempts: {}  passes: {passes}  fails: {fails}  unique plans: {}",
        attempts.len(),
        plans.len()
    );
    println!(
        "failure categories: {}",
        categories
            .iter()
            .filter(|(k, _)| *k != "pass")
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!(
        "llm: {} calls, {} tokens — {}",
        total_calls,
        total_tokens,
        llm.iter()
            .map(|(a, (c, t))| format!("{a} {c}/{t}t"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for a in &attempts {
        println!(
            "  i{}/c{} {} {}{}",
            a["iteration"], a["chain"], a["category"],
            a["latency_ms"].as_f64().map(|m| format!("{m:.4} ms ")).unwrap_or_default(),
            a["error"].as_str().unwrap_or("")
        );
    }
    print_winner(&events);
    Ok(())
}

/// Aggregate a campaign into a self-benchmark: win rate, speedup distribution,
/// failure taxonomy, and token cost (across every target's journal).
fn eval_campaign(campaign_id: &str) -> Result<()> {
    let state = campaign::load(campaign_id)?;
    let runs_dir = mock_runs_dir()?;
    let mut cats: BTreeMap<String, u32> = BTreeMap::new();
    let mut causes: BTreeMap<String, u32> = BTreeMap::new();
    let mut speedups: Vec<f64> = Vec::new();
    let mut tokens = 0u64;
    let mut calls = 0u64;

    for t in &state.targets {
        if let Some(rid) = &t.run_id {
            if let Ok(events) = Journal::replay(&runs_dir, rid) {
                for e in &events {
                    match e {
                        Event::AttemptFailed { category, .. } => {
                            *cats.entry(category.clone()).or_default() += 1
                        }
                        Event::LlmCall { prompt_tokens, completion_tokens, .. } => {
                            tokens += prompt_tokens + completion_tokens;
                            calls += 1;
                        }
                        Event::RunFinished { root_cause: Some(c), .. } => {
                            *causes.entry(c.clone()).or_default() += 1
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(s) = t.best_speedup {
            speedups.push(s);
        }
    }

    let count = |status: &str| state.targets.iter().filter(|t| t.status == status).count();
    let optimized = count("optimized");
    let matched = count("matched");
    let fallback = count("fallback");
    let failed = count("failed");
    speedups.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = speedups.get(speedups.len() / 2).copied().unwrap_or(1.0);
    let max = speedups.last().copied().unwrap_or(1.0);
    let wins = speedups.iter().filter(|s| **s > 1.0).count();

    println!("campaign {campaign_id}: {} targets", state.targets.len());
    println!(
        "outcomes: optimized {optimized}  matched {matched}  fallback {fallback}  failed {failed}"
    );
    println!(
        "speedups: median {med:.3}x  max {max:.3}x  wins(>1.0x) {wins}/{}",
        state.targets.len()
    );
    println!(
        "failures: {}",
        cats.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!(
        "cost: {calls} LLM calls, {tokens} tokens  ({} tokens / optimized target)",
        tokens / optimized.max(1) as u64
    );
    if !causes.is_empty() {
        println!(
            "fallback causes: {}",
            causes
                .iter()
                .map(|(k, v)| format!("{v}× {k}"))
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    Ok(())
}

/// The most recently modified run id under `.kernelopt/runs/`.
fn latest_run() -> Result<String> {
    let dir = mock_runs_dir()?;
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
            best = Some((mtime, name));
        }
    }
    best.map(|(_, n)| n).context("no runs found")
}

/// Tail a run's journal and print formatted progress (like `tail -f`).
/// Stops once the run reports `RunFinished` (unless `follow`), then prints the
/// winner and the artifact paths so the diff is easy to find.
fn watch_run(run_id: &str, once: bool, follow: bool) -> Result<()> {
    let run_dir = mock_runs_dir()?.join(run_id);
    let path = run_dir.join("journal.jsonl");
    if !path.exists() {
        anyhow::bail!("no journal at {} (is the run id right?)", path.display());
    }
    println!("watching {run_id} — Ctrl-C to stop");
    let mut offset = 0u64;
    let mut events: Vec<Event> = Vec::new();
    let mut finished = false;
    loop {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let bytes = text.len() as u64;
            if bytes > offset {
                let tail = &text[offset as usize..];
                if let Some(last_nl) = tail.rfind('\n') {
                    for line in tail[..last_nl].lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(e) = serde_json::from_str::<Event>(line) {
                            println!("{}", format_event(&e));
                            if matches!(e, Event::RunFinished { .. }) {
                                finished = true;
                            }
                            events.push(e);
                        }
                    }
                    offset += (last_nl + 1) as u64;
                }
            }
        }
        if once || (finished && !follow) {
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if finished {
        print_winner(&events);
        println!("\nartifacts:");
        println!("  report:  {}", run_dir.join("report.md").display());
        println!("  diff:    {}", run_dir.join("report.diff").display());
        println!("  journal: {}", path.display());
    }
    Ok(())
}


/// Print the winner's "what changed / why faster" section.
fn print_winner(events: &[Event]) {
    let Some(w) = winner_summary(events) else { return };
    println!("\n## winner — what changed / why it's faster");
    println!("candidate: {}", w["candidate"].as_str().unwrap_or(""));
    if let Some(c) = w["commit"].as_str() {
        println!("commit:    {c}");
    }
    if let Some(x) = w["what_changed"].as_str().filter(|x| !x.is_empty()) {
        println!("what:      {x}");
    }
    if let Some(x) = w["why"].as_str().filter(|x| !x.is_empty()) {
        println!("why:       {x}");
    }
    if let Some(x) = w["how"].as_str().filter(|x| !x.is_empty()) {
        println!("how:       {x}");
    }
    if let Some(m) = w.get("measurement") {
        let shape = m["shape"].as_str().unwrap_or("(all shapes)");
        let runs = m["runs"].as_u64().unwrap_or(1);
        let base = m["baseline_ms"].as_f64();
        let fin = m["final_ms"].as_f64().or_else(|| w["latency_ms"].as_f64());
        let sp = m["speedup"].as_f64().unwrap_or(1.0);
        let noise = m["noise_pct"].as_f64();
        if let (Some(b), Some(f)) = (base, fin) {
            println!(
                "measured:  op-level median latency on {shape}, {runs} runs: {b:.4} ms -> {f:.4} ms = {sp:.2}x{}",
                noise.map(|n| format!(" (noise {:.2}%)", n * 100.0)).unwrap_or_default()
            );
            println!(
                "           {sp:.2}x = baseline/final: latency fell to {:.0}% of baseline",
                100.0 / sp
            );
            if let (Some(g), Some(r)) = (m["cand_gbs"].as_f64(), m["roofline_gbs"].as_f64()) {
                if r > 0.0 {
                    println!(
                        "           {g:.0} GB/s = {:.0}% of the {r:.0} GB/s memory roofline",
                        g / r * 100.0
                    );
                }
            }
        }
    } else if let (Some(ms), Some(base)) = (w["latency_ms"].as_f64(), w["baseline_ms"].as_f64()) {
        println!(
            "measured:  op-level latency {ms:.4} ms vs baseline {base:.4} ms → {:.2}x",
            w["speedup"].as_f64().unwrap_or(1.0)
        );
    }
    println!("plan:      {}", w["plan"].as_str().unwrap_or(""));
}

fn print_report(run_id: &str, events: &[Event]) {
    println!("# KernelOpt report — {run_id}\n");
    for e in events {
        match e {
            Event::RunStarted { model_file, provider, model, .. } => {
                println!("- target: `{model_file}` via {provider}/{model}");
            }
            Event::StageCompleted { stage, .. } => println!("- stage completed: {stage}"),
            Event::CandidateEvaluated { iteration, chain, passed, latency_ms, plan, .. } => {
                println!(
                    "- candidate i{iteration}/c{chain}: {}{} — plan: {}",
                    if *passed { "PASS" } else { "FAIL" },
                    latency_ms.map(|m| format!(" @ {m:.4} ms")).unwrap_or_default(),
                    truncate_line(plan, 80)
                );
            }
            Event::GatesVerdict { passed, detail, .. } => {
                println!("- gates: {} — `{}`", if *passed { "PASS" } else { "REJECT" }, detail);
            }
            Event::RunFinished { outcome, speedup, root_cause } => {
                println!("\n## outcome: {outcome}");
                if let Some(s) = speedup {
                    println!("speedup: {s:.3}x");
                }
                if let Some(c) = root_cause {
                    println!("root cause: {c}");
                }
            }
            _ => {}
        }
    }
    print_winner(events);
}

fn truncate_line(s: &str, n: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= n {
        one_line
    } else {
        format!("{}…", one_line.chars().take(n).collect::<String>())
    }
}
