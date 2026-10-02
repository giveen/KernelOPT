//! kernelopt — dispatch-aware agentic GPU kernel optimization.

use kernelopt::attribution;
use kernelopt::backend::{self, Backend, Target};
use kernelopt::campaign::{self, CampaignOptions};
use kernelopt::config::{self, Config, Hyper, ProfilerMode};
use kernelopt::cuda_pipeline::{winner_summary, CudaPipeline, E2eConfig, EditMode, PipelineResult, DEFAULT_VERIFY_TIMEOUT_S};
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
        /// Print the raw JSON result instead of a human summary.
        #[arg(long)]
        json: bool,
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
        #[arg(long = "bench-arg", allow_hyphen_values = true)]
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
        /// Print the raw JSON result instead of a human summary.
        #[arg(long)]
        json: bool,
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
        #[arg(long = "bench-arg", allow_hyphen_values = true)]
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
        /// Print the raw JSON result instead of a human summary.
        #[arg(long)]
        json: bool,
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
    /// Rank kernels by real engine share (nsys/ncu, or Graphsignal).
    Profile {
        /// Target checkout (env: NINFER_REPO/LLAMACPP_REPO).
        #[arg(long)]
        repo: Option<String>,
        /// Backend: auto | ninfer | llamacpp.
        #[arg(long, default_value = "auto")]
        mode: String,
        /// Workload to profile (argv after --cmd; place last).
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
        /// Engine-share backend: auto | nsys | ncu | graphsignal.
        #[arg(long, default_value = "auto")]
        engine: String,
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
    /// Look up a CUDA/CUB (or HIP) symbol in the docs sources.
    Docs {
        /// Symbol to look up, e.g. `cub::WarpMergeSort` (omit with --login).
        symbol: Option<String>,
        /// Run the OAuth login for the docs MCP server and cache the token.
        #[arg(long)]
        login: bool,
    },
    /// Structural code map: who calls a symbol (via codebase-memory-mcp).
    Map {
        /// Repo (or worktree) to index.
        #[arg(long)]
        repo: Option<String>,
        /// Symbol to trace callers for (else lists top-level symbols).
        #[arg(long)]
        symbol: Option<String>,
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
    /// Interactive setup: detect what it can, ask a few questions, then run it.
    Wizard {
        /// Target checkout (else NINFER_REPO/LLAMACPP_REPO, else prompt).
        #[arg(long)]
        repo: Option<String>,
        /// Backend: auto | ninfer | llamacpp.
        #[arg(long)]
        mode: Option<String>,
        /// Optimize this op only (else choose from the discovered list).
        #[arg(long)]
        op: Option<String>,
        /// Optimize every target (campaign) without asking.
        #[arg(long)]
        all: bool,
        /// Engine-E2E model: a path, a name, or `auto`.
        #[arg(long)]
        e2e_weights: Option<String>,
        /// Loop preset: quick | standard | thorough.
        #[arg(long)]
        preset: Option<String>,
        /// Accept detected defaults with no prompts (scripts/CI).
        #[arg(long)]
        yes: bool,
        /// Print the equivalent command and exit (run nothing).
        #[arg(long)]
        dry_run: bool,
        /// Persist the choices to `.env` so later plain runs pick them up.
        #[arg(long)]
        save: bool,
        #[command(flatten)]
        llm: LlmArgs,
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
    /// Cross-shape regression tolerance (default 1.01).
    #[arg(long)]
    regress_margin: Option<f64>,
}

fn hyper_from(llm: &LlmArgs, lp: &LoopArgs, default_t: u32) -> Hyper {
    let _ = llm;
    // CLI -> env (`wizard --save` persists the preset) -> paper default.
    let env_u32 = |name: &str| env_opt(name).and_then(|v| v.parse::<u32>().ok());
    Hyper {
        t_iterations: lp
            .iterations
            .or_else(|| env_u32("KERNELOPT_ITERATIONS"))
            .unwrap_or(default_t),
        n_plans: lp.plans.or_else(|| env_u32("KERNELOPT_PLANS")).unwrap_or(4),
        k_retries: lp.retries.or_else(|| env_u32("KERNELOPT_RETRIES")).unwrap_or(4),
        b_beam: lp.beam.or_else(|| env_u32("KERNELOPT_BEAM")).unwrap_or(4),
        regression_margin: lp
            .regress_margin
            .or_else(|| env_opt("KERNELOPT_REGRESS_MARGIN").and_then(|v| v.parse::<f64>().ok()))
            .unwrap_or(1.01),
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
            json,
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
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_run_result(&result, true);
            }
            Ok(())
        }

        Cmd::RunNinfer { op, repo, llm: llm_args, kernel_file, build_dir, bench_args, e2e_weights, e2e_engine, e2e_cmd, e2e_prompt, e2e_max_new, run_id, ncu_set, quiet, watch, bench_shape, final_rounds, edit_mode, json, loop_ } => {
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
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_run_result(&serde_json::to_value(&result)?, !watch);
            }
            if result.paused {
                eprintln!("paused; re-run with --run-id {session_id} to continue");
            }
            Ok(())
        }

        Cmd::RunLlamacpp { op, repo, llm: llm_args, kernel_file, build_dir, bench_args, e2e_weights, e2e_engine, e2e_cmd, e2e_prompt, e2e_max_new, run_id, ncu_set, quiet, watch, bench_shape, final_rounds, edit_mode, json, loop_ } => {
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
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_run_result(&serde_json::to_value(&result)?, !watch);
            }
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

        Cmd::Profile { repo, mode, cmd, listen_port, cuda_graph_trace, top, cwd, no_setup, source, json, engine } => {
            let envs: &[&str] = match mode.to_ascii_lowercase().as_str() {
                "llamacpp" | "llama.cpp" | "llama" => &["LLAMACPP_REPO"],
                "ninfer" => &["NINFER_REPO"],
                _ => &["NINFER_REPO", "LLAMACPP_REPO"],
            };
            let repo = resolve_repo(repo, envs, "NINFER_REPO or LLAMACPP_REPO")?;
            let backend = backend::resolve_backend(&repo, Some(&mode))?;
            let targets = backend::discover_targets(&repo, backend)?;
            let engine = if engine.eq_ignore_ascii_case("auto") {
                kernelopt::engine_share::auto_engine().to_string()
            } else {
                engine.to_ascii_lowercase()
            };
            eprintln!("[profile] engine share via {engine} (workload: {})", cmd.join(" "));
            let (kernels, summary) = match engine.as_str() {
                "nsys" | "ncu" => {
                    let k = if engine == "nsys" {
                        kernelopt::engine_share::nsys_kernel_times(&cmd, 1800)?
                    } else {
                        kernelopt::engine_share::ncu_kernel_times(&cmd, 1800)?
                    };
                    (k, serde_json::json!({}))
                }
                _ => {
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
                    let payload = resp["signals"].clone();
                    let k = signals::kernel_times(&payload, top);
                    let s = signals::summarize(&payload, top);
                    (k, s)
                }
            };
            let (ranking, unattributed) = attribution::rank_targets(&kernels, &targets);

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "backend": backend.as_str(),
                        "engine": engine,
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
                    "engine share — {engine} ({} kernels, {} targets)",
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
                    Event::RunFinished { outcome, speedup, root_cause, .. } => Some(format!(
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
            kernelopt::git::revert(&worktree, &to)?;
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

        Cmd::Docs { symbol, login } => {
            if login {
                let t = kernelopt::docs_oauth::login()?;
                let left = (t.expires_at - chrono::Utc::now().timestamp()).max(0);
                println!("logged in — docs token cached (valid ~{left}s)");
                return Ok(());
            }
            let Some(sym) = symbol else {
                anyhow::bail!("provide a symbol to look up, or use --login");
            };
            match kernelopt::docs::lookup(&sym) {
                Some(text) => println!("{}", kernelopt::docs::clean_markdown(&text)),
                None => eprintln!("no docs found for {sym}"),
            }
            Ok(())
        }

        Cmd::Map { repo, symbol } => {
            let repo = resolve_repo(repo, &["NINFER_REPO", "LLAMACPP_REPO"], "repo")?;
            let Some(cm) = kernelopt::codemap::open(&repo) else {
                anyhow::bail!(
                    "codebase-memory-mcp not found — install it (`kernelopt setup`) or set KERNELOPT_CODEMAP_CMD"
                );
            };
            println!("indexed {} as {}", repo.display(), cm.project);
            match symbol {
                Some(sym) => {
                    let callers = cm.callers(&sym, 20);
                    if callers.is_empty() {
                        println!("no callers found for {sym}");
                    } else {
                        println!("callers of {sym}:");
                        for c in callers {
                            println!("  {c}");
                        }
                    }
                }
                None => {
                    println!("(pass --symbol <fn> to trace its callers)");
                }
            }
            Ok(())
        }

        Cmd::Providers { provider, model, base_url, api_key, no_ping } => {
            providers(provider, model, base_url, api_key, no_ping)
        }

        Cmd::Models { mode, repo, json } => list_models(mode, repo, json),

        Cmd::Wizard { repo, mode, op, all, e2e_weights, preset, yes, dry_run, save, llm } => {
            wizard(repo, mode, op, all, e2e_weights, preset, yes, dry_run, save, llm)
        }
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
        verify_timeout_s: std::cell::Cell::new(DEFAULT_VERIFY_TIMEOUT_S),
        recent_failures: Vec::new(),
        toolchain: None,
        recent_results: Vec::new(),
        bench_options: None,
        baseline_shapes: Vec::new(),
        codemap: None,
        caller_context: None,
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

/// Outcome of probing an OpenAI-compatible endpoint the way the pipeline uses it.
struct LlmProbe {
    base: String,
    models: Option<usize>,
    model_listed: bool,
    completion_ok: Option<bool>,
    tool_ok: Option<bool>,
    error: Option<String>,
}

/// Probe an endpoint: `GET /models` (auth), a 1-token completion, and a forced
/// tool call (what the pipeline actually depends on).
fn probe_llm(
    provider: &str,
    model: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<LlmProbe> {
    let preset = config::provider_preset(provider);
    let base = base_url
        .map(|s| s.to_string())
        .or_else(|| preset.as_ref().map(|p| p.base_url.clone()))
        .with_context(|| format!("no base URL for provider {provider:?}"))?;
    let key = api_key.map(|s| s.to_string()).or_else(|| {
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
        model.to_string(),
        key,
        format!("kernelopt/{}", env!("CARGO_PKG_VERSION")),
        preset_session_header(provider),
        Duration::from_secs(120),
    );
    let mut p = LlmProbe {
        base,
        models: None,
        model_listed: false,
        completion_ok: None,
        tool_ok: None,
        error: None,
    };
    match client.list_models() {
        Ok(models) => {
            p.models = Some(models.len());
            p.model_listed = models.iter().any(|m| m == model);
        }
        Err(e) => {
            p.error = Some(format!("{e:#}"));
            return Ok(p);
        }
    }
    p.completion_ok = Some(client.ping().is_ok());
    let tools = vec![llm::ToolDef {
        name: "probe".into(),
        description: "Echo a value back to the caller".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        }),
    }];
    p.tool_ok = Some(matches!(
        client.complete(
            "You are a connectivity probe.",
            "Call the probe tool with value \"ok\".",
            &tools,
            "kernelopt-providers-test",
            Some("probe"),
        ),
        Ok(c) if c.tool_calls.iter().any(|t| t.name == "probe")
    ));
    Ok(p)
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

    let p = probe_llm(&provider, &model, base_url.as_deref(), api_key.as_deref())?;
    println!("testing {provider} @ {} (model: {model})", p.base);
    match (&p.models, &p.error) {
        (Some(n), _) => {
            println!("  auth OK — {n} models visible");
            if p.model_listed {
                println!("  model {model:?} available");
            } else {
                println!("  WARNING: {model:?} not in model list");
            }
        }
        (None, Some(e)) => {
            println!("  models check failed: {e}");
            return Ok(());
        }
        (None, None) => {}
    }
    if !no_ping {
        match p.completion_ok {
            Some(true) => println!("  completion probe… OK"),
            Some(false) => println!("  completion probe… FAILED"),
            None => {}
        }
        match p.tool_ok {
            Some(true) => println!("  tool-call probe… OK"),
            Some(false) => println!(
                "  tool-call probe… NO TOOL CALL — content only; forced tool calls may be \
                 unsupported here (the client falls back to \"auto\"/no tool_choice)"
            ),
            None => {}
        }
    }
    Ok(())
}

/// Interactive setup: detect what it can, ask the few real choices, run it.
#[allow(clippy::too_many_arguments)]
fn wizard(
    repo: Option<String>,
    mode: Option<String>,
    op: Option<String>,
    all: bool,
    e2e_weights: Option<String>,
    preset: Option<String>,
    yes: bool,
    dry_run: bool,
    save: bool,
    llm: LlmArgs,
) -> Result<()> {
    // 1. Target checkout: flag -> env -> prompt.
    let repo_spec = repo
        .or_else(|| env_opt("NINFER_REPO"))
        .or_else(|| env_opt("LLAMACPP_REPO"));
    let repo_spec = match repo_spec {
        Some(r) => r,
        None if yes => {
            anyhow::bail!("no target checkout: pass --repo or set NINFER_REPO/LLAMACPP_REPO")
        }
        None => ask_line("Target checkout path: ")?,
    };
    let repo = PathBuf::from(kernelopt::dotenv::expand_tilde(&repo_spec));
    if !repo.is_dir() {
        anyhow::bail!(
            "repo not found: {} (set NINFER_REPO/LLAMACPP_REPO or pass --repo)",
            repo.display()
        );
    }
    let backend = backend::resolve_backend(&repo, mode.as_deref())?;
    println!("· target: {}  ({})", repo.display(), backend.as_str());

    // 2. Targets.
    let targets = backend::discover_targets(&repo, backend)?;
    if targets.is_empty() {
        anyhow::bail!("no optimizable targets in {}", repo.display());
    }
    println!("· {} target(s) discovered", targets.len());
    let (op, all) = if let Some(op) = op {
        (Some(op), false)
    } else if all || yes {
        (None, true)
    } else {
        choose_target(&targets)?
    };

    // 3. Optimizer LLM (accept the resolved .env defaults by default).
    let resolved = llm.resolve();
    let (provider, model) = if yes {
        (resolved.provider, resolved.model)
    } else {
        prompt_llm(resolved)?
    };
    println!("· optimizer: {provider}/{model}");

    // 4. Engine-E2E model (optional).
    let e2e = if let Some(e) = e2e_weights {
        Some(e)
    } else if yes {
        None
    } else {
        choose_e2e(&repo, backend)?
    };

    // 5. Loop preset.
    let preset = if let Some(p) = preset {
        p
    } else if yes {
        "standard".into()
    } else {
        choose_preset()?
    };
    let (iterations, beam, plans) = kernelopt::wizard::preset(&preset);
    println!("· preset: {preset} ({iterations} iterations, beam {beam}, {plans} plans)");

    // Optionally persist the choices so later plain runs pick them up.
    if save {
        let repo_key = match backend {
            Backend::Ninfer => Some("NINFER_REPO"),
            Backend::Llamacpp => Some("LLAMACPP_REPO"),
            // Custom repos are passed with --repo (no canonical env var).
            Backend::Custom => None,
        };
        let mut pairs: Vec<(String, String)> = vec![
            ("KERNELOPT_PROVIDER".into(), provider.clone()),
            ("KERNELOPT_MODEL".into(), model.clone()),
            ("KERNELOPT_ITERATIONS".into(), iterations.to_string()),
            ("KERNELOPT_BEAM".into(), beam.to_string()),
            ("KERNELOPT_PLANS".into(), plans.to_string()),
        ];
        if let Some(k) = repo_key {
            pairs.insert(0, (k.into(), repo.to_string_lossy().to_string()));
        }
        if let Some(e) = &e2e {
            pairs.push(("KERNELOPT_E2E_WEIGHTS".into(), e.clone()));
        }
        let path = kernelopt::dotenv::default_path();
        kernelopt::dotenv::set_vars(&path, &pairs)?;
        println!(
            "· saved to {} ({})",
            path.display(),
            pairs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(", ")
        );
    }

    let plan = kernelopt::wizard::Plan {
        backend,
        repo,
        op,
        all,
        provider,
        model,
        e2e,
        iterations,
        beam,
        plans,
        watch: true,
    };

    println!("\n  {}\n", plan.display());
    if dry_run {
        return Ok(());
    }
    if !yes && !confirm("Run this now?", true)? {
        println!("not run.");
        return Ok(());
    }
    // Re-invoke ourselves with exactly this argv (same code path as the CLI).
    let exe = std::env::current_exe().context("current_exe")?;
    let status = std::process::Command::new(exe).args(plan.argv()).status()?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

/// Default, human-readable result for the `run*` commands. The raw object is
/// printed only with `--json` (for scripts).
fn print_run_result(v: &serde_json::Value, show_paths: bool) {
    let outcome = v["outcome"].as_str().unwrap_or("?");
    let speedup = v["speedup"].as_f64();
    println!(
        "\noutcome:    {outcome}{}",
        speedup.map(|s| format!("  ({s:.3}x)")).unwrap_or_default()
    );
    if let (Some(b), Some(f)) = (v["baseline_ms"].as_f64(), v["final_ms"].as_f64()) {
        println!("latency:    {b:.4} ms → {f:.4} ms");
    }
    match speedup {
        Some(s) => println!(
            "speedup:    {s:.3}x{}",
            if outcome == "fallback" {
                "  (pinned shape only — rejected, see root cause)"
            } else {
                ""
            }
        ),
        None => println!("speedup:    not measured"),
    }
    if let Some(sr) = v["stop_reason"].as_str().filter(|s| !s.is_empty()) {
        println!("stopped:    {sr}");
    }
    println!(
        "llm:        {} calls · {} tokens",
        v["llm_calls"].as_u64().unwrap_or(0),
        v["tokens"].as_u64().unwrap_or(0)
    );
    if let Some(c) = v["root_cause"].as_str() {
        println!("root cause: {c}");
    }
    if show_paths {
        if let Some(d) = v["diff_path"].as_str() {
            let p = std::path::Path::new(d);
            if let Some(dir) = p.parent() {
                println!("report:     {}", rel_path(&dir.join("report.md")));
            }
            println!("diff:       {}", rel_path(p));
        }
    }
    println!();
}

/// Render a path relative to the cwd (`./…`) when it lives under it.
fn rel_path(p: &std::path::Path) -> String {
    match std::env::current_dir()
        .ok()
        .and_then(|c| p.strip_prefix(c).ok().map(|r| r.to_path_buf()))
    {
        Some(r) => format!("./{}", r.display()),
        None => p.display().to_string(),
    }
}

fn ask_line(prompt: &str) -> Result<String> {
    use std::io::Write;
    eprint!("{prompt}");
    let _ = std::io::stderr().flush();
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).context("reading stdin")?;
    Ok(s.trim().to_string())
}

fn confirm(prompt: &str, default: bool) -> Result<bool> {
    let d = if default { "[Y/n]" } else { "[y/N]" };
    let a = ask_line(&format!("{prompt} {d} "))?;
    Ok(match a.to_ascii_lowercase().as_str() {
        "" => default,
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default,
    })
}

fn choose_target(targets: &[Target]) -> Result<(Option<String>, bool)> {
    println!("\nWhich target?");
    println!("   0) all of them  (campaign)");
    for (i, t) in targets.iter().enumerate() {
        let variant = t.variant.as_deref().map(|v| format!("/{v}")).unwrap_or_default();
        println!("  {:>2}) {}  ({}{})", i + 1, t.op, t.family, variant);
    }
    let n = targets.len();
    let ans = ask_line(&format!("choose [0-{n}, default 0]: "))?;
    match kernelopt::wizard::parse_choice(&ans, n).unwrap_or(0) {
        0 => Ok((None, true)),
        k => Ok((Some(targets[k - 1].op.clone()), false)),
    }
}

fn choose_e2e(repo: &std::path::Path, backend: Backend) -> Result<Option<String>> {
    let models = kernelopt::models::discover(repo, Some(backend.as_str()));
    if models.is_empty() {
        println!("· engine E2E: no local models found — skipping (set KERNELOPT_MODELS_DIR to enable)");
        return Ok(None);
    }
    println!("\nEngine E2E model (Gate 3 — proves the kernel helps a real model):");
    println!("   0) skip");
    let n = models.len().min(12);
    for (i, m) in models.iter().take(n).enumerate() {
        println!(
            "  {:>2}) {}  ({}, {})",
            i + 1,
            m.name,
            m.engine,
            kernelopt::models::human_size(m.size_bytes)
        );
    }
    let ans = ask_line(&format!("choose [0-{n}, default 0]: "))?;
    Ok(match kernelopt::wizard::parse_choice(&ans, n).unwrap_or(0) {
        0 => None,
        k => Some(models[k - 1].name.clone()),
    })
}

fn choose_preset() -> Result<String> {
    println!("\nHow hard should it search?");
    for (i, (name, desc)) in kernelopt::wizard::PRESETS.iter().enumerate() {
        println!("  {}) {name:<9} {desc}", i + 1);
    }
    let n = kernelopt::wizard::PRESETS.len();
    let ans = ask_line(&format!("choose [1-{n}, default 2]: "))?;
    let k = kernelopt::wizard::parse_choice(&ans, n).unwrap_or(2);
    Ok(kernelopt::wizard::PRESETS[k - 1].0.to_string())
}

/// Ask for the optimizer LLM, probing it first (auth + tool call) so a broken
/// model is caught before a run starts.
fn prompt_llm(mut resolved: ResolvedLlm) -> Result<(String, String)> {
    loop {
        println!(
            "\nOptimizer LLM: {}/{}",
            resolved.provider, resolved.model
        );
        match probe_llm(
            &resolved.provider,
            &resolved.model,
            resolved.base_url.as_deref(),
            resolved.api_key.as_deref(),
        ) {
            Ok(p) => match (p.models, p.tool_ok) {
                (Some(n), Some(true)) => println!("  probe: OK — {n} models, tool call works"),
                (Some(n), Some(false)) => println!(
                    "  probe: WARNING — {n} models, but no tool call; the pipeline needs tool calls"
                ),
                (_, _) => println!(
                    "  probe: {}",
                    p.error.unwrap_or_else(|| "inconclusive".into())
                ),
            },
            Err(e) => println!("  probe failed: {e:#}"),
        }
        if confirm("Use it?", true)? {
            return Ok((resolved.provider, resolved.model));
        }
        let provider = ask_line(&format!("provider [{}]: ", resolved.provider))?;
        let model = ask_line(&format!("model [{}]: ", resolved.model))?;
        if !provider.is_empty() {
            resolved.provider = provider;
        }
        if !model.is_empty() {
            resolved.model = model;
        }
    }
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
    let mut llm: BTreeMap<String, (u32, u64, u64)> = BTreeMap::new();
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
            Event::LlmCall { agent, prompt_tokens, completion_tokens, cached_tokens } => {
                let e = llm.entry(agent.clone()).or_default();
                e.0 += 1;
                e.1 += prompt_tokens + completion_tokens;
                e.2 += cached_tokens;
            }
            Event::RunFinished { outcome: o, speedup, root_cause, .. } => {
                outcome = Some((o.clone(), *speedup, root_cause.clone()));
            }
            _ => {}
        }
    }

    let (total_calls, total_tokens, total_cached) = llm
        .values()
        .fold((0u32, 0u64, 0u64), |(c, t, ch), (c2, t2, ch2)| {
            (c + c2, t + t2, ch + ch2)
        });
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
                "cached_prompt_tokens": total_cached,
                "llm_by_agent": llm.iter().map(|(k, (c, t, ch))| (k.clone(), serde_json::json!({"calls": c, "tokens": t, "cached": ch}))).collect::<serde_json::Map<_, _>>(),
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
    let cache_note = if total_cached > 0 {
        format!(
            "  (prefix cache: {} of prompt in, {:.0}% cached)",
            total_cached,
            100.0 * total_cached as f64 / total_tokens.max(1) as f64
        )
    } else {
        "  (no prefix-cache reporting from the endpoint)".to_string()
    };
    println!(
        "llm: {} calls, {} tokens — {}{}",
        total_calls,
        total_tokens,
        llm.iter()
            .map(|(a, (c, t, ch))| {
                let c = if *ch > 0 { format!("{c}/{t}t+{ch}c") } else { format!("{c}/{t}t") };
                format!("{} {c}", kernelopt::journal::agent_label(a))
            })
            .collect::<Vec<_>>()
            .join(", "),
        cache_note
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
    let unverified = count("unverified");
    let fallback = count("fallback");
    let failed = count("failed");
    speedups.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = speedups.get(speedups.len() / 2).copied().unwrap_or(1.0);
    let max = speedups.last().copied().unwrap_or(1.0);
    let wins = speedups.iter().filter(|s| **s > 1.0).count();

    println!("campaign {campaign_id}: {} targets", state.targets.len());
    println!(
        "outcomes: optimized {optimized}  matched {matched}  unverified {unverified}  fallback {fallback}  failed {failed}"
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
            Event::RunFinished { outcome, speedup, root_cause, .. } => {
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
