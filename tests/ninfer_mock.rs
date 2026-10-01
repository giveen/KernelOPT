//! Walking-skeleton E2E for the CUDA pipeline (ninfer `add_bias`): real compile
//! + real ctest + real bench against a pre-built worktree, scripted mock LLM.
//!
//!   cargo test --test ninfer_mock -- --ignored --nocapture

use kernelopt::backend::Backend;
use kernelopt::config::{Config, Hyper, ProfilerMode};
use kernelopt::cuda_pipeline::CudaPipeline;
use kernelopt::journal::Journal;
use kernelopt::llm::{Completion, MockClient, ToolCall};
use kernelopt::memory::{ExperienceMemory, StrategyTracker};
use kernelopt::ninfer;
use kernelopt::runner_bridge::RunnerBridge;
use std::path::PathBuf;

fn repo() -> PathBuf {
    PathBuf::from(
        std::env::var("NINFER_REPO").expect("set NINFER_REPO to a ninfer checkout"),
    )
}

#[test]
#[ignore = "requires CUDA GPU + a pre-built .kernelopt/ninfer/build"]
fn ninfer_identity_walk_add_bias() {
    let repo = repo();
    let project_root = std::env::current_dir().unwrap();
    let base = project_root.join(".kernelopt/ninfer");
    let worktree = base.join("worktree");
    let build_dir = base.join("build");
    assert!(
        build_dir.join("bench/ninfer_add_bias_bench").exists(),
        "pre-build the worktree first (see docs/ninfer-mode.md)"
    );

    let target = ninfer::discover_target(&repo, "add_bias").unwrap();
    let baseline = std::fs::read_to_string(repo.join(&target.target_file)).unwrap();
    let candidate = format!("{baseline}\n// kernelopt identity walk (walking skeleton)\n");

    let scripted = vec![
        Completion {
            content: None,
            tool_calls: vec![ToolCall {
                name: "submit_plan".into(),
                arguments: serde_json::json!({"plan": {"change": "identity walk"}}),
            }],
            usage: Some(kernelopt::llm::Usage { prompt_tokens: 5, completion_tokens: 5 }),
        },
        Completion {
            content: None,
            tool_calls: vec![ToolCall {
                name: "submit_kernel".into(),
                arguments: serde_json::json!({"kernel_source": candidate, "change_summary": "comment"}),
            }],
            usage: Some(kernelopt::llm::Usage { prompt_tokens: 5, completion_tokens: 5 }),
        },
        Completion {
            content: Some("{\"item_id\":\"m1\",\"iteration\":0,\"speedup\":1.0,\"rewrite_type\":\"none\",\"framework\":\"cuda\",\"direction\":\"walk\",\"profiling_signal\":\"\",\"strategy_title\":\"walk\",\"strategy_description\":\"identity\"}".into()),
            tool_calls: vec![],
            usage: None,
        },
    ];

    let hyper = Hyper { t_iterations: 1, b_beam: 1, k_retries: 1, ..Default::default() };
    let cfg = Config::load(
        "mock".into(), "mock".into(), None, None, hyper,
        ProfilerMode::Ncu, "basic".into(), "get_inputs".into(), None,
    )
    .unwrap();

    let run_id = format!("ninfer_mock_{}", chrono::Utc::now().format("%H%M%S"));
    let run_dir = project_root.join(&cfg.runs_dir).join(&run_id);
    for sub in ["candidates", "bench", "ncu"] {
        std::fs::create_dir_all(run_dir.join(sub)).unwrap();
    }
    let mut journal = Journal::create(&cfg.runs_dir, &run_id).unwrap();
    let llm = MockClient::new(scripted);
    let runner = RunnerBridge::new(cfg.runner_dir.clone());

    let mut mem = ExperienceMemory::new(8, 1.05, 1.20);
    let mut tracker = StrategyTracker::default();
    let checkpoint_path = run_dir.join("checkpoint.json");
    let mut pipe = CudaPipeline {
        cfg: &cfg,
        llm: &llm,
        runner: &runner,
        journal: &mut journal,
        session_id: run_id.clone(),
        repo,
        target,
        worktree,
        build_dir,
        gpu_lock_path: kernelopt::gpu_lock::default_path(std::path::Path::new(".")),
        run_dir,
        ncu_set: "basic".into(),
        e2e: None,
        edit_mode: kernelopt::cuda_pipeline::EditMode::Full,
        baseline_e2e: None,
        base_sha: None,
        last_bench_noise_pct: None,
        last_bench_label: None,
        last_bench_gbs: None,
        last_bench_roofline_gbs: None,
        memory: &mut mem,
        tracker: &mut tracker,
        patience: 2,
        target_speedup: 0.0,
        min_improvement: 0.01,
        max_iterations: 1,
        deadline: None,
        llm_call_budget: None,
        llm_calls: 0,
        tokens: 0,
        checkpoint_path: Some(checkpoint_path),
        interrupted: None,
        verbose: false,
        watch: false,
        bench_shape: None,
        final_rounds: 1,
        verify_timeout_s: std::cell::Cell::new(kernelopt::cuda_pipeline::DEFAULT_VERIFY_TIMEOUT_S),
        recent_failures: Vec::new(),
    };

    let result = pipe.run().unwrap();
    println!("result: {result:#?}");
    assert!(
        result.outcome == "optimized" || result.outcome == "matched",
        "unexpected outcome {}",
        result.outcome
    );
    assert!(result.speedup.unwrap() > 0.9);
    let _ = Backend::Ninfer;
}
