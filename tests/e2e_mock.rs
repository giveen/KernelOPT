//! Happy-path E2E test: a scripted mock LLM submits a valid candidate that
//! must pass Gates 1–2, bench, DiverseSelect, restitch, and Gates 3–4.
//! Requires CUDA; run with: cargo test --test e2e_mock -- --ignored --nocapture

use kernelopt::config::{Config, Hyper, ProfilerMode};
use kernelopt::journal::Journal;
use kernelopt::llm::{Completion, MockClient, ToolCall};
use kernelopt::pipeline::Pipeline;
use kernelopt::runner_bridge::RunnerBridge;

const CANDIDATE_TAIL: &str = r#"

_BASE = None

def init_with_model(m):
    global _BASE
    _BASE = m

def kernel_function(x):
    return _BASE(x)
"#;

#[test]
#[ignore = "requires CUDA GPU"]
fn happy_path_mock_walk() {
    let model_file = std::fs::canonicalize("examples/mlp.py").unwrap();
    let source = std::fs::read_to_string(&model_file).unwrap();
    let candidate = format!("{source}{CANDIDATE_TAIL}");

    // Scripted: 1 planner plan, 1 executor submission (passes gates 1-2).
    let scripted = vec![
        Completion {
            content: None,
            tool_calls: vec![ToolCall {
                name: "submit_plan".into(),
                arguments: serde_json::json!({"plan": {"change": "identity baseline walk"}}),
            }],
            usage: Some(kernelopt::llm::Usage { prompt_tokens: 10, completion_tokens: 5, cached_tokens: 0 }),
        },
        Completion {
            content: None,
            tool_calls: vec![ToolCall {
                name: "submit_kernel".into(),
                arguments: serde_json::json!({"kernel_source": candidate, "change_summary": "identity"}),
            }],
            usage: Some(kernelopt::llm::Usage { prompt_tokens: 10, completion_tokens: 5, cached_tokens: 0 }),
        },
    ];

    let hyper = Hyper { t_iterations: 1, b_beam: 1, k_retries: 1, ..Default::default() };
    let cfg = Config::load(
        "mock".into(),
        "mock".into(),
        None,
        None,
        hyper,
        ProfilerMode::Both,
        "full".into(),
        "get_inputs".into(),
        None,
    )
    .unwrap();

    let run_id = format!("mock_walk_{}", chrono::Utc::now().format("%H%M%S"));
    let mut journal = Journal::create(&cfg.runs_dir, &run_id).unwrap();
    let llm = MockClient::new(scripted);
    let runner = RunnerBridge::new(cfg.runner_dir.clone());

    let wrapped = RunnerBridge::new(cfg.runner_dir.clone()); // not graphsignal-wrapped in tests
    let mut pipe = Pipeline {
        cfg: &cfg,
        llm: &llm,
        runner: &runner,
        wrapped_runner: &wrapped,
        journal: &mut journal,
        session_id: run_id.clone(),
        model_file: model_file.to_string_lossy().to_string(),
        baseline_weights: None,
        memory: kernelopt::memory::ExperienceMemory::new(8, 1.05, 1.20),
        tracker: kernelopt::memory::StrategyTracker::default(),
        next_signals_port: 18299,
        last_bench_signals: None,
    };

    let result = pipe.run(&model_file.to_string_lossy()).unwrap();
    println!("result: {result:#}");
    assert_eq!(result["outcome"], "optimized");
    assert!(result["speedup"].as_f64().unwrap() > 0.9); // identity walk ~1.0x
}
