//! Bridge smoke test: exercise the Rust→runner path for a fast ninfer command
//! sequence (worktree create → bench → reset). Ignored: needs a ninfer checkout
//! and a pre-built `.kernelopt/ninfer/build`.
//!
//!   cargo test --test bridge_probe -- --ignored --nocapture

use kernelopt::runner_bridge::RunnerBridge;

#[test]
#[ignore = "requires a ninfer checkout + prebuilt worktree"]
fn worktree_create_bench_reset_sequence_via_bridge() {
    let repo = std::env::var("NINFER_REPO").expect("set NINFER_REPO to a ninfer checkout");
    let base = std::env::current_dir().unwrap().join(".kernelopt/ninfer");
    let worktree = base.join("worktree").to_string_lossy().to_string();
    let build_dir = base.join("build").to_string_lossy().to_string();
    let bridge = RunnerBridge::new("runner".into());

    let create = serde_json::json!({
        "command": "cuda_worktree",
        "repo": repo,
        "worktree_dir": worktree,
        "branch": "kernelopt/ninfer-add_bias",
        "base": "HEAD",
        "action": "create",
    });
    let r = bridge.call(&create).unwrap();
    assert_eq!(r["ok"], serde_json::json!(true), "create: {r:#}");

    let bench = serde_json::json!({
        "command": "cuda_bench",
        "build_dir": build_dir,
        "binary": "ninfer_add_bias_bench",
        "args": ["--d", "1152", "--columns", "4096"],
    });
    let b = bridge.call(&bench).unwrap();
    assert_eq!(b["ok"], serde_json::json!(true), "bench: {b:#}");
    assert!(b["median_us"].as_f64().unwrap() > 0.0);

    for _ in 0..3 {
        let reset = serde_json::json!({
            "command": "cuda_worktree",
            "worktree_dir": worktree,
            "base": "HEAD",
            "action": "reset",
        });
        let r = bridge.call(&reset).unwrap();
        assert_eq!(r["ok"], serde_json::json!(true), "reset: {r:#}");
    }
}
