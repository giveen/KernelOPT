//! Read-only integration test for `kernelopt discover` against a real ninfer
//! checkout (docs/ninfer-mode.md §9). Soft-skips when no checkout is present so
//! `cargo test` passes on machines without ninfer.
//!
//! Run explicitly with:
//!   NINFER_REPO=/path/to/ninfer cargo test --test discover_real -- --nocapture

use std::path::PathBuf;

fn ninfer_repo() -> Option<PathBuf> {
    std::env::var("NINFER_REPO")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("src/ops").is_dir())
}

#[test]
fn discovers_real_ninfer_ops() {
    let Some(repo) = ninfer_repo() else {
        eprintln!("skip: no ninfer checkout (set NINFER_REPO)");
        return;
    };

    // P0 basic op: everything must line up.
    let basic = kernelopt::ninfer::discover(&repo, "add_bias").unwrap();
    assert_eq!(basic.family, "add_bias");
    assert_eq!(basic.variant, None);
    assert!(
        basic.kernel_files.iter().any(|f| f.ends_with("kernel/add_bias.cuh")),
        "kernel file missing: {:?}",
        basic.kernel_files
    );
    assert!(
        basic.contract_files.iter().any(|f| f.ends_with("ops/add_bias.h")),
        "contract missing: {:?}",
        basic.contract_files
    );
    assert!(
        basic.tests.iter().any(|t| t.target == "ninfer_add_bias_test"),
        "test target missing: {:?}",
        basic.tests
    );
    assert!(
        basic.benches.iter().any(|b| b.target == "ninfer_add_bias_bench"),
        "bench target missing: {:?}",
        basic.benches
    );
    eprintln!("add_bias: {:#?}", basic);

    // P2 quant variant: `fp8_linear_add` must resolve to family `linear_add`.
    let fp8 = kernelopt::ninfer::discover(&repo, "fp8_linear_add").unwrap();
    assert_eq!(fp8.family, "linear_add");
    assert_eq!(fp8.variant.as_deref(), Some("fp8"));
    assert!(!fp8.kernel_files.is_empty(), "no fp8 kernel files");
    assert!(
        fp8.kernel_files.iter().all(|f| f.contains("linear_add/fp8")),
        "kernel files escaped the fp8 variant dir: {:?}",
        fp8.kernel_files
    );
    assert!(
        fp8.tests.iter().any(|t| t.sources.iter().any(|s| s.contains("test_fp8.cpp"))),
        "fp8 test missing: {:?}",
        fp8.tests
    );
    assert!(
        fp8.benches.iter().any(|b| b.target == "ninfer_fp8_linear_add_bench"),
        "fp8 bench missing: {:?}",
        fp8.benches
    );
    eprintln!("fp8_linear_add: {:#?}", fp8);
}
