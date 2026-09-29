//! llama.cpp backend discovery: map a ggml op onto the `ggml/src/ggml-cuda`
//! kernel file(s) that implement it, and describe the `test-backend-ops`
//! gates. See `docs/llamacpp-mode.md`.
//!
//! The op→file mapping is curated (a `.cu` filename does not always equal its
//! ggml op name, e.g. `MUL_MAT` lives in `mmq.cu`/`mmvq.cu`/`mmvf.cu` and
//! `SWIGLU` lives in `unary.cu`). `perf = true` marks ops covered by
//! `test-backend-ops perf` (a timing authority exists).

use crate::backend::{Backend, Target};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const CUDA_DIR: &str = "ggml/src/ggml-cuda";

/// (ggml op name, in perf suite?, kernel file stems)
const OPS: &[(&str, bool, &[&str])] = &[
    ("MUL_MAT", true, &["mmq", "mmvq", "mmvf", "mmf"]),
    ("MUL_MAT_ID", true, &["mmid"]),
    ("FLASH_ATTN_EXT", true, &["fattn", "fattn-tile"]),
    ("SOFT_MAX", true, &["softmax", "softcap"]),
    ("L2_NORM", true, &["norm"]),
    ("RMS_NORM", false, &["norm"]),
    ("NORM", false, &["norm"]),
    ("GROUP_NORM", false, &["norm"]),
    ("ROPE", true, &["rope"]),
    ("GLU", true, &["unary"]),
    ("SWIGLU", true, &["unary"]),
    ("GELU", false, &["unary"]),
    ("SILU", false, &["unary"]),
    ("LEAKY_RELU", true, &["unary"]),
    ("CONT", true, &["cpy"]),
    ("CPY", true, &["cpy", "convert"]),
    ("ADD", true, &["binbcast"]),
    ("ADD_ID", true, &["add-id"]),
    ("CONCAT", false, &["concat"]),
    ("CONV_2D", true, &["conv2d"]),
    ("CONV_2D_DW", true, &["conv2d-dw"]),
    ("CONV_3D", true, &["conv3d"]),
    ("CONV_TRANSPOSE_2D", true, &["conv2d-transpose"]),
    ("ARGMAX", true, &["argmax"]),
    ("ARGSORT", true, &["argsort"]),
    ("CUMSUM", true, &["cumsum"]),
    ("MEAN", true, &["mean"]),
    ("SUM", true, &["sum"]),
    ("SUM_ROWS", true, &["sumrows"]),
    ("TOP_K", true, &["top-k"]),
    ("TRI", true, &["tri"]),
    ("SOLVE_TRI", true, &["solve_tri"]),
    ("IM2COL", true, &["im2col"]),
    ("COL2IM_1D", true, &["col2im-1d"]),
    ("ACC", true, &["acc"]),
    ("GATED_DELTA_NET", true, &["gated_delta_net"]),
    ("LIGHTNING_INDEXER", true, &["lightning-indexer"]),
    ("SSM_CONV", true, &["ssm-conv"]),
    ("SSM_SCAN", true, &["ssm-scan"]),
    ("PAD_REFLECT_1D", true, &["pad_reflect_1d"]),
    ("PAD", false, &["pad"]),
    ("GET_ROWS", false, &["getrows"]),
    ("SET_ROWS", false, &["set-rows"]),
    ("CLAMP", false, &["clamp"]),
    ("SCALE", false, &["scale"]),
    ("FILL", false, &["fill"]),
    ("ARANGE", false, &["arange"]),
    ("DIAG", false, &["diag"]),
    ("ROLL", false, &["roll"]),
    ("SNAKE", false, &["snake"]),
    ("FWHT", false, &["fwht"]),
    ("POOL_2D", false, &["pool2d"]),
    ("UPSCALE", false, &["upscale"]),
    ("OUT_PROD", false, &["out-prod"]),
    ("CROSS_ENTROPY_LOSS", false, &["cross-entropy-loss"]),
    ("OPT_STEP_ADAMW", false, &["opt-step-adamw"]),
];

fn norm(s: &str) -> String {
    s.to_ascii_uppercase().replace('-', "_").replace('.', "")
}

/// All runnable llama.cpp targets whose kernel files exist in `repo`.
pub fn discover_targets(repo: &Path) -> Result<Vec<Target>> {
    let cuda = repo.join(CUDA_DIR);
    if !cuda.is_dir() {
        anyhow::bail!("{} is not a llama.cpp tree (no {CUDA_DIR})", repo.display());
    }
    let mut out = Vec::new();
    for (op, perf, stems) in OPS {
        if let Some(t) = build_target(&cuda, op, *perf, stems) {
            out.push(t);
        }
    }
    if out.is_empty() {
        anyhow::bail!("no known ggml-cuda kernels found under {}", cuda.display());
    }
    Ok(out)
}

/// One target by op name (e.g. `SOFT_MAX`) or kernel stem (e.g. `softmax`).
pub fn discover_target(repo: &Path, op: &str) -> Result<Target> {
    let cuda = repo.join(CUDA_DIR);
    if !cuda.is_dir() {
        anyhow::bail!("{} is not a llama.cpp tree (no {CUDA_DIR})", repo.display());
    }
    let want = norm(op);
    for (name, perf, stems) in OPS {
        let stem_match = stems.iter().any(|s| norm(s) == want);
        if norm(name) == want || stem_match {
            return build_target(&cuda, name, *perf, stems)
                .with_context(|| format!("op {name} maps to missing kernel files in {}", cuda.display()));
        }
    }
    anyhow::bail!(
        "unknown llama.cpp op {op:?}; known ops: {}",
        OPS.iter().map(|(o, _, _)| *o).collect::<Vec<_>>().join(", ")
    )
}

fn build_target(cuda: &Path, op: &str, perf: bool, stems: &[&str]) -> Option<Target> {
    let mut kernel_files = Vec::new();
    let mut warnings = Vec::new();
    for stem in stems {
        for ext in ["cu", "cuh"] {
            let rel = format!("{CUDA_DIR}/{stem}.{ext}");
            if cuda.join(format!("{stem}.{ext}")).is_file() {
                kernel_files.push(rel);
            }
        }
    }
    if kernel_files.is_empty() {
        return None;
    }
    // Prefer a .cu as the edit unit; .cuh-only kernels are still editable.
    let target_file = kernel_files
        .iter()
        .find(|f| f.ends_with(".cu"))
        .cloned()
        .unwrap_or_else(|| kernel_files[0].clone());
    if !target_file.ends_with(".cu") {
        warnings.push(format!("{op}: only a header implementation found ({target_file})"));
    }
    if !perf {
        warnings.push(format!("{op}: not in the test-backend-ops perf suite — correctness-only"));
    }
    Some(Target {
        backend: Backend::Llamacpp,
        op: op.to_string(),
        family: op.to_string(),
        variant: None,
        kernel_files,
        context_files: vec![format!("{CUDA_DIR}/common.cuh")],
        contract_files: Vec::new(),
        target_file,
        build_targets: vec!["ggml-cuda".into(), "test-backend-ops".into()],
        test_filters: vec![op.to_string()],
        test_sources: Vec::new(),
        bench_binary: None,
        bench_args: Vec::new(),
        timing: perf,
        warnings,
        configure_args: crate::backend::llamacpp_configure_args(),
        test_cmd: None,
        bench_cmd: None,
        bench_format: None,
    })
}

/// Kernel files under `ggml/src/ggml-cuda` not covered by the curated map
/// (surfaced for visibility, not run by default).
pub fn unmapped_kernel_files(repo: &Path) -> Vec<String> {
    let cuda = repo.join(CUDA_DIR);
    let mapped: std::collections::HashSet<String> = OPS
        .iter()
        .flat_map(|(_, _, stems)| stems.iter().map(|s| s.to_string()))
        .collect();
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&cuda) {
        let mut names: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        names.sort();
        for p in names {
            if p.extension().and_then(|e| e.to_str()) != Some("cu") {
                continue;
            }
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if !mapped.contains(&stem) {
                out.push(format!("{CUDA_DIR}/{stem}.cu"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_normalizes() {
        assert_eq!(norm("soft-max"), "SOFT_MAX");
        assert_eq!(norm("llama.cpp"), "LLAMACPP");
    }
}
