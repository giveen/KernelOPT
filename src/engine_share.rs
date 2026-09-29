//! Engine-share profiling: rank kernels by GPU time in a real workload run.
//!
//! Replaces the Graphsignal dependency for ninfer/llama.cpp. Backends:
//!   - `nsys` (default) — kernels **and** memcpy, CUDA graphs, no admin counters
//!   - `ncu`            — kernel-only fallback (needs `RmProfilingAdminOnly=0`)
//!   - `graphsignal`    — optional; the only backend with ROCm support
//!
//! All three feed the same `attribution::rank_targets` / `order_by_share`.

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Preferred backend when the user asks for `auto`: nsys > ncu > graphsignal.
pub fn auto_engine() -> &'static str {
    if crate::exec::find_nsys().is_some() {
        "nsys"
    } else if crate::exec::find_ncu().is_some() {
        "ncu"
    } else {
        "graphsignal"
    }
}

/// Trace `workload` with Nsight Systems and return `(symbol, ns)` for kernels
/// and memory copies (so copies aren't invisible, unlike with ncu).
pub fn nsys_kernel_times(workload: &[String], timeout_secs: u64) -> Result<Vec<(String, f64)>> {
    let dir = std::env::current_dir()
        .context("cwd")?
        .join(".kernelopt")
        .join("nsys");
    std::fs::create_dir_all(&dir).context("creating .kernelopt/nsys")?;
    let prefix = dir.join(uuid::Uuid::new_v4().simple().to_string());
    crate::exec::nsys_profile(workload, &prefix, timeout_secs)?;
    let rep = PathBuf::from(format!("{}.nsys-rep", prefix.display()));

    let mut out = crate::parse::parse_nsys_sum(&crate::exec::nsys_stats(
        &rep,
        "cuda_gpu_kern_sum",
        600,
    )?);
    if let Ok(mem) = crate::exec::nsys_stats(&rep, "cuda_gpu_mem_time_sum", 600) {
        out.extend(crate::parse::parse_nsys_sum(&mem));
    }
    Ok(out)
}

/// Run `workload` under ncu and sum per-kernel durations (kernel-only).
pub fn ncu_kernel_times(workload: &[String], timeout_secs: u64) -> Result<Vec<(String, f64)>> {
    let resp = crate::exec::ncu(workload, "basic", Some(20), Some(500), None, timeout_secs)?;
    Ok(crate::parse::ncu_kernel_times(
        resp["raw_stdout"].as_str().unwrap_or(""),
    ))
}
