//! Graphsignal payload summarizer (attribution only — never gate timings).
//!
//! Extracts the planner-relevant slice of a `/signals` payload: per-kernel
//! cumulative time ranking (`cuda_kernels_nanoseconds` frames), memcpy/sync
//! totals, GPU telemetry, dropped-record sanity, and console errors.

use serde_json::{json, Value};

/// Metric names we care about (Graphsignal conventions).
const KERNELS: &str = "cuda_kernels_nanoseconds";
const KERNELS_ROCM: &str = "rocm_kernels_nanoseconds";
const GRAPHS: &str = "cuda_graphs_nanoseconds";
const MEMCPY_NS: &str = "cuda_memcpy_nanoseconds";
const SYNC_NS: &str = "cuda_sync_nanoseconds";
const DROPPED: &str = "cuda_dropped_records_total";
const TRACE_MODE: &str = "cuda_graph_trace_mode";
const GPU_UTIL: &str = "gpu_utilization_percent";

fn find_metric<'a>(payload: &'a Value, name: &str) -> Option<&'a Value> {
    payload
        .get("metrics")?
        .as_array()?
        .iter()
        .find(|m| m.get("name").and_then(|n| n.as_str()) == Some(name))
}

fn frames(metric: Option<&Value>, top: usize) -> Vec<(String, f64, u64)> {
    let mut out = Vec::new();
    if let Some(fs) = metric
        .and_then(|m| m.pointer("/stats/frames"))
        .and_then(|f| f.as_array())
    {
        for f in fs {
            let name = f.get("name").and_then(|n| n.as_str()).unwrap_or("?").to_string();
            let value = f.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let samples = f.get("samples").and_then(|s| s.as_u64()).unwrap_or(0);
            out.push((name, value, samples));
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top);
    }
    out
}

fn total_ns(metric: Option<&Value>) -> Option<f64> {
    metric
        .and_then(|m| m.pointer("/stats/total"))
        .and_then(|t| t.as_f64())
}

fn gauge(metric: Option<&Value>) -> Option<f64> {
    metric
        .and_then(|m| m.pointer("/stats/value"))
        .and_then(|v| v.as_f64())
}

/// Ranked `(kernel symbol, cumulative ns)` from a `/signals` payload (CUDA or
/// ROCm). Used for engine-share attribution of optimization targets.
pub fn kernel_times(payload: &Value, top: usize) -> Vec<(String, f64)> {
    let metric = find_metric(payload, KERNELS).or_else(|| find_metric(payload, KERNELS_ROCM));
    frames(metric, top)
        .into_iter()
        .map(|(name, ns, _)| (name, ns))
        .collect()
}

/// Build a compact planner-facing summary from a raw /signals payload.
pub fn summarize(payload: &Value, top_kernels: usize) -> Value {
    if payload.get("error").is_some() || payload.get("metrics").is_none() {
        return json!({
            "available": false,
            "note": payload.get("error").and_then(|e| e.as_str()).unwrap_or("no metrics in payload"),
        });
    }

    let kernels = frames(find_metric(payload, KERNELS), top_kernels);
    let graphs = frames(find_metric(payload, GRAPHS), 5);
    let dropped = find_metric(payload, DROPPED)
        .and_then(|m| m.pointer("/stats/total"))
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0);
    let trace_mode = gauge(find_metric(payload, TRACE_MODE));
    let gpu_util = gauge(find_metric(payload, GPU_UTIL));

    let kernel_lines: Vec<Value> = kernels
        .iter()
        .map(|(name, ns, samples)| {
            json!({
                "kernel": name,
                "cumulative_ms": ns / 1e6,
                "launches": samples,
            })
        })
        .collect();

    let graph_lines: Vec<Value> = graphs
        .iter()
        .map(|(name, ns, samples)| {
            json!({
                "graph": name,
                "cumulative_ms": ns / 1e6,
                "replays": samples,
            })
        })
        .collect();

    json!({
        "available": true,
        "source": "graphsignal (attribution only; do not treat as gate timing)",
        "trace_mode": trace_mode.map(|m| if m as u64 == 1 { "node" } else { "graph" }),
        "top_kernels": kernel_lines,
        "cuda_graphs": graph_lines,
        "memcpy_ms_total": total_ns(find_metric(payload, MEMCPY_NS)).map(|ns| ns / 1e6),
        "sync_ms_total": total_ns(find_metric(payload, SYNC_NS)).map(|ns| ns / 1e6),
        "gpu_utilization_percent": gpu_util,
        "dropped_records": dropped,
        "dropped_warning": if dropped > 0.0 {
            Some("cuda_dropped_records_total > 0: kernel timings above are INCOMPLETE")
        } else {
            None
        },
        "errors": payload
            .get("errors")
            .and_then(|e| e.as_array())
            .map(|errs| {
                errs.iter()
                    .filter(|e| e.get("level").and_then(|l| l.as_str()) == Some("error"))
                    .map(|e| {
                        e.get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("?")
                            .chars()
                            .take(300)
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        json!({
            "metrics": [
                {"name": "cuda_kernels_nanoseconds", "type": "profile", "stats": {"frames": [
                    {"name": "triton_poi_fused_addmm_gelu_0", "value": 8_100_000_000u64, "samples": 3200},
                    {"name": "triton_mm_27", "value": 5_500_000_000u64, "samples": 410},
                    {"name": "vectorized_elementwise", "value": 1_200_000_000u64, "samples": 900}
                ]}},
                {"name": "cuda_graph_trace_mode", "type": "gauge", "stats": {"value": 1}},
                {"name": "cuda_dropped_records_total", "type": "counter", "stats": {"total": 0}},
                {"name": "gpu_utilization_percent", "type": "gauge", "stats": {"value": 97.0}},
                {"name": "cuda_memcpy_nanoseconds", "type": "profile", "stats": {"total": 42_000_000u64}},
                {"name": "cuda_sync_nanoseconds", "type": "profile", "stats": {"total": 8_000_000u64}}
            ],
            "errors": [
                {"level": "error", "message": "CUDA out of memory"}
            ]
        })
    }

    #[test]
    fn extracts_kernel_ranking_and_warnings() {
        let s = summarize(&fixture(), 2);
        assert_eq!(s["available"], true);
        assert_eq!(s["trace_mode"], "node");
        let kernels = s["top_kernels"].as_array().unwrap();
        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[0]["kernel"], "triton_poi_fused_addmm_gelu_0");
        assert!((kernels[0]["cumulative_ms"].as_f64().unwrap() - 8100.0).abs() < 0.1);
        assert_eq!(s["dropped_records"], 0.0);
        assert_eq!(s["dropped_warning"], Value::Null);
        assert_eq!(s["errors"].as_array().unwrap().len(), 1);
        assert!((s["memcpy_ms_total"].as_f64().unwrap() - 42.0).abs() < 1e-6);
    }

    #[test]
    fn handles_missing_payload() {
        let s = summarize(&json!({"error": "could not read"}), 5);
        assert_eq!(s["available"], false);
        let s2 = summarize(&json!({}), 5);
        assert_eq!(s2["available"], false);
    }

    #[test]
    fn dropped_records_warn() {
        let mut p = fixture();
        p["metrics"][2]["stats"]["total"] = json!(17);
        let s = summarize(&p, 5);
        assert!(s["dropped_warning"].is_string());
    }
}
