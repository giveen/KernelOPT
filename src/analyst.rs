//! Deterministic strategy analyst (paper §4.2): classifies the NCU bottleneck
//! tier — near-optimal (either SOL > 80%), memory-bound (mem > comp, both
//! ≤ 80%), compute-bound (vice versa), underutilized (equal, ≤ 80%) — and
//! injects the top-3 NCU rules by estimated speedup into planner context.

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    NearOptimal,
    MemoryBound,
    ComputeBound,
    Underutilized,
}

impl Tier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::NearOptimal => "near_optimal",
            Tier::MemoryBound => "memory_bound",
            Tier::ComputeBound => "compute_bound",
            Tier::Underutilized => "underutilized",
        }
    }
}

/// Paper thresholds: either SOL > 80% → near-optimal; else compare the two.
pub fn classify_tier(sol_memory: Option<f64>, sol_compute: Option<f64>) -> Tier {
    const NEAR_OPTIMAL: f64 = 80.0;
    let mem = sol_memory.unwrap_or(0.0);
    let comp = sol_compute.unwrap_or(0.0);
    if mem > NEAR_OPTIMAL || comp > NEAR_OPTIMAL {
        Tier::NearOptimal
    } else if mem > comp {
        Tier::MemoryBound
    } else if comp > mem {
        Tier::ComputeBound
    } else {
        Tier::Underutilized
    }
}

/// One planner-facing context per kernel, ranked rules trimmed to top-N.
pub fn planning_context(ncu_context: &Value, top_rules: usize) -> Value {
    let kernels = ncu_context
        .get("kernels")
        .and_then(|k| k.as_array())
        .cloned()
        .unwrap_or_default();

    let mut per_kernel = Vec::new();
    for k in &kernels {
        let mem = k.get("sol_memory").and_then(|v| v.as_f64());
        let comp = k.get("sol_compute").and_then(|v| v.as_f64());
        let tier = classify_tier(mem, comp);
        per_kernel.push(serde_json::json!({
            "name": k.get("name").and_then(|n| n.as_str()).unwrap_or("?"),
            "sol_memory_pct": mem,
            "sol_compute_pct": comp,
            "duration_us": k.get("duration_us"),
            "registers": k.get("registers"),
            "achieved_occupancy_pct": k.get("achieved_occupancy"),
            "bottleneck_tier": tier.as_str(),
        }));
    }

    // Sort rules by estimated speedup, keep top-N (paper: top-3 injected).
    let mut rules: Vec<&Value> = ncu_context
        .get("rules")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    rules.sort_by(|a, b| {
        let av = a.get("estimated_speedup").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let bv = b.get("estimated_speedup").and_then(|v| v.as_f64()).unwrap_or(0.0);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<Value> = rules.into_iter().take(top_rules).cloned().collect();

    serde_json::json!({
        "kernels": per_kernel,
        "top_rules": top,
        "ncu_set": ncu_context.get("ncu_set"),
        "replay_mode": ncu_context.get("replay_mode"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tiers_match_paper_thresholds() {
        assert_eq!(classify_tier(Some(85.0), Some(40.0)), Tier::NearOptimal);
        assert_eq!(classify_tier(Some(30.0), Some(85.0)), Tier::NearOptimal);
        assert_eq!(classify_tier(Some(70.0), Some(40.0)), Tier::MemoryBound);
        assert_eq!(classify_tier(Some(40.0), Some(70.0)), Tier::ComputeBound);
        assert_eq!(classify_tier(Some(50.0), Some(50.0)), Tier::Underutilized);
    }

    #[test]
    fn planning_context_ranks_rules() {
        let ctx = json!({
            "kernels": [{"name": "triton_mm_27", "sol_memory": 45.0, "sol_compute": 20.0,
                         "duration_us": 38.9, "registers": 40, "achieved_occupancy": 62.5}],
            "rules": [
                {"kernel": "triton_mm_27", "rule": "OPT-1", "estimated_speedup": 12.5},
                {"kernel": "triton_mm_27", "rule": "OPT-2", "estimated_speedup": 35.1},
                {"kernel": "triton_mm_27", "rule": "OPT-3", "estimated_speedup": 2.0},
                {"kernel": "triton_mm_27", "rule": "OPT-4", "estimated_speedup": 8.8}
            ]
        });
        let pc = planning_context(&ctx, 3);
        assert_eq!(pc["kernels"][0]["bottleneck_tier"], "memory_bound");
        let rules = pc["top_rules"].as_array().unwrap();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0]["rule"], "OPT-2");
    }
}
