//! Search structures (paper §4.3): DiverseSelect (two-phase beam selection),
//! UCB(c=1.4) exploration/exploitation, and the meltdown detector.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub chain: u32,      // chain origin
    pub iteration: u32,
    pub source: String,  // kernel source
    pub plan: String,
    pub latency_ms: Option<f64>,
    pub passed: bool,
    /// Git commit of this candidate in the worktree (if committed).
    #[serde(default)]
    pub commit: Option<String>,
    /// Executor's summary of what it changed.
    #[serde(default)]
    pub change_summary: Option<String>,
    /// Planner's implementation hints (the "how").
    #[serde(default)]
    pub hints: Option<String>,
    /// Planner's evidence (the "why it should be faster").
    #[serde(default)]
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamNode {
    pub candidate: Candidate,
    pub expansions: u32, // times this chain was expanded (for UCB)
}

/// DiverseSelect (Eq. 3): phase 1 = best per chain (exploration);
/// phase 2 = fill remaining slots by global latency (exploitation).
pub fn diverse_select(candidates: &[Candidate], beam_width: usize) -> Vec<Candidate> {
    let mut chains: Vec<u32> = candidates.iter().map(|c| c.chain).collect();
    chains.sort();
    chains.dedup();

    let mut selected: Vec<Candidate> = Vec::new();

    // Phase 1: best per chain among passed candidates.
    for chain in &chains {
        let mut best: Option<&Candidate> = None;
        for c in candidates {
            if c.chain == *chain && c.passed {
                if best.map(|b| c.latency_ms < b.latency_ms).unwrap_or(true) {
                    best = Some(c);
                }
            }
        }
        if let Some(b) = best {
            selected.push(b.clone());
        }
    }

    // Phase 2: fill remaining slots by global latency among passed.
    let mut rest: Vec<&Candidate> = candidates
        .iter()
        .filter(|c| c.passed && !selected.iter().any(|s| s.id == c.id))
        .collect();
    rest.sort_by(|a, b| {
        a.latency_ms
            .unwrap_or(f64::INFINITY)
            .partial_cmp(&b.latency_ms.unwrap_or(f64::INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for c in rest {
        if selected.len() >= beam_width {
            break;
        }
        selected.push(c.clone());
    }

    // If nothing passed, keep baseline slots empty (caller falls back).
    selected
}

/// UCB score for a chain: exploitation (best latency) vs exploration
/// (few expansions). Lower latency → higher score; fewer expansions → bonus.
pub fn ucb_score(best_latency: Option<f64>, expansions: u32, total_rounds: u32, c: f64) -> f64 {
    let exploit = 1.0 / (1.0 + best_latency.unwrap_or(f64::INFINITY));
    let explore = c * (libm_ln(1.0 + total_rounds as f64) / libm_ln(1.0 + expansions as f64)).sqrt();
    exploit + explore
}

fn libm_ln(x: f64) -> f64 {
    x.ln()
}

/// Meltdown detector: when the last ≤6 plan directions collapse to ≤2 unique
/// normalized approaches, trigger diversity enforcement.
pub fn meltdown_detected(recent_directions: &[String], window: usize, min_unique: usize) -> bool {
    let recent: Vec<String> = recent_directions
        .iter()
        .rev()
        .take(window)
        .map(|d| normalize_direction(d))
        .collect();
    if recent.len() < 3 {
        return false; // not enough signal yet
    }
    let mut unique: Vec<String> = recent.clone();
    unique.sort();
    unique.dedup();
    unique.len() <= min_unique
}

fn normalize_direction(d: &str) -> String {
    d.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, chain: u32, latency: Option<f64>, passed: bool) -> Candidate {
        Candidate {
            id: id.into(),
            chain,
            iteration: 0,
            source: String::new(),
            plan: String::new(),
            latency_ms: latency,
            passed,
            commit: None,
            change_summary: None,
            hints: None,
            evidence: None,
        }
    }

    #[test]
    fn diverse_select_best_per_chain_then_global() {
        let cands = vec![
            cand("a1", 0, Some(2.0), true),
            cand("a2", 0, Some(1.5), true),
            cand("b1", 1, Some(3.0), true),
            cand("b2", 1, Some(2.8), true),
            cand("c1", 2, Some(1.0), true),
        ];
        let sel = diverse_select(&cands, 3);
        let ids: Vec<&str> = sel.iter().map(|c| c.id.as_str()).collect();
        // best per chain: a2, b2, c1
        assert_eq!(ids, vec!["a2", "b2", "c1"]);
    }

    #[test]
    fn diverse_select_fills_by_global_when_chains_few() {
        let cands = vec![
            cand("a1", 0, Some(2.0), true),
            cand("a2", 0, Some(1.0), true),
            cand("b1", 1, Some(1.5), true),
        ];
        let sel = diverse_select(&cands, 3);
        let ids: Vec<&str> = sel.iter().map(|c| c.id.as_str()).collect();
        // best per chain: a2 (1.0), b1 (1.5); remaining slot → next best overall (a1, 2.0)
        assert_eq!(ids, vec!["a2", "b1", "a1"]);
    }

    #[test]
    fn diverse_select_skips_failed() {
        let cands = vec![
            cand("a1", 0, Some(0.5), false),
            cand("a2", 0, Some(1.0), true),
        ];
        let sel = diverse_select(&cands, 2);
        assert_eq!(sel.len(), 1);
        assert_eq!(sel[0].id, "a2");
    }

    #[test]
    fn meltdown_triggers_on_collapse() {
        let dirs: Vec<String> = vec![
            "increase XBLOCK".into(),
            "increase XBLOCK!".into(),
            "Increase XBLOCK".into(),
            "add autotune".into(),
        ];
        assert!(meltdown_detected(&dirs, 6, 2));
        let varied: Vec<String> = vec![
            "increase XBLOCK".into(),
            "add autotune".into(),
            "vectorized loads".into(),
            "evict_first hints".into(),
        ];
        assert!(!meltdown_detected(&varied, 6, 2));
    }

    #[test]
    fn ucb_prefers_underexpanded() {
        let explored = ucb_score(Some(1.0), 10, 10, 1.4);
        let fresh = ucb_score(Some(1.0), 1, 10, 1.4);
        assert!(fresh > explored);
    }
}
