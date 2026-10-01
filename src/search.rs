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
    diverse_select_indices(candidates, beam_width)
        .into_iter()
        .map(|i| candidates[i].clone())
        .collect()
}

/// DiverseSelect over beam nodes, preserving each node's UCB expansion count.
pub fn diverse_select_nodes(nodes: &[BeamNode], beam_width: usize) -> Vec<BeamNode> {
    let candidates: Vec<Candidate> = nodes.iter().map(|n| n.candidate.clone()).collect();
    diverse_select_indices(&candidates, beam_width)
        .into_iter()
        .map(|i| nodes[i].clone())
        .collect()
}

/// Index form of DiverseSelect (shared by the `Candidate` and `BeamNode` fns).
fn diverse_select_indices(candidates: &[Candidate], beam_width: usize) -> Vec<usize> {
    let mut chains: Vec<u32> = candidates.iter().map(|c| c.chain).collect();
    chains.sort();
    chains.dedup();

    let mut selected: Vec<usize> = Vec::new();
    let mut selected_ids: Vec<String> = Vec::new();

    // Phase 1: best per chain among passed candidates.
    for chain in &chains {
        let mut best: Option<usize> = None;
        for (i, c) in candidates.iter().enumerate() {
            if c.chain == *chain && c.passed {
                if best.map(|b| c.latency_ms < candidates[b].latency_ms).unwrap_or(true) {
                    best = Some(i);
                }
            }
        }
        if let Some(i) = best {
            selected.push(i);
            selected_ids.push(candidates[i].id.clone());
        }
    }

    // Phase 2: fill remaining slots by global latency among passed.
    let mut rest: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.passed && !selected_ids.contains(&c.id))
        .map(|(i, _)| i)
        .collect();
    rest.sort_by(|&a, &b| {
        candidates[a]
            .latency_ms
            .unwrap_or(f64::INFINITY)
            .partial_cmp(&candidates[b].latency_ms.unwrap_or(f64::INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in rest {
        if selected.len() >= beam_width {
            break;
        }
        selected.push(i);
    }

    // If nothing passed, keep baseline slots empty (caller falls back).
    selected
}

/// UCB score for a chain: exploitation (best latency) vs exploration
/// (few expansions). Lower latency → higher score; fewer expansions → bonus.
/// An underexpanded arm (0 expansions) scores `+∞` so it is tried first.
pub fn ucb_score(best_latency: Option<f64>, expansions: u32, total_rounds: u32, c: f64) -> f64 {
    let exploit = 1.0 / (1.0 + best_latency.unwrap_or(f64::INFINITY));
    // UCB1 convention: an unvisited arm must be explored. This also avoids a
    // divide-by-zero (ln(1 + 0) = 0) in the exploration term.
    if expansions == 0 {
        return f64::INFINITY;
    }
    let explore = c * (libm_ln(1.0 + total_rounds as f64) / libm_ln(1.0 + expansions as f64)).sqrt();
    exploit + explore
}

/// Distribute one iteration's `n_plans` expansions across the beam by UCB(c).
///
/// Each slot picks the arm maximizing
/// `ucb_score(latency, expansions, total, c)`, then charges that arm one
/// expansion so later slots in the same iteration see it as more expanded.
/// Returns the chosen arm index per slot, in order. Ties break to the lower
/// index, so unexpanded arms are tried before any arm repeats. `n_plans == 0`
/// falls back to expanding every arm once (the pre-UCB behavior).
pub fn allocate_expansions(arms: &[BeamNode], n_plans: u32, c: f64) -> Vec<usize> {
    if arms.is_empty() {
        return Vec::new();
    }
    let n = if n_plans == 0 { arms.len() as u32 } else { n_plans };
    let mut pending: Vec<u32> = arms.iter().map(|a| a.expansions).collect();
    let mut total: u32 = pending.iter().sum();
    let mut out: Vec<usize> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut best: Option<(usize, f64)> = None;
        for (i, arm) in arms.iter().enumerate() {
            let score = ucb_score(arm.candidate.latency_ms, pending[i], total, c);
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((i, score));
            }
        }
        let idx = best.map(|(i, _)| i).unwrap_or(0);
        pending[idx] += 1;
        total += 1;
        out.push(idx);
    }
    out
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

    #[test]
    fn ucb_unexpanded_arm_is_infinite() {
        assert!(ucb_score(Some(0.5), 0, 10, 1.4).is_infinite());
    }

    #[test]
    fn allocate_tries_every_arm_before_repeating() {
        let arms: Vec<BeamNode> = (0..3)
            .map(|i| BeamNode {
                candidate: cand(&format!("c{i}"), i as u32, Some(1.0 + i as f64), true),
                expansions: 0,
            })
            .collect();
        let mut slots = allocate_expansions(&arms, 3, 1.4);
        slots.sort();
        assert_eq!(slots, vec![0, 1, 2]);
    }

    #[test]
    fn allocate_prefers_faster_arm_after_exploration() {
        // Both arms tried once; the faster one should win the majority of the
        // next batch while the slower still gets explored.
        let arms = vec![
            BeamNode { candidate: cand("fast", 0, Some(0.5), true), expansions: 1 },
            BeamNode { candidate: cand("slow", 1, Some(2.0), true), expansions: 1 },
        ];
        let slots = allocate_expansions(&arms, 4, 1.4);
        let fast = slots.iter().filter(|&&i| i == 0).count();
        assert!(fast > 2, "faster arm should win the majority: {slots:?}");
    }

    #[test]
    fn allocate_defaults_to_one_per_arm_when_n_zero() {
        let arms: Vec<BeamNode> = (0..4)
            .map(|i| BeamNode {
                candidate: cand(&format!("c{i}"), i as u32, Some(1.0), true),
                expansions: 0,
            })
            .collect();
        assert_eq!(allocate_expansions(&arms, 0, 1.4).len(), 4);
    }

    #[test]
    fn allocate_empty_beam_is_empty() {
        assert!(allocate_expansions(&[], 4, 1.4).is_empty());
    }

    #[test]
    fn diverse_select_nodes_preserves_expansions() {
        let nodes = vec![
            BeamNode { candidate: cand("a", 0, Some(2.0), true), expansions: 7 },
            BeamNode { candidate: cand("b", 1, Some(1.0), true), expansions: 5 },
        ];
        let sel = diverse_select_nodes(&nodes, 2);
        assert_eq!(sel.len(), 2);
        assert!(sel.iter().any(|n| n.candidate.id == "a" && n.expansions == 7));
        assert!(sel.iter().any(|n| n.candidate.id == "b" && n.expansions == 5));
    }
}
