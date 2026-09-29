//! Map Graphsignal CUDA kernel symbols onto discovered optimization targets.
//!
//! Graphsignal ranks GPU time by kernel symbol (`cuda_kernels_nanoseconds`
//! frames). This module answers the paper's P4 question — *what does the engine
//! actually spend time on?* — by attributing that time to the ops KernelOPT can
//! edit, so a campaign can optimize the hot ones first.
//!
//! Matching is heuristic (symbols are mangled and filenames don't always equal
//! op names): a kernel is attributed to the target whose op/family/variant
//! tokens it contains, most specific first.

use crate::backend::Target;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct TargetShare {
    pub op: String,
    pub family: String,
    /// Cumulative nanoseconds attributed to this target.
    pub ns: f64,
    /// Share of total attributed GPU kernel time.
    pub share_pct: f64,
    /// The kernel symbols that matched.
    pub kernels: Vec<String>,
}

/// Token-match score of a kernel symbol against a target (0 = no match).
fn score(symbol: &str, target: &Target) -> u32 {
    let sym = symbol.to_ascii_lowercase();
    let mut s = 0u32;
    if !target.op.is_empty() && sym.contains(&target.op.to_ascii_lowercase()) {
        s += 4;
    }
    if !target.family.is_empty() && sym.contains(&target.family.to_ascii_lowercase()) {
        s += 3;
    }
    if let Some(v) = &target.variant {
        if !v.is_empty() && sym.contains(&v.to_ascii_lowercase()) {
            s += 2;
        }
    }
    s
}

/// Attribute `kernels` (symbol, ns) to `targets`. Returns the ranking (desc by
/// time) and the kernels no target matched.
pub fn rank_targets(
    kernels: &[(String, f64)],
    targets: &[Target],
) -> (Vec<TargetShare>, Vec<(String, f64)>) {
    let total: f64 = kernels.iter().map(|(_, ns)| ns).sum();
    let mut agg: HashMap<usize, (f64, Vec<String>)> = HashMap::new();
    let mut unattributed: Vec<(String, f64)> = Vec::new();

    for (symbol, ns) in kernels {
        let best = targets
            .iter()
            .enumerate()
            .map(|(i, t)| (i, score(symbol, t)))
            .max_by_key(|(_, s)| *s);
        match best {
            Some((i, s)) if s > 0 => {
                let entry = agg.entry(i).or_insert((0.0, Vec::new()));
                entry.0 += ns;
                entry.1.push(symbol.clone());
            }
            _ => unattributed.push((symbol.clone(), *ns)),
        }
    }

    let mut ranking: Vec<TargetShare> = agg
        .into_iter()
        .map(|(i, (ns, kernels))| TargetShare {
            op: targets[i].op.clone(),
            family: targets[i].family.clone(),
            ns,
            share_pct: if total > 0.0 { ns / total * 100.0 } else { 0.0 },
            kernels,
        })
        .collect();
    ranking.sort_by(|a, b| b.ns.partial_cmp(&a.ns).unwrap_or(std::cmp::Ordering::Equal));
    unattributed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    (ranking, unattributed)
}

/// Reorder targets so the highest engine-share ops come first; ops with no
/// measured share keep their original (complexity) order at the end.
pub fn order_by_share(targets: &mut Vec<Target>, share: &HashMap<String, f64>) {
    targets.sort_by(|a, b| {
        let sa = share.get(&a.op).copied().unwrap_or(-1.0);
        let sb = share.get(&b.op).copied().unwrap_or(-1.0);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.complexity().cmp(&b.complexity()))
            .then_with(|| a.op.cmp(&b.op))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    fn target(op: &str, family: &str, variant: Option<&str>) -> Target {
        Target {
            backend: Backend::Ninfer,
            op: op.into(),
            family: family.into(),
            variant: variant.map(|v| v.to_string()),
            kernel_files: vec![format!("src/ops/{op}.cu")],
            context_files: vec![],
            contract_files: vec![],
            target_file: format!("src/ops/{op}.cu"),
            build_targets: vec![],
            test_filters: vec![],
            bench_binary: None,
            bench_args: vec![],
            timing: true,
            warnings: vec![],
        }
    }

    #[test]
    fn attributes_variant_over_family() {
        let targets = vec![
            target("fp8_linear_add", "linear_add", Some("fp8")),
            target("q8_linear_add", "linear_add", Some("q8")),
            target("add_bias", "add_bias", None),
        ];
        let kernels = vec![
            ("fp8_linear_add_a8_kernel".to_string(), 100.0),
            ("add_bias_bf16x8_kernel".to_string(), 50.0),
            ("unrelated_kernel".to_string(), 10.0),
        ];
        let (rank, un) = rank_targets(&kernels, &targets);
        assert_eq!(rank[0].op, "fp8_linear_add");
        assert!((rank[0].share_pct - 100.0 / 160.0 * 100.0).abs() < 0.01);
        assert_eq!(rank[1].op, "add_bias");
        assert_eq!(un.len(), 1);
        assert_eq!(un[0].0, "unrelated_kernel");
    }

    #[test]
    fn order_by_share_puts_hot_first() {
        let mut targets = vec![
            target("add_bias", "add_bias", None),
            target("fp8_linear_add", "linear_add", Some("fp8")),
        ];
        let mut share = HashMap::new();
        share.insert("fp8_linear_add".to_string(), 80.0);
        order_by_share(&mut targets, &share);
        assert_eq!(targets[0].op, "fp8_linear_add");
    }
}
