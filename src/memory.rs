//! Experience memory (paper §4.5): bounded FIFO queue with asymmetric
//! thresholds (store if speedup ≥ s+ or regression ≥ s−), Summarizer's
//! append/replace/skip contract, and a cross-run strategy tracker.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceItem {
    pub item_id: String,
    pub iteration: u32,
    pub speedup: f64,
    pub rewrite_type: String,
    pub framework: String,
    pub direction: String,
    pub profiling_signal: String,
    pub strategy_title: String,
    pub strategy_description: String,
    #[serde(default)]
    pub slow_pseudocode: String,
    #[serde(default)]
    pub fast_pseudocode: String,
    #[serde(default)]
    pub applicable_when: String,
    #[serde(default)]
    pub do_not_apply_when: String,
    #[serde(default)]
    pub framework_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUpdate {
    pub action: String, // append | replace | skip
    #[serde(default)]
    pub replace_item_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Parse the Summarizer's "two JSON objects separated by a blank line" output.
pub fn parse_summarizer_output(text: &str) -> Result<(Option<ExperienceItem>, Option<MemoryUpdate>)> {
    let mut blocks = text.split("\n\n");
    // Tolerate prose before the first JSON object.
    let objs: Vec<&str> = text
        .split("\n\n")
        .filter(|b| b.trim_start().starts_with('{'))
        .collect();
    let _ = &mut blocks;

    let item: Option<ExperienceItem> = objs
        .first()
        .and_then(|b| serde_json::from_str(b.trim()).ok());
    let update: Option<MemoryUpdate> = objs
        .get(1)
        .and_then(|b| serde_json::from_str(b.trim()).ok());
    Ok((item, update))
}

pub struct ExperienceMemory {
    queue: Vec<ExperienceItem>,
    capacity: usize,
    s_plus: f64,
    s_minus: f64,
}

impl serde::Serialize for ExperienceMemory {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("ExperienceMemory", 4)?;
        st.serialize_field("queue", &self.queue)?;
        st.serialize_field("capacity", &self.capacity)?;
        st.serialize_field("s_plus", &self.s_plus)?;
        st.serialize_field("s_minus", &self.s_minus)?;
        st.end()
    }
}

impl<'de> serde::Deserialize<'de> for ExperienceMemory {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Raw {
            queue: Vec<ExperienceItem>,
            capacity: usize,
            s_plus: f64,
            s_minus: f64,
        }
        let r = Raw::deserialize(d)?;
        Ok(Self {
            queue: r.queue,
            capacity: r.capacity,
            s_plus: r.s_plus,
            s_minus: r.s_minus,
        })
    }
}

impl ExperienceMemory {
    pub fn new(capacity: usize, s_plus: f64, s_minus: f64) -> Self {
        Self {
            queue: Vec::new(),
            capacity,
            s_plus,
            s_minus,
        }
    }

    /// Should this experience be stored at all? (asymmetric thresholds)
    pub fn should_store(&self, speedup: f64) -> bool {
        speedup >= self.s_plus || (1.0 / speedup) >= self.s_minus
    }

    /// Apply the Summarizer's decision; returns the action taken.
    pub fn apply(&mut self, item: ExperienceItem, update: MemoryUpdate) -> &'static str {
        match update.action.as_str() {
            "append" => {
                self.queue.push(item);
                if self.queue.len() > self.capacity {
                    self.queue.remove(0);
                }
                "append"
            }
            "replace" => {
                if let (Some(id), Some(new_item)) = (update.replace_item_id.clone(), Some(item)) {
                    if let Some(pos) = self.queue.iter().position(|e| e.item_id == id) {
                        self.queue[pos] = new_item;
                        return "replace";
                    }
                }
                "skip"
            }
            _ => "skip",
        }
    }

    /// Render memory as planner context (serialized items).
    pub fn context(&self) -> String {
        if self.queue.is_empty() {
            return "(memory empty — no past optimization experience)".into();
        }
        self.queue
            .iter()
            .map(|e| {
                format!(
                    "- [{}] {} (speedup {:.2}, direction: {}): {} | applies when: {} | avoid when: {}",
                    e.item_id,
                    e.strategy_title,
                    e.speedup,
                    e.direction,
                    e.strategy_description,
                    e.applicable_when,
                    e.do_not_apply_when
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Cross-run strategy tracker: per-kernel-type success rates; <30% success
/// after ≥3 attempts → AVOID.
#[derive(Default, Serialize, Deserialize)]
pub struct StrategyTracker {
    /// kernel_type -> strategy -> (attempts, successes)
    stats: HashMap<String, HashMap<String, (u32, u32)>>,
}

impl StrategyTracker {
    pub fn record(&mut self, kernel_type: &str, strategy: &str, success: bool) {
        let e = self
            .stats
            .entry(kernel_type.to_string())
            .or_default()
            .entry(strategy.to_string())
            .or_insert((0, 0));
        e.0 += 1;
        if success {
            e.1 += 1;
        }
    }

    pub fn avoid_flags(&self, kernel_type: &str) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(m) = self.stats.get(kernel_type) {
            for (strategy, (attempts, successes)) in m {
                if *attempts >= 3 {
                    let rate = *successes as f64 / *attempts as f64;
                    if rate < 0.30 {
                        out.push(strategy.clone());
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(speedup: f64) -> ExperienceItem {
        ExperienceItem {
            item_id: format!("it{speedup}"),
            iteration: 1,
            speedup,
            rewrite_type: "tiling".into(),
            framework: "triton".into(),
            direction: "bigger XBLOCK".into(),
            profiling_signal: "SOL mem 45%".into(),
            strategy_title: "bigger tiles".into(),
            strategy_description: "increase tile when memory-bound and occupancy low".into(),
            slow_pseudocode: String::new(),
            fast_pseudocode: String::new(),
            applicable_when: String::new(),
            do_not_apply_when: String::new(),
            framework_notes: String::new(),
        }
    }

    #[test]
    fn asymmetric_thresholds() {
        let m = ExperienceMemory::new(8, 1.05, 1.20);
        assert!(m.should_store(1.06)); // win ≥5%
        assert!(!m.should_store(1.04)); // marginal
        assert!(m.should_store(0.80)); // regression ≥20%
        assert!(!m.should_store(0.90)); // small regression
    }

    #[test]
    fn fifo_capacity() {
        let mut m = ExperienceMemory::new(2, 1.05, 1.20);
        for s in [1.1, 1.2, 1.3] {
            m.apply(item(s), MemoryUpdate { action: "append".into(), replace_item_id: None, reason: None });
        }
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn summarizer_two_json() {
        let text = r#"{"item_id":"a","iteration":0,"speedup":1.2,"rewrite_type":"t","framework":"triton","direction":"d","profiling_signal":"s","strategy_title":"t","strategy_description":"x"}

{"action":"append","replace_item_id":null,"reason":"new"}"#;
        let (item, upd) = parse_summarizer_output(text).unwrap();
        assert!(item.is_some());
        assert_eq!(upd.unwrap().action, "append");
    }

    #[test]
    fn avoid_after_3_low_success() {
        let mut t = StrategyTracker::default();
        for _ in 0..3 {
            t.record("pointwise", "warp_specialization", false);
        }
        t.record("pointwise", "autotune", true);
        assert_eq!(t.avoid_flags("pointwise"), vec!["warp_specialization"]);
    }

    #[test]
    fn memory_serde_roundtrip_preserves_learning() {
        let mut m = ExperienceMemory::new(8, 1.05, 1.20);
        m.apply(
            item(1.3),
            MemoryUpdate { action: "append".into(), replace_item_id: None, reason: None },
        );
        let text = serde_json::to_string(&m).unwrap();
        let restored: ExperienceMemory = serde_json::from_str(&text).unwrap();
        assert_eq!(restored.len(), 1);
        assert!(restored.should_store(1.06)); // thresholds survived
        assert!(restored.context().contains("bigger tiles"));
    }
}
