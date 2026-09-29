//! Interactive setup for the common case.
//!
//! The flag surface grew large; `kernelopt wizard` detects what it can (repo,
//! backend, targets, provider, local models) and asks only the real choices,
//! then shows the equivalent command and runs it. This module holds the pure
//! parts (presets, argv construction, choice parsing) so they are testable.

use crate::backend::Backend;
use std::path::PathBuf;

/// Loop presets: (iterations, beam).
pub fn preset(name: &str) -> (u32, u32) {
    match name.trim().to_ascii_lowercase().as_str() {
        "quick" | "fast" => (2, 1),
        "thorough" | "deep" => (5, 3),
        _ => (3, 2), // "standard"
    }
}

pub const PRESETS: [(&str, &str); 3] = [
    ("quick", "2 iterations, beam 1 — a fast first look"),
    ("standard", "3 iterations, beam 2 — the default"),
    ("thorough", "5 iterations, beam 3 — more search, more cost"),
];

/// A fully-resolved wizard choice.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub backend: Backend,
    pub repo: PathBuf,
    /// A single op, or `None` with `all = true` for a campaign.
    pub op: Option<String>,
    pub all: bool,
    pub provider: String,
    pub model: String,
    /// `--e2e-weights` (path / bare name / `auto`), or None.
    pub e2e: Option<String>,
    pub iterations: u32,
    pub beam: u32,
    pub watch: bool,
}

impl Plan {
    /// The equivalent `kernelopt …` argv (the wizard runs exactly this).
    pub fn argv(&self) -> Vec<String> {
        let mut a: Vec<String> = Vec::new();
        if self.all {
            a.push("campaign".into());
            a.push("--mode".into());
            a.push(self.backend.as_str().into());
            a.push("--max-iterations".into());
            a.push(self.iterations.to_string());
            if let Some(op) = &self.op {
                a.push("--op".into());
                a.push(op.clone());
            }
        } else {
            a.push(
                match self.backend {
                    Backend::Ninfer => "run-ninfer",
                    Backend::Llamacpp => "run-llamacpp",
                }
                .into(),
            );
            if let Some(op) = &self.op {
                a.push("--op".into());
                a.push(op.clone());
            }
            a.push("--iterations".into());
            a.push(self.iterations.to_string());
            a.push("--beam".into());
            a.push(self.beam.to_string());
        }
        a.push("--repo".into());
        a.push(self.repo.to_string_lossy().into());
        a.push("--provider".into());
        a.push(self.provider.clone());
        a.push("--model".into());
        a.push(self.model.clone());
        if let Some(e) = &self.e2e {
            a.push("--e2e-weights".into());
            a.push(e.clone());
        }
        if self.watch {
            a.push("--watch".into());
        }
        a
    }

    /// Shell-quoted command line for display.
    pub fn display(&self) -> String {
        let mut out = String::from("kernelopt");
        for a in self.argv() {
            out.push(' ');
            if a.contains(' ') {
                out.push_str(&format!("{a:?}"));
            } else {
                out.push_str(&a);
            }
        }
        out
    }
}

/// Parse a `1..=n` menu choice; `0` means "all"/"none" (caller decides).
pub fn parse_choice(input: &str, n: usize) -> Option<usize> {
    let t = input.trim();
    if t.is_empty() {
        return None;
    }
    let v: usize = t.parse().ok()?;
    (v <= n).then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(all: bool) -> Plan {
        Plan {
            backend: Backend::Ninfer,
            repo: PathBuf::from("/repo"),
            op: Some("add_bias".into()),
            all,
            provider: "opencode-go".into(),
            model: "glm-5.3-flash".into(),
            e2e: Some("qwen3_8_27b_nvfp4".into()),
            iterations: 3,
            beam: 2,
            watch: true,
        }
    }

    #[test]
    fn presets_are_sane() {
        assert_eq!(preset("quick"), (2, 1));
        assert_eq!(preset("standard"), (3, 2));
        assert_eq!(preset("thorough"), (5, 3));
        assert_eq!(preset("bogus"), (3, 2));
    }

    #[test]
    fn run_argv_has_op_and_loop() {
        let a = plan(false).argv();
        assert_eq!(a[0], "run-ninfer");
        assert!(a.windows(2).any(|w| w == ["--op", "add_bias"]));
        assert!(a.windows(2).any(|w| w == ["--iterations", "3"]));
        assert!(a.windows(2).any(|w| w == ["--beam", "2"]));
        assert!(a.windows(2).any(|w| w == ["--e2e-weights", "qwen3_8_27b_nvfp4"]));
        assert_eq!(a.last().unwrap(), "--watch");
    }

    #[test]
    fn campaign_argv_uses_max_iterations() {
        let a = plan(true).argv();
        assert_eq!(a[0], "campaign");
        assert!(a.windows(2).any(|w| w == ["--mode", "ninfer"]));
        assert!(a.windows(2).any(|w| w == ["--max-iterations", "3"]));
        assert!(!a.iter().any(|x| x == "--beam"));
    }

    #[test]
    fn display_quotes_spaces() {
        let mut p = plan(false);
        p.repo = PathBuf::from("/my dir/repo");
        assert!(p.display().contains("\"/my dir/repo\""), "{}", p.display());
    }

    #[test]
    fn choice_parsing() {
        assert_eq!(parse_choice("2", 3), Some(2));
        assert_eq!(parse_choice("0", 3), Some(0));
        assert_eq!(parse_choice("4", 3), None);
        assert_eq!(parse_choice("", 3), None);
        assert_eq!(parse_choice("x", 3), None);
    }
}
