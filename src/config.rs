//! Configuration: provider presets + paper hyperparameters (all overridable).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Known OpenAI-compatible provider presets.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderPreset {
    pub name: String,
    pub base_url: String,
    pub api_key_env: Option<String>,
    /// OpenCode Go asks clients to identify themselves and send a session header.
    pub user_agent: Option<String>,
    pub session_header: Option<String>,
    pub needs_key: bool,
}

pub fn provider_preset(name: &str) -> Option<ProviderPreset> {
    let (name, base_url, api_key_env): (String, String, Option<&str>) = match name {
        "opencode-go" => (
            "opencode-go".into(),
            "https://opencode.ai/zen/go/v1".into(),
            Some("OPENCODE_API_KEY"),
        ),
        "openai" => (
            "openai".into(),
            "https://api.openai.com/v1".into(),
            Some("OPENAI_API_KEY"),
        ),
        "openrouter" => (
            "openrouter".into(),
            "https://openrouter.ai/api/v1".into(),
            Some("OPENROUTER_API_KEY"),
        ),
        "ollama" => (
            "ollama".into(),
            "http://localhost:11434/v1".into(),
            None,
        ),
        "vllm" => ("vllm".into(), "http://localhost:8000/v1".into(), None),
        "lmstudio" => (
            "lmstudio".into(),
            "http://localhost:1234/v1".into(),
            None,
        ),
        _ => return None,
    };
    Some(ProviderPreset {
        user_agent: Some(format!("kernelopt/{}", env!("CARGO_PKG_VERSION"))),
        session_header: if name == "opencode-go" {
            Some("x-opencode-session".into())
        } else {
            None
        },
        name,
        base_url,
        api_key_env: api_key_env.map(|s| s.to_string()),
        needs_key: api_key_env.is_some(),
    })
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hyper {
    /// Beam-search iterations (paper T=5).
    #[serde(default = "d_t")]
    pub t_iterations: u32,
    /// Plans per iteration (paper N=4).
    #[serde(default = "d_n")]
    pub n_plans: u32,
    /// Executor retries (paper K=4).
    #[serde(default = "d_k")]
    pub k_retries: u32,
    /// Beam width (paper B=4).
    #[serde(default = "d_b")]
    pub b_beam: u32,
    /// Performance gate noise margin (paper γ=1.03).
    #[serde(default = "d_gamma")]
    pub gamma: f64,
    /// Experience memory capacity (paper Q=8).
    #[serde(default = "d_q")]
    pub q_memory: usize,
    /// Store experience if speedup >= s_plus (1.05)…
    #[serde(default = "d_s_plus")]
    pub s_plus: f64,
    /// …or regression >= s_minus (1.20).
    #[serde(default = "d_s_minus")]
    pub s_minus: f64,
    /// UCB exploration constant.
    #[serde(default = "d_ucb_c")]
    pub ucb_c: f64,
    /// LLM per-call timeout seconds (paper: 900).
    #[serde(default = "d_llm_timeout")]
    pub llm_timeout_s: u64,
    /// Search-time allclose tolerance (relaxed to keep TF32 candidates).
    #[serde(default = "d_search_tol")]
    pub search_tol: f64,
    /// Final E2E tolerance (KernelBench standard).
    #[serde(default = "d_final_tol")]
    pub final_tol: f64,
}

fn d_t() -> u32 {
    5
}
fn d_n() -> u32 {
    4
}
fn d_k() -> u32 {
    4
}
fn d_b() -> u32 {
    4
}
fn d_gamma() -> f64 {
    1.03
}
fn d_q() -> usize {
    8
}
fn d_s_plus() -> f64 {
    1.05
}
fn d_s_minus() -> f64 {
    1.20
}
fn d_ucb_c() -> f64 {
    1.4
}
fn d_llm_timeout() -> u64 {
    900
}
fn d_search_tol() -> f64 {
    1e-3
}
fn d_final_tol() -> f64 {
    1e-4
}

impl Default for Hyper {
    fn default() -> Self {
        toml::from_str("").unwrap()
    }
}

/// Profiling tier selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ProfilerMode {
    /// Graphsignal sidecar (attribution) + NCU deep context.
    Both,
    Ncu,
    Graphsignal,
    None,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub provider: String,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub api_key: Option<String>,
    pub hyper: Hyper,
    pub profiler: ProfilerMode,
    pub ncu_set: String,
    pub inputs_fn: String,
    /// OpenAI reasoning_effort sent with every completion ("low" cuts
    /// thinking-token burn on reasoning models; None = provider default).
    pub reasoning_effort: Option<String>,
    pub runs_dir: PathBuf,
    pub runner_dir: PathBuf,
    pub prompts_dir: PathBuf,
}

impl Config {
    pub fn load(
        provider: String,
        model: String,
        base_url: Option<String>,
        api_key: Option<String>,
        hyper: Hyper,
        profiler: ProfilerMode,
        ncu_set: String,
        inputs_fn: String,
        reasoning_effort: Option<String>,
    ) -> Result<Self> {
        // Optional config.toml at project root supplies defaults.
        let file: FileConfig = {
            let path = PathBuf::from("config.toml");
            if path.exists() {
                toml::from_str(&std::fs::read_to_string(&path).context("reading config.toml")?)
                    .context("parsing config.toml")?
            } else {
                FileConfig::default()
            }
        };

        let preset = provider_preset(&provider);
        let api_key_env = api_key
            .is_none()
            .then(|| {
                file.api_key_env
                    .clone()
                    .or_else(|| preset.as_ref().and_then(|p| p.api_key_env.clone()))
            })
            .flatten();

        let api_key = api_key.or_else(|| {
            api_key_env
                .as_ref()
                .and_then(|env| std::env::var(env).ok())
                .filter(|s| !s.is_empty())
        });

        let base_url = if provider == "mock" {
            // MockClient never touches the network.
            "mock://local".to_string()
        } else {
            base_url
                .or(file.base_url)
                .or_else(|| preset.as_ref().map(|p| p.base_url.clone()))
                .with_context(|| format!("no base URL for provider {provider:?} (use --base-url)"))?
        };

        if preset.as_ref().map(|p| p.needs_key).unwrap_or(false) && api_key.is_none() {
            let env = api_key_env.clone().unwrap_or_else(|| "API_KEY".into());
            anyhow::bail!(
                "provider {provider:?} needs an API key: set {env} or pass --api-key"
            );
        }

        let reasoning_effort = resolve_reasoning_effort(reasoning_effort, file.reasoning_effort)?;

        let root = PathBuf::from(".");
        Ok(Config {
            provider,
            model,
            base_url: Some(base_url),
            api_key_env,
            api_key,
            hyper,
            profiler,
            ncu_set,
            inputs_fn,
            reasoning_effort,
            runs_dir: root.join(".kernelopt/runs"),
            runner_dir: root.join("runner"),
            prompts_dir: root.join("prompts"),
        })
    }
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    base_url: Option<String>,
    api_key_env: Option<String>,
    /// Default OpenAI `reasoning_effort` (overridden by `--reasoning-effort`).
    reasoning_effort: Option<String>,
}

/// Resolve the OpenAI `reasoning_effort` sent with every completion.
///
/// Precedence: CLI flag → `config.toml` → `"low"`. `none`/`off`/empty disable
/// the parameter (provider default). Valid levels: `minimal|low|medium|high`.
pub fn resolve_reasoning_effort(
    cli: Option<String>,
    file: Option<String>,
) -> Result<Option<String>> {
    let raw = cli.or(file).unwrap_or_else(|| "low".to_string());
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "off" => Ok(None),
        level @ ("minimal" | "low" | "medium" | "high") => Ok(Some(level.to_string())),
        other => anyhow::bail!(
            "invalid reasoning effort {other:?}: expected none|minimal|low|medium|high"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_precedence_and_validation() {
        // Default when nothing is set.
        assert_eq!(resolve_reasoning_effort(None, None).unwrap(), Some("low".into()));
        // config.toml fills in when the CLI is silent.
        assert_eq!(
            resolve_reasoning_effort(None, Some("high".into())).unwrap(),
            Some("high".into())
        );
        // CLI wins.
        assert_eq!(
            resolve_reasoning_effort(Some("minimal".into()), Some("high".into())).unwrap(),
            Some("minimal".into())
        );
        // Disable.
        assert_eq!(resolve_reasoning_effort(Some("none".into()), None).unwrap(), None);
        assert_eq!(resolve_reasoning_effort(Some("off".into()), None).unwrap(), None);
        // Case-insensitive.
        assert_eq!(
            resolve_reasoning_effort(Some("MEDIUM".into()), None).unwrap(),
            Some("medium".into())
        );
        assert!(resolve_reasoning_effort(Some("bogus".into()), None).is_err());
    }
}
