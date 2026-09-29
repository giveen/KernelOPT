//! LLM client: OpenAI-compatible chat/completions with tool calling.
//!
//! Provider-agnostic by design: OpenCode Go, OpenAI, OpenRouter, Ollama,
//! vLLM, LM Studio, or any custom base URL. The mock provider (same trait)
//! enables full-pipeline tests with zero tokens.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Raw JSON arguments object.
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
}

/// Transport-agnostic LLM backend.
pub trait LlmClient: Send {
    /// `tool_choice`: None = omit; Some("auto"|"required") sent verbatim;
    /// any other string is treated as a named function to force.
    fn complete(
        &self,
        system: &str,
        user: &str,
        tools: &[ToolDef],
        session_id: &str,
        tool_choice: Option<&str>,
    ) -> Result<Completion>;
}

// ---------- OpenAI-compatible wire types ----------

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Clone)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: WireToolFn,
}

#[derive(Serialize, Clone)]
struct WireToolFn {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<WireTool>>,
    /// "auto"|"required" as strings; named-function form as an object —
    /// both per the OpenAI tool_choice schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    /// OpenAI-standard reasoning control ("low"|"medium"|"high"|"minimal").
    /// Verified: OpenCode Go accepts this; vendor variants (`reasoning`,
    /// `thinking`, `enable_thinking`) are rejected by its strict upstream.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    // Session identity goes in the x-opencode-session HEADER only — strict
    // OpenAI-compatible upstreams reject unknown body fields.
}

#[derive(Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: String,
    function: WireToolSpec,
}

#[derive(Serialize)]
struct WireToolSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default, deserialize_with = "null_as_default")]
    tool_calls: Vec<WireToolCallOut>,
}

/// Some providers send `"tool_calls": null` instead of omitting the field.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::Deserialize<'de> + Default,
{
    let v: Option<T> = serde::Deserialize::deserialize(deserializer)?;
    Ok(v.unwrap_or_default())
}

#[derive(Deserialize)]
struct WireToolCallOut {
    function: WireToolFnOut,
}

#[derive(Deserialize)]
struct WireToolFnOut {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// Real HTTP client for any OpenAI-compatible endpoint.
pub struct OpenAiCompatClient {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub user_agent: String,
    pub session_header: Option<String>,
    pub timeout: Duration,
    pub reasoning_effort: Option<String>,
    http: reqwest::blocking::Client,
}

impl OpenAiCompatClient {
    pub fn new(
        base_url: String,
        model: String,
        api_key: Option<String>,
        user_agent: String,
        session_header: Option<String>,
        timeout: Duration,
    ) -> Self {
        let http = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .user_agent(&user_agent)
            .build()
            .expect("reqwest client");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            api_key,
            user_agent,
            session_header,
            timeout,
            reasoning_effort: None,
            http,
        }
    }

    /// Set reasoning effort (e.g. "low" to cut thinking-token burn on
    /// reasoning models like GLM/DeepSeek/Qwen).
    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    fn clone_for_retry(&self) -> OpenAiCompatClient {
        OpenAiCompatClient {
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            user_agent: self.user_agent.clone(),
            session_header: self.session_header.clone(),
            timeout: self.timeout,
            reasoning_effort: None,
            http: self.http.clone(),
        }
    }

    fn authed(&self, url: &str) -> reqwest::blocking::RequestBuilder {
        let mut req = self.http.get(url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req
    }

    /// GET /models — provider reachability + model id check.
    pub fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let resp = self
            .authed(&url)
            .send()
            .with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("models endpoint error {status}: {}", truncate(&text, 800));
        }
        let parsed: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing /models response: {}", truncate(&text, 300)))?;
        let ids: Vec<String> = parsed
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        Ok(ids)
    }

    /// Minimal auth+completion probe (1 token).
    pub fn ping(&self) -> Result<String> {
        let c = self.complete(
            "You are a connectivity probe. Reply with the single word: pong",
            "ping",
            &[],
            "kernelopt-providers-test",
            None,
        )?;
        Ok(c.content.unwrap_or_else(|| "(empty)".into()))
    }
}

impl LlmClient for OpenAiCompatClient {
    fn complete(
        &self,
        system: &str,
        user: &str,
        tools: &[ToolDef],
        session_id: &str,
        tool_choice: Option<&str>,
    ) -> Result<Completion> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut messages = vec![ChatMessage {
            role: "system".into(),
            content: system.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }];
        messages.push(ChatMessage {
            role: "user".into(),
            content: user.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });

        let wire_tools = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| WireTool {
                        kind: "function".into(),
                        function: WireToolSpec {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: t.parameters.clone(),
                        },
                    })
                    .collect::<Vec<_>>(),
            )
        };

        // tool_choice: "auto"/"required" verbatim; any other string names a
        // function to force (OpenAI named-object form).
        let wire_tool_choice = tool_choice.map(|tc| {
            if tc == "auto" || tc == "required" || tc == "none" {
                serde_json::json!(tc)
            } else {
                serde_json::json!({"type": "function", "function": {"name": tc}})
            }
        });
        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            tools: wire_tools,
            tool_choice: wire_tool_choice,
            reasoning_effort: self.reasoning_effort.clone(),
        };

        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        if let Some(header) = &self.session_header {
            req = req.header(header, session_id);
        }

        let resp = req
            .json(&body)
            .send()
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            // Self-healing: some models/endpoints reject reasoning_effort —
            // retry once without it when the error names the parameter.
            if self.reasoning_effort.is_some()
                && status.as_u16() == 400
                && text.contains("reasoning_effort")
            {
                let retry = OpenAiCompatClient {
                    reasoning_effort: None,
                    ..self.clone_for_retry()
                };
                eprintln!("[llm] reasoning_effort rejected by endpoint; retrying without it");
                return retry.complete(system, user, tools, session_id, tool_choice);
            }
            anyhow::bail!("LLM API error {status}: {}", truncate(&text, 2000));
        }
        let parsed: ChatResponse =
            serde_json::from_str(&text).with_context(|| format!("parsing response: {}", truncate(&text, 500)))?;

        let choice = parsed.choices.first().context("empty choices")?;
        Ok(Completion {
            content: choice.message.content.clone(),
            tool_calls: choice
                .message
                .tool_calls
                .iter()
                .map(|tc| ToolCall {
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::Null),
                })
                .collect(),
            usage: parsed.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
            }),
        })
    }
}

fn truncate(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut end = n;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Retry decorator: exponential backoff on transport/5xx/429 errors.
/// The paper allows a 900 s per-call timeout; we add at most 3 retries.
pub struct RetryClient {
    pub inner: OpenAiCompatClient,
    pub max_retries: u32,
}

impl RetryClient {
    pub fn new(inner: OpenAiCompatClient, max_retries: u32) -> Self {
        Self { inner, max_retries }
    }
}

impl LlmClient for RetryClient {
    fn complete(
        &self,
        system: &str,
        user: &str,
        tools: &[ToolDef],
        session_id: &str,
        tool_choice: Option<&str>,
    ) -> Result<Completion> {
        let mut attempt = 0u32;
        loop {
            match self.inner.complete(system, user, tools, session_id, tool_choice) {
                Ok(c) => return Ok(c),
                Err(e) if attempt < self.max_retries => {
                    let backoff = Duration::from_secs(2u64.pow(attempt) * 2); // 2s, 4s, 8s
                    eprintln!(
                        "[llm] attempt {} failed ({e}); retrying in {:?}",
                        attempt + 1,
                        backoff
                    );
                    std::thread::sleep(backoff);
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

// ---------- Mock provider (zero-token tests) ----------

/// Scripted mock: pops canned completions; empty queue returns a harmless ack.
pub struct MockClient {
    pub scripted: std::sync::Mutex<Vec<Completion>>,
}

impl MockClient {
    pub fn new(scripted: Vec<Completion>) -> Self {
        Self {
            scripted: std::sync::Mutex::new(scripted),
        }
    }
}

impl LlmClient for MockClient {
    fn complete(
        &self,
        _system: &str,
        _user: &str,
        _tools: &[ToolDef],
        _session_id: &str,
        _tool_choice: Option<&str>,
    ) -> Result<Completion> {
        let mut q = self.scripted.lock().unwrap();
        if q.is_empty() {
            return Ok(Completion {
                content: Some("mock: nothing scripted".into()),
                tool_calls: vec![],
                usage: None,
            });
        }
        Ok(q.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_returns_scripted_then_default() {
        let client = MockClient::new(vec![Completion {
            content: Some("plan!".into()),
            tool_calls: vec![ToolCall {
                name: "submit_plan".into(),
                arguments: serde_json::json!({"plan": {"change": "bump XBLOCK"}}),
            }],
            usage: None,
        }]);
        let tools = vec![];
        let c = client.complete("s", "u", &tools, "sess", None).unwrap();
        assert_eq!(c.tool_calls.len(), 1);
        assert_eq!(c.tool_calls[0].name, "submit_plan");
        // queue exhausted -> default
        let c2 = client.complete("s", "u", &tools, "sess", None).unwrap();
        assert!(c2.tool_calls.is_empty());
    }
}
