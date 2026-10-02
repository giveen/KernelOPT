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
    /// Prompt tokens served from the provider's prefix cache (0 when the
    /// endpoint doesn't report it). Defaulted for older journals.
    #[serde(default)]
    pub cached_tokens: u64,
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
    #[serde(default, deserialize_with = "content_as_text")]
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

/// `content` is a string per the OpenAI spec, but some servers send an array of
/// content parts (`[{"type":"text","text":"…"}]`) — accept both.
fn content_as_text<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match v {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(s),
        Some(serde_json::Value::Array(parts)) => Some(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
        ),
        Some(other) => Some(other.to_string()),
    })
}

#[derive(Deserialize)]
struct WireToolCallOut {
    function: WireToolFnOut,
}

#[derive(Deserialize)]
struct WireToolFnOut {
    name: String,
    /// `arguments` is a JSON *string* per the OpenAI spec, but some compatible
    /// servers send the object directly — accept both.
    #[serde(default, deserialize_with = "string_or_json")]
    arguments: serde_json::Value,
}

/// Accept `arguments` as either a JSON string (parse it) or an object (use it).
fn string_or_json<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(deserializer)?;
    Ok(match v {
        serde_json::Value::String(s) => {
            serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
        }
        other => other,
    })
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    /// OpenAI-compatible: `prompt_tokens_details.cached_tokens` (flat on some
    /// endpoints); Anthropic: `cache_read_input_tokens`.
    #[serde(default)]
    prompt_tokens_details: Option<WirePromptDetails>,
    #[serde(default)]
    cached_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct WirePromptDetails {
    #[serde(default)]
    cached_tokens: u64,
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

/// How `tool_choice` is sent while self-healing provider divergences.
#[derive(Clone, Copy, PartialEq)]
enum Choice<'a> {
    Given(Option<&'a str>),
    Auto,
    Omit,
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
        self.complete_with(
            system,
            user,
            tools,
            session_id,
            Choice::Given(tool_choice),
            false,
            false,
        )
    }
}

impl OpenAiCompatClient {
    /// `complete` plus progressive fallbacks for OpenAI-compatible servers that
    /// diverge on `reasoning_effort`, `tool_choice`, or tool schemas. Each
    /// fallback is logged once and never loops (the drop flags are sticky).
    fn complete_with(
        &self,
        system: &str,
        user: &str,
        tools: &[ToolDef],
        session_id: &str,
        choice: Choice<'_>,
        drop_reasoning: bool,
        drop_tools: bool,
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

        let wire_tools = if drop_tools || tools.is_empty() {
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
        let choice_str = match choice {
            Choice::Given(c) => c,
            Choice::Auto => Some("auto"),
            Choice::Omit => None,
        };
        let wire_tool_choice = choice_str.map(|tc| {
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
            reasoning_effort: if drop_reasoning {
                None
            } else {
                self.reasoning_effort.clone()
            },
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
            let code = status.as_u16();
            let lower = text.to_ascii_lowercase();
            let retryable = code == 400 || code == 422;
            // Some models/endpoints reject reasoning_effort.
            if retryable
                && !drop_reasoning
                && self.reasoning_effort.is_some()
                && lower.contains("reasoning")
            {
                eprintln!("[llm] {status}: endpoint rejected reasoning_effort; retrying without it");
                return self.complete_with(system, user, tools, session_id, choice, true, drop_tools);
            }
            // Some servers reject a *forced* tool_choice (named or "auto").
            if retryable
                && !drop_tools
                && !tools.is_empty()
                && choice != Choice::Omit
                && lower.contains("tool_choice")
            {
                let next = if matches!(choice, Choice::Given(Some(_))) {
                    Choice::Auto
                } else {
                    Choice::Omit
                };
                eprintln!(
                    "[llm] {status}: endpoint rejected tool_choice; retrying with {}",
                    if next == Choice::Auto { "\"auto\"" } else { "no tool_choice" }
                );
                return self.complete_with(system, user, tools, session_id, next, drop_reasoning, drop_tools);
            }
            // Some servers do not support tool schemas at all.
            if retryable
                && !drop_tools
                && !tools.is_empty()
                && (lower.contains("tool") || lower.contains("function"))
                && (lower.contains("support")
                    || lower.contains("unrecognized")
                    || lower.contains("unknown")
                    || lower.contains("invalid")
                    || lower.contains("not allowed")
                    || lower.contains("extra"))
            {
                eprintln!("[llm] {status}: endpoint rejected tool schemas; retrying without tools");
                return self.complete_with(system, user, tools, session_id, Choice::Omit, drop_reasoning, true);
            }
            anyhow::bail!("LLM API error {status}: {}", truncate(&text, 2000));
        }
        let parsed: ChatResponse =
            serde_json::from_str(&text).with_context(|| format!("parsing response: {}", truncate(&text, 500)))?;

        let msg = &parsed.choices.first().context("empty choices")?.message;
        Ok(Completion {
            content: msg.content.clone(),
            tool_calls: msg
                .tool_calls
                .iter()
                .map(|tc| ToolCall {
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                })
                .collect(),
            usage: parsed.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                cached_tokens: u
                    .prompt_tokens_details
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0)
                    .max(u.cached_tokens.unwrap_or(0))
                    .max(u.cache_read_input_tokens.unwrap_or(0)),
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

    // ---- provider-compat parsing (the shapes real servers actually send) ---- //

    #[test]
    fn parses_object_tool_arguments_and_array_content() {
        // Some OpenAI-compatible servers send `arguments` as an object and
        // `content` as an array of parts.
        let body = r#"{
          "choices": [{"message": {
            "content": [{"type":"text","text":"hello "},{"type":"text","text":"world"}],
            "tool_calls": [{"type":"function","function":{"name":"submit_plan","arguments":{"plan":{"change":"x"}}}}]
          }}],
          "usage": {"prompt_tokens": 3, "completion_tokens": 4}
        }"#;
        let parsed: ChatResponse = serde_json::from_str(body).unwrap();
        let msg = &parsed.choices[0].message;
        assert_eq!(msg.content.as_deref(), Some("hello world"));
        assert_eq!(msg.tool_calls[0].function.arguments["plan"]["change"], "x");
        assert_eq!(parsed.usage.as_ref().unwrap().completion_tokens, 4);
    }

    #[test]
    fn parses_string_tool_arguments() {
        let body = r#"{"choices":[{"message":{"content":"ok","tool_calls":[
          {"function":{"name":"submit_kernel","arguments":"{\"kernel_source\":\"int main(){}\"}"}}]}}]}"#;
        let parsed: ChatResponse = serde_json::from_str(body).unwrap();
        assert_eq!(
            parsed.choices[0].message.tool_calls[0].function.arguments["kernel_source"],
            "int main(){}"
        );
    }

    #[test]
    fn tolerates_null_tool_calls() {
        let body = r#"{"choices":[{"message":{"content":"hi","tool_calls":null}}]}"#;
        let parsed: ChatResponse = serde_json::from_str(body).unwrap();
        assert!(parsed.choices[0].message.tool_calls.is_empty());
        assert_eq!(parsed.choices[0].message.content.as_deref(), Some("hi"));
    }

    #[test]
    fn cache_usage_parses_both_shapes() {
        let openai: ChatResponse = serde_json::from_str(concat!(
            r#"{"choices":[{"message":{"content":null,"tool_calls":[]}}],"#,
            r#""usage":{"prompt_tokens":100,"completion_tokens":5,"#,
            r#""prompt_tokens_details":{"cached_tokens":70}}}"#
        ))
        .unwrap();
        let u = openai.usage.unwrap();
        let cached = u
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0)
            .max(u.cached_tokens.unwrap_or(0))
            .max(u.cache_read_input_tokens.unwrap_or(0));
        assert_eq!(cached, 70);

        let anthropic: ChatResponse = serde_json::from_str(concat!(
            r#"{"choices":[{"message":{"content":null,"tool_calls":[]}}],"#,
            r#""usage":{"prompt_tokens":100,"completion_tokens":5,"#,
            r#""cache_read_input_tokens":42}}"#
        ))
        .unwrap();
        let u = anthropic.usage.unwrap();
        let cached = u
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0)
            .max(u.cached_tokens.unwrap_or(0))
            .max(u.cache_read_input_tokens.unwrap_or(0));
        assert_eq!(cached, 42);
    }

    // ---- self-healing against an OpenAI-compatible server that diverges ---- //

    fn read_request_body(sock: &mut std::net::TcpStream) -> String {
        use std::io::Read;
        let mut data = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = sock.read(&mut buf).unwrap_or(0);
            if n == 0 {
                break;
            }
            data.extend_from_slice(&buf[..n]);
            if let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&data[..pos]).to_lowercase();
                let clen = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if data.len() >= pos + 4 + clen {
                    break;
                }
            }
        }
        String::from_utf8_lossy(&data)
            .split("\r\n\r\n")
            .nth(1)
            .unwrap_or("")
            .to_string()
    }

    fn mock_server(
        replies: Vec<(u16, &'static str)>,
    ) -> (String, std::thread::JoinHandle<Vec<String>>) {
        use std::io::Write;
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for (code, payload) in replies {
                let (mut sock, _) = listener.accept().unwrap();
                bodies.push(read_request_body(&mut sock));
                let reason = if code == 200 { "OK" } else { "Bad Request" };
                let resp = format!(
                    "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = sock.write_all(resp.as_bytes());
            }
            bodies
        });
        (format!("http://{addr}/v1"), handle)
    }

    fn test_client(base: String) -> OpenAiCompatClient {
        OpenAiCompatClient::new(base, "m".into(), None, "ua".into(), None, Duration::from_secs(5))
    }

    #[test]
    fn self_heals_reasoning_effort_rejection() {
        let ok = r#"{"choices":[{"message":{"content":"ok","tool_calls":[]}}]}"#;
        let (base, server) = mock_server(vec![
            (400, r#"{"error":{"message":"Unrecognized request argument supplied: reasoning_effort"}}"#),
            (200, ok),
        ]);
        let client = test_client(base).with_reasoning_effort(Some("low".into()));
        let out = client.complete("s", "u", &[], "sess", None).unwrap();
        assert_eq!(out.content.as_deref(), Some("ok"));
        let bodies = server.join().unwrap();
        assert!(bodies[0].contains("reasoning_effort"), "first body: {}", bodies[0]);
        assert!(!bodies[1].contains("reasoning_effort"), "retry body: {}", bodies[1]);
    }

    #[test]
    fn self_heals_tool_choice_rejection() {
        let ok = r#"{"choices":[{"message":{"content":"ok","tool_calls":[]}}]}"#;
        let (base, server) = mock_server(vec![
            (400, r#"{"error":{"message":"tool_choice is not supported"}}"#),
            (200, ok),
        ]);
        let client = test_client(base);
        let tools = vec![ToolDef {
            name: "submit_plan".into(),
            description: "d".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let out = client
            .complete("s", "u", &tools, "sess", Some("submit_plan"))
            .unwrap();
        assert_eq!(out.content.as_deref(), Some("ok"));
        let bodies = server.join().unwrap();
        assert!(bodies[0].contains("\"name\":\"submit_plan\""), "first body: {}", bodies[0]);
        assert!(bodies[1].contains("\"tool_choice\":\"auto\""), "retry body: {}", bodies[1]);
    }
}
