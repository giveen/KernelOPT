//! OAuth (dynamic client registration + authorization code + PKCE) and a
//! cached token for the docs MCP servers.
//!
//! `kernelopt docs --login` runs the flow once and caches the token under
//! `.kernelopt/docs_token.json`; later lookups refresh it automatically.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_REGISTER: &str = "https://api.copilot.nsight.ngc.nvidia.com/register";
const DEFAULT_TOKEN: &str = "https://api.copilot.nsight.ngc.nvidia.com/token";
const DEFAULT_AUTHZ: &str = "https://api.copilot.nsight.ngc.nvidia.com/authorize";

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct StoredToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    /// Unix seconds.
    pub expires_at: i64,
    pub client_id: String,
    pub client_secret: String,
    pub token_endpoint: String,
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn token_path() -> Option<PathBuf> {
    let dir = std::env::current_dir().ok()?.join(".kernelopt");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("docs_token.json"))
}

pub fn load() -> Option<StoredToken> {
    let text = std::fs::read_to_string(token_path()?).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn save(t: &StoredToken) -> Result<()> {
    let path = token_path().context("no token path")?;
    std::fs::write(path, serde_json::to_string_pretty(t)?)?;
    Ok(())
}

/// A valid access token from the cache, refreshing if it is (nearly) expired.
pub fn access_token() -> Option<String> {
    let mut t = load()?;
    if t.access_token.is_empty() {
        return None;
    }
    if t.expires_at - 60 <= now() {
        refresh(&mut t)?;
    }
    Some(t.access_token)
}

pub fn refresh(t: &mut StoredToken) -> Option<()> {
    let refresh_token = t.refresh_token.clone()?;
    let endpoint = if t.token_endpoint.is_empty() {
        DEFAULT_TOKEN.to_string()
    } else {
        t.token_endpoint.clone()
    };
    let form = form_encode(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", &refresh_token),
        ("client_id", &t.client_id),
        ("client_secret", &t.client_secret),
    ]);
    let v = post_form(&endpoint, &form).ok()?;
    update_from(t, &v);
    let _ = save(t);
    (!t.access_token.is_empty()).then_some(())
}

/// Interactive login (DCR → authorize → token), caching the result.
pub fn login() -> Result<StoredToken> {
    let http = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").context("binding callback port")?;
    let port = listener.local_addr()?.port();
    let redirect = format!("http://127.0.0.1:{port}/callback");

    // 1) Dynamic client registration.
    let reg: serde_json::Value = http
        .post(DEFAULT_REGISTER)
        .json(&serde_json::json!({
            "client_name": "kernelopt",
            "redirect_uris": [redirect],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post"
        }))
        .send()
        .context("register request")?
        .json()
        .context("parsing registration response")?;
    let client_id = reg["client_id"].as_str().context("no client_id in registration")?.to_string();
    let client_secret = reg["client_secret"].as_str().unwrap_or("").to_string();

    // 2) PKCE.
    let verifier = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let challenge = pkce_challenge(&verifier);
    let state = uuid::Uuid::new_v4().simple().to_string();
    let url = format!(
        "{DEFAULT_AUTHZ}?{}",
        form_encode(&[
            ("response_type", "code"),
            ("client_id", &client_id),
            ("redirect_uri", &redirect),
            ("scope", "openid"),
            ("state", &state),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
        ])
    );
    println!("Open this URL in a browser and sign in, then return here:\n\n{url}\n");

    // 3) Wait for the redirect.
    let (code, err) = wait_for_code(listener, Duration::from_secs(900))?;
    if let Some(e) = err {
        anyhow::bail!("authorization error: {e}");
    }
    let code = code.context("no `code` in the callback")?;

    // 4) Exchange the code.
    let form = form_encode(&[
        ("grant_type", "authorization_code"),
        ("code", &code),
        ("redirect_uri", &redirect),
        ("client_id", &client_id),
        ("client_secret", &client_secret),
        ("code_verifier", &verifier),
    ]);
    let v = post_form(DEFAULT_TOKEN, &form).context("token exchange")?;
    let mut t = StoredToken {
        client_id,
        client_secret,
        token_endpoint: DEFAULT_TOKEN.to_string(),
        ..Default::default()
    };
    update_from(&mut t, &v);
    if t.access_token.is_empty() {
        anyhow::bail!("token exchange returned no access_token: {v}");
    }
    save(&t)?;
    Ok(t)
}

fn update_from(t: &mut StoredToken, v: &serde_json::Value) {
    if let Some(a) = v["access_token"].as_str() {
        t.access_token = a.to_string();
    }
    if let Some(r) = v["refresh_token"].as_str() {
        t.refresh_token = Some(r.to_string());
    }
    let expires = v["expires_in"].as_i64().unwrap_or(3600);
    t.expires_at = now() + expires;
}

fn post_form(url: &str, form: &str) -> Result<serde_json::Value> {
    let resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?
        .post(url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(form.to_string())
        .send()?;
    let text = resp.text()?;
    serde_json::from_str(&text).with_context(|| format!("token endpoint body: {text}"))
}

fn wait_for_code(
    listener: std::net::TcpListener,
    timeout: Duration,
) -> Result<(Option<String>, Option<String>)> {
    use std::io::{Read, Write};
    listener.set_nonblocking(true)?;
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((mut sock, _)) => {
                let mut buf = [0u8; 8192];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let target = req
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or("")
                    .to_string();
                let query = target.splitn(2, '?').nth(1).unwrap_or("");
                let params = parse_query(query);
                let body = "KernelOPT: login captured. You can close this tab.";
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
                return Ok((params.get("code").cloned(), params.get("error").cloned()));
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() > deadline {
                    anyhow::bail!("login timed out");
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn parse_query(q: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    for pair in q.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            out.insert(percent_decode(k), percent_decode(v));
        }
    }
    out
}

fn pkce_challenge(verifier: &str) -> String {
    use base64::Engine;
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(h.finalize())
}

fn form_encode(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_matches_rfc_example() {
        // RFC 7636 Appendix B.
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn percent_roundtrip() {
        let s = "http://127.0.0.1:8123/callback?x=1";
        assert_eq!(percent_decode(&percent_encode(s)), s);
    }
}
