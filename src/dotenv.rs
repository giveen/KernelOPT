//! Minimal `.env` loader (no external dependency).
//!
//! Precedence: the real process environment always wins — a variable already
//! present is never overwritten, so `NINFER_REPO=x kernelopt …` beats `.env`.
//!
//! File location: `$KERNELOPT_ENV_FILE` if set, otherwise `./.env`.
//! Syntax: `KEY=VALUE` per line; blank lines and `#` comments are ignored; an
//! optional `export ` prefix is allowed; surrounding single/double quotes are
//! stripped; `\n`/`\t`/`\"` escapes are honored inside double quotes.

use std::path::PathBuf;

/// Parse `KEY=VALUE` pairs from dotenv text.
pub fn parse(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty()
            || !key
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }
        out.push((key.to_string(), unquote(value.trim())));
    }
    out
}

fn unquote(value: &str) -> String {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let first = bytes[0];
        let last = bytes[value.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            let inner = &value[1..value.len() - 1];
            if first == b'"' {
                return inner
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\\"", "\"");
            }
            return inner.to_string();
        }
    }
    value.to_string()
}

/// Apply parsed pairs to the process environment, without overriding anything
/// already set. Returns how many variables were newly set.
pub fn apply(text: &str) -> usize {
    let mut set = 0;
    for (key, value) in parse(text) {
        if std::env::var_os(&key).is_none() {
            std::env::set_var(&key, value);
            set += 1;
        }
    }
    set
}

/// Load `$KERNELOPT_ENV_FILE` or `./.env` if present. Returns the path and the
/// number of variables newly set (None when no file exists).
pub fn load_default() -> Option<(PathBuf, usize)> {
    let path = std::env::var("KERNELOPT_ENV_FILE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".env"));
    if !path.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    Some((path, apply(&text)))
}

/// Expand a leading `~` to `$HOME` (dotenv files often use it for paths).
pub fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_forms() {
        let text = r#"
# comment
NINFER_REPO=/opt/ninfer
export LLAMACPP_REPO="~/llama.cpp"
KERNELOPT_MODEL=deepseek-v4-pro   # trailing comment kept as value
EMPTY=
QUOTED='a b c'
ESCAPED="line\nbreak"
INVALID KEY=value
"#;
        let pairs = parse(text);
        let map: std::collections::HashMap<_, _> = pairs.iter().cloned().collect();
        assert_eq!(map["NINFER_REPO"], "/opt/ninfer");
        assert_eq!(map["LLAMACPP_REPO"], "~/llama.cpp");
        assert_eq!(map["KERNELOPT_MODEL"], "deepseek-v4-pro   # trailing comment kept as value");
        assert_eq!(map["EMPTY"], "");
        assert_eq!(map["QUOTED"], "a b c");
        assert_eq!(map["ESCAPED"], "line\nbreak");
        assert!(!map.contains_key("INVALID"));
    }

    #[test]
    fn tilde_expansion() {
        std::env::set_var("HOME", "/home/tester");
        assert_eq!(expand_tilde("~/ninfer"), "/home/tester/ninfer");
        assert_eq!(expand_tilde("/abs/path"), "/abs/path");
        assert_eq!(expand_tilde("relative"), "relative");
    }
}
