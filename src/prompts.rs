//! Prompt template rendering: `{{var}}` substitution over editable assets in
//! `prompts/`. Unknown variables render as empty strings; never hard-code
//! agent prompts in Rust — they live in version-controlled markdown.

/// Render `{{var}}` placeholders from a JSON object of strings.
pub fn render(template: &str, vars: &serde_json::Value) -> String {
    let mut out = template.to_string();
    if let Some(map) = vars.as_object() {
        for (key, value) in map {
            let placeholder = format!("{{{{{key}}}}}");
            let rendered = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            out = out.replace(&placeholder, &rendered);
        }
    }
    // Scrub any unresolved placeholders (prompts tolerate missing context).
    while let Some(start) = out.find("{{") {
        let Some(end_rel) = out[start..].find("}}") else { break };
        let end = start + end_rel + 2;
        out.replace_range(start..end, "");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn substitutes_and_scrubs() {
        let t = "Hello {{name}}! {{missing}} tail {{#if x}}block{{/if}}";
        let s = render(t, &json!({"name": "world", "#if x": "", "/if": ""}));
        assert!(s.contains("Hello world!"));
        assert!(!s.contains("{{"));
    }

    #[test]
    fn plain_text_untouched() {
        let t = "no placeholders here";
        assert_eq!(render(t, &json!({})), t);
    }
}
