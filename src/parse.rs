//! Parsers for external tool output (ncu, ctest, nvcc, git numstat, benches).
//!
//! These lived in the Python runner; moving them here puts the measurement
//! logic next to the verdicts (`performance_verdict`, the roofline guard, Gate
//! 5) and removes a subprocess hop per parse. The JSON shapes match the old
//! runner responses, so consumers are unchanged.

use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

// --------------------------------------------------------------------------- //
// ncu CSV
// --------------------------------------------------------------------------- //

/// SOL metric rows: (json key, ncu "Metric Name", expected unit).
const SOL_METRICS: &[(&str, &str, Option<&str>)] = &[
    ("sol_compute", "Compute (SM) Throughput", Some("%")),
    ("sol_memory", "Memory Throughput", Some("%")),
    ("duration", "Duration", None),
    ("registers", "Registers Per Thread", Some("register/thread")),
    ("occupancy", "Achieved Occupancy", Some("%")),
];

/// ncu prints progress lines before the CSV header; keep from the header on.
pub fn extract_ncu_csv(stdout: &str) -> String {
    for (i, line) in stdout.lines().enumerate() {
        if line.starts_with("\"ID\"") || line.starts_with("ID,") {
            return stdout.lines().skip(i).collect::<Vec<_>>().join("\n");
        }
    }
    stdout.to_string()
}

/// Split one CSV line, honoring double-quoted fields (ncu and the benches).
fn csv_split(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_q {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_q = false;
                }
            } else {
                cur.push(c);
            }
        } else {
            match c {
                '"' => in_q = true,
                ',' => {
                    out.push(cur.trim().to_string());
                    cur.clear();
                }
                _ => cur.push(c),
            }
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn unit_matches(expected: Option<&str>, unit: &str) -> bool {
    match expected {
        None => matches!(unit, "ns" | "us" | "µs" | "ms"),
        Some("%") => unit == "%",
        Some(e) => unit.contains(e) || e.contains(unit),
    }
}

fn duration_us(value: f64, unit: &str) -> f64 {
    match unit {
        "ns" => value / 1000.0,
        "us" | "µs" => value,
        "ms" => value * 1000.0,
        _ => value,
    }
}

/// ncu values come as `82.51`, `1,234.56`, `<`, … → a number (0.0 on failure).
pub fn to_float(raw: &str) -> f64 {
    raw.replace(',', "")
        .replace('%', "")
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
}

/// Parse ncu CSV into `{kernels:[…], rules:[…]}` (same shape as the runner).
pub fn parse_ncu_csv(csv_text: &str) -> Value {
    let mut lines = csv_text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return json!({"kernels": [], "rules": []});
    };
    let cols = csv_split(header);
    let col = |name: &str| cols.iter().position(|h| h == name);
    let (i_kernel, i_metric, i_value) = (
        col("Kernel Name").or_else(|| col("Kernel Name (correlation ID)")),
        col("Metric Name"),
        col("Metric Value"),
    );
    let (i_unit, i_section, i_rule) = (
        col("Metric Unit"),
        col("Section Name"),
        col("Rule Description").or_else(|| col("Rule Type")),
    );
    let (Some(ik), Some(im), Some(iv)) = (i_kernel, i_metric, i_value) else {
        return json!({"kernels": [], "rules": []});
    };

    let mut order: Vec<String> = Vec::new();
    let mut kernels: HashMap<String, Map<String, Value>> = HashMap::new();
    let mut units: HashMap<String, Map<String, Value>> = HashMap::new();
    let mut rules: Vec<Value> = Vec::new();

    for line in lines {
        let f = csv_split(line);
        let get = |i: Option<usize>| {
            i.and_then(|i| f.get(i))
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        let name = get(Some(ik));
        if name.is_empty() {
            continue;
        }
        let metric = get(Some(im));
        let unit = get(i_unit);
        let raw = get(Some(iv));
        if metric.is_empty() || raw.is_empty() {
            continue;
        }
        if let Some((key, expected)) = SOL_METRICS
            .iter()
            .find(|(_, m, _)| *m == metric)
            .map(|(k, _, e)| (*k, *e))
        {
            if !unit_matches(expected, &unit) {
                continue; // e.g. a byte/s row also named "Memory Throughput"
            }
            let mut value = to_float(&raw);
            if key == "duration" {
                value = duration_us(value, &unit);
            }
            let e = kernels.entry(name.clone()).or_insert_with(|| {
                order.push(name.clone());
                Map::new()
            });
            // First valid row wins so a later analysis row can't clobber the SOL %.
            if !e.contains_key(key) {
                e.insert(key.to_string(), json!(value));
                if !unit.is_empty() {
                    units
                        .entry(name.clone())
                        .or_default()
                        .insert(key.to_string(), json!(unit));
                }
            }
        } else if metric.starts_with("OPT")
            || metric.starts_with("Rule")
            || get(i_section).contains("Recommendation")
        {
            rules.push(json!({
                "kernel": name,
                "rule": metric,
                "estimated_speedup": to_float(&raw),
                "details": get(i_rule),
            }));
        }
    }

    let kernel_list: Vec<Value> = order
        .iter()
        .map(|name| {
            let e = &kernels[name];
            let g = |k: &str| e.get(k).cloned().unwrap_or(Value::Null);
            json!({
                "name": name,
                "sol_compute": g("sol_compute"),
                "sol_memory": g("sol_memory"),
                "duration_us": g("duration"),
                "registers": g("registers"),
                "achieved_occupancy": g("occupancy"),
                "units": Value::Object(units.get(name).cloned().unwrap_or_default()),
            })
        })
        .collect();
    rules.sort_by(|a, b| {
        let (av, bv) = (
            a["estimated_speedup"].as_f64().unwrap_or(0.0),
            b["estimated_speedup"].as_f64().unwrap_or(0.0),
        );
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    json!({"kernels": kernel_list, "rules": rules})
}

// --------------------------------------------------------------------------- //
// ctest (Gate 2)
// --------------------------------------------------------------------------- //

/// Per-test status from ctest output → `{cases, failing_cases}`.
pub fn parse_ctest(text: &str) -> Value {
    let re = Regex::new(
        r"Test\s+#\d+:\s*(?P<name>\S+)\s*\.*\s*(?P<status>Passed|Failed|\*\*\*Failed|\*\*\*Timeout|Skipped|Not Run)",
    )
    .expect("ctest regex");
    let mut cases: Vec<Value> = Vec::new();
    for c in re.captures_iter(text) {
        let status = &c["status"];
        let s = if status.contains("Failed") || status.contains("Timeout") {
            "failed".to_string()
        } else {
            status.to_lowercase()
        };
        cases.push(json!({"name": &c["name"], "status": s}));
    }
    let failing: Vec<Value> = cases
        .iter()
        .filter(|c| c["status"] == "failed")
        .cloned()
        .collect();
    json!({"cases": cases, "failing_cases": failing})
}

// --------------------------------------------------------------------------- //
// nvcc / gcc diagnostics (Gate 1)
// --------------------------------------------------------------------------- //

/// Structured, de-duplicated diagnostics from nvcc/gcc output.
pub fn parse_compiler_errors(text: &str) -> Vec<Value> {
    // nvcc: path(line): error: msg     gcc/clang: path:line:col: error: msg
    let re = Regex::new(
        r"(?m)^(?P<file>[^:\n()]+?)(?:\((?P<line_p>\d+)\)|:(?P<line_c>\d+)(?::(?P<col>\d+))?):\s*(?P<sev>fatal error|error|warning)\s*:\s*(?P<msg>.*)$",
    )
    .expect("diag regex");
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for c in re.captures_iter(text) {
        let line = c.name("line_p").or_else(|| c.name("line_c")).map(|m| m.as_str());
        let col = c.name("col").map(|m| m.as_str());
        let file = c.name("file").map(|m| m.as_str()).unwrap_or("");
        let msg = c.name("msg").map(|m| m.as_str().trim()).unwrap_or("");
        let key = (file.to_string(), line.map(str::to_string), col.map(str::to_string), msg.to_string());
        if !seen.insert(key) {
            continue;
        }
        let msg = if msg.chars().count() > 500 {
            format!("{}…", msg.chars().take(500).collect::<String>())
        } else {
            msg.to_string()
        };
        out.push(json!({
            "file": file,
            "line": line.and_then(|l| l.parse::<u64>().ok()),
            "column": col.and_then(|l| l.parse::<u64>().ok()),
            "severity": &c["sev"],
            "message": msg,
        }));
        if out.len() >= 40 {
            break;
        }
    }
    out
}

/// `git diff --numstat` → `{files_changed, file_count, insertions, deletions}`.
pub fn parse_numstat(text: &str) -> Value {
    let (mut files, mut ins, mut del) = (0u64, 0u64, 0u64);
    let mut paths: Vec<String> = Vec::new();
    for line in text.lines() {
        let mut it = line.split('\t');
        let (Some(a), Some(b), Some(p)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        files += 1;
        paths.push(p.trim().to_string());
        ins += a.trim().parse::<u64>().unwrap_or(0);
        del += b.trim().parse::<u64>().unwrap_or(0);
    }
    json!({
        "files_changed": paths,
        "file_count": files,
        "insertions": ins,
        "deletions": del,
    })
}

// --------------------------------------------------------------------------- //
// Runner-response views: compute the parsed fields in Rust from the raw output
// --------------------------------------------------------------------------- //

fn raw_output(resp: &Value) -> String {
    format!(
        "{}\n{}",
        resp["raw_stdout"].as_str().unwrap_or(""),
        resp["raw_stderr"].as_str().unwrap_or("")
    )
}

/// Gate 1 view: `{passed, exit_code, compiler_errors, raw_tail}` from a
/// `cuda_compile` response (which now carries the raw build output).
pub fn compile_view(resp: &Value) -> Value {
    let combined = raw_output(resp);
    let all = parse_compiler_errors(&combined);
    let errors: Vec<Value> = all
        .into_iter()
        .filter(|e| e["severity"] != "warning")
        .collect();
    let exit = resp["exit_code"].as_i64();
    let passed = exit.map(|c| c == 0).unwrap_or(resp["passed"] == json!(true));
    let tail: String = combined
        .chars()
        .rev()
        .take(4000)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    json!({"passed": passed, "exit_code": exit, "compiler_errors": errors, "raw_tail": tail})
}

/// Gate 2 view for llama.cpp: `test-backend-ops test` prints `N/M tests passed`.
pub fn parse_llama_verify(text: &str) -> Value {
    let pass_re = Regex::new(r"(\d+)\s*/\s*(\d+)\s+tests passed").expect("llama pass regex");
    let ansi = Regex::new(r"\x1b\[[0-9;]*m").expect("ansi regex");
    let (mut passed, mut total) = (None, None);
    if let Some(c) = pass_re.captures(text) {
        passed = c[1].parse::<u64>().ok();
        total = c[2].parse::<u64>().ok();
    }
    let mut failing = Vec::new();
    for line in text.lines() {
        let s = ansi.replace_all(line, "").trim().to_string();
        if s.contains("FAIL") || s.contains("compare failed") {
            failing.push(json!({"name": s.chars().take(220).collect::<String>()}));
        }
    }
    let ok_counts = matches!((passed, total), (Some(p), Some(t)) if p == t);
    json!({
        "tests_passed": passed,
        "tests_total": total,
        "failing_cases": failing,
        "ok_counts": ok_counts,
    })
}

/// Gate 2 view: `{passed, exit_code, cases, failing_cases}` from a verify
/// response. ctest for ninfer, `N/M tests passed` for llama.cpp.
pub fn verify_view(resp: &Value, backend: crate::backend::Backend) -> Value {
    let text = raw_output(resp);
    let exit = resp["exit_code"].as_i64();
    let ran_clean = exit.map(|c| c == 0).unwrap_or(true);
    match backend {
        crate::backend::Backend::Ninfer => {
            let parsed = parse_ctest(&text);
            let failing = parsed["failing_cases"].clone();
            let passed =
                ran_clean && failing.as_array().map(|a| a.is_empty()).unwrap_or(false);
            json!({
                "passed": passed,
                "exit_code": exit,
                "cases": parsed["cases"],
                "failing_cases": failing,
            })
        }
        crate::backend::Backend::Llamacpp => {
            let parsed = parse_llama_verify(&text);
            let failing = parsed["failing_cases"].clone();
            let passed = ran_clean
                && parsed["ok_counts"] == json!(true)
                && failing.as_array().map(|a| a.is_empty()).unwrap_or(false);
            json!({
                "passed": passed,
                "exit_code": exit,
                "tests_passed": parsed["tests_passed"],
                "tests_total": parsed["tests_total"],
                "failing_cases": failing,
            })
        }
    }
}

/// `cuda_diff` view: `{file_count, insertions, deletions}` from the raw numstat.
pub fn numstat_view(resp: &Value) -> Value {
    parse_numstat(resp["numstat_raw"].as_str().unwrap_or(""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncu_extracts_and_parses() {
        let stdout = "\
==PROF== Connected to process 123
==PROF== Profiling \"k\" - 0%
\"ID\",\"Kernel Name\",\"Section Name\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"
\"0\",\"void k<float>\",\"SpeedOfLight\",\"Memory Throughput\",\"%\",\"55.96\"
\"0\",\"void k<float>\",\"SpeedOfLight\",\"Compute (SM) Throughput\",\"%\",\"78.97\"
\"0\",\"void k<float>\",\"MemoryWorkloadAnalysis\",\"Memory Throughput\",\"byte/s\",\"123456\"
\"0\",\"void k<float>\",\"LaunchStats\",\"Registers Per Thread\",\"register/thread\",\"32\"
\"0\",\"void k<float>\",\"Occupancy\",\"Achieved Occupancy\",\"%\",\"11.08\"
\"0\",\"void k<float>\",\"SpeedOfLight\",\"Duration\",\"us\",\"818.88\"
\"0\",\"void k<float>\",\"SchedulerStats\",\"OPT_Reuse\",\"\",\"1.25\"
";
        let csv = extract_ncu_csv(stdout);
        assert!(csv.starts_with("\"ID\""));
        let ctx = parse_ncu_csv(&csv);
        let k = &ctx["kernels"][0];
        assert_eq!(k["name"], "void k<float>");
        assert_eq!(k["sol_memory"], 55.96); // byte/s row must NOT clobber the %
        assert_eq!(k["sol_compute"], 78.97);
        assert_eq!(k["duration_us"], 818.88);
        assert_eq!(k["registers"], 32.0);
        assert_eq!(k["achieved_occupancy"], 11.08);
        assert_eq!(ctx["rules"][0]["rule"], "OPT_Reuse");
    }

    #[test]
    fn ctest_statuses() {
        let text = "\
1/1 Test #63: ninfer_add_bias_test .............   Passed    1.70 sec
Test #2: ninfer_gelu_test ....***Failed    3.4 sec
Test #3: ninfer_cast_test .......   Skipped
";
        let p = parse_ctest(text);
        assert_eq!(p["cases"].as_array().unwrap().len(), 3);
        assert_eq!(p["failing_cases"][0]["name"], "ninfer_gelu_test");
    }

    #[test]
    fn nvcc_and_gcc_diagnostics() {
        let text = "\
src/ops/kernel/a.cuh(8): error: #include expects \"FILENAME\"
src/ops/launcher/a.cu:40:5: error: expected a \";\"
/home/x/a.cuh(12): warning: unused variable
src/ops/kernel/a.cuh(8): error: #include expects \"FILENAME\"
";
        let e = parse_compiler_errors(text);
        assert_eq!(e.len(), 3, "{e:?}");
        assert_eq!(e[0]["file"], "src/ops/kernel/a.cuh");
        assert_eq!(e[0]["line"], 8);
        assert_eq!(e[0]["severity"], "error");
        assert_eq!(e[1]["file"], "src/ops/launcher/a.cu");
        assert_eq!(e[1]["line"], 40);
        assert_eq!(e[1]["column"], 5);
        assert_eq!(e[2]["severity"], "warning");
    }

    #[test]
    fn numstat_totals() {
        let s = parse_numstat("3\t1\tsrc/a.cu\n-\t-\tbin/x\n2\t0\tsrc/b.cu\n");
        assert_eq!(s["file_count"], 3);
        assert_eq!(s["insertions"], 5);
        assert_eq!(s["deletions"], 1);
    }
}
