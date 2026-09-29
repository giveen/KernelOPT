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
// bench parsing + aggregation (Gate 4)
// --------------------------------------------------------------------------- //

/// One shape's measurement (median latency, label, bandwidth if reported).
#[derive(Debug, Clone, Default)]
pub struct BenchRow {
    pub median_us: f64,
    pub label: String,
    pub line: Option<String>,
    pub effective_gbs: Option<f64>,
    pub roofline_gbs: Option<f64>,
}

/// One bench invocation's rows.
#[derive(Debug, Clone, Default)]
pub struct ParsedBench {
    pub rows: Vec<BenchRow>,
    pub median_us: Option<f64>,
    pub best_median_us: Option<f64>,
}

/// Aggregated result across repeats (the Gate-4 representative + noise).
#[derive(Debug, Clone, Default)]
pub struct MergedBench {
    pub rows: Vec<BenchRow>,
    pub representative_us: Option<f64>,
    pub representative_label: Option<String>,
    pub representative_gbs: Option<f64>,
    pub representative_roofline_gbs: Option<f64>,
    pub noise_pct: Option<f64>,
}

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    Some(if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    })
}

fn median_opt(mut v: Vec<f64>) -> Option<f64> {
    v.retain(|x| x.is_finite());
    median(&v)
}

fn median_column(cols: &[String]) -> Option<usize> {
    if let Some(i) = cols.iter().position(|c| c.trim() == "median_us") {
        return Some(i);
    }
    cols.iter().position(|c| {
        let l = c.to_lowercase();
        l.contains("median") && l.contains("us")
    })
}

/// A human label for a CSV bench row (route/op/path + T when present).
fn row_label(fields: &[String], cols: &[String]) -> String {
    let mut parts = Vec::new();
    for k in ["route", "op", "path", "policy", "T"] {
        if let Some(i) = cols.iter().position(|c| c.trim() == k) {
            if let Some(v) = fields.get(i) {
                parts.push(format!("{k}={v}"));
            }
        }
    }
    if parts.is_empty() {
        "row".to_string()
    } else {
        parts.join(" ")
    }
}

fn finish(rows: Vec<BenchRow>) -> ParsedBench {
    let medians: Vec<f64> = rows.iter().map(|r| r.median_us).collect();
    let best = medians.iter().cloned().fold(f64::INFINITY, f64::min);
    ParsedBench {
        best_median_us: if best.is_finite() { Some(best) } else { None },
        median_us: median(&medians),
        rows,
    }
}

/// Parse an op-bench CSV (`--csv-out`): rows of floats + aggregate median.
pub fn parse_bench_csv(text: &str) -> ParsedBench {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return ParsedBench::default();
    };
    let cols = csv_split(header);
    let Some(mc) = median_column(&cols) else {
        return ParsedBench::default();
    };
    let gbs_col = cols.iter().position(|c| c.trim() == "effective_gbs");
    let roof_col = cols.iter().position(|c| c.trim() == "roofline_gbs");
    let mut rows = Vec::new();
    for line in lines {
        let f = csv_split(line);
        let Some(median_us) = f.get(mc).and_then(|v| v.trim().parse::<f64>().ok()) else {
            continue;
        };
        let num = |i: Option<usize>| i.and_then(|i| f.get(i)).and_then(|v| v.trim().parse::<f64>().ok());
        rows.push(BenchRow {
            median_us,
            label: row_label(&f, &cols),
            line: None,
            effective_gbs: num(gbs_col),
            roofline_gbs: num(roof_col),
        });
    }
    finish(rows)
}

/// Fallback for benches without `--csv-out` (e.g. add_bias): one line per shape.
pub fn parse_bench_stdout(text: &str) -> ParsedBench {
    let med_re = Regex::new(r"(?i)median\s*=\s*([0-9.]+)\s*us").expect("bench median regex");
    let gbs_re = Regex::new(r"(?i)([0-9.]+)\s*GB/s").expect("bench gbs regex");
    let roof_re = Regex::new(r"(?i)of\s+([0-9.]+)\s*GB/s\s+roofline").expect("bench roof regex");
    let mut rows = Vec::new();
    for line in text.lines() {
        let Some(c) = med_re.captures(line) else {
            continue;
        };
        let median_us = c[1].parse::<f64>().unwrap_or(0.0);
        let label = line.split("median=").next().unwrap_or("").trim().to_string();
        rows.push(BenchRow {
            median_us,
            label,
            line: Some(line.trim().to_string()),
            effective_gbs: gbs_re.captures(line).and_then(|c| c[1].parse::<f64>().ok()),
            roofline_gbs: roof_re.captures(line).and_then(|c| c[1].parse::<f64>().ok()),
        });
    }
    finish(rows)
}

/// Pick CSV when the bench wrote one, else fall back to the console output.
pub fn parse_bench_ninfer(stdout: &str, csv: Option<&str>) -> ParsedBench {
    match csv.filter(|c| !c.trim().is_empty()) {
        Some(c) => parse_bench_csv(c),
        None => parse_bench_stdout(stdout),
    }
}

/// llama.cpp `test-backend-ops perf` console lines: `<n> runs - <t> us/run`.
pub fn parse_bench_llama(text: &str) -> ParsedBench {
    let us_re = Regex::new(r"(\d+)\s+runs\s+-\s+([0-9.]+)\s+us/run").expect("llama us regex");
    let ansi = Regex::new(r"\x1b\[[0-9;]*m").expect("ansi regex");
    let gbs_re = Regex::new(r"([0-9.]+)\s*GB/s").expect("llama gbs regex");
    let mut rows = Vec::new();
    for raw in text.lines() {
        let line = ansi.replace_all(raw, "").to_string();
        let Some(c) = us_re.captures(&line) else {
            continue;
        };
        rows.push(BenchRow {
            median_us: c[2].parse::<f64>().unwrap_or(0.0),
            label: format!("{} runs", &c[1]),
            line: Some(line.trim().chars().take(220).collect()),
            effective_gbs: gbs_re.captures(&line).and_then(|c| c[1].parse::<f64>().ok()),
            roofline_gbs: None,
        });
    }
    finish(rows)
}

/// Aggregate repeated runs per shape. The representative is the row whose label
/// contains `shape_filter` (else the *slowest* shape — least launch-overhead-
/// dominated). `noise_pct` is the median relative spread across repeats.
pub fn merge_bench_runs(runs: &[ParsedBench], shape_filter: Option<&str>) -> MergedBench {
    let n = runs.iter().map(|r| r.rows.len()).min().unwrap_or(0);
    let mut rows: Vec<BenchRow> = Vec::new();
    let mut spreads: Vec<f64> = Vec::new();
    for i in 0..n {
        let vals: Vec<f64> = runs.iter().map(|r| r.rows[i].median_us).collect();
        let med = median(&vals).unwrap_or(0.0);
        if med > 0.0 {
            let (mx, mn) = vals
                .iter()
                .fold((f64::MIN, f64::MAX), |(mx, mn), v| (mx.max(*v), mn.min(*v)));
            spreads.push((mx - mn) / med);
        }
        rows.push(BenchRow {
            median_us: med,
            label: runs[0].rows[i].label.clone(),
            line: runs[0].rows[i].line.clone(),
            effective_gbs: median_opt(
                runs.iter().filter_map(|r| r.rows[i].effective_gbs).collect(),
            ),
            roofline_gbs: median_opt(
                runs.iter().filter_map(|r| r.rows[i].roofline_gbs).collect(),
            ),
        });
    }
    let mut candidates: Vec<usize> = (0..rows.len()).collect();
    if let Some(f) = shape_filter {
        let fl = f.to_lowercase();
        let matching: Vec<usize> = (0..rows.len())
            .filter(|&i| rows[i].label.to_lowercase().contains(&fl))
            .collect();
        if !matching.is_empty() {
            candidates = matching;
        }
    }
    let rep = candidates
        .into_iter()
        .max_by(|&a, &b| {
            rows[a]
                .median_us
                .partial_cmp(&rows[b].median_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    MergedBench {
        representative_us: rep.map(|i| rows[i].median_us),
        representative_label: rep.map(|i| rows[i].label.clone()),
        representative_gbs: rep.and_then(|i| rows[i].effective_gbs),
        representative_roofline_gbs: rep.and_then(|i| rows[i].roofline_gbs),
        noise_pct: median_opt(spreads),
        rows,
    }
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

    #[test]
    fn bench_stdout_parses_gbs_and_roofline() {
        let text = "\
add_bias [1152,4096 ]            median=    4.94 us  min=    4.91 us  p95=    4.96 us    3821.5 GB/s  (213.3% of 1792 GB/s roofline)
add_bias [4608,16384]            median=  181.08 us  min=  179.70 us  p95=  182.79 us    1667.7 GB/s  (93.1% of 1792 GB/s roofline)
";
        let p = parse_bench_stdout(text);
        assert_eq!(p.rows.len(), 2);
        assert_eq!(p.rows[0].label, "add_bias [1152,4096 ]");
        assert_eq!(p.rows[0].effective_gbs, Some(3821.5));
        assert_eq!(p.rows[0].roofline_gbs, Some(1792.0));
        assert_eq!(p.rows[1].median_us, 181.08);
    }

    #[test]
    fn bench_merge_uses_slowest_shape_and_noise() {
        let r1 = parse_bench_stdout("x median= 10.0 us\nx median= 100.0 us\n");
        let r2 = parse_bench_stdout("x median= 10.2 us\nx median= 101.0 us\n");
        let m = merge_bench_runs(&[r1, r2], None);
        assert_eq!(m.representative_us, Some(100.5));
        assert!(m.noise_pct.unwrap() >= 0.0);
        // A shape filter picks the matching row even when it is not the slowest.
        let r = parse_bench_csv("median_us,route\n10.0,\"a [1152,8]\"\n100.0,\"a [1152,4096]\"\n");
        let m2 = merge_bench_runs(&[r], Some("1152,8"));
        assert_eq!(m2.representative_us, Some(10.0));
        // No match -> fall back to the slowest shape.
        let r = parse_bench_csv("median_us,route\n10.0,\"a [1152,8]\"\n100.0,\"a [1152,4096]\"\n");
        assert_eq!(merge_bench_runs(&[r], Some("nope")).representative_us, Some(100.0));
    }

    #[test]
    fn bench_llama_parses_us_per_run() {
        let p = parse_bench_llama("  CUDA0 SOFT_MAX(1024): 5 runs - 12.34 us/run - 100.5 GB/s\n");
        assert_eq!(p.rows.len(), 1);
        assert_eq!(p.rows[0].median_us, 12.34);
        assert_eq!(p.rows[0].effective_gbs, Some(100.5));
    }
}
