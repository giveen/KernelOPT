//! ninfer-mode inventory: map a KernelOpt op token (e.g. `fp8_linear_add`,
//! `add_bias`) onto the real files, contract header, tests, and bench targets
//! inside a ninfer checkout.
//!
//! Design authority: `docs/ninfer-mode.md` §2–3, §9. This is the `discover`
//! stage — pure filesystem/CMake parsing, no GPU, no LLM. Everything it reports
//! is read-only; it never edits the ninfer tree.

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// One discovered op, ready to feed the ninfer pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct OpInventory {
    /// The requested token, verbatim.
    pub op: String,
    /// Op family directory under `src/ops/` (e.g. `linear_add`).
    pub family: String,
    /// Quant/branch variant when the token is `<variant>_<family>` (e.g. `fp8`).
    pub variant: Option<String>,
    pub repo: String,
    /// `src/ops/<family>` when it exists.
    pub family_dir: Option<String>,
    /// Editable CUDA kernel sources (`.cu`/`.cuh`); the Executor's write set.
    pub kernel_files: Vec<String>,
    /// Launch wrappers inside the family dir (read-only context).
    pub launcher_files: Vec<String>,
    /// Wrapper translation units (`src/ops/wrapper/…`, read-only context).
    pub wrapper_files: Vec<String>,
    /// Plan/dispatch/launch helpers inside the family dir (read-only context).
    pub context_files: Vec<String>,
    /// Op contract headers (`include/ninfer/ops/…`; semantic authority).
    pub contract_files: Vec<String>,
    /// `sources.cmake` files that register the kernel sources.
    pub sources_cmake: Vec<String>,
    /// Test executables + sources + every ctest name derived from them.
    pub tests: Vec<TestTarget>,
    /// Bench executables + sources (Gate 4 authority).
    pub benches: Vec<BenchTarget>,
    /// Non-fatal problems worth surfacing (missing contract, empty inventory…).
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TestTarget {
    pub target: String,
    pub sources: Vec<String>,
    /// ctest names: the target itself plus any `add_test(NAME … COMMAND target)`.
    pub test_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchTarget {
    pub target: String,
    pub sources: Vec<String>,
}

/// Discover one op inside `repo`. Returns a structured inventory; empty buckets
/// are meaningful (a missing contract or bench target is a warning, not an error).
pub fn discover(repo: &Path, op: &str) -> Result<OpInventory> {
    let repo = repo
        .canonicalize()
        .with_context(|| format!("ninfer repo not found: {}", repo.display()))?;
    let src_ops = repo.join("src/ops");
    if !src_ops.is_dir() {
        anyhow::bail!("{} does not look like a ninfer tree (no src/ops)", repo.display());
    }

    let mut warnings = Vec::new();
    let (family, variant) = resolve_family(&src_ops, op);

    let rel = |p: &Path| -> String {
        p.strip_prefix(&repo)
            .unwrap_or(p)
            .to_string_lossy()
            .replace('\\', "/")
            .to_string()
    };

    // ---- kernel / launcher / context files under the family dir -------------
    let family_dir = src_ops.join(&family);
    let family_dir_exists = family_dir.is_dir();
    let mut kernel_files = Vec::new();
    let mut launcher_files = Vec::new();
    let mut context_files = Vec::new();

    if family_dir_exists {
        let scan_root = match &variant {
            Some(v) if family_dir.join(v).is_dir() => family_dir.join(v),
            _ => family_dir.clone(),
        };
        let mut files = Vec::new();
        walk_files(&scan_root, &mut files)?;
        for f in files {
            let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let relpath = rel(&f);
            // If we had to scan the whole family (no exact variant dir), keep
            // only paths that actually mention the variant.
            if let Some(v) = &variant {
                if !family_dir.join(v).is_dir() && !relpath.contains(v.as_str()) {
                    continue;
                }
            }
            let ext = f.extension().and_then(|s| s.to_str()).unwrap_or("");
            let lower = name.to_ascii_lowercase();
            if ext == "cmake" {
                continue;
            }
            if lower.contains("plan") || lower.contains("dispatch") || lower.contains("launch") {
                context_files.push(relpath);
            } else if ext == "cu" || ext == "cuh" {
                kernel_files.push(relpath);
            } else {
                // headers/helpers that are not kernels or launchers: context.
                context_files.push(relpath);
            }
        }
    } else {
        // "basic" op: kernel lives in src/ops/kernel/<op>.cuh.
        for cand in [format!("src/ops/kernel/{op}.cuh"), format!("src/ops/kernel/{op}.cu")] {
            let p = repo.join(&cand);
            if p.is_file() {
                kernel_files.push(cand);
            }
        }
        if kernel_files.is_empty() {
            warnings.push(format!(
                "no family dir src/ops/{family} and no src/ops/kernel/{op}.cuh — op token may be wrong"
            ));
        }
    }
    kernel_files.sort();
    kernel_files.dedup();
    launcher_files.sort();
    launcher_files.dedup();
    context_files.sort();
    context_files.dedup();

    // ---- launcher / wrapper (basic ops) / wrapper (families) ----------------
    for cand in [
        format!("src/ops/launcher/{op}.cu"),
        format!("src/ops/launcher/{op}.h"),
    ] {
        let p = repo.join(&cand);
        if p.is_file() && !launcher_files.contains(&cand) {
            launcher_files.push(cand);
        }
    }
    launcher_files.sort();
    launcher_files.dedup();

    let mut wrapper_files = Vec::new();
    let wrapper_dir = repo.join("src/ops/wrapper");
    if wrapper_dir.is_dir() {
        let mut files = Vec::new();
        walk_files(&wrapper_dir, &mut files)?;
        for f in files {
            let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let matches = name.contains(op) || name.contains(&family);
            if matches {
                wrapper_files.push(rel(&f));
            }
        }
    }
    wrapper_files.sort();
    wrapper_files.dedup();

    // ---- contract headers ---------------------------------------------------
    let mut contract_files = Vec::new();
    for cand in [format!("include/ninfer/ops/{family}.h"), format!("include/ninfer/ops/{op}.h")] {
        let p = repo.join(&cand);
        if p.is_file() && !contract_files.contains(&cand) {
            contract_files.push(cand);
        }
    }
    if contract_files.is_empty() {
        warnings.push(format!(
            "no contract header at include/ninfer/ops/{family}.h — semantic authority unresolved"
        ));
    }

    // ---- sources.cmake registration ----------------------------------------
    let mut sources_cmake = Vec::new();
    let family_sources = format!("src/ops/{family}/sources.cmake");
    if repo.join(&family_sources).is_file() {
        sources_cmake.push(family_sources);
    }
    // basic_sources.cmake registers the flat launcher/wrapper ops; only include
    // it when it actually mentions this op.
    let basic = repo.join("src/ops/basic_sources.cmake");
    if basic.is_file() {
        let mentions = std::fs::read_to_string(&basic)
            .map(|t| t.contains(&format!("/{op}.")) || t.contains(&family))
            .unwrap_or(false);
        if mentions {
            sources_cmake.push("src/ops/basic_sources.cmake".to_string());
        }
    }

    // ---- tests + benches from CMake declarations ---------------------------
    let tests = parse_test_targets(&repo, op, &family, variant.as_deref())?;
    let benches = parse_bench_targets(&repo, op, &family, variant.as_deref())?;
    if tests.is_empty() {
        warnings.push(format!("no test target found for op {op:?}"));
    }
    if benches.is_empty() {
        warnings.push(format!("no bench target found for op {op:?}"));
    }

    Ok(OpInventory {
        op: op.to_string(),
        family,
        variant,
        repo: repo.to_string_lossy().to_string(),
        family_dir: family_dir_exists.then(|| rel(&family_dir)),
        kernel_files,
        launcher_files,
        wrapper_files,
        context_files,
        contract_files,
        sources_cmake,
        tests,
        benches,
        warnings,
    })
}

/// Pick the longest `src/ops/<dir>` that is a `_`-delimited affix of `op`.
/// `bf16_linear_add` → (`linear_add`, `bf16`); `add_bias` → (`add_bias`, None).
fn resolve_family(src_ops: &Path, op: &str) -> (String, Option<String>) {
    let mut families: Vec<String> = std::fs::read_dir(src_ops)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
        .collect();
    families.sort_by_key(|f| std::cmp::Reverse(f.len()));

    for f in &families {
        if op == f {
            return (f.clone(), None);
        }
        if let Some(v) = op.strip_suffix(f).and_then(|s| s.strip_suffix('_')) {
            return (f.clone(), Some(v.to_string()));
        }
        if let Some(v) = op.strip_prefix(f).and_then(|s| s.strip_prefix('_')) {
            return (f.clone(), Some(v.to_string()));
        }
    }
    (op.to_string(), None)
}

/// Recursively collect regular files under `root`, skipping build/VCS noise.
fn walk_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root)
        .with_context(|| format!("reading {}", root.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == ".git" || name == "build" || name == "__pycache__" {
            continue;
        }
        if p.is_dir() {
            walk_files(&p, out)?;
        } else {
            out.push(p);
        }
    }
    Ok(())
}

fn rel_to(repo: &Path, p: &Path) -> String {
    p.strip_prefix(repo)
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
        .to_string()
}

/// Collect every `.cmake` under `dir` (recursive), skipping build dirs.
fn cmake_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        walk_cmake(dir, &mut files)?;
    }
    Ok(files)
}

fn walk_cmake(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
    {
        let p = e.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == ".git" || name == "build" {
            continue;
        }
        if p.is_dir() {
            walk_cmake(&p, out)?;
        } else if p.extension().and_then(|s| s.to_str()) == Some("cmake") {
            out.push(p);
        }
    }
    Ok(())
}

/// A parsed `ninfer_add_op_*` declaration from a CMake file.
struct CmakeTarget {
    name: String,
    sources: Vec<String>,
}

/// Extract balanced argument text for each `fname(` call in `text`.
fn find_calls(text: &str, fname: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(pos) = text[from..].find(fname) {
        let abs = from + pos;
        // Word boundary: previous char must not be ident-ish.
        let boundary_ok = abs == 0
            || !matches!(bytes[abs - 1], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_');
        let after = abs + fname.len();
        let rest = &text[after..];
        let trimmed = rest.trim_start();
        if !boundary_ok || !trimmed.starts_with('(') {
            from = after;
            continue;
        }
        let open = after + (rest.len() - trimmed.len());
        let mut depth = 0i32;
        let mut i = open;
        let mut in_quote = false;
        while i < bytes.len() {
            let c = bytes[i];
            if c == b'"' {
                in_quote = !in_quote;
            } else if !in_quote {
                if c == b'(' {
                    depth += 1;
                } else if c == b')' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }
            i += 1;
        }
        if i < bytes.len() {
            out.push(text[open + 1..i].to_string());
        }
        from = i.saturating_add(1);
    }
    out
}

/// Extract double-quoted substrings, in order.
fn quoted_strings(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            out.push(s[start..j].to_string());
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// First whitespace-delimited token of `args`.
fn first_token(args: &str) -> String {
    args.trim_start()
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .find(|t| !t.is_empty())
        .unwrap_or("")
        .to_string()
}

/// All whitespace-delimited tokens, stripping surrounding quotes/parens.
fn all_tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|t| t.trim_matches(|c| c == '"' || c == '(' || c == ')'))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Span `(open_paren, close_paren)` of the first `name(` call at/after `from`.
fn call_span(text: &str, name: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut search = from;
    while let Some(pos) = text[search..].find(name) {
        let abs = search + pos;
        let boundary = abs == 0
            || !matches!(bytes[abs - 1], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_');
        let after = abs + name.len();
        let rest = &text[after..];
        let trimmed = rest.trim_start();
        if boundary && trimmed.starts_with('(') {
            let open = after + (rest.len() - trimmed.len());
            let mut depth = 0i32;
            let mut i = open;
            let mut in_quote = false;
            while i < bytes.len() {
                let c = bytes[i];
                if c == b'"' {
                    in_quote = !in_quote;
                } else if !in_quote {
                    if c == b'(' {
                        depth += 1;
                    } else if c == b')' {
                        depth -= 1;
                        if depth == 0 {
                            return Some((open, i));
                        }
                    }
                }
                i += 1;
            }
            return None;
        }
        search = after;
    }
    None
}

/// Items of `set(<list> item item …)`, in order.
fn set_list_items(text: &str, list: &str) -> Vec<String> {
    let mut from = 0usize;
    while let Some((open, close)) = call_span(text, "set", from) {
        let toks = all_tokens(&text[open + 1..close]);
        if toks.first().map(|s| s.as_str()) == Some(list) {
            return toks.into_iter().skip(1).collect();
        }
        from = close + 1;
    }
    Vec::new()
}

/// Expand ninfer's `set(<list> …)` + `foreach(<var> IN LISTS <list>) … endforeach()`
/// test declarations into literal calls so the call parser sees every op.
fn expand_foreach(text: &str) -> String {
    let mut out = text.to_string();
    for _ in 0..128 {
        let Some((fopen, fclose)) = call_span(&out, "foreach", 0) else {
            break;
        };
        let toks = all_tokens(&out[fopen + 1..fclose]);
        let var = toks.first().cloned().unwrap_or_default();
        let list = toks
            .iter()
            .position(|t| t == "LISTS")
            .and_then(|i| toks.get(i + 1))
            .cloned()
            .unwrap_or_default();
        let Some((eopen, eclose)) = call_span(&out, "endforeach", fclose) else {
            break;
        };
        let body = &out[fclose + 1..eopen];
        let items = set_list_items(&out, &list);
        let expanded: String = items
            .iter()
            .map(|item| body.replace(&format!("${{{var}}}"), item))
            .collect();
        out = format!("{}{}{}", &out[..fopen], expanded, &out[eclose + 1..]);
    }
    out
}

/// Parse `ninfer_add_op_test` / `ninfer_add_test` declarations from a CMake file.
fn parse_target_decls(path: &Path, fname: &str) -> Result<Vec<CmakeTarget>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let text = expand_foreach(&raw);
    let base = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for args in find_calls(&text, fname) {
        let name = first_token(&args);
        if name.is_empty() {
            continue;
        }
        let sources = match args.find("SOURCES") {
            Some(sp) => {
                let tail = &args[sp + "SOURCES".len()..];
                let tail = tail.split("LIBRARIES").next().unwrap_or(tail);
                quoted_strings(tail)
                    .into_iter()
                    .map(|q| {
                        let resolved = q.replace("${CMAKE_CURRENT_LIST_DIR}", &base.to_string_lossy());
                        resolved.replace('\\', "/")
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        out.push(CmakeTarget { name, sources });
    }
    Ok(out)
}

/// `add_test(NAME x COMMAND y)` → (name, executable).
fn parse_add_test(path: &Path) -> Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for args in find_calls(&text, "add_test") {
        let name = args
            .find("NAME")
            .map(|i| first_token(&args[i + 4..]))
            .unwrap_or_default();
        let command = args
            .find("COMMAND")
            .map(|i| first_token(&args[i + 7..]))
            .unwrap_or_default();
        if !name.is_empty() {
            out.push((name, command));
        }
    }
    Ok(out)
}

fn token_matches(name: &str, op: &str, family: &str, variant: Option<&str>) -> bool {
    if name.contains(op) {
        return true;
    }
    match variant {
        Some(v) => name.contains(family) && name.contains(v),
        None => false,
    }
}

/// Candidate exact names for an op's test/bench target, most specific first.
fn expected_names(op: &str, family: &str, variant: Option<&str>, suffix: &str) -> Vec<String> {
    let mut v = vec![format!("ninfer_{op}_{suffix}"), format!("ninfer_{family}_{suffix}")];
    if let Some(var) = variant {
        v.push(format!("ninfer_{family}_{var}_{suffix}"));
        v.push(format!("ninfer_{var}_{family}_{suffix}"));
    }
    v
}

fn parse_test_targets(
    repo: &Path,
    op: &str,
    family: &str,
    variant: Option<&str>,
) -> Result<Vec<TestTarget>> {
    let tests_dir = repo.join("tests/ops");
    let mut decls: Vec<(PathBuf, CmakeTarget)> = Vec::new();
    for cmake in cmake_files(&tests_dir)? {
        for d in parse_target_decls(&cmake, "ninfer_add_op_test")? {
            decls.push((cmake.clone(), d));
        }
        for d in parse_target_decls(&cmake, "ninfer_add_test")? {
            decls.push((cmake.clone(), d));
        }
    }
    // Map any `add_test(NAME … COMMAND <exe>)` derived names onto their exe.
    let mut derived: Vec<(String, String)> = Vec::new();
    for cmake in cmake_files(&tests_dir)? {
        derived.extend(parse_add_test(&cmake)?);
    }

    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (_cmake, d) in decls {
        if !token_matches(&d.name, op, family, variant) {
            continue;
        }
        if !seen.insert(d.name.clone()) {
            continue;
        }
        let mut test_names = vec![d.name.clone()];
        for (n, exe) in &derived {
            if exe == &d.name && !test_names.contains(n) {
                test_names.push(n.clone());
            }
        }
        out.push(TestTarget {
            sources: d
                .sources
                .iter()
                .map(|s| rel_to(repo, Path::new(s)))
                .collect(),
            target: d.name,
            test_names,
        });
    }
    out.sort_by(|a, b| a.target.cmp(&b.target));
    let expected = expected_names(op, family, variant, "test");
    let exact: Vec<TestTarget> = out
        .iter()
        .filter(|t| expected.iter().any(|e| e == &t.target))
        .cloned()
        .collect();
    Ok(if exact.is_empty() { out } else { exact })
}

fn parse_bench_targets(
    repo: &Path,
    op: &str,
    family: &str,
    variant: Option<&str>,
) -> Result<Vec<BenchTarget>> {
    let bench_dir = repo.join("bench");
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for cmake in cmake_files(&bench_dir)? {
        for d in parse_target_decls(&cmake, "ninfer_add_op_bench")? {
            if !token_matches(&d.name, op, family, variant) {
                continue;
            }
            if !seen.insert(d.name.clone()) {
                continue;
            }
            out.push(BenchTarget {
                sources: d
                    .sources
                    .iter()
                    .map(|s| rel_to(repo, Path::new(s)))
                    .collect(),
                target: d.name,
            });
        }
    }
    out.sort_by(|a, b| a.target.cmp(&b.target));
    let expected = expected_names(op, family, variant, "bench");
    let exact: Vec<BenchTarget> = out
        .iter()
        .filter(|t| expected.iter().any(|e| e == &t.target))
        .cloned()
        .collect();
    Ok(if exact.is_empty() { out } else { exact })
}

/// Discover one op as a backend-agnostic [`Target`].
pub fn discover_target(repo: &Path, op: &str) -> Result<crate::backend::Target> {
    Ok(to_target(&discover(repo, op)?))
}

/// Enumerate every runnable ninfer op in a checkout (variants preferred over
/// whole families, so the campaign optimizes the smallest meaningful unit).
pub fn discover_targets(repo: &Path) -> Result<Vec<crate::backend::Target>> {
    use std::collections::{HashMap, HashSet};

    let src_ops = repo.join("src/ops");
    if !src_ops.is_dir() {
        anyhow::bail!("{} does not look like a ninfer tree (no src/ops)", repo.display());
    }

    let mut tokens: Vec<String> = Vec::new();
    // Flat ops: one kernel header per op under src/ops/kernel/.
    if let Ok(rd) = std::fs::read_dir(src_ops.join("kernel")) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("cuh") {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    tokens.push(stem.to_string());
                }
            }
        }
    }

    // Family ops: variants live in subdirectories (`linear_add/fp8`).
    let mut variants_by_family: HashMap<String, Vec<String>> = HashMap::new();
    if let Ok(rd) = std::fs::read_dir(&src_ops) {
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let family = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if matches!(family.as_str(), "common" | "kernel" | "launcher" | "wrapper") {
                continue;
            }
            let mut subs = Vec::new();
            if let Ok(rd2) = std::fs::read_dir(&p) {
                for e2 in rd2.flatten() {
                    let sp = e2.path();
                    if sp.is_dir() && has_cu(&sp) {
                        if let Some(sub) = sp.file_name().and_then(|s| s.to_str()) {
                            subs.push(sub.to_string());
                        }
                    }
                }
            }
            if subs.is_empty() {
                tokens.push(family);
            } else {
                for sub in &subs {
                    tokens.push(format!("{sub}_{family}"));
                }
                variants_by_family.insert(family, subs);
            }
        }
    }

    let mut targets: Vec<crate::backend::Target> = Vec::new();
    let mut seen = HashSet::new();
    for tok in &tokens {
        if !seen.insert(tok.clone()) {
            continue;
        }
        if let Ok(inv) = discover(repo, tok) {
            if !inv.tests.is_empty() && !inv.kernel_files.is_empty() {
                targets.push(to_target(&inv));
            }
        }
    }

    // Fall back to the family token when no variant resolved to a runnable test.
    for family in variants_by_family.keys() {
        let covered = targets.iter().any(|t| &t.family == family);
        if !covered && seen.insert(family.clone()) {
            if let Ok(inv) = discover(repo, family) {
                if !inv.tests.is_empty() && !inv.kernel_files.is_empty() {
                    targets.push(to_target(&inv));
                }
            }
        }
    }

    if targets.is_empty() {
        anyhow::bail!("no runnable ninfer ops found under {}", src_ops.display());
    }
    Ok(targets)
}

fn has_cu(dir: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return false;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if has_cu(&p) {
                return true;
            }
        } else if p.extension().and_then(|s| s.to_str()) == Some("cu") {
            return true;
        }
    }
    false
}

/// Convert a ninfer inventory into the uniform pipeline `Target`.
pub fn to_target(inv: &OpInventory) -> crate::backend::Target {
    let test_filters: Vec<String> = inv
        .tests
        .iter()
        .flat_map(|t| t.test_names.clone())
        .collect();
    let test_sources: Vec<String> = inv
        .tests
        .iter()
        .flat_map(|t| t.sources.clone())
        .filter(|s| s.ends_with(".cpp") || s.ends_with(".cu"))
        .collect();
    let bench_binary = inv.benches.first().map(|b| b.target.clone());
    let mut build_targets: Vec<String> = inv.tests.iter().map(|t| t.target.clone()).collect();
    if let Some(b) = &bench_binary {
        build_targets.push(b.clone());
    }
    let mut context_files = inv.context_files.clone();
    context_files.extend(inv.launcher_files.iter().cloned());
    context_files.extend(inv.wrapper_files.iter().cloned());
    crate::backend::Target {
        backend: crate::backend::Backend::Ninfer,
        op: inv.op.clone(),
        family: inv.family.clone(),
        variant: inv.variant.clone(),
        kernel_files: inv.kernel_files.clone(),
        context_files,
        contract_files: inv.contract_files.clone(),
        target_file: inv.kernel_files.first().cloned().unwrap_or_default(),
        build_targets,
        test_filters,
        test_sources,
        bench_binary,
        bench_args: Vec::new(),
        timing: !inv.benches.is_empty(),
        warnings: inv.warnings.clone(),
        configure_args: crate::backend::ninfer_configure_args(),
        test_cmd: None,
        bench_cmd: None,
        bench_format: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"// fixture\n").unwrap();
    }

    fn fixture_root() -> PathBuf {
        std::env::temp_dir().join(format!("kernelopt_ninfer_fixture_{}", uuid::Uuid::new_v4().simple()))
    }

    /// A minimal ninfer-like tree with one basic op and one quant family.
    fn build_fixture(root: &Path) {
        // basic op `fake_op`
        touch(&root.join("src/ops/kernel/fake_op.cuh"));
        touch(&root.join("src/ops/launcher/fake_op.cu"));
        touch(&root.join("src/ops/launcher/fake_op.h"));
        touch(&root.join("src/ops/wrapper/fake_op.cpp"));
        touch(&root.join("include/ninfer/ops/fake_op.h"));
        touch(&root.join("src/ops/basic_sources.cmake"));
        touch(&root.join("tests/ops/test_fake_op.cpp"));
        touch(&root.join("bench/ops/fake_op_bench.cu"));

        // family op `linear_add` with an `fp8` variant
        touch(&root.join("src/ops/linear_add/fp8/fp8_linear_add_kernel.cu"));
        touch(&root.join("src/ops/linear_add/fp8/fp8_linear_add_plan.h"));
        touch(&root.join("src/ops/linear_add/fp8/fp8_linear_add_plan.cpp"));
        touch(&root.join("src/ops/linear_add/sources.cmake"));
        touch(&root.join("include/ninfer/ops/linear_add.h"));
        touch(&root.join("src/ops/wrapper/linear_add.cpp"));
        touch(&root.join("tests/ops/linear_add/linear_add_test_common.cpp"));
        touch(&root.join("tests/ops/linear_add/test_fp8.cpp"));
        touch(&root.join("bench/ops/fp8_linear_add_bench.cu"));
        touch(&root.join("bench/ops/q8_linear_add_bench.cu"));

        std::fs::write(
            root.join("tests/ops/tests.cmake"),
            r#"
ninfer_add_op_test(ninfer_fake_op_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fake_op.cpp"
  LIBRARIES ninfer_ops)
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("tests/ops/linear_add/tests.cmake"),
            r#"
ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES ninfer_ops)
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("bench/ops/benchmarks.cmake"),
            r#"
ninfer_add_op_bench(ninfer_fake_op_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/fake_op_bench.cu")
ninfer_add_op_bench(ninfer_fp8_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/fp8_linear_add_bench.cu")
ninfer_add_op_bench(ninfer_q8_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q8_linear_add_bench.cu")
"#,
        )
        .unwrap();
    }

    #[test]
    fn discovers_basic_op() {
        let root = fixture_root();
        build_fixture(&root);
        let inv = discover(&root, "fake_op").unwrap();

        assert_eq!(inv.family, "fake_op");
        assert_eq!(inv.variant, None);
        assert_eq!(inv.kernel_files, vec!["src/ops/kernel/fake_op.cuh"]);
        assert!(inv.launcher_files.contains(&"src/ops/launcher/fake_op.cu".to_string()));
        assert!(inv.wrapper_files.contains(&"src/ops/wrapper/fake_op.cpp".to_string()));
        assert_eq!(inv.contract_files, vec!["include/ninfer/ops/fake_op.h"]);
        assert_eq!(inv.tests.len(), 1);
        assert_eq!(inv.tests[0].target, "ninfer_fake_op_test");
        assert_eq!(inv.tests[0].sources, vec!["tests/ops/test_fake_op.cpp"]);
        assert_eq!(inv.benches.len(), 1);
        assert_eq!(inv.benches[0].target, "ninfer_fake_op_bench");
        assert!(inv.warnings.is_empty(), "unexpected warnings: {:?}", inv.warnings);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn discovers_quant_variant() {
        let root = fixture_root();
        build_fixture(&root);
        let inv = discover(&root, "fp8_linear_add").unwrap();

        assert_eq!(inv.family, "linear_add");
        assert_eq!(inv.variant.as_deref(), Some("fp8"));
        assert_eq!(inv.kernel_files, vec!["src/ops/linear_add/fp8/fp8_linear_add_kernel.cu"]);
        assert!(inv.context_files.contains(&"src/ops/linear_add/fp8/fp8_linear_add_plan.h".to_string()));
        assert_eq!(inv.contract_files, vec!["include/ninfer/ops/linear_add.h"]);
        assert_eq!(inv.tests.len(), 1);
        assert_eq!(inv.tests[0].target, "ninfer_linear_add_fp8_test");
        // Only the fp8 bench, not q8.
        assert_eq!(inv.benches.len(), 1);
        assert_eq!(inv.benches[0].target, "ninfer_fp8_linear_add_bench");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn unknown_op_warns() {
        let root = fixture_root();
        build_fixture(&root);
        let inv = discover(&root, "does_not_exist").unwrap();
        assert!(inv.kernel_files.is_empty());
        assert!(inv.tests.is_empty());
        assert!(!inv.warnings.is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn cmake_helpers() {
        let text = r#"
ninfer_add_op_test(ninfer_x_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_x.cpp" "${CMAKE_CURRENT_LIST_DIR}/test_y.cpp"
  LIBRARIES ninfer_ops)
"#;
        let calls = find_calls(text, "ninfer_add_op_test");
        assert_eq!(calls.len(), 1);
        assert_eq!(first_token(&calls[0]), "ninfer_x_test");
        let q = quoted_strings(calls[0].split("LIBRARIES").next().unwrap());
        assert_eq!(q.len(), 2);
    }
}
