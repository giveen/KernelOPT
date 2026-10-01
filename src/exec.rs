//! Direct invocation of the external tools (cmake/ctest/bench/ncu/
//! test-backend-ops). Replaces the Python runner's `cuda_*`/`llama_*` commands;
//! parsing lives in `parse`. Everything runs with a wall-clock timeout and
//! captures output without pipe deadlock.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub struct RunOut {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// Run with a timeout, capturing stdout/stderr on reader threads (so a full
/// pipe can't deadlock the child) and killing it if it overruns.
pub fn run_capture(cmd: &mut Command, timeout_secs: u64) -> Result<RunOut> {
    // Own process group so a timeout can kill the whole tree (e.g. ctest → the
    // test binary), not just the direct child — no orphaned GPU/CPU spinners.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {:?}", cmd.get_program()))?;
    let mut out = child.stdout.take().expect("stdout pipe");
    let mut err = child.stderr.take().expect("stderr pipe");
    let oh = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let eh = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    });
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let mut timed_out = false;
    let code = loop {
        match child.try_wait()? {
            Some(st) => break st.code(),
            None if Instant::now() >= deadline => {
                kill_tree(&mut child);
                timed_out = true;
                break None;
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
    };
    Ok(RunOut {
        code,
        stdout: oh.join().unwrap_or_default(),
        stderr: eh.join().unwrap_or_default(),
        timed_out,
    })
}

/// Kill a child and everything in its process group. The group is created by
/// `process_group(0)` in `run_capture`, so a tool that forks helpers (ctest →
/// test binary, ninja → compilers) leaves nothing running after a timeout.
fn kill_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // Negative pid = process group (which equals the child's pid here).
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Gate 1: configure (optional) + build. Returns a `compile_view`-ready payload.
#[allow(clippy::too_many_arguments)]
pub fn build(
    worktree: &Path,
    build_dir: &Path,
    targets: &[String],
    configure: bool,
    force_configure: bool,
    configure_args: &[String],
    jobs: usize,
    timeout_secs: u64,
) -> Result<Value> {
    if configure && (force_configure || !build_dir.join("CMakeCache.txt").exists()) {
        // Wipe a cache configured for a different source tree (cmake refuses to
        // reuse it).
        if let Ok(text) = std::fs::read_to_string(build_dir.join("CMakeCache.txt")) {
            if let Some(home) = text
                .lines()
                .find_map(|l| l.strip_prefix("CMAKE_HOME_DIRECTORY:"))
                .map(|s| s.trim().to_string())
            {
                if std::fs::canonicalize(&home).ok() != std::fs::canonicalize(worktree).ok() {
                    let _ = std::fs::remove_dir_all(build_dir);
                }
            }
        }
        let args: Vec<String> = if configure_args.is_empty() {
            ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON", "-DNINFER_BUILD_BENCHMARKS=ON"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            configure_args.to_vec()
        };
        let mut cmd = Command::new("cmake");
        cmd.arg("-S")
            .arg(worktree)
            .arg("-B")
            .arg(build_dir)
            .arg("-G")
            .arg("Ninja")
            .args(&args);
        let o = run_capture(&mut cmd, timeout_secs)?;
        if o.timed_out {
            bail!("cmake configure timed out after {timeout_secs}s");
        }
        if o.code != Some(0) {
            return Ok(json!({
                "ok": true, "passed": false, "stage": "configure", "exit_code": o.code,
                "raw_stdout": o.stdout, "raw_stderr": o.stderr,
            }));
        }
    }

    let mut cmd = Command::new("cmake");
    cmd.arg("--build")
        .arg(build_dir)
        .arg("-j")
        .arg(jobs.to_string());
    if !targets.is_empty() {
        cmd.arg("--target").args(targets);
    }
    let o = run_capture(&mut cmd, timeout_secs)?;
    if o.timed_out {
        bail!("cmake --build timed out after {timeout_secs}s");
    }
    Ok(json!({
        "ok": true, "passed": o.code == Some(0), "exit_code": o.code,
        "targets": targets, "raw_stdout": o.stdout, "raw_stderr": o.stderr,
    }))
}

/// Gate 2 (ninfer): ctest.
pub fn ctest(build_dir: &Path, tests: &[String], timeout_secs: u64) -> Result<Value> {
    let mut cmd = Command::new("ctest");
    cmd.arg("--test-dir")
        .arg(build_dir)
        .arg("--output-on-failure");
    if !tests.is_empty() {
        let pat = format!(
            "^({})$",
            tests
                .iter()
                .map(|t| regex::escape(t))
                .collect::<Vec<_>>()
                .join("|")
        );
        cmd.arg("-R").arg(pat);
    }
    let o = run_capture(&mut cmd, timeout_secs)?;
    if o.timed_out {
        bail!("ctest timed out after {timeout_secs}s");
    }
    Ok(json!({
        "ok": true, "passed": o.code == Some(0), "exit_code": o.code,
        "tests": tests, "raw_stdout": o.stdout, "raw_stderr": o.stderr,
    }))
}

/// Probe `--help` for supported flags (cheap: help exits before GPU init).
fn bench_capabilities(binary: &Path, timeout_secs: u64) -> HashSet<String> {
    let mut cmd = Command::new(binary);
    cmd.arg("--help");
    let Ok(o) = run_capture(&mut cmd, timeout_secs) else {
        return HashSet::new();
    };
    let text = format!("{}{}", o.stdout, o.stderr);
    ["--csv-out", "--warmup", "--repeat", "--profile", "--t-sweep"]
        .into_iter()
        .filter(|f| text.contains(f))
        .map(String::from)
        .collect()
}

/// Gate 4: run the bench `repeats` times, returning raw output per run.
#[allow(clippy::too_many_arguments)]
pub fn bench(
    binary: &Path,
    args: &[String],
    csv_out: Option<&Path>,
    warmup: Option<u32>,
    repeat: Option<u32>,
    repeats: u32,
    timeout_secs: u64,
) -> Result<Value> {
    if !binary.exists() {
        bail!("bench binary not found: {}", binary.display());
    }
    let caps = bench_capabilities(binary, 60);
    let mut argv: Vec<String> = vec![binary.to_string_lossy().to_string()];
    argv.extend(args.iter().cloned());
    if let Some(c) = csv_out {
        if caps.contains("--csv-out") && !args.iter().any(|a| a == "--csv-out") {
            if let Some(p) = c.parent() {
                let _ = std::fs::create_dir_all(p);
            }
            argv.push("--csv-out".into());
            argv.push(c.to_string_lossy().to_string());
        }
    }
    if let Some(w) = warmup {
        if caps.contains("--warmup") {
            argv.push("--warmup".into());
            argv.push(w.to_string());
        }
    }
    if let Some(r) = repeat {
        if caps.contains("--repeat") {
            argv.push("--repeat".into());
            argv.push(r.to_string());
        }
    }

    let repeats = repeats.max(1);
    let mut runs = Vec::new();
    let mut last_code = None;
    for _ in 0..repeats {
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]);
        let o = run_capture(&mut cmd, timeout_secs)?;
        if o.timed_out {
            bail!("bench timed out after {timeout_secs}s");
        }
        last_code = o.code;
        let csv = csv_out
            .and_then(|p| std::fs::read_to_string(p).ok())
            .filter(|s| !s.trim().is_empty());
        runs.push(json!({"stdout": o.stdout, "stderr": o.stderr, "csv": csv}));
    }
    Ok(json!({
        "ok": true, "passed": last_code == Some(0), "exit_code": last_code,
        "command": argv.join(" "), "repeats": repeats, "runs": runs,
    }))
}

/// Discover the `ncu` binary (PATH, then CUDA_HOME/CUDA_PATH).
pub fn find_ncu() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let p = Path::new(dir).join("ncu");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    for env in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = std::env::var(env) {
            let p = Path::new(&root).join("bin").join("ncu");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

/// Planner context: profile `target_argv` with ncu (raw CSV out).
#[allow(clippy::too_many_arguments)]
pub fn ncu(
    target_argv: &[String],
    ncu_set: &str,
    launch_skip: Option<u32>,
    launch_count: Option<u32>,
    kernel_name: Option<&str>,
    timeout_secs: u64,
) -> Result<Value> {
    let ncu = find_ncu().context("ncu binary not found")?;
    let mut cmd = Command::new(&ncu);
    cmd.arg("--csv")
        .arg(format!("--set={ncu_set}"))
        .arg("--target-processes=all");
    if let Some(s) = launch_skip {
        cmd.arg(format!("--launch-skip={s}"));
    }
    if let Some(c) = launch_count {
        cmd.arg(format!("--launch-count={c}"));
    }
    if let Some(k) = kernel_name {
        cmd.arg(format!("--kernel-name={k}"));
    }
    cmd.args(target_argv);
    let o = run_capture(&mut cmd, timeout_secs)?;
    if o.timed_out {
        bail!("ncu timed out after {timeout_secs}s");
    }
    Ok(json!({
        "ok": true, "exit_code": o.code,
        "raw_stdout": o.stdout, "raw_stderr": o.stderr,
    }))
}

/// Run an arbitrary argv (custom-backend test/bench commands), capturing output.
pub fn run_argv(argv: &[String], timeout_secs: u64) -> Result<Value> {
    let Some(prog) = argv.first() else {
        bail!("empty command");
    };
    let mut cmd = Command::new(prog);
    cmd.args(&argv[1..]);
    let o = run_capture(&mut cmd, timeout_secs)?;
    if o.timed_out {
        bail!("command timed out after {timeout_secs}s: {}", argv.join(" "));
    }
    Ok(json!({
        "ok": true, "passed": o.code == Some(0), "exit_code": o.code,
        "raw_stdout": o.stdout, "raw_stderr": o.stderr,
    }))
}

/// Discover the `nsys` binary (PATH, then CUDA_HOME/CUDA_PATH).
pub fn find_nsys() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let p = Path::new(dir).join("nsys");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    for env in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = std::env::var(env) {
            let p = Path::new(&root).join("bin").join("nsys");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

/// Trace `workload` with Nsight Systems (CUDA only) into `<out_prefix>.nsys-rep`.
/// Unlike ncu, nsys needs no admin counters and captures kernels *and* memcpy,
/// including kernels inside CUDA graphs.
pub fn nsys_profile(workload: &[String], out_prefix: &Path, timeout_secs: u64) -> Result<Value> {
    let nsys = find_nsys().context("nsys binary not found")?;
    let mut cmd = Command::new(&nsys);
    cmd.arg("profile")
        .arg("-t")
        .arg("cuda")
        .arg("--force-overwrite=true")
        .arg("-o")
        .arg(out_prefix)
        .args(workload);
    let o = run_capture(&mut cmd, timeout_secs)?;
    if o.timed_out {
        bail!("nsys profile timed out after {timeout_secs}s");
    }
    Ok(json!({
        "ok": true, "exit_code": o.code,
        "raw_stdout": o.stdout, "raw_stderr": o.stderr,
    }))
}

/// Run one `nsys stats` report (e.g. `cuda_gpu_kern_sum`) and return its CSV.
pub fn nsys_stats(rep: &Path, report: &str, timeout_secs: u64) -> Result<String> {
    let nsys = find_nsys().context("nsys binary not found")?;
    let mut cmd = Command::new(&nsys);
    cmd.arg("stats")
        .arg("--format")
        .arg("csv")
        .arg("--report")
        .arg(report)
        .arg(rep);
    let o = run_capture(&mut cmd, timeout_secs)?;
    if o.timed_out {
        bail!("nsys stats timed out after {timeout_secs}s");
    }
    Ok(o.stdout)
}

pub fn llama_test(
    binary: &Path,
    backend: &str,
    ops: &[String],
    timeout_secs: u64,
) -> Result<Value> {
    if !binary.exists() {
        bail!("test-backend-ops not found: {}", binary.display());
    }
    let mut cmd = Command::new(binary);
    cmd.arg("test").arg("-b").arg(backend);
    if !ops.is_empty() {
        cmd.arg("-o").arg(ops.join(","));
    }
    let o = run_capture(&mut cmd, timeout_secs)?;
    if o.timed_out {
        bail!("llama_verify timed out after {timeout_secs}s");
    }
    Ok(json!({
        "ok": true, "passed": o.code == Some(0), "exit_code": o.code,
        "ops": ops, "raw_stdout": o.stdout, "raw_stderr": o.stderr,
    }))
}

/// Gate 4 (llama.cpp): `test-backend-ops perf`, raw output per run.
pub fn llama_perf(
    binary: &Path,
    backend: &str,
    ops: &[String],
    args: &[String],
    repeats: u32,
    timeout_secs: u64,
) -> Result<Value> {
    if !binary.exists() {
        bail!("test-backend-ops not found: {}", binary.display());
    }
    let mut base: Vec<String> = vec![
        binary.to_string_lossy().to_string(),
        "perf".into(),
        "-b".into(),
        backend.into(),
    ];
    if !ops.is_empty() {
        base.push("-o".into());
        base.push(ops.join(","));
    }
    base.extend(args.iter().cloned());

    let repeats = repeats.max(1);
    let mut runs = Vec::new();
    let mut last = None;
    for _ in 0..repeats {
        let mut cmd = Command::new(&base[0]);
        cmd.args(&base[1..]);
        let o = run_capture(&mut cmd, timeout_secs)?;
        if o.timed_out {
            bail!("llama_bench timed out after {timeout_secs}s");
        }
        last = o.code;
        runs.push(json!({"stdout": o.stdout, "stderr": o.stderr, "csv": Value::Null}));
    }
    Ok(json!({
        "ok": true, "passed": last == Some(0), "exit_code": last,
        "repeats": repeats, "runs": runs,
    }))
}

/// Best-effort, runtime toolchain summary for the Planner: compiler + release
/// and the target GPU arch. Every value is detected from the host (compiler
/// `--version`, the build's CMake cache, the live GPU) — nothing is hardcoded,
/// so it adapts to any NVIDIA/CUDA, AMD/ROCm, or CPU-only machine. `None` when
/// nothing is detectable.
pub fn detect_toolchain(build_dir: &Path) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(v) = tool_version("nvcc", "release ") {
        parts.push(format!("nvcc {v}"));
    } else if let Some(v) = tool_version("hipcc", "HIP version:") {
        parts.push(format!("hipcc {v}"));
    }
    // Prefer the arch the build was configured for; else the live GPU. CUDA and
    // HIP caches use different keys; the two GPU probes cover NVIDIA and AMD.
    if let Some(a) = cache_value(build_dir, "CMAKE_CUDA_ARCHITECTURES")
        .or_else(|| cache_value(build_dir, "CMAKE_HIP_ARCHITECTURES"))
    {
        parts.push(format!("target arch {a}"));
    } else if let Some(cc) = nvidia_compute_cap() {
        parts.push(format!("target sm_{cc}"));
    } else if let Some(a) = amd_arch() {
        parts.push(format!("target {a}"));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(", "))
    }
}

/// Leading version token after `needle` in `<tool> --version` (nvcc → "13.3").
fn tool_version(tool: &str, needle: &str) -> Option<String> {
    let out = Command::new(tool).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let rest = text.get(text.find(needle)? + needle.len()..)?;
    let tok: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    (!tok.is_empty()).then_some(tok)
}

/// `KEY` value from a CMake cache (the arch the target is actually built for).
fn cache_value(build_dir: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(build_dir.join("CMakeCache.txt")).ok()?;
    text.lines()
        .find_map(|l| l.split_once('=').filter(|(k, _)| k.starts_with(key)))
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// NVIDIA GPU compute capability as digits ("12.0" → "120"), if `nvidia-smi`
/// can report it.
fn nvidia_compute_cap() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    compute_cap_digits(&String::from_utf8_lossy(&out.stdout))
}

/// AMD GPU target arch (`gfxNNNN`), via `amdgpu-arch` (ships with ROCm) with a
/// `rocminfo` fallback. `None` on non-AMD hosts.
fn amd_arch() -> Option<String> {
    if let Ok(o) = Command::new("amdgpu-arch").output() {
        if o.status.success() {
            if let Some(a) = String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .find(|l| l.starts_with("gfx"))
            {
                return Some(a.to_string());
            }
        }
    }
    let o = Command::new("rocminfo").output().ok()?;
    if !o.status.success() {
        return None;
    }
    String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .find(|t| t.starts_with("gfx"))
        .map(|s| s.trim_matches(|c| c == ',' || c == ';').to_string())
}

/// "12.0" → "120"; already-digit strings pass through. `None` if empty.
fn compute_cap_digits(s: &str) -> Option<String> {
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    (!digits.is_empty()).then_some(digits)
}

/// Free GPU memory in GB (NVIDIA, first device) — best effort.
pub fn free_gpu_gb() -> Option<f64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mib: f64 = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()?;
    Some(mib / 1024.0)
}

/// Estimated on-disk size of a model path in GB (a file, or the summed files of
/// a directory). `None` for a bare name / missing path.
pub fn model_size_gb(path: &str) -> Option<f64> {
    let meta = std::fs::metadata(path).ok()?;
    let bytes = if meta.is_file() {
        meta.len()
    } else if meta.is_dir() {
        let mut total = 0u64;
        for entry in std::fs::read_dir(path).ok()? {
            if let Ok(m) = entry.ok()?.metadata() {
                if m.is_file() {
                    total += m.len();
                }
            }
        }
        total
    } else {
        return None;
    };
    (bytes > 0).then_some(bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_cap_digits_normalizes() {
        assert_eq!(compute_cap_digits("12.0\n"), Some("120".into()));
        assert_eq!(compute_cap_digits("9.0"), Some("90".into()));
        assert_eq!(compute_cap_digits("n/a"), None);
    }

    #[test]
    fn model_size_gb_reads_files() {
        assert_eq!(model_size_gb("/no/such/path/does-not-exist"), None);
        let f = std::env::temp_dir()
            .join(format!("kopt-size-{}", uuid::Uuid::new_v4().simple()));
        std::fs::write(&f, vec![0u8; 2_000_000]).unwrap();
        let gb = model_size_gb(&f.to_string_lossy()).unwrap();
        assert!((gb - 0.002).abs() < 1e-6, "{gb}");
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn cache_value_reads_arch() {
        let dir = std::env::temp_dir()
            .join(format!("kernelopt-cache-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("CMakeCache.txt"),
            "// cuda arch\nCMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120a\nOTHER=VALUE\n",
        )
        .unwrap();
        assert_eq!(cache_value(&dir, "CMAKE_CUDA_ARCHITECTURES"), Some("120a".into()));
        assert_eq!(cache_value(&dir, "CMAKE_HIP_ARCHITECTURES"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
