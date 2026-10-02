//! `kernelopt setup`: check (and where possible, install) the external tools
//! KernelOPT depends on or benefits from.
//!
//! Required-ish: git, cmake (+ctest), a build runner (ninja/make), a GPU
//! compiler (nvcc/hipcc), python3. Recommended: nsys/ncu profilers (ship with
//! the CUDA toolkit), the Graphsignal venv (`setup-graphsignal`), and the
//! codebase-memory-mcp binary (auto-installed by `setup --install`).
//! Optional: hip-docs-mcp (needs ROCm), docs MCP tokens (`docs --login`).

use anyhow::{Context, Result};
use std::path::PathBuf;

/// How much KernelOPT needs a tool.
#[derive(Clone, Copy, PartialEq)]
pub enum Need {
    Required,
    Recommended,
    Optional,
}

pub struct ToolStatus {
    pub name: &'static str,
    pub need: Need,
    pub found: bool,
    pub detail: String,
    pub hint: String,
    /// Whether `setup --install` can install it.
    pub installable: bool,
}

fn on_path(tool: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p)
            .map(|d| d.join(tool))
            .find(|p| p.is_file())
    })
}

fn version_of(tool: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(tool).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    text.lines().next().map(|l| l.trim().chars().take(80).collect())
}

fn prog(out: &mut Vec<ToolStatus>, name: &'static str, need: Need, args: &[&str], hint: &'static str) {
    let (found, detail) = match on_path(name) {
        Some(_) => (true, version_of(name, args).unwrap_or_default()),
        None => (false, String::new()),
    };
    out.push(ToolStatus {
        name,
        need,
        found,
        detail,
        hint: hint.to_string(),
        installable: false,
    });
}

/// Presence plus a minimum version floor (e.g. cmake 3.28, python3 3.10).
/// Below the floor counts as missing, with the actual version shown.
fn prog_min(
    out: &mut Vec<ToolStatus>,
    name: &'static str,
    need: Need,
    args: &[&str],
    hint: &'static str,
    min: (u64, u64, u64),
    min_str: &'static str,
) {
    let (found, detail) = match on_path(name).and_then(|_| full_version_of(name, args)) {
        Some((text, ver)) if ver >= min => (true, first_line(&text)),
        Some((text, _)) => (
            false,
            format!("found {}, need >= {}", first_line(&text), min_str),
        ),
        None => (false, String::new()),
    };
    out.push(ToolStatus {
        name,
        need,
        found,
        detail,
        hint: hint.to_string(),
        installable: false,
    });
}

/// Full `--version` output plus the first `X.Y[.Z]` found in it.
fn full_version_of(tool: &str, args: &[&str]) -> Option<(String, (u64, u64, u64))> {
    let out = std::process::Command::new(tool).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_version(&text).map(|v| (text, v))
}

/// First `X.Y[.Z]` version triple in text.
fn parse_version(text: &str) -> Option<(u64, u64, u64)> {
    let re = regex::Regex::new(r"(\d+)\.(\d+)(?:\.(\d+))?").ok()?;
    let c = re.captures(text)?;
    Some((
        c[1].parse().ok()?,
        c[2].parse().ok()?,
        c.get(3).map(|m| m.as_str().parse().unwrap_or(0)).unwrap_or(0),
    ))
}

fn first_line(text: &str) -> String {
    text.lines()
        .next()
        .map(|l| l.trim().chars().take(80).collect())
        .unwrap_or_default()
}

/// GPU presence: NVIDIA GPUs counted, else ROCm presence.
fn gpu_status() -> ToolStatus {
    let base = ToolStatus {
        name: "GPU",
        need: Need::Required,
        found: false,
        detail: String::new(),
        hint: "no NVIDIA (nvidia-smi) or AMD (rocm-smi) GPU detected".to_string(),
        installable: false,
    };
    if let Some(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let names: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if !names.is_empty() {
            return ToolStatus {
                found: true,
                detail: format!("{}× {}", names.len(), names[0]),
                ..base
            };
        }
    }
    if on_path("rocm-smi").is_some() {
        return ToolStatus {
            found: true,
            detail: "ROCm GPU (rocm-smi present)".to_string(),
            ..base
        };
    }
    base
}

fn euid() -> u32 {
    unsafe { libc::geteuid() }
}

/// Can this user collect GPU perf counters (what `ncu --set full` needs)?
/// Pure for testing: root always can; otherwise the NVIDIA driver must not
/// restrict profiling to admin (`RmProfilingAdminOnly: 0`).
fn profiling_allowed(params: Option<&str>, uid: u32) -> bool {
    if uid == 0 {
        return true;
    }
    params
        .and_then(|t| {
            t.lines().find_map(|l| {
                let (k, v) = l.split_once(':')?;
                (k.trim() == "RmProfilingAdminOnly").then(|| v.trim() == "0")
            })
        })
        .unwrap_or(false)
}

/// NCU profiling-permission row. `None` when there is no NVIDIA stack to check
/// (e.g. ROCm-only or CPU-only hosts).
fn ncu_profiling_status() -> Option<ToolStatus> {
    let has_nvidia = on_path("nvidia-smi").is_some() || on_path("ncu").is_some();
    if !has_nvidia {
        return None;
    }
    let params = std::fs::read_to_string("/proc/driver/nvidia/params").ok();
    let ok = profiling_allowed(params.as_deref(), euid());
    Some(ToolStatus {
        name: "ncu profiling (perf counters)",
        need: Need::Recommended,
        found: ok,
        detail: if ok {
            if euid() == 0 {
                "running as root".to_string()
            } else {
                "RmProfilingAdminOnly=0 (non-root profiling allowed)".to_string()
            }
        } else {
            String::new()
        },
        hint: "run as root, grant CAP_SYS_ADMIN, or set NVreg_RestrictProfilingToAdminHost=0 via /etc/modprobe.d + reboot".to_string(),
        installable: false,
    })
}

/// Check every known external tool. Read-only; safe to run anywhere.
pub fn check_all() -> Vec<ToolStatus> {
    let mut out = Vec::new();

    prog(&mut out, "git", Need::Required, &["--version"], "system package manager (git)");
    // ninfer checkouts require cmake >= 3.28; the runner needs python >= 3.10.
    prog_min(&mut out, "cmake", Need::Required, &["--version"], "system package manager (cmake); provides ctest", (3, 28, 0), "3.28");
    prog_min(&mut out, "ctest", Need::Required, &["--version"], "ships with cmake", (3, 28, 0), "3.28");
    prog_min(&mut out, "python3", Need::Required, &["--version"], "system package manager (python3)", (3, 10, 0), "3.10");


    // Either build runner is fine.
    let ninja = on_path("ninja").map(|p| ("ninja", p));
    let make = on_path("make").map(|p| ("make", p));
    match ninja.or(make) {
        Some((which, _)) => out.push(ToolStatus {
            name: "build runner",
            need: Need::Required,
            found: true,
            detail: format!("{which} on PATH"),
            hint: String::new(),
            installable: false,
        }),
        None => out.push(ToolStatus {
            name: "build runner",
            need: Need::Required,
            found: false,
            detail: String::new(),
            hint: "ninja or make via system package manager".to_string(),
            installable: false,
        }),
    }
    // Either GPU compiler is fine.
    let nvcc = on_path("nvcc");
    let hipcc = on_path("hipcc");
    match (nvcc, hipcc) {
        (Some(_), _) => out.push(ToolStatus {
            name: "nvcc",
            need: Need::Required,
            found: true,
            detail: version_of("nvcc", &["--version"]).unwrap_or_default(),
            hint: String::new(),
            installable: false,
        }),
        (None, Some(_)) => out.push(ToolStatus {
            name: "hipcc",
            need: Need::Required,
            found: true,
            detail: version_of("hipcc", &["--version"]).unwrap_or_default(),
            hint: String::new(),
            installable: false,
        }),
        (None, None) => out.push(ToolStatus {
            name: "nvcc/hipcc",
            need: Need::Required,
            found: false,
            detail: String::new(),
            hint: "CUDA toolkit (nvcc) or ROCm (hipcc)".to_string(),
            installable: false,
        }),
    }
    prog(&mut out, "nsys", Need::Recommended, &["--version"], "ships with the CUDA toolkit");
    prog(&mut out, "ncu", Need::Recommended, &["--version"], "ships with the CUDA toolkit");
    out.push(gpu_status());
    if let Some(s) = ncu_profiling_status() {
        out.push(s);
    }

    // Managed Graphsignal venv (installed by `setup-graphsignal`).
    let gs = std::env::current_dir()
        .map(|c| c.join(".kernelopt/graphsignal"))
        .unwrap_or_default();
    out.push(ToolStatus {
        name: "graphsignal venv",
        need: Need::Recommended,
        found: gs.join("venv").is_dir() || gs.join("bin").is_dir(),
        detail: String::new(),
        hint: "kernelopt setup-graphsignal".to_string(),
        installable: false,
    });

    // codebase-memory-mcp binary (auto-installable below).
    match on_path("codebase-memory-mcp") {
        Some(p) => out.push(ToolStatus {
            name: "codebase-memory-mcp",
            need: Need::Recommended,
            found: true,
            detail: p.display().to_string(),
            hint: String::new(),
            installable: true,
        }),
        None => out.push(ToolStatus {
            name: "codebase-memory-mcp",
            need: Need::Recommended,
            found: false,
            detail: String::new(),
            hint: "kernelopt setup --install (downloads the static binary)".to_string(),
            installable: true,
        }),
    }

    // hip-docs-mcp (needs ROCm to run; installable but useless without it).
    out.push(ToolStatus {
        name: "hip-docs-mcp",
        need: Need::Optional,
        found: on_path("hip-docs-mcp").is_some(),
        detail: String::new(),
        hint: "uv/pip install from AMDResearch/intellikit (needs ROCm to run)".to_string(),
        installable: false,
    });

    // LLM provider: env presence only (no network probe here).
    let provider = std::env::var("KERNELOPT_PROVIDER").unwrap_or_else(|_| "opencode-go".into());
    let model = std::env::var("KERNELOPT_MODEL").unwrap_or_default();
    out.push(ToolStatus {
        name: "LLM provider",
        need: Need::Required,
        found: !model.is_empty(),
        detail: if model.is_empty() {
            "KERNELOPT_MODEL unset".to_string()
        } else {
            format!("{provider}/{model} (see `kernelopt providers`)")
        },
        hint: "set KERNELOPT_PROVIDER/KERNELOPT_MODEL (or --provider/--model)".to_string(),
        installable: false,
    });

    out
}

/// Render the check table for humans.
pub fn render_table(status: &[ToolStatus]) -> String {
    let mut s = String::from("tool                     need         status\n");
    for t in status {
        let need = match t.need {
            Need::Required => "required   ",
            Need::Recommended => "recommended",
            Need::Optional => "optional   ",
        };
        let status = if t.found {
            if t.detail.is_empty() {
                "OK".to_string()
            } else {
                format!("OK ({})", truncate(&t.detail, 50))
            }
        } else {
            format!("MISSING — {}", t.hint)
        };
        s.push_str(&format!("{:<25}{need}  {status}\n", t.name));
    }
    s
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n.saturating_sub(1)).collect::<String>())
    }
}

/// Download the codebase-memory-mcp static binary for this platform into
/// `~/.local/bin` (or `dir` when given). Returns the installed path.
pub fn install_codemap(dir: Option<PathBuf>) -> Result<PathBuf> {
    let asset = codemap_asset_name().context("unsupported platform for codebase-memory-mcp")?;
    let tag = latest_codemap_tag()?;
    let url = format!("https://github.com/DeusData/codebase-memory-mcp/releases/download/{tag}/{asset}");
    let dest_dir = dir.unwrap_or_else(home_local_bin);
    std::fs::create_dir_all(&dest_dir)?;
    let tgz = dest_dir.join(format!("{asset}.download"));
    download(&url, &tgz)?;
    // Extract the single binary from the tarball.
    let out = std::process::Command::new("tar")
        .args(["-xzf"])
        .arg(&tgz)
        .arg("-C")
        .arg(&dest_dir)
        .output()
        .context("extracting codebase-memory-mcp (needs tar)")?;
    if !out.status.success() {
        anyhow::bail!("extract failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let _ = std::fs::remove_file(&tgz);
    let bin = find_extracted_binary(&dest_dir).context("binary not found in archive")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = std::fs::metadata(&bin)?.permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&bin, perm)?;
    }
    Ok(bin)
}

fn home_local_bin() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local/bin")
}

/// Asset stem for this OS/arch, e.g. `codebase-memory-mcp-linux-amd64.tar.gz`.
pub fn codemap_asset_name() -> Option<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let plat = match (os, arch) {
        ("linux", "x86_64") => "linux-amd64",
        ("linux", "aarch64") => "linux-arm64",
        ("macos", "aarch64") => "darwin-arm64",
        ("macos", "x86_64") => "darwin-amd64",
        ("windows", "x86_64") => "windows-amd64",
        _ => return None,
    };
    Some(format!("codebase-memory-mcp-{plat}.tar.gz"))
}

/// Latest release tag from the GitHub API.
pub fn latest_codemap_tag() -> Result<String> {
    let text = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?
        .get("https://api.github.com/repos/DeusData/codebase-memory-mcp/releases/latest")
        .header("User-Agent", "kernelopt")
        .header("Accept", "application/vnd.github+json")
        .send()
        .context("querying GitHub releases")?
        .text()?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    v["tag_name"]
        .as_str()
        .map(String::from)
        .context("no tag_name in release response")
}

fn download(url: &str, dest: &std::path::Path) -> Result<()> {
    let mut resp = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?
        .get(url)
        .header("User-Agent", "kernelopt")
        .send()
        .with_context(|| format!("downloading {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("download failed: HTTP {}", resp.status());
    }
    let mut f = std::fs::File::create(dest)?;
    std::io::copy(&mut resp, &mut f)?;
    Ok(())
}

fn find_extracted_binary(dir: &std::path::Path) -> Option<PathBuf> {
    let mut found = None;
    for e in walkdir_files(dir) {
        let name = e.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if name == "codebase-memory-mcp" || name == "codebase-memory-mcp.exe" {
            // Prefer a top-level or bin/ match.
            if found.is_none() {
                found = Some(e);
            }
        }
    }
    found
}

fn walkdir_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_name_matches_host() {
        let a = codemap_asset_name().expect("test host should be supported");
        assert!(a.starts_with("codebase-memory-mcp-"));
        assert!(a.ends_with(".tar.gz"));
    }

    #[test]
    fn parse_version_reads_first_triple() {
        assert_eq!(parse_version("cmake version 4.3.4"), Some((4, 3, 4)));
        assert_eq!(parse_version("Python 3.14.7"), Some((3, 14, 7)));
        assert_eq!(parse_version("nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.3, V13.3.73"), Some((13, 3, 0)));
        assert_eq!(parse_version("no version here"), None);
        assert!((3, 27, 0) < (3, 28, 0));
    }

    #[test]
    fn profiling_allowed_needs_root_or_unrestricted_driver() {
        let open = "RmProfilingAdminOnly: 0\nOther: 1\n";
        let locked = "RmProfilingAdminOnly: 1\n";
        assert!(profiling_allowed(Some(open), 1000));
        assert!(!profiling_allowed(Some(locked), 1000));
        assert!(!profiling_allowed(None, 1000));
        assert!(profiling_allowed(Some(locked), 0));
        assert!(profiling_allowed(None, 0));
    }

    #[test]
    fn table_renders_missing_with_hint() {        let t = vec![ToolStatus {
            name: "ncu",
            need: Need::Recommended,
            found: false,
            detail: String::new(),
            hint: "CUDA toolkit".into(),
            installable: false,
        }];
        let s = render_table(&t);
        assert!(s.contains("ncu") && s.contains("MISSING") && s.contains("CUDA toolkit"), "{s}");
    }
}
