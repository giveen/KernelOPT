//! Git worktree / diff / patch operations (moved from the Python runner).
//!
//! The user's own checkout is never modified: candidates live in a linked
//! worktree on a dedicated branch. Everything shells out to `git` directly, so
//! a run no longer pays a Python hop for each worktree operation.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn git(dir: &Path, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .with_context(|| format!("git -C {} {}", dir.display(), args.join(" ")))
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).trim().to_string()
}

fn tail(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= n {
        return s.to_string();
    }
    chars[chars.len() - n..].iter().collect()
}

/// Absolute path of the repository's shared `.git` (or None).
fn common_dir(path: &Path) -> Option<PathBuf> {
    let o = git(path, &["rev-parse", "--git-common-dir"]).ok()?;
    if !o.status.success() {
        return None;
    }
    let d = PathBuf::from(stdout(&o));
    let joined = if d.is_absolute() { d } else { path.join(d) };
    Some(std::fs::canonicalize(&joined).unwrap_or(joined))
}

/// Create (or reuse) the isolated worktree on `branch`, based on the repo's
/// `base` (resolved against the repo, never the worktree branch).
pub fn create(repo: &Path, worktree: &Path, branch: &str, base: &str) -> Result<Value> {
    if let Some(parent) = worktree.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let resolved = git(repo, &["rev-parse", base])?;
    let base_sha = if resolved.status.success() {
        stdout(&resolved)
    } else {
        base.to_string()
    };
    let wt = worktree.to_string_lossy().to_string();

    if worktree.exists() {
        let same_repo = common_dir(worktree) == common_dir(repo);
        let cur = git(worktree, &["rev-parse", "--abbrev-ref", "HEAD"])?;
        if same_repo && cur.status.success() && stdout(&cur) == branch {
            // Reuse so the (expensive) build dir stays warm.
            let _ = git(worktree, &["reset", "--hard", &base_sha]);
            let _ = git(worktree, &["clean", "-fd"]);
            return Ok(json!({
                "ok": true, "worktree": wt, "branch": branch,
                "base": base_sha, "head": base_sha, "reused": true,
            }));
        }
        // Stale worktree (wrong repo or branch): remove and recreate.
        let old_common = common_dir(worktree);
        let _ = std::fs::remove_dir_all(worktree);
        if let Some(c) = old_common.and_then(|c| c.parent().map(Path::to_path_buf)) {
            let _ = git(&c, &["worktree", "prune"]);
        }
        let _ = git(repo, &["worktree", "prune"]);
    }

    let o = git(repo, &["worktree", "add", "-B", branch, &wt, &base_sha])?;
    if !o.status.success() {
        bail!(
            "git worktree add failed: {}",
            tail(String::from_utf8_lossy(&o.stderr).as_ref(), 600)
        );
    }
    Ok(json!({"ok": true, "worktree": wt, "branch": branch, "base": base_sha, "head": base_sha}))
}

/// Discard every edit and untracked file (each candidate starts clean).
pub fn reset(worktree: &Path, base: &str) -> Result<Value> {
    let o = git(worktree, &["reset", "--hard", base])?;
    if !o.status.success() {
        bail!("git reset failed: {}", tail(String::from_utf8_lossy(&o.stderr).as_ref(), 600));
    }
    let _ = git(worktree, &["clean", "-fd"]);
    Ok(json!({"ok": true, "reset_to": base}))
}

/// Hard-reset to `r` (a candidate sha or tag).
pub fn revert(worktree: &Path, r: &str) -> Result<Value> {
    let o = git(worktree, &["reset", "--hard", r])?;
    if !o.status.success() {
        bail!(
            "git reset failed: {}",
            tail(
                if o.stderr.is_empty() { String::from_utf8_lossy(&o.stdout) } else { String::from_utf8_lossy(&o.stderr) }
                    .as_ref(),
                600
            )
        );
    }
    let head = stdout(&git(worktree, &["rev-parse", "HEAD"])?);
    Ok(json!({"ok": true, "reverted_to": r, "head": head}))
}

/// Commit the current worktree state; optionally tag it so it stays reachable
/// after the branch is reset back to base.
pub fn commit(worktree: &Path, message: &str, tag: Option<&str>) -> Result<Value> {
    let _ = git(worktree, &["add", "-A"]);
    let o = Command::new("git")
        .arg("-C")
        .arg(worktree)
        .args([
            "-c",
            "user.name=kernelopt",
            "-c",
            "user.email=kernelopt@localhost",
            "commit",
            "-m",
            message,
        ])
        .output()
        .context("git commit")?;
    if !o.status.success() {
        let msg = if o.stderr.is_empty() {
            String::from_utf8_lossy(&o.stdout).to_string()
        } else {
            String::from_utf8_lossy(&o.stderr).to_string()
        };
        bail!("git commit failed: {}", tail(msg.trim(), 600));
    }
    let sha = stdout(&git(worktree, &["rev-parse", "HEAD"])?);
    if let Some(t) = tag {
        let _ = git(worktree, &["tag", "-f", t, &sha]);
    }
    Ok(json!({"ok": true, "sha": sha, "tag": tag, "message": message}))
}

/// `{dirty, porcelain}` for the worktree.
pub fn status(worktree: &Path) -> Result<Value> {
    let o = git(worktree, &["status", "--porcelain"])?;
    let porcelain = String::from_utf8_lossy(&o.stdout).to_string();
    Ok(json!({"ok": true, "dirty": !porcelain.trim().is_empty(), "porcelain": porcelain}))
}

/// Unified diff + numstat of the worktree vs `base`.
pub fn diff(worktree: &Path, base: &str, paths: &[String]) -> Result<Value> {
    let mut d = Command::new("git");
    d.arg("-C").arg(worktree).args(["--no-pager", "diff", base]);
    let mut s = Command::new("git");
    s.arg("-C")
        .arg(worktree)
        .args(["--no-pager", "diff", "--numstat", base]);
    if !paths.is_empty() {
        d.arg("--").args(paths);
        s.arg("--").args(paths);
    }
    let od = d.output().context("git diff")?;
    if !od.status.success() {
        bail!("git diff failed: {}", tail(String::from_utf8_lossy(&od.stderr).as_ref(), 600));
    }
    let os = s.output().context("git diff --numstat")?;
    let numstat = String::from_utf8_lossy(&os.stdout).to_string();
    let stat = crate::parse::parse_numstat(&numstat);
    let mut v = json!({
        "ok": true,
        "diff": String::from_utf8_lossy(&od.stdout),
        "base": base,
        "numstat_raw": numstat,
    });
    if let (Some(dst), Some(src)) = (v.as_object_mut(), stat.as_object()) {
        for (k, val) in src {
            dst.insert(k.clone(), val.clone());
        }
    }
    Ok(v)
}

/// Apply a unified diff to the worktree (`git apply`, falling back to `patch`).
pub fn apply_patch(worktree: &Path, patch: &str) -> Result<Value> {
    let patch = if patch.ends_with('\n') {
        patch.to_string()
    } else {
        format!("{patch}\n")
    };
    let run = |cmd: &mut Command, input: &str| -> Result<Output> {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(input.as_bytes())?;
        Ok(child.wait_with_output()?)
    };
    let mut g = Command::new("git");
    g.arg("-C")
        .arg(worktree)
        .args(["apply", "--whitespace=nowarn", "--recount", "-"]);
    let o = run(&mut g, &patch)?;
    if o.status.success() {
        return Ok(json!({"ok": true, "applied": true}));
    }
    let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
    let mut p = Command::new("patch");
    p.arg("-p1").arg("-d").arg(worktree);
    let o2 = run(&mut p, &patch)?;
    if !o2.status.success() {
        let msg = if err.is_empty() { "git apply failed".to_string() } else { err };
        bail!("patch_apply: {}", tail(&msg, 1000));
    }
    Ok(json!({"ok": true, "applied": true}))
}
