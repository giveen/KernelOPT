"""Tests for the worktree git actions (commit/log/reset/revert)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "runner"))

from kernelopt_runner import ninfer  # noqa: E402


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_commit_log_reset_revert(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "tester", cwd=repo)
    (repo / "k.cu").write_text("base\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    wt = str(tmp_path / "wt")
    created = ninfer.cuda_worktree(
        {"repo": str(repo), "worktree_dir": wt, "branch": "kernelopt/t", "action": "create"}
    )
    assert created["ok"] is True
    base = created["head"]

    # edit + commit + tag
    with open(os.path.join(wt, "k.cu"), "w") as f:
        f.write("base\nchange\n")
    commit = ninfer.cuda_worktree(
        {"worktree_dir": wt, "action": "commit", "message": "cand i0", "tag": "kernelopt/t/i0"}
    )
    assert commit["ok"] is True and commit["sha"]

    # log reads it
    log = ninfer.cuda_worktree({"worktree_dir": wt, "action": "log", "max": 5})
    assert any(c["subject"] == "cand i0" for c in log["commits"])

    # reset returns to base (branch stays attached)
    ninfer.cuda_worktree({"worktree_dir": wt, "base": base, "action": "reset"})
    assert open(os.path.join(wt, "k.cu")).read() == "base\n"

    # revert restores the tagged candidate
    rev = ninfer.cuda_worktree({"worktree_dir": wt, "action": "revert", "ref": "kernelopt/t/i0"})
    assert rev["ok"] is True
    assert "change" in open(os.path.join(wt, "k.cu")).read()

    ninfer.cuda_worktree(
        {"repo": str(repo), "worktree_dir": wt, "branch": "kernelopt/t", "action": "remove"}
    )


def test_create_recreates_when_repo_changes(tmp_path):
    def make_repo(name, content):
        r = tmp_path / name
        r.mkdir()
        _git("init", "-q", cwd=r)
        _git("config", "user.email", "t@example.com", cwd=r)
        _git("config", "user.name", "tester", cwd=r)
        (r / "k.cu").write_text(content)
        _git("add", "-A", cwd=r)
        _git("commit", "-q", "-m", "base", cwd=r)
        head = subprocess.run(
            ["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        return r, head

    repo_a, head_a = make_repo("repoA", "A\n")
    repo_b, head_b = make_repo("repoB", "B\n")
    wt = str(tmp_path / "wt3")

    r1 = ninfer.cuda_worktree(
        {"repo": str(repo_a), "worktree_dir": wt, "branch": "kernelopt/x", "action": "create"}
    )
    assert r1["head"] == head_a
    # Same path + branch, different repo -> must recreate from repo B.
    r2 = ninfer.cuda_worktree(
        {"repo": str(repo_b), "worktree_dir": wt, "branch": "kernelopt/x", "action": "create"}
    )
    assert r2["head"] == head_b, r2
    assert open(os.path.join(wt, "k.cu")).read() == "B\n"


def test_apply_patch(tmp_path):
    repo = tmp_path / "repo2"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "tester", cwd=repo)
    (repo / "k.cu").write_text("line1\nline2\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    wt = str(tmp_path / "wt2")
    ninfer.cuda_worktree(
        {"repo": str(repo), "worktree_dir": wt, "branch": "kernelopt/t2", "action": "create"}
    )
    patch = (
        "diff --git a/k.cu b/k.cu\n"
        "--- a/k.cu\n"
        "+++ b/k.cu\n"
        "@@ -1,2 +1,3 @@\n"
        " line1\n"
        "+added\n"
        " line2\n"
    )
    ok = ninfer.cuda_apply_patch({"worktree_dir": wt, "patch": patch})
    assert ok["ok"] is True and ok["applied"] is True
    assert "added" in open(os.path.join(wt, "k.cu")).read()

    bad = ninfer.cuda_apply_patch(
        {
            "worktree_dir": wt,
            "patch": "diff --git a/nope.cu b/nope.cu\n--- a/nope.cu\n+++ b/nope.cu\n@@ -1 +1 @@\n-x\n+y\n",
        }
    )
    assert bad["ok"] is False
    assert bad["error"]["kind"] == "patch_apply"

    ninfer.cuda_worktree(
        {"repo": str(repo), "worktree_dir": wt, "branch": "kernelopt/t2", "action": "remove"}
    )
