"""ninfer-mode runner commands (protocol v1).

These drive a real ninfer CMake worktree: build a single op's targets, run the
op's own test suite, benchmark it, profile it, and surface a unified diff.
No GPU work happens in this module's parsers, so they are unit-testable without
CUDA; the subprocess wrappers run `cmake`/`ctest`/`ninfer_<op>_bench`/`ncu`/`git`.

Design authority: `docs/ninfer-mode.md` §3. Commands:

  cuda_worktree  — create/remove/reset the per-run git worktree (never the user's tree)
  cuda_compile   — Gate 1: cmake --build, structured nvcc/gcc diagnostics
  cuda_verify    — Gate 2: ctest the op's own suite, per-case failures
  cuda_bench     — Gate 4: run ninfer_<op>_bench, parse CSV or stdout
  cuda_ncu       — planner context: ncu --csv on the bench --profile path
  cuda_engine_bench — optional Gate 3: whole-engine ninfer_bench rate
  cuda_diff      — final artifact: git diff of the worktree vs base

Every command returns {"ok": true, ...} for a completed run (even when the gate
fails) and {"ok": false, "error": {...}} only for transport/tool failures.
"""
from __future__ import annotations

import csv
import io
import os
import re
import statistics
import subprocess

try:  # package import
    from .ncu import _extract_csv, parse_ncu_csv
except ImportError:  # pragma: no cover - direct script use
    from ncu import _extract_csv, parse_ncu_csv  # type: ignore

BUILD_DEFAULT_TIMEOUT = 3600
BENCH_DEFAULT_TIMEOUT = 1800
NCU_DEFAULT_TIMEOUT = 1800


def run(request: dict) -> dict:
    command = request.get("command")
    handlers = {
        "cuda_worktree": cuda_worktree,
        "cuda_compile": cuda_compile,
        "cuda_verify": cuda_verify,
        "cuda_bench": cuda_bench,
        "cuda_ncu": cuda_ncu,
        "cuda_engine_bench": cuda_engine_bench,
        "cuda_diff": cuda_diff,
        "cuda_apply_patch": cuda_apply_patch,
    }
    handler = handlers.get(command)
    if handler is None:
        raise ValueError(f"ninfer runner: unknown command {command!r}")
    return handler(request)


def cuda_apply_patch(request: dict) -> dict:
    """Apply a unified diff to the worktree (patch-mode editing).

    The Executor emits a `diff --git a/<path> b/<path>` patch; we apply it with
    `git apply` (fallback: `patch -p1`) so the model never has to reproduce the
    whole file.
    """
    import tempfile

    wt_arg = request.get("worktree_dir") or request.get("worktree")
    if not wt_arg:
        return _err("protocol", "cuda_apply_patch requires 'worktree_dir'")
    worktree = os.path.abspath(wt_arg)
    patch = request.get("patch") or ""
    if not patch.strip():
        return _err("protocol", "cuda_apply_patch requires a non-empty 'patch'")
    timeout = int(request.get("timeout_s", 120))

    fd, path = tempfile.mkstemp(suffix=".patch", prefix="kernelopt_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(patch)
            if not patch.endswith("\n"):
                f.write("\n")
        proc = _run(
            ["git", "-C", worktree, "apply", "--whitespace=nowarn", "--recount", path],
            None,
            timeout,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or "git apply failed"
            proc2 = _run(["patch", "-p1", "-d", worktree, "-i", path], None, timeout)
            if proc2.returncode != 0:
                return _err("patch_apply", err[-1000:])
        return {"ok": True, "applied": True}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _err(kind: str, message: str, traceback_text: str = "") -> dict:
    return {"ok": False, "error": {"kind": kind, "message": message, "traceback": traceback_text}}


def _run(cmd: list[str], cwd: str | None, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------- #
# cuda_worktree
# --------------------------------------------------------------------------- #

def _git_common_dir(path: str) -> str | None:
    """Absolute path of the repository's shared `.git` (or None)."""
    r = _run(["git", "-C", path, "rev-parse", "--git-common-dir"], None, 60)
    if r.returncode != 0:
        return None
    d = r.stdout.strip()
    return os.path.realpath(d if os.path.isabs(d) else os.path.join(path, d))


def cuda_worktree(request: dict) -> dict:
    """Manage the isolated git worktree. Actions: create | remove | reset | status.

    The user's own checkout is never modified: `create` adds a linked worktree at
    `worktree_dir` on a fresh branch; `remove` deletes both.
    """
    repo = os.path.abspath(request["repo"]) if request.get("repo") else None
    wt_arg = request.get("worktree_dir") or request.get("worktree")
    if not wt_arg:
        return _err("protocol", "cuda_worktree requires 'worktree_dir'")
    worktree = os.path.abspath(wt_arg)
    branch = request.get("branch", "kernelopt/run")
    base = request.get("base", "HEAD")
    action = request.get("action", "create")
    timeout = int(request.get("timeout_s", 600))

    if action in ("create", "remove") and repo is None:
        return _err("protocol", f"cuda_worktree action {action!r} requires 'repo'")

    if action == "create":
        os.makedirs(os.path.dirname(worktree), exist_ok=True)
        # Resolve the base against the REPO (the clean checkout), never the
        # worktree branch — otherwise a prior candidate commit on the branch
        # becomes the "base" and pollutes the baseline.
        resolved = _run(["git", "-C", repo, "rev-parse", base], None, timeout)
        base_sha = resolved.stdout.strip() if resolved.returncode == 0 else base
        # Reuse an existing worktree on the same branch so the (expensive) CMake
        # build dir stays warm; ninja only rebuilds files we actually touch.
        if os.path.exists(worktree):
            # A worktree belongs to exactly one repo. If the path was created
            # from a different repo (e.g. NINFER_REPO changed), its objects and
            # branches are unrelated — remove and recreate from THIS repo.
            same_repo = _git_common_dir(worktree) == _git_common_dir(repo)
            cur = _run(["git", "-C", worktree, "rev-parse", "--abbrev-ref", "HEAD"], None, timeout)
            if same_repo and cur.returncode == 0 and cur.stdout.strip() == branch:
                _run(["git", "-C", worktree, "reset", "--hard", base_sha], None, timeout)
                _run(["git", "-C", worktree, "clean", "-fd"], None, timeout)
                return {
                    "ok": True,
                    "worktree": worktree,
                    "branch": branch,
                    "base": base_sha,
                    "head": base_sha,
                    "reused": True,
                }
            # Stale worktree (wrong repo or branch): clean it out first.
            old_common = _git_common_dir(worktree)
            import shutil

            shutil.rmtree(worktree, ignore_errors=True)
            if old_common:
                old_repo = os.path.dirname(old_common)
                _run(["git", "-C", old_repo, "worktree", "prune"], None, timeout)
            _run(["git", "-C", repo, "worktree", "prune"], None, timeout)
        proc = _run(
            ["git", "-C", repo, "worktree", "add", "-B", branch, worktree, base_sha], None, timeout
        )
        if proc.returncode != 0:
            return _err("runtime", f"git worktree add failed: {proc.stderr.strip()[-600:]}")
        return {
            "ok": True,
            "worktree": worktree,
            "branch": branch,
            "base": base_sha,
            "head": base_sha,
        }

    if action == "remove":
        _run(["git", "-C", repo, "worktree", "remove", "--force", worktree], None, timeout)
        _run(["git", "-C", repo, "branch", "-D", branch], None, timeout)
        return {"ok": True, "removed": worktree, "branch": branch}

    if action == "reset":
        # Discard every edit and untracked file so each candidate starts clean.
        # `reset --hard` (not `checkout`) keeps the branch attached to `base`.
        proc = _run(["git", "-C", worktree, "reset", "--hard", base], None, timeout)
        if proc.returncode != 0:
            return _err("runtime", f"git reset failed: {proc.stderr.strip()[-600:]}")
        _run(["git", "-C", worktree, "clean", "-fd"], None, timeout)
        return {"ok": True, "worktree": worktree, "reset_to": base}

    if action == "commit":
        # Commit the current worktree state (a candidate) and optionally tag it so
        # it stays reachable after the branch is reset back to base.
        message = request.get("message", "kernelopt candidate")
        tag = request.get("tag")
        _run(["git", "-C", worktree, "add", "-A"], None, timeout)
        proc = _run(
            [
                "git", "-C", worktree,
                "-c", "user.name=kernelopt",
                "-c", "user.email=kernelopt@localhost",
                "commit", "-m", message,
            ],
            None, timeout,
        )
        if proc.returncode != 0:
            return _err("runtime", f"git commit failed: {(proc.stderr or proc.stdout).strip()[-600:]}")
        sha = _run(["git", "-C", worktree, "rev-parse", "HEAD"], None, timeout).stdout.strip()
        if tag:
            _run(["git", "-C", worktree, "tag", "-f", tag, sha], None, timeout)
        return {"ok": True, "sha": sha, "tag": tag, "message": message}

    if action == "log":
        n = int(request.get("max", 20))
        ref = request.get("ref", "HEAD")
        proc = _run(
            ["git", "-C", worktree, "log", "--oneline", "-n", str(n), ref], None, timeout
        )
        commits = []
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            commits.append({"sha": parts[0], "subject": parts[1] if len(parts) > 1 else ""})
        return {"ok": True, "commits": commits}

    if action == "revert":
        ref = request.get("ref")
        if not ref:
            return _err("protocol", "cuda_worktree revert requires 'ref' (sha or tag)")
        proc = _run(["git", "-C", worktree, "reset", "--hard", ref], None, timeout)
        if proc.returncode != 0:
            return _err("runtime", f"git reset failed: {(proc.stderr or proc.stdout).strip()[-600:]}")
        sha = _run(["git", "-C", worktree, "rev-parse", "HEAD"], None, timeout).stdout.strip()
        return {"ok": True, "worktree": worktree, "reverted_to": ref, "head": sha}

    if action == "status":
        proc = _run(["git", "-C", worktree, "status", "--porcelain"], None, timeout)
        return {"ok": True, "dirty": bool(proc.stdout.strip()), "porcelain": proc.stdout}

    return _err("protocol", f"unknown cuda_worktree action {action!r}")


# --------------------------------------------------------------------------- #
# cuda_compile (Gate 1)
# --------------------------------------------------------------------------- #

# nvcc:  path(line): error: msg        gcc/clang:  path:line:col: error: msg
_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n()]+?)"
    r"(?:\((?P<line_p>\d+)\)|:(?P<line_c>\d+)(?::(?P<col>\d+))?)"
    r":\s*(?P<sev>fatal error|error|warning)\s*:\s*(?P<msg>.*)$",
    re.MULTILINE,
)


def parse_compiler_errors(text: str, max_errors: int = 40, max_msg: int = 500) -> list[dict]:
    """Structured, de-duplicated diagnostics from nvcc/gcc stderr."""
    out: list[dict] = []
    seen: set[tuple] = set()
    for m in _DIAG_RE.finditer(text or ""):
        line = m.group("line_p") or m.group("line_c")
        col = m.group("col")
        key = (m.group("file"), line, col, m.group("msg"))
        if key in seen:
            continue
        seen.add(key)
        msg = m.group("msg").strip()
        if len(msg) > max_msg:
            msg = msg[:max_msg] + "…"
        out.append(
            {
                "file": m.group("file"),
                "line": int(line) if line else None,
                "column": int(col) if col else None,
                "severity": m.group("sev"),
                "message": msg,
            }
        )
        if len(out) >= max_errors:
            break
    return out


def cuda_compile(request: dict) -> dict:
    worktree = os.path.abspath(request["worktree"])
    build_dir = os.path.abspath(request["build_dir"])
    targets = list(request.get("targets") or [])
    jobs = int(request.get("jobs", os.cpu_count() or 4))
    timeout = int(request.get("timeout_s", BUILD_DEFAULT_TIMEOUT))

    if request.get("configure") and (
        request.get("force_configure")
        or not os.path.exists(os.path.join(build_dir, "CMakeCache.txt"))
    ):
        # If the build dir was configured for a different source tree (e.g.
        # NINFER_REPO changed), wipe it — cmake refuses to reuse a cache whose
        # CMAKE_HOME_DIRECTORY differs.
        cache = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.exists(cache):
            home = None
            try:
                with open(cache) as f:
                    for line in f:
                        if line.startswith("CMAKE_HOME_DIRECTORY:"):
                            home = line.split("=", 1)[1].strip()
                            break
            except OSError:
                home = None
            if home and os.path.realpath(home) != os.path.realpath(worktree):
                import shutil

                shutil.rmtree(build_dir, ignore_errors=True)

        configure_args = request.get("configure_args") or [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=ON",
            "-DNINFER_BUILD_BENCHMARKS=ON",
        ]
        cfg = _run(
            ["cmake", "-S", worktree, "-B", build_dir, "-G", "Ninja", *configure_args],
            None,
            timeout,
        )
        if cfg.returncode != 0:
            return {
                "ok": True,
                "passed": False,
                "stage": "configure",
                "compiler_errors": parse_compiler_errors(cfg.stderr),
                "raw_tail": (cfg.stdout + cfg.stderr)[-4000:],
            }

    cmd = ["cmake", "--build", build_dir, "-j", str(jobs)]
    if targets:
        cmd += ["--target", *targets]
    try:
        proc = _run(cmd, None, timeout)
    except subprocess.TimeoutExpired:
        return _err("timeout", f"cmake --build timed out after {timeout}s")

    combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
    errors = [e for e in parse_compiler_errors(combined) if e["severity"] != "warning"]
    return {
        "ok": True,
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "targets": targets,
        "compiler_errors": errors,
        "diagnostic_count": len(parse_compiler_errors(combined)),
        "raw_tail": combined[-4000:],
    }


# --------------------------------------------------------------------------- #
# cuda_verify (Gate 2)
# --------------------------------------------------------------------------- #

_TEST_LINE_RE = re.compile(
    r"Test\s+#\d+:\s*(?P<name>\S+)\s*\.*\s*(?P<status>Passed|Failed|\*\*\*Failed|\*\*\*Timeout|Skipped|Not Run)",
    re.MULTILINE,
)


def parse_ctest_output(text: str) -> dict:
    """Per-test status from ctest output; failures carry their output block."""
    cases: list[dict] = []
    for m in _TEST_LINE_RE.finditer(text or ""):
        status = m.group("status")
        cases.append(
            {
                "name": m.group("name"),
                "status": "failed" if "Failed" in status or "Timeout" in status else status.lower(),
            }
        )
    failing = [c for c in cases if c["status"] == "failed"]
    return {"cases": cases, "failing_cases": failing}


def cuda_verify(request: dict) -> dict:
    build_dir = os.path.abspath(request["build_dir"])
    tests = list(request.get("tests") or [])
    timeout = int(request.get("timeout_s", BUILD_DEFAULT_TIMEOUT))
    log_path = request.get("log_path")

    cmd = ["ctest", "--test-dir", build_dir, "--output-on-failure"]
    if tests:
        pattern = "^(" + "|".join(re.escape(t) for t in tests) + ")$"
        cmd += ["-R", pattern]
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        cmd += ["-O", log_path]

    try:
        proc = _run(cmd, None, timeout)
    except subprocess.TimeoutExpired:
        return _err("timeout", f"ctest timed out after {timeout}s")

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_ctest_output(combined)
    passed = proc.returncode == 0 and not parsed["failing_cases"]
    return {
        "ok": True,
        "passed": passed,
        "exit_code": proc.returncode,
        "tests": tests,
        "cases": parsed["cases"],
        "failing_cases": parsed["failing_cases"],
        "suite_output": combined[-8000:],
    }


# --------------------------------------------------------------------------- #
# cuda_bench (Gate 4)
# --------------------------------------------------------------------------- #

_MEDIAN_STDOUT_RE = re.compile(r"median\s*=\s*([0-9.]+)\s*us", re.IGNORECASE)
_GBS_STDOUT_RE = re.compile(r"([0-9.]+)\s*GB/s", re.IGNORECASE)


def parse_bench_csv(text: str) -> dict:
    """Parse any op-bench CSV: rows of floats + aggregate median across rows."""
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []
    median_key = _median_column(reader.fieldnames or [])
    for raw in reader:
        row: dict = {}
        for k, v in raw.items():
            if k is None:
                continue
            row[k.strip()] = _num(v)
        if row:
            row["label"] = _row_label(row)
            rows.append(row)
    medians = [r[median_key] for r in rows if median_key and _is_num(r.get(median_key))]
    return {
        "rows": rows,
        "columns": list(reader.fieldnames or []),
        "median_us": statistics.median(medians) if medians else None,
        "best_median_us": min(medians) if medians else None,
        "row_count": len(rows),
    }


def parse_bench_stdout(text: str) -> dict:
    """Fallback parser for benches without --csv-out (e.g. add_bias).

    Each matching line is one measurement; the aggregate median across the
    printed sweep is the scalar gate metric.
    """
    rows: list[dict] = []
    for line in (text or "").splitlines():
        m = _MEDIAN_STDOUT_RE.search(line)
        if not m:
            continue
        row = {
            "median_us": float(m.group(1)),
            "label": line.split("median=")[0].strip(),
            "line": line.strip(),
        }
        g = _GBS_STDOUT_RE.search(line)
        if g:
            row["effective_gbs"] = float(g.group(1))
        rows.append(row)
    medians = [r["median_us"] for r in rows]
    return {
        "rows": rows,
        "columns": ["median_us", "effective_gbs"],
        "median_us": statistics.median(medians) if medians else None,
        "best_median_us": min(medians) if medians else None,
        "row_count": len(rows),
    }


def _row_label(row: dict) -> str:
    """A human label for a CSV bench row (route/op/path + T when present)."""
    parts = [f"{k}={row[k]}" for k in ("route", "op", "path", "policy", "T") if k in row]
    return " ".join(parts) if parts else "row"


def _median_column(columns: list[str]) -> str | None:
    for c in columns:
        if c.strip() == "median_us":
            return c.strip()
    for c in columns:
        if "median" in c.lower() and "us" in c.lower():
            return c.strip()
    return None


def _is_num(v) -> bool:
    return isinstance(v, (int, float))


def _num(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s == "":
        return ""
    try:
        return float(s)
    except ValueError:
        return s


def bench_capabilities(binary_path: str, timeout: int = 60) -> set[str]:
    """Probe `--help` for supported flags. Cheap: help exits before GPU init."""
    try:
        proc = _run([binary_path, "--help"], None, timeout)
    except (subprocess.TimeoutExpired, OSError):
        return set()
    text = (proc.stdout or "") + (proc.stderr or "")
    caps = set()
    for flag in ("--csv-out", "--warmup", "--repeat", "--profile", "--t-sweep"):
        if flag in text:
            caps.add(flag)
    return caps


def cuda_bench(request: dict) -> dict:
    build_dir = os.path.abspath(request["build_dir"])
    binary = request["binary"]
    binary_path = binary if os.path.isabs(binary) else os.path.join(build_dir, "bench", binary)
    args = [str(a) for a in (request.get("args") or [])]
    warmup = request.get("warmup")
    repeat = request.get("repeat")
    csv_out = request.get("csv_out")
    timeout = int(request.get("timeout_s", BENCH_DEFAULT_TIMEOUT))

    if not os.path.exists(binary_path):
        return _err("runtime", f"bench binary not found: {binary_path}")

    caps = bench_capabilities(binary_path) if request.get("probe", True) else set()
    argv = [binary_path, *args]
    if csv_out and "--csv-out" in caps and "--csv-out" not in args:
        os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
        argv += ["--csv-out", csv_out]
    if warmup is not None and "--warmup" in caps:
        argv += ["--warmup", str(int(warmup))]
    if repeat is not None and "--repeat" in caps:
        argv += ["--repeat", str(int(repeat))]

    # Repeat the whole bench so a single noisy sample can't masquerade as a win.
    repeats = max(1, int(request.get("repeats", 1)))
    parsed_runs: list[dict] = []
    proc = None
    for _ in range(repeats):
        try:
            proc = _run(argv, None, timeout)
        except subprocess.TimeoutExpired:
            return _err("timeout", f"bench timed out after {timeout}s")
        stdout = proc.stdout or ""
        csv_text = ""
        used_csv = False
        if csv_out and os.path.exists(csv_out):
            with open(csv_out, "r") as f:
                csv_text = f.read()
            used_csv = bool(csv_text.strip())
        parsed = parse_bench_csv(csv_text) if used_csv else parse_bench_stdout(stdout)
        if parsed["rows"]:
            parsed_runs.append(parsed)

    if not parsed_runs:
        return {
            "ok": True,
            "passed": False,
            "error_kind": "no_measurements",
            "exit_code": proc.returncode if proc else None,
            "command": " ".join(argv),
            "stdout_tail": (proc.stdout or "")[-3000:] if proc else "",
            "stderr_tail": (proc.stderr or "")[-2000:] if proc else "",
            "rows": [],
            "median_us": None,
            "representative_us": None,
            "noise_pct": None,
            "row_count": 0,
        }
    merged = _merge_bench_runs(parsed_runs, request.get("shape_filter"))
    return {
        "ok": True,
        "passed": proc.returncode == 0 and merged["representative_us"] is not None,
        "exit_code": proc.returncode,
        "command": " ".join(argv),
        "used_csv": used_csv,
        "capabilities": sorted(caps),
        "csv_path": csv_out if used_csv else None,
        "repeats": repeats,
        "stdout_tail": (proc.stdout or "")[-2000:],
        **merged,
    }


def _merge_bench_runs(runs: list[dict], shape_filter: str | None = None) -> dict:
    """Aggregate repeated bench runs per shape (row index).

    The representative row is the one whose label contains `shape_filter` (when
    given), else the *slowest* shape — the least launch-overhead-dominated, most
    stable comparison point.
    `noise_pct` is the median relative spread across repeats (measurement noise).
    """
    n = min(len(r["rows"]) for r in runs)
    rows: list[dict] = []
    spreads: list[float] = []
    for i in range(n):
        per_run = [r["rows"][i].get("median_us") for r in runs]
        vals = [v for v in per_run if isinstance(v, (int, float))]
        if not vals:
            continue
        med = statistics.median(vals)
        row = {
            "median_us": med,
            "min_us": min(vals),
            "max_us": max(vals),
            "samples": vals,
            "label": runs[0]["rows"][i].get("label"),
            "line": runs[0]["rows"][i].get("line"),
        }
        if med and med > 0:
            spreads.append((max(vals) - min(vals)) / med)
        rows.append(row)
    medians = [r["median_us"] for r in rows]
    candidates = rows
    if shape_filter:
        f = shape_filter.lower()
        matching = [r for r in rows if f in (r.get("label") or "").lower()]
        if matching:
            candidates = matching
    representative = max(candidates, key=lambda r: r["median_us"]) if candidates else None
    return {
        "rows": rows,
        "row_count": len(rows),
        "median_us": statistics.median(medians) if medians else None,
        "representative_us": (representative or {}).get("median_us"),
        "representative_label": (representative or {}).get("label"),
        "representative_samples": (representative or {}).get("samples"),
        "noise_pct": statistics.median(spreads) if spreads else None,
    }


# --------------------------------------------------------------------------- #
# cuda_ncu (planner context)
# --------------------------------------------------------------------------- #

def cuda_ncu(request: dict) -> dict:
    from .ncu import find_ncu

    ncu = request.get("ncu_path") or find_ncu()
    if not ncu:
        return _err("runtime", "ncu binary not found")

    build_dir = os.path.abspath(request["build_dir"])
    if request.get("binary_path"):
        binary_path = request["binary_path"]
    else:
        binary = request["binary"]
        binary_path = binary if os.path.isabs(binary) else os.path.join(build_dir, "bench", binary)
    if not os.path.exists(binary_path):
        return _err("runtime", f"bench binary not found: {binary_path}")

    args = [str(a) for a in (request.get("args") or [])]
    ncu_set = request.get("ncu_set", "full")
    timeout = int(request.get("timeout_s", NCU_DEFAULT_TIMEOUT))
    report = request.get("report_path")

    cmd = [ncu, "--csv", f"--set={ncu_set}", "--target-processes=all"]
    if request.get("launch_skip"):
        cmd.append(f"--launch-skip={int(request['launch_skip'])}")
    if request.get("launch_count"):
        cmd.append(f"--launch-count={int(request['launch_count'])}")
    if request.get("kernel_name"):
        cmd.append(f"--kernel-name={request['kernel_name']}")
    caps = bench_capabilities(binary_path) if request.get("probe", True) else set()
    argv = [binary_path, *args]
    if "--profile" in caps and request.get("profile", True):
        argv.append("--profile")
    cmd += argv

    try:
        proc = _run(cmd, None, timeout)
    except subprocess.TimeoutExpired:
        return _err("timeout", f"ncu timed out after {timeout}s")

    csv_text = _extract_csv(proc.stdout or "")
    context = parse_ncu_csv(csv_text) if csv_text.strip() else {"kernels": [], "rules": []}
    if not context["kernels"]:
        combined = (proc.stdout or "") + (proc.stderr or "")
        if "ERR_NVGPUCTRPERM" in combined:
            return _err(
                "runtime",
                "ERR_NVGPUCTRPERM: GPU performance counters are admin-only "
                "(RmProfilingAdminOnly=1). Set NVreg_RmProfilingAdminOnly=0 and reload.",
            )
        return _err(
            "runtime",
            f"ncu produced no kernel data (exit {proc.returncode}): " + combined[-600:],
        )

    if report and csv_text.strip():
        os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
        with open(report, "w") as f:
            f.write(csv_text)
        context["raw_csv_path"] = report
    context["ncu_command"] = " ".join(cmd)
    return {"ok": True, "context": context}


# --------------------------------------------------------------------------- #
# cuda_engine_bench (optional Gate 3)
# --------------------------------------------------------------------------- #

def cuda_engine_bench(request: dict) -> dict:
    """Whole-engine `ninfer_bench` rate. Requires a built .ninfer artifact."""
    build_dir = os.path.abspath(request["build_dir"])
    weights = request.get("weights")
    if not weights or not os.path.exists(weights):
        return _err("runtime", f"engine weights not found: {weights}")
    binary_path = os.path.join(build_dir, "bench", request.get("binary", "ninfer_bench"))
    if not os.path.exists(binary_path):
        return _err("runtime", f"engine bench binary not found: {binary_path}")

    timeout = int(request.get("timeout_s", BENCH_DEFAULT_TIMEOUT))
    out_path = request.get("output_file")
    argv = [binary_path, "--weights", weights, "-o", "csv"]
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        argv += ["--output-file", out_path]
    for key, flag in (("n_prompt", "-p"), ("n_gen", "-n"), ("repetitions", "-r"), ("warmup", "--warmup")):
        if request.get(key) is not None:
            argv += [flag, str(request[key])]
    if request.get("no_cuda_graph"):
        argv.append("--no-cuda-graph")

    try:
        proc = _run(argv, None, timeout)
    except subprocess.TimeoutExpired:
        return _err("timeout", f"engine bench timed out after {timeout}s")

    text = ""
    if out_path and os.path.exists(out_path):
        with open(out_path, "r") as f:
            text = f.read()
    if not text.strip():
        text = proc.stdout or ""
    rows = _parse_csv_rows(text)
    return {
        "ok": True,
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "command": " ".join(argv),
        "rows": rows,
        "row_count": len(rows),
        "stdout_tail": (proc.stdout or "")[-2000:],
    }


def _parse_csv_rows(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    return [{(k or "").strip(): _num(v) for k, v in row.items() if k is not None} for row in reader]


# --------------------------------------------------------------------------- #
# cuda_diff (final artifact)
# --------------------------------------------------------------------------- #

_NUMSTAT_RE = re.compile(r"^(?P<ins>\d+|-)\t(?P<del>\d+|-)\t(?P<path>.+)$", re.MULTILINE)


def parse_numstat(text: str) -> dict:
    files, ins, dele = [], 0, 0
    for m in _NUMSTAT_RE.finditer(text or ""):
        files.append(m.group("path").strip())
        if m.group("ins") != "-":
            ins += int(m.group("ins"))
        if m.group("del") != "-":
            dele += int(m.group("del"))
    return {"files_changed": files, "file_count": len(files), "insertions": ins, "deletions": dele}


def cuda_diff(request: dict) -> dict:
    worktree = os.path.abspath(request["worktree"])
    base = request.get("base", "HEAD")
    paths = request.get("paths") or []
    timeout = int(request.get("timeout_s", 300))

    diff_cmd = ["git", "-C", worktree, "--no-pager", "diff", base]
    stat_cmd = ["git", "-C", worktree, "--no-pager", "diff", "--numstat", base]
    if paths:
        diff_cmd += ["--", *paths]
        stat_cmd += ["--", *paths]

    diff = _run(diff_cmd, None, timeout)
    stat = _run(stat_cmd, None, timeout)
    if diff.returncode != 0:
        return _err("runtime", f"git diff failed: {diff.stderr.strip()[-600:]}")
    summary = parse_numstat(stat.stdout)
    return {"ok": True, "diff": diff.stdout, "base": base, **summary}
