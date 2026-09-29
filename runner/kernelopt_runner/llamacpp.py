"""llama.cpp mode runner commands (protocol v1).

The optimization unit is a `ggml/src/ggml-cuda/*.cu` kernel file; the authority
is llama.cpp's own `test-backend-ops` binary:

  llama_verify — Gate 2: `test-backend-ops test -o <OP> -b CUDA0`
  llama_bench  — Gate 4: `test-backend-ops perf -o <OP> -b CUDA0`, parse us/run
  llama_ncu    — planner context: ncu on `test-backend-ops perf -o <OP>`

Build (Gate 1) reuses the generic `cuda_compile` (CMake); worktree/diff reuse
`cuda_worktree`/`cuda_diff`. Design authority: docs/ninfer-mode.md (mode mapping).
"""
from __future__ import annotations

import os
import re
import statistics

from .ninfer import _err, _run, _merge_bench_runs

DEFAULT_BACKEND = "CUDA0"
TEST_TIMEOUT = 1800
BENCH_TIMEOUT = 1800

_US_RE = re.compile(r"(\d+)\s+runs\s+-\s+([0-9.]+)\s+us/run")
_PASS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+tests passed")
_GBS_RE = re.compile(r"([0-9.]+)\s*GB/s")


def _binary(request: dict) -> str:
    if request.get("binary_path"):
        return request["binary_path"]
    build_dir = os.path.abspath(request["build_dir"])
    return os.path.join(build_dir, "bin", "test-backend-ops")


def _op_args(ops: list[str]) -> list[str]:
    return ["-o", ",".join(ops)] if ops else []


def run(request: dict) -> dict:
    command = request.get("command")
    handlers = {
        "llama_verify": llama_verify,
        "llama_bench": llama_bench,
        "llama_ncu": llama_ncu,
    }
    handler = handlers.get(command)
    if handler is None:
        raise ValueError(f"llamacpp runner: unknown command {command!r}")
    return handler(request)


def parse_verify_output(text: str) -> dict:
    """`test-backend-ops test` prints `N/M tests passed` per backend."""
    m = _PASS_RE.search(text or "")
    passed = total = None
    if m:
        passed, total = int(m.group(1)), int(m.group(2))
    failing: list[dict] = []
    for line in (text or "").splitlines():
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if "FAIL" in stripped or "compare failed" in stripped:
            failing.append({"name": stripped[:220]})
    return {
        "tests_passed": passed,
        "tests_total": total,
        "failing_cases": failing,
        "ok_counts": passed is not None and total is not None and passed == total,
    }


def llama_verify(request: dict) -> dict:
    binary = _binary(request)
    if not os.path.exists(binary):
        return _err("runtime", f"test-backend-ops not found: {binary}")
    backend = request.get("backend", DEFAULT_BACKEND)
    timeout = int(request.get("timeout_s", TEST_TIMEOUT))
    cmd = [binary, "test", "-b", backend, *_op_args(request.get("ops") or [])]
    try:
        proc = _run(cmd, None, timeout)
    except Exception as exc:  # subprocess.TimeoutExpired / OSError
        return _err("timeout", f"llama_verify failed: {exc}")

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_verify_output(text)
    passed = proc.returncode == 0 and parsed["ok_counts"] and not parsed["failing_cases"]
    return {
        "ok": True,
        "passed": passed,
        "exit_code": proc.returncode,
        "ops": request.get("ops") or [],
        "tests_passed": parsed["tests_passed"],
        "tests_total": parsed["tests_total"],
        "failing_cases": parsed["failing_cases"],
        "suite_output": text[-8000:],
    }


def parse_bench_stdout(text: str) -> dict:
    """Perf console lines: `<n> runs - <t> us/run - ...`."""
    rows: list[dict] = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        m = _US_RE.search(line)
        if not m:
            continue
        row = {
            "runs": int(m.group(1)),
            "us_per_run": float(m.group(2)),
            "median_us": float(m.group(2)),  # alias for the shared merge helper
            "line": line.strip()[:220],
        }
        g = _GBS_RE.search(line)
        if g:
            row["effective_gbs"] = float(g.group(1))
        rows.append(row)
    medians = [r["us_per_run"] for r in rows]
    return {
        "rows": rows,
        "median_us": statistics.median(medians) if medians else None,
        "best_median_us": min(medians) if medians else None,
        "row_count": len(rows),
    }


def llama_bench(request: dict) -> dict:
    binary = _binary(request)
    if not os.path.exists(binary):
        return _err("runtime", f"test-backend-ops not found: {binary}")
    backend = request.get("backend", DEFAULT_BACKEND)
    timeout = int(request.get("timeout_s", BENCH_TIMEOUT))
    args = [str(a) for a in (request.get("args") or [])]
    cmd = [binary, "perf", "-b", backend, *_op_args(request.get("ops") or []), *args]
    repeats = max(1, int(request.get("repeats", 1)))
    runs: list[dict] = []
    proc = None
    try:
        for _ in range(repeats):
            proc = _run(cmd, None, timeout)
            parsed = parse_bench_stdout(proc.stdout or "")
            if parsed["rows"]:
                runs.append(parsed)
    except Exception as exc:
        return _err("timeout", f"llama_bench failed: {exc}")

    stdout = (proc.stdout if proc else "") or ""
    if not runs:
        return {
            "ok": True,
            "passed": False,
            "error_kind": "no_measurements",
            "exit_code": proc.returncode if proc else None,
            "command": " ".join(cmd),
            "stdout_tail": stdout[-3000:],
            "rows": [],
            "median_us": None,
            "representative_us": None,
            "noise_pct": None,
            "row_count": 0,
        }
    merged = _merge_bench_runs(runs)
    return {
        "ok": True,
        "passed": proc.returncode == 0 and merged["representative_us"] is not None,
        "exit_code": proc.returncode,
        "command": " ".join(cmd),
        "repeats": repeats,
        "stdout_tail": stdout[-1500:],
        **merged,
    }


def llama_ncu(request: dict) -> dict:
    """Profile `test-backend-ops perf -o <OP>` with ncu (delegates to cuda_ncu)."""
    from .ninfer import cuda_ncu

    req = {
        "build_dir": request["build_dir"],
        "binary_path": _binary(request),
        "args": ["perf", "-b", request.get("backend", DEFAULT_BACKEND),
                 *_op_args(request.get("ops") or [])],
        "ncu_set": request.get("ncu_set", "basic"),
        "launch_skip": request.get("launch_skip", 3),
        "launch_count": request.get("launch_count", 1),
        "timeout_s": request.get("timeout_s", 1200),
        "report_path": request.get("report_path"),
        "probe": False,
    }
    return cuda_ncu(req)
