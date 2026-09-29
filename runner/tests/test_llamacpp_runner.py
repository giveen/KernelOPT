"""Tests for llama.cpp-mode runner commands.

Parsers run always; subprocess tests skip unless a llama.cpp build is present.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "runner")
sys.path.insert(0, RUNNER)

from kernelopt_runner import llamacpp  # noqa: E402

LLAMACPP = os.environ.get("LLAMACPP_REPO")
LLAMACPP_BUILD = os.path.join(LLAMACPP, "build") if LLAMACPP else None
TEST_BACKEND_OPS = (
    os.path.join(LLAMACPP_BUILD, "bin", "test-backend-ops") if LLAMACPP_BUILD else None
)

needs_build = pytest.mark.skipif(
    not TEST_BACKEND_OPS or not os.path.exists(TEST_BACKEND_OPS),
    reason="set LLAMACPP_REPO to a built llama.cpp checkout",
)


def runner_cmd(request: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = RUNNER + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "kernelopt_runner"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=RUNNER,
        env=env,
        timeout=600,
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    return json.loads(proc.stdout)


VERIFY_SAMPLE = """
  RMS_NORM(type=f32,ne=[64,5,4,3]): compare failed
  213/214 tests passed
  Backend CUDA0: FAIL
"""


def test_parse_verify_output_counts_and_failures():
    parsed = llamacpp.parse_verify_output(VERIFY_SAMPLE)
    assert parsed["tests_passed"] == 213
    assert parsed["tests_total"] == 214
    assert parsed["ok_counts"] is False
    assert any("compare failed" in f["name"] for f in parsed["failing_cases"])


PERF_SAMPLE = """
  SOFT_MAX(type=f32,ne=[4096,4096,5,1],mask=0):  2080 runs -   480.81 us/run - 1312.40 GB/s
  SOFT_MAX(type=f32,ne=[77,4096,5,1],mask=0):   59928 runs -    17.33 us/run -  678.14 GB/s
"""


def test_parse_bench_stdout_aggregates():
    parsed = llamacpp.parse_bench_stdout(PERF_SAMPLE)
    assert parsed["row_count"] == 2
    assert parsed["median_us"] == pytest.approx((480.81 + 17.33) / 2)
    assert parsed["best_median_us"] == 17.33
    assert parsed["rows"][0]["runs"] == 2080


def test_unknown_llama_command_is_error():
    resp = runner_cmd({"command": "llama_nope"})
    assert resp["ok"] is False
    assert resp["error"]["kind"] in ("runtime", "protocol")


@needs_build
def test_llama_verify_soft_max_real():
    resp = runner_cmd(
        {"command": "llama_verify", "build_dir": LLAMACPP_BUILD, "ops": ["SOFT_MAX"]}
    )
    assert resp["ok"] is True, resp
    assert resp["passed"] is True, resp
    assert resp["tests_passed"] == resp["tests_total"] > 0


@needs_build
def test_llama_bench_soft_max_real():
    resp = runner_cmd(
        {"command": "llama_bench", "build_dir": LLAMACPP_BUILD, "ops": ["SOFT_MAX"]}
    )
    assert resp["ok"] is True, resp
    assert resp["median_us"] and resp["median_us"] > 0
    assert resp["row_count"] >= 1
