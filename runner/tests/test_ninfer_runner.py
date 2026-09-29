"""Tests for ninfer-mode runner commands.

Parsers are pure and always run. The subprocess tests are skipped unless a real
ninfer build is present, so this file is safe in CI without CUDA.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "runner")

sys.path.insert(0, RUNNER)

from kernelopt_runner import ninfer  # noqa: E402

NINFER = os.environ.get("NINFER_REPO")
NINFER_BUILD = os.path.join(NINFER, "build") if NINFER else None
ADD_BIAS_BENCH = (
    os.path.join(NINFER_BUILD, "bench", "ninfer_add_bias_bench") if NINFER_BUILD else None
)

needs_build = pytest.mark.skipif(
    not ADD_BIAS_BENCH or not os.path.exists(ADD_BIAS_BENCH),
    reason="set NINFER_REPO to a built ninfer checkout",
)
needs_repo = pytest.mark.skipif(
    not NINFER or not os.path.isdir(os.path.join(NINFER, "src", "ops")),
    reason="set NINFER_REPO to a ninfer checkout",
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
        timeout=300,
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #

NVCC_SAMPLE = """
/opt/ninfer/src/ops/kernel/add_bias.cuh(42): error: identifier "foo" is undefined
          ^
/opt/ninfer/src/ops/kernel/add_bias.cuh(42): error: identifier "foo" is undefined
2 errors detected in the compilation of "add_bias.cu".
"""


def test_parse_compiler_errors_dedupes():
    errs = ninfer.parse_compiler_errors(NVCC_SAMPLE)
    assert len(errs) == 1
    assert errs[0]["file"].endswith("add_bias.cuh")
    assert errs[0]["line"] == 42
    assert errs[0]["severity"] == "error"


CTEST_SAMPLE = """
Test project <build-dir>
    Start 51: ninfer_add_bias_test
1/3 Test #51: ninfer_add_bias_test .............   Passed    1.45 sec
    Start 52: ninfer_linear_add_fp8_test
2/3 Test #52: ninfer_linear_add_fp8_test ......***Failed    2.10 sec
    Start 53: ninfer_linear_add_bf16_test
3/3 Test #53: ninfer_linear_add_bf16_test .....   Passed    1.10 sec

75% tests passed, 1 tests failed out of 3
"""


def test_parse_ctest_output_finds_failure():
    parsed = ninfer.parse_ctest_output(CTEST_SAMPLE)
    names = {c["name"]: c["status"] for c in parsed["cases"]}
    assert names["ninfer_add_bias_test"] == "passed"
    assert names["ninfer_linear_add_fp8_test"] == "failed"
    assert parsed["failing_cases"] == [{"name": "ninfer_linear_add_fp8_test", "status": "failed"}]


BF16_CSV = """route,T,median_us,min_us,p95_us,effective_gbs
a,1,43.008,42.432,43.008,1463.62
b,8,45.056,43.008,45.056,1402.18
"""


def test_parse_bench_csv_aggregates_median():
    parsed = ninfer.parse_bench_csv(BF16_CSV)
    assert parsed["row_count"] == 2
    assert parsed["median_us"] == pytest.approx((43.008 + 45.056) / 2)
    assert parsed["best_median_us"] == 43.008
    assert parsed["rows"][0]["route"] == "a"


ADD_BIAS_STDOUT = """
add_bias [1152,4096 ]            median=    4.94 us  min=    4.91 us  p95=    4.96 us    3821.5 GB/s  (213.3% of 1792 GB/s roofline)
add_bias [4608,16384]            median=  181.08 us  min=  179.70 us  p95=  182.79 us    1667.7 GB/s  (93.1% of 1792 GB/s roofline)
"""


def test_parse_bench_stdout_fallback():
    parsed = ninfer.parse_bench_stdout(ADD_BIAS_STDOUT)
    assert parsed["row_count"] == 2
    assert parsed["median_us"] == pytest.approx((4.94 + 181.08) / 2)
    assert parsed["rows"][0]["effective_gbs"] == 3821.5


def test_parse_bench_stdout_extracts_roofline():
    parsed = ninfer.parse_bench_stdout(ADD_BIAS_STDOUT)
    assert parsed["rows"][0]["roofline_gbs"] == pytest.approx(1792.0)


def test_merge_bench_runs_carries_bandwidth():
    runs = [
        {
            "rows": [
                {"median_us": 10.0, "label": "a", "effective_gbs": 600.0, "roofline_gbs": 1792.0}
            ]
        }
    ]
    m = ninfer._merge_bench_runs(runs)
    assert m["representative_gbs"] == pytest.approx(600.0)
    assert m["representative_roofline_gbs"] == pytest.approx(1792.0)


def test_merge_bench_runs_uses_slowest_shape_and_noise():
    import statistics

    runs = [
        {"rows": [{"median_us": 10.0, "line": "small"}, {"median_us": 100.0, "line": "big"}]},
        {"rows": [{"median_us": 10.2, "line": "small"}, {"median_us": 101.0, "line": "big"}]},
    ]
    m = ninfer._merge_bench_runs(runs)
    # representative = slowest shape's median across repeats (not the cross-shape median)
    assert m["representative_us"] == pytest.approx(100.5)
    assert m["median_us"] == pytest.approx(statistics.median([10.1, 100.5]))
    assert m["noise_pct"] is not None and m["noise_pct"] >= 0.0


def test_merge_bench_runs_shape_filter():
    runs = [
        {"rows": [{"median_us": 10.0, "label": "add_bias [1152,8]"}, {"median_us": 100.0, "label": "add_bias [1152,4096]"}]}
    ]
    m = ninfer._merge_bench_runs(runs, shape_filter="1152,8")
    assert m["representative_label"] == "add_bias [1152,8]"
    assert m["representative_us"] == 10.0
    # No match -> fall back to the slowest shape.
    m2 = ninfer._merge_bench_runs(runs, shape_filter="does-not-exist")
    assert m2["representative_us"] == 100.0


def test_parse_numstat():
    text = "3\t1\tsrc/ops/linear_add/fp8/fp8_linear_add_a8.cu\n-\t-\tbin/thing\n"
    s = ninfer.parse_numstat(text)
    assert s["file_count"] == 2
    assert s["insertions"] == 3
    assert s["deletions"] == 1


# --------------------------------------------------------------------------- #
# protocol dispatch
# --------------------------------------------------------------------------- #

def test_unknown_cuda_command_is_error():
    resp = runner_cmd({"command": "cuda_nope"})
    assert resp["ok"] is False
    assert resp["error"]["kind"] in ("runtime", "protocol")


# --------------------------------------------------------------------------- #
# real-build integration (skipped without a ninfer build)
# --------------------------------------------------------------------------- #

@needs_build
def test_cuda_bench_add_bias_real():
    resp = runner_cmd(
        {
            "command": "cuda_bench",
            "build_dir": NINFER_BUILD,
            "binary": "ninfer_add_bias_bench",
            "args": ["--d", "1152", "--columns", "4096"],
        }
    )
    assert resp["ok"] is True, resp
    assert resp["median_us"] and resp["median_us"] > 0
    assert resp["row_count"] >= 1


@needs_build
def test_cuda_bench_same_binary_is_stable():
    """Negative control: re-benching an unchanged binary must NOT show a win.

    Two independent measurements of the same binary must agree within a few
    percent — that agreement is the noise floor the perf gate relies on.
    """

    def bench():
        return runner_cmd(
            {
                "command": "cuda_bench",
                "build_dir": NINFER_BUILD,
                "binary": "ninfer_add_bias_bench",
                "args": ["--d", "1152", "--columns", "4096"],
                "repeats": 3,
            }
        )

    a, b = bench(), bench()
    assert a["ok"] and b["ok"], (a, b)
    rel = abs(a["median_us"] - b["median_us"]) / a["median_us"]
    assert rel < 0.05, f"same binary drifted {rel:.1%}: {a['median_us']} vs {b['median_us']}"


@needs_build
def test_cuda_bench_detects_a_real_difference():
    """Positive control: the bench can distinguish a much slower config.

    A 16x larger column count must be measurably slower; this proves the
    measurement pipeline can detect a real performance difference, so a
    `matched` verdict is a finding and not a blind spot.
    """

    def bench(columns):
        return runner_cmd(
            {
                "command": "cuda_bench",
                "build_dir": NINFER_BUILD,
                "binary": "ninfer_add_bias_bench",
                "args": ["--d", "1152", "--columns", str(columns)],
                "repeats": 3,
            }
        )

    small, big = bench(4096), bench(65536)
    assert small["ok"] and big["ok"], (small, big)
    assert big["median_us"] > small["median_us"] * 1.5, (
        small["median_us"],
        big["median_us"],
    )


@needs_build
def test_cuda_verify_add_bias_real():
    resp = runner_cmd(
        {
            "command": "cuda_verify",
            "build_dir": NINFER_BUILD,
            "tests": ["ninfer_add_bias_test"],
        }
    )
    assert resp["ok"] is True, resp
    assert resp["passed"] is True, resp
    assert any(c["name"] == "ninfer_add_bias_test" for c in resp["cases"])


@needs_build
def test_cuda_diff_real_repo():
    resp = runner_cmd(
        {"command": "cuda_diff", "worktree": NINFER, "base": "HEAD"}
    )
    assert resp["ok"] is True, resp
    assert "diff" in resp


@needs_repo
def test_cuda_worktree_create_reuse_remove(tmp_path):
    wt = str(tmp_path / "wt")
    branch = "kernelopt/test_reuse"
    create = {
        "command": "cuda_worktree",
        "repo": NINFER,
        "worktree_dir": wt,
        "branch": branch,
        "action": "create",
    }
    first = runner_cmd(create)
    assert first["ok"] is True, first
    assert not first.get("reused")
    second = runner_cmd(create)
    assert second["ok"] is True, second
    assert second.get("reused") is True
    assert runner_cmd({"command": "cuda_worktree", "worktree_dir": wt,
                       "base": "HEAD", "action": "reset"})["ok"] is True
    # `worktree` is accepted as an alias for `worktree_dir`.
    assert runner_cmd({"command": "cuda_worktree", "worktree": wt,
                       "base": "HEAD", "action": "reset"})["ok"] is True
    removed = runner_cmd({"command": "cuda_worktree", "repo": NINFER,
                          "worktree_dir": wt, "branch": branch, "action": "remove"})
    assert removed["ok"] is True, removed
