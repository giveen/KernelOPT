"""Tests for the kernelopt_runner protocol commands (requires CUDA)."""
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(ROOT, "examples")
RUNNER = os.path.join(ROOT, "runner")

REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


@REQUIRES_CUDA
def test_trace_mlp_classifies_kernels():
    resp = runner_cmd({"command": "trace", "model_file": os.path.join(EXAMPLES, "mlp.py")})
    assert resp["ok"] is True
    assert resp["classification"] == "optimizable"
    classes = {k["class"] for k in resp["sub_kernels"]}
    assert "triton" in classes and "extern" in classes


@REQUIRES_CUDA
def test_bench_mlp_returns_latency():
    resp = runner_cmd({"command": "bench", "model_file": os.path.join(EXAMPLES, "mlp.py")})
    assert resp["ok"] is True
    assert resp["mean_ms"] > 0


@REQUIRES_CUDA
def test_verify_identity_candidate():
    """A candidate that is the model itself (wrapped as kernel_function) must pass."""
    model_path = os.path.join(EXAMPLES, "mlp.py")
    with open(model_path) as f:
        src = f.read()
    candidate = src + (
        "\n\n\n_BASE = None\n"
        "def init_with_model(m):\n"
        "    global _BASE\n"
        "    _BASE = m\n"
        "def kernel_function(x):\n"
        "    return _BASE(x)\n"
    )
    resp = runner_cmd(
        {
            "command": "verify",
            "model_file": model_path,
            "candidate_code": candidate,
            "seeds": [0, 1],
        }
    )
    assert resp["ok"] is True
    assert resp["gate1"]["passed"] is True
    assert resp["gate2"]["passed"] is True
    assert resp["all_passed"] is True


@REQUIRES_CUDA
def test_verify_broken_candidate_fails_gate1():
    model_path = os.path.join(EXAMPLES, "mlp.py")
    candidate = "def kernel_function(x):\n    return undefined_thing(x)\n"
    resp = runner_cmd(
        {"command": "verify", "model_file": model_path, "candidate_code": candidate, "seeds": [0]}
    )
    assert resp["all_passed"] is False
    assert resp["gate1"]["passed"] is False
    assert resp["gate1"]["error"]["kind"] in ("compile", "runtime")


def test_unknown_command_is_protocol_error():
    resp = runner_cmd({"command": "nope"})
    assert resp["ok"] is False
    assert resp["error"]["kind"] == "protocol"
