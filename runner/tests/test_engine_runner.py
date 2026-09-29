"""Tests for the engine (model-level) E2E runner commands."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "runner")
sys.path.insert(0, RUNNER)

from kernelopt_runner import engine  # noqa: E402


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
        timeout=120,
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    return json.loads(proc.stdout)


PPL_SAMPLE = """
domain                            tokens        mean_nll             ppl
custom                                20        2.406996       11.100563
overall                               20        2.406996       11.100563
"""


def test_perplexity_regex():
    m = engine._PPL_RE.search(PPL_SAMPLE)
    assert m and abs(float(m.group(1)) - 11.100563) < 1e-6


def test_detect_engine(tmp_path):
    assert engine.detect_engine("m.ninfer") == "ninfer"
    assert engine.detect_engine("m.gguf") == "llamacpp"
    d = tmp_path / "hf"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert engine.detect_engine(str(d)) == "hf"


def test_custom_cmd_is_deterministic():
    a = runner_cmd({"command": "engine_generate", "build_dir": "/tmp", "cmd": ["echo", "hello"], "max_new": 1})
    b = runner_cmd({"command": "engine_generate", "build_dir": "/tmp", "cmd": ["echo", "hello"], "max_new": 1})
    assert a["ok"] is True and a["text"] == "hello"
    assert a["digest"] == b["digest"]


def test_generate_missing_model_is_error():
    resp = runner_cmd({"command": "engine_generate", "build_dir": "/tmp", "model": "/tmp/nope.gguf"})
    assert resp["ok"] is False
    assert resp["error"]["kind"] == "runtime"


def test_generate_missing_weights_is_error():
    resp = runner_cmd(
        {"command": "engine_generate", "build_dir": "/tmp", "weights": "/tmp/nope.ninfer"}
    )
    assert resp["ok"] is False
    assert resp["error"]["kind"] == "runtime"


def test_unknown_engine_command_is_error():
    resp = runner_cmd({"command": "engine_nope"})
    assert resp["ok"] is False
    assert resp["error"]["kind"] in ("runtime", "protocol")
