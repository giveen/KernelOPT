"""Tests for the Graphsignal runner module (no network, no GPU)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "runner")
sys.path.insert(0, RUNNER)

from kernelopt_runner import graphsignal  # noqa: E402


def _fake_run(tmp_path, name="graphsignal-run"):
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    run = bindir / name
    run.write_text("#!/bin/sh\necho 'graphsignal-run test'\n")
    run.chmod(0o755)
    return str(run)


def test_explicit_env_wins(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    monkeypatch.setenv("GRAPHSIGNAL_RUN", run)
    assert graphsignal.find_graphsignal_run({"managed_dir": str(tmp_path)}) == run


def test_find_returns_existing_path(tmp_path):
    _fake_run(tmp_path)
    found = graphsignal.find_graphsignal_run({"managed_dir": str(tmp_path)})
    assert found and os.path.exists(found)


def test_setup_is_idempotent_when_present(tmp_path):
    _fake_run(tmp_path)
    res = graphsignal.setup({"managed_dir": str(tmp_path), "cuda": "13"})
    assert res["ok"] is True
    assert res.get("already") is True
    assert res["cuda"] == "13"


def test_resolve_source_defaults_to_fork():
    src = graphsignal.resolve_source({})
    assert "graphsignal" in src and "@" in src  # git URL pinned to a ref


def test_resolve_source_pypi_sentinel():
    assert graphsignal.resolve_source({"source": "pypi"}) is None


def test_resolve_source_vendored(tmp_path):
    third = tmp_path / "third_party" / "graphsignal"
    third.mkdir(parents=True)
    (third / "pyproject.toml").write_text("[project]\nname='graphsignal'\n")
    managed = tmp_path / ".kernelopt" / "graphsignal"
    src = graphsignal.resolve_source({"managed_dir": str(managed)})
    assert src == str(third)


def test_managed_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELOPT_GRAPH_SIGNAL_DIR", str(tmp_path))
    assert graphsignal.managed_root() == str(tmp_path)


def test_detect_cuda_major_does_not_crash():
    v = graphsignal.detect_cuda_major()
    assert v is None or v.isdigit()
