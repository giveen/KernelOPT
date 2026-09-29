"""`graphsignal_profile` — run a workload under the Graphsignal sidecar and
return its `/signals` payload.

Graphsignal (https://github.com/graphsignal/graphsignal) observes a GPU process
from a sidecar and serves everything it measures at `http://127.0.0.1:<port>/signals`
while the workload runs. We shell out to `graphsignal-run` (the installed CLI)
rather than importing the package, so the only dependency is the binary.

Request:
  cmd               — workload argv, e.g. ["ninfer_bench", "-o", "json", ...]
  listen_port       — /signals port (default 18259)
  cuda_graph_trace  — "node" | "graph" | "" (node = per-kernel inside graphs)
  cwd               — working directory for the workload
  settle_s          — initial wait before the first read (default 1.0)
  poll_s            — poll interval while running (default 1.0)
  timeout_s         — kill the workload after this many seconds (default 1800)
  min_metrics       — keep polling until the payload has at least this many
                      metrics (default 1), then settle_s more

Response:
  {ok, exit_code, signals, stdout_tail, stderr_tail, graphsignal_run, polls}
  `signals` is the last non-empty /signals payload ({} when nothing was read).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

DEFAULT_PORT = 18259

# Upstream PyPI graphsignal has no llama.cpp / NInfer launchers or recorders;
# this fork does. Override with KERNELOPT_GRAPH_SIGNAL_SOURCE or --source.
DEFAULT_SOURCE = "git+https://github.com/giveen/graphsignal@main"


def managed_root(request: dict | None = None) -> str:
    """KernelOPT-managed Graphsignal install (no user-level install needed)."""
    if request and request.get("managed_dir"):
        return str(request["managed_dir"])
    env = os.environ.get("KERNELOPT_GRAPH_SIGNAL_DIR")
    if env:
        return env
    return os.path.join(os.getcwd(), ".kernelopt", "graphsignal")


def managed_run(request: dict | None = None) -> str:
    return os.path.join(managed_root(request), "venv", "bin", "graphsignal-run")


def project_root(request: dict | None = None) -> str:
    # <root>/.kernelopt/graphsignal -> <root>
    return os.path.dirname(os.path.dirname(managed_root(request)))


def vendored_source(request: dict | None = None) -> str | None:
    """A `third_party/graphsignal` checkout at the project root, if present."""
    cand = os.path.join(project_root(request), "third_party", "graphsignal")
    return cand if os.path.isfile(os.path.join(cand, "pyproject.toml")) else None


def resolve_source(request: dict | None = None) -> str | None:
    """Where to install Graphsignal from.

    `--source pypi` forces upstream; otherwise: `--source` → vendored
    `third_party/graphsignal` → `KERNELOPT_GRAPH_SIGNAL_SOURCE` → the fork.
    """
    request = request or {}
    raw = request.get("source")
    if raw:
        return None if str(raw).strip().lower() == "pypi" else str(raw)
    vendored = vendored_source(request)
    if vendored:
        return vendored
    return os.environ.get("KERNELOPT_GRAPH_SIGNAL_SOURCE") or DEFAULT_SOURCE


def has_fork_features(run_path: str) -> bool | None:
    """True when the install has llama.cpp/NInfer launchers (the fork).

    None when it cannot be determined. Upstream installs return False.
    """
    try:
        with open(run_path) as f:
            shebang = f.readline().strip()
    except OSError:
        return None
    interp = shebang[2:].strip().split()[0] if shebang.startswith("#!") else sys.executable
    code = (
        "import importlib.util as u, sys;"
        "sys.exit(0 if u.find_spec('graphsignal.launchers.llama_launcher') else 3)"
    )
    try:
        p = subprocess.run([interp, "-c", code], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if p.returncode == 0:
        return True
    if p.returncode == 3:
        return False
    return None


def detect_cuda_major() -> str | None:
    """CUDA major version (12/13) from nvcc or CUDA_HOME/CUDA_PATH."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, timeout=20
            ).stdout
            m = re.search(r"release (\d+)\.", out)
            if m:
                return m.group(1)
        except Exception:  # noqa: BLE001
            pass
    root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if root:
        version_json = os.path.join(root, "version.json")
        if os.path.exists(version_json):
            try:
                v = json.load(open(version_json)).get("version")
                if v:
                    return str(v).split(".")[0]
            except Exception:  # noqa: BLE001
                pass
    return None


def find_graphsignal_run(request: dict | None = None) -> str | None:
    """Locate `graphsignal-run`, preferring an install with llama.cpp/NInfer
    launchers (the fork). An explicit `GRAPHSIGNAL_RUN` always wins."""
    for env in ("GRAPHSIGNAL_RUN", "GRAPHSIGNAL_RUN_PATH"):
        p = os.environ.get(env)
        if p and os.path.exists(p):
            return p
    candidates = [c for c in (managed_run(request), shutil.which("graphsignal-run")) if c and os.path.exists(c)]
    for c in candidates:
        if has_fork_features(c) is True:
            return c
    return candidates[0] if candidates else None


def setup(request: dict) -> dict:
    """Install Graphsignal into `.kernelopt/graphsignal/venv` (idempotent).

    Source precedence: `--source` (local checkout or git URL) → pinned PyPI
    `--version` → latest PyPI. The CUDA extra is chosen from `--cuda` or
    detected from the toolchain. Nothing is installed system-wide.
    """
    cuda = str(request.get("cuda") or detect_cuda_major() or "12")
    source = resolve_source(request)
    version = request.get("version")
    force = bool(request.get("force"))
    timeout = int(request.get("timeout_s", 1800))
    root = managed_root(request)
    run_path = managed_run(request)

    if os.path.exists(run_path) and not force:
        return {
            "ok": True,
            "already": True,
            "graphsignal_run": run_path,
            "cuda": cuda,
            "fork_features": has_fork_features(run_path),
        }

    os.makedirs(root, exist_ok=True)
    venv = os.path.join(root, "venv")

    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if not os.path.exists(os.path.join(venv, "bin", "python")):
        p = _run([sys.executable, "-m", "venv", venv])
        if p.returncode != 0:
            return _err("runtime", f"venv creation failed: {p.stderr[-800:]}")

    pip = os.path.join(venv, "bin", "pip")
    _run([pip, "install", "--upgrade", "pip"])

    if source:
        if re.match(r"^(git\+)?(https?|ssh|git)://", source):
            spec = f"graphsignal[cu{cuda}] @ {source}"
        else:
            spec = f"{os.path.abspath(os.path.expanduser(source))}[cu{cuda}]"
    elif version:
        spec = f"graphsignal[cu{cuda}]=={version}"
    else:
        spec = f"graphsignal[cu{cuda}]"

    p = _run([pip, "install", spec])
    if p.returncode != 0:
        return _err("runtime", f"pip install {spec!r} failed: {(p.stderr or p.stdout)[-1500:]}")
    if not os.path.exists(run_path):
        return _err("runtime", f"install finished but {run_path} is missing")

    fork = has_fork_features(run_path)
    if fork is False:
        print(
            "[graphsignal] warning: installed package lacks llama.cpp/NInfer "
            "launchers (upstream?); use --source <fork> for engine metrics",
            file=sys.stderr,
        )
    ver = _run([run_path, "--version"])
    marker = {
        "cuda": cuda,
        "source": source,
        "version": version,
        "graphsignal_run": run_path,
        "reported": (ver.stdout or "").strip(),
        "fork_features": fork,
    }
    with open(os.path.join(root, "setup.json"), "w") as f:
        json.dump(marker, f, indent=2)
    return {"ok": True, **marker}


def read_signals_once(port: int, timeout: float = 2.0) -> dict:
    """One GET /signals; returns {} on any failure (profiling is best-effort)."""
    url = f"http://127.0.0.1:{port}/signals"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - endpoint may not be up yet
        return {}


def _err(kind: str, message: str) -> dict:
    return {"ok": False, "error": {"kind": kind, "message": message, "traceback": ""}}


def run(request: dict) -> dict:
    command = request.get("command")
    if command == "graphsignal_setup":
        return setup(request)
    return profile(request)


def profile(request: dict) -> dict:
    cmd = request.get("cmd") or []
    if not cmd:
        return _err("protocol", "graphsignal_profile requires 'cmd'")

    gs = find_graphsignal_run(request)
    if not gs and request.get("auto_setup", True):
        print(
            "[graphsignal] not found; provisioning into .kernelopt/graphsignal/venv …",
            file=sys.stderr,
        )
        res = setup(
            {
                "cuda": request.get("cuda"),
                "source": request.get("source"),
                "version": request.get("version"),
                "managed_dir": request.get("managed_dir"),
                "timeout_s": request.get("setup_timeout_s", 1800),
            }
        )
        if res.get("ok"):
            gs = res["graphsignal_run"]
        else:
            return _err(
                "runtime",
                "graphsignal-run not found and auto-setup failed: "
                + res["error"]["message"],
            )
    if not gs:
        return _err(
            "runtime",
            "graphsignal-run not found (run `kernelopt setup-graphsignal`, or set GRAPHSIGNAL_RUN)",
        )

    fork = has_fork_features(gs)
    if fork is False:
        print(
            "[graphsignal] warning: this graphsignal-run lacks llama.cpp/NInfer "
            "launchers; CUDA kernel attribution still works, engine metrics will not",
            file=sys.stderr,
        )

    port = int(request.get("listen_port", DEFAULT_PORT))
    trace = str(request.get("cuda_graph_trace") or "").strip()
    settle_s = float(request.get("settle_s", 1.0))
    poll_s = float(request.get("poll_s", 1.0))
    timeout_s = float(request.get("timeout_s", 1800))
    min_metrics = int(request.get("min_metrics", 1))
    cwd = request.get("cwd")

    argv = [gs, "--listen-port", str(port)]
    if trace:
        argv += ["--cuda-graph-trace", trace]
    argv += [str(a) for a in cmd]

    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except OSError as exc:
        return _err("runtime", f"failed to launch graphsignal-run: {exc}")

    last: dict = {}
    polls = 0
    started = time.time()
    time.sleep(settle_s)
    try:
        while proc.poll() is None:
            if time.time() - started > timeout_s:
                proc.kill()
                break
            payload = read_signals_once(port)
            polls += 1
            if payload.get("metrics"):
                last = payload
                if len(payload["metrics"]) >= min_metrics:
                    # One more settle so counters advance, then stop reading.
                    time.sleep(min(poll_s, 1.0))
                    final = read_signals_once(port)
                    if final.get("metrics"):
                        last = final
                    break
            time.sleep(poll_s)
        # The endpoint can outlive the process briefly; try a few final reads.
        for _ in range(5):
            payload = read_signals_once(port, timeout=1.0)
            if payload.get("metrics"):
                last = payload
                break
            time.sleep(0.3)
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    return {
        "ok": True,
        "exit_code": proc.returncode,
        "graphsignal_run": gs,
        "fork_features": fork,
        "listen_port": port,
        "cuda_graph_trace": trace or "graph",
        "polls": polls,
        "signals": last,
        "stdout_tail": (stdout or "")[-3000:],
        "stderr_tail": (stderr or "")[-2000:],
    }
