"""`ncu` command — deep profiling context for the Planner (paper §4.2).

Request:
  model_file        — user model file (get_model/get_inputs)
  ncu_path          — path to the ncu binary (default: discovered)
  ncu_set           --set argument: "full" (paper) or "basic"
  config            — Profiler-agent submission:
      {kernel_name, launch_skip, launch_count, replay_mode, kernel_args}
  iters             — steady-state iterations in the harness (default 3)

Triton kernels are JIT-compiled → mangled names that don't match the Python
function name, and kernel replay is incompatible with them — hence the
Profiler agent sets replay_mode='application' and kernel_name=''. Those
choices are enforced here as hard rules, not just prompt guidance.

Response:
  context: {kernels: [{name, sol_memory, sol_compute, duration_us, registers,
                       achieved_occupancy}], rules: [{kernel, rule, estimated_speedup}],
            raw_csv_path, tier_hint}
"""
from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile

from .common import get_model_and_inputs, load_module

HARNESS_TEMPLATE = '''
import sys
import torch

sys.path.insert(0, {model_dir!r})
from kernelopt_user_model import get_model, {inputs_fn}

model = get_model().cuda()
model.eval()
inputs = [{inputs_fn}]()
if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]

with torch.no_grad():
    for _ in range(3):
        model(*inputs)
    torch.cuda.synchronize()
    for _ in range({iters}):
        model(*inputs)
    torch.cuda.synchronize()
'''

# SOL metric names in ncu CSV output: key -> (Metric Name, expected unit).
# `Memory Throughput` and `Compute (SM) Throughput` appear in BOTH the SOL
# section (unit %) and the analysis sections (unit byte/s, inst/s), so the unit
# must be checked or the later row clobbers the real SOL value.
SOL_METRICS = {
    "sol_compute": ("Compute (SM) Throughput", "%"),
    "sol_memory": ("Memory Throughput", "%"),
    "duration": ("Duration", None),  # time unit varies (ns/us/ms)
    "registers": ("Registers Per Thread", "register/thread"),
    "occupancy": ("Achieved Occupancy", "%"),
}


def _unit_matches(expected: str | None, unit: str) -> bool:
    if expected is None:  # duration: any time unit
        return unit in ("ns", "us", "µs", "ms")
    if expected == "%":
        return unit == "%"
    return expected in unit or unit in expected


def _duration_us(value: float, unit: str) -> float:
    return {"ns": value / 1000.0, "us": value, "µs": value, "ms": value * 1000.0}.get(unit, value)


def find_ncu() -> str | None:
    p = shutil.which("ncu")
    if p:
        return p
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root:
            candidate = os.path.join(root, "bin", "ncu")
            if os.path.exists(candidate):
                return candidate
    return None


def run(request: dict) -> dict:
    model_file = request["model_file"]
    ncu_set = request.get("ncu_set", "full")
    cfg = request.get("config") or {}
    iters = int(request.get("iters", 3))

    ncu = request.get("ncu_path") or find_ncu()
    if not ncu:
        return {
            "ok": False,
            "error": {"kind": "runtime", "message": "ncu binary not found", "traceback": ""},
        }

    # Normalize + enforce the JIT-kernel rules.
    replay_mode = cfg.get("replay_mode") or "application"
    if replay_mode not in ("application", "kernel"):
        replay_mode = "application"
    kernel_name = cfg.get("kernel_name")
    if kernel_name:  # Triton/Helion: names are mangled; empty means "profile all"
        kernel_name = None

    # Write the model file where the harness imports it from.
    model_dir = os.path.dirname(os.path.abspath(model_file))
    load_module(model_file, "kernelopt_user_model")  # fail fast if broken

    harness = HARNESS_TEMPLATE.format(
        model_dir=model_dir,
        inputs_fn=request.get("inputs_fn", "get_inputs"),
        iters=iters,
    )
    tmpdir = tempfile.mkdtemp(prefix="kernelopt_ncu_")
    harness_path = os.path.join(tmpdir, "harness.py")
    with open(harness_path, "w") as f:
        f.write(harness)

    cmd = [
        ncu,
        "--csv",
        f"--set={ncu_set}",
        f"--replay-mode={replay_mode}",
        "--target-processes=all",
    ]
    launch_skip = int(cfg.get("launch_skip") or 0)
    if launch_skip > 0:
        cmd.append(f"--launch-skip={launch_skip}")
    launch_count = cfg.get("launch_count")
    if launch_count:
        cmd.append(f"--launch-count={int(launch_count)}")
    cmd.append(sys_executable())
    cmd.append(harness_path)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=int(request.get("timeout_s", 900))
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": {"kind": "timeout", "message": "ncu run timed out", "traceback": ""},
        }

    csv_text = _extract_csv(proc.stdout)
    context = parse_ncu_csv(csv_text) if csv_text.strip() else {"kernels": [], "rules": []}
    if not context["kernels"]:
        # No kernels parsed: distinguish the common permission failure.
        combined = (proc.stdout or "") + (proc.stderr or "")
        if "ERR_NVGPUCTRPERM" in combined:
            return {
                "ok": False,
                "error": {
                    "kind": "runtime",
                    "message": "ERR_NVGPUCTRPERM: GPU performance counters are admin-only "
                    "(RmProfilingAdminOnly=1). Set NVreg_RmProfilingAdminOnly=0 in "
                    "/etc/modprobe.d/ and reload the driver / reboot.",
                    "traceback": "",
                },
            }
        return {
            "ok": False,
            "error": {
                "kind": "runtime",
                "message": f"ncu produced no kernel data (exit {proc.returncode}): "
                + combined[-600:],
                "traceback": "",
            },
        }

    raw_csv_path = os.path.join(tmpdir, "ncu_report.csv")
    with open(raw_csv_path, "w") as f:
        f.write(csv_text)

    context["raw_csv_path"] = raw_csv_path
    context["ncu_command"] = " ".join(cmd)
    context["replay_mode"] = replay_mode
    return {"ok": True, "context": context}


def sys_executable() -> str:
    import sys

    return sys.executable


def _extract_csv(stdout: str) -> str:
    """ncu prints progress lines before the CSV header; keep from the header on."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            return "\n".join(lines[i:])
    return stdout


def parse_ncu_csv(csv_text: str) -> dict:
    """Parse ncu CSV into a typed context.

    Generic shape: rows of
      ID, ..., Kernel Name, Section Name, Metric Name, Metric Unit, Metric Value
    We key per-kernel metrics by metric name and collect rule rows separately.
    """
    kernels: dict[str, dict] = {}
    rules: list[dict] = []
    metric_to_key = {v[0]: (k, v[1]) for k, v in SOL_METRICS.items()}

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        name = (
            row.get("Kernel Name")
            or row.get("Kernel Name (correlation ID)")
            or ""
        ).strip()
        if not name:
            continue
        metric = (row.get("Metric Name") or "").strip()
        unit = (row.get("Metric Unit") or "").strip()
        raw = (row.get("Metric Value") or "").strip()
        if not metric or raw == "":
            continue
        value = _to_float(raw)

        entry = kernels.setdefault(name, {"name": name, "metrics": {}})
        if metric in metric_to_key:
            key, expected_unit = metric_to_key[metric]
            if not _unit_matches(expected_unit, unit):
                continue  # e.g. byte/s row named "Memory Throughput"
            if key == "duration":
                value = _duration_us(value, unit)
            # First valid row wins so a later analysis row can't clobber the SOL %.
            if key not in entry:
                entry[key] = value
                if unit:
                    entry.setdefault("units", {})[key] = unit
        elif metric.startswith(("OPT", "Rule")) or "Recommendation" in (row.get("Section Name") or ""):
            est = _to_float(raw)
            rules.append(
                {
                    "kernel": name,
                    "rule": metric,
                    "estimated_speedup": est,
                    "details": (row.get("Rule Description") or row.get("Rule Type") or "").strip(),
                }
            )

    # Per-kernel summary + rank rules by estimated speedup (paper: top-3 injected).
    kernel_list = []
    for entry in kernels.values():
        kernel_list.append(
            {
                "name": entry["name"],
                "sol_compute": entry.get("sol_compute"),
                "sol_memory": entry.get("sol_memory"),
                "duration_us": entry.get("duration"),
                "registers": entry.get("registers"),
                "achieved_occupancy": entry.get("occupancy"),
                "units": entry.get("units", {}),
            }
        )
    rules.sort(key=lambda r: r["estimated_speedup"], reverse=True)
    return {"kernels": kernel_list, "rules": rules}


def _to_float(raw: str) -> float:
    """ncu values come as '82.51', '1,234.56', '<', etc."""
    cleaned = raw.replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0
