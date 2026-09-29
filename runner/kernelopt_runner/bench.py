"""`bench` command — clean wall-clock timing via triton.testing.do_bench.

Subprocess-isolated (the runner is invoked fresh by Rust each time) so CUDA
state never leaks between measurements. Paper protocol: warmup=25 ms, rep=100 ms.

Request:
  model_file     — path to user model file
  variant_file   — optional path to a variant (e.g. re-stitched model) to bench
  inputs_fn      — name of the inputs function (default get_inputs)
  warmup_ms/reptime_ms — defaults 25 / 100 (paper §5)
  graphsignal    — optional dict {enable: bool, listen_port: int, cuda_graph_trace: str}
                   When enabled, this process is expected to already be running
                   under `graphsignal-run`; we read /signals before exit and embed
                   the payload (attribution only — never used for gate timings).
"""
from __future__ import annotations

from .common import get_model_and_inputs, load_module


def run(request: dict) -> dict:
    import torch
    from triton.testing import do_bench

    model_file = request["model_file"]
    variant_file = request.get("variant_file")  # future: re-stitched model
    inputs_fn = request.get("inputs_fn", "get_inputs")
    warmup_ms = float(request.get("warmup_ms", 25))
    rep_ms = float(request.get("rep_ms", 100))
    gs = request.get("graphsignal") or {}

    target_file = variant_file or model_file
    module = load_module(target_file, "kernelopt_bench_model")
    model, inputs = get_model_and_inputs(module, inputs_fn)
    model.eval()

    def step():
        with torch.no_grad():
            model(*inputs)

    # Warmup on the GPU, then hand timing to do_bench (which does its own).
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    mean_ms = do_bench(step, warmup=warmup_ms, rep=rep_ms, return_mode="mean")
    result = {
        "ok": True,
        "mean_ms": float(mean_ms),
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "bench_target": os_friendly(target_file),
    }

    if gs.get("enable"):
        from . import signals

        settle_s = float(gs.get("settle_s", 3))
        payload = signals.read_signals(
            port=int(gs.get("listen_port", 18259)), settle_s=settle_s
        )
        result["graphsignal"] = {"enabled": True, "payload": payload}

    return result


def os_friendly(path: str) -> str:
    import os

    return os.path.basename(path)
