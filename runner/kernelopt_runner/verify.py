"""`verify` command — Gates 1 & 2 (per-candidate).

Gate 1 (static): dry-run execution catches syntax errors and runtime crashes.
Gate 2 (multi-seed correctness): 3 seeds, allclose(rtol=atol=1e-3) against the
original PyTorch model executed in eager mode.

Request:
  model_file      — path to user model file (get_model/get_inputs)
  candidate_code  — full Python source of the candidate (kernel file)
  kernel_fn       — name of the @triton.jit function (optional; default: discover)
  wrapper_fn      — name of the callable wrapper (optional; default: kernel_function)
  seeds           — list of ints (default [0, 1, 2])
  rtol / atol     — floats (default 1e-3)

The candidate file may either:
  (a) define get_inputs() returning the block inputs, or
  (b) rely on the harness feeding the model's get_inputs() directly.

Weight extraction (paper §4.4): when Inductor externalizes model parameters,
a candidate must not construct fresh random weights. The harness therefore
injects the real eager baseline model into the candidate before execution:

  def init_with_model(model):   # optional hook
      global _baseline
      _baseline = model

Gate 2 runs the eager baseline on the SAME inputs as the candidate.
"""
from __future__ import annotations

import os
import tempfile
import types

from .common import load_module
from .weights import load_baseline


def _write_and_load(code: str, name: str) -> types.ModuleType:
    fd, path = tempfile.mkstemp(suffix=".py", prefix=f"kernelopt_cand_{name}_")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    return load_module(path, name)


def _find_kernel_and_wrapper(module) -> tuple[str, str]:
    kernel_fn = None
    wrapper_fn = None
    for name in dir(module):
        if name in ("kernel_function", "fused_kernel_call"):
            wrapper_fn = name
            break
        obj = getattr(module, name)
        if type(obj).__name__ in ("JITFunction", "Autotuner"):
            kernel_fn = name
    if wrapper_fn is None:
        for name in ("kernel_function", "fused_kernel_call"):
            if hasattr(module, name):
                wrapper_fn = name
                break
    if kernel_fn is None:
        # Any module attribute with a 'cache' dict is a JITFunction.
        for name in dir(module):
            obj = getattr(module, name)
            if type(obj).__name__ in ("JITFunction", "Autotuner"):
                kernel_fn = name
                break
    return kernel_fn or "", wrapper_fn or ""


def run(request: dict) -> dict:
    import torch

    model_file = request["model_file"]
    candidate_code = request["candidate_code"]
    seeds = request.get("seeds", [0, 1, 2])
    rtol = float(request.get("rtol", 1e-3))
    atol = float(request.get("atol", 1e-3))
    inputs_fn = request.get("inputs_fn", "get_inputs")
    baseline_seed = int(request.get("baseline_seed", 0))
    weights_path = request.get("baseline_weights_path")

    # ---- Load baseline model & inputs (canonical weights) ----
    model, inputs = load_baseline(model_file, inputs_fn, baseline_seed, weights_path)
    model.eval()

    # ---- Gate 1: dry-run of candidate (compile + single execution) ----
    gate1 = {"passed": False, "error": None}
    candidate = None
    try:
        candidate = _write_and_load(candidate_code, "candidate")
        kernel_fn_name, wrapper_fn_name = _find_kernel_and_wrapper(candidate)
        if not wrapper_fn_name and not kernel_fn_name:
            raise RuntimeError(
                "no @triton.jit kernel or kernel_function wrapper found in candidate"
            )
        wrapper = getattr(candidate, wrapper_fn_name or kernel_fn_name)
        if hasattr(candidate, "init_with_model"):
            candidate.init_with_model(model)
        with torch.no_grad():
            _ = wrapper(*[t.clone() for t in inputs])
        torch.cuda.synchronize()
        gate1["passed"] = True
    except Exception as exc:  # noqa: BLE001
        import traceback as tb

        gate1["error"] = {
            "kind": "compile" if isinstance(exc, (SyntaxError, NameError, AttributeError)) else "runtime",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": tb.format_exc(),
        }
        return {"ok": True, "gate1": gate1, "gate2": {"passed": False, "skipped": True}, "all_passed": False}

    # ---- Gate 2: multi-seed correctness vs eager baseline ----
    gate2 = {"passed": False, "results": [], "error": None}
    all_passed = True
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            if hasattr(candidate, "get_inputs"):
                cand_inputs = candidate.get_inputs()
                if not isinstance(cand_inputs, (list, tuple)):
                    cand_inputs = [cand_inputs]
            else:
                cand_inputs = inputs
            cand_inputs = [t.cuda() for t in cand_inputs]

            with torch.no_grad():
                # Kernel workbench mode: judge kernel_function against the
                # eager BASELINE MODEL output (real weights), not against
                # itself — a trivial wrapper would otherwise always pass.
                # get_inputs() supplies the matmul input (weight/bias live in
                # the model); kernel_function returns the fused epilogue.
                ref = model(*[t.clone() for t in cand_inputs])
                test = getattr(candidate, wrapper_fn_name)(*[t.clone() for t in cand_inputs])
            torch.cuda.synchronize()

            passed = torch.allclose(ref, test, rtol=rtol, atol=atol)
            max_abs_err = float((ref - test).abs().max()) if passed is not None else float("inf")
            gate2["results"].append(
                {"seed": seed, "passed": bool(passed), "max_abs_err": max_abs_err}
            )
            all_passed = all_passed and bool(passed)
        gate2["passed"] = all_passed
    except Exception as exc:  # noqa: BLE001
        import traceback as tb

        gate2["error"] = {
            "kind": "correctness",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": tb.format_exc(),
        }
        all_passed = False

    return {"ok": True, "gate1": gate1, "gate2": gate2, "all_passed": gate1["passed"] and gate2["passed"] and all_passed}
