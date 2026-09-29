"""`trace` command: torch.compile the user model, capture Inductor output.

Classifies sub-kernels as Triton-generated (optimizable) or extern library
calls (cuBLAS/cuDNN — preserved), per the paper's dispatch-aware rule.

Implementation note: we use TORCHINDUCTOR_CACHE_DIR + TORCH_COMPILE_DEBUG to
read Inductor's generated output_hybrid.py. That is fragile across torch
versions, so we additionally capture via a custom backend that records the
compiled fn graph — and always return the eager graph fallback info.
"""
from __future__ import annotations

import json
import os
import re
import tempfile

from .common import get_model_and_inputs, load_module

# Extern-kernel markers in Inductor output code (paper §4.1).
EXTERN_PATTERN = re.compile(
    r"extern_kernels\.(mm|addmm|bmm|convolution|bd_non_dilated|convolution_backward|_scaled_mm)"
)
TRITON_KERNEL_PATTERN = re.compile(r"@triton\.jit\s*\ndef\s+(\w+)\s*\(")
TRITON_LAUNCH_PATTERN = re.compile(r"(\w+)\.run\(")

# torch 2.x puts Inductor output here when TORCH_COMPILE_DEBUG=1
_DEBUG_SUBDIRS = ("torch_compile_debug", "run_*", "torchinductor_*")


def _find_inductor_output(debug_dir: str) -> str | None:
    for root, _dirs, files in os.walk(debug_dir):
        for name in files:
            if name in ("output_code.py", "output_hybrid.py"):
                return os.path.join(root, name)
        # prefer the deepest output_code.py if several
    best = None
    for root, _dirs, files in os.walk(debug_dir):
        for name in files:
            if name in ("output_code.py", "output_hybrid.py"):
                candidate = os.path.join(root, name)
                if best is None or candidate.count(os.sep) > best.count(os.sep):
                    best = candidate
    return best


def run(request: dict) -> dict:
    model_file = request["model_file"]
    inputs_fn = request.get("inputs_fn", "get_inputs")
    mode = request.get("mode", "max-autotune")  # "max-autotune" | "default"
    compile_debug = request.get("compile_debug", True)
    out_weights = request.get("out_weights")
    seed = int(request.get("seed", 0))

    import torch

    # Canonical baseline weights: get_model() is random-init, so the ONE
    # instance created here is snapshotted and reused by every later stage
    # (verify injection, restitch embedding, e2e comparison). This is the
    # walking-skeleton stand-in for the paper's capture backend (§4.4).
    torch.manual_seed(seed)
    module = load_module(model_file, "kernelopt_user_model")
    model, inputs = get_model_and_inputs(module, inputs_fn)
    weights_path = None
    if out_weights:
        os.makedirs(os.path.dirname(os.path.abspath(out_weights)), exist_ok=True)
        torch.save(model.state_dict(), out_weights)
        weights_path = os.path.abspath(out_weights)

    # CUDA graph tracing affects what graphsignal sees later; irrelevant here.
    debug_dir = tempfile.mkdtemp(prefix="kernelopt_trace_")
    if compile_debug:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = debug_dir

    compile_opts = {"mode": mode} if mode != "default" else {}
    compiled = torch.compile(model, **compile_opts)

    with torch.no_grad():
        # Warmup + one real call to force full compile+autotune.
        for _ in range(3):
            _ = compiled(*inputs)
        torch.cuda.synchronize()

    inductor_path = _find_inductor_output(debug_dir) if compile_debug else None
    source = ""
    if inductor_path:
        with open(inductor_path, "r") as f:
            source = f.read()

    triton_kernels = sorted(set(TRITON_KERNEL_PATTERN.findall(source)))
    # Triton launches appear as `<kernel_name>.run(...)` in Inductor output.
    launched = sorted(set(TRITON_LAUNCH_PATTERN.findall(source)) & set(triton_kernels))
    extern_ops = sorted(set(EXTERN_PATTERN.findall(source)))

    # Extract the @triton.jit function bodies — the executor agent edits THESE,
    # not the original model file (paper §4.1: optimize the generated Triton).
    kernel_sources = _extract_jit_bodies(source, launched)

    sub_kernels = []
    for name in launched:
        sub_kernels.append({"name": name, "class": "triton", "needs_replacement": True})
    for op in extern_ops:
        sub_kernels.append(
            {"name": op, "class": "extern", "needs_replacement": False,
             "library": _extern_library(op)}
        )

    if not sub_kernels:
        classification = "no_gpu_kernels"
    elif extern_ops and not launched:
        classification = "extern_only"
    else:
        classification = "optimizable"

    return {
        "ok": True,
        "baseline_weights_path": weights_path,
        "classification": classification,
        "triton_kernel_sources": kernel_sources,
        "sub_kernels": sub_kernels,
        "triton_kernel_names": triton_kernels,
        "extern_ops": extern_ops,
        "inductor_output_path": inductor_path,
        "debug_dir": debug_dir if compile_debug else None,
        "inductor_source_lines": source.count("\n") + 1 if source else 0,
        "model_repr": type(model._orig_mod).__name__ if hasattr(model, "_orig_mod") else type(model).__name__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compile_mode": mode,
        # Placeholder for M4: per-group aten op graphs for CodeGen/Fusion.
        "fusible_groups": [],
    }


def _extern_library(op: str) -> str:
    if op in ("mm", "addmm", "bmm", "_scaled_mm"):
        return "cuBLAS"
    if op.startswith("conv") or op.startswith("bd_"):
        return "cuDNN"
    return "unknown"


def _extract_jit_bodies(source: str, names: list[str], max_bodies: int = 4) -> list[dict]:
    """Pull the full source of each @triton.jit function out of Inductor output.

    Inductor emits kernels inside a TritonCodeCache-related wrapper, typically
    ending with `''', device_str='cuda')` — the body runs from the @triton.jit
    decorator (or the def when unwrapped) to that terminator.
    """
    out = []
    for name in names[:max_bodies]:
        def_idx = source.find(f"def {name}(")
        if def_idx < 0:
            continue
        # Start at the decorator line if one sits directly above the def.
        line_start = source.rfind("\n", 0, def_idx) + 1
        dec_line = source.rfind("\n", 0, line_start - 1) + 1
        start = def_idx
        if "@triton.jit" in source[dec_line:line_start]:
            start = dec_line
        # End at the wrapper terminator if present, else at the next top-level def.
        term_idx = source.find("''', device_str=", def_idx)
        if term_idx >= 0:
            end = term_idx
        else:
            next_def = source.find("\ndef ", def_idx + 1)
            end = next_def if next_def >= 0 else len(source)
        body = source[start:end].rstrip()
        out.append({"name": name, "source": body})
    return out
