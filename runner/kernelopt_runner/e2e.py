"""`e2e_verify` command — Gates 3 & 4 (end-to-end, on the re-stitched model).

Gate 3 (model-level correctness): stricter tolerance (default 1e-4). When
strict allclose fails (common for TF32 tensor-core paths), a float64
reference disambiguates (paper §4.4, Eq. 4):

    r_f32 = baseline output (FP32)     r_f64 = baseline output (FP64)
    o_f32 = optimized output (FP32)
    d_ref = ||r_f32 - r_f64||inf       d_opt = ||o_f32 - r_f64||inf

    if d_ref >= 1e-8:  pass iff rho = d_opt / d_ref <= rho_max (default 10)
    else (exact ops):  pass iff d_opt / max(||r_f32||inf, 1e-12) <= 1e-4
    absolute branch:   d_opt <= max(5e-3, 1e-3 * ||r_f32||inf)  (fused kernels)

Gate 4 (performance) is measured by Rust via separate `bench` calls
(subprocess isolation); this command reports correctness only.

Request:
  baseline_file / variant_file — paths with get_model/get_inputs
  seeds, rtol, atol, rho_max
"""
from __future__ import annotations

from .common import get_model_and_inputs, load_module
from .weights import load_baseline


def run(request: dict) -> dict:
    import torch

    baseline_file = request["baseline_file"]
    variant_file = request["variant_file"]
    seeds = request.get("seeds", [0, 1, 2])
    rtol = float(request.get("rtol", 1e-4))
    atol = float(request.get("atol", 1e-4))
    rho_max = float(request.get("rho_max", 10.0))
    inputs_fn = request.get("inputs_fn", "get_inputs")

    import copy

    bmod = load_module(baseline_file, "kernelopt_baseline")
    vmod = load_module(variant_file, "kernelopt_variant")
    baseline_seed = int(request.get("baseline_seed", 0))
    weights_path = request.get("baseline_weights_path")
    bmodel, _ = load_baseline(baseline_file, inputs_fn, baseline_seed, weights_path)
    vmodel, _ = get_model_and_inputs(vmod, inputs_fn)
    bmodel.eval()
    vmodel.eval()
    # Module.double() mutates in place; the f64 reference needs its own copy.
    bmodel64 = copy.deepcopy(bmodel).double()

    per_seed = []
    all_passed = True
    for seed in seeds:
        torch.manual_seed(seed)
        if hasattr(vmod, "get_inputs"):
            inputs = vmod.get_inputs()
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
        else:
            inputs = get_model_and_inputs(bmod, inputs_fn)[1]
        inputs = [t.cuda() for t in inputs]

        with torch.no_grad():
            r32 = bmodel(*[t.clone() for t in inputs]).float()
            o32 = vmodel(*[t.clone() for t in inputs]).float()
            r64 = bmodel64(*[t.clone().double() for t in inputs]).float()
        torch.cuda.synchronize()

        strict = bool(torch.allclose(r32, o32, rtol=rtol, atol=atol))
        d_ref = float((r32 - r64).abs().max())
        d_opt = float((o32 - r64).abs().max())
        r_inf = float(r32.abs().max())

        if strict:
            verdict, reason = True, "strict_allclose"
        elif d_ref >= 1e-8:
            rho = d_opt / d_ref if d_ref > 0 else float("inf")
            verdict, reason = rho <= rho_max, f"rho={rho:.2f} (max {rho_max})"
        else:
            scale_rel = d_opt / max(r_inf, 1e-12)
            verdict, reason = scale_rel <= 1e-4, f"scale_rel={scale_rel:.2e}"
            # absolute branch (fused kernels anchor precision artificially)
            if not verdict and d_opt <= max(5e-3, 1e-3 * r_inf):
                verdict, reason = True, f"absolute bound: d_opt={d_opt:.2e}"

        per_seed.append(
            {
                "seed": seed,
                "strict": strict,
                "passed": bool(verdict),
                "reason": reason,
                "d_ref": d_ref,
                "d_opt": d_opt,
            }
        )
        all_passed = all_passed and verdict

    return {"ok": True, "gate3": {"passed": all_passed, "results": per_seed}}
