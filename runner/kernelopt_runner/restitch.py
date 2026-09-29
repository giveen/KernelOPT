"""`restitch` command — build the re-stitched (optimized) model file.

The optimizer's source defines the model class; the emitted file embeds the
BASELINE model's real weights (base64 torch.save state_dict) and overrides
`get_model()` to load them — a candidate must never evaluate with freshly
initialized weights (the paper's weight-extraction rule, §4.4).

Emitted file layout:
    <header + provenance comments>
    <optimizer source verbatim>            # class Model, helpers, get_inputs
    _WEIGHTS_B64 = "..."                   # baseline state_dict
    def get_model():                       # shadows the source's get_model
        m = <model_class>(); m.load_state_dict(...); return m
"""
from __future__ import annotations

import base64
import io
import os

HEADER = '"""KernelOpt re-stitched model (auto-generated; do not edit by hand)."""\n'


def run(request: dict) -> dict:
    import torch

    source_code = request["source_code"]
    out_path = request["out_path"]
    baseline_file = request["baseline_file"]
    model_class = request.get("model_class", "Model")
    inputs_fn = request.get("inputs_fn", "get_inputs")
    provenance = request.get("provenance", {})

    from .weights import load_baseline

    baseline_seed = int(request.get("baseline_seed", 0))
    weights_path = request.get("baseline_weights_path")
    baseline_model, _ = load_baseline(
        baseline_file, inputs_fn, baseline_seed, weights_path
    )

    buf = io.BytesIO()
    torch.save(baseline_model.state_dict(), buf)
    weights_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(HEADER)
        if provenance:
            f.write("# provenance: " + repr(provenance) + "\n")
        f.write(source_code)
        if not source_code.endswith("\n"):
            f.write("\n")
        f.write(f"\n\nimport base64 as _base64\nimport io as _io\n")
        f.write(f'_WEIGHTS_B64 = "{weights_b64}"\n')
        f.write(f"def get_model():\n")
        f.write(f"    m = {model_class}()\n")
        f.write(
            "    m.load_state_dict(torch.load("
            "_io.BytesIO(_base64.b64decode(_WEIGHTS_B64)), map_location='cuda'))\n"
        )
        f.write("    return m.cuda()\n")

    # Structural sanity.
    missing = [name for name in (inputs_fn,) if f"def {name}(" not in source_code]
    if model_class and not (
        f"class {model_class}" in source_code or f"def {model_class}" in source_code
    ):
        missing.append(model_class)
    if missing:
        return {
            "ok": False,
            "error": {
                "kind": "compile",
                "message": f"re-stitched source missing {missing}",
                "traceback": "",
            },
        }

    return {"ok": True, "out_path": os.path.abspath(out_path)}
