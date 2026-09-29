"""Shared weight loading: canonical baseline state_dict for all stages.

A fixture's get_model() is random-init, so every stage must instantiate the
baseline the same way (fixed seed) or load the canonical snapshot written by
`trace`. Both paths produce identical weights; the snapshot is authoritative
when present.
"""
from __future__ import annotations

import os

from .common import get_model_and_inputs, load_module


def load_baseline(model_file: str, inputs_fn: str, seed: int, weights_path: str | None):
    """Return (model, inputs) with canonical baseline weights."""
    import torch

    if weights_path and os.path.exists(weights_path):
        module = load_module(model_file, "kernelopt_baseline_loaded")
        model_fn = getattr(module, "get_model")
        model = model_fn().cuda()
        state = torch.load(weights_path, map_location="cuda")
        model.load_state_dict(state)
        inputs = getattr(module, inputs_fn)()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return model, list(inputs)

    # Snapshot absent: deterministic seed reproduces trace-time weights.
    torch.manual_seed(seed)
    module = load_module(model_file, "kernelopt_baseline_seeded")
    model, inputs = get_model_and_inputs(module, inputs_fn)
    return model, inputs
