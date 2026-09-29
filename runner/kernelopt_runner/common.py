"""Shared helpers: user-model loading, structured errors, do_bench."""
from __future__ import annotations

import importlib.util
import os
import sys
import types


def load_module(path: str, name_hint: str) -> types.ModuleType:
    """Import a Python file by path."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    spec = importlib.util.spec_from_file_location(name_hint, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name_hint] = module
    spec.loader.exec_module(module)
    return module


def get_model_and_inputs(module: types.ModuleType, inputs_fn: str = "get_inputs"):
    model_fn = getattr(module, "get_model")
    model = model_fn()
    model = model.cuda()
    inputs = getattr(module, inputs_fn)()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    return model, list(inputs)


def call_with_inputs(model, inputs):
    """Call model with a list/tuple of positional inputs."""
    return model(*inputs)
