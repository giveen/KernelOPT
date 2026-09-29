"""Model-level (engine) E2E checks, engine/format-aware.

The op-level gates prove a kernel is correct/fast in isolation; these prove the
**whole engine** still produces the same output and did not get slower.

Supported model formats (auto-detected from the path, or forced with `engine`):

  *.ninfer                                  -> ninfer engine
  *.gguf                                    -> llama.cpp (`llama-cli`)
  a directory of *.safetensors (HF layout)  -> hf (transformers) | vllm | sglang
  anything                                  -> `cmd`: a custom command you supply

Commands:
  engine_generate     deterministic greedy generation -> output digest + wall time
  engine_perplexity   perplexity (ninfer / llama.cpp) -> numeric correctness metric

Request:
  build_dir, model (or weights), prompt, max_new, seed, timeout_s
  engine?   ninfer|llamacpp|hf|vllm|sglang   (default: auto-detect)
  cmd?      ["python", "my_engine.py", ...]  (bypasses engine dispatch)
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time

from .ninfer import _err

_PPL_RE = re.compile(r"^overall\s+\d+\s+[0-9.]+\s+([0-9.]+)", re.MULTILINE)
_LLAMA_PPL_RE = re.compile(r"\[\d+\]\d+.*?ppl\s*=\s*([0-9.]+)", re.IGNORECASE)


def run(request: dict) -> dict:
    command = request.get("command")
    if command == "engine_generate":
        return engine_generate(request)
    if command == "engine_perplexity":
        return engine_perplexity(request)
    raise ValueError(f"engine runner: unknown command {command!r}")


def detect_engine(model: str) -> str:
    p = str(model or "")
    if p.endswith(".ninfer"):
        return "ninfer"
    if p.endswith(".gguf"):
        return "llamacpp"
    if os.path.isdir(p):
        try:
            names = os.listdir(p)
        except OSError:
            names = []
        if any(n.endswith(".safetensors") for n in names) or "config.json" in names:
            return "hf"
    if p.endswith(".safetensors"):
        return "hf"
    return "hf"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _result(proc, elapsed: float, text: str, max_new: int) -> dict:
    return {
        "ok": True,
        "exit_code": proc.returncode,
        "text": text[:1000],
        "digest": _digest(text),
        "elapsed_s": elapsed,
        "tokens_per_s": (max_new / elapsed) if elapsed > 0 else None,
    }


def engine_generate(request: dict) -> dict:
    model = request.get("model") or request.get("weights")
    cmd = request.get("cmd")
    if cmd:
        return _custom(request, cmd)
    if not model or not os.path.exists(model):
        return _err("runtime", f"model not found: {model}")
    engine = (request.get("engine") or detect_engine(model)).lower()
    if engine == "ninfer":
        return _ninfer_generate(request, model)
    if engine in ("llamacpp", "llama", "gguf"):
        return _llamacpp_generate(request, model)
    if engine in ("hf", "transformers", "safetensors"):
        return _python_engine_generate(request, model, "hf")
    if engine in ("vllm", "sglang"):
        return _python_engine_generate(request, model, engine)
    return _err("protocol", f"unknown engine {engine!r} for {model}")


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _custom(request: dict, cmd: list[str]) -> dict:
    timeout = int(request.get("timeout_s", 900))
    started = time.time()
    try:
        proc = _run([str(a) for a in cmd], timeout)
    except subprocess.TimeoutExpired:
        return _err("timeout", f"custom E2E command timed out after {timeout}s")
    elapsed = time.time() - started
    text = (proc.stdout or "").strip()
    if proc.returncode != 0 or not text:
        return _err("runtime", f"custom command failed (exit {proc.returncode}): {(proc.stderr or '')[-800:]}")
    return _result(proc, elapsed, text, int(request.get("max_new", 0)))


def _ninfer_generate(request: dict, model: str) -> dict:
    build_dir = os.path.abspath(request["build_dir"])
    binary = request.get("binary") or os.path.join(build_dir, "apps", "ninfer")
    if not os.path.exists(binary):
        return _err("runtime", f"ninfer binary not found: {binary} (build target 'ninfer')")
    max_new = int(request.get("max_new", 16))
    argv = [
        binary, model,
        "--prompt", request.get("prompt", "The capital of France is"),
        "--max-new", str(max_new),
        "--greedy", "--seed", str(int(request.get("seed", 0))),
        "--no-thinking", "--log-level", "error",
    ]
    return _timed(argv, max_new, int(request.get("timeout_s", 900)), "ninfer generation")


def _llamacpp_generate(request: dict, model: str) -> dict:
    build_dir = os.path.abspath(request["build_dir"])
    binary = request.get("binary") or os.path.join(build_dir, "bin", "llama-cli")
    if not os.path.exists(binary):
        return _err("runtime", f"llama-cli not found: {binary}")
    max_new = int(request.get("max_new", 16))
    argv = [
        binary, "-m", model,
        "-p", request.get("prompt", "The capital of France is"),
        "-n", str(max_new),
        "--temp", "0", "-s", str(int(request.get("seed", 0))),
        "--no-display-prompt", "-st", "--log-disable", "--simple-io",
    ]
    return _timed(argv, max_new, int(request.get("timeout_s", 900)), "llama.cpp generation")


def _timed(argv: list[str], max_new: int, timeout: int, label: str) -> dict:
    started = time.time()
    try:
        proc = _run(argv, timeout)
    except subprocess.TimeoutExpired:
        return _err("timeout", f"{label} timed out after {timeout}s")
    elapsed = time.time() - started
    text = (proc.stdout or "").strip()
    if proc.returncode != 0 or not text:
        return _err("runtime", f"{label} failed (exit {proc.returncode}): {(proc.stderr or '')[-800:]}")
    return _result(proc, elapsed, text, max_new)


_HF_SNIPPET = r"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path, prompt, max_new, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="cuda")
inputs = tok(prompt, return_tensors="pt").to(model.device)
torch.manual_seed(seed)
out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
"""

_VLLM_SNIPPET = r"""
import sys
from vllm import LLM, SamplingParams
path, prompt, max_new, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
llm = LLM(model=path)
out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=max_new, seed=seed))
print(out[0].outputs[0].text)
"""

_SGLANG_SNIPPET = r"""
import sys
import sglang as sgl
path, prompt, max_new, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sgl.set_default_backend(sgl.RuntimeEndpoint(path))
@sgl.function
def gen(s): s += sgl.gen("out", max_tokens=max_new, temperature=0)
print(gen.run(prompt=prompt)["out"])
"""


def _python_engine_generate(request: dict, model: str, engine: str) -> dict:
    snippet = {"hf": _HF_SNIPPET, "vllm": _VLLM_SNIPPET, "sglang": _SGLANG_SNIPPET}[engine]
    max_new = int(request.get("max_new", 16))
    argv = [
        sys.executable, "-c", snippet, model,
        request.get("prompt", "The capital of France is"),
        str(max_new), str(int(request.get("seed", 0))),
    ]
    return _timed(argv, max_new, int(request.get("timeout_s", 900)), f"{engine} generation")


def engine_perplexity(request: dict) -> dict:
    model = request.get("model") or request.get("weights")
    if not model or not os.path.exists(model):
        return _err("runtime", f"model not found: {model}")
    engine = (request.get("engine") or detect_engine(model)).lower()
    build_dir = os.path.abspath(request["build_dir"])
    timeout = int(request.get("timeout_s", 900))

    if engine == "ninfer":
        binary = os.path.join(build_dir, "apps", "ninfer-perplexity")
        cmd = [binary, model, "--log-level", "warning"]
        if request.get("corpus"):
            cmd += ["--corpus", request["corpus"]]
            if request.get("quick"):
                cmd.append("--quick")
        elif request.get("text"):
            cmd += ["--text", request["text"]]
        else:
            return _err("protocol", "perplexity requires 'corpus' or 'text'")
        regex = _PPL_RE
    elif engine in ("llamacpp", "llama", "gguf"):
        binary = os.path.join(build_dir, "bin", "llama-perplexity")
        if not request.get("text"):
            return _err("protocol", "llama.cpp perplexity requires 'text'")
        cmd = [binary, "-m", model, "-f", request["text"], "--log-disable"]
        regex = _LLAMA_PPL_RE
    else:
        return _err("protocol", f"perplexity not supported for engine {engine!r}")

    if not os.path.exists(cmd[0]):
        return _err("runtime", f"binary not found: {cmd[0]}")
    try:
        proc = _run(cmd, timeout)
    except subprocess.TimeoutExpired:
        return _err("timeout", f"perplexity timed out after {timeout}s")
    m = regex.search((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0 or not m:
        return _err(
            "runtime",
            f"perplexity failed (exit {proc.returncode}): {(proc.stderr or proc.stdout or '')[-800:]}",
        )
    return {"ok": True, "perplexity": float(m.group(1))}
