# KernelOpt — Model-level (engine) verification

Op-level gates prove a kernel is correct and fast **in isolation**. They do not
prove the **model** got better: a kernel can win its microbenchmark yet slow the
engine down (dispatch overhead, lost fusion) or subtly change numerics. This is
the paper's Gate 3.

## How it works

KernelOPT runs the **whole engine** on the baseline build and on the candidate
build, then compares output and speed:

```
baseline:  <engine> <model> … --greedy/--temp 0 --seed 0   → output digest + wall time
candidate: same command on the rebuilt engine
             → digest must MATCH, wall time must be ≤ γ × baseline
```

- **Correctness** — deterministic greedy decoding (fixed seed, temperature 0) is
  reproducible, so the output digest is identical **iff** the kernel change
  preserved the model's semantics.
- **Performance** — the same run is timed; a candidate slower than baseline by
  more than γ (1.03) fails, catching "faster kernel, slower model".
- **Numeric option** — perplexity gives a stronger numeric comparison where the
  engine supports it.

The engine binaries are built **from the worktree**, so the candidate kernel is
what executes.

## Model formats and engines

The model path is auto-detected; `--e2e-engine` overrides it.

| Model path | Engine | Command |
|---|---|---|
| `*.ninfer` | `ninfer` | `build/apps/ninfer` (greedy, `--no-thinking`) |
| `*.gguf` | `llamacpp` | `build/bin/llama-cli` (`--temp 0`, `-st`) |
| a directory with `*.safetensors` / `config.json` | `hf` (default) | `transformers` offline greedy |
| same dir, `--e2e-engine vllm` / `sglang` | `vllm` / `sglang` | their offline APIs |
| anything | `cmd` | your `--e2e-cmd <argv…>` (bypasses dispatch) |

So: **llama.cpp takes GGUF; vLLM/SGLang (and any HF model) take safetensors; ninfer
takes `.ninfer`.** If your engine isn't covered, `--e2e-cmd` runs any command and
compares its stdout digest + wall time.

Perplexity (`engine_perplexity`) is implemented for `ninfer`
(`ninfer-perplexity --corpus|--text`) and `llamacpp` (`llama-perplexity -f`).

## Usage

```bash
# ninfer
kernelopt run-ninfer --op fp8_linear_add --repo "$NINFER_REPO" \
    --e2e-weights "$NINFER_REPO/models/qwen3_8_27b_nvfp4.v3.ninfer" \
    --e2e-prompt "The capital of France is" --e2e-max-new 16

# llama.cpp (GGUF)
kernelopt run-llamacpp --op SOFT_MAX --repo "$LLAMACPP_REPO" \
    --e2e-weights /path/to/model.gguf

# vLLM/SGLang (HF safetensors dir)
kernelopt run-llamacpp --op SOFT_MAX --repo "$LLAMACPP_REPO" \
    --e2e-weights /path/to/hf-model-dir --e2e-engine vllm

# anything else: your command (place --e2e-cmd last)
kernelopt run-ninfer --op add_bias --repo "$NINFER_REPO" \
    --e2e-cmd python my_engine_check.py --model /path/to/model
```

The baseline engine run is captured during the baseline stage; the candidate run
happens at finalize. Both are journaled:

```json
{"type":"GatesVerdict","stage":"engine_e2e","passed":true,
 "detail":{"same_tokens":true,"not_slower":true,"baseline_elapsed_s":5.1,"candidate_elapsed_s":5.0,"gamma":1.03}}
```

In a campaign, pass `--e2e-weights` once and every target gets the Gate 3 check.

## Cost and caveats

- Building the engine app and loading a model is the expensive part, so Gate 3 is
  **opt-in** (`--e2e-weights`).
- Generation wall-time is a proxy for engine throughput; it needs no corpus. For a
  rigorous number, the engine's own benchmark (`ninfer_bench -o json`,
  `llama-bench`) is the authority and can be added alongside.
- `hf`/`vllm`/`sglang` run via a small Python harness using the runner's
  interpreter; they require `transformers` / `vllm` / `sglang` to be importable.
- ninfer requires a v3 `.ninfer` artifact (v2 is rejected with an upgrade hint).
- Without `--e2e-weights`, the run reports `optimized`/`matched` on the op-level
  gates alone.
