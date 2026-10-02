# KernelOPT

Dispatch-aware, agentic GPU-kernel optimization. Point KernelOPT at a **single
kernel, a directory of kernels, or a whole model** — Triton kernels in a compiled
PyTorch model, a CUDA op in ninfer, a ggml-cuda kernel in llama.cpp, or **any
CMake-based GPU repo (CUDA or HIP/ROCm)** that declares its build/test/bench in
`kernelopt.toml`. It profiles them and drives a profile → plan → edit → verify →
gate loop until each is faster — or a budget runs out. Your working tree is never
touched: edits happen in an isolated git worktree, every accepted candidate is
**committed and tagged** (so it can be read or reverted instantly), and the winner
is handed back as a unified diff.

An implementation of *KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel
Optimization* (arXiv:2609.30059), extended from its Triton/Inductor setting to
CUDA kernel trees (ninfer, llama.cpp) and a generic descriptor-driven backend
(CUDA and HIP/ROCm), with engine-share ranking via the NVIDIA suite (nsys/ncu) or
Graphsignal (CUDA or ROCm).

| Mode | Target tree | Optimization unit | Correctness | Timing |
|---|---|---|---|---|
| `triton` | compiled PyTorch model | one Inductor `@triton.jit` kernel | eager PyTorch + `allclose` | `triton.testing.do_bench` |
| `ninfer` | ninfer checkout | one `src/ops/<family>/<impl>/*.cu` | the Op's own `ctest` suite | `ninfer_<op>_bench` |
| `llamacpp` | llama.cpp checkout | one `ggml/src/ggml-cuda/*.cu` | `test-backend-ops test -o <OP>` | `test-backend-ops perf -o <OP>` |
| `custom` | any CMake GPU repo (CUDA or HIP/ROCm) with `kernelopt.toml` | a file you declare | your `test_cmd` | your `bench_cmd` |

## How it works

```
discover ─▶ baseline ─▶ ┌ plan ─▶ edit ─▶ compile ─▶ correctness ─▶ bench ┐ ─▶ γ/roofline gate ─▶ shape gate ─▶ diff
                        └────────── iterate until patience/target/budget ──┘
```

LLM agents (Planner → Executor → Summarizer) work with an NCU/Graphsignal-guided
UCB-guided beam search and shared experience memory. Five gates filter candidates: static
compile, multi-seed correctness, **model-level (engine) verification**, a
performance gate (pinned shape, interleaved fresh baseline/candidate rounds, a
noise floor and a sign test), and **measured-shape correctness** — the op's own
suite may not cover the launch path the perf gate measures. A perf win must also
be *physically plausible*: a candidate that appears to beat the device's memory
roofline is rejected as "doing less work", not accepted as a win. If nothing
passes, the baseline is preserved.

The Planner can also pull context on demand (`search_repo`/`read_file`, ripgrep)
instead of receiving a full file dump, and every accepted candidate (compile +
correctness + bench) is committed and tagged for fast revert. The Executor edits
by whole file (default) or by unified diff (`--edit-mode patch`, opt-in) to avoid
reproduction drift; either way the run outputs a reviewable unified diff and your
checkout is never modified.

**Measurement rigor:** an op-level speedup is measured on a pinned shape with 3
repeats, then re-confirmed with interleaved fresh baseline/candidate re-benches
(`--final-rounds`). A candidate is only called `optimized` if it beats the baseline
by more than the measured noise, wins every round, is physically plausible
(below a generous multiple of the memory roofline), and is correct on the
measured shape; otherwise it is `matched` (correct but not faster) or `fallback`
(rejected). Correctness-only targets (no bench) report `unverified`: a candidate
was applied, but no speedup could be measured. The reported number is always explicit (`N×` = baseline/final, with
the shape, ms values, noise, and bandwidth-vs-roofline).

**One GPU job at a time:** bench/verify/NCU/engine-E2E take an exclusive
cross-process lock (`.kernelopt/gpu.lock`), so a campaign and a single run — or
two campaigns — serialize on the device instead of corrupting each other's timing
or VRAM.

## Requirements

- An NVIDIA GPU (or AMD with ROCm) + recent driver.
- CUDA toolkit (`nvcc`, `nsys`, `ncu`) — or ROCm (`hipcc`).
- Rust stable, Python 3.10+, `cmake` ≥ 3.28, `ninja`/`make`, `git`.
- A target checkout (ninfer, llama.cpp, or your own repo) that builds.
- An LLM key + a model with tool calls.

Full ordered checklist: **[docs/prereqs.md](docs/prereqs.md)** — driver,
toolkit, profiling permission, build tools, LLM key, verify.

## Build

```bash
cargo build --release        # binary: target/release/kernelopt
```

## Setup

New machine? Start with the ordered checklist: **[docs/prereqs.md](docs/prereqs.md)**
(GPU driver → CUDA toolkit → profiling permission → build tools → target
checkout → LLM key + tool-call model → verify). Then:

```bash
cp .env.example .env     # set NINFER_REPO / LLAMACPP_REPO / KERNELOPT_MODEL / OPENCODE_API_KEY
kernelopt setup          # full prereq table
kernelopt setup --smoke  # prove the compiler works (seconds, no GPU needed)
```

`.env` is read automatically (real env wins), so `--repo`/`--provider`/`--model`
are optional. If you pass `--repo "$NINFER_REPO"`, export it first
(`set -a; source .env; set +a`) — otherwise the shell expands it to empty.

## Documentation lookup (CUDA/HIP API docs)

When a candidate fails to compile, KernelOPT looks the offending symbol up and
attaches it to the Executor's retry:

```bash
kernelopt docs cub::WarpMergeSort     # or: __reduce_max_sync, hipMalloc, rocwmma::…
kernelopt docs --login                # one-time NVIDIA login (token auto-refreshed)
```

Optional; falls back to local headers. Full detail:
[docs/docs-lookup.md](docs/docs-lookup.md).

## Quickstart

### First time? Use the wizard

```bash
kernelopt wizard            # detects repo/backend/targets, asks a few questions, runs
kernelopt wizard --dry-run  # just show the command it would run
kernelopt wizard --save     # also persist the choices to .env for later runs
kernelopt wizard --yes      # accept detected defaults, no prompts (scripts/CI)
```

It reads your `.env`, probes the optimizer LLM (auth + a forced tool call), lists
the discovered kernels and local models, offers `quick | standard | thorough`
presets, then prints and runs the exact command — so you can graduate to the
flags once you know what you want.

### One kernel

```bash
# Compiled PyTorch model (Triton kernels)
kernelopt run examples/mlp.py
kernelopt run examples/pointwise_fused.py

# ninfer Op — smallest kernel, good first run. --watch = one-terminal live view
kernelopt run-ninfer --op add_bias --iterations 3 --beam 2 --retries 3 --watch

# llama.cpp ggml-cuda kernel
kernelopt run-llamacpp --op SOFT_MAX --iterations 3 --beam 2 --watch
```

Every run prints a `run id` and writes `report.md` + `report.diff` under
`.kernelopt/runs/<run_id>/`. `--watch` shows progress in the same terminal and
prints those paths when the run finishes.

## Go further

**[docs/advanced.md](docs/advanced.md)** — harder measurements, engine
verification, custom engines, campaigns, profiling, run control, outputs,
testing, troubleshooting, repo layout.

## Documentation

| Doc | Contents |
|---|---|
| [docs/advanced.md](docs/advanced.md) | **Beyond the basics**: harder measurements, engine verification, campaigns, profiling, run control, troubleshooting |
| [docs/cli.md](docs/cli.md) | **Every command and option**: what `discover`, `profile`, `campaign`, `run*` do, the full flag reference, and campaign option effects |
| [docs/configuration.md](docs/configuration.md) | `.env`, providers, thinking level, `config.toml`, hyperparameters |
| [docs/prereqs.md](docs/prereqs.md) | Ordered setup checklist: driver, CUDA toolkit, profiling permission, build tools, LLM key, verify |
| [docs/outputs.md](docs/outputs.md) | Artifacts, run/campaign state, applying diffs, disk |
| [docs/campaign.md](docs/campaign.md) | Campaign walkthrough: discovery, ordering, budgets, resume |
| [docs/monitoring.md](docs/monitoring.md) | Live progress, what the LLM is doing, pausing (Ctrl-C), `watch`, campaign status |
| [docs/kernel-editing.md](docs/kernel-editing.md) | How kernels are read, edited (full-file), isolated, and surfaced as a diff |
| [docs/docs-lookup.md](docs/docs-lookup.md) | CUDA/HIP API docs lookup: NVIDIA `cuda-docs` MCP login, ROCm headers, `kernelopt docs` |
| [docs/testing-kernel-wins.md](docs/testing-kernel-wins.md) | RST recommendations: oracles, benchmark controls, measured-shape coverage and kernel versus application wins |
| [docs/model-e2e.md](docs/model-e2e.md) | Model-level (engine) verification: same tokens, not slower |
| [docs/ninfer-mode.md](docs/ninfer-mode.md) | ninfer mapping (gates, workbench, prompts) |
| [docs/llamacpp-mode.md](docs/llamacpp-mode.md) | llama.cpp mapping (`test-backend-ops` gates) |
| [docs/custom-mode.md](docs/custom-mode.md) | **custom mode**: optimize any CUDA repo via `kernelopt.toml` |
| [docs/graphsignal.md](docs/graphsignal.md) | Engine-share profiling, managed install, attribution rules |

## Outputs

Run artifacts live in `.kernelopt/runs/<run_id>/` (`journal.jsonl`,
`report.md`, `report.diff`, candidate/bench/ncu dumps); campaigns in
`.kernelopt/campaigns/<id>/`. Accepted candidates are committed and tagged in the
worktree (`kernelopt/<run_id>/<candidate>`); nothing is committed to **your** repo.

Apply the winner with `git apply`, inspect with `kernelopt history`, or revert with
`kernelopt revert` — details in [docs/advanced.md](docs/advanced.md#outputs).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

This project implements the method and adapts the agent prompts from
**KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization**
(Poddar, Prasad, Samanta, Chakraborty, Goyal, Rathaur; arXiv:2609.30059).

```bibtex
@misc{poddar2026kernelopt,
  title        = {KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization},
  author       = {Poddar, Aheli and Prasad, Sanskar and Samanta, Arindam and
                  Chakraborty, Subha and Goyal, Vishal and Rathaur, Rohit Singh},
  year         = {2026},
  eprint       = {2609.30059},
  archivePrefix= {arXiv},
  primaryClass = {cs.DC},
  url          = {https://arxiv.org/abs/2609.30059}
}
```

See [`CITATION.cff`](CITATION.cff) and [`NOTICE`](NOTICE). The Planner, Executor,
and Summarizer prompts under `prompts/` are adapted from the paper's appendix
(arXiv:2609.30059). Graphsignal: https://github.com/graphsignal/graphsignal
(fork adds llama.cpp/NInfer).
