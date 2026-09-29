# KernelOPT

Dispatch-aware, agentic GPU-kernel optimization. Point KernelOPT at a **directory
of kernels**; it profiles them and drives a profile → plan → edit → verify → gate
loop until each is faster — or a budget runs out. Your working tree is never
touched: edits happen in an isolated git worktree, every accepted candidate is
**committed and tagged** (so it can be read or reverted instantly), and the winner
is handed back as a unified diff.

An implementation of *KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel
Optimization* (arXiv:2609.30059), extended to CUDA kernel trees.

| Mode | Target tree | Optimization unit | Correctness | Timing |
|---|---|---|---|---|
| `triton` | compiled PyTorch model | one Inductor `@triton.jit` kernel | eager PyTorch + `allclose` | `triton.testing.do_bench` |
| `ninfer` | ninfer checkout | one `src/ops/<family>/<impl>/*.cu` | the Op's own `ctest` suite | `ninfer_<op>_bench` |
| `llamacpp` | llama.cpp checkout | one `ggml/src/ggml-cuda/*.cu` | `test-backend-ops test -o <OP>` | `test-backend-ops perf -o <OP>` |

## How it works

```
discover ─▶ baseline ─▶ ┌ plan ─▶ edit ─▶ compile ─▶ correctness ─▶ bench ┐ ─▶ γ/roofline gate ─▶ shape gate ─▶ diff
                        └────────── iterate until patience/target/budget ──┘
```

LLM agents (Planner → Executor → Summarizer) work with an NCU/Graphsignal-guided
beam search and shared experience memory. Five gates filter candidates: static
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
(rejected). The reported number is always explicit (`N×` = baseline/final, with
the shape, ms values, noise, and bandwidth-vs-roofline).

**One GPU job at a time:** bench/verify/NCU/engine-E2E take an exclusive
cross-process lock (`.kernelopt/gpu.lock`), so a campaign and a single run — or
two campaigns — serialize on the device instead of corrupting each other's timing
or VRAM.

## Requirements

- **Rust** (stable, edition 2021) + Cargo.
- **Python 3** (stdlib only for the CUDA modes; PyTorch/Triton for `triton` mode).
- For `ninfer`/`llamacpp`: `cmake`, `ninja`, `nvcc`, a CUDA GPU, and a configured
  build of the target repo.
- Optional: `ncu` for profiling context; Graphsignal (auto-provisioned) for
  engine-share attribution. GPU counters may need `NVreg_RmProfilingAdminOnly=0`.

## Build

```bash
cargo build --release        # binary: target/release/kernelopt
```

## Setup

```bash
cp .env.example .env     # set NINFER_REPO / LLAMACPP_REPO / KERNELOPT_MODEL / OPENCODE_API_KEY
```

`.env` is read automatically (real env wins), so `--repo`/`--provider`/`--model`
are optional. If you pass `--repo "$NINFER_REPO"`, export it first
(`set -a; source .env; set +a`) — otherwise the shell expands it to empty.

## Examples

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

### Measure harder, or pin the shape

```bash
# Gate 4 measures the slowest shape by default; pin one and add a final round
kernelopt run-ninfer --op add_bias --bench-shape "4304,4096" --final-rounds 3 --watch

# Executor submits a unified diff instead of the whole file (opt-in)
kernelopt run-ninfer --op add_bias --edit-mode patch --watch
```

### Whole-engine (model-level) verification

```bash
# What local models can I test against? (searches <repo>/models, KERNELOPT_MODELS_DIR, ~/models)
kernelopt models

# Gate 3: the candidate must produce the same tokens and not be slower, end to end
kernelopt run-ninfer --op bf16_linear_add --e2e-weights qwen3_8_27b_nvfp4 --watch  # by name
kernelopt run-ninfer --op bf16_linear_add --e2e-weights auto --watch              # newest local
kernelopt run-llamacpp --op SOFT_MAX --e2e-weights /path/to/model.gguf --watch
```

`--e2e-weights` takes a path, a bare name (from `kernelopt models`), or `auto`;
it also reads `KERNELOPT_E2E_WEIGHTS` from `.env`.

`--e2e-cmd` / `--profile-cmd` consume the rest of the argv — put them **last**.

### Sweep a directory (campaign)

```bash
# Discover + optimize every target; stop after 8h or 3 stalled targets
kernelopt campaign --mode ninfer --patience 3 --target-speedup 1.15 \
    --budget-hours 8 --max-iterations 4 --watch

# Restrict to specific ops (--op is repeatable) / resume a previous campaign
kernelopt campaign --mode ninfer --op add_bias --op gelu --watch
kernelopt campaign --mode ninfer --resume 20260929_063819_ninfer --watch
```

`Ctrl-C` pauses gracefully; `--watch` shows the campaign status, and
`kernelopt status <id> --campaign` re-checks later.

### Discover and profile

```bash
# What's optimizable here? (one target, or every target)
kernelopt discover --list
kernelopt discover --op add_bias

# What local models can I run the engine-E2E against?
kernelopt models --mode ninfer

# Rank kernels by real engine share (Graphsignal; attribution only)
kernelopt profile --mode llamacpp --cuda-graph-trace node \
    --cmd "$LLAMACPP_REPO/build/bin/test-backend-ops" perf -o SOFT_MAX -b CUDA0

# Provision the profiler into .kernelopt/graphsignal/venv (idempotent)
kernelopt setup-graphsignal
```

### Inspect and control a run

```bash
kernelopt watch   --latest           # live progress; stops when the run finishes and
                                     # prints report.md / report.diff / journal paths
                                     # (--follow to keep tailing, --once to dump and exit)
kernelopt status  <run_id>            # events / candidates / tokens (--campaign for campaigns)
kernelopt report  <run_id>            # report + winner: what changed / why faster
kernelopt history <run_id>            # candidate commits (sha, latency, plan)
kernelopt revert  <run_id> --to <ref> # reset the worktree to a candidate
kernelopt analyze <run_id>            # failure taxonomy, retries, tokens, plans, winner
kernelopt resume  <run_id>            # resume an interrupted run from its journal
kernelopt eval    <campaign_id>       # self-benchmark: win rate, speedups, cost
```

Providers: `opencode-go` (default), `openai`, `openrouter`, `ollama`, `vllm`,
`lmstudio`, `mock`. `--reasoning-effort` / `--thinking` (default `low`) sets the
thinking level. See `.env.example` for all keys.

## Documentation

| Doc | Contents |
|---|---|
| [docs/cli.md](docs/cli.md) | **Every command and option**: what `discover`, `profile`, `campaign`, `run*` do, the full flag reference, and campaign option effects |
| [docs/configuration.md](docs/configuration.md) | `.env`, providers, thinking level, `config.toml`, hyperparameters |
| [docs/outputs.md](docs/outputs.md) | Artifacts, run/campaign state, applying diffs, disk |
| [docs/campaign.md](docs/campaign.md) | Campaign walkthrough: discovery, ordering, budgets, resume |
| [docs/monitoring.md](docs/monitoring.md) | Live progress, what the LLM is doing, pausing (Ctrl-C), `watch`, campaign status |
| [docs/kernel-editing.md](docs/kernel-editing.md) | How kernels are read, edited (full-file), isolated, and surfaced as a diff |
| [docs/model-e2e.md](docs/model-e2e.md) | Model-level (engine) verification: same tokens, not slower |
| [docs/ninfer-mode.md](docs/ninfer-mode.md) | ninfer mapping (gates, workbench, prompts) |
| [docs/llamacpp-mode.md](docs/llamacpp-mode.md) | llama.cpp mapping (`test-backend-ops` gates) |
| [docs/graphsignal.md](docs/graphsignal.md) | Engine-share profiling, managed install, attribution rules |

## Outputs

Run artifacts live in `.kernelopt/runs/<run_id>/` (`journal.jsonl`,
`report.md`, `report.diff`, candidate/bench/ncu dumps); campaigns in
`.kernelopt/campaigns/<id>/`. Accepted candidates are committed and tagged in the
worktree (`kernelopt/<run_id>/<candidate>`); nothing is committed to **your** repo.

```bash
git -C "$NINFER_REPO" apply .kernelopt/runs/<run_id>/report.diff
kernelopt history <run_id>    # read the candidate commits
kernelopt revert  <run_id> --to <sha|tag>
```

Full layout in [docs/outputs.md](docs/outputs.md); the git model in
[docs/kernel-editing.md](docs/kernel-editing.md).

## Testing

```bash
cargo test                                   # Rust unit + discovery tests
cargo test --test ninfer_mock -- --ignored   # real compile+ctest+bench, scripted LLM
NINFER_REPO=… LLAMACPP_REPO=… python3 -m pytest runner/tests/ -q
```

Tests needing a repo/build skip when the env vars are unset.

## Troubleshooting

- **`Error: repo not found:` (empty path)** — you passed `--repo "$NINFER_REPO"`
  but the variable isn't set in your *shell* (it's only in `.env`, which the tool
  reads itself). Omit `--repo` (`.env` supplies it), or load it:
  `set -a; source .env; set +a`.
- **`--e2e-cmd` / `--profile-cmd` swallowed your next flag** — they consume the
  rest of the argv; place them **last**.
- **`ncu` yields no data / `ERR_NVGPUCTRPERM`** — counters are admin-only; set
  `NVreg_RmProfilingAdminOnly=0` and reload the driver. KernelOPT degrades
  gracefully (runs without NCU context).
- **Wrong repo / no targets** — `kernelopt discover --repo <dir> --list`; pass
  `--mode ninfer|llamacpp` if auto-detection is ambiguous.
- **Baseline fails** — KernelOPT refuses to optimize a red tree; the run ends
  `fallback` with the cause in `report.md`.
- **Cold first build is slow** — the worktree build is configured once, then
  reused per op. Delete `.kernelopt/<backend>/` to reclaim disk.
- **A run pauses before benching** — another `kernelopt` process holds the GPU
  lock (`.kernelopt/gpu.lock`). Bench/verify/NCU/E2E serialize on it so the device
  only runs one job; wait for the other run, or remove a stale lockfile if no
  process is using it.
- **The model never calls the tools (`no_tool_call` retries)** — the pipeline
  drives the model entirely through tool calls. Verify your provider/model with
  `kernelopt providers --provider <p> --model <m>`; its **tool-call probe** shows
  whether forced tool calls work. The client auto-falls-back to `tool_choice:
  "auto"` then no `tool_choice`, but a model that cannot emit tool calls at all
  cannot drive the pipeline.

## Repository layout

```
src/
  main.rs            CLI
  cuda_pipeline.rs   backend-agnostic profile→plan→edit→verify→gate loop
  pipeline.rs        compiled-model (triton) pipeline
  campaign.rs        directory→targets→loop driver, persistent state
  backend.rs         Backend/Target abstraction + auto-detection
  attribution.rs     Graphsignal kernel→target engine-share attribution
  ninfer.rs          ninfer discovery        llamacpp.rs  llama.cpp discovery
  tools.rs           Planner retrieval (ripgrep search / bounded reads)
  signals.rs         /signals summarizer      analyst.rs   NCU bottleneck tier
  dotenv.rs          .env loader             memory.rs    experience memory
  journal.rs         append-only run log      gpu_lock.rs  cross-process GPU lock
runner/kernelopt_runner/   Python runner (cuda_*, llama_*, graphsignal_*, engine_*)
prompts/                   agent prompts
docs/                      this directory
```

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
and Summarizer prompts under `prompts/` are adapted from the paper's appendix.

## References

- Paper: **KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization**,
  arXiv:2609.30059.
- Graphsignal: https://github.com/graphsignal/graphsignal (fork adds llama.cpp/NInfer).
