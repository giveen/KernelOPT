# KernelOpt — Advanced guide

You know `setup` and the `wizard`; this is everything after that: harder
measurements, engine verification, custom engines, campaigns, profiling,
run control, the docs-lookup deep dive, outputs, testing, troubleshooting,
and the repo layout.

## Measure harder, or pin the shape

```bash
# Gate 4 measures the slowest shape by default; pin one and add a final round
kernelopt run-ninfer --op add_bias --bench-shape "4304,4096" --final-rounds 3 --watch

# Executor submits a unified diff instead of the whole file (opt-in)
kernelopt run-ninfer --op add_bias --edit-mode patch --watch
```

## Whole-engine (model-level) verification

```bash
# What local models can I test against? (searches <repo>/models, KERNELOPT_MODELS_DIR, ~/models)
kernelopt models

# Gate 3: the candidate must produce the same tokens and not be slower, end to end
kernelopt run-ninfer --op bf16_linear_add --e2e-weights qwen3_8_27b_nvfp4 --watch  # by name
kernelopt run-ninfer --op bf16_linear_add --e2e-weights auto --watch              # newest local
kernelopt run-llamacpp --op SOFT_MAX --e2e-weights /path/to/model.gguf --watch
```

`--e2e-weights` takes a path, a bare name (from `kernelopt models`), or `auto`;
it also reads `KERNELOPT_E2E_WEIGHTS` from `.env`. See
[model-e2e.md](model-e2e.md).

`--e2e-cmd` / `--profile-cmd` consume the rest of the argv — put them **last**.

## Your own inference engine (custom)

```bash
# Declare build/test/bench in <repo>/kernelopt.toml, then:
kernelopt wizard --repo /path/to/my-engine            # auto-detects kernelopt.toml
kernelopt campaign --repo /path/to/my-engine --mode custom --op add_bias \
    --max-targets 1 --max-iterations 3 --watch
```

See [custom-mode.md](custom-mode.md) — the full engine (LLM loop,
measurement rigor, gates) applies to any CUDA repo, no code changes.

## Sweep a directory (campaign)

```bash
# Discover + optimize every target; stop after 8h or 3 stalled targets
kernelopt campaign --mode ninfer --patience 3 --target-speedup 1.15 \
    --budget-hours 8 --max-iterations 4 --watch

# Restrict to specific ops (--op is repeatable) / resume a previous campaign
kernelopt campaign --mode ninfer --op add_bias --op gelu --watch
kernelopt campaign --mode ninfer --resume 20260929_063819_ninfer --watch
```

`Ctrl-C` pauses gracefully; `--watch` shows the campaign status, and
`kernelopt status <id> --campaign` re-checks later. See
[campaign.md](campaign.md).

## Discover and profile

```bash
# What's optimizable here? (one target, or every target)
kernelopt discover --list
kernelopt discover --op add_bias

# What local models can I run the engine-E2E against?
kernelopt models --mode ninfer

# Rank kernels by real engine share (nsys default; --engine ncu|graphsignal)
kernelopt profile --mode llamacpp --engine nsys \
    --cmd "$LLAMACPP_REPO/build/bin/test-backend-ops" perf -o SOFT_MAX -b CUDA0

# Graphsignal is optional — only used for --engine graphsignal (the ROCm path)
kernelopt setup-graphsignal
```

## Inspect and control a run

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

## Documentation lookup (CUDA/HIP API docs)

When a candidate fails to compile, KernelOPT looks the offending symbol up in
CUDA/HIP documentation and attaches a short excerpt to the Executor's retry, so
the model fixes the API instead of re-guessing. You can also query it directly:

```bash
kernelopt docs cub::WarpMergeSort     # or: __reduce_max_sync, hipMalloc, rocwmma::…
```

Sources are tried in order — **stdio MCP → HTTP MCP → local headers** — and the
first hit wins; a miss at one source falls through to the next.

### NVIDIA CUDA docs (MCP, recommended)

One-time login (OAuth: dynamic client registration + PKCE); the token is cached
at `.kernelopt/docs_token.json` and **refreshed automatically**:

```bash
kernelopt docs --login     # opens an authorize URL; sign in with your NVIDIA account
kernelopt docs "cub::WarpMergeSort 64-bit keys"
```

Or supply a token directly (no login):

```bash
export KERNELOPT_DOCS_TOKEN=<access-token>
```

### AMD ROCm / HIP

Auto-enabled when ROCm is detected (`hipcc`/`rocm-smi`/`rocminfo`/`amdgpu-arch`
on `PATH`, `ROCM_PATH`/`HIP_PATH`, or `/opt/rocm`); HIP symbols are then looked up
in the local ROCm headers. To use AMD's `hip-docs-mcp` server instead, point at
its command:

```bash
export KERNELOPT_DOCS_MCP_CMD="uv run --directory /path/to/intellikit/rocm_mcp hip-docs-mcp"
```

Force on/off with `KERNELOPT_DOCS_ROCM=1` / `=0`. (AMD's `rocm-mcp` package
imports `amdsmi`, so the server needs ROCm installed to start.)

| Env var | Purpose |
|---|---|
| `KERNELOPT_DOCS_TOKEN` | Bearer token for the HTTP docs MCP (NVIDIA `cuda-docs`). Alias: `KERNELOPT_CUDA_DOCS_TOKEN` |
| `KERNELOPT_DOCS_URL` | Override the HTTP MCP endpoint (default: NVIDIA `cuda-docs`). Alias: `KERNELOPT_CUDA_DOCS_URL` |
| `KERNELOPT_DOCS_MCP_CMD` | stdio MCP command (e.g. `hip-docs-mcp`); tried before HTTP |
| `KERNELOPT_DOCS_ROCM` | `1`/`0` to force ROCm docs on/off (unset = auto-detect) |

Everything is optional: with no config, lookups fall back to the **local
CUDA/CCCL/ROCm headers** already on the machine, and only fire on a confident
API symbol. See [docs-lookup.md](docs-lookup.md).

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

Full layout in [outputs.md](outputs.md); the git model in
[kernel-editing.md](kernel-editing.md).

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
- **`ncu` yields no data / `ERR_NVGPUCTRPERM`** — counters are admin-only; see
  [prereqs.md](prereqs.md) (profiling permission).
  KernelOPT degrades gracefully (runs without NCU context).
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
  exec.rs            tool invocation (cmake/ctest/bench/ncu)   parse.rs  output parsers
  git.rs             worktree/diff/patch      models.rs    local-model discovery
  wizard.rs          guided setup
runner/kernelopt_runner/   Python runner (triton trace/verify/bench/ncu, engine_*, graphsignal_*)
prompts/                   agent prompts
docs/                      this directory
```

## References

- Paper: **KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization**,
  arXiv:2609.30059.
- Graphsignal: https://github.com/graphsignal/graphsignal (fork adds llama.cpp/NInfer).
