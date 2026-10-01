# KernelOpt — Command reference

Precedence for every setting: **CLI flag → environment/`.env` → `config.toml` →
built-in default**. `--repo` is required unless the matching env var
(`NINFER_REPO`/`LLAMACPP_REPO`) is set — there are no built-in repo paths.

## Command behavior

### `discover` — inventory, runs nothing

Scans a checkout and reports the optimizable targets. It never builds, edits, or
benchmarks anything.

- **Backend detection** (`--mode auto`): a tree with `src/ops` +
  `include/ninfer/ops` is ninfer; a tree with `ggml/src/ggml-cuda` is llama.cpp.
- **ninfer**: enumerates every op family and quant variant (e.g. `fp8_linear_add`),
  plus the flat ops under `src/ops/kernel/`. Variants are preferred over whole
  families, and only targets with a matching test are kept. It also resolves the
  contract header, launcher/wrapper files, the ctest names, and the bench binary.
- **llama.cpp**: maps ggml ops onto their `ggml/src/ggml-cuda` kernel file(s) via
  a curated table (a filename doesn't always equal the op name). `timing: false`
  marks ops outside the `test-backend-ops perf` suite (correctness-only).
- **Output**: `--op <TOKEN>` prints one `Target` JSON; `--list` prints every
  target. A `Target` is exactly what the pipeline consumes:

  | Field | Meaning |
  |---|---|
  | `backend`, `op`, `family`, `variant` | identity and grouping |
  | `kernel_files` | all editable `.cu`/`.cuh` files for the op |
  | `target_file` | the single file the Executor edits (first, or `--kernel-file`) |
  | `context_files`, `contract_files` | read-only context (launcher/wrapper/contract) |
  | `build_targets` | CMake targets built for Gate 1 |
  | `test_filters` | ctest names (ninfer) or `-o` op filters (llama.cpp) — Gate 2 |
  | `bench_binary`, `bench_args`, `timing` | Gate 4 authority |
  | `warnings` | e.g. "not in the perf suite", missing contract |

  For llama.cpp, `.cu` files not in the curated map are listed on stderr.

### `profile` — rank kernels by real engine share

Answers "what does the engine actually spend time on?" and is **attribution
only** (never a gate timing). `--engine auto|nsys|ncu|graphsignal` picks the
backend (`auto` = **nsys → ncu → graphsignal**).

- **nsys** (default): `nsys profile -t cuda` + `nsys stats` — kernels **and
  memcpy**, CUDA graphs included, no admin counters.
- **ncu** (fallback): profiles the workload and sums per-kernel `Duration`
  (kernel-only; needs `NVreg_RmProfilingAdminOnly=0`).
- **graphsignal**: `graphsignal-run` + `GET /signals` (auto-provisioned; the
  only backend with ROCm support).

Whichever backend, the per-kernel times are attributed to discovered targets by
matching op/family/variant tokens (most specific first); unmatched kernels are
reported separately. Prints the ranking (`op`, share %, time, kernels) or `--json`.

`--top` caps how many kernels are considered; `--cwd` sets the workload's working
directory. `--cuda-graph-trace` / `--no-setup` / `--source` apply to the
Graphsignal backend only. See [graphsignal.md](graphsignal.md).

### `campaign` — the self-looping driver

```
discover all targets → (optionally order by engine share) → for each target:
    create run dir + journal → CudaPipeline (baseline, then iterate)
    → persist state → next target → stop on budget / all done
```

It creates `.kernelopt/campaigns/<id>/state.json` (target queue + per-target
status/speedup/iterations/cursor) and `campaign.jsonl`, updated after every
target, so it is resumable. Experience memory and the strategy tracker are shared
across the targets of a run (not persisted across `--resume`). Each target's
artifacts land in `.kernelopt/runs/<run_id>/`.

Per target, the loop stops at whichever comes first:

- **patience** consecutive non-improving iterations,
- **target-speedup** reached,
- **max-iterations** hit,
- the global **budget** (time or LLM calls).

Patience/target/max-iterations finalize the target; a **budget** stop *pauses* it
(status stays `running`, checkpoint kept) so `--resume` continues it instead of
restarting. See [campaign.md](campaign.md#4-state-and-resume).

#### What each campaign option does

| Option | Effect |
|---|---|
| `--repo <DIR>` | The tree to scan and optimize. |
| `--mode <MODE>` | `auto` (detect), `ninfer`, or `llamacpp`. Sets discovery and the gate commands. |
| `--op <OP>` (repeatable) | Only process targets whose op **or** family matches (case-insensitive). E.g. `--op linear_add` runs every quant variant; `--op fp8_linear_add` runs just that one. |
| `--max-targets <M>` | Process at most M targets, then stop (the queue keeps the rest for `--resume`). `0` = all. |
| `--budget-hours <H>` | Wall-clock deadline for the whole campaign. Checked before each target and between iterations; a target stops mid-run when it expires. `0` = none. |
| `--budget-llm-calls <N>` | Stop the campaign once N LLM completions have been spent across all targets. The remaining budget is handed to each target. `0` = none. |
| `--patience <P>` | Per target: give up after P consecutive iterations that fail to improve the best latency by at least `--min-improvement`. `2` = paper-faithful. |
| `--target-speedup <X>` | Per target: stop early once `baseline / best ≥ X` (e.g. `1.2` = "got 20%, move on"). `0` = off. |
| `--min-improvement <F>` | The fractional gain that counts as an improvement and resets patience. `0.01` = 1%. Prevents noise from keeping patience at zero. |
| `--max-iterations <N>` | Hard safety cap on iterations per target (the campaign's loop cap; distinct from the shared `--iterations`). |
| `--order <complexity\|engine>` | `complexity` (default): simplest targets first (fewest kernel files). `engine`: run `--profile-cmd` once, rank targets by measured GPU share, and process the hottest ops first (unattributed targets last). |
| `--profile-cmd <ARGV…>` | The workload to run under Graphsignal for `--order engine` (e.g. `ninfer_bench --weights …` or `llama-server …`). **Place it last**; everything after it is the command. |
| `--profile-port <PORT>` | Graphsignal `/signals` port for `--order engine`. |
| `--profile-trace <node\|graph>` | CUDA-graph trace granularity for `--order engine` (`node` = per-kernel, needed to rank). |
| `--e2e-weights <PATH>` | Enables the whole-engine Gate 3 per target (ninfer only): the winner is re-checked with `ninfer_bench` on this `.ninfer` artifact. |
| `--ncu-set <full\|basic>` | How much NCU context the Planner sees (`full` = paper). |
| `--resume <ID>` | Continue a previous campaign: loads `state.json` (queue/cursor), `memory.json`/`tracker.json` (learning), and each interrupted target's `checkpoint.json`, so it skips finished targets and resumes a paused one mid-search. Discovery/filter/order are skipped because the queue already exists. See [campaign.md](campaign.md#4-state-and-resume). |
| `--quiet` | Suppress live per-iteration progress lines (stderr); target headers and the final summary remain. See [monitoring.md](monitoring.md). |

Plus the shared LLM and loop options (`--provider`, `--model`, `--reasoning-effort`,
`--iterations` T, `--plans` N, `--beam` B, `--retries` K) — these configure each
target's iterations, not the campaign's target cap.

**How the per-target limits interact** (example: `--patience 2
--target-speedup 1.2 --max-iterations 25 --budget-hours 8`): each target keeps
iterating until it has gone 2 iterations without a ≥1%-improvement, *or* it
reaches 1.2×, *or* it hits 25 iterations, *or* the 8-hour campaign deadline
arrives — whichever happens first.

### `run` / `run-ninfer` / `run-llamacpp` — one target, five gates

All three run the same pipeline on a single target and write a journal + report:

1. **Baseline** — isolated worktree, build, run the correctness suite (must be
   green), bench, profile.
2. **Loop** — Planner proposes, Executor edits the one kernel file, compile
   (Gate 1) → correctness (Gate 2) → bench; errors are fed back for up to K
   retries; every candidate is journaled.
3. **Finalize** — restore the winner, rebuild/re-test (Gate 1–2), then:
   - **Gate 4 perf** on a pinned shape with 3 repeats and interleaved fresh
     baseline/candidate rounds (`--final-rounds`); a win must clear the measured
     noise, win every round, and be *physically plausible* (not above a generous
     multiple of the memory roofline — otherwise it is "doing less work").
   - **Gate 5 measured-shape correctness** — the measured shape is appended to the
     op's test and run against baseline and candidate (ninfer; unsupported tests
     skip). Rejects a candidate that is correct on the suite but wrong at the shape.
   - for `run-ninfer`, optionally **Gate 3** model-level verification
     (`--e2e-weights`).
4. **Artifact** — a unified diff (`report.diff`) and `report.md`; the user's tree
   is never modified.

Bench/verify/NCU/E2E take an exclusive cross-process lock (`.kernelopt/gpu.lock`),
so only one GPU job runs at a time, even across separate `kernelopt` processes.

Differences: `run` operates on a compiled PyTorch model (Triton kernels, eager
`allclose` correctness, `do_bench` timing); `run-ninfer` on a ninfer Op (ctest +
op bench); `run-llamacpp` on a ggml-cuda kernel (`test-backend-ops test|perf`).
See [ninfer-mode.md](ninfer-mode.md), [llamacpp-mode.md](llamacpp-mode.md), and
[kernel-editing.md](kernel-editing.md) for how the file is read/edited/patched.

### `wizard` — guided setup

```bash
kernelopt wizard [--dry-run] [--yes] [--repo DIR] [--op OP] [--all] \
    [--preset quick|standard|thorough] [--e2e-weights NAME]
```

Detects the repo (`--repo`, else `NINFER_REPO`/`LLAMACPP_REPO`), the backend, and
the kernel targets; asks which target (or **all** → campaign), which optimizer
LLM (defaults from `.env`), an optional engine-E2E model (from `kernelopt
models`), and a loop preset; then prints and runs the equivalent command.

- `--yes` — accept detected defaults, no prompts (all targets, `standard` preset).
- `--dry-run` — print the command and exit without running it.
- `--save` — upsert the choices into `.env` (`NINFER_REPO`/`LLAMACPP_REPO`,
  `KERNELOPT_PROVIDER`, `KERNELOPT_MODEL`, `KERNELOPT_ITERATIONS`,
  `KERNELOPT_BEAM`, `KERNELOPT_PLANS`, and `KERNELOPT_E2E_WEIGHTS`), so later
  plain commands pick them up. Existing keys are rewritten in place; comments are
  preserved.
- The optimizer LLM is **probed** (auth + a forced tool call) before you accept
  it, so a broken model is caught before a run starts.
- Presets set `--iterations`/`--beam`/`--plans`: `quick` 2/1/2, `standard` 3/2/4,
  `thorough` 5/3/6 (persisted as `KERNELOPT_ITERATIONS`/`KERNELOPT_BEAM`/
  `KERNELOPT_PLANS`). Plans are 2× beam so UCB allocation has room (N > B).

### `setup-graphsignal` — provision the profiler

Installs Graphsignal into `.kernelopt/graphsignal/venv` (idempotent, nothing
system-wide). Default source is the fork that adds llama.cpp/NInfer launchers;
`--upstream`/`--source pypi` opt into upstream. Writes `setup.json` recording the
source, CUDA version, and whether the fork's launchers are present.
See [graphsignal.md](graphsignal.md).

---

## CLI reference

### Shared: LLM options

Used by `run`, `run-ninfer`, `run-llamacpp`, `campaign`.

| Option | Default | Meaning |
|---|---|---|
| `--provider <PRESET>` | `opencode-go` | Provider preset (`opencode-go`, `openai`, `openrouter`, `ollama`, `vllm`, `lmstudio`, `mock`). Env `KERNELOPT_PROVIDER`. |
| `--model <ID>` | `deepseek-v4-pro` | Model id at the provider. Env `KERNELOPT_MODEL`. |
| `--base-url <URL>` | preset | Override the OpenAI-compatible endpoint. Env `KERNELOPT_BASE_URL`. |
| `--api-key <KEY>` | preset env | API key. Env `KERNELOPT_API_KEY`, else the preset's key var (`OPENCODE_API_KEY`, …). |
| `--reasoning-effort <LEVEL>` / `--thinking` | `low` | Thinking level: `none\|minimal\|low\|medium\|high`. Env `KERNELOPT_REASONING_EFFORT`, `config.toml`. |

### Shared: loop options

Used by `run`, `run-ninfer`, `run-llamacpp`, `campaign`. Paper values in parentheses.

| Option | Default | Meaning |
|---|---|---|
| `--iterations <T>` | `5` | Beam iterations for a single-target run (paper T=5). Campaign uses `--max-iterations`. |
| `--plans <N>` | `4` | Expansions per iteration, allocated across the beam by UCB(c=1.4) (paper N=4). |
| `--beam <B>` | `4` | Beam width: frontier nodes kept per iteration by DiverseSelect (paper B=4). |
| `--retries <K>` | `4` | Executor retries per plan, fed compile/correctness errors (paper K=4). |

### `kernelopt run <MODEL_FILE>`

Optimize a compiled PyTorch model (file exposes `get_model()` and `get_inputs()`).

| Option | Default | Meaning |
|---|---|---|
| `<MODEL_FILE>` | — | Path to the model file (positional). |
| `--profiler <MODE>` | `both` | Profiling tier: `both` (Graphsignal + NCU), `ncu`, `graphsignal`, `none`. |
| `--ncu-set <SET>` | `full` | NCU section set (`full` = paper; `basic` for small-context models). |
| `--inputs-fn <NAME>` | `get_inputs` | Name of the inputs function in the model file. |

Plus the shared LLM and loop options.

### `kernelopt run-ninfer`

Optimize one ninfer Op in an isolated worktree.

| Option | Default | Meaning |
|---|---|---|
| `--op <TOKEN>` | required | ninfer op token, e.g. `add_bias`, `bf16_linear_add`, `fp8_linear_swiglu`. |
| `--repo <DIR>` | `$NINFER_REPO` | ninfer checkout. |
| `--kernel-file <PATH>` | first from `discover` | Override the editable kernel file (repo-relative). |
| `--build-dir <DIR>` | `.kernelopt/ninfer/build` | CMake build dir (must be configured against the worktree). |
| `--bench-arg <ARG>` | — | Extra argv forwarded to `ninfer_<op>_bench` (repeatable, e.g. `--bench-arg --t-sweep --bench-arg 1,8`). |
| `--e2e-weights <PATH>` / `--e2e-model` | off | Model enabling the model-level Gate 3: `.ninfer`, `.gguf`, or an HF safetensors dir. See [model-e2e.md](model-e2e.md). |
| `--e2e-engine <ENGINE>` | auto | Engine override: `ninfer`, `llamacpp`, `hf`, `vllm`, `sglang`. |
| `--e2e-cmd <ARGV…>` | — | Custom E2E command (place last); bypasses engine dispatch. |
| `--e2e-prompt <TEXT>` | `The capital of France is` | Prompt for the engine E2E generation. |
| `--e2e-max-new <N>` | `16` | Tokens generated for the engine E2E check. |
| `--ncu-set <SET>` | `full` | NCU section set for planner context. |
| `--quiet` | off | Suppress live per-iteration progress (stderr). |
| `--watch` | off | Tail the run's journal live in this terminal (single-terminal view). |
| `--edit-mode <MODE>` | `full` | `full` = Executor returns the whole file; `patch` = returns a unified diff applied with `git apply` (no reproduction drift). See [kernel-editing.md](kernel-editing.md). |
| `--bench-shape <SUBSTR>` | slowest | Pin the representative bench shape (substring of the bench row label). |
| `--final-rounds <N>` | `2` | Interleaved fresh baseline/candidate re-bench rounds at finalize (sign test). |

Plus the shared LLM and loop options.

### `kernelopt run-llamacpp`

Optimize one llama.cpp `ggml-cuda` kernel in an isolated worktree.

| Option | Default | Meaning |
|---|---|---|
| `--op <OP\|STEM>` | required | ggml op name (`SOFT_MAX`) or kernel stem (`softmax`). |
| `--repo <DIR>` | `$LLAMACPP_REPO` | llama.cpp checkout. |
| `--kernel-file <PATH>` | first from `discover` | Override the editable kernel file (repo-relative). |
| `--build-dir <DIR>` | `.kernelopt/llamacpp/build` | CMake build dir (configured against the worktree). |
| `--bench-arg <ARG>` | — | Extra argv forwarded to `test-backend-ops perf` (repeatable). |
| `--ncu-set <SET>` | `basic` | NCU section set for planner context. |
| `--e2e-weights <PATH>` / `--e2e-model` | off | Model for Gate 3: `.gguf`, `.ninfer`, or HF safetensors dir. |
| `--e2e-engine <ENGINE>` | auto | Engine override: `llamacpp`, `ninfer`, `hf`, `vllm`, `sglang`. |
| `--e2e-cmd <ARGV…>` | — | Custom E2E command (place last). |
| `--quiet` | off | Suppress live per-iteration progress (stderr). |
| `--watch` | off | Tail the run's journal live in this terminal (single-terminal view). |
| `--edit-mode <MODE>` | `full` | `full` = Executor returns the whole file; `patch` = returns a unified diff applied with `git apply` (no reproduction drift). See [kernel-editing.md](kernel-editing.md). |
| `--bench-shape <SUBSTR>` | slowest | Pin the representative bench shape (substring of the bench row label). |
| `--final-rounds <N>` | `2` | Interleaved fresh baseline/candidate re-bench rounds at finalize (sign test). |

Plus the shared LLM and loop options.

### `kernelopt campaign`

Point at a directory and optimize every discovered kernel in a resumable loop.
See "What each campaign option does" above.

| Option | Default | Meaning |
|---|---|---|
| `--repo <DIR>` | required | Target checkout. |
| `--mode <MODE>` | `auto` | Backend: `auto`, `ninfer`, `llamacpp`. |
| `--op <OP>` | all | Restrict to ops/families (repeatable). |
| `--max-targets <M>` | `0` | Stop after M targets. |
| `--budget-hours <H>` | `0` | Wall-clock budget. |
| `--budget-llm-calls <N>` | `0` | Stop after N LLM completions. |
| `--patience <P>` | `2` | Non-improving iterations before moving on. |
| `--target-speedup <X>` | `0` | Stop a target at `baseline/best ≥ X`. |
| `--min-improvement <F>` | `0.01` | Gain that resets patience. |
| `--max-iterations <N>` | `25` | Iteration cap per target. |
| `--e2e-weights <PATH>` | off | Model for the model-level Gate 3 (`.ninfer`, `.gguf`, or HF safetensors dir). |
| `--e2e-engine <ENGINE>` | auto | Engine override: `ninfer`, `llamacpp`, `hf`, `vllm`, `sglang`. |
| `--e2e-cmd <ARGV…>` | — | Custom E2E command (place last). |
| `--ncu-set <SET>` | `full` | NCU section set. |
| `--resume <ID>` | new | Resume a campaign by id. |
| `--order <MODE>` | `complexity` | `complexity` or `engine`. |
| `--profile-cmd <ARGV…>` | — | Workload for `--order engine` (place last). |
| `--profile-port <PORT>` | `18259` | `/signals` port for `--order engine`. |
| `--profile-trace <MODE>` | `node` | CUDA graph trace for `--order engine`. |
| `--quiet` | off | Suppress live per-iteration progress (stderr). |
| `--watch` | off | Tail the run's journal live in this terminal (single-terminal view). |
| `--edit-mode <MODE>` | `full` | `full` = Executor returns the whole file; `patch` = returns a unified diff applied with `git apply` (no reproduction drift). See [kernel-editing.md](kernel-editing.md). |
| `--bench-shape <SUBSTR>` | slowest | Pin the representative bench shape (substring of the bench row label). |
| `--final-rounds <N>` | `2` | Interleaved fresh baseline/candidate re-bench rounds at finalize (sign test). |

Plus the shared LLM and loop options.

### `kernelopt discover`

Inventory targets without running anything.

| Option | Default | Meaning |
|---|---|---|
| `--op <TOKEN>` | list all | Show one target. |
| `--repo <DIR>` | env | Target checkout. |
| `--mode <MODE>` | `auto` | Backend: `auto`, `ninfer`, `llamacpp`. |
| `--list` | off | List every runnable target (JSON). |

### `kernelopt profile`

Rank kernels by real engine share using Graphsignal (attribution only).

| Option | Default | Meaning |
|---|---|---|
| `--repo <DIR>` | env | Target checkout. |
| `--mode <MODE>` | `auto` | Backend: `auto`, `ninfer`, `llamacpp`. |
| `--cmd <ARGV…>` | required | Workload to run under `graphsignal-run` (place last). |
| `--listen-port <PORT>` | `18259` | Graphsignal `/signals` port. |
| `--cuda-graph-trace <MODE>` | `node` | `node` = per-kernel inside CUDA graphs; `graph` = whole replays. |
| `--top <N>` | `30` | Max kernels to rank. |
| `--cwd <DIR>` | current | Working directory for the workload. |
| `--no-setup` | off | Don't auto-provision Graphsignal. |
| `--source <SRC>` | fork | Auto-provision source: path, git URL, or `pypi`. |
| `--json` | off | Emit JSON (ranking + `/signals` summary). |

### `kernelopt setup-graphsignal`

Provision Graphsignal into KernelOPT's managed venv (no separate install).

| Option | Default | Meaning |
|---|---|---|
| `--cuda <12\|13>` | detected | CUDA major version for the install extra. |
| `--source <SRC>` | fork | Local checkout or git URL (default: the llama.cpp/NInfer fork). |
| `--version <V>` | latest | Pin a PyPI version. |
| `--upstream` | off | Install upstream PyPI (no llama.cpp/NInfer launchers). |
| `--force` | off | Reinstall even if already present. |

### `kernelopt providers` / `status` / `report` / `resume`

| Command | Options | Meaning |
|---|---|---|
| `providers` | `--provider`, `--model`, `--base-url`, `--api-key`, `--no-ping` | List presets and probe auth/models + a forced **tool call** (skip the probes with `--no-ping`). |
| `models` | `--mode`, `--repo`, `--json` | List local model artifacts usable for the engine-E2E (Gate 3) check; `--e2e-weights` then accepts a bare name or `auto`. |
| `status <RUN_ID>` | `--campaign` | Replay a run's journal (events, passing candidates, tokens); with `--campaign` summarize a campaign's target queue. |
| `watch [RUN_ID]` | `--latest`, `--once`, `--follow` | Tail a run's journal live; stops on finish and prints the artifact paths, `--follow` keeps tailing, `--once` prints and exits. |
| `history <RUN_ID>` | — | List a run's candidate commits (sha, status, latency, plan). |
| `revert <RUN_ID>` | `--to <sha\|tag>` | Reset the run's worktree to a candidate commit. |
| `analyze <RUN_ID>` | `--json` | Diagnose a run: failure taxonomy (compile/correctness/bench/no-tool-call), retries, token spend by agent, plan diversity, and the winner summary. |
| `eval <CAMPAIGN_ID>` | — | Aggregate a campaign into a **self-benchmark**: win rate, speedup distribution (median/max), failure taxonomy, and token cost. |
| `report <RUN_ID>` | — | Render the run's markdown report from the journal, ending with the **winner: what changed / why it's faster** (executor summary, planner evidence, measured speedup). |
| `resume <RUN_ID>` | — | Report the last recorded state of an interrupted run. |

See [monitoring.md](monitoring.md) for live progress, pausing (Ctrl-C), and
resuming.

### Environment / `.env`

See [configuration.md](configuration.md). Key vars: `NINFER_REPO`, `LLAMACPP_REPO`,
`KERNELOPT_PROVIDER`, `KERNELOPT_MODEL`, `KERNELOPT_BASE_URL`,
`KERNELOPT_API_KEY`, `KERNELOPT_REASONING_EFFORT`, `KERNELOPT_ENV_FILE`,
`KERNELOPT_GRAPH_SIGNAL_DIR`, `KERNELOPT_GRAPH_SIGNAL_SOURCE`,
`GRAPHSIGNAL_RUN`, and the provider key vars.
