# KernelOpt — ninfer Mode Design

How the KernelOpt agent loop (paper-faithful: profile → plan → edit → verify → gate → keep-or-revert)
maps onto `$NINFER_REPO`. Read this before touching the ninfer tree.

## 1. Target model

KernelOpt optimizes **CUDA kernel implementations behind ninfer Op contracts**. The paper's
torch.compile mapping carries over almost 1:1:

| KernelOpt concept | Paper (torch.compile) | ninfer |
|---|---|---|
| Optimization unit | one Inductor Triton kernel | one `.cu`/`.cuh` kernel file inside `src/ops/<family>/<impl>/` |
| Preserved vendor dispatch | `extern_kernels` (cuBLAS/cuDNN) | cuBLAS/cuDNN calls inside launchers — never rewritten |
| Verification authority | eager PyTorch + allclose | the Op's **own test suite** (`tests/ops/test_*.cpp`, criteria in `op_check.h`) |
| Timing authority | `do_bench` (subprocess-isolated) | `build/bench/ninfer_<op>_bench --csv-out` (cold-cache methodology built in) |
| Profiling | `ncu --set full` | `ncu --set full` — the bench binaries even ship `--profile` |
| Keep/abort decision | four-gate cascade | same cascade, ninfer-backed (below) |

Rule from `docs/maintainer/op-development.md` that shapes everything: **the contract in
`include/ninfer/ops/<family>.h` is the semantic authority**. The agent edits implementations,
never contracts, never wrapper semantics.

## 2. Workbench layout (all agent state lives in OUR tree)

```
.kernelopt/ninfer/
├── src/ops/...            # candidate copies of kernel files (git worktree, branch per run)
├── build/                 # a SECOND cmake build dir — the main tree's build is never reused
├── runs/<run_id>/
│   ├── journal.jsonl      # same journal as Triton mode
│   ├── candidates/        # every submission (pass or fail) + nvcc stderr
│   ├── ncu/               # raw CSV per profiled kernel
│   ├── bench/             # CSV outputs per bench invocation
│   └── report.md
```

**Git discipline (the ninfer tree is a clean checkout at a real commit):**
- `git worktree add` into `.kernelopt/ninfer/worktree` at the current HEAD; all edits happen there.
- One branch per op: `kernelopt/ninfer-<op>` (stable so the separate CMake build dir stays warm —
  Ninja only rebuilds files we actually touch). The user's working tree is never touched.
- A candidate that survives all four gates is surfaced as a patch (`git diff`) for the user to
  accept — KernelOpt does not commit to the user's tree, matching "compiler baseline preserved."
- Teardown resets the worktree (discarding candidate edits) but leaves it and the build dir in
  place for the next run; delete `.kernelopt/ninfer/` to reclaim disk. Concurrent runs on the
  same op are out of scope for v1.

## 3. Runner commands (new, in `runner/kernelopt_runner/`, protocol v1)

All of the following are implemented in `runner/kernelopt_runner/ninfer.py` and dispatched by
`__main__.py` (`cuda_*` prefix). `cuda_worktree` manages the isolated worktree; `cuda_engine_bench`
is the optional whole-engine Gate 3 wrapper around `ninfer_bench`.

### `cuda_worktree` — isolation
```
in:  {repo, worktree_dir, branch, base, action: create|remove|reset|status}
out: {worktree, branch, base, head, reused?}
```
`create` reuses an existing worktree already on `branch` (warm build cache) and otherwise adds a
linked worktree; `reset` discards edits/untracked files; `remove` deletes worktree + branch.

### `cuda_compile` — Gate 1 (static validation)
```
in:  {worktree, kernel_files[], build_target}
do:  cmake --build .kernelopt/ninfer/build --target <op test + bench targets> -j
out: {passed, compiler_errors[]}          # nvcc stderr, deduped, truncated per error
```
nvcc template errors are the #1 candidate killer (CUTLASS/CuTe-style code) — errors are fed back
verbatim to the Executor, same loop as Triton compile errors.

### `cuda_verify` — Gate 2 (correctness, borrowed authority)
```
in:  {worktree, op, criterion_note}
do:  ctest the op's own test suite (e.g. tests/ops/test_linear_add*) in the worktree build
out: {passed, suite_output, failing_cases[]}
```
**Key decision:** we do NOT invent an allclose layer. ninfer's suites already encode per-Op
criteria (fp32 accumulate in double, bf16 rounding boundaries, per-family tolerances). The paper's
multi-seed requirement maps to the suites' deterministic-input seeding; where a suite has a seed
flag we sweep 3 seeds. Gate 2 verdict = the repo's own tests.

### `cuda_bench` — Gate 4 (performance)
```
in:  {worktree, op, t_values[], warmup, repeat}
do:  run build/bench/ninfer_<op>_bench --csv-out <run>/bench/<ts>.csv [--warmup --repeat]
out: {median_ms per t, effective_gbs, tflops}   # parsed from the bench's own CSV
```
Bench methodology is ninfer's own (cold-cache flush, warmup/repeat defaults, T sweep) — we reuse
it wholesale instead of writing a cudaEvent harness. Gate 4 verdict: median across the T sweep,
γ = 1.03 noise margin, ≥2 targeted retries on rejection (paper-faithful).

### `cuda_ncu` — planner context (attribution tier)
```
in:  {worktree, op, ncu_set, launch_skip, launch_count}
do:  ncu --set full --csv ... ninfer_<op>_bench --profile (single T; the flag requires it)
out: {kernels[{name, sol_memory, sol_compute, duration_us, registers, occupancy}], rules top-3}
```
Same typed context + analyst as Triton mode (identical SOL thresholds). CUDA kernel names are real
symbols — no mangling, so `kernel_name` filtering actually works here (easier than Triton).

### `cuda_diff` — final artifact
```
out: unified diff of the worktree vs base commit (the deliverable the user reviews)
```

## 4. Gate cascade (ninfer mapping)

1. **V_stat**: `cuda_compile` (nvcc clean).
2. **V_corr**: `cuda_verify` (repo's own Op tests, 3 seeds where seeded).
3. **V_model**: the paper's E2E model check maps to `ninfer_bench` (whole-engine route) when the
   op is on the engine path — catches the paper's "faster kernel, slower model" trap (dispatch
   overhead, lost fusion). Optional per-run flag for expensive models; default on for P2+ ops.
4. **V_perf**: `cuda_bench` sweep ≤ γ × baseline sweep.

No candidate passes → worktree discarded, report explains root cause (library dominance /
no headroom / correctness / perf-gate), user tree untouched.

## 5. Agent prompts (new assets, paper discipline preserved)

- `prompts/cuda-executor.md` — the paper's Executor rules (signature identity, minimal diff,
  no restructuring) restated for CUDA:
  - edit ONLY the target kernel file; launcher/wrapper/contract files are read-only context;
  - never change the Op contract or observable numerics (contract header quoted in prompt);
  - preserve fp32 accumulation on reductions/dot products (ninfer's numerical principles);
  - never replace hand-written kernels with cuBLAS/cuDNN/thrust calls (anti-cheating, matches
    the paper's CodeGen bans);
  - no new dependencies, no build-system edits, `static_assert`/contracts respected;
  - templates: full type signatures, no `auto` shorthand in kernel signatures (nvcc error hygiene).
- `prompts/cuda-planner.md` — the paper's Planner structure (diagnose → inspect → memory →
  ONE plan → submit) with CUDA strategy list: occupancy/launch config, vectorized loads
  (`__ldg`, 128-bit), shared-mem staging, register pressure, warp-level primitives, persistent
  kernels, `cp.async`/TMA where arch-appropriate, algorithmic rewrites (skip-empty segments,
  split-K choices). TF32/BF16-accumulation warning carries over verbatim.
- Memory/AVOID tracker: unchanged; kernel_type = op family (linear_add, softmax_attention, …).

## 6. Pipeline flow

```
kernelopt run-ninfer --op linear_add --repo "$NINFER_REPO" \
    --provider opencode-go --model glm-5.3-flash --reasoning-effort low
```

1. `discover`: parse `src/ops/<family>/` — kernel files, launcher, contract header, test names,
   bench binary name (from `bench/ops/benchmarks.cmake`).
2. `baseline`: build clean → run tests (must pass before anything) → bench sweep → ncu context.
3. Beam loop (paper Algorithm 1: T×N×B×K, DiverseSelect, meltdown detector): plan → edit
   worktree kernel → compile → test → bench → summarize→memory.
4. Gates 3–4: `ninfer_bench` E2E (optional) + perf sweep γ=1.03.
5. Report + `cuda_diff`; journal every step; resumable via `kernelopt resume` (unchanged).

## 7. Phased targets (op families, easiest → hardest)

| Phase | Op family | Why first | Risk |
|---|---|---|---|
| P0 | `add_bias`, `cast`, `argmax` | trivial pointwise/reduction kernels; suites tiny; loop exercises end-to-end | low |
| P1 | `bf16_linear_add` | single kernel + clear roofline (`kFp8Fp32AccumulatePeak`-style % metrics in bench CSV) | low-med |
| P2 | `fp8_linear_add`, `fp8_linear_swiglu` | quantized paths, K-split variants — real headroom (the repo's own commit log shows perf iteration here) | med |
| P3 | `softmax_attention` (packed/causal variants) | MMA/tile code, CuTe-adjacent; compile-heavy; biggest wins but slowest loop | high |
| P4 | CUDA-graph E2E | graphsignal `--cuda-graph-trace node` on `ninfer_bench`/serve to rank ops by real engine share | med |

P4 closes the loop with the paper's biggest lesson: optimize what the *engine* actually spends
time on, not what looks slow in isolation.

## 8. Deliberately out of scope (v1 of ninfer mode)

- Editing `include/ninfer/ops/*.h` contracts (authority rule).
- Multi-GPU / NCCL paths (`allreduce.cu`).
- ROCm/HIP (graphsignal supports it; ninfer build here is CUDA 13 / sm_120a).
- Automatic PR creation — output is a reviewed patch + report, per the "baseline preserved"
  contract.

## 9. Verification of the tool itself

> **Status (implemented):**
> - `discover` — `kernelopt discover --repo <dir> [--mode auto|ninfer|llamacpp] [--op <token>|--list]`
>   (`src/ninfer.rs`, `src/llamacpp.rs`; fixture unit tests + read-only `tests/discover_real.rs`).
> - Runner commands (`runner/kernelopt_runner/ninfer.py`, `llamacpp.py`): `cuda_worktree`,
>   `cuda_compile`, `cuda_verify`, `cuda_bench`, `cuda_ncu`, `cuda_engine_bench`, `cuda_diff`,
>   `llama_verify`, `llama_bench`, `llama_ncu`.
> - Pipeline + CLI — `kernelopt run-ninfer|run-llamacpp --op <token> --repo <path> …`
>   (`src/cuda_pipeline.rs`, backend-agnostic); prompts `prompts/cuda-planner.md`,
>   `prompts/cuda-executor.md`.
> - Campaign mode — `kernelopt campaign --repo <dir>` (`src/campaign.rs`); see
>   [campaign.md](campaign.md). llama.cpp mapping in [llamacpp-mode.md](llamacpp-mode.md).
> - Walking skeleton `tests/ninfer_mock.rs` (real compile+ctest+bench, scripted
>   LLM) — `cargo test --test ninfer_mock -- --ignored --nocapture`.

- `kernelopt discover --op fp8_linear_add` → correct file/test/bench inventory (unit test
  with a fixture tree; integration test against the real repo, read-only). ✅
- `cuda_compile` on an intentionally broken kernel copy → structured nvcc errors. ✅
  (nvcc `file(line): error:` and gcc `file:line:col: error:` both parsed/deduped.)
- `cuda_bench` CSV parse vs a golden CSV fixture. ✅ (plus stdout fallback for benches
  without `--csv-out`, e.g. `add_bias`.)
- Full mock-LLM run on `add_bias` (compile+test+bench real, LLM scripted) — zero tokens. ✅
  (`tests/ninfer_mock.rs`, ignored; needs a pre-built worktree build dir.)

## 10. Running ninfer mode

One-time: the build dir must be configured **against the worktree** (source paths are
baked into CMake/Ninja, so the main `ninfer/build` cannot be reused):

```
kernelopt run-ninfer --op add_bias --repo "$NINFER_REPO" \
    --provider opencode-go --model glm-5.3-flash --reasoning-effort low \
    --iterations 5 --beam 4 --retries 4
```

Artifacts land in `.kernelopt/runs/<run_id>/`: `journal.jsonl`, `candidates/`,
`bench/*.csv`, `ncu/*.csv`, `report.diff`. The user's ninfer tree is never modified;
the surviving change is the unified diff for review. `run-ninfer` accepts
`--kernel-file`, `--build-dir`, repeated `--bench-arg` (e.g. `--bench-arg --t-sweep
--bench-arg 1,8`), and `--e2e-weights <artifact.ninfer>` to enable the whole-engine
Gate 3.

Deliberate simplification (v1): the optimization unit is a single kernel file (the
first from `discover`, or `--kernel-file`); Gate 3 is skipped unless `--e2e-weights`
is supplied.
