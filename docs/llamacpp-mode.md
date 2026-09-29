# KernelOpt — llama.cpp Mode

How the KernelOpt agent loop maps onto a llama.cpp checkout. The unit of
optimization is a CUDA kernel file under `ggml/src/ggml-cuda/`; the correctness
and timing authority is llama.cpp's own `test-backend-ops` binary.

Set `LLAMACPP_REPO` (or pass `--repo`) to your checkout; nothing here assumes a
particular filesystem layout.

## 1. Mapping

| KernelOpt concept | ninfer mode | llama.cpp mode |
|---|---|---|
| Optimization unit | `src/ops/<family>/<impl>/*.cu` | `ggml/src/ggml-cuda/<op>.cu` |
| Preserved vendor dispatch | cuBLAS/cuDNN in launchers | cuBLAS in `ggml-cuda.cu` (never rewritten) |
| Gate 1 static | `cmake --build … --target <op test+bench>` | `cmake --build … --target ggml-cuda test-backend-ops` |
| Gate 2 correctness | `ctest` the Op suite | `test-backend-ops test -o <OP> -b CUDA0` |
| Gate 4 timing | `ninfer_<op>_bench --csv-out` | `test-backend-ops perf -o <OP> -b CUDA0` (`us/run`) |
| Planner context | `ncu --set full` on the op bench | `ncu` on `test-backend-ops perf -o <OP>` |
| Final artifact | `git diff` of the worktree | `git diff` of the worktree |

The op→file mapping is curated in `src/llamacpp.rs` (`OPS`): a `.cu` filename does
not always equal its ggml op name (`MUL_MAT` → `mmq.cu`/`mmvq.cu`/`mmvf.cu`;
`SWIGLU` → `unary.cu`; `RMS_NORM`/`NORM`/`L2_NORM` → `norm.cu`). Ops marked
`perf = true` are in the `test-backend-ops perf` suite and therefore have a
timing authority (Gate 4); the rest run correctness-only.

## 2. Commands

```
kernelopt discover --repo "$LLAMACPP_REPO" --mode llamacpp --list
kernelopt run-llamacpp --op SOFT_MAX --repo "$LLAMACPP_REPO" --provider opencode-go --model <model>
kernelopt campaign --repo "$LLAMACPP_REPO" --mode llamacpp --patience 2 --target-speedup 1.2
```

The build dir must be configured **against the worktree** (Ninja bakes in source
paths), so the pipeline keeps its own build under `.kernelopt/llamacpp/`:

```
.kernelopt/llamacpp/
├── worktree/     # linked git worktree, branch kernelopt/llamacpp-<op> (reused)
└── build/        # CMake build dir, configured from the worktree
```

`run-llamacpp` builds with the shared CMake path and diffs via `src/git.rs`; the
gates run `test-backend-ops` and `ncu` directly from `src/exec.rs`, with output
parsed in `src/parse.rs` (the old `llama_*` Python commands are gone).

## 3. Gate cascade

1. **V_stat** — cmake builds `ggml-cuda` + `test-backend-ops` (nvcc/gcc
   diagnostics fed back to the Executor).
2. **V_corr** — `test-backend-ops test -o <OP> -b CUDA0`;
   `N/M tests passed` plus per-case FAIL lines are the verdict.
3. **V_model** — not wired in v1 (would need a GGUF and `llama-bench`).
4. **V_perf** — `test-backend-ops perf -o <OP> -b CUDA0`; parses
   `N runs - T us/run`; the median across cases must be ≤ γ × baseline.

Correctness-only ops (`perf = false`) pass Gate 2 and are reported as `matched`
(no timing gate).

## 4. Deliberately out of scope (v1)

- Editing headers shared across many ops (`common.cuh`, `ggml-cuda.cu` dispatch).
- ROCm/HIP, Metal, and other non-CUDA backends.
- Whole-model `llama-bench` Gate 3.
- Multi-GPU / `allreduce.cu`.
- Automatic PR creation — output is a reviewed patch + report.
