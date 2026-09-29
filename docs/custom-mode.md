# KernelOpt — custom mode (any CUDA repo)

`ninfer` and `llama.cpp` are just built-in presets. **custom** mode optimizes
*any* CUDA kernel repo that declares how to build, test, and bench itself — no
KernelOPT code changes.

## `kernelopt.toml` (repo root)

```toml
[project]
name = "my-engine"
configure_args = ["-DCMAKE_BUILD_TYPE=Release", "-DMY_BUILD_TESTS=ON"]

[[target]]
op = "add_bias"                                  # the token you address it by
file = "src/ops/add_bias.cu"                     # the ONE file the Executor edits
kernel_files  = ["src/ops/add_bias.cu"]          # optional; defaults to [file]
context_files = ["src/ops/launcher.cu"]          # read-only (launcher/dispatch)
contract_files = ["include/add_bias.h"]          # semantic authority (read-only)
build_targets = ["my_add_bias_test", "my_add_bias_bench"]

# Gate 2 (correctness): exit 0 == correct.
test_cmd  = ["ctest", "--test-dir", "{build}", "-R", "my_add_bias_test"]

# Gate 4 (performance): run 3x, parse the pinned/representative shape.
bench_cmd = ["{build}/bench/my_add_bias_bench", "--csv-out", "{csv}"]
bench_format = "csv"                             # csv | stdout | llama
timing = true                                    # false => correctness-only
```

Placeholders: `{repo}`, `{build}`, `{csv}`.

## Using it

```bash
# What's optimizable?
kernelopt discover --repo /path/to/my-engine --mode custom --list

# Optimize one target (a one-target campaign)
kernelopt campaign --repo /path/to/my-engine --mode custom \
    --op add_bias --max-targets 1 --max-iterations 3 --watch

# Or the guided path (auto-detects kernelopt.toml)
kernelopt wizard --repo /path/to/my-engine
```

`--mode` is optional when the repo has `kernelopt.toml` (auto-detected first).

## What you get for free
The whole engine applies: the plan→edit→compile→correctness→bench loop, the LLM
agents, experience memory, beam search, the GPU lock, and the measurement rigor —
3 repeats on a pinned shape, interleaved fresh baseline/candidate rounds, the
noise floor, the sign test, the memory-roofline plausibility guard, and Gate 5
(measured-shape correctness, when a `run_case`-style test is found).

## Gates
- **Gate 1 (build)**: `cmake -S <repo> -B .kernelopt/custom/build <configure_args>`
  then `cmake --build --target <build_targets>`.
- **Gate 2 (correctness)**: your `test_cmd` — exit 0 passes.
- **Gate 3 (engine E2E)**: optional; pass `--e2e-cmd <your command>`.
- **Gate 4 (performance)**: your `bench_cmd`, parsed per `bench_format`.

## Profiles
Engine-share ranking (`kernelopt profile --mode custom`) and campaign
`--order engine` use **nsys** (default), **ncu**, or **graphsignal** — see
[monitoring.md](monitoring.md).

## Notes / limits
- `bench_format`: `csv` (op benches that emit `median_us`), `stdout`
  (`median= … us` lines), or `llama` (`N runs - T us/run`).
- Discovery is descriptor-driven on purpose: which file is editable, which header
  is the contract, and which test covers which launch path are project-specific.
- `build_targets` should include your test + bench targets so Gate 1 compiles them.
