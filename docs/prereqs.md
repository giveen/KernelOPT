# KernelOpt — Prerequisites (ordered checklist)

Get these in order. Each step ends with how to verify it. When everything is
green, the wizard gates on it automatically — but walking this list first saves
a failed run. `kernelopt setup` checks most of it; `kernelopt setup --smoke`
proves the compiler works.

## 0. Hardware

- An **NVIDIA GPU** with a recent driver, or an **AMD GPU** with ROCm (ROCm
  support is newer; CUDA is the well-tested path).
- **VRAM**: the optimizer itself needs little, but Gate 3 (engine E2E) loads a
  whole model — a 27B model needs ~30 GB free. Without it, skip `--e2e-weights`.
- **Disk**: CUDA toolkit (~5 GB) + target checkouts and their builds (several
  GB each under `.kernelopt/`).

## 1. GPU driver

NVIDIA: install a recent driver for your distro
([CUDA toolkit downloads](https://developer.nvidia.com/cuda-downloads) bundle
one), then:

```bash
nvidia-smi   # must list your GPU
```

If `nvidia-smi` fails, nothing below matters — fix the driver first.

## 2. CUDA toolkit (nvcc, nsys, ncu)

Install a recent CUDA 12/13 toolkit so `nvcc`, `nsys`, and `ncu` are on `PATH`
(or set `CUDA_HOME`/`CUDA_PATH`):

```bash
nvcc --version   # Cuda compilation tools, release 13.x
ncu --version
nsys --version
```

`ncu` (Nsight Compute) is what profiles kernels for the planner; `nsys`
(Nsight Systems) is used for engine-share attribution. Both ship with the
toolkit — there is no standalone install.

## 3. Non-root GPU profiling permission (NVIDIA)

`ncu --set full` reads GPU performance counters, which the driver restricts by
default. Check:

```bash
cat /proc/driver/nvidia/params | grep -i profil   # want RmProfilingAdminOnly: 0
```

If it is `1` and you are not root, profiling fails. Options (pick one):

- run as root, or
- grant `CAP_SYS_ADMIN`, or
- open profiling persistently (needs a reboot):

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminHost=0" \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u    # Debian/Ubuntu (RHEL/Fedora: rebuild initramfs with dracut)
sudo reboot
```

`kernelopt setup` reports this as `ncu profiling (perf counters)`.

## 4. ROCm path (AMD instead of NVIDIA)

Install ROCm so `hipcc` and `rocm-smi` are available. KernelOPT detects `hipcc`
automatically; `ncu`/`nsys` rows in `setup` don't apply. HIP docs lookup
auto-enables when ROCm is detected (see [docs-lookup.md](docs-lookup.md)).

## 5. Build tools and languages

```bash
git --version
cmake --version    # need >= 3.28 (ninfer requires it)
ctest --version    # ships with cmake
ninja --version    # or: make --version (either build runner is fine)
python3 --version  # need >= 3.10 (the GPU runner)
```

Plus the **Rust toolchain** to build KernelOPT itself:

```bash
cargo build --release   # produces ./target/release/kernelopt
```

## 6. Target checkout (the code to optimize)

```bash
git clone <ninfer-or-llamacpp-url> ~/ninfer   # or wherever you keep it
```

The checkout itself must build — KernelOPT builds it in an isolated worktree,
but it can't fix a broken tree. If the target's own build fails, stop here.

## 7. LLM provider: key + a model with tool calls

```bash
cp .env.example .env   # then edit it
```

```bash
KERNELOPT_PROVIDER=opencode-go
KERNELOPT_MODEL=<a model that supports tool/function calls>
OPENCODE_API_KEY=sk-...
```

The pipeline *requires* tool calls (planner/executor submit via tools). Verify
before spending anything:

```bash
kernelopt providers               # auth OK? model listed?
kernelopt providers --model NAME  # completion probe + tool-call probe
```

A model that answers but never calls tools will stall every run — the wizard
probes this and now refuses to accept a broken model by default.

## 8. Verify everything at once

```bash
kernelopt setup          # full table: required / recommended / optional
kernelopt setup --smoke  # compiles a minimal kernel: proves nvcc+headers work
```

Then start with the wizard, which re-checks required tools before spending
anything:

```bash
kernelopt wizard
```

## 9. Optional extras

| What | How | Why |
|---|---|---|
| `codebase-memory-mcp` | `kernelopt setup --install` | Structural call-graph context for the planner (`kernelopt map`) |
| Graphsignal venv | `kernelopt setup-graphsignal` | Engine-share profiling backend |
| CUDA docs MCP | `kernelopt docs --login` | Authoritative CUDA/CUTLASS/CCCL docs on compile errors ([docs-lookup.md](docs-lookup.md)) |
| Engine E2E model | `--e2e-weights <model>` / `kernelopt models` | Gate 3: proves a kernel win helps a real model ([model-e2e.md](model-e2e.md)) |

## Troubleshooting (symptom → fix)

| Symptom | Fix |
|---|---|
| `ncu` fails with `ERR_NVGPUCTRPERM` / permission errors | Step 3 (profiling permission) |
| `cmake` errors about minimum version | Step 5 (`cmake >= 3.28`) |
| Baseline bench exits 1 with a usage error | Pass the bench's required flags via `--bench-arg` (KernelOPT auto-detects most; see [cli.md](cli.md)) |
| `cudaMalloc failed: out of memory` at engine baseline | Model too big for VRAM: smaller model, free memory, or drop `--e2e-weights` (Gate 3 is optional) |
| Planner never calls tools / run stalls at plan time | Step 7: switch to a model with tool-call support |
| `codebase-memory-mcp not found` | Step 9 (`setup --install`) or `KERNELOPT_CODEMAP=0` to skip it |
| Wizard refuses to start (missing required tool) | Read the hint it prints; `kernelopt setup` for the full table |

Still stuck? The per-stage errors in `.kernelopt/runs/<id>/report.md` and the
journal (`kernelopt status <id>`) say exactly which gate failed and why.
