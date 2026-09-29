# KernelOpt — Graphsignal integration (engine-share attribution)

> **Optional now.** Engine-share ranking defaults to the NVIDIA suite —
> **nsys** (`--engine nsys`) or **ncu** (`--engine ncu`) — which needs no
> Graphsignal fork, venv, or `/signals`. Use `--engine graphsignal` only when you
> want CUPTI sidecar attribution (e.g. ROCm, or a CUDA graph trace). See
> [cli.md](cli.md) `profile`.

[Graphsignal](https://github.com/graphsignal/graphsignal) is a GPU profiler that
observes an inference engine or any GPU process from a sidecar and serves
everything it measures at `http://127.0.0.1:<port>/signals` as JSON.

KernelOPT uses it to answer the paper's P4 question — **what does the engine
actually spend time on?** — and to order a campaign by real GPU share instead of
guessing. Graphsignal output is **attribution only**: it never becomes a gate
timing (those stay with the mode's own bench: `ninfer_<op>_bench` /
`test-backend-ops perf`).

## No separate install

KernelOPT provisions Graphsignal into its own managed venv; the user does not
install anything system-wide.

**Source matters:** upstream PyPI Graphsignal has **no llama.cpp or NInfer
launchers/recorders**. KernelOPT defaults to the fork that adds them
(`github.com/giveen/graphsignal`); override with `--source` or
`KERNELOPT_GRAPH_SIGNAL_SOURCE`.

```
kernelopt setup-graphsignal                    # fork (default), auto-detect CUDA 12/13
kernelopt setup-graphsignal --source /path/to/graphsignal
kernelopt setup-graphsignal --source git+https://github.com/<you>/graphsignal@main
kernelopt setup-graphsignal --source third_party/graphsignal   # vendored submodule
kernelopt setup-graphsignal --upstream         # PyPI upstream (no llama.cpp/NInfer)
kernelopt setup-graphsignal --cuda 13 --version 1.0.0
```

Source precedence: `--source` → vendored `third_party/graphsignal` →
`KERNELOPT_GRAPH_SIGNAL_SOURCE` → the fork. `--source pypi` / `--upstream` forces
upstream.

The first `kernelopt profile` (or a campaign with `--order engine`) also
auto-provisions if nothing is found. Lookup order for the binary:
`$GRAPHSIGNAL_RUN` → `.kernelopt/graphsignal/venv/bin/graphsignal-run` →
`graphsignal-run` on `PATH`; among those, an install with the fork's
llama.cpp/NInfer launchers is preferred. Set `KERNELOPT_GRAPH_SIGNAL_DIR` to
relocate the managed venv.

For a pinned, in-tree source:

```
git submodule add <fork-url> third_party/graphsignal
kernelopt setup-graphsignal --source third_party/graphsignal
```

## Rank kernels by engine share

```
kernelopt profile --repo "$NINFER_REPO" --mode ninfer \
    --cuda-graph-trace node \
    --cmd <workload argv …>
```

Example (llama.cpp):

```
kernelopt profile --repo "$LLAMACPP_REPO" --mode llamacpp --cuda-graph-trace node \
    --cmd "$LLAMACPP_REPO/build/bin/test-backend-ops" perf -o SOFT_MAX -b CUDA0
```

Output:

```
engine share — graphsignal trace=node (2 kernels, 56 targets)
op                          share           time  kernels
SOFT_MAX                   100.0%     1239.13 ms  2
```

The runner command behind it is `graphsignal_profile`: it launches the workload
under `graphsignal-run`, polls `/signals` while it runs, and returns the last
payload. `--json` emits the ranking plus `signals::summarize` (transfer/sync
totals, GPU telemetry, dropped-record warnings, console errors).

Kernel symbols are mangled and filenames don't always equal op names, so
attribution is heuristic: a kernel is matched to the target whose
op/family/variant tokens it contains (most specific first). Unmatched kernels are
reported separately.

## Order a campaign by engine share

```
kernelopt campaign --repo "$NINFER_REPO" --mode ninfer \
    --order engine \
    --profile-cmd "$NINFER_REPO/build/bench/ninfer_bench" --weights <artifact.ninfer> \
    --patience 2 --target-speedup 1.2
```

With `--order engine`, the campaign runs the workload once under Graphsignal,
attributes GPU time to targets, and processes the hottest ops first (ops with no
measured share fall back to complexity order). `--order complexity` is the
default.

## `cuda_graph_trace`

Engines that capture decode into a CUDA graph report whole replays in
`cuda_graphs_nanoseconds`; pass `--cuda-graph-trace node` to get per-kernel time
in `cuda_kernels_nanoseconds` (needed for per-op ranking). It costs more but
needs no elevated privileges.

## Privacy

Graphsignal runs locally and binds `127.0.0.1` by default. Nothing is uploaded
unless `GRAPHSIGNAL_API_KEY` is set. KernelOPT never sets it.

## GPU probes (optional, no download)

Graphsignal also ships a header-only probe API (`include/graphsignal/probe.h`,
`probe_cuda.h`, Apache-2.0) for instrumenting kernels from the inside. It is
self-contained and can be vendored from the same source checkout when a kernel is
named and the remaining question is *which part* of it is slow.
