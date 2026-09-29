# KernelOpt — Monitoring, pausing, and resuming

Long campaigns run for hours; this is how to see what's happening, pause, and
pick up where you left off.

## One terminal: `--watch`

Add `--watch` to any run command and it renders the **journal view live in the same
terminal** — no second terminal:

```bash
kernelopt run-ninfer --op add_bias --watch
kernelopt campaign --mode ninfer --watch
```

`--watch` tails the run's own `journal.jsonl` (timestamped: start, baseline
stages, LLM calls, candidates, gates, finish) and prints it to stderr while the
pipeline runs in the same process. `--quiet` still silences the terse per-iteration
lines; `--watch` and `--quiet` are independent.

## Live progress

Optimization commands also stream per-iteration progress to **stderr** (stdout
stays the final JSON result), so you can watch it and still pipe the result. Lines
are timestamped and printed in full (plans and compiler errors are not truncated):

```
[21:07:12] [fp8_linear_add] iteration 1/25 (best —)
[21:07:19]   c0 plan: vectorize the epilogue loads with uint4 and drop the per-element branch
[21:07:41]   c0 exec a0: rewrite epilogue to 128-bit loads with a tail path
[21:07:48]   c0 ✗ compile: fp8_linear_add_epilogue.cuh:42: identifier "uint4" is undefined
[21:07:52]   c0 exec a1: include <cuda_runtime.h> and cast the pointer to uint4
[21:08:05]   c0 ✓ 4.91 ms · 1.04x
[21:08:05] [fp8_linear_add] iteration 2/25 (best 4.91 ms)
```

What each line shows:

| Line | Meaning |
|---|---|
| `iteration i/N` | the current beam iteration and the best latency so far |
| `cN plan:` | the **Planner's** chosen change (what the LLM is thinking) |
| `cN exec aK:` | the **Executor's** `change_summary` for attempt K |
| `cN ✗ compile/correctness` | the candidate was rejected; the error fed back to the Executor |
| `cN ✓ <ms> · <x>` | the candidate passed compile + correctness + bench |

`--quiet` suppresses these per-iteration lines (campaign target headers remain).
The raw per-call detail is always in the run's `journal.jsonl`.

## Watching a run

`watch` tails a run's journal and renders it live — for attaching to a **detached**
run (started with `&`, `nohup`, or in another session), or just to re-read one:

```bash
kernelopt watch --latest          # follow the most recent run (Ctrl-C to stop)
kernelopt watch <run_id>          # follow a specific run
kernelopt watch --latest --once   # print what exists and exit
```

To run detached and attach later:

```bash
nohup kernelopt run-ninfer --op add_bias > run.log 2>&1 &
kernelopt watch --latest
```

```
▶ start ninfer:add_bias via mock/mock
· stage compile_baseline: {"errors":[],"passed":true}
· stage bench_baseline: {"median_ms":0.00509}
  llm planner: 5+5 tok
  i0/c0 PASS 0.0050 ms — identity walk
■ finished optimized 1.032x
```

For a campaign:

```bash
kernelopt status <campaign-id> --campaign
```

```
{ "campaign": "20260928_...", "targets": 52, "processed": 7, "optimized": 3, ... }

op                       status      iters  speedup  stop
add_bias                 optimized       2   1.031x  patience
fp8_linear_add           running         4        -  -
...
```

## Pausing

**Ctrl-C pauses gracefully.** KernelOPT finishes the runner step in flight (a
compile/bench is never left half-done), saves the per-target checkpoint, marks the
target `running`, writes the state, and prints a resume hint:

```
[interrupt] finishing the current step, then pausing…
  -> paused (interrupted); resume with:
     kernelopt campaign --repo … --mode ninfer --resume 20260928_...
```

The runner subprocess is placed in its own process group, so Ctrl-C does not kill
a build mid-flight.

The **budget** (`--budget-hours`, `--budget-llm-calls`) pauses the same way when it
expires mid-target.

## Resuming

Campaign:

```bash
kernelopt campaign --repo "$NINFER_REPO" --mode ninfer --resume <campaign-id>
```

Single run (`run-ninfer` / `run-llamacpp`): re-run with the same `--run-id` so the
checkpoint is picked up:

```bash
kernelopt run-ninfer --op fp8_linear_add --repo "$NINFER_REPO" \
    --provider opencode-go --model <model> --run-id <run-id>
```

Resume skips finished targets, restores experience memory and AVOID flags, and
continues the interrupted target from its next iteration — see
[campaign.md](campaign.md#4-state-and-resume).

## Why not a TUI?

A full-screen TUI is a large dependency and a different failure mode for a tool
that mostly runs unattended. The journal is the source of truth, so `watch` (tail
it) plus `status --campaign` (summarize it) give live visibility without a TUI.
`--json` on `profile` and the machine-readable stdout results cover scripting. If
an interactive dashboard is wanted later, it can render the same journal/state
files without touching the pipeline.
