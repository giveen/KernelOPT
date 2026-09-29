# KernelOpt — Outputs and artifacts

Everything KernelOPT produces lives under `.kernelopt/` (git-ignored). Nothing is
committed to your target repo.

```
.kernelopt/
├── runs/<run_id>/
│   ├── journal.jsonl      # every event (durable; powers status/report/resume)
│   ├── checkpoint.json    # per-target search state (written each iteration)
│   ├── candidates/        # every submission (pass or fail)
│   ├── bench/*.csv        # per-invocation bench output
│   ├── ncu/*.csv          # raw profiler reports
│   ├── report.md          # human-readable summary
│   └── report.diff        # CUDA modes: unified diff of the winning change
├── campaigns/<id>/
│   ├── state.json         # target queue + per-target status (resume here)
│   ├── campaign.jsonl     # one line per finished target
│   ├── memory.json        # experience memory (persisted across resume)
│   └── tracker.json       # strategy tracker / AVOID flags (persisted)
├── graphsignal/           # managed profiler venv (see graphsignal.md)
└── <backend>/{worktree,build}/   # isolated worktree + warm CMake build dir
```

Accepted candidates are committed in the worktree and tagged
`kernelopt/<run_id>/<candidate>`; `kernelopt history <run_id>` lists them and
`kernelopt revert <run_id> --to <ref>` restores one. See
[kernel-editing.md](kernel-editing.md#history-and-revert-git-commits).

## Applying the result

For CUDA modes (`ninfer`/`llamacpp`) the deliverable is `report.diff`:

```bash
git -C "$NINFER_REPO" apply .kernelopt/runs/<run_id>/report.diff
```

For `triton` mode the deliverable is the re-stitched model written to
`.kernelopt/tmp/restitched_*.py` (the run result records its path).

## Inspecting a run

```bash
kernelopt status <run_id>    # events, passing candidates, tokens
kernelopt report <run_id>    # markdown report from the journal
kernelopt resume <run_id>    # last recorded state
```

## Campaign state

`state.json` holds the target queue and per-target outcome:

| Field | Meaning |
|---|---|
| `targets[]` | `op`, `family`, `status` (`pending`/`running`/`optimized`/`matched`/`fallback`/`failed`), `best_speedup`, `iterations`, `stop_reason`, `run_id`, `llm_calls`, `error` |
| `cursor` | next target index (resume point) |
| `llm_calls`, `tokens`, `elapsed_s` | cumulative totals |

`memory.json`/`tracker.json` carry cross-target learning across `--resume`;
`runs/<run_id>/checkpoint.json` lets an interrupted target continue from its next
iteration. Details in [campaign.md](campaign.md#4-state-and-resume).

## Campaign termination

A target keeps iterating until **any** of:

- `--patience P` consecutive non-improving iterations (default 2);
- `--target-speedup X` reached (`baseline / best ≥ X`);
- `--max-iterations N` hard cap (default 25);
- the global budget (`--budget-hours`, `--budget-llm-calls`) is exhausted.

`--min-improvement` (default 1%) is the fractional gain that counts as an
improvement and resets patience. The campaign stops when all targets are
processed or the budget is exhausted; `--max-targets` caps how many are run.
See [campaign.md](campaign.md) and [cli.md](cli.md).

## Disk

The isolated worktree and its CMake build dir are reused per op to keep the build
warm. Delete `.kernelopt/<backend>/` to reclaim space (the next run rebuilds).
