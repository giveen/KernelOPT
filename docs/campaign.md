# KernelOpt — Campaign Mode (point at a directory)

Campaign mode turns KernelOpt from "optimize one kernel" into a long-running,
resumable loop over **every kernel in a directory**.

```
kernelopt campaign --repo <dir> [--mode auto|ninfer|llamacpp|custom] [options]
```

`--mode auto` detects the backend from repo markers: `src/ops` +
`include/ninfer/ops` → ninfer; `ggml/src/ggml-cuda` → llama.cpp.

## 1. Flow

1. **Discover** every runnable target (`discover_targets`): ninfer ops/quant
   variants, or the curated llama.cpp op→kernel map. Targets are ordered
   simplest-first (fewest editable files).
2. **Loop** each target through the full pipeline (`CudaPipeline`): baseline
   worktree+build, then plan → edit → compile → correctness → bench iterations.
3. **Persist** after every target: `state.json` (queue/cursor), `memory.json` +
   `tracker.json` (cross-target learning), and each target's
   `runs/<run_id>/checkpoint.json` after every iteration (`--resume <id>`
   continues). Journals/reports/diffs land in `.kernelopt/runs/<run_id>/`.
4. **Stop** on the global budget, or when all targets are processed.

Experience memory and the strategy tracker are shared across targets and
persisted across `--resume` (paper §4.5 cross-run), so a failure on one op
informs the next — even in a later session.

## 2. Per-target termination

Unlike the paper's fixed `T = 5`, a target keeps iterating until any of:

- **patience** — `--patience P` consecutive non-improving iterations (default 2,
  paper-faithful);
- **target** — `--target-speedup X`: stop once `baseline / best ≥ X`
  (e.g. `1.2` = "get 20% or move on");
- **max-iterations** — `--max-iterations N` hard cap (default 25);
- **budget** — the global wall-clock / LLM-call budget is exhausted.

"Improvement" means a best latency at least `--min-improvement` (default 1%)
below the previous best, which resets patience.

## 3. Budgets

```
--budget-hours H        # wall-clock cap for the whole campaign (0 = none)
--budget-llm-calls N    # stop after N LLM completions (0 = none)
--max-targets M         # stop after M targets (0 = all)
```

Budgets are checked between iterations and before each target. When the budget
expires mid-target the target is **paused** (checkpoint kept, status stays
`running`) rather than finalized, so `--resume` continues it. Ctrl-C pauses the
same way. See [monitoring.md](monitoring.md).

## 4. State and resume

```
.kernelopt/campaigns/<id>/
├── state.json     # target queue + per-target status/speedup/iterations
├── campaign.jsonl # one line per finished target
├── memory.json    # experience memory (paper §4.5), persisted across resume
└── tracker.json   # strategy tracker (AVOID flags), persisted across resume

.kernelopt/runs/<run_id>/
├── journal.jsonl
├── checkpoint.json   # per-target search state, written after each iteration
└── …
```

`state.json` fields: `targets[]` (`status` ∈ pending/running/optimized/matched/
unverified/fallback/failed), `cursor`, `llm_calls`, `tokens`, `elapsed_s`.

**How `--resume <id>` avoids repeating work:**

1. **Finished targets are skipped.** `cursor` only advances past targets that
   reached a terminal status, so completed targets are never redone.
2. **Cross-target learning is restored.** `memory.json` and `tracker.json` are
   loaded, so the Planner's `memory_context` and `avoid_flags` carry over — it
   won't re-propose directions already learned.
3. **An interrupted target resumes mid-search.** A target caught by the budget is
   marked `running` (not finalized) and its `checkpoint.json` holds the best
   candidate, the next iteration index, recent directions, patience counter, and
   LLM/token totals. On resume the target keeps its `run_id`, loads the
   checkpoint, and continues from the next iteration instead of restarting.
4. **Accounting is delta-based.** A resumed target reports cumulative LLM
   usage, and the campaign adds only the delta, so totals aren't double-counted.

Budgets are checked between iterations and before each target; when the budget
expires mid-target the target is *paused* (checkpoint kept) rather than finalized,
and `--resume` picks it up.

## 5. Example

```
export NINFER_REPO=/path/to/ninfer
kernelopt campaign --repo "$NINFER_REPO" --mode ninfer \
    --provider opencode-go --model <model> --reasoning-effort low \
    --patience 3 --target-speedup 1.15 --budget-hours 8 --max-iterations 30

# later, continue where it stopped:
kernelopt campaign --repo "$NINFER_REPO" --mode ninfer --resume <campaign-id>
```
