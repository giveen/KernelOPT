# KernelOpt — How kernels are read, edited, and patched

The optimization unit is exactly **one kernel file** per target. KernelOpt never
rewrites launchers, contracts, or build files, and never edits your checkout.

## Read

1. **Discovery** scans the repo (`discover`) and resolves a `Target`:
   - `kernel_files` — every editable `.cu`/`.cuh` for the op,
   - `target_file` — the one file that will be edited (the first, or `--kernel-file`),
   - `contract_files`, `context_files` — read-only context (contract header,
     launcher, wrapper, dispatch/plan),
   - `test_filters`, `bench_binary` — the gate authorities.
2. The pipeline materializes the isolated worktree and reads the target file's
   text from it (`std::fs::read_to_string(worktree/<target_file>)`).
3. That text is embedded in the prompts, along with the **contract header text**
   and the profiling context.

### What the model actually sees (context policy)

The editing model is full-file submission (below), so the Executor must see the
whole file. The Planner does not, and pays for it only when the file is small:

| Agent | Small file (≤ 20k chars) | Large file (> 20k chars) |
|---|---|---|
| **Planner** | full source inlined | **outline** (kernel/device/template lines) + `search_repo` / `read_file` tools |
| **Executor** | full source inlined | full source inlined (required to reproduce it) + a size warning if > 40k chars |

The Planner is given **retrieval tools** so it pulls only what a plan needs
instead of a full dump:

- `search_repo(query, glob?, max?)` — ripgrep over the worktree (falls back to a
  built-in line scan when `rg` is absent).
- `read_file(path, start?, end?)` — a bounded, line-numbered region of any file in
  the worktree.

This matters for real files: llama.cpp `mmq.cuh` is 1619 lines and
`fattn-mma-f16.cuh` 2183; ninfer softmax variants run 260–670. Inlining those for
planning is wasteful, and the Planner only needs the function it will change.

**Why not ripgrep for the Executor too?** Because the Executor returns the
*complete* file — it cannot reconstruct what it never saw. Retrieval and
full-file submission are in tension; the split above keeps the Planner cheap
without breaking the Executor. For very large files the Executor warning is the
signal to either pick a smaller `--kernel-file` or add a patch-based edit path
(not implemented).


## Edit modes

`--edit-mode full|patch` selects how the Executor **produces** the change. Both
run inside the isolated worktree; either way the run's deliverable is a
**reviewable unified diff** (`report.diff`) and your checkout is never modified.
Patch mode is **opt-in** (`full` is the default).

**`full` (default)** — the Executor returns the complete file (below). Always
applies, but the model must reproduce the whole file, so it can drift (mangled
comments, duplicated types); the compile gate catches it and the error is fed back
for a retry.

**`patch`** — the Executor returns a **unified diff** for the target file only;
KernelOPT applies it with `git apply` (fallback `patch -p1`) and reads the result
back. Benefits: no reproduction drift, and far fewer output tokens on large files
(`mmq.cuh`, `fattn-*.cuh`). New failure mode: a malformed/mis-anchored patch —
surfaced as `patch_apply` in `analyze` and fed back for a retry. Patches are
archived next to candidates as `*.patch`.

Use `--edit-mode patch` to A/B against `full` on a real op.

## Edit (full mode)

There is **no diff/patch application in the loop**. The Executor returns the
**complete new content** of the target file via `submit_kernel(kernel_source,
change_summary)`; KernelOpt:

1. strips markdown code fences (`strip_code_fences`),
2. archives the submission for postmortem at
   `.kernelopt/runs/<run_id>/candidates/i<iter>_c<chain>_a<attempt>.cu`,
3. writes it wholesale into the worktree:
   `std::fs::write(worktree/<target_file>, candidate)`.

Then it compiles (Gate 1), runs correctness (Gate 2), and benches (Gate 4). On
failure the error text is fed back and the Executor submits a **new complete
file** for the next attempt. Only `target_file` is ever written.

**Why full-file instead of patches:** LLMs produce patches unreliably (context
drift, hunk mismatch), and a failed patch wastes a round-trip. Full-file
submissions always apply; the compile + correctness gates catch a truncated or
drifted file. The trade-off is larger outputs and the model occasionally
reformatting unrelated lines — visible in the final diff.

## Isolation

All writes go to a **linked git worktree**, never your tree:

```
.kernelopt/<backend>/worktree/     # branch kernelopt/<backend>-<op>
```

Between beam chains the pipeline resets the worktree to the recorded base commit
(`cuda_worktree reset` → `git reset --hard <base>` + `git clean -fd`), so each
chain starts from the pristine baseline kernel. The build dir
(`.kernelopt/<backend>/build`) is configured against the worktree, so edited
sources are what gets compiled.

## History and revert (git commits)

Every candidate that passes Gates 1–2 is **committed** in the worktree and
**tagged**, so the branch can be reset to base without losing the attempts:

- commit message: `kernelopt i<iter>/c<chain>: <plan> (latency <ms>)`
- tag: `kernelopt/<run_id>/i<iter>_c<chain>_a<attempt>`

The base commit is recorded at baseline (`stage worktree` → `head`); resets go
back to it. This makes reverting instant (`git reset --hard <ref>`) and the whole
search readable:

```bash
kernelopt history <run_id>                 # candidate commits: sha, status, latency, plan
kernelopt revert <run_id> --to <sha|tag>   # put the worktree at that candidate
```

or directly in the worktree:

```bash
git -C .kernelopt/ninfer/worktree log --oneline            # the search, newest first
git -C .kernelopt/ninfer/worktree tag -l 'kernelopt/<run>/*'
git -C .kernelopt/ninfer/worktree show <tag>               # the exact change
git -C .kernelopt/ninfer/worktree diff <base> <tag>
```

The final winner is materialized by a fast `git reset --hard <winner-commit>`
(no file rewriting), then re-verified.

## Patch (the deliverable)

The loop produces no commits **in your repo** (the worktree branch is scratch).
The deliverable is a unified diff of the base commit against the winner:

```
git -C <worktree> --no-pager diff <base-sha> -- <target_file>   # -> runs/<id>/report.diff
```

Apply it to your checkout with:

```bash
git -C "$NINFER_REPO" apply .kernelopt/runs/<run_id>/report.diff
```

If no candidate passes all gates, nothing is written to your tree and the report
explains why (baseline preserved).

## Inspecting a change

- `report.diff` — the accepted change.
- `candidates/*.cu` — every submission, pass or fail, with its attempt id.
- `journal.jsonl` — plan, `change_summary`, errors, and timings per attempt
  (`kernelopt watch <run_id>` renders it live).
