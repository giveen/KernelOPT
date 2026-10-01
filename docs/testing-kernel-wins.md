# Testing whether a kernel optimization is a real win

This is a recommended review workflow, not a change to KernelOPT's automated
verdicts. It applies [Rapid Software Testing (RST)](https://www.satisfice.com/rapid-testing-methodology):
use risk and experiments to decide what evidence is needed, then question what
passing checks actually establish. Scale the investigation to the change;
these are prompts for judgment, not a mandatory checklist for every candidate.

## Start with a claim that can be challenged

Before searching, record the decision and the failure you are trying to expose.
For example: "Does this vectorized load reduce device time at the production
shape without dropping tail elements or slowing the caller?"

Freeze the baseline and candidate source/build identities, hardware, toolchain,
inputs, dtype, shape, strides, alignment, launch geometry and timing boundary.
Record residency, warmup and synchronization policy. A resident-buffer kernel
measurement and an engine run including allocation or loading answer different
questions.

Choose an oracle for each risk before testing:

| Risk | Useful oracle or experiment |
|---|---|
| Tail rows skipped or stale output reused | Independent reference over the complete output; initialize output with a sentinel; include ragged sizes |
| Wrong stride, offset or alias assumption | Supported non-contiguous/offset cases and contract-based rejection of unsupported cases |
| Reduction or precision change | Independent higher-precision reference plus absolute/relative error and non-finite checks; justify tolerances before seeing the candidate |
| Candidate compiled but never dispatched | Separate diagnostic trace identifying the executed variant and workload |
| Stateful corruption | Multi-step continuation, reset and partial-update scenarios against a trusted state reference |
| Faster kernel, slower caller | Same-workload application measurements with actual completed work counted |

Test the timed shape and nearby dispatch boundaries: for tile width `B`, consider
`B-1`, `B`, `B+1`, multiple tiles and a remainder. Sample relevant data regimes,
not only random seeds: zeros, cancellation, large/small magnitudes and supported
non-finite behavior. Let the operation's contract determine which cases matter.

## Check the checking machinery

Use a valid counterpart and a deliberate defect to test each important oracle.
For a copy-like operation, skipping the final row should fail full-output
comparison even if it produces an impressive timing. A wrong index or a missing
state update may be a better control for another operation. Keep mutations in
throwaway candidates, never in the deliverable.

A baseline-versus-baseline timing run helps expose drift or a harness that
reports ordinary noise as a win. Test the report/parser with known synthetic
slower and unchanged results as well. These controls investigate the harness;
they do not prove the candidate correct.

Do not loosen a tolerance because a candidate fails. Investigate the discrepancy,
retain the first failure, and document any independently justified contract
change before starting a new comparison. An oracle sharing the candidate's
indexing or reduction implementation can reproduce the same bug.

## Separate diagnosis from timing

Use profiling to establish the route and explain a bottleneck, then disable
profiling for the performance comparison. Keep captures and timing receipts
separate. Inspect other device activity: an advisory lock only coordinates
processes using the same lock path, not every process on the GPU.

Use fresh, interleaved baseline/candidate sessions, varying which arm runs first.
Retain per-session measurements and the predeclared warmup policy. Compare the
observed difference to baseline variation and an effect large enough to matter.
A small number of all-winning rounds is evidence, not a general statistical
guarantee. Account for repeated candidate search by confirming the frozen winner
in fresh sessions outside the selection loop.

Report each important shape rather than only an aggregate. A median can hide a
regression in a tail shape, short request or low-acceptance workload. For inference,
report actual input/output tokens, time to first token, prefill and decode
separately; requests with different output lengths do not represent equal work.
Use engine timing rather than network event counts as a decode-rate proxy.

A roofline check is a diagnostic clue. Unexpected bandwidth can mean skipped
work, incorrect byte accounting or missing synchronization, but cache-resident
traffic can legitimately exceed a DRAM bandwidth estimate. Investigate residency
and bytes actually moved before interpreting the number. Being below a roofline
does not establish correctness.

## State the scope of the result

The current CUDA pipeline's `optimized` label describes its configured gates.
[Engine verification](model-e2e.md) is opt-in. The measured-shape gate can be
`skipped` for unsupported test signatures; that is not a passed shape check.
Inspect the gate details in the [journal/report](outputs.md), not only the label.
Custom test commands are only as strong as the checks behind their exit status.

Recommended language for a review:

| Evidence obtained | Defensible claim |
|---|---|
| Correctness checks pass, timing inconclusive | Correct on the tested cases; no demonstrated speedup |
| Exact measured shape passes and fresh timing improves | Kernel win at the stated shape, dtype, hardware and timing boundary |
| That kernel actually runs in the application and application measurements improve | Application win on the tested workload, with listed regressions and limits |
| Required oracle unavailable, route unconfirmed or gate skipped | Provisional result; specify the missing evidence |

Output equality on a fixed prompt is a useful regression check, not proof of
model-wide semantic equivalence or quality. If arithmetic intentionally changes,
separate numerical agreement from task-quality evaluation. Choose representative
held-out tasks or metrics appropriate to the model; do not treat faster output
with repetition or early termination as an equivalent workload.

Also estimate whether the target matters. Under the simplifying assumption that
other costs stay fixed, a kernel occupying fraction `f` of runtime and sped up by
`s` yields application speedup `1 / ((1-f) + f/s)`. A 2x improvement to a 5% share
predicts only about 1.026x overall. Dispatch, fusion and memory effects can break
that assumption, so measure the application before claiming the gain.

## Leave a useful receipt and a stopping reason

Alongside the existing run artifacts, record:

- Claim, frozen identities, workload and exact timing boundary.
- Oracles, tolerances, valid/known-bad controls and executed routes.
- Raw per-session baseline/candidate timings, actual work and regressions.
- Gates passed, failed or skipped; uncovered shapes, hardware and state scenarios.
- Decision and why the investigation stopped: meaningful confirmed gain,
  correctness failure, budget exhausted or evidence unavailable.
- For a correct but slower candidate, what new evidence would justify reopening
  it, such as a changed dispatch cost or a larger share in the integrated model.

Preserve informative failures and correct-but-slower candidates without promoting
them. Avoid rerunning an unchanged experiment without a new question. Passing
checks should support a bounded claim and a concrete decision.
