# Agent prompts

These prompts implement the multi-agent loop described in:

> **KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization**
> Aheli Poddar, Sanskar Prasad, Arindam Samanta, Subha Chakraborty, Vishal Goyal,
> Rohit Singh Rathaur. arXiv:2609.30059 (2026).
> https://arxiv.org/abs/2609.30059

| File | Agent / purpose |
|---|---|
| `planner.md` | Planner (compiled-model / Triton) — adapted from the paper's appendix |
| `executor.md` | Executor (Triton) — adapted from the paper's appendix |
| `summarizer.md` | Summarizer (experience memory) — adapted from the paper's appendix |
| `profiler.md` | Profiler (NCU configuration) |
| `codegen.md`, `fusion.md` | Inductor-aware synthesis / fusion grouping |
| `cuda-planner.md` | Planner for CUDA targets (ninfer / llama.cpp / custom) — this project. Takes `{{backend}}` |
| `cuda-executor.md` | Executor, full-file edit mode — this project. Takes `{{backend}}` |
| `cuda-executor-patch.md` | Executor, patch (unified-diff) edit mode — this project. Takes `{{backend}}` |

Templates use `{{var}}` placeholders rendered by `src/prompts.rs`. The prompts
adapted from the paper are condensed there; the versions here are the working,
extended forms. Please cite the paper if you build on them (see the repository
`CITATION.cff` / `NOTICE`).
