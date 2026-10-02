# KernelOpt — CUDA/HIP documentation lookup

When a candidate fails to **compile**, the cause is usually API availability or
calling convention — e.g. `no instance of overloaded function "__reduce_max_sync"
matches the argument list`, or `cub::WarpMergeSort` has a deleted constructor on a
new arch/CCCL. KernelOPT extracts the offending symbol from the diagnostic, looks
it up, and **attaches a short excerpt to the Executor's retry** (and to the
Planner's `RECENT FAILURES`) so the model fixes the API instead of re-guessing.

You can also query it yourself:

```bash
kernelopt docs <symbol>
```

It only fires on a **confident API symbol** (a namespaced name like
`cub::WarpMergeSort`, a `__…` intrinsic, or a known builtin prefix such as
`atomic…`, `hip…`, `rocwmma::`, `__builtin_amdgcn…`). Generic tokens (`lambda`,
`unsigned`, user types) are skipped, so no noise is injected.

## Resolution order

1. **stdio MCP server** — `KERNELOPT_DOCS_MCP_CMD`, else AMD's `hip-docs-mcp`
   when ROCm is present and installed.
2. **HTTP MCP server** — a token (`KERNELOPT_DOCS_TOKEN`) against `KERNELOPT_DOCS_URL`
   (default: NVIDIA `cuda-docs`).
3. **Local headers** — the CUDA/CCCL headers on the machine, plus ROCm headers
   when ROCm is detected.

A miss at one source falls through to the next, so a broken/expired server never
blocks a run — you get local headers instead.

## NVIDIA CUDA docs (HTTP MCP)

The `cuda-docs` MCP does semantic search over the current CUDA Toolkit, cuDNN,
CUTLASS and CCCL documentation and code samples.

### One-time login

```bash
kernelopt docs --login
```

This performs OAuth **dynamic client registration** + **authorization code with
PKCE (S256)**, prints an authorize URL, and waits on a local `127.0.0.1` callback
(15-minute timeout) for the redirect. Sign in with your NVIDIA account, then the
token is cached at:

```
.kernelopt/docs_token.json
```

Subsequent lookups use the cache and **refresh it automatically** when it nears
expiry, so you only log in once (until the refresh token is revoked/expires).

### Without login

```bash
export KERNELOPT_DOCS_TOKEN=<access-token>   # or KERNELOPT_CUDA_DOCS_TOKEN
kernelopt docs "cub::WarpMergeSort 64-bit keys"
```

Point at a different endpoint with `KERNELOPT_DOCS_URL` (or
`KERNELOPT_CUDA_DOCS_URL`).

Example output (cleaned — anchor/link noise is stripped):

```
# cub::WarpMergeSort

template<typename KeyT, int ITEMS_PER_THREAD, int LOGICAL_WARP_THREADS = detail::warp_threads,
         typename ValueT = NullType> class WarpMergeSort : public cub::BlockMergeSortStrategy<…>
:   The WarpMergeSort class provides methods for sorting items partitioned across a CUDA warp
    using a merge sorting method.
    … example: sorting 64 integer keys across 16 threads …
```

## AMD ROCm / HIP

ROCm support is **auto-enabled when ROCm is detected**: `hipcc`, `rocm-smi`,
`rocminfo`, or `amdgpu-arch` on `PATH`; `ROCM_PATH`/`HIP_PATH`; or `/opt/rocm`.
HIP symbols are then looked up in the local ROCm headers (offline, no auth).

To use AMD's [`hip-docs-mcp`](https://github.com/AMDResearch/intellikit) server
(stdio) instead, point KernelOPT at its command:

```bash
export KERNELOPT_DOCS_MCP_CMD="uv run --directory /path/to/intellikit/rocm_mcp hip-docs-mcp"
kernelopt docs hipMalloc
```

Notes:
- AMD's `rocm-mcp` package imports `amdsmi`, which needs `libamd_smi.so` — the
  server only starts on a host with ROCm installed. On such a host, KernelOPT
  auto-uses `hip-docs-mcp` if it is on `PATH`.
- Force on/off with `KERNELOPT_DOCS_ROCM=1` / `=0` (unset = auto-detect).

## Environment variables

| Variable | Purpose |
|---|---|
| `KERNELOPT_DOCS_TOKEN` | Bearer token for the HTTP MCP docs server. Alias: `KERNELOPT_CUDA_DOCS_TOKEN` |
| `KERNELOPT_DOCS_URL` | HTTP MCP endpoint (default: NVIDIA `cuda-docs`). Alias: `KERNELOPT_CUDA_DOCS_URL` |
| `KERNELOPT_DOCS_MCP_CMD` | stdio MCP command (tried before HTTP) |
| `KERNELOPT_DOCS_ROCM` | Force ROCm docs on (`1`) or off (`0`); unset = auto-detect |

## In a run

No configuration is required to get the **local-header** fallback. Once a token
is cached (or set) the CUDA MCP is used automatically on compile failures:

```
· attached CUDA docs for the failing symbol
```

The excerpt is added to the Executor's retry for that attempt; the Planner's
`RECENT FAILURES` line keeps the compact error. Lookups add tokens per compile
retry, so they only run when a symbol looks like a real API.

## Limitations

- Local-header lookup is heuristic (grep + filename match): it returns the first
  plausible declaration, not a curated answer.
- The HTTP path needs a token; the stdio path needs the server installed. Both
  are optional — local headers are always available.
- The token file lives under `.kernelopt/` (git-ignored); it holds an access +
  refresh token for your NVIDIA account.
