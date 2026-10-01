# KernelOpt — Configuration

Precedence for every setting: **CLI flag → environment/`.env` → `config.toml` →
built-in default**.

## `.env` (recommended)

Copy [`.env.example`](../.env.example) to `.env` in the directory you run from
(or point `KERNELOPT_ENV_FILE` at any file). Real environment variables always
override `.env`, and `~` is expanded in path values. `.env` is git-ignored.

| Key | Purpose |
|---|---|
| `NINFER_REPO` / `LLAMACPP_REPO` | target checkout paths |
| `KERNELOPT_PROVIDER` | provider preset (default `opencode-go`) |
| `KERNELOPT_MODEL` | model id (default `deepseek-v4-pro`) |
| `KERNELOPT_BASE_URL` | custom OpenAI-compatible base URL |
| `KERNELOPT_API_KEY` | API key (else the preset's key var) |
| `KERNELOPT_REASONING_EFFORT` | thinking level: `none\|minimal\|low\|medium\|high` |
| `OPENCODE_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | provider keys |
| `KERNELOPT_ENV_FILE` | load a different env file |
| `KERNELOPT_GRAPH_SIGNAL_DIR` | relocate the managed Graphsignal venv |
| `KERNELOPT_GRAPH_SIGNAL_SOURCE` | Graphsignal install source (default: the fork) |
| `GRAPHSIGNAL_RUN` | use an existing `graphsignal-run` binary |

## Providers

Any OpenAI-compatible endpoint. Presets:

| Preset | Base URL | Key env |
|---|---|---|
| `opencode-go` (default) | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `ollama` / `vllm` / `lmstudio` | local | — |
| `mock` | scripted, zero tokens (tests) | — |

```bash
kernelopt providers --provider opencode-go --model <model>   # auth + model + tool-call probe
```

The probe checks three things: `GET /models` (auth + model id), a 1-token
completion, and a **forced tool call** — the pipeline drives the model entirely
through tool calls, so this is the compatibility check that matters.

The client is defensive about OpenAI-compatible servers that diverge, and falls
back automatically (logging once to stderr):

- rejects `reasoning_effort` → retried without it;
- rejects a forced `tool_choice` (named) → retried with `"auto"`, then with no
  `tool_choice`;
- rejects tool schemas entirely → retried without tools;
- returns `tool_calls: null`, object-valued `arguments`, or array `content` →
  all accepted.

Local servers (`ollama`/`vllm`/`lmstudio`) default `reasoning_effort` **off**
(they don't accept it); set `--reasoning-effort` explicitly to force it.

## Thinking level

`--reasoning-effort` (alias `--thinking`) controls how much the model thinks
before answering; it is sent as OpenAI `reasoning_effort`. Defaults to **`low`**
for hosted providers (faster, cheaper) and **off** for local ones, accepts
`none|minimal|low|medium|high`, and is validated.

```bash
kernelopt run-ninfer --op add_bias --repo "$NINFER_REPO" --thinking high
```

Set it per command, in `.env` (`KERNELOPT_REASONING_EFFORT`), or in `config.toml`.

## `config.toml` (optional, project root)

```toml
base_url = "https://opencode.ai/zen/go/v1"
api_key_env = "OPENCODE_API_KEY"
reasoning_effort = "low"        # overridden by --reasoning-effort / .env
```

## Paper hyperparameters

All overridable per command: `T=5`, `N=4`, `K=4`, `B=4`, UCB `c=1.4`, `γ=1.03`,
memory `Q=8`, `s+=1.05`, `s−=1.20`. On the CLI these are `--iterations` (T), `--plans` (N),
`--retries` (K), `--beam` (B); the UCB exploration constant and the performance margin γ
are fixed in `config.rs`.
