# RouterForLazyPeople

A local LLM routing proxy for people who want things to just work.

Sits between your chat apps and your local models — starts them on demand, stops them when idle, routes requests to the right size model automatically, and benchmarks performance over time.

```
Your App (OpenAI / Anthropic SDK / OpenClaw / Open WebUI)
         ↓
   Router :9001  ←  auto-classifies request complexity
         ↓
   fast (35B)  |  mid (27B)  |  deep (122B reasoning)
   started on demand, stopped after idle timeout
```

---

## Features

- **Lazy loading** — models start on first request, stop automatically after idle
- **Smart routing** — keyword + token-count classifier picks fast / mid / deep tier
- **Multi-engine** — llama.cpp, vLLM, SGLang, TensorRT-LLM, HuggingFace TGI
- **Auto-discovery** — scans your model directories and registers GGUF / HF / TRT-LLM models automatically
- **OpenAI-compatible** — drop-in for any OpenAI SDK client, Open WebUI, OpenClaw, Cursor, Continue, Jan
- **Anthropic-compatible** — drop-in for Anthropic SDK, Claude Code, `@anthropic-ai/sdk`
- **Gemini-compatible** — supports Google Gemini `generateContent` and `streamGenerateContent` payloads
- **Benchmarking** — tracks TTFT, latency, tokens/sec per backend; export as CSV
- **System diagnostics** — detects GPU, CUDA, CPU architecture, engine versions, install recommendations
- **Auto-update** — one command updates llama.cpp and Python deps
- **Beginner-friendly** — all config in two YAML files, no Python knowledge required to operate

---

## Quick Start: DGX Spark / NVIDIA Workstation

Run this in a terminal on the DGX Spark. SSH is fine; the router prints a LAN URL after startup.

```bash
git clone https://github.com/Yao-Qinghong/RouterForLazyPeople
cd RouterForLazyPeople

# Must be 3.10 or newer. The CLI refuses to build .venv on older Python.
python3 --version

# First start creates .venv, installs dependencies, scans for models, and starts the router.
./router-start start

# Success check: router is reachable and backends/models are visible.
./router-start status
```

After startup:
- OpenAI-compatible clients: `http://localhost:9001/v1`
- Anthropic-compatible clients: `http://localhost:9001/anthropic`
- Model list in a browser: `http://localhost:9001/v1/models`

If your models already live in a default scan directory such as `~/.lmstudio/models`, `~/models`, `~/llm-models`, `~/.cache/huggingface/hub`, or `~/trt-engines`, you can often stop here.

If `./router-start status` works but the model list is empty:
- Run `./router-start sysinfo` and check whether GPU / CUDA / llama.cpp were detected
- Edit `config/settings.yaml` and add your model folders to `scan_dirs`
- Or edit `config/backends.yaml` and point `fast`, `mid`, and `deep` at exact model files
- Run `./router-start rescan`, then check `./router-start status` again

When the model list looks right, run the speed benchmark once:

```bash
./router-start bench
```

`bench` starts each router-managed backend, measures prompt-processing / generation speed, caches the result, and refreshes router state so benchmark-informed routing can use it.

---

## Requirements

- Python 3.10+
- Current focus: NVIDIA DGX Spark or another NVIDIA CUDA Linux machine
- Also supported: macOS with llama.cpp/Metal; CPU-only llama.cpp for small models
- Windows: not currently supported or tested
- NVIDIA GPU recommended (CPU-only works for llama.cpp)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built locally for GGUF models

If you connect from a laptop to a DGX Spark, run the router on the Spark and put the printed LAN base URL into your client app. Keep `localhost` only for clients running on the same machine as the router.

---

## Configuration

All config lives in two files. No Python editing needed.

### `config/backends.yaml` — your models

```yaml
backends:
  fast:
    engine: llama.cpp
    port: 8080
    model: "~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    ctx_size: 131072
    idle_timeout: 300      # stop after 5 min idle
    description: "Qwen3.5-35B — fast"

  mid:
    engine: llama.cpp
    port: 8082
    model: "~/.lmstudio/models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf"
    ctx_size: 131072
    idle_timeout: 180

  deep:
    engine: llama.cpp
    port: 8081
    model: "~/.lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"
    ctx_size: 131072
    reasoning: true
    reasoning_budget: 2048
    idle_timeout: 600
```

Commented examples for vLLM, SGLang, HuggingFace TGI, and TensorRT-LLM are included in the file.

### `config/settings.yaml` — router behaviour

Ports, log rotation, model scan directories, routing keywords, tier thresholds, proxy concurrency — all editable without touching Python code.

It also contains optional auth, CORS, audit logging, model aliases, and preload settings.
The `rate_limit` section is reserved for future use and is not enforced by the router yet.

---

## CLI

```bash
./router-start start             # beginner-safe launcher; creates venv on first run
python cli.py start              # same CLI if you already selected Python 3.10+
python cli.py start --update     # update llama.cpp first, then start
python cli.py stop               # stop the router
python cli.py status             # show which backends are running
python cli.py bench              # measure PP/TG speed for each backend and refresh routing data
python cli.py benchmark          # show request metrics from real router traffic
python cli.py benchmark --export report.csv
python cli.py sysinfo            # GPU, CUDA, CPU, engine versions, install hints
python cli.py sysinfo --all      # include all auto-discovered models
python cli.py update             # update llama.cpp + pip deps
python cli.py update --restart   # update and restart router
python cli.py rescan             # re-scan for new model files (no restart needed)
python cli.py logs               # tail the router log
```

`python cli.py update` chooses a llama.cpp rebuild mode automatically:
- CUDA on systems with a usable CUDA toolchain
- Metal on macOS
- CPU-only everywhere else

llama.cpp updates are transactional: if the pull/build fails, the CLI checks the source tree back out to the previous commit and restores the previous `llama-server` binary when one existed.

## More Docs

- Overview and onboarding: this README
- Public API and functional contract: [`docs/API_SPEC.md`](docs/API_SPEC.md)
- Internal technical design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Operations and runtime behavior: [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
- Docs index: [`docs/README.md`](docs/README.md)

### Startup output

```
Started with PID 12345

┌─ Connect your LLM client to ───────────────────────────────┐
│  OpenAI base URL : http://localhost:9001/v1                │
│  LAN / remote    : http://192.168.1.42:9001/v1             │
│  API key         : any string (no auth required)           │
└────────────────────────────────────────────────────────────┘

  Works with: Open WebUI · OpenClaw · LM Studio API · Cursor · Continue · Jan

  Anthropic SDK (Claude Code, @anthropic-ai/sdk):
    base_url  : http://localhost:9001/anthropic
    api_key   : any string
```

---

## Connecting Your Apps

### OpenAI SDK / Open WebUI / Jan / Continue / Cursor

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9001/v1",
    api_key="anything",
)
response = client.chat.completions.create(
    model="fast",   # or "mid", "deep", or any backend key
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Anthropic SDK / Claude Code

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:9001/anthropic",
    api_key="anything",
)
msg = client.messages.create(
    model="claude-3-5-sonnet-20241022",   # maps to "mid" backend automatically
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Model name → backend mapping:

| Anthropic model | Local backend |
|---|---|
| `claude-3-haiku-*`, `claude-3-5-haiku-*` | fast |
| `claude-3-sonnet-*`, `claude-3-5-sonnet-*` | mid |
| `claude-3-opus-*`, `claude-4-*` | deep |

### Gemini API

```bash
curl -X POST \
  "http://localhost:9001/gemini/v1beta/models/gemini-2.0-flash-latest:generateContent" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: anything" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "Hello"}]}
    ]
  }'
```

### OpenClaw

In `openclaw.json`:

```json
{
  "models": {
    "providers": {
      "local-router": {
        "baseUrl": "http://localhost:9001/v1",
        "apiKey": "anything",
        "api": "openai-completions"
      }
    }
  }
}
```

Models are auto-discovered via `GET /v1/models` — no manual list needed.

### Explicit routing (any client)

Prefix your message with `[route:backend-key]` to force a specific backend:

```
[route:deep] Explain the proof of Fermat's Last Theorem step by step.
```

Or use a query parameter: `POST /v1/chat/completions?backend=fast`

---

## API Summary

The full public route contract now lives in [`docs/API_SPEC.md`](docs/API_SPEC.md).
The most commonly used endpoints are:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/status` | Backend run-state (running, idle seconds, PID) |
| `GET` | `/backends` | All registered backends |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `GET` | `/v1/models/{model_id}` | OpenAI-compatible model detail |
| `GET` | `/engines` | Installed engine availability |
| `GET` | `/metrics` | Request metrics from live router traffic |
| `GET` | `/metrics/export` | Download full history as CSV |
| `GET` | `/metrics/prometheus` | Prometheus text exposition |
| `GET` | `/benchmarks` | Cached speed-test results from `bench` |
| `GET` | `/sysinfo` | GPU, CUDA, CPU, engine versions, recommendations |
| `POST` | `/start/{key}` | Start a backend |
| `POST` | `/stop/{key}` | Stop a backend |
| `POST` | `/restart/{key}` | Restart a backend |
| `POST` | `/rescan` | Re-discover models and reload config |
| `POST` | `/reload-config` | Reload mutable router settings from `settings.yaml` without restart |
| `POST` | `/retune/{key}` | Force re-tune a TRT-LLM backend |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/completions` | OpenAI-compatible legacy completions |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings |
| `POST` | `/v1/{path}` | OpenAI-compatible inference proxy |
| `POST` | `/anthropic/v1/messages` | Anthropic Messages API proxy |
| `POST` | `/v1/messages` | Anthropic Messages API (alias) |
| `POST` | `/gemini/v1beta/models/{model}:generateContent` | Gemini-compatible generation |
| `WS` | `/v1/chat/completions/ws` | WebSocket chat streaming bridge |

---

## Smart Routing

The router classifies each request automatically:

| Condition | Backend |
|---|---|
| Prompt > 4000 tokens | deep |
| Keywords: reason, think, analyze, step by step, debug, refactor… | deep |
| Keywords: write, implement, code, function, fix, review… | mid |
| Everything else | fast |

Override any time with `[route:key]` prefix or `?backend=key` query param.

## Auth And Limits

Authentication is optional and configured in `config/settings.yaml`.
When enabled, inference and admin routes require either `Authorization: Bearer ...` or `x-api-key`.

Built-in rate limiting is not enforced yet, even though the parser accepts a `rate_limit` section for forward compatibility.
If you need throttling today, put the router behind a reverse proxy or API gateway.

---

## Supported Engines

| Engine | Format | Notes |
|---|---|---|
| **llama.cpp** | GGUF | Best for single-GPU, recommended for most users |
| **vLLM** | HuggingFace | High throughput, needs CUDA 12.1+ |
| **SGLang** | HuggingFace | Fast structured generation, needs CUDA 12.1+ |
| **TensorRT-LLM** | TRT engines | Maximum speed on NVIDIA hardware, needs CUDA 12.2+ |
| **HuggingFace TGI** | HuggingFace | Broad model support |

TensorRT-LLM backends include an **auto-tuner** that searches for the largest context window that fits in GPU memory — no manual config needed.

---

## System Diagnostics

```bash
python cli.py sysinfo
```

```
── System Info ────────────────────────────────────────────────
  OS      : Linux x86_64  (6.8.0)
  CPU     : Intel Core i9-14900K  (24 cores)
  RAM     : 64.0 GB

── GPU ────────────────────────────────────────────────────────
  GPU 0   : NVIDIA A100 80GB PCIe
  VRAM    : 80.0 GB total  |  72.1 GB free
  Driver  : 550.90.07
  CUDA    : 12.4

── Engine Versions & Recommendations ──────────────────────────
  llama.cpp      installed: b4286    recommended: b4500+    ✓ compatible
  vLLM           NOT installed       recommended: 0.6.4
                   install: pip install vllm==0.6.4
  TRT-LLM        NOT installed       recommended: 0.14.0
                   install: pip install tensorrt-llm==0.14.0

── Port Conflicts ─────────────────────────────────────────────
  No conflicts detected.
```

---

## Auto-Discovery

The router scans these directories at startup (configurable in `settings.yaml`):

- `~/.lmstudio/models`
- `~/models`
- `~/llm-models`
- `~/.cache/huggingface/hub`
- `~/trt-engines`

Any `.gguf` file, HuggingFace checkpoint, or TRT-LLM engine directory found is registered automatically with size-based tier assignment.

To add a new model without restarting:
```bash
# drop the file in a scan directory, then:
python cli.py rescan
# or: curl -X POST http://localhost:9001/rescan
```

### User overrides

Create `~/.llm-router/overrides.json` to hide or patch any backend:

```json
{
  "exclude": ["some-model-slug"],
  "overrides": {
    "fast": {"ctx_size": 65536, "idle_timeout": 600}
  }
}
```

---

## Keeping Things Updated

```bash
# Update llama.cpp (git pull + cmake rebuild) and Python deps:
python cli.py update

# Nightly cron (update at 3am, no auto-restart):
0 3 * * * cd ~/RouterForLazyPeople && python cli.py update >> ~/.llm-router/logs/update.log 2>&1
```

Python dependencies are **pinned** in `requirements.txt` for stability. Update them deliberately by editing the file and running `python cli.py update`.

---

## Project Structure

```
RouterForLazyPeople/
├── config/
│   ├── settings.yaml       # all tunable settings
│   └── backends.yaml       # your model definitions
├── router/
│   ├── main.py             # FastAPI app, all routes
│   ├── config.py           # YAML loader
│   ├── registry.py         # backend registry builder
│   ├── discovery.py        # auto-discovery (GGUF, HF, TRT)
│   ├── engines.py          # engine detection + command builders
│   ├── routing.py          # request classifier
│   ├── lifecycle.py        # BackendManager (start/stop/watchdog)
│   ├── proxy.py            # OpenAI + Anthropic proxy handlers
│   ├── anthropic_compat.py # Anthropic ↔ OpenAI translation
│   ├── metrics.py          # benchmarking (TTFT, latency, tok/s)
│   ├── sysinfo.py          # hardware + engine detection
│   └── trt_tuner.py        # TRT-LLM memory auto-tuner
├── cli.py                  # command-line interface
└── requirements.txt        # pinned dependencies
```

For the stricter spec split, see [`docs/README.md`](docs/README.md).

---

## License

MIT
