# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start / stop the router
python cli.py               # start (default when no subcommand given)
python cli.py start         # same as above
python cli.py stop
python cli.py status

# Install as a system service (auto-starts on every login — macOS launchd or Linux systemd)
python cli.py service install
python cli.py service uninstall
python cli.py service status

# Run tests
pytest                        # all tests
pytest tests/test_routing.py  # single file

# Lint / type check (not yet configured — add ruff/mypy if needed)

# Hardware diagnostics
python cli.py sysinfo

# Force backend rescan
python cli.py rescan

# Rebuild TRT-LLM config from scratch
curl -X POST http://localhost:9001/retune/<backend-key>

# Docker
docker compose up -d
```

## Architecture

### Request flow

```
client → router/main.py (FastAPI)
           ├── AuthMiddleware (router/auth.py, off by default)
           ├── POST /v1/chat/completions  → proxy.handle_proxy()
           ├── POST /anthropic/v1/messages → proxy.handle_anthropic_proxy()
           │     └── anthropic_compat.py translates ↔ OpenAI format
           ├── POST /v1/messages          (same as above, alias)
           └── POST /gemini/…             → proxy.handle_gemini_proxy()
                 └── gemini_compat.py translates ↔ OpenAI format

proxy.py:
  1. resolve model alias (config.model_aliases: "gpt-4" → "deep")
  2. routing.classify(payload, backends, config) → backend key
  3. lifecycle.BackendManager.ensure_running(key)  ← lazy-starts subprocess
  4. forward request via httpx (streaming or buffered)
  5. record metrics (metrics.MetricsStore.record)
```

### Configuration (no Python editing for operators)

- **`config/settings.yaml`** — all tunable knobs: host/port, logging, llama_bin, scan_dirs, routing thresholds, keywords, timeouts, proxy settings, metrics, model_aliases, preload list, auth, CORS, rate limiting.
- **`config/backends.yaml`** — manual backend definitions (slug → port, model path, engine, tier, idle_timeout, etc.).

`router/config.py` loads both files into `AppConfig` dataclasses. Search order for settings.yaml: CLI arg → `$LLM_ROUTER_CONFIG` → `./config/settings.yaml` → `~/.llm-router/settings.yaml`.

### AppConfig injection pattern

All modules accept `config: AppConfig` as a parameter — no globals. The config object is created once in `main.py` and passed down. This makes unit testing straightforward: construct a minimal `AppConfig` and pass it directly.

### Backend lifecycle (router/lifecycle.py)

`BackendManager` manages subprocesses for inference servers:
- **Lazy start**: `ensure_running(key)` is called on every request; if the process isn't running, it spawns it and polls `/health` until ready or `startup_wait` expires.
- **Per-backend asyncio.Lock** prevents duplicate starts from concurrent requests.
- **Idle eviction**: `idle_watchdog()` runs every 30s and calls `stop(key)` for backends silent longer than `idle_timeout`.
- **Ollama exception**: uses `_OllamaSentinel` instead of a real `Popen` object; `poll()` pings the Ollama HTTP API to test liveness.
- **TRT-LLM**: delegates to `trt_tuner.TRTLLMTuner` which searches 6 progressively-smaller memory configs until health check passes, then saves the winning config to `data_dir/tuning/<key>.json`.

### Adding a new engine

1. Add constants + `is_engine_available()` branch + `build_<engine>_cmd()` in `router/engines.py`.
2. Add `elif engine == "..."` branch in `BackendManager._build_cmd()` (`lifecycle.py`).
3. Add version detection in `_detect_engine_versions()` and a row in `COMPATIBILITY` (`sysinfo.py`).

### Format compatibility layers

| Client format | Entry point | Translator |
|---|---|---|
| OpenAI | `POST /v1/chat/completions` | none (native) |
| Anthropic | `POST /anthropic/v1/messages` or `/v1/messages` | `router/anthropic_compat.py` |
| Google Gemini | `POST /gemini/…` | `router/gemini_compat.py` |

Both translators handle streaming (SSE) and non-streaming paths separately.

### Discovery (router/discovery.py + router/registry.py)

`build_backend_registry(config)` merges manually-defined backends with auto-discovered ones:
- GGUF files in `config.scan_dirs.gguf`
- HuggingFace checkpoints in `config.scan_dirs.hf`
- TRT-LLM engines in `config.scan_dirs.trtllm`

Tier assignment uses `config.tier_thresholds` (file size bins). Applies overrides from `~/.llm-router/overrides.json` (exclude/patch individual backends).

### Metrics (router/metrics.py)

- In-memory deque (last 1000 requests, per-backend stats)
- Flushed to `~/.llm-router/metrics/YYYY-MM-DD.jsonl` every `flush_interval_sec`
- Endpoints: `GET /metrics` (JSON), `GET /metrics/prometheus` (Prometheus text), `GET /metrics/export` (CSV download)

### Reference artifacts

- **`router.py`** (root, 1141 lines) — original monolith kept for historical reference. **Not used by the running system.**
- **`start_router.sh`** — original shell launcher. **Not used.**
