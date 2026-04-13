# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start / stop the router
./router-start              # start (shell wrapper — works on any system)
./router-start stop
./router-start status

# Install as a system service (auto-starts on every login — macOS launchd or Linux systemd)
./router-start service install
./router-start service uninstall
./router-start service status

# Run tests
pytest                        # all tests
pytest tests/test_routing.py  # single file

# Benchmark all backends (measures real PP/TG tok/s, saves to ~/.llm-router/benchmarks/)
python cli.py bench
python cli.py bench --backend <backend-key>   # single backend only

# Hardware diagnostics (GPU, CUDA, CPU, installed engine versions, install recommendations)
python cli.py sysinfo

# Force backend rescan (re-detect running servers + re-scan model dirs)
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
  2. model name → backend key direct match (if payload.model == a backend key)
  3. routing.classify(payload, backends, config) → tier → _pick() → backend key
  4. lifecycle.BackendManager.ensure_running(key)  ← lazy-starts subprocess
  5. forward request via httpx (streaming or buffered)
  6. record metrics (metrics.MetricsStore.record)
```

### Routing classification (router/routing.py)

`classify()` assigns a tier in priority order:

1. `[route:key]` prefix in any message → exact backend key (highest priority)
2. Structural signals determine minimum tier:
   - Tool use / function calling → `deep`
   - JSON schema response format → at least `mid`
   - Token count > `token_threshold_deep` (4000) → `deep`
   - System prompt > 2000 tokens → at least `mid`
   - More than 10 messages → at least `mid`
   - Token count > `token_threshold_mid` (500) → at least `mid`
3. Keywords as **soft tiebreaker** — can push `fast→mid`, but cannot force `deep` alone. Keywords only confirm `mid→deep` when combined with a structural signal (token count > mid threshold).
4. Default → `fast`

`_pick(backends, tier)` selects the best backend within a tier:

1. Highest measured TG tok/s from `python cli.py bench` results (cached in `_bench`)
2. Engine capability rank fallback: `trt-llm > trt-llm-docker > vllm > sglang > llama.cpp > huggingface > openai > ollama`
3. Round-robin only among backends with identical score (load balancing)

Benchmark data is injected at startup via `set_benchmark_results(load_all_results(config))` and refreshed after `/rescan`. Without benchmarks, routing falls back to engine rank silently.

### Configuration (no Python editing for operators)

- **`config/settings.yaml`** — all tunable knobs: `engines_enabled` (default: `["llama.cpp"]`), host/port, logging, llama_bin, scan_dirs, routing thresholds, keywords, timeouts, proxy settings, metrics, model_aliases, preload list, auth, CORS, rate limiting.
- **`config/backends.yaml`** — manual backend definitions (slug → port, model path, engine, tier, idle_timeout, etc.). Empty by default; auto-discovery handles fresh installs.

`router/config.py` loads both files into `AppConfig` dataclasses. Search order for settings.yaml: CLI arg → `$LLM_ROUTER_CONFIG` → `./config/settings.yaml` → `~/.llm-router/settings.yaml`.

### AppConfig injection pattern

All modules accept `config: AppConfig` as a parameter — no globals. The config object is created once in `main.py` and passed down. This makes unit testing straightforward: construct a minimal `AppConfig` and pass it directly.

### Backend lifecycle (router/lifecycle.py)

`BackendManager` manages subprocesses for inference servers:
- **Lazy start**: `ensure_running(key)` is called on every request; if the process isn't running, it spawns it and polls `/health` until ready or `startup_wait` expires.
- **Per-backend asyncio.Lock** prevents duplicate starts from concurrent requests.
- **Registry lock**: `update_registry()` is async and acquires `_registry_lock` so `/rescan` doesn't race with in-flight requests.
- **Backend snapshotting**: `snapshot_backends()` returns the current dict reference. Proxy handlers call it once at request start for a consistent view across `await` points.
- **Log handle safety**: `_close_log(key)` is called on every failure path and inside `_open_log()` before reopening, preventing file descriptor leaks during retries.
- **Idle eviction**: `idle_watchdog()` runs every 30s and calls `stop(key)` for backends silent longer than `idle_timeout`.
- **External servers** (LM Studio, custom OpenAI-compatible): uses `_ExternalSentinel` instead of `Popen`; `poll()` pings `/v1/models` to test liveness; `kill()`/`terminate()` are no-ops (we don't own them).
- **Ollama**: uses `_OllamaSentinel`; `poll()` pings the Ollama HTTP API.
- **TRT-LLM**: delegates to `trt_tuner.TRTLLMTuner` which searches 6 progressively-smaller memory configs until health check passes, then saves the winning config to `data_dir/tuning/<key>.json`.

### Auto-detection of running servers (router/discovery.py)

`detect_running_servers(config)` is called by `build_backend_registry()` on every start and rescan:
- Probes LM Studio at `localhost:1234/v1/models` → registers as `engine=openai`
- Probes Ollama at `localhost:11434/api/tags` → registers one backend per model
- Parallel-probes `config.discovery.probe_ports` (default: 8080, 8000, 8001, 8002, 30000) for any OpenAI-compatible server

MoE tier classification: names like `35B-A3B` or `8x7B` extract active params instead of using file size. Thresholds: `< 7B active → fast`, `7–20B active → mid`, `> 20B active → deep`.

### Adding a new engine

1. Add constants + `is_engine_available()` branch + `build_<engine>_cmd()` in `router/engines.py`.
2. Add `elif engine == "..."` branch in `BackendManager._build_cmd()` (`lifecycle.py`).
3. Add version detection in `_detect_engine_versions()` and a row in `COMPATIBILITY` (`sysinfo.py`).

### Benchmarking (router/benchmark.py)

`measure_backend(key, cfg, config)` runs two streaming requests directly to the backend (bypassing the router):
- **PP benchmark**: ~400-token prompt, `max_tokens=1` — TTFT measures prefill speed
- **TG benchmark**: short prompt, `max_tokens=80` — measures tok/s after first token

Results saved to `~/.llm-router/benchmarks/<key>.json`. `load_all_results()` reads all cached results; `set_benchmark_results()` injects them into `routing.py`'s `_bench` cache. Tier thresholds: `≥ 30 tok/s → fast`, `≥ 10 → mid`, `< 10 → deep`.

### System detection (router/sysinfo.py)

`detect_system()` returns a dict with OS, CPU arch/model/cores, RAM, GPU (name, VRAM total/free), CUDA version, installed engine versions, stable version recommendations with install commands, and any port conflicts. Used by `GET /sysinfo` and `python cli.py sysinfo`. Never raises — all fields have safe fallbacks.

### Format compatibility layers

| Client format | Entry point | Translator |
|---|---|---|
| OpenAI | `POST /v1/chat/completions` | none (native) |
| Anthropic | `POST /anthropic/v1/messages` or `/v1/messages` | `router/anthropic_compat.py` |
| Google Gemini | `POST /gemini/…` | `router/gemini_compat.py` |

Both translators handle streaming (SSE) and non-streaming paths separately.

### Discovery (router/discovery.py + router/registry.py)

`build_backend_registry(config)` merges in priority order (manual wins on key collision):
1. Manually-defined backends from `config/backends.yaml`
2. Auto-detected running servers (`detect_running_servers`)
3. Discovered model files — merged as `{**gguf, **hf, **trt}`, so on slug collision **TRT-LLM > HF > GGUF** takes the winning config

Applies overrides from `~/.llm-router/overrides.json` (exclude/patch individual backends).

### Metrics (router/metrics.py)

- In-memory deque (last 1000 requests, per-backend stats)
- Flushed to `~/.llm-router/metrics/YYYY-MM-DD.jsonl` every `flush_interval_sec`
- Endpoints: `GET /metrics` (JSON), `GET /metrics/prometheus` (Prometheus text), `GET /metrics/export` (CSV download)

### Key API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /status` | Backend run-state + cached benchmark data per backend |
| `GET /backends` | All registered backends (key, engine, tier, port, size) |
| `GET /benchmarks` | Raw cached benchmark results for all backends |
| `GET /sysinfo` | Hardware, CUDA, engine versions, install recommendations |
| `GET /v1/models` | OpenAI-compatible model list (backend keys as model IDs) |
| `POST /rescan` | Re-detect running servers + re-scan model dirs + refresh benchmarks |
| `POST /retune/{key}` | Force TRT-LLM re-tune from scratch |

### Entry point

- **`router-start`** — shell wrapper that auto-selects `.venv/bin/python` → `python3` → `python`. Used as the ExecStart in both launchd (macOS) and systemd (Linux) service units.
