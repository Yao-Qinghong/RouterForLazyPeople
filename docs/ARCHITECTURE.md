# Architecture

This document is the technical and internal design spec for RouterForLazyPeople.
It describes module responsibilities, startup flow, routing flow, and configuration resolution.

## Components

- `router/main.py`: FastAPI app factory, lifespan wiring, middleware registration, route registration, and global error handling
- `router/config.py`: typed config loading, path expansion, and config-file resolution
- `router/registry.py`: merges manual backends with auto-discovered backends and user overrides
- `router/discovery.py`: scans GGUF, HuggingFace, and TRT-LLM sources and assigns inferred metadata
- `router/lifecycle.py`: backend process lifecycle, health checks, preload, restart, idle eviction
- `router/proxy.py`: OpenAI proxying, retries, backpressure, audit logging, and metrics recording
- `router/anthropic_compat.py`: Anthropic-to-OpenAI translation and reverse response adaptation
- `router/gemini_compat.py`: Gemini-to-OpenAI translation and reverse response adaptation
- `router/metrics.py`: in-memory request ring buffer plus persisted metrics history
- `router/benchmark.py`: active PP/TG speed measurement, benchmark-cache persistence, and CLI summary formatting
- `router/sysinfo.py`: platform, GPU, CUDA, engine version, and port-conflict diagnostics

## Startup Flow

1. `create_app()` loads the active settings file and the matching backends file.
2. Logging is configured.
3. The backend registry is built from manual config plus auto-discovery.
4. Cached benchmark results are loaded and passed to the routing classifier.
5. `BackendManager`, `MetricsStore`, and the proxy semaphore are initialized.
6. System diagnostics are captured and stored on `app.state`.
7. Background tasks start for idle eviction and metrics flushing.
8. Optional preload starts configured backends asynchronously.

## Request Routing

1. A compatibility layer translates non-OpenAI payloads into an OpenAI-like shape when needed.
2. The router resolves the backend using query param, alias, route prefix, and classifier precedence.
3. Tool/function-calling requests are treated as deep-tier work unless an explicit route overrides them.
4. When benchmark data exists, tier fallbacks prefer faster measured engines over only using static engine priority.
5. Backpressure is applied through a shared semaphore.
6. `BackendManager.ensure_running()` lazily starts the target backend if needed.
7. The request is proxied to the local backend.
8. Metrics and optional audit events are recorded.
9. Response adaptation runs for Anthropic and Gemini surfaces.

## Backend Lifecycle

- Backends are started lazily on first use unless listed in `preload`.
- Health checks gate a backend being considered ready.
- Idle backends are stopped by the watchdog after their configured timeout.
- `restart()` performs stop, wait, and start through the same lifecycle path.
- TRT-LLM start is special-cased through the tuner and can persist a working memory configuration.
- Ollama backends are treated as an external managed server rather than a normal subprocess.

## Config Precedence

Settings resolution order:

1. Explicit path passed to `load_config()`
2. `LLM_ROUTER_CONFIG`
3. Project-local `config/settings.yaml`
4. `~/.llm-router/settings.yaml`

Backends resolution order:

1. Explicit path passed to `load_config()`
2. `backends.yaml` beside the active settings file
3. `~/.llm-router/backends.yaml`

Other config rules:

- `~`-prefixed paths are expanded during load.
- `AppConfig.settings_file` and `AppConfig.backends_file` are retained so reload operations keep using the same source files.
- `overrides.json` is applied after registry merge and can exclude or patch discovered or manual backends.

## Runtime Boundaries

- Middleware is constructed at app creation time. Auth and CORS are not hot-swapped by `/reload-config`.
- `/reload-config` only updates mutable runtime structures already held on `app.state.config`.
- Metrics are summarized from the in-memory ring buffer and persisted to JSONL on a flush loop.
- Benchmark results are separate from request metrics: `bench` writes cache files, and startup/rescan load those files into the classifier.
- WebSocket chat streaming bypasses parts of the normal HTTP response path but still uses routing and lazy backend start.

## Deliberate Non-Goals

- The router is not an all-in-one production API gateway.
- Built-in rate limiting is not implemented even though config parsing accepts a placeholder section.
- Full dynamic reconfiguration of middleware and backend subprocess flags is not part of the current architecture.
