# API Spec

This document is the public and functional specification for RouterForLazyPeople.
It describes the externally visible behavior that clients and integrators can rely on.

## Scope

**Phase 1 (tested, supported):**
- OpenAI-compatible chat completions (`/v1/chat/completions`)
- OpenAI-compatible model list (`/v1/models`)
- Backend discovery and control endpoints
- Engine: llama.cpp only

**Phase 2+ (code exists, untested):**
- Anthropic Messages API compatibility
- Gemini API compatibility
- Additional engines (vLLM, SGLang, TRT-LLM, HuggingFace, Ollama)
- Embeddings, legacy completions, WebSocket bridge, catch-all inference proxy

Clients should only depend on phase-1 endpoints for production use. Phase-2+ endpoints are available for experimentation but may have bugs or incomplete behavior.

## Client Compatibility

- OpenAI SDK and OpenAI-compatible apps use the `/v1/*` surface.
- Anthropic SDK and Claude Code use `/anthropic/v1/messages` or `/v1/messages`.
- Gemini-compatible clients use `/gemini/v1beta/models/{model}:generateContent` and `/gemini/v1beta/models/{model}:streamGenerateContent`.

## Backend Selection Rules

Selection precedence is:

1. `?backend=<key>` query parameter
2. Configured `model_aliases`
3. Explicit `[route:key]` prefix in the first message content
4. Automatic classification from token thresholds and keyword lists

Automatic classification behavior:

- Prompts above `routing.token_threshold_deep` route to the `deep` tier.
- Prompts above `routing.token_threshold_mid` route to the `mid` tier.
- `routing.deep_keywords` prefer the `deep` tier.
- `routing.mid_keywords` prefer the `mid` tier.
- Tool/function-calling requests prefer the `deep` tier.
- Within the selected tier, `_pick()` ranks backends by: (1) capability match (tool support, JSON schema), (2) highest measured TG tok/s from benchmarks, (3) engine capability rank, (4) round-robin among ties.
- Everything else falls back to `fast`.
- If no backend exists in the classified tier, `_pick()` falls back to the first registered backend regardless of tier. This is a known limitation — the fallback is not health-aware or benchmark-ranked. A future version should either return 503 or apply the same ranking logic across all tiers.

Thinking / reasoning behavior:

- Normal proxied chat requests are passed through. The router does not add `/think`, `/no_think`, `reasoning_effort`, or chain-of-thought instructions.
- Client-supplied `/think` and `/no_think` prompt text is forwarded unchanged.
- Backend config field `reasoning` is an engine startup / parsing setting for supported engines; it is not the same thing as injecting a prompt-level thinking directive.
- `bench` uses a `/no_think` prompt directive by default so cached routing speed reflects direct-answer throughput. CLI users may opt into `bench --thinking` for a reasoning-mode speed check.
- `bench` measures currently running backends by default. Starting stopped backends for measurement requires the explicit CLI option `--start-stopped`.

## Auth Behavior

- Auth is optional and configured in `config/settings.yaml`.
- When enabled, clients may send a key via `Authorization: Bearer ...` or `x-api-key`.
- Public routes remain accessible without auth:
  - `GET /health`
  - `GET /status`
  - `GET /backends`
  - `GET /benchmarks`
  - `GET /v1/models`
  - `GET /v1/models/{model_id}`
  - `GET /engines`
  - `GET /sysinfo`
  - `GET /metrics`
  - `GET /metrics/export`
  - `GET /metrics/prometheus`
  - `GET /docs`, `GET /openapi.json`, `GET /redoc` (FastAPI auto-generated)
- Inference routes require `inference` or `all` scope.
- Admin/control routes require `admin` or `all` scope.

## Public Endpoints

| Method | Path | Phase | Contract |
|---|---|---|---|
| `GET` | `/health` | 1 | Liveness check, returns router health |
| `GET` | `/status` | 1 | Runtime state of registered backends |
| `GET` | `/backends` | 1 | Backend registry summary |
| `GET` | `/v1/models` | 1 | OpenAI-style model list from registered backends and aliases |
| `GET` | `/v1/models/{model_id}` | 1 | OpenAI-style single-model lookup |
| `GET` | `/engines` | 1 | Installed engine availability |
| `GET` | `/sysinfo` | 1 | System diagnostics, engine versions, and recommendations |
| `GET` | `/metrics` | 1 | Aggregated request metrics |
| `GET` | `/metrics/export` | 1 | CSV export of metrics history |
| `GET` | `/metrics/prometheus` | 1 | Prometheus exposition format |
| `GET` | `/benchmarks` | 1 | Cached PP/TG speed benchmark results written by `python cli.py bench` |
| `POST` | `/start/{key}` | 1 | Start a backend |
| `POST` | `/stop/{key}` | 1 | Stop a backend |
| `POST` | `/restart/{key}` | 1 | Restart a backend |
| `POST` | `/rescan` | 1 | Rebuild registry from config plus auto-discovery |
| `POST` | `/reload-config` | 1 | Reload mutable router settings from the active `settings.yaml` |
| `POST` | `/retune/{key}` | 2+ | Clear TRT-LLM tuning cache and re-tune on start |
| `POST` | `/v1/chat/completions` | 1 | OpenAI-compatible chat completions |
| `POST` | `/v1/completions` | 2+ | OpenAI-compatible legacy completions |
| `POST` | `/v1/embeddings` | 2+ | OpenAI-compatible embeddings |
| `POST` | `/v1/{path}` | 2+ | Catch-all OpenAI-compatible inference proxy |
| `POST` | `/anthropic/v1/messages` | 2+ | Anthropic Messages API compatibility |
| `POST` | `/v1/messages` | 2+ | Anthropic Messages alias |
| `POST` | `/gemini/v1beta/models/{model}:generateContent` | 2+ | Gemini-compatible non-streaming generation |
| `POST` | `/gemini/v1beta/models/{model}:streamGenerateContent` | 2+ | Gemini-compatible streaming generation |
| `WS` | `/v1/chat/completions/ws` | 2+ | WebSocket bridge for chat streaming |

## Route-Specific Notes

- `POST /reload-config` reloads mutable runtime settings only. It does not rebuild middleware or mutate already-running backend process flags in place.
- `POST /rescan` preserves already-running processes while updating the backend registry.
- `POST /rescan` also refreshes the cached benchmark data used by automatic routing.
- `GET /engines` reports availability gated by `engines_enabled` in `settings.yaml`. An installed engine that is not listed in `engines_enabled` will show as unavailable.
- `GET /benchmarks` returns cached active speed-test results. Request-traffic metrics stay under `/metrics`.
- `POST /v1/{path}` is the generic catch-all for OpenAI-compatible inference paths not exposed as dedicated route helpers.
- `GET /v1/models` includes configured aliases when they resolve to a registered backend.

## Unsupported Or Reserved Behavior

- Built-in rate limiting is not part of the functional contract yet. The config parser accepts `rate_limit`, but the router does not enforce those limits.
- TLS termination, production-grade gateway throttling, and edge auth hardening are outside the router’s public API contract.
