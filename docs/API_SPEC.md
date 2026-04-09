# API Spec

This document is the public and functional specification for RouterForLazyPeople.
It describes the externally visible behavior that clients and integrators can rely on.

## Scope

- OpenAI-compatible request proxying
- Anthropic Messages API compatibility
- Gemini API compatibility
- Backend discovery and control endpoints
- Auth expectations and routing semantics

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
- Cached `bench` results are used to prefer faster measured backends when the router must choose among compatible fallbacks.
- Everything else falls back to `fast`.
- If a tier is missing, routing falls back to any available backend.

## Auth Behavior

- Auth is optional and configured in `config/settings.yaml`.
- When enabled, clients may send a key via `Authorization: Bearer ...` or `x-api-key`.
- Public routes remain accessible without auth:
  - `GET /health`
  - `GET /status`
  - `GET /backends`
  - `GET /v1/models`
  - `GET /v1/models/{model_id}`
  - `GET /engines`
  - `GET /sysinfo`
  - `GET /metrics`
  - `GET /metrics/export`
  - `GET /metrics/prometheus`
  - `GET /benchmarks`
- Inference routes require `inference` or `all` scope.
- Admin/control routes require `admin` or `all` scope.

## Public Endpoints

| Method | Path | Contract |
|---|---|---|
| `GET` | `/health` | Liveness check, returns router health |
| `GET` | `/status` | Runtime state of registered backends |
| `GET` | `/backends` | Backend registry summary |
| `GET` | `/v1/models` | OpenAI-style model list from registered backends and aliases |
| `GET` | `/v1/models/{model_id}` | OpenAI-style single-model lookup |
| `GET` | `/engines` | Installed engine availability |
| `GET` | `/sysinfo` | System diagnostics, engine versions, and recommendations |
| `GET` | `/metrics` | Aggregated request metrics |
| `GET` | `/metrics/export` | CSV export of metrics history |
| `GET` | `/metrics/prometheus` | Prometheus exposition format |
| `GET` | `/benchmarks` | Cached PP/TG speed benchmark results written by `python cli.py bench` |
| `POST` | `/start/{key}` | Start a backend |
| `POST` | `/stop/{key}` | Stop a backend |
| `POST` | `/restart/{key}` | Restart a backend |
| `POST` | `/rescan` | Rebuild registry from config plus auto-discovery |
| `POST` | `/reload-config` | Reload mutable router settings from the active `settings.yaml` |
| `POST` | `/retune/{key}` | Clear TRT-LLM tuning cache and re-tune on start |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/completions` | OpenAI-compatible legacy completions |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings |
| `POST` | `/v1/{path}` | Catch-all OpenAI-compatible inference proxy |
| `POST` | `/anthropic/v1/messages` | Anthropic Messages API compatibility |
| `POST` | `/v1/messages` | Anthropic Messages alias |
| `POST` | `/gemini/v1beta/models/{model}:generateContent` | Gemini-compatible non-streaming generation |
| `POST` | `/gemini/v1beta/models/{model}:streamGenerateContent` | Gemini-compatible streaming generation |
| `WS` | `/v1/chat/completions/ws` | WebSocket bridge for chat streaming |

## Route-Specific Notes

- `POST /reload-config` reloads mutable runtime settings only. It does not rebuild middleware or mutate already-running backend process flags in place.
- `POST /rescan` preserves already-running processes while updating the backend registry.
- `POST /rescan` also refreshes the cached benchmark data used by automatic routing.
- `GET /benchmarks` returns cached active speed-test results. Request-traffic metrics stay under `/metrics`.
- `POST /v1/{path}` is the generic catch-all for OpenAI-compatible inference paths not exposed as dedicated route helpers.
- `GET /v1/models` includes configured aliases when they resolve to a registered backend.

## Unsupported Or Reserved Behavior

- Built-in rate limiting is not part of the functional contract yet. The config parser accepts `rate_limit`, but the router does not enforce those limits.
- TLS termination, production-grade gateway throttling, and edge auth hardening are outside the router’s public API contract.
