# Architecture

This is the implementation-contract spec for RouterForLazyPeople.
It defines domain objects, state machines, contracts, and failure policy — not just file names.

---

## Phase 1 Contract

Phase 1 is: **OpenAI-compatible proxy for llama.cpp backends, serving OpenClaw and OpenCode.**

In scope:
- `POST /v1/chat/completions` — non-streaming and SSE streaming
- `GET /v1/models` — model list
- Tool/function calling
- Structured output / JSON schema mode (`response_format.type == "json_schema"`)
- Lazy single-backend start on DGX Spark
- Benchmark-informed within-tier backend selection

Deferred (not phase-1 targets):
- `developer` role rewrite to `system` before forwarding (not yet implemented)
- Anthropic and Gemini proxy surfaces
- WebSocket chat bridge
- vLLM, SGLang, TRT-LLM, HuggingFace engines
- Embeddings endpoint
- Rate limiting enforcement
- Preflight context-overflow rejection (prompt token count vs `ctx_size`)

`engines_enabled` must remain `["llama.cpp"]` during phase-1 development and testing.

---

## Assumptions

- **Trusted LAN only.** Auth is optional. No edge security.
- **Single node, no HA.** One router process. One active backend at a time is the designed operating point.
- **No automatic cloud fallback.** The router never selects a remote provider automatically. Remote/external backends require explicit `?backend=` or `[route:key]` routing.
- **DGX Spark / CUDA Linux is the primary target.** macOS with Metal is a development convenience.
- **Operator-managed hardware.** The router does not manage GPU allocation across processes.

---

## Constraints

- Concurrent in-flight requests are capped by a semaphore (`proxy.max_concurrent_requests`, default: 20). When full the request queues for up to `proxy.queue_timeout_sec` (default: 30s), then returns 503 on OpenAI/Gemini routes or 529 on Anthropic routes.
- Backend start budget: `startup_wait` seconds from subprocess spawn to `/health` passing (default: 30s, per-backend field). This is separate from per-request inference timeout.
- One backend active at a time is the DGX Spark operating point. Multiple backends can be registered, but concurrent VRAM usage is the operator's responsibility.
- Stale or missing benchmark data silently falls back to engine-rank ordering. It does not block routing.

---

## Core Domain Objects

### BackendDescriptor

```python
@dataclass
class BackendDescriptor:
    key: str            # unique slug, e.g. "llama-mistral-7b"
    engine: str         # "llama.cpp" | "vllm" | "ollama" | "openai" | ...
    tier: str           # "fast" | "mid" | "deep"
    host: str
    port: int
    model_path: str | None   # absolute path for managed engines; None for external
    ctx_size: int            # context window in tokens
    n_parallel: int          # parallel slots (llama.cpp --parallel); default 1
    idle_timeout: int        # seconds before idle eviction; 0 = never
    is_external: bool        # True → router does not own the process
    capabilities: Capabilities
```

### Capabilities

```python
@dataclass
class Capabilities:
    streaming: bool          # supports SSE streaming
    tools: bool              # supports tool/function calling
    structured_output: bool  # supports JSON schema / grammar mode
    embeddings: bool         # supports /v1/embeddings
    context_limit: int       # hard context ceiling in tokens
    local: bool              # True if process runs on same node as router
```

llama.cpp defaults:
`streaming=True, tools=True, structured_output=True, embeddings=False, context_limit=ctx_size, local=True`

### NormalizedError

All backend errors returned to clients must use this shape:

```json
{
  "error": {
    "message": "<human-readable>",
    "type": "<error_type>",
    "code": <http_status>
  }
}
```

| Condition | HTTP | type |
|---|---|---|
| No backend available | 503 | `backend_unavailable` |
| Backend start timeout | 504 | `backend_start_timeout` |
| Semaphore queue timeout | 503 (OpenAI/Gemini) or 529 (Anthropic) | `overloaded_error` (Anthropic) / plain error (OpenAI) |
| Backend 4xx (bad request) | forward 4xx | `backend_error` |
| Backend 5xx | 502 | `backend_error` |
| Partial stream failure | 502 | `stream_error` |

---

## Backend Lifecycle State Machine

```
discovered
    │  build_backend_registry()
    ▼
registered ◄─── rescan()
    │  ensure_running() called
    ▼
starting
    ├── startup_wait expires without /health passing ──► unhealthy
    │  /health passes
    ▼
ready ◄── restart() completes
    ├── /health fails after 3 consecutive probes ──► unhealthy
    ├── process exits / crash detected ──────────────► unhealthy
    │  idle_timeout exceeded
    ▼
stopping
    │  SIGTERM + wait
    ▼
stopped
    │  ensure_running() called again
    └──────────────────────────────────────────────► starting
```

External backends (`is_external=True`) skip `starting` and enter a `ready/unhealthy` polling cycle via `poll()` against their `/v1/models` or `/health` endpoint.

**Unhealthy quarantine:** A backend in `unhealthy` is excluded from routing. It stays quarantined until `restart()` is called explicitly. The router does not auto-restart unhealthy backends during normal operation.

**Crash mid-request:** If the backend process exits while a request is in flight, the proxy detects the broken connection, returns 502 `stream_error`, and transitions the backend to `unhealthy`. The request is not retried.

---

## Routing Algorithm

Selection runs in strict priority order. The first matching rule wins.

1. `?backend=<key>` — must resolve to a `ready` backend; 503 if not.
2. `model_aliases` entry — alias resolves to a backend key, then treated as rule 1.
3. `[route:key]` prefix in first message content — same rules as rule 1.
4. Automatic classification:
   - a. Classify tier from payload (see Tier Classification below).
   - b. Filter candidates: `state == ready AND tier == classified_tier AND has_required_capabilities(payload)`.
   - c. **If no candidates in the classified tier:** current code falls back to the first registered backend (any tier, no ranking). This is a known limitation — a future version should either return 503 or apply the full ranking logic across all tiers.
   - d. Among candidates: rank by measured TG tok/s (bench cache), then engine rank, then round-robin for ties.

**Capability filter (step b):** Payloads with `tools` require `capabilities.tools == True`. Payloads with `response_format.type == "json_schema"` require `capabilities.structured_output == True`.

**Local-first:** `local=True` backends are ranked ahead of `local=False` regardless of benchmark scores.

**No automatic cloud fallback:** Backends with `local=False` are never selected by the automatic classifier.

### Tier Classification

Priority order; first match wins:

1. Payload contains tool definitions → `deep`
2. Token count > `token_threshold_deep` (default: 4000) → `deep`
3. System prompt > 2000 tokens OR message count > 10 → `mid`
4. Token count > `token_threshold_mid` (default: 500) → `mid`
5. Deep keywords (soft signal): can push `fast→mid`; cannot force `deep` alone. Keywords only confirm `mid→deep` when combined with a structural signal.
6. Default → `fast`

### Engine Rank Fallback (within tier)

When no bench data is available, selection within a tier uses this rank:
`trt-llm > trt-llm-docker > vllm > sglang > llama.cpp > huggingface > openai > ollama`

---

## llama.cpp Operational Contract

llama.cpp is the phase-1 backend. These are hard rules.

### Startup

- Command: `llama-server --host 127.0.0.1 --port <port> --model <path> --ctx-size <ctx_size> --parallel <n_parallel>` plus any `extra_args` from config.
- Readiness probe: `GET http://127.0.0.1:<port>/health` polled every 2s.
  - Pass: HTTP 200 with `{"status": "ok"}`.
  - After 3 consecutive failures once previously ready → transition to `unhealthy`.
- If `startup_wait` expires without readiness: log stderr tail, set state `unhealthy`, close log handle.
- No pre-warming prompt is sent. Backend is ready as soon as `/health` passes.

### Timeouts (these are separate)

| Timeout | What it covers | Default | Configurable |
|---|---|---|---|
| `startup_wait` | Spawn to first `/health` pass | 30s | Per-backend (`backends.yaml`) |
| `proxy.timeout_sec` | Request forwarded to final byte | 300s | `settings.yaml` |

For SSE streaming: `proxy.timeout_sec` covers time to first token. After first token arrives, chunks are forwarded without an additional per-chunk timeout.

### Concurrency and Slot Policy

- `n_parallel` in backend config sets `--parallel` on llama-server (parallel completion slots).
- When semaphore is exhausted: request queues for up to `queue_timeout_sec` (default 30s), then returns 503 (OpenAI/Gemini) or 529 (Anthropic).
- When llama-server slots are exhausted (backend returns 503): proxy returns 503. Do not retry.

### Context Overflow

- **Not yet implemented.** There is no preflight token-count vs `ctx_size` check. Prompts are forwarded to the backend as-is; if the backend rejects with a 4xx, the behavior depends on the request mode: non-streaming requests pass the 4xx through directly, while streaming requests emit the error as an SSE error event inside a 200 response.

### Retry Policy

| Request type | Retry on start failure | Retry on inference failure |
|---|---|---|
| Plain text, no streaming | Once if backend just became ready | No |
| Streaming, no output sent yet | Once if zero bytes sent to client | No |
| Streaming, output started | No | No |
| Tool calling | No | No |
| Structured output | No | No |

"Start failure" = `ensure_running()` fails. "Inference failure" = request reached backend and failed.

### llama-server Error Mapping

| llama-server response | Proxy action |
|---|---|
| HTTP 200, SSE stream | Forward chunks. On broken pipe after output started: send `data: [DONE]`, mark backend `unhealthy` if process exited. |
| HTTP 400 | Forward as-is. |
| HTTP 503 (slots full) | Return 503 `backend_unavailable`. Do not retry. |
| HTTP 5xx other | Return 502 `backend_error`. |
| Connection refused | State → `unhealthy`. Return 503 `backend_unavailable`. |
| Process exited mid-stream | Return 502 `stream_error`. State → `unhealthy`. |

---

## OpenClaw / OpenCode Compatibility

These are phase-1 acceptance requirements, not aspirational claims.

### Required Endpoint Behavior

| Feature | OpenClaw | OpenCode |
|---|---|---|
| `GET /v1/models` with stable `id` | Required | Required |
| `POST /v1/chat/completions` non-streaming | Required | Required |
| `POST /v1/chat/completions` SSE streaming | Required | Required |
| Tool/function calling (OpenAI schema) | Required | Optional |
| `response_format: json_schema` | Required | Required |
| Auth header accepted when auth disabled | Required | Required |

### SSE Streaming Contract

- Each chunk: `data: <json>\n\n` with a valid `choices[0].delta` object.
- Stream terminates with `data: [DONE]\n\n`.
- No bare newlines or non-SSE content before `[DONE]`.
- Proxy must not buffer the full response before forwarding.

### OpenClaw-Specific

- `model` field in responses must match what OpenClaw registered as the model identifier. Use the backend key or configured alias as the `id` in `/v1/models` and in response `model` fields — stable across restarts.
- Tool calls must follow OpenAI schema exactly: `tool_calls[].function.name`, `tool_calls[].function.arguments` (JSON string), `tool_calls[].id`.
- `developer` role rewrite to `system`: **not yet implemented** — currently forwarded unchanged. Deferred to a future phase.
- OpenClaw setup: `baseURL = http://localhost:9001/v1`, model name = a registered backend key.

### OpenCode-Specific

- OpenCode selects by `provider/model` identity. Expose backends as model IDs in `/v1/models`. IDs must be stable across restarts.
- `baseURL = http://localhost:9001/v1`. No additional path prefix.
- Router must accept and ignore `Authorization: Bearer <key>` when auth is disabled.
- Non-tool responses: `choices[0].message.content` must be a string (not null, not array).

---

## Config Schema

### settings.yaml

| Field | Type | Default | Requires restart? |
|---|---|---|---|
| `router.host` | str | `"0.0.0.0"` | Yes |
| `router.port` | int | `9001` | Yes |
| `engines_enabled` | list[str] | `["llama.cpp"]` | Rescan |
| `proxy.max_concurrent_requests` | int | `20` | Yes |
| `proxy.timeout_sec` | int (s) | `300` | `/reload-config` |
| `proxy.queue_timeout_sec` | int (s) | `30` | `/reload-config` |
| `routing.token_threshold_deep` | int | `4000` | `/reload-config` |
| `routing.token_threshold_mid` | int | `500` | `/reload-config` |
| `routing.deep_keywords` | list[str] | `[]` | `/reload-config` |
| `routing.mid_keywords` | list[str] | `[]` | `/reload-config` |
| `scan_dirs` | list[str] | platform defaults | Rescan |
| `data_dir` | str | `~/.llm-router` | Yes |

`startup_wait` (default: 30s) is a per-backend field in `backends.yaml`, not a top-level setting.

### Startup Validation (abort on error, not warning)

- `port` is already in use.
- `engines_enabled` contains an unknown engine name.
- `backends.yaml` references an engine not in `engines_enabled`.
- `backends.yaml` backend has a `model_path` that does not exist on disk.

### Hot-Reload Boundaries

| Mechanism | Updates |
|---|---|
| `/reload-config` | Routing thresholds, keywords, timeouts, log level |
| `/rescan` | Backend registry, benchmark cache (running processes untouched) |
| Router restart | Host/port binding, middleware (auth, CORS), semaphore size |

Middleware changes (auth enable/disable, CORS origins) always require restart.

---

## Failure Mode Table

| Failure | Detection | Router Action | Client Response |
|---|---|---|---|
| Backend crash before request | `poll()` returns None | State → `unhealthy`; excluded from routing | 503 `backend_unavailable` |
| Backend crash mid-request | Broken connection | Close stream; state → `unhealthy` | 502 `stream_error` |
| Startup timeout | `startup_wait` expires | State → `unhealthy`; log stderr tail | 504 `backend_start_timeout` |
| Semaphore full | Queue timeout after `queue_timeout_sec` | Queues, then rejects | 503 (OpenAI/Gemini) or 529 (Anthropic) |
| Slots full (llama-server 503) | Backend returns 503 | No retry | 503 `backend_unavailable` |
| Context overflow | Not yet implemented | Forwarded to backend as-is | Backend 4xx passed through |
| OOM / process exits with signal | Poll fails | State → `unhealthy` | 502 `backend_error` |
| Bad config on startup | Validation | Abort; print error | N/A |
| Bad config on `/reload-config` | Validation | Keep current config | 400 (admin endpoint) |
| Partial stream failure | Broken pipe after first chunk | Send `data: [DONE]`; log error | Stream truncated |
| Stale/missing bench data | Age/presence check on load | Engine-rank ordering; no alert | No client impact |
| No backend in classified tier | Routing step 4c | Falls back to first registered backend (known limitation) | Depends on fallback backend |

---

## Phase 1 Acceptance Criteria

All must pass on a fresh DGX Spark install.

| Test | Pass Condition |
|---|---|
| `GET /v1/models` | JSON with at least one `id` matching a registered llama.cpp backend |
| Non-streaming chat | `choices[0].message.content` is a string; response within `proxy.timeout_sec` |
| SSE streaming | Valid SSE chunks; terminates with `data: [DONE]` |
| Tool call | `choices[0].message.tool_calls` in OpenAI schema |
| `response_format: json_schema` | Response is valid JSON matching schema |
| `developer` role message | **Deferred** — not yet implemented |
| Lazy start | Backend not running; request starts it; response succeeds within `startup_wait + proxy.timeout_sec` |
| Startup timeout | Backend never becomes healthy; client receives 504 within `startup_wait + 5s` |
| Context overflow | **Deferred** — no preflight check yet; backend 4xx is passed through |
| Semaphore saturation | Requests beyond `max_concurrent_requests` queue for `queue_timeout_sec`, then receive 503/529 |
| OpenClaw end-to-end | `baseURL=http://localhost:9001/v1`; chat + tool call works |
| OpenCode end-to-end | `baseURL=http://localhost:9001/v1`; chat works |
| `/reload-config` | Routing threshold change takes effect without restart |
| `/rescan` | New GGUF in scan dir appears in `/v1/models` after rescan |

---

## Architecture Boundaries (Non-Goals)

- Not a production API gateway. No TLS, no edge auth, no enforced rate limiting.
- Not an HA system. Single process, single node, no replication.
- Not a cloud provider router. Remote backends are never auto-selected.
- Not a multi-tenant system. Single operator, trusted local clients.
- Full dynamic reconfiguration of middleware and backend subprocess flags is not supported.

---

## Appendix: Adding a Future Engine

Minimum steps to add a new engine:

1. Add engine name to the known engine list in `engines.py`.
2. Implement `build_<engine>_cmd()` returning a process invocation list.
3. Define `Capabilities` defaults for the engine.
4. Implement `GET /health` probe or equivalent.
5. Add the engine to `OPERATIONS.md` under "Additional Engines".
6. Gate behind `engines_enabled`. Default remains `["llama.cpp"]` until the engine passes its own acceptance criteria.

An engine is not supported until it has its own acceptance matrix and passes it.

## Appendix: Runtime Wiring

For orientation — not the contract.

- `router/main.py`: FastAPI app factory, lifespan, middleware, route registration
- `router/config.py`: typed config loading, path expansion, file resolution
- `router/registry.py`: merges manual + discovered backends + overrides
- `router/discovery.py`: scans GGUF/HF/TRT sources (gated by `engines_enabled`)
- `router/lifecycle.py`: process lifecycle, health probes, idle eviction, registry snapshotting
- `router/proxy.py`: request forwarding, semaphore, streaming, metrics recording
- `router/routing.py`: tier classification, `_pick()`, bench cache
- `router/benchmark.py`: PP/TG measurement, cache persistence
- `router/sysinfo.py`: platform, GPU, CUDA, engine version diagnostics
- `router/anthropic_compat.py` / `router/gemini_compat.py`: format translation layers (phase 2+)

Config resolution order:

1. Explicit path passed to `load_config()`
2. `LLM_ROUTER_CONFIG` env var
3. `config/settings.yaml` beside the project root
4. `~/.llm-router/settings.yaml`

`overrides.json` (`~/.llm-router/overrides.json`) is applied after registry merge and can exclude or patch individual backends.
