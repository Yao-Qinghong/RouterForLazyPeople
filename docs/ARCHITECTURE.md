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
- **Single node, no HA.** One router process. The designed operating point on DGX Spark is one active backend at a time, but the router does not enforce a hard limit — multiple backends can be running concurrently if VRAM permits.
- **No automatic cloud fallback.** The router never selects a remote provider automatically. Remote/external backends require explicit `?backend=` or `[route:key]` routing.
- **DGX Spark / CUDA Linux is the primary target.** macOS with Metal is a development convenience.
- **Operator-managed hardware.** The router does not manage GPU allocation across processes.

---

## Constraints

- Concurrent in-flight requests are capped by a semaphore (`proxy.max_concurrent_requests`, default: 20). When full the request queues for up to `proxy.queue_timeout_sec` (default: 30s), then returns 503 on OpenAI/Gemini routes or 529 on Anthropic routes.
- Backend start budget: `startup_wait` seconds from subprocess spawn to `/health` passing (default: 30s, per-backend field). This is separate from per-request inference timeout.
- **Backend handoff (VRAM eviction):** When `ensure_running(key)` is called and the new backend has a `vram_estimate_gb`, the router calls `_evict_for_vram()` which stops idle backends oldest-first (by `last_used` timestamp) until enough VRAM is free. Backends with active in-flight requests (`active_requests > 0`) are never evicted. If eviction cannot free enough VRAM, the request fails with 503. When `vram_estimate_gb` is not set (e.g. auto-discovered models without size data), no eviction occurs and concurrent starts are allowed — VRAM management is the operator's responsibility.
- **No drain on handoff.** There is no request-drain period. Eviction calls `stop(key)` immediately (SIGTERM → SIGKILL after 10s). In-flight requests to the evicted backend will fail with a broken connection. This is acceptable for the single-operator DGX Spark use case but would need a drain period for shared use.
- Stale or missing benchmark data silently falls back to engine-rank ordering. It does not block routing.

---

## Core Domain Objects

### BackendConfig

The actual runtime type is `BackendConfig` in `router/config.py`. Key fields:

```python
@dataclass
class BackendConfig:
    engine: str = "llama.cpp"
    port: int = 8080
    model: str = ""              # absolute path for GGUF / HF checkpoint
    model_dir: str = ""          # directory for multi-file models
    tier: str = ""               # "fast" | "mid" | "deep" (auto-assigned if empty)
    ctx_size: int = 32768
    idle_timeout: int = 300      # seconds before idle eviction; 0 = never
    startup_wait: int = 30       # seconds to wait for /health on start
    auto_discovered: bool = False
    capabilities: BackendCapabilities = ...
```

External backends (LM Studio, Ollama, running servers) use sentinel process objects (`_ExternalSentinel`, `_OllamaSentinel`) instead of `Popen`; they are not distinguished by a boolean field.

### BackendCapabilities

```python
@dataclass
class BackendCapabilities:
    supports_tools: bool = False
    supports_json_schema: bool = False
    max_context: int = 32768
    code_quality: str = "good"   # "weak" | "good" | "strong"
```

Capabilities are inferred at registration time by `_infer_capabilities(engine, size_gb, name)`:
- Large models (>=25 GB or 32B+ names): `supports_tools=True, supports_json_schema=True, code_quality="strong"`
- Medium models (>=8 GB or 13B+ names): `supports_tools=True, supports_json_schema=True, code_quality="good"`
- Small models: `supports_tools=False, supports_json_schema=False, code_quality="weak"`

### Model Identity

Backend keys serve as model IDs in `/v1/models` and in response `model` fields. They must be stable across restarts for OpenClaw and OpenCode compatibility.

**Key generation rules:**

| Source | Key format | Example |
|---|---|---|
| `backends.yaml` (manual) | The YAML key verbatim | `fast`, `deep`, `my-llama` |
| GGUF auto-discovery | `_slug(filename, path)` | `qwen3-5-35b-a3b-ud-q4-k-xl` |
| HuggingFace auto-discovery | `_slug("hf-" + dirname, path)` | `hf-meta-llama-3-1-8b` |
| TRT-LLM auto-discovery | `_slug("trt-" + dirname, path)` | `trt-nemotron-nano` |
| Ollama | `"ollama-" + slugified_name` | `ollama-llama3` |
| LM Studio (external) | `"lmstudio"` | `lmstudio` |
| Probed ports (external) | `"local-" + port` | `local-8080` |

**`_slug(name, path)`** (`discovery.py:94`): lowercase, replace `[^a-z0-9]` with `-`, strip leading/trailing `-`. If longer than 40 chars, truncate to 35 and append `-` + first 4 hex of MD5(path).

**Alias exposure:** Configured `model_aliases` (e.g. `gpt-4: deep`) are listed as additional entries in `/v1/models` alongside their target backend key. Both the alias and the canonical key are valid `model` values in requests.

**Collision rule:** Manual backends (`backends.yaml`) win on key collision with auto-discovered backends. Among auto-discovered sources, the merge order is `{**gguf, **hf, **trt}`, so on slug collision TRT-LLM > HF > GGUF takes the winning config.

**Stability guarantee:** For a given model file at a given path, the slug is deterministic. Moving or renaming the file changes the key. Manual `backends.yaml` keys are fully operator-controlled and always stable.

### Error Responses

Error shapes vary by route. There is no single normalized envelope today.

**OpenAI routes** (`/v1/chat/completions`): flat `{"error": "<message>", ...}` for router-level errors (overload, no backend). Backend 4xx/5xx on non-streaming requests are forwarded as-is. On streaming requests, backend errors are emitted as SSE error events (`data: {"error": {...}}`) inside a 200 response.

**Anthropic routes** (`/anthropic/v1/messages`, `/v1/messages`): Anthropic-style `{"type": "error", "error": {"type": "...", "message": "..."}}` envelope. Overload returns 529 with `overloaded_error`.

**Gemini routes** (`/gemini/...`): `{"error": {"message": "..."}}` envelope. Overload returns 503.

| Condition | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| No backend available | 503 | 503 | 400 |
| Backend start timeout | 504 | 504 | 504 |
| Semaphore queue timeout | 503 | 529 | 503 |
| Backend 4xx (non-streaming) | forward as-is | forward as-is | forward as-is |
| Backend error (streaming) | SSE error event in 200 | SSE error event in 200 | SSE error event in 200 |

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
    ├── mark_unhealthy() called (60s penalty) ─────► unhealthy (deprioritized)
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

**Unhealthy penalty:** `mark_unhealthy(key)` imposes a transient 60-second penalty. Unhealthy backends are not excluded — they are sorted to the end of the candidate list and only tried as a last resort. The penalty expires automatically after 60 seconds; no explicit `restart()` is required.

**Crash mid-request:** If the backend process exits while a request is in flight, the proxy detects the broken connection, returns 502 `stream_error`, and transitions the backend to `unhealthy`. The request is not retried.

**VRAM eviction:** `_evict_for_vram(needed_gb)` is called inside `ensure_running()` before spawning a new backend. It stops idle backends (no active requests, oldest first) until `query_free_vram()` reports enough free VRAM. Backends with `active_requests > 0` are skipped. If GPU info is unavailable (`query_free_vram()` returns None), eviction is skipped entirely.

---

## Routing Algorithm

Selection runs in strict priority order. The first matching rule wins.

1. `?backend=<key>` — resolves to the named backend. The backend is lazily started via `ensure_running()` if not already running. Returns 503 if start fails.
2. `model_aliases` entry — alias resolves to a backend key, then treated as rule 1.
3. `[route:key]` prefix in first message content — same rules as rule 1.
4. Automatic classification:
    - a. Classify tier from payload (see Tier Classification below).
    - b. Select candidates: backends matching the classified tier, sorted by benchmark score then engine rank. Unhealthy backends are pushed to the end (not excluded). Capability filtering applies when multiple candidates exist.
    - c. **Capability filter is fail-open:** If the payload requires tools or JSON schema and no backend in the tier declares support, the filter is skipped and all tier backends remain candidates. The request may route to an incapable backend and fail at the backend level. This is a known limitation.
    - d. **No-tier fallback is fail-open:** If no backend exists in the classified tier, the code falls back to the first registered backend regardless of tier, capability, or health. This is a known limitation — a future version should return 503 instead.
    - e. For each candidate in order: call `ensure_running()` (lazy start). The first backend that starts successfully handles the request.

**Capability filter detail (`select_candidates()` in `routing.py:221`):** When multiple candidates exist in the tier, payloads with `tools` narrow to backends where `capabilities.supports_tools == True`, and payloads with `response_format.type == "json_schema"` narrow to `capabilities.supports_json_schema == True`. If the narrowed list is empty, the filter result is discarded and all tier backends remain candidates. This is a preference, not a hard gate.

**Known fail-open risks:**
- A tool-calling request can route to a small model that does not support tools. The backend will likely return malformed output, not a clean error.
- A JSON-schema request can route to a model without grammar/schema support. The backend may return unstructured text.
- When no tier match exists, the fallback ignores tier, capability, and health entirely.

These are accepted phase-1 limitations. Operators should ensure at least one capable backend exists per tier they expect to use, or use explicit `?backend=` routing for tool/schema workloads.

**No automatic cloud fallback:** The router does not distinguish local vs remote backends at the routing level — there is no `local` field. Remote/external backends (e.g. OpenAI API keys) require explicit `?backend=` or `[route:key]` routing by design assumption, not by a code-enforced filter.

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

- Command: `llama-server --host 0.0.0.0 --port <port> --model <path> --ctx-size <ctx_size>` plus optional flags (`--flash-attn`, `--reasoning`) detected from the binary. `--parallel` is **not** currently set by the router.
- Readiness probe: `GET http://<host>:<port>/health` polled every 1s.
  - Pass: any HTTP 200 response (body content is not checked).
  - No consecutive-failure counter exists today — a single failed probe does not transition state.
- If `startup_wait` expires without readiness: log stderr tail, set state `unhealthy`, close log handle.
- No pre-warming prompt is sent. Backend is ready as soon as `/health` passes.

### Timeouts (these are separate)

| Timeout | What it covers | Default | Configurable |
|---|---|---|---|
| `startup_wait` | Spawn to first `/health` pass | 30s | Per-backend (`backends.yaml`) |
| `proxy.timeout_sec` | Request forwarded to final byte | 300s | `settings.yaml` |

For SSE streaming: `proxy.timeout_sec` covers time to first token. After first token arrives, chunks are forwarded without an additional per-chunk timeout.

### Concurrency and Slot Policy

- The router does **not** set `--parallel` on llama-server. Parallel slot count is llama-server's default (1) unless the operator adds it via `extra_args` in `backends.yaml`.
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

### backends.yaml

Each key under `backends:` defines a backend. Manual backends take priority over auto-discovered ones on key collision.

| Field | Type | Default | Required | Notes |
|---|---|---|---|---|
| `engine` | str | `"llama.cpp"` | No | Must be in `ALL_ENGINES` |
| `port` | int | `8080` | No | Must be unique across all backends |
| `model` | str | `""` | Yes* | Path to GGUF file or HF checkpoint. *Required for `llama.cpp`, `vllm`, `sglang`, `huggingface` |
| `model_dir` | str | `""` | No | Alternative to `model` for directory-based models |
| `tier` | str | `""` | No | `"fast"` / `"mid"` / `"deep"`. Auto-assigned from file size if empty |
| `ctx_size` | int | `32768` | No | Context window passed to engine as `--ctx-size` |
| `gpu_layers` | int | `999` | No | GPU layers for llama.cpp (`--n-gpu-layers`) |
| `flash_attn` | bool | `True` | No | Enable flash attention (llama.cpp, if binary supports it) |
| `reasoning` | bool | `False` | No | Enable reasoning tag parsing (llama.cpp `--reasoning`) |
| `reasoning_budget` | int | `null` | No | Reasoning token budget (llama.cpp `--reasoning-budget`) |
| `idle_timeout` | int | `300` | No | Seconds before idle eviction. `0` = never evict |
| `startup_wait` | int | `30` | No | Seconds to wait for `/health` pass after spawn |
| `extra_args` | list | `[]` | No | Additional CLI args passed to the engine command |
| `description` | str | `""` | No | Human-readable label shown in `/status` |
| `capabilities` | object | inferred | No | Override `supports_tools`, `supports_json_schema`, `max_context`, `code_quality` |
| `size_gb` | float | `null` | No | Model size in GB. Used for tier assignment and VRAM estimation |
| `vram_estimate_gb` | float | `null` | No | Estimated VRAM needed. Triggers eviction logic in `ensure_running()` |
| `dtype` | str | `"auto"` | No | Data type for vLLM/SGLang/HF engines |
| `tensor_parallel_size` | int | `1` | No | Tensor parallelism for vLLM |
| `quantization` | str | `null` | No | Quantization method for vLLM |
| `tokenizer` | str | `""` | No | Custom tokenizer path |

Fields not listed here (`gpu_memory_fraction`, `trust_remote_code`, `enforce_eager`, `enable_prefix_caching`, `wrapper_script`, `model_type`, `trt_config`, `docker_config`) are engine-specific and rarely needed for phase-1 llama.cpp use.

### Startup Validation

Abort (`ConfigError`):
- `backends.yaml` references an unknown engine (not in `ALL_ENGINES`).
- `backends.yaml` backend for a local engine (`llama.cpp`, `vllm`, `sglang`, `huggingface`) has no `model` or `model_dir` field at all.
- Two backends in `backends.yaml` share the same port.

Warn only (does not abort):
- `backends.yaml` backend has a `model_path` that does not exist on disk (logged as warning, backend still registered).

Not validated:
- `engines_enabled` values are not checked against a known engine list.
- Router port already in use (detected at bind time by uvicorn, not at config load).

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
| Backend crash before request | `poll()` returns None | State → `unhealthy`; deprioritized (sorted to end) | 503 `backend_unavailable` |
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

## Open Questions and Known Limitations

Tracked here so they are not buried in normative text. Each item is a decision that needs to be made before the behavior can be considered stable.

| # | Area | Question | Current behavior | Risk |
|---|---|---|---|---|
| 1 | Routing | Should no-tier fallback return 503 instead of routing to the first registered backend? | Falls back to first backend regardless of tier/capability/health | Silent misrouting to wrong-size or incapable model |
| 2 | Routing | Should capability filter be fail-closed (503) when no capable backend exists? | Filter is skipped; request routes to incapable backend | Malformed tool/schema output instead of clean error |
| 3 | Lifecycle | Should backend eviction drain in-flight requests before SIGTERM? | No drain; immediate SIGTERM | In-flight requests fail on backend swap |
| 4 | Lifecycle | Should `mark_unhealthy()` exclude backends entirely instead of deprioritizing? | 60s deprioritization, still tried as last resort | Unhealthy backend serves requests during penalty window |
| 5 | llama.cpp | Should the router set `--parallel` to match semaphore size? | Not set; llama-server defaults to 1 slot | Concurrent requests queue inside llama-server even when semaphore allows them |
| 6 | llama.cpp | Should there be a stream-idle timeout after first token? | No timeout after first token arrives | Hung streams hold semaphore slot and connection indefinitely |
| 7 | llama.cpp | Should client disconnect cancel the backend request? | No cancellation; backend continues generating | Wasted GPU compute on abandoned requests |
| 8 | llama.cpp | Should repeated startup failures trigger backoff or circuit-breaker? | No backoff; each request retries `ensure_running()` from scratch | Rapid retry storm on a persistently-failing backend |
| 9 | Compatibility | Should `developer` role be rewritten to `system` for phase 1? | Forwarded unchanged | Some agentic clients may depend on it |
| 10 | Compatibility | Should context overflow be rejected preflight (400) or forwarded? | Forwarded to backend | Backend may return unhelpful error or OOM |
| 11 | Config | Should `engines_enabled` unknown values be validated at startup? | Not validated | Typo in config silently disables an engine |
| 12 | Identity | Should auto-discovered model keys include a version/quant suffix for disambiguation? | Slug is derived from filename which usually includes quant | Different quants of same model at different paths get different slugs, but renaming a file changes the key |

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
