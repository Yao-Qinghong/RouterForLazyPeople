# ARCHITECTURE.md Doc Tightening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten ARCHITECTURE.md (and reconcile API_SPEC.md + README.md) so the doc set is a reliable implementation contract for phase-1 llama.cpp on DGX Spark.

**Architecture:** Pure documentation changes across three files. No code changes. Each task rewrites or adds a specific section of ARCHITECTURE.md, then a final task reconciles scope across API_SPEC.md and README.md. The doc is the deliverable.

**Tech Stack:** Markdown only. Verification is grep/read against the codebase to confirm accuracy.

---

### Task 1: Add backend handoff and drain policy

The review's top finding: the doc says "one active backend at a time" but never defines what happens when routing picks a different model while one is already running. The code already has `_evict_for_vram()` in `lifecycle.py:453` which evicts idle backends oldest-first when VRAM is needed, and `ensure_running()` at `lifecycle.py:502` calls it before starting a new backend. This must be documented.

**Files:**
- Modify: `docs/ARCHITECTURE.md:36-48` (Assumptions + Constraints)
- Modify: `docs/ARCHITECTURE.md:111-141` (Backend Lifecycle State Machine)

- [ ] **Step 1: Read the current Assumptions and Constraints sections**

Verify lines 33-48 of ARCHITECTURE.md.

- [ ] **Step 2: Rewrite the single-backend assumption with handoff semantics**

Replace the vague "One backend active at a time is the designed operating point" bullet in Constraints (line 47) and the matching Assumptions bullet (line 36) with concrete handoff behavior:

```markdown
## Assumptions

- **Trusted LAN only.** Auth is optional. No edge security.
- **Single node, no HA.** One router process. The designed operating point on DGX Spark is one active backend at a time, but the router does not enforce a hard limit — multiple backends can be running concurrently if VRAM permits.
- **No automatic cloud fallback.** The router never selects a remote provider automatically. Remote/external backends require explicit `?backend=` or `[route:key]` routing.
- **DGX Spark / CUDA Linux is the primary target.** macOS with Metal is a development convenience.
- **Operator-managed hardware.** The router does not manage GPU allocation across processes.
```

And in Constraints, replace the "One backend active" bullet with:

```markdown
- **Backend handoff (VRAM eviction):** When `ensure_running(key)` is called and the new backend has a `vram_estimate_gb`, the router calls `_evict_for_vram()` which stops idle backends oldest-first (by `last_used` timestamp) until enough VRAM is free. Backends with active in-flight requests (`active_requests > 0`) are never evicted. If eviction cannot free enough VRAM, the request fails with 503. When `vram_estimate_gb` is not set (e.g. auto-discovered models without size data), no eviction occurs and concurrent starts are allowed — VRAM management is the operator's responsibility.
- **No drain on handoff.** There is no request-drain period. Eviction calls `stop(key)` immediately (SIGTERM → SIGKILL after 10s). In-flight requests to the evicted backend will fail with a broken connection. This is acceptable for the single-operator DGX Spark use case but would need a drain period for shared use.
```

- [ ] **Step 3: Add VRAM eviction to the lifecycle state machine section**

After the "Unhealthy penalty" paragraph (line 139), add:

```markdown
**VRAM eviction:** `_evict_for_vram(needed_gb)` is called inside `ensure_running()` before spawning a new backend. It stops idle backends (no active requests, oldest first) until `query_free_vram()` reports enough free VRAM. Backends with `active_requests > 0` are skipped. If GPU info is unavailable (`query_free_vram()` returns None), eviction is skipped entirely.
```

- [ ] **Step 4: Verify against code**

Run: `grep -n "evict_for_vram\|active_requests\|SIGTERM\|SIGKILL\|10s\|vram_estimate" router/lifecycle.py | head -20`

Confirm: `_evict_for_vram` exists, `active_requests` check exists, SIGTERM/SIGKILL timeout is 10s, `vram_estimate_gb` is checked.

- [ ] **Step 5: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: add backend handoff and VRAM eviction policy to ARCHITECTURE.md"
```

---

### Task 2: Make routing fail-closed on capability misses

The review says capability filtering is fail-open (filter skipped if nothing qualifies) and the no-tier fallback is "first registered backend." Both of these are true in the code and must be documented honestly with their risk.

**Files:**
- Modify: `docs/ARCHITECTURE.md:145-160` (Routing Algorithm section)

- [ ] **Step 1: Read the current Routing Algorithm section**

Verify lines 142-177 of ARCHITECTURE.md.

- [ ] **Step 2: Rewrite routing steps 4b-4d and capability filter with fail-open warnings**

Replace lines 152-160 with:

```markdown
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
```

- [ ] **Step 3: Remove the old "No automatic cloud fallback" paragraph**

The existing paragraph at line 160 is already accurate but can stay as-is — it doesn't conflict.

- [ ] **Step 4: Verify against code**

Run: `grep -n "filter is skipped\|all tier backends\|capable\|tier_backends" router/routing.py`

Confirm: capability filter returns `tier_backends` unchanged when `capable` list is empty (lines 248-249 and 253-254 of routing.py).

- [ ] **Step 5: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: document fail-open routing and capability filter risks"
```

---

### Task 3: Define stable model ID generation rules

The review says stable model identity is required but ID generation for discovered models is not defined. The code uses `_slug()` in `discovery.py:94` which lowercases, replaces non-alphanumeric chars with hyphens, and truncates + MD5-suffixes at 40 chars.

**Files:**
- Modify: `docs/ARCHITECTURE.md:52-89` (Core Domain Objects section — add after BackendCapabilities)

- [ ] **Step 1: Read `_slug()` and key generation in discovery.py**

Verify `router/discovery.py:94-98` for the slug function, and lines 309, 408, 531, 572, 661 for how each source generates keys.

- [ ] **Step 2: Add a "Model Identity" subsection after BackendCapabilities**

Insert after line 89 (after the BackendCapabilities section, before Error Responses):

```markdown
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
```

- [ ] **Step 3: Verify against code**

Run: `grep -n "_slug\|slug =" router/discovery.py | head -15`

Confirm: slug generation matches the documented rules.

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: define model identity and slug generation rules"
```

---

### Task 4: Add an Open Questions section

The review says known limitations are buried in normative text instead of tracked as decisions. Collect them into one section.

**Files:**
- Modify: `docs/ARCHITECTURE.md` (add new section before Appendices, after Architecture Boundaries)

- [ ] **Step 1: Add the Open Questions section**

Insert before "## Appendix: Adding a Future Engine":

```markdown
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
```

- [ ] **Step 2: Verify each item references real code behavior**

Spot-check 3-4 items against the codebase:
- Item 1: `routing.py:239-240` — `return list(backends.keys())[:limit]` (first registered)
- Item 5: `engines.py:216-232` — no `--parallel` in `build_llama_cmd()`
- Item 6: `proxy.py:200` — `proxy.timeout_sec` covers TTFT only for streaming
- Item 7: no `on_disconnect` or cancellation logic in `proxy.py`

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: add Open Questions section tracking known limitations"
```

---

### Task 5: Expand config schema for backends.yaml

The review says there is no full config schema for `backends.yaml`. The code defines `BackendConfig` with ~25 fields in `config.py:167-198`. Document the operator-facing ones.

**Files:**
- Modify: `docs/ARCHITECTURE.md:275-294` (Config Schema section)

- [ ] **Step 1: Read BackendConfig fields**

Verify `router/config.py:167-198` for the full field list.

- [ ] **Step 2: Add a backends.yaml schema table after the settings.yaml table**

Insert after the `startup_wait` note (line 294) and before "### Startup Validation":

```markdown
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
```

- [ ] **Step 3: Verify field names and defaults against code**

Run: `grep -n "field\|: " router/config.py | sed -n '167,198p'`

Confirm: field names and defaults match `BackendConfig` dataclass.

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: add full backends.yaml config schema"
```

---

### Task 6: Tighten llama.cpp day-2 controls documentation

The review identifies missing: slot policy ownership, stream-idle timeout, disconnect cancellation, repeated-failure backoff. These don't exist in the code, so document them as "not implemented" with the risk, not as aspirational specs.

**Files:**
- Modify: `docs/ARCHITECTURE.md:180-234` (llama.cpp Operational Contract)

- [ ] **Step 1: Read the current llama.cpp section**

Verify lines 180-234 of ARCHITECTURE.md.

- [ ] **Step 2: Add day-2 gaps subsection after llama-server Error Mapping**

Insert after the error mapping table (after line 234), before the `---`:

```markdown
### Known Gaps (day-2 controls not yet implemented)

These are behaviors the llama.cpp contract does not yet specify or enforce. They are tracked as open questions (#5–#8 in the Open Questions section).

- **Slot alignment:** The router semaphore (`max_concurrent_requests`, default 20) and llama-server slot count (`--parallel`, default 1) are independent. With default settings, the semaphore admits 20 concurrent requests but llama-server can only serve 1 at a time — the rest queue inside llama-server and may time out. Operators should either set `extra_args: ["--parallel", "20"]` in `backends.yaml` or reduce `max_concurrent_requests` to 1.
- **Stream-idle timeout:** After the first SSE token arrives, there is no per-chunk timeout. A hung backend that stops emitting tokens will hold the connection and semaphore slot until `proxy.timeout_sec` expires (default 300s) or the client disconnects.
- **Client disconnect:** When a client closes the connection mid-stream, the proxy stops forwarding but does not cancel the backend request. llama-server continues generating tokens until completion, wasting GPU compute.
- **Repeated-failure backoff:** If a backend fails to start repeatedly (e.g. missing model file, OOM), each incoming request retries `ensure_running()` from scratch with no backoff or circuit-breaker. Under load this produces a retry storm.
- **Health probe specificity:** The readiness probe accepts any HTTP 200 from `/health`. It does not verify the response body (`{"status": "ok"}`) or that the correct model is loaded. A healthy but wrong-model llama-server would pass the probe.
```

- [ ] **Step 3: Verify each gap against code**

- Slot alignment: `engines.py:216-232` has no `--parallel`; `config.py:86` has `max_concurrent_requests: int = 20`
- Stream-idle: `proxy.py:200` — timeout covers TTFT only
- Client disconnect: no `on_disconnect` handler in proxy.py
- Repeated-failure: `lifecycle.py:502-545` — no backoff logic
- Health probe: `lifecycle.py:124-126` — `if r.status_code == 200: return True`

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: document llama.cpp day-2 control gaps"
```

---

### Task 7: Tighten OpenClaw/OpenCode compatibility

The review says acceptance criteria are too weak. Add concrete parameter support and gap documentation.

**Files:**
- Modify: `docs/ARCHITECTURE.md:237-272` (OpenClaw / OpenCode Compatibility section)

- [ ] **Step 1: Read the current compatibility section**

Verify lines 237-272 of ARCHITECTURE.md.

- [ ] **Step 2: Add supported parameter matrix and gap documentation**

After the "Auth header accepted" row in the Required Endpoint Behavior table (line 250), add:

```markdown
### Supported OpenAI Parameters

Parameters the router passes through to the backend. The router does not validate or transform these — support depends on the backend engine.

| Parameter | Passed through | Notes |
|---|---|---|
| `model` | Yes | Resolved to backend key, then rewritten to backend's model name |
| `messages` | Yes | Array of `{role, content}` objects |
| `stream` | Yes | Controls SSE vs buffered response |
| `temperature` | Yes | |
| `top_p` | Yes | |
| `max_tokens` | Yes | |
| `stop` | Yes | |
| `tools` / `functions` | Yes | Used by routing classifier to select `deep` tier |
| `tool_choice` | Yes | Passed through; no router-level validation |
| `response_format` | Yes | `json_schema` type used by routing classifier |
| `n` | Yes | Multiple completions; backend support varies |
| `presence_penalty` | Yes | |
| `frequency_penalty` | Yes | |
| `logprobs` | Yes | |
| `seed` | Yes | |
| `user` | Yes | |

**Not supported / not validated:**
- `parallel_tool_calls`: passed through but not validated. Backend may ignore it.
- `usage` in responses: present only if the backend includes it. The router does not synthesize usage data.
- `finish_reason`: forwarded as-is from the backend. The router does not validate or normalize it.
- `tool_choice: "required"` or `tool_choice: {"type": "function", "function": {"name": "..."}}`: passed through. Whether the backend honors it depends on the model and engine.

### Compatibility Gaps

- **Tool-call streaming:** Tool calls in SSE streams follow whatever format the backend emits. The router does not validate that streamed tool calls accumulate into valid OpenAI schema.
- **`developer` role:** Not rewritten to `system`. Agentic clients (OpenClaw) that send `developer` role messages may get unexpected behavior from backends that don't recognize it.
- **Long-prompt handling:** No preflight rejection. Oversized prompts are forwarded and may produce backend-specific errors rather than a clean 400.
- **Stable `usage` field:** Not guaranteed. Some backends omit `prompt_tokens` or `completion_tokens`. Clients that rely on `usage` for billing or context tracking should not depend on it.
```

- [ ] **Step 3: Verify parameter passthrough**

Run: `grep -n "payload\|json=\|forward" router/proxy.py | head -10`

Confirm: the proxy forwards the full payload dict to the backend without filtering parameters.

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: tighten OpenClaw/OpenCode compatibility with parameter matrix"
```

---

### Task 8: Reconcile scope across API_SPEC.md and README.md

The review's third top finding: ARCHITECTURE.md says phase 1 is only `/v1/chat/completions` + `/v1/models` on llama.cpp, while API_SPEC.md and README.md advertise Anthropic, Gemini, websocket, embeddings, catch-all inference as current surface. Add phase labels to the other docs.

**Files:**
- Modify: `docs/API_SPEC.md:1-10` (Scope section)
- Modify: `docs/API_SPEC.md:68-97` (Public Endpoints table)
- Modify: `README.md:24-36` (Features section)

- [ ] **Step 1: Read API_SPEC.md scope and endpoints**

Verify lines 1-97 of API_SPEC.md.

- [ ] **Step 2: Add phase labels to API_SPEC.md scope**

Replace lines 1-13 of API_SPEC.md with:

```markdown
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
```

- [ ] **Step 3: Add phase column to API_SPEC.md endpoints table**

Replace the Public Endpoints table (lines 70-97) with a version that includes a Phase column:

```markdown
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
```

- [ ] **Step 4: Add phase labels to README.md features**

Replace lines 24-36 of README.md (the Features section) with:

```markdown
## Features

**Phase 1 (tested, supported):**
- **Lazy loading** — models start on first request, stop automatically after idle
- **Smart routing** — keyword + token-count classifier picks fast / mid / deep tier
- **OpenAI-compatible** — drop-in for any OpenAI SDK client, Open WebUI, OpenClaw, Cursor, Continue, Jan
- **Auto-discovery** — scans your model directories and registers GGUF models automatically
- **Benchmarking** — tracks TTFT, latency, tokens/sec per backend; export as CSV
- **System diagnostics** — detects GPU, CUDA, CPU architecture, engine versions, install recommendations
- **Beginner-friendly** — all config in two YAML files, no Python knowledge required to operate

**Phase 2+ (code exists, untested — use at your own risk):**
- **Multi-engine** — vLLM, SGLang, TensorRT-LLM, HuggingFace TGI available via `engines_enabled`
- **Anthropic-compatible** — drop-in for Anthropic SDK, Claude Code
- **Gemini-compatible** — supports Google Gemini `generateContent` and `streamGenerateContent` payloads
- **HF / TRT-LLM auto-discovery** — scans HuggingFace and TRT-LLM model directories (gated by `engines_enabled`)
- **Auto-update** — one command updates llama.cpp and Python deps
```

- [ ] **Step 5: Verify the phase-1 endpoint list matches ARCHITECTURE.md**

Cross-check: ARCHITECTURE.md phase-1 scope (lines 12-18) lists `/v1/chat/completions` and `/v1/models`. The API_SPEC phase-1 column should include those plus admin/status endpoints but NOT Anthropic, Gemini, embeddings, WS, or catch-all.

- [ ] **Step 6: Commit**

```bash
git add docs/ARCHITECTURE.md docs/API_SPEC.md README.md
git commit -m "doc: reconcile phase-1 scope across ARCHITECTURE, API_SPEC, and README"
```

---

### Task 9: Clean up the engine appendix

The review says the future-engine section leaks implementation details (sentinel classes) and reduces extensibility to "add engine name." Tighten it to match the actual code steps.

**Files:**
- Modify: `docs/ARCHITECTURE.md:374-385` (Appendix: Adding a Future Engine)

- [ ] **Step 1: Rewrite the engine appendix**

Replace lines 374-385 with:

```markdown
## Appendix: Adding a Future Engine

Minimum steps to add a new engine (based on existing engine implementations):

1. **`router/engines.py`:** Add the engine constant (e.g. `ENGINE_NEWENGINE = "newengine"`), add it to `ALL_ENGINES`, implement `is_engine_available()` check, and implement `build_<engine>_cmd(cfg, config)` returning the subprocess command list.
2. **`router/lifecycle.py`:** Add an `elif engine == "newengine"` branch in `_start_process()` (or a dedicated `_start_<engine>_backend()` method if the engine needs special lifecycle handling like Docker or Ollama).
3. **`router/config.py`:** Add default `BackendCapabilities` for the engine in `_infer_capabilities()`.
4. **`router/engines.py`:** Define the health endpoint URL pattern (most engines use `/health` or `/v1/models`; add to `health_url()` if non-standard).
5. **`router/sysinfo.py`:** Add version detection in `_detect_engine_versions()`.
6. **Gate behind `engines_enabled`.** The engine must not activate unless explicitly listed. Default remains `["llama.cpp"]`.
7. **Write an acceptance matrix** for the new engine (similar to the phase-1 acceptance criteria). The engine is not supported until its matrix passes.

An engine is not supported until it has its own acceptance matrix and passes it.
```

- [ ] **Step 2: Verify the steps match actual engine implementations**

Run: `grep -n "ENGINE_\|build_.*_cmd\|health_url\|_detect_engine" router/engines.py | head -20`

Confirm: the pattern matches existing engines (llama.cpp, vllm, sglang, etc.).

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "doc: tighten engine appendix to match actual extension pattern"
```

---

### Task 10: Final review pass

Read the full updated ARCHITECTURE.md and cross-check for internal consistency.

**Files:**
- Read: `docs/ARCHITECTURE.md` (full file)
- Read: `docs/API_SPEC.md` (full file)
- Read: `README.md` (lines 24-36)

- [ ] **Step 1: Read all three files end-to-end**

- [ ] **Step 2: Check for internal contradictions**

Grep for terms that should be consistent:
- `max_concurrent_requests` default should be `20` everywhere
- `startup_wait` default should be `30` everywhere
- `proxy.timeout_sec` should be the field name everywhere (not `request_timeout`)
- Phase-1 scope should be consistent across all three docs
- No remaining references to `Capabilities` (old name) — should be `BackendCapabilities`
- No remaining references to `n_parallel` being set by the router
- No remaining references to `local=True` / `local=False`

Run: `grep -n "request_timeout\|default.*10[^0-9]\|Capabilities[^A-Z]\|n_parallel\|local=True\|local=False" docs/ARCHITECTURE.md docs/API_SPEC.md README.md`

- [ ] **Step 3: Fix any inconsistencies found**

If the grep finds stale references, fix them inline.

- [ ] **Step 4: Final commit**

```bash
git add docs/ARCHITECTURE.md docs/API_SPEC.md README.md
git commit -m "doc: final consistency pass across architecture docs"
```
