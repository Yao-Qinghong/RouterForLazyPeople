# RouterForLazyPeople V2 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the LLM router to be production-ready for OpenClaw/OpenCode as a daily local-first LLM proxy, focusing on llama.cpp + OpenAI-compatible path first.

**Architecture:** Proper capability-based routing with fail-closed semantics, explicit `400` capability errors vs `503` backend unavailability, safe rescan/config reconciliation with in-flight request draining, and proper error-envelope translation for Anthropic/Gemini adapters. (A normalized request/response path is *deferred* — see Task 7 for why it needs its own planning cycle.)

**Tech Stack:** Python 3.10+, FastAPI, httpx, pytest
**Test Framework:** pytest with pytest-asyncio (follow existing patterns)

---

## Phase 1: Critical Safety Fixes (llama.cpp + OpenAI Path Only)

### Task 1: Port Allocation Safety - Prevent Discovery from Colliding with Manual Backends

**Files:**
- Modify: `router/registry.py:63-129`
- Modify: `router/discovery.py:269-338` (GGUF), `router/discovery.py:340-467` (HF)
- Test: `tests/test_registry.py` (create new file)

**Problem:** Discovery allocates ports starting at 8100, but manual backends in `backends.yaml` already use 8101 and 8107. The merged registry is never revalidated for port conflicts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
from pathlib import Path

import yaml

import router.discovery as discovery_module
import router.registry as registry_module
from router.config import load_config
from router.registry import build_backend_registry

def test_discovery_avoids_manual_backend_ports(tmp_path, monkeypatch):
    """Discovery should skip ports already used by manual backends."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "first.gguf").write_text("fake")
    (model_dir / "second.gguf").write_text("fake")

    settings_yaml = tmp_path / "settings.yaml"
    backends_yaml = tmp_path / "backends.yaml"

    settings_yaml.write_text(yaml.safe_dump({
        "router": {"host": "0.0.0.0", "port": 9001},
        "llama_bin": str(tmp_path / "llama-server"),
        "data_dir": str(tmp_path / "data"),
        "scan_dirs": {"gguf": [str(model_dir)], "hf": [], "trtllm": []},
        "discovery": {"port_start": 8100, "port_end": 8299, "probe_ports": []},
        "tier_thresholds_gb": {"fast": 10, "mid": 40},
        "idle_timeouts_sec": {"fast": 300, "mid": 600, "deep": 900},
        "routing": {
            "token_threshold_mid": 500,
            "token_threshold_deep": 4000,
            "mid_keywords": [],
            "deep_keywords": [],
        },
        "proxy": {
            "max_concurrent_requests": 20,
            "queue_timeout_sec": 30,
            "timeout_sec": 300,
            "retry_attempts": 1,
            "retry_on_status": [502, 503, 504],
            "retry_backoff_sec": 1,
        },
        "model_aliases": {},
        "engines_enabled": ["llama.cpp"],
    }, sort_keys=False))

    backends_yaml.write_text(yaml.safe_dump({
        "backends": {
            "manual-backend": {
                "engine": "llama.cpp",
                "port": 8101,
                "model": str(tmp_path / "manual.gguf"),
                "tier": "mid",
            }
        }
    }, sort_keys=False))

    monkeypatch.setattr(discovery_module, "_file_size_gb", lambda path: 2.0)
    monkeypatch.setattr(registry_module, "detect_running_servers", lambda config: {})

    config = load_config(settings_yaml, backends_yaml)
    registry = build_backend_registry(config)

    ports = {key: cfg.get("port") for key, cfg in registry.items()}
    assert len(ports) == len(set(ports.values())), "Port collision detected"
    assert ports["manual-backend"] == 8101
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_discovery_avoids_manual_backend_ports -v`
Expected: FAIL or test doesn't exist yet

- [ ] **Step 3: Implement port tracking in registry.py**

Modify `router/registry.py:63-129`:

```python
def build_backend_registry(config: "AppConfig") -> dict:
    registry = load_backends(config)
    manual_count = len(registry)

    def _cfg_port(cfg) -> int | None:
        if hasattr(cfg, "get"):
            return cfg.get("port")
        return getattr(cfg, "port", None)

    used_ports = {
        port for cfg in registry.values()
        if (port := _cfg_port(cfg)) is not None
    }

    running = detect_running_servers(config)
    for slug, cfg in running.items():
        if slug not in registry and cfg.get("engine", "") in enabled:
            registry[slug] = cfg
            if cfg.get("port") is not None:
                used_ports.add(cfg["port"])

    port_counter = [
        _find_next_available_port(
            config.discovery.port_start,
            used_ports,
            config.discovery.port_end,
        )
    ]
```

Add helper function at top of file:

```python
def _find_next_available_port(start: int, used: set, end: int) -> int:
    port = start
    while port in used and port <= end:
        port += 1
    return port if port <= end else (end + 1)
```

- [ ] **Step 4: Update discovery functions to use port counter that skips used ports**

Modify `router/discovery.py:269` - change `discover_gguf_models` signature and implementation to skip used ports:

```python
def discover_gguf_models(
    config: "AppConfig",
    port_counter: list[int],
    used_ports: set[int] | None = None,
) -> dict:
    used_ports = used_ports or set()
    # ... inside the loop where port is assigned ...
    while port_counter[0] in used_ports and port_counter[0] <= config.discovery.port_end:
        port_counter[0] += 1
    if port_counter[0] > config.discovery.port_end:
        break
    # ... assign port and then ...
    used_ports.add(port_counter[0])
    port_counter[0] += 1
```

- [ ] **Step 5: Wire up used_ports through registry.py**

```python
# In build_backend_registry, after detecting running servers:
gguf = discover_gguf_models(config, port_counter, used_ports) if ENGINE_LLAMA in enabled else {}
hf   = discover_hf_models(config, port_counter, used_ports)   if ENGINE_HF in enabled else {}
```

- [ ] **Step 6: Validate the merged registry as a last line of defense**

```python
def _assert_unique_ports(registry: dict) -> None:
    seen: dict[int, str] = {}
    for slug, cfg in registry.items():
        port = cfg.get("port") if hasattr(cfg, "get") else getattr(cfg, "port", None)
        if port is None:
            continue
        if port in seen:
            raise ConfigError(
                f"Merged registry port collision: port {port} used by '{seen[port]}' and '{slug}'"
            )
        seen[port] = slug
```

Call `_assert_unique_ports(registry)` just before returning from `build_backend_registry()`.

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_registry.py::test_discovery_avoids_manual_backend_ports -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add router/registry.py router/discovery.py tests/test_registry.py
git commit -m "fix: prevent discovery port collision with manual backends"
```

### Task 2: Safe Rescan - Backend Config Change Detection

**Files:**
- Modify: `router/lifecycle.py:83-99` (update_registry method)
- Test: `tests/test_lifecycle.py`

**Problem:** `/rescan` can silently retarget a live backend key to new config while the old process keeps running, causing wrong-port routing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lifecycle.py
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from router.lifecycle import BackendManager
from router.config import BackendConfig

def _make_manager_config(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path,
        trtllm_docker=SimpleNamespace(
            image="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7",
            container_port=8000,
            hf_cache_dir=tmp_path / "hf-cache",
            log_dir=tmp_path / "docker-logs",
            env={},
            serve_defaults={},
        ),
    )

@pytest.mark.asyncio
async def test_update_registry_detects_runtime_config_change(tmp_path):
    """Rescan should detect when a running backend's config changes."""
    backend1 = BackendConfig(
        engine="llama.cpp",
        port=8100,
        model="/path/to/model1.gguf",
        tier="fast",
        ctx_size=32768,
        gpu_layers=999,
        flash_attn=True,
    )

    manager = BackendManager({"test-backend": backend1}, _make_manager_config(tmp_path))

    proc = MagicMock()
    proc.poll.return_value = None
    manager.processes["test-backend"] = proc

    backend2 = BackendConfig(
        engine="llama.cpp",
        port=8101,
        model="/path/to/model2.gguf",
        tier="fast",
        ctx_size=65536,
        gpu_layers=999,
        flash_attn=False,
    )

    stale = await manager.update_registry({"test-backend": backend2})

    assert stale["test-backend"]["reason"] == "config_changed"
    assert stale["test-backend"]["old_port"] == 8100
    assert stale["test-backend"]["new_port"] == 8101


@pytest.mark.asyncio
async def test_update_registry_drains_before_stop(tmp_path):
    """Rescan should wait for in-flight requests to finish before stopping."""
    backend1 = BackendConfig(
        engine="llama.cpp", port=8100, model="/m1.gguf", tier="fast",
        ctx_size=32768, gpu_layers=999, flash_attn=True,
    )
    manager = BackendManager({"test-backend": backend1}, _make_manager_config(tmp_path))
    proc = MagicMock()
    proc.poll.return_value = None
    manager.processes["test-backend"] = proc
    manager.active_requests["test-backend"] = 2  # simulate in-flight

    # Drain completes when active_requests drops to 0.
    async def _finish_requests():
        await asyncio.sleep(0.2)
        manager.active_requests["test-backend"] = 0
    asyncio.create_task(_finish_requests())

    backend2 = BackendConfig(
        engine="llama.cpp", port=8101, model="/m2.gguf", tier="fast",
        ctx_size=65536, gpu_layers=999, flash_attn=False,
    )
    with patch.object(manager, "stop") as mock_stop:
        await manager.update_registry({"test-backend": backend2})
        # stop() must only be called AFTER active_requests drained to 0
        mock_stop.assert_called_once_with("test-backend")
        assert manager.active_requests["test-backend"] == 0


@pytest.mark.asyncio
async def test_update_registry_drain_timeout_still_stops(tmp_path):
    """If drain times out, stop() is still called (best-effort semantics)."""
    backend1 = BackendConfig(
        engine="llama.cpp", port=8100, model="/m1.gguf", tier="fast",
        ctx_size=32768, gpu_layers=999, flash_attn=True,
    )
    manager = BackendManager({"test-backend": backend1}, _make_manager_config(tmp_path))
    proc = MagicMock()
    proc.poll.return_value = None
    manager.processes["test-backend"] = proc
    manager.active_requests["test-backend"] = 1  # never drops

    backend2 = BackendConfig(
        engine="llama.cpp", port=8101, model="/m2.gguf", tier="fast",
        ctx_size=65536, gpu_layers=999, flash_attn=False,
    )
    with patch.object(manager, "_drain_backend", new_callable=AsyncMock) as mock_drain:
        mock_drain.return_value = False  # simulate timeout
        with patch.object(manager, "stop") as mock_stop:
            await manager.update_registry({"test-backend": backend2})
            mock_stop.assert_called_once_with("test-backend")
```

Import additions at the top of `tests/test_lifecycle.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch
```

- [ ] **Step 2: Implement fingerprint comparison and drain in lifecycle.py**

Add helper method to `BackendManager`:

```python
import json

_RUNTIME_FINGERPRINT_FIELDS = (
    "engine", "port", "model", "model_dir", "ctx_size", "gpu_layers",
    "flash_attn", "reasoning", "reasoning_budget", "tokenizer",
    "dtype", "gpu_memory_fraction", "tensor_parallel_size", "quantization",
    "enforce_eager", "enable_prefix_caching", "trust_remote_code",
    "wrapper_script", "model_type",
)

def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def _backend_fingerprint(self, cfg) -> tuple:
    """Create a runtime fingerprint for config-change detection."""
    payload = {
        field: _cfg_get(cfg, field)
        for field in _RUNTIME_FINGERPRINT_FIELDS
    }
    payload["extra_args"] = tuple(_cfg_get(cfg, "extra_args", []) or [])
    payload["trt_config"] = json.dumps(_cfg_get(cfg, "trt_config", {}) or {}, sort_keys=True)
    payload["docker_config"] = json.dumps(_cfg_get(cfg, "docker_config", {}) or {}, sort_keys=True)
    return tuple(sorted(payload.items()))

async def _drain_backend(self, key: str, timeout: float = 10.0) -> bool:
    """Wait for in-flight requests on a backend to complete.

    Returns True if drained cleanly, False on timeout. Caller should still
    proceed with stop() on False — the goal is best-effort drain, not guarantee.
    """
    deadline = time.time() + timeout
    while self.active_requests.get(key, 0) > 0 and time.time() < deadline:
        await asyncio.sleep(0.1)
    return self.active_requests.get(key, 0) == 0

async def update_registry(self, new_backends: dict):
    if self._registry_lock is None:
        self._registry_lock = asyncio.Lock()
    async with self._registry_lock:
        stale = {}
        for key, new_cfg in new_backends.items():
            if key in self.backends and self.is_running(key):
                old_cfg = self.backends[key]
                old_fp = self._backend_fingerprint(old_cfg)
                new_fp = self._backend_fingerprint(new_cfg)
                if old_fp != new_fp:
                    old_port = _cfg_get(old_cfg, "port")
                    stale[key] = {
                        "reason": "config_changed",
                        "old_port": old_port,
                        "new_port": _cfg_get(new_cfg, "port"),
                    }
                    logger.warning(
                        f"[{key}] Config changed while running on port {old_port}. "
                        f"Draining in-flight requests before stopping."
                    )

        # Drain and stop stale backends. Drain is best-effort: if in-flight
        # requests don't finish within the timeout, we stop anyway and those
        # requests will error — but this is strictly better than yanking the
        # process out from under them with no warning.
        for key in stale:
            drained = await self._drain_backend(key, timeout=10.0)
            if not drained:
                active = self.active_requests.get(key, 0)
                logger.warning(
                    f"[{key}] Did not drain within 10s (active_requests={active}); "
                    f"stopping anyway — in-flight requests on the old config will error."
                )
            self.stop(key)

        self.backends = new_backends
        for k in new_backends:
            if k not in self.starting:
                self.starting[k] = asyncio.Lock()
            if k not in self.active_requests:
                self.active_requests[k] = 0

        return stale
```

- [ ] **Step 3: Update main.py /rescan endpoint to handle stale backends**

Modify `router/main.py:540-567`:

```python
@app.post("/rescan", summary="Re-discover models and reload config")
async def rescan(request: Request):
    cfg = request.app.state.config
    manager = request.app.state.manager
    clear_engine_cache()
    clear_llama_flag_cache()
    running_before = {k for k in manager.backends if manager.is_running(k)}
    new_backends = build_backend_registry(cfg)
    stale_backends = await manager.update_registry(new_backends)

    from router.benchmark import load_all_results
    from router.routing import set_benchmark_results
    _bench_results = load_all_results(cfg)
    set_benchmark_results(_bench_results)
    _apply_measured_tiers(manager.backends, _bench_results)
    discovered = sum(1 for v in new_backends.values() if v.get("auto_discovered"))
    return {
        "total": len(new_backends),
        "discovered": discovered,
        "engines": available_engines(cfg),
        "running": list(running_before),
        "stale_stopped": list(stale_backends.keys()),
        "backends": list(new_backends.keys()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lifecycle.py::test_update_registry_detects_runtime_config_change -v`

- [ ] **Step 5: Commit**

```bash
git add router/lifecycle.py router/main.py tests/test_lifecycle.py
git commit -m "fix: detect and handle backend config changes on rescan"
```

### Task 3: Llama.cpp Slot Control via extra_args

**Files:**
- Modify: `router/engines.py:216-233` (build_llama_cmd)
- Test: `tests/test_engines.py`

**Problem:** `build_llama_cmd()` ignores `extra_args`, preventing operators from setting `--parallel` to align router concurrency with llama-server slots.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engines.py
from unittest.mock import MagicMock

from router.engines import build_llama_cmd

def test_build_llama_cmd_includes_extra_args():
    """llama.cpp command should include extra_args for --parallel control."""
    config = MagicMock()
    config.llama_bin = "/fake/llama-server"

    cfg = {
        "model": "/path/to/model.gguf",
        "port": 8100,
        "ctx_size": 32768,
        "gpu_layers": 999,
        "extra_args": ["--parallel", "4", "--threads-http", "8"],
    }
    
    cmd = build_llama_cmd(cfg, config)
    
    # Assert: extra_args are included
    assert "--parallel" in cmd
    assert "4" in cmd
    assert "--threads-http" in cmd
    assert "8" in cmd
```

- [ ] **Step 2: Fix build_llama_cmd to include extra_args**

Modify `router/engines.py:216-233`:

```python
def build_llama_cmd(cfg: dict, config: "AppConfig") -> list[str]:
    supported = _detect_llama_flags(str(config.llama_bin))
    cmd = [
        str(config.llama_bin),
        "--model",        cfg["model"],
        "--ctx-size",     str(cfg.get("ctx_size", 32768)),
        "--n-gpu-layers", str(cfg.get("gpu_layers", 999)),
        "--host",         "0.0.0.0",
        "--port",         str(cfg["port"]),
    ]
    if "--flash-attn" in supported:
        cmd += ["--flash-attn", "on" if cfg.get("flash_attn", True) else "off"]
    if "--reasoning" in supported:
        cmd += ["--reasoning", "on" if cfg.get("reasoning", False) else "off"]
    if "--reasoning-budget" in supported and cfg.get("reasoning_budget"):
        cmd += ["--reasoning-budget", str(cfg["reasoning_budget"])]
    
    # Append extra_args for operator control (--parallel, --threads-http, etc.)
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)
    
    return cmd
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_engines.py::test_build_llama_cmd_includes_extra_args -v`

- [ ] **Step 4: Commit**

```bash
git add router/engines.py tests/test_engines.py
git commit -m "fix: include extra_args in llama.cpp command builder"
```

### Task 4: Fail-Closed Capability Routing — **SHIPPED** (commit `2e7ab99`)

**Files touched:**
- `router/routing.py` — `_filter_capable()` helper, AND-semantics gate in `_pick()` and `select_candidates()`, `extract_route_key()` / `InvalidRouteKey`
- `router/proxy.py` — fail-fast 400 for invalid `?backend=` / `[route:key]` on every surface
- `router/main.py` — same for WebSocket top-level `backend` and embedded `[route:key]`
- `docs/ARCHITECTURE.md`, `docs/API_SPEC.md`
- `tests/test_routing.py`, `tests/test_api.py`

**Shipped design (diverges from the original sketch above — keep this section authoritative):**

The original sketch split error codes by *cause* (`400 capability_error` when no backend in the registry can satisfy the request type, `503 backend_unavailable` otherwise). Implementation collapsed those into a single rule that splits by *who is at fault*:

- **`400 invalid_request_error`** — only when the caller named a backend that does not exist (`?backend=<typo>`, `[route:<typo>]`, WS top-level `backend: <typo>`). The client can fix the request; retrying the same payload will keep failing.
- **`503 service_unavailable`** with `Retry-After` — every classifier-cannot-satisfy case: empty tier, no tier backend declares the required capability, all candidates fail to start. The server might recover; retry is appropriate. Gemini keeps `400 INVALID_ARGUMENT` per its surface convention.

This is simpler to reason about (caller fault vs. server state) and matches what SDK retry policies already do — they retry 503 and surface 400 to the user.

**Capability filter — AND semantics, not elif:**

`_filter_capable(backends, keys, signals)` applies `supports_tools` then `supports_json_schema` filters sequentially. A request that combines `tools` and `response_format=json_schema` must land on a backend declaring **both**. Both `_pick()` (single best backend) and `select_candidates()` (top-N for retry) call the same helper.

```python
def _filter_capable(backends, keys, signals):
    if not signals:
        return keys
    filtered = keys
    if signals.has_tools:
        filtered = [k for k in filtered
                    if getattr(backends[k].get("capabilities"), "supports_tools", False)]
        if not filtered:
            return []
    if signals.needs_json_schema:
        filtered = [k for k in filtered
                    if getattr(backends[k].get("capabilities"), "supports_json_schema", False)]
        if not filtered:
            return []
    return filtered
```

**Direct-key bypass — strict, not advisory:**

The original sketch made `[route:key]` an *advisory* override that logged a warning and proceeded even when the backend lacked the requested capability. The shipped design is stricter:

- A direct-key match (`?backend=<key>`, `model_aliases` resolution, `[route:<key>]`, WS top-level `backend`) is a hard bypass — no capability filtering, no warning. The operator named the backend, and the operator owns the consequences.
- The validation gate is at the *key-existence* level. An invalid key fails fast with 400 on every surface; a valid key bypasses tier/capability gating entirely.

This avoids the worst-of-both behavior the advisory variant would cause (warning logged, then a confusing backend error from a model that can't do tools).

**Resolution chain (HTTP):** `?backend=` → `model_aliases` → `[route:key]` → model-name heuristic → classifier. The first three validate against the registry and short-circuit; only the classifier applies tier+capability filtering.

**`[route:key]` content shapes handled:** OpenAI string `content`, Anthropic `[{"type": "text", ...}]` blocks, Gemini `parts[].text`. The prefix is stripped from the payload before forwarding so it never reaches the backend.

**WebSocket parity:** `extract_route_key()` runs *before* the top-level `backend` short-circuit, so a `[route:<typo>]` cannot leak silently when the client also supplies `backend`. WS errors use the same `invalid_request_error` envelope as HTTP and keep the connection open so the client can fix the payload.

**Tests landed:**
- `tests/test_routing.py::TestCombinedCapabilityFilter` — six tests covering: combined no-match returns `[]`; mixed candidates returns only both-capable; `_pick()` returns both-capable / `""`; direct-key bypass; classifier end-to-end.
- `tests/test_routing.py::test_explicit_route_unknown_key_raises[_from_classify_candidates]` — `[route:typo]` raises `InvalidRouteKey`.
- `tests/test_api.py` — invalid `?backend=` and `[route:key]` → 400 on OpenAI/Anthropic/Gemini; WS typo → `invalid_request_error`; WS prefix-strip-with-top-level-backend; combined tools+json_schema with tool-only deep backend → 503.

**Acceptance:** `python3 -m pytest -q tests/test_routing.py tests/test_api.py` → 82 passed.

- [x] All steps complete (see commit `2e7ab99`).

---

## Phase 2: Anthropic/Gemini Hardening

### Task 5: Safe Retry Policy - No Retry for Tool/Schema Requests

**Files:**
- Modify: `router/proxy.py:387-500` (_proxy_nonstream)
- Modify: `router/proxy.py:856-932` (Anthropic non-streaming)
- Test: `tests/test_proxy.py`

**Problem:** `_proxy_nonstream()` retries every request on 502/503/504, but tool-calling and structured output must not retry per docs/ARCHITECTURE.md:249.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy.py
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from router.proxy import _proxy_nonstream

@pytest.mark.asyncio
async def test_no_retry_for_tool_calls():
    """Tool-calling requests should not be retried on backend errors."""
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Get the weather"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    
    # Mock client that returns 503
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"error": "slot full"}
    mock_resp.text = "slot full"
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("router.proxy.get_client", return_value=mock_client):
        response = await _proxy_nonstream(
            target_url="http://localhost:8100/v1/chat/completions",
            payload=payload,
            path="chat/completions",
            backend_key="test-backend",
            cfg={"port": 8100, "engine": "llama.cpp"},
            start_time=0,
            request_id="test-123",
            metrics_store=MagicMock(),
            config=SimpleNamespace(proxy=SimpleNamespace(
                retry_attempts=3,
                retry_on_status=[502, 503, 504],
                retry_backoff_sec=1.0,
                timeout_sec=300,
            )),
        )

    mock_client.post.assert_called_once()
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_no_retry_for_json_schema():
    """Structured output requests should not be retried."""
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Parse this"}],
        "response_format": {"type": "json_schema"},
    }
    
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.json.return_value = {"error": "bad gateway"}
    mock_resp.text = "bad gateway"
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("router.proxy.get_client", return_value=mock_client):
        response = await _proxy_nonstream(
            target_url="http://localhost:8100/v1/chat/completions",
            payload=payload,
            path="chat/completions",
            backend_key="test-backend",
            cfg={"port": 8100, "engine": "llama.cpp"},
            start_time=0,
            request_id="test-456",
            metrics_store=MagicMock(),
            config=SimpleNamespace(proxy=SimpleNamespace(
                retry_attempts=3,
                retry_on_status=[502, 503, 504],
                retry_backoff_sec=1.0,
                timeout_sec=300,
            )),
        )

    mock_client.post.assert_called_once()
    assert response.status_code == 502
```

- [ ] **Step 2: Implement can_retry helper in proxy.py**

Add at top of `router/proxy.py` after imports:

```python
def _can_retry(payload: dict, status: int) -> bool:
    """
    Determine if a request can be safely retried.
    
    Returns False for:
    - Tool-calling requests (tools/functions in payload)
    - Structured output requests (response_format with json_schema/json_object)
    - 503 Slot Full (backend explicitly says it's busy)
    """
    # Never retry tool calls or structured output
    if payload.get("tools") or payload.get("functions"):
        return False
    response_format = (payload.get("response_format") or {})
    if response_format.get("type") in ("json_schema", "json_object"):
        return False
    # Never retry 503 - backend is explicitly overloaded
    if status == 503:
        return False
    return True
```

- [ ] **Step 3: Update _proxy_nonstream to use can_retry**

Modify `router/proxy.py:409-415`:

```python
# Retry on transient errors
if status in config.proxy.retry_on_status and attempt < max_retries:
    if not _can_retry(payload, status):
        logger.debug(f"[{backend_key}] /{path} → {status} — not retrying (unsafe)")
        break
    logger.warning(
        f"[{backend_key}] /{path} → {status} (attempt {attempt + 1}/{max_retries + 1}), retrying"
    )
    await asyncio.sleep(config.proxy.retry_backoff_sec * (attempt + 1))
    continue
```

- [ ] **Step 4: Update Anthropic non-streaming path similarly**

Modify `router/proxy.py:868-873`:

```python
if resp_status in config.proxy.retry_on_status and attempt < max_retries:
    if not _can_retry(oai_payload, resp_status):
        logger.debug(f"[{backend_key}] anthropic → {resp_status} — not retrying (unsafe)")
        break
    logger.warning(
        f"[{backend_key}] anthropic → {resp_status} (attempt {attempt + 1}), retrying"
    )
    await asyncio.sleep(config.proxy.retry_backoff_sec * (attempt + 1))
    continue
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_proxy.py -v -k "no_retry"`

- [ ] **Step 6: Commit**

```bash
git add router/proxy.py tests/test_proxy.py
git commit -m "fix: disable retries for tool-calling and structured output"
```

### Task 6: Anthropic/Gemini Error Translation Fix

**Files:**
- Modify: `router/proxy.py:907-930` (Anthropic non-streaming error path)
- Modify: `router/proxy.py:1097-1109` (Gemini non-streaming error path)
- Test: `tests/test_proxy.py`

**Problem:** Both handlers run backend 4xx/5xx through success translators, returning error status with empty assistant payload instead of proper error envelope.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proxy.py
import pytest
from router.proxy import _extract_error_message

def test_extract_error_message_handles_nested_flat_and_raw_shapes():
    assert _extract_error_message({"error": {"message": "nested"}}) == "nested"
    assert _extract_error_message({"error": "flat"}) == "flat"
    assert _extract_error_message({"message": "top-level"}) == "top-level"
    assert _extract_error_message({"raw": "raw backend text"}) == "raw backend text"
    assert _extract_error_message("plain text") == "plain text"
```

- [ ] **Step 2: Add a shared error-extraction helper in proxy.py**

```python
def _extract_error_message(body: object, fallback: str = "Unknown error") -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str) and err.strip():
            return err
        for key in ("message", "detail", "raw"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value
    elif isinstance(body, str) and body.strip():
        return body
    return fallback
```

- [ ] **Step 3: Fix Anthropic non-streaming error path**

Modify `router/proxy.py:905-930`:

```python
total_ms = (time.time() - start_time) * 1000

if resp_status >= 400:
    message = _extract_error_message(oai_body)
    return JSONResponse(
        status_code=resp_status,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"Backend '{backend_key}' returned {resp_status}: {message}",
            },
        },
        headers={"anthropic-version": "2023-06-01"},
    )

anthropic_body = openai_to_anthropic(oai_body, original_model)
prompt_tokens  = anthropic_body["usage"]["input_tokens"]
comp_tokens    = anthropic_body["usage"]["output_tokens"]
```

- [ ] **Step 4: Fix Gemini non-streaming error path**

Modify `router/proxy.py:1091-1109`:

```python
resp_status = resp.status_code
try:
    oai_body = resp.json()
except Exception:
    oai_body = {}

total_ms = (time.time() - start_time) * 1000

if resp_status >= 400:
    message = _extract_error_message(oai_body)
    return JSONResponse(
        status_code=resp_status,
        content={
            "error": {
                "code": resp_status,
                "message": f"Backend '{backend_key}' returned {resp_status}: {message}",
            },
        },
    )

gemini_body = openai_to_gemini(oai_body, model)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_proxy.py -v -k "error"`

- [ ] **Step 6: Commit**

```bash
git add router/proxy.py tests/test_proxy.py
git commit -m "fix: return proper error envelopes for Anthropic/Gemini on backend errors"
```

### Task 7: Normalized Request/Response Path — **DEFERRED**

**Status:** Extracted to its own planning cycle. Do not execute as part of this plan.

**Why deferred:** Collapsing three proxy handlers into one is a week-scale refactor, not a task. The sketch previously in this section only covered OpenAI-shape roundtrip — the hard parts were unaddressed:

- **Streaming event normalization across three SSE dialects.** OpenAI emits `data: {...}\n\n` deltas with `choices[].delta.content` / `choices[].delta.tool_calls`. Anthropic emits typed events (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`) with per-block indices. Gemini emits `data: {...}\n\n` with `candidates[].content.parts[]` and cumulative (not delta) `finishReason`. A unified `NormalizedEvent` stream needs to be reverse-translatable into each dialect, lossless, and not introduce buffering that breaks TTFT claims.
- **Tool-call streaming semantics.** OpenAI streams tool-call arguments as incremental JSON-string deltas; Anthropic streams them as `input_json_delta` content-block events; Gemini emits them as complete `functionCall` parts in a single chunk. The normalized form must represent all three without losing the shape the client expects.
- **Usage accounting timing.** OpenAI emits usage in the final chunk (optional); Anthropic emits `message_delta` with `usage.output_tokens` mid-stream plus final in `message_stop`; Gemini emits `usageMetadata` only at stream end. The metrics path depends on where we attach usage.
- **Error-during-stream handling.** Once bytes are flushed, we can't change status code — each dialect has a different way to signal in-stream errors, and the normalized layer needs a consistent policy.

**What to do instead:** After Phase 1 + Tasks 5–6 land and stabilize, open a dedicated plan (tentatively `docs/superpowers/plans/YYYY-MM-DD-normalized-path.md`) that:

1. Starts by auditing every streaming event shape emitted by each of the three current handlers.
2. Defines `NormalizedEvent` as a superset that can reverse-translate losslessly, with explicit tests per dialect.
3. Migrates the three handlers one at a time, keeping the old handler in place until the normalized equivalent ships with equivalent streaming tests.
4. Lands on a single `_proxy_normalized` only after all three dialect adapters pass their existing test suites against the normalized path.

Token/schema-format converters alone (the `NormalizedRequest`/`NormalizedResponse` dataclasses from the old Task 7 sketch) are not worth shipping without the streaming piece — they would add a layer of indirection for no behavioral benefit.

---

## Phase 3: Observability and Hardening (Can Wait)

### Task 8: Startup-Failure Circuit Breaking

**Files:**
- Modify: `router/lifecycle.py` (add circuit breaker state)
- Test: `tests/test_lifecycle.py`

- [ ] Implement circuit breaker that tracks consecutive startup failures
- [ ] After N failures, mark backend as circuit-open for M seconds
- [ ] Return fast 503 instead of attempting startup when circuit is open

### Task 9: Request Routing Decision Logging

**Files:**
- Modify: `router/routing.py` (add decision metadata)
- Modify: `router/proxy.py` (log routing decisions)

- [ ] Return routing decision metadata from classify_candidates
- [ ] Log: selected backend, rejected candidates with reasons, capability gates

### Task 10: Context Window Preflight Check

**Files:**
- Create: `router/context_check.py` (new module)
- Modify: `router/proxy.py` (add preflight check)

- [ ] Estimate token count from request
- [ ] Compare against backend's ctx_size
- [ ] Return 413 Request Entity Too Large if exceeds context

---

## Self-Review

### Spec Coverage

| Audit Finding | Task | Phase |
|---------------|------|-------|
| Port collision | Task 1 | Phase 1 |
| Rescan config drift (+ in-flight drain) | Task 2 | Phase 1 |
| llama.cpp extra_args | Task 3 | Phase 1 |
| Fail-closed routing (+ direct-key bypass) | Task 4 | **Shipped** (`2e7ab99`) |
| Unsafe retries | Task 5 | Phase 2 |
| Anthropic/Gemini errors | Task 6 | Phase 2 |
| Normalized path | Task 7 | **Deferred — own plan** |
| Circuit breaker | Task 8 | Phase 3 |
| Routing logs | Task 9 | Phase 3 |
| Context preflight | Task 10 | Phase 3 |

### Placeholder Scan

No TBD/TODO placeholders remain in the executable tasks (1–6, 8–10). Task 7 is explicitly marked deferred with its own scope notes; it is not part of execution for this plan.

### Type Consistency

- `BackendConfig` used consistently
- `AppConfig` passed through all functions

---

## Execution Handoff

**Executable scope:** Tasks 1–6 (Phase 1 + Phase 2). Task 7 is deferred with its own planning cycle. Tasks 8–10 (Phase 3) remain stubbed for later.

**Recommended bundling:** Phase 1 (Tasks 1–4) + Phase 2 (Tasks 5–6) ship as one coordinated PR — six focused commits, each test-backed, each scoped to one behavior. Total surface: `router/{registry,discovery,lifecycle,engines,routing,proxy,main}.py` + matching tests.

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Best for keeping each task's scope enforced by a clean context window.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch with checkpoints between tasks.

**Which approach?**
