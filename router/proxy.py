from __future__ import annotations

"""
router/proxy.py — OpenAI-compatible /v1/ request proxy

Forwards inference requests to the appropriate backend with:
  - Backpressure via asyncio.Semaphore
  - Configurable retries on transient errors
  - TTFT (time-to-first-token) capture for streaming responses
  - Improved streaming token counting
  - Optional audit logging
  - Model alias resolution
  - Metrics recording after each request
  - Structured JSON error responses
"""

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from router.metrics import RequestRecord, MetricsStore, extract_token_counts
from router.routing import classify_candidates

if TYPE_CHECKING:
    from router.config import AppConfig
    from router.lifecycle import BackendManager

logger = logging.getLogger("llm-router.proxy")
audit_logger = logging.getLogger("llm-router.audit")

# Module-level semaphore — initialized by init_semaphore() at app startup
_semaphore: asyncio.Semaphore | None = None

# ─────────────────────────────────────────────────────────────
# Shared httpx client pool — avoids per-request connection setup
# ─────────────────────────────────────────────────────────────
_clients: dict[int, httpx.AsyncClient] = {}


def get_client(port: int, timeout: float) -> httpx.AsyncClient:
    """Return a shared AsyncClient for a backend port, creating one if needed."""
    client = _clients.get(port)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout)
        _clients[port] = client
    return client


def evict_client(port: int):
    """Remove a client from the pool (sync-safe, for use in lifecycle.stop()).

    The underlying connection is not explicitly closed — Python/httpx will
    clean it up via GC.  The next get_client() call creates a fresh one.
    """
    _clients.pop(port, None)


async def close_client(port: int):
    """Close and remove the client for a specific port."""
    client = _clients.pop(port, None)
    if client and not client.is_closed:
        await client.aclose()


async def close_all_clients():
    """Close all pooled clients (called at shutdown)."""
    for client in _clients.values():
        if hasattr(client, "is_closed") and not client.is_closed:
            await client.aclose()
    _clients.clear()

_MODEL_MATCH_ENGINE_PRIORITY = {
    # Prefer already-running passthrough servers for exact model IDs so clients
    # can select a real model name without knowing an internal backend key.
    "openai": 0,
    "trt-llm": 1,
    "trt-llm-docker": 2,
    "vllm": 3,
    "sglang": 4,
    "llama.cpp": 5,
    "huggingface": 6,
    "ollama": 7,
}


def init_semaphore(max_concurrent: int):
    """Call once at startup to set up the backpressure semaphore."""
    global _semaphore
    _semaphore = asyncio.Semaphore(max_concurrent)


def _is_path_like_model_name(model: str) -> bool:
    if not model:
        return False

    expanded = os.path.expanduser(model)
    return (
        expanded.startswith("/")
        or model.startswith(("~/", "./", "../"))
        or expanded.endswith((".gguf", ".bin", ".pt", ".pth", ".safetensors", ".engine"))
        or os.path.exists(expanded)
    )


def _backend_model_name(cfg: dict) -> str | None:
    """Return a client-facing model id when the backend has one."""
    model = cfg.get("model", "")
    if not model or _is_path_like_model_name(model):
        return None
    return model


def _preferred_model_backend(
    candidates: list[str],
    backends: dict,
    manager: "BackendManager" | None = None,
) -> str:
    def _score(key: str) -> tuple[int, int, str]:
        running = 0 if manager and manager.is_running(key) else 1
        engine = backends[key].get("engine", "")
        engine_rank = _MODEL_MATCH_ENGINE_PRIORITY.get(engine, 99)
        return (running, engine_rank, key)

    return min(candidates, key=_score)


def build_model_aliases(
    backends: dict,
    configured_aliases: dict[str, str] | None = None,
    manager: "BackendManager" | None = None,
) -> dict[str, str]:
    """
    Build a model-name → backend-key map.

    Explicit config aliases win. Auto aliases are generated from backend model
    ids when they are safe to expose to clients.
    """
    auto_aliases: dict[str, list[str]] = {}
    for key, cfg in backends.items():
        model = _backend_model_name(cfg)
        if model and model not in backends:
            auto_aliases.setdefault(model, []).append(key)

    resolved = {
        model: _preferred_model_backend(candidates, backends, manager)
        for model, candidates in auto_aliases.items()
    }
    for alias, target in (configured_aliases or {}).items():
        if target in backends:
            resolved[alias] = target
    return resolved


def resolve_requested_model(
    model: str,
    backends: dict,
    configured_aliases: dict[str, str] | None = None,
    manager: "BackendManager" | None = None,
) -> str | None:
    """
    Resolve a client-supplied model string to a backend key.

    Order:
      1. Exact backend key
      2. Explicit config alias
      3. Exact backend model id (auto alias)
    """
    if not model:
        return None
    if model in backends:
        return model
    return build_model_aliases(backends, configured_aliases, manager).get(model)


def _audit_request(request_id: str, backend_key: str, endpoint: str,
                   payload: dict, config: "AppConfig", api_key_name: str = ""):
    """Log request metadata for audit trail."""
    if not config.audit.enabled:
        return
    entry = {
        "type": "request",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend_key,
        "endpoint": endpoint,
        "model": payload.get("model", ""),
        "stream": payload.get("stream", False),
        "api_key": api_key_name,
    }
    if config.audit.log_request_body and not config.audit.redact_content:
        entry["body"] = payload
    elif config.audit.log_request_body and config.audit.redact_content:
        entry["message_count"] = len(payload.get("messages", []))
        entry["has_tools"] = "tools" in payload
        entry["max_tokens"] = payload.get("max_tokens")
    audit_logger.info(json.dumps(entry))


def _audit_response(request_id: str, status_code: int, latency_ms: float,
                    tokens: dict, config: "AppConfig"):
    """Log response metadata for audit trail."""
    if not config.audit.enabled:
        return
    entry = {
        "type": "response",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_code": status_code,
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": tokens.get("prompt", 0),
        "completion_tokens": tokens.get("completion", 0),
    }
    audit_logger.info(json.dumps(entry))


async def handle_proxy(
    path: str,
    request: Request,
    manager: "BackendManager",
    metrics_store: MetricsStore,
    config: "AppConfig",
) -> StreamingResponse | JSONResponse:
    """
    Main proxy handler for POST /v1/{path}.

    1. Parse JSON body
    2. Resolve model alias if configured
    3. Determine backend (query param → alias → explicit route prefix → classifier)
    4. Validate backend key
    5. Acquire semaphore (backpressure) or return 503
    6. Start backend if not running
    7. Proxy request with retry, capturing TTFT and metrics
    8. Release semaphore
    """
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    api_key_name = getattr(request.state, "api_key_name", "")
    # Snapshot backends once so routing + lookup use a consistent view
    # even if /rescan swaps manager.backends concurrently.
    backends = manager.snapshot_backends()

    # ── Parse body ────────────────────────────────────────────
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body", "request_id": request_id},
        )

    # ── Determine backend ─────────────────────────────────────
    # Priority: ?backend= param → model alias → classifier candidates with fallback
    explicit_key = (
        request.query_params.get("backend")
        or resolve_requested_model(
            payload.get("model", ""),
            backends,
            config.model_aliases,
            manager,
        )
    )

    if explicit_key:
        candidates = [explicit_key]
    else:
        candidates = classify_candidates(
            payload, backends, config, limit=3,
            healthy_fn=manager.is_healthy,
        )

    # Validate at least first candidate exists
    if not candidates or (len(candidates) == 1 and candidates[0] not in backends):
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown backend '{candidates[0] if candidates else ''}'",
                "valid_backends": list(backends.keys()),
                "request_id": request_id,
            },
        )

    _audit_request(request_id, candidates[0], path, payload, config, api_key_name)

    # ── Backpressure ──────────────────────────────────────────
    if _semaphore is None:
        init_semaphore(config.proxy.max_concurrent_requests)

    try:
        await asyncio.wait_for(
            _semaphore.acquire(),
            timeout=config.proxy.queue_timeout_sec,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Router overloaded — too many concurrent requests",
                "retry_after": config.proxy.queue_timeout_sec,
                "request_id": request_id,
            },
        )

    try:
        # ── Try candidates with fallback ──────────────────────
        last_error = None
        backend_key = candidates[0]

        for candidate in candidates:
            if candidate not in backends:
                continue
            try:
                await manager.ensure_running(candidate)
                backend_key = candidate
                last_error = None
                break
            except RuntimeError as e:
                last_error = e
                manager.mark_unhealthy(candidate)
                logger.warning(f"[{candidate}] failed to start, trying next candidate: {e}")
                continue

        if last_error:
            cfg = backends.get(backend_key, {})
            return JSONResponse(
                status_code=503,
                content={
                    "error": str(last_error),
                    "backend": backend_key,
                    "log": cfg.get("log", ""),
                    "request_id": request_id,
                },
            )

        cfg = backends[backend_key]
        target_url = f"http://localhost:{cfg['port']}/v1/{path}"
        is_stream = payload.get("stream", False)

        if is_stream:
            return await _proxy_stream(
                target_url, payload, path, backend_key, cfg,
                start_time, request_id, metrics_store, config, manager,
            )
        else:
            async with manager.request_lease(backend_key):
                return await _proxy_nonstream(
                    target_url, payload, path, backend_key, cfg,
                    start_time, request_id, metrics_store, config,
                )

    finally:
        _semaphore.release()


async def _proxy_nonstream(
    target_url: str,
    payload: dict,
    path: str,
    backend_key: str,
    cfg: dict,
    start_time: float,
    request_id: str,
    metrics_store: MetricsStore,
    config: "AppConfig",
) -> JSONResponse:
    max_retries = config.proxy.retry_attempts
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            client = get_client(cfg["port"], config.proxy.timeout_sec)
            resp = await client.post(target_url, json=payload)

            total_ms = (time.time() - start_time) * 1000
            status = resp.status_code

            # Retry on transient errors
            if status in config.proxy.retry_on_status and attempt < max_retries:
                logger.warning(
                    f"[{backend_key}] /{path} → {status} (attempt {attempt + 1}/{max_retries + 1}), retrying"
                )
                await asyncio.sleep(config.proxy.retry_backoff_sec * (attempt + 1))
                continue

            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}

            prompt_tokens, completion_tokens = extract_token_counts(body)
            tps = (completion_tokens / (total_ms / 1000)) if total_ms > 0 else 0.0

            metrics_store.record(RequestRecord(
                request_id=request_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                backend_key=backend_key,
                engine=cfg.get("engine", ""),
                model_path=cfg.get("model", cfg.get("model_dir", "")),
                endpoint=path,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=total_ms,       # non-streaming: TTFT = total latency
                total_latency_ms=total_ms,
                tokens_per_sec=round(tps, 2),
                status_code=status,
                error=None if status < 400 else resp.text[:200],
            ))

            _audit_response(request_id, status, total_ms,
                           {"prompt": prompt_tokens, "completion": completion_tokens}, config)

            logger.info(f"[{backend_key}] /{path} → {status} ({total_ms:.0f}ms)")
            return JSONResponse(content=body, status_code=status)

        except httpx.ConnectError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"[{backend_key}] connect error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                await asyncio.sleep(config.proxy.retry_backoff_sec * (attempt + 1))
                continue

        except httpx.TimeoutException:
            total_ms = (time.time() - start_time) * 1000
            metrics_store.record(RequestRecord(
                request_id=request_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                backend_key=backend_key,
                engine=cfg.get("engine", ""),
                model_path=cfg.get("model", cfg.get("model_dir", "")),
                endpoint=path,
                prompt_tokens=0, completion_tokens=0,
                ttft_ms=total_ms, total_latency_ms=total_ms,
                tokens_per_sec=0.0, status_code=504,
                error="timeout",
            ))
            return JSONResponse(
                status_code=504,
                content={
                    "error": f"Backend '{backend_key}' timed out after {config.proxy.timeout_sec}s",
                    "backend": backend_key,
                    "request_id": request_id,
                },
            )

    # All retries exhausted (connect errors)
    total_ms = (time.time() - start_time) * 1000
    metrics_store.record(RequestRecord(
        request_id=request_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend_key=backend_key,
        engine=cfg.get("engine", ""),
        model_path=cfg.get("model", cfg.get("model_dir", "")),
        endpoint=path,
        prompt_tokens=0, completion_tokens=0,
        ttft_ms=total_ms, total_latency_ms=total_ms,
        tokens_per_sec=0.0, status_code=502,
        error=str(last_error)[:200] if last_error else "connection_failed",
    ))
    return JSONResponse(
        status_code=502,
        content={
            "error": f"Backend '{backend_key}' unreachable after {max_retries + 1} attempts",
            "backend": backend_key,
            "request_id": request_id,
        },
    )


async def _proxy_stream(
    target_url: str,
    payload: dict,
    path: str,
    backend_key: str,
    cfg: dict,
    start_time: float,
    request_id: str,
    metrics_store: MetricsStore,
    config: "AppConfig",
    manager: "BackendManager" = None,
) -> StreamingResponse:
    """Stream the response, capturing TTFT on the first chunk with improved token counting."""

    async def stream_with_metrics() -> AsyncIterator[bytes]:
        ttft_ms = None
        completion_tokens = 0
        prompt_tokens = 0
        actual_status = 200
        _lease = manager.request_lease(backend_key) if manager else contextlib.nullcontext()

        try:
            async with _lease:
                client = get_client(cfg["port"], config.proxy.timeout_sec)
                async with client.stream("POST", target_url, json=payload) as resp:
                        actual_status = resp.status_code

                        # Backend returned an error — emit SSE error event
                        if actual_status >= 400:
                            error_body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                            logger.warning(f"[{backend_key}] backend returned {actual_status}: {error_body[:200]}")
                            error_event = {"error": {
                                "message": f"Backend '{backend_key}' returned {actual_status}",
                                "type": "backend_error",
                                "code": actual_status,
                            }}
                            yield f"data: {json.dumps(error_event)}\n\ndata: [DONE]\n\n".encode()
                            total_ms = (time.time() - start_time) * 1000
                            metrics_store.record(RequestRecord(
                                request_id=request_id,
                                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                backend_key=backend_key,
                                engine=cfg.get("engine", ""),
                                model_path=cfg.get("model", cfg.get("model_dir", "")),
                                endpoint=path,
                                prompt_tokens=0, completion_tokens=0,
                                ttft_ms=total_ms, total_latency_ms=total_ms,
                                tokens_per_sec=0.0,
                                status_code=actual_status,
                                error=error_body[:200],
                            ))
                            return

                        _metrics_buf = ""
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                if ttft_ms is None:
                                    ttft_ms = (time.time() - start_time) * 1000

                                # Buffer decoded text for line-safe SSE metrics parsing
                                _metrics_buf += chunk.decode("utf-8", errors="replace")
                                while "\n" in _metrics_buf:
                                    line, _metrics_buf = _metrics_buf.split("\n", 1)
                                    line = line.strip()
                                    if not line.startswith("data:"):
                                        continue
                                    data_str = line[5:].strip()
                                    if not data_str or data_str == "[DONE]":
                                        continue
                                    try:
                                        data = json.loads(data_str)
                                        if "usage" in data:
                                            u = data["usage"]
                                            prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                                            completion_tokens = u.get("completion_tokens", completion_tokens)
                                        choices = data.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            text = delta.get("content", "")
                                            if text:
                                                completion_tokens += max(1, len(text) // 4)
                                    except json.JSONDecodeError:
                                        pass

                                yield chunk

                total_ms = (time.time() - start_time) * 1000
                tps = (completion_tokens / (total_ms / 1000)) if total_ms > 0 and completion_tokens > 0 else 0.0

                metrics_store.record(RequestRecord(
                    request_id=request_id,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    backend_key=backend_key,
                    engine=cfg.get("engine", ""),
                    model_path=cfg.get("model", cfg.get("model_dir", "")),
                    endpoint=path,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_ms=ttft_ms or total_ms,
                    total_latency_ms=total_ms,
                    tokens_per_sec=round(tps, 2),
                    status_code=actual_status,
                    error=None,
                ))

                _audit_response(request_id, actual_status, total_ms,
                               {"prompt": prompt_tokens, "completion": completion_tokens}, config)

                logger.info(
                    f"[{backend_key}] streaming /{path} done "
                    f"(TTFT={ttft_ms:.0f}ms, total={total_ms:.0f}ms, ~{completion_tokens} tokens)"
                )

        except Exception as e:
            logger.warning(f"[{backend_key}] stream error: {e}")
            total_ms = (time.time() - start_time) * 1000
            # Emit SSE error event so clients see the failure
            error_event = {"error": {"message": str(e), "type": "stream_error"}}
            yield f"data: {json.dumps(error_event)}\n\ndata: [DONE]\n\n".encode()
            metrics_store.record(RequestRecord(
                request_id=request_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                backend_key=backend_key,
                engine=cfg.get("engine", ""),
                model_path=cfg.get("model", cfg.get("model_dir", "")),
                endpoint=path,
                prompt_tokens=0, completion_tokens=0,
                ttft_ms=ttft_ms or total_ms,
                total_latency_ms=total_ms,
                tokens_per_sec=0.0, status_code=500,
                error=str(e)[:200],
            ))

    return StreamingResponse(stream_with_metrics(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────
# Anthropic Messages API proxy
# ─────────────────────────────────────────────────────────────

async def handle_anthropic_proxy(
    request: Request,
    manager: "BackendManager",
    metrics_store: MetricsStore,
    config: "AppConfig",
) -> StreamingResponse | JSONResponse:
    """
    Handle POST /anthropic/v1/messages and /v1/messages.

    1. Parse Anthropic request body
    2. Determine backend (query param → model name mapping → classifier)
    3. Translate request to OpenAI format
    4. Proxy to local backend with retry
    5. Translate response back to Anthropic format
    """
    from router.anthropic_compat import (
        anthropic_to_openai, openai_to_anthropic,
        stream_openai_to_anthropic, model_to_backend,
    )

    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    api_key_name = getattr(request.state, "api_key_name", "")
    backends = manager.snapshot_backends()

    # ── Parse body ────────────────────────────────────────────
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error",
                     "message": "Invalid JSON body"}, "request_id": request_id},
        )

    original_model = payload.get("model", "claude-3-5-sonnet-20241022")
    is_stream      = payload.get("stream", False)

    # ── Determine backend candidates ─────────────────────────
    explicit_key = request.query_params.get("backend")
    if not explicit_key:
        explicit_key = resolve_requested_model(
            original_model,
            backends,
            config.model_aliases,
            manager,
        )
    if not explicit_key:
        explicit_key = model_to_backend(original_model)

    if explicit_key and explicit_key in backends:
        candidates = [explicit_key]
    else:
        oai_for_classify = anthropic_to_openai(payload)
        candidates = classify_candidates(
            oai_for_classify, backends, config, limit=3,
            healthy_fn=manager.is_healthy,
        )

    if not candidates or not any(c in backends for c in candidates):
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {
                "type": "invalid_request_error",
                "message": f"No backend available for model '{original_model}'. "
                           f"Valid backends: {list(backends.keys())}",
            }},
        )

    # ── Backpressure ──────────────────────────────────────────
    if _semaphore is None:
        init_semaphore(config.proxy.max_concurrent_requests)

    try:
        await asyncio.wait_for(
            _semaphore.acquire(),
            timeout=config.proxy.queue_timeout_sec,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=529,   # Anthropic uses 529 for overload
            content={"type": "error", "error": {
                "type": "overloaded_error",
                "message": "Router overloaded — too many concurrent requests",
            }},
        )

    try:
        # ── Start backend with fallback ───────────────────────
        backend_key = None
        last_error = None
        for candidate in candidates:
            if candidate not in backends:
                continue
            try:
                await manager.ensure_running(candidate)
                backend_key = candidate
                last_error = None
                break
            except RuntimeError as e:
                last_error = e
                manager.mark_unhealthy(candidate)
                logger.warning(f"[{candidate}] anthropic startup failed, trying next: {e}")
                continue

        if backend_key is None:
            return JSONResponse(
                status_code=503,
                content={"type": "error", "error": {
                    "type": "api_error",
                    "message": str(last_error) if last_error else "All backends failed to start",
                }},
            )

        _audit_request(request_id, backend_key, "anthropic/messages", payload, config, api_key_name)
        cfg        = backends[backend_key]
        oai_payload = anthropic_to_openai(payload)
        target_url  = f"http://localhost:{cfg['port']}/v1/chat/completions"

        if is_stream:
            converter = stream_openai_to_anthropic(original_model)

            async def anthropic_stream() -> AsyncIterator[bytes]:
                ttft_ms = None
                completion_tokens = 0
                actual_status = 200
                try:
                    async with manager.request_lease(backend_key):
                        client = get_client(cfg["port"], config.proxy.timeout_sec)
                        async with client.stream("POST", target_url, json=oai_payload) as resp:
                                actual_status = resp.status_code
                                if actual_status >= 400:
                                    error_body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                                    logger.warning(f"[{backend_key}] anthropic backend returned {actual_status}")
                                    err = {"type": "error", "error": {
                                        "type": "api_error",
                                        "message": f"Backend returned {actual_status}: {error_body[:200]}",
                                    }}
                                    yield f"event: error\ndata: {json.dumps(err)}\n\n".encode()
                                    total_ms = (time.time() - start_time) * 1000
                                    metrics_store.record(RequestRecord(
                                        request_id=request_id,
                                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                        backend_key=backend_key, engine=cfg.get("engine", ""),
                                        model_path=cfg.get("model", cfg.get("model_dir", "")),
                                        endpoint="anthropic/messages",
                                        prompt_tokens=0, completion_tokens=0,
                                        ttft_ms=total_ms, total_latency_ms=total_ms,
                                        tokens_per_sec=0.0, status_code=actual_status,
                                        error=error_body[:200],
                                    ))
                                    return
                                raw_stream = resp.aiter_bytes()
                                async for chunk in converter(raw_stream):
                                    if ttft_ms is None and chunk:
                                        ttft_ms = (time.time() - start_time) * 1000
                                    try:
                                        for line in chunk.decode("utf-8", errors="replace").splitlines():
                                            if not line.startswith("data:"):
                                                continue
                                            data = json.loads(line[5:].strip())
                                            if data.get("type") == "content_block_delta":
                                                delta = data.get("delta", {})
                                                text = delta.get("text", "")
                                                if text:
                                                    completion_tokens += max(1, len(text) // 4)
                                            elif data.get("type") == "message_delta":
                                                u = data.get("usage", {})
                                                if u.get("output_tokens"):
                                                    completion_tokens = u["output_tokens"]
                                    except Exception:
                                        pass
                                    yield chunk
                        total_ms = (time.time() - start_time) * 1000
                        tps = (completion_tokens / (total_ms / 1000)) if total_ms > 0 and completion_tokens > 0 else 0.0
                        metrics_store.record(RequestRecord(
                            request_id=request_id,
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            backend_key=backend_key, engine=cfg.get("engine", ""),
                            model_path=cfg.get("model", cfg.get("model_dir", "")),
                            endpoint="anthropic/messages",
                            prompt_tokens=0, completion_tokens=completion_tokens,
                            ttft_ms=ttft_ms or total_ms, total_latency_ms=total_ms,
                            tokens_per_sec=round(tps, 2), status_code=actual_status, error=None,
                        ))
                        _audit_response(request_id, actual_status, total_ms,
                                       {"prompt": 0, "completion": completion_tokens}, config)
                        logger.info(f"[{backend_key}] anthropic stream done ({total_ms:.0f}ms, ~{completion_tokens} tokens)")
                except Exception as e:
                    logger.warning(f"[{backend_key}] anthropic stream error: {e}")
                    err = {"type": "error", "error": {"type": "stream_error", "message": str(e)}}
                    yield f"event: error\ndata: {json.dumps(err)}\n\n".encode()

            return StreamingResponse(
                anthropic_stream(),
                media_type="text/event-stream",
                headers={"anthropic-version": "2023-06-01"},
            )

        else:
            async with manager.request_lease(backend_key):
                # Non-streaming with retry support
                max_retries = config.proxy.retry_attempts
                oai_body = {}
                resp_status = 500

                for attempt in range(max_retries + 1):
                    try:
                        client = get_client(cfg["port"], config.proxy.timeout_sec)
                        resp = await client.post(target_url, json=oai_payload)
                        resp_status = resp.status_code

                        if resp_status in config.proxy.retry_on_status and attempt < max_retries:
                            logger.warning(
                                f"[{backend_key}] anthropic → {resp_status} (attempt {attempt + 1}), retrying"
                            )
                            await asyncio.sleep(config.proxy.retry_backoff_sec * (attempt + 1))
                            continue

                        try:
                            oai_body = resp.json()
                        except Exception:
                            oai_body = {}
                        break

                    except httpx.ConnectError as e:
                        if attempt < max_retries:
                            await asyncio.sleep(config.proxy.retry_backoff_sec * (attempt + 1))
                            continue
                        raise
                    except httpx.TimeoutException:
                        total_ms = (time.time() - start_time) * 1000
                        return JSONResponse(
                            status_code=504,
                            content={"type": "error", "error": {
                                "type": "timeout_error",
                                "message": f"Backend timed out after {config.proxy.timeout_sec}s",
                            }},
                            headers={"anthropic-version": "2023-06-01"},
                        )

                total_ms = (time.time() - start_time) * 1000

                anthropic_body = openai_to_anthropic(oai_body, original_model)
                prompt_tokens  = anthropic_body["usage"]["input_tokens"]
                comp_tokens    = anthropic_body["usage"]["output_tokens"]
                tps = (comp_tokens / (total_ms / 1000)) if total_ms > 0 else 0.0

                metrics_store.record(RequestRecord(
                    request_id=request_id,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    backend_key=backend_key, engine=cfg.get("engine", ""),
                    model_path=cfg.get("model", cfg.get("model_dir", "")),
                    endpoint="anthropic/messages",
                    prompt_tokens=prompt_tokens, completion_tokens=comp_tokens,
                    ttft_ms=total_ms, total_latency_ms=total_ms,
                    tokens_per_sec=round(tps, 2),
                    status_code=resp_status, error=None,
                ))

                _audit_response(request_id, resp_status, total_ms,
                               {"prompt": prompt_tokens, "completion": comp_tokens}, config)

                logger.info(f"[{backend_key}] anthropic /messages → {resp_status} ({total_ms:.0f}ms)")
                return JSONResponse(
                    content=anthropic_body,
                    status_code=resp_status,
                    headers={"anthropic-version": "2023-06-01"},
                )

    finally:
        _semaphore.release()


# ─────────────────────────────────────────────────────────────
# Gemini API proxy
# ─────────────────────────────────────────────────────────────

async def handle_gemini_proxy(
    model: str,
    is_stream: bool,
    request: Request,
    manager: "BackendManager",
    metrics_store: MetricsStore,
    config: "AppConfig",
) -> StreamingResponse | JSONResponse:
    """
    Handle Gemini generateContent and streamGenerateContent.
    Translates to OpenAI format, proxies, translates back.
    """
    from router.gemini_compat import (
        gemini_to_openai, openai_to_gemini,
        stream_openai_to_gemini, gemini_model_to_backend,
    )

    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    backends = manager.snapshot_backends()

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON body"}})

    # Determine backend candidates
    explicit_key = (
        request.query_params.get("backend")
        or resolve_requested_model(model, backends, config.model_aliases, manager)
        or gemini_model_to_backend(model)
    )
    if explicit_key and explicit_key in backends:
        candidates = [explicit_key]
    else:
        oai_temp = gemini_to_openai(payload)
        candidates = classify_candidates(
            oai_temp, backends, config, limit=3,
            healthy_fn=manager.is_healthy,
        )

    if not candidates or not any(c in backends for c in candidates):
        return JSONResponse(status_code=400, content={
            "error": {"message": f"No backend for model '{model}'"}
        })

    # Backpressure
    if _semaphore is None:
        init_semaphore(config.proxy.max_concurrent_requests)
    try:
        await asyncio.wait_for(_semaphore.acquire(), timeout=config.proxy.queue_timeout_sec)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=503, content={"error": {"message": "Overloaded"}})

    try:
        backend_key = None
        last_error = None
        for candidate in candidates:
            if candidate not in backends:
                continue
            try:
                await manager.ensure_running(candidate)
                backend_key = candidate
                last_error = None
                break
            except RuntimeError as e:
                last_error = e
                manager.mark_unhealthy(candidate)
                logger.warning(f"[{candidate}] gemini startup failed, trying next: {e}")
                continue

        if backend_key is None:
            return JSONResponse(status_code=503, content={
                "error": {"message": str(last_error) if last_error else "All backends failed"}
            })

        cfg = backends[backend_key]
        oai_payload = gemini_to_openai(payload, is_stream=is_stream)
        target_url = f"http://localhost:{cfg['port']}/v1/chat/completions"

        if is_stream:
            converter = stream_openai_to_gemini(model)

            async def gemini_stream() -> AsyncIterator[bytes]:
                try:
                    async with manager.request_lease(backend_key):
                        client = get_client(cfg["port"], config.proxy.timeout_sec)
                        async with client.stream("POST", target_url, json=oai_payload) as resp:
                                if resp.status_code >= 400:
                                    error_body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                                    logger.warning(f"[{backend_key}] gemini backend returned {resp.status_code}")
                                    total_ms = (time.time() - start_time) * 1000
                                    metrics_store.record(RequestRecord(
                                        request_id=request_id,
                                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                        backend_key=backend_key, engine=cfg.get("engine", ""),
                                        model_path=cfg.get("model", cfg.get("model_dir", "")),
                                        endpoint=f"gemini/{model}",
                                        prompt_tokens=0, completion_tokens=0,
                                        ttft_ms=total_ms, total_latency_ms=total_ms,
                                        tokens_per_sec=0.0, status_code=resp.status_code,
                                        error=error_body[:200],
                                    ))
                                    error_event = {"error": {
                                        "code": resp.status_code,
                                        "message": f"Backend '{backend_key}' returned {resp.status_code}",
                                    }}
                                    yield f"data: {json.dumps(error_event)}\n\n".encode()
                                    return
                                async for chunk in converter(resp.aiter_bytes()):
                                    yield chunk
                        total_ms = (time.time() - start_time) * 1000
                        metrics_store.record(RequestRecord(
                            request_id=request_id,
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            backend_key=backend_key, engine=cfg.get("engine", ""),
                            model_path=cfg.get("model", cfg.get("model_dir", "")),
                            endpoint=f"gemini/{model}",
                            prompt_tokens=0, completion_tokens=0,
                            ttft_ms=total_ms, total_latency_ms=total_ms,
                            tokens_per_sec=0.0, status_code=200, error=None,
                        ))
                except Exception as e:
                    logger.warning(f"[{backend_key}] gemini stream error: {e}")

            return StreamingResponse(gemini_stream(), media_type="text/event-stream")
        else:
            async with manager.request_lease(backend_key):
                client = get_client(cfg["port"], config.proxy.timeout_sec)
                resp = await client.post(target_url, json=oai_payload)
                resp_status = resp.status_code
                try:
                    oai_body = resp.json()
                except Exception:
                    oai_body = {}
                total_ms = (time.time() - start_time) * 1000
                gemini_body = openai_to_gemini(oai_body, model)
                metrics_store.record(RequestRecord(
                    request_id=request_id,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    backend_key=backend_key, engine=cfg.get("engine", ""),
                    model_path=cfg.get("model", cfg.get("model_dir", "")),
                    endpoint=f"gemini/{model}",
                    prompt_tokens=0, completion_tokens=0,
                    ttft_ms=total_ms, total_latency_ms=total_ms,
                    tokens_per_sec=0.0, status_code=resp_status, error=None,
                ))
                logger.info(f"[{backend_key}] gemini /{model} → {resp_status} ({total_ms:.0f}ms)")
                return JSONResponse(content=gemini_body, status_code=resp_status)

    finally:
        _semaphore.release()
