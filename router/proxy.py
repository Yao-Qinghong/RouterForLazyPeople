"""
router/proxy.py — OpenAI-compatible /v1/ request proxy

Forwards inference requests to the appropriate backend with:
  - Backpressure via asyncio.Semaphore
  - TTFT (time-to-first-token) capture for streaming responses
  - Metrics recording after each request
  - Structured JSON error responses
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from router.metrics import RequestRecord, MetricsStore, extract_token_counts
from router.routing import classify

if TYPE_CHECKING:
    from router.config import AppConfig
    from router.lifecycle import BackendManager

logger = logging.getLogger("llm-router.proxy")

# Module-level semaphore — initialized by init_semaphore() at app startup
_semaphore: asyncio.Semaphore | None = None


def init_semaphore(max_concurrent: int):
    """Call once at startup to set up the backpressure semaphore."""
    global _semaphore
    _semaphore = asyncio.Semaphore(max_concurrent)


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
    2. Determine backend (query param → explicit route prefix → classifier)
    3. Validate backend key
    4. Acquire semaphore (backpressure) or return 503
    5. Start backend if not running
    6. Proxy request, capturing TTFT and metrics
    7. Release semaphore
    """
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()

    # ── Parse body ────────────────────────────────────────────
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body", "request_id": request_id},
        )

    # ── Determine backend ─────────────────────────────────────
    backend_key = (
        request.query_params.get("backend")
        or classify(payload, manager.backends, config)
    )

    if backend_key not in manager.backends:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown backend '{backend_key}'",
                "valid_backends": list(manager.backends.keys()),
                "request_id": request_id,
            },
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
            status_code=503,
            content={
                "error": "Router overloaded — too many concurrent requests",
                "retry_after": config.proxy.queue_timeout_sec,
                "request_id": request_id,
            },
        )

    try:
        # ── Start backend if needed ───────────────────────────
        try:
            await manager.ensure_running(backend_key)
        except RuntimeError as e:
            cfg = manager.backends[backend_key]
            return JSONResponse(
                status_code=503,
                content={
                    "error": str(e),
                    "backend": backend_key,
                    "log": cfg.get("log", ""),
                    "request_id": request_id,
                },
            )

        manager.last_used[backend_key] = time.time()
        cfg = manager.backends[backend_key]
        target_url = f"http://localhost:{cfg['port']}/v1/{path}"
        is_stream = payload.get("stream", False)

        if is_stream:
            return await _proxy_stream(
                target_url, payload, path, backend_key, cfg,
                start_time, request_id, metrics_store, config,
            )
        else:
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
    try:
        async with httpx.AsyncClient(timeout=config.proxy.timeout_sec) as client:
            resp = await client.post(target_url, json=payload)

        total_ms = (time.time() - start_time) * 1000
        status = resp.status_code

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

        logger.info(f"[{backend_key}] /{path} → {status} ({total_ms:.0f}ms)")
        return JSONResponse(content=body, status_code=status)

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
) -> StreamingResponse:
    """Stream the response, capturing TTFT on the first chunk."""

    async def stream_with_metrics() -> AsyncIterator[bytes]:
        ttft_ms = None
        completion_tokens = 0

        try:
            async with httpx.AsyncClient(timeout=config.proxy.timeout_sec) as client:
                async with client.stream("POST", target_url, json=payload) as resp:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            if ttft_ms is None:
                                ttft_ms = (time.time() - start_time) * 1000
                            # Rough token count from SSE chunks
                            if b'"content"' in chunk or b'"text"' in chunk:
                                completion_tokens += 1
                            yield chunk

            total_ms = (time.time() - start_time) * 1000
            tps = (completion_tokens / (total_ms / 1000)) if total_ms > 0 else 0.0

            metrics_store.record(RequestRecord(
                request_id=request_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                backend_key=backend_key,
                engine=cfg.get("engine", ""),
                model_path=cfg.get("model", cfg.get("model_dir", "")),
                endpoint=path,
                prompt_tokens=0,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms or total_ms,
                total_latency_ms=total_ms,
                tokens_per_sec=round(tps, 2),
                status_code=200,
                error=None,
            ))
            logger.info(
                f"[{backend_key}] streaming /{path} done "
                f"(TTFT={ttft_ms:.0f}ms, total={total_ms:.0f}ms)"
            )

        except Exception as e:
            logger.warning(f"[{backend_key}] stream error: {e}")
            total_ms = (time.time() - start_time) * 1000
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
    4. Proxy to local backend
    5. Translate response back to Anthropic format
    """
    from router.anthropic_compat import (
        anthropic_to_openai, openai_to_anthropic,
        stream_openai_to_anthropic, model_to_backend,
    )

    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()

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

    # ── Determine backend ─────────────────────────────────────
    backend_key = request.query_params.get("backend")
    if not backend_key:
        # Try model name → tier mapping first
        backend_key = model_to_backend(original_model)
        # Fall back to keyword classifier on the translated messages
        if not backend_key or backend_key not in manager.backends:
            oai_for_classify = anthropic_to_openai(payload)
            from router.routing import classify
            backend_key = classify(oai_for_classify, manager.backends, config)

    if backend_key not in manager.backends:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {
                "type": "invalid_request_error",
                "message": f"No backend available for model '{original_model}'. "
                           f"Valid backends: {list(manager.backends.keys())}",
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
        # ── Start backend ─────────────────────────────────────
        try:
            await manager.ensure_running(backend_key)
        except RuntimeError as e:
            cfg = manager.backends[backend_key]
            return JSONResponse(
                status_code=503,
                content={"type": "error", "error": {
                    "type": "api_error",
                    "message": str(e),
                }, "log": cfg.get("log", "")},
            )

        manager.last_used[backend_key] = time.time()
        cfg        = manager.backends[backend_key]
        oai_payload = anthropic_to_openai(payload)
        target_url  = f"http://localhost:{cfg['port']}/v1/chat/completions"

        if is_stream:
            converter = stream_openai_to_anthropic(original_model)

            async def anthropic_stream() -> AsyncIterator[bytes]:
                ttft_ms = None
                try:
                    async with httpx.AsyncClient(timeout=config.proxy.timeout_sec) as client:
                        async with client.stream("POST", target_url, json=oai_payload) as resp:
                            raw_stream = resp.aiter_bytes()
                            async for chunk in converter(raw_stream):
                                if ttft_ms is None and chunk:
                                    ttft_ms = (time.time() - start_time) * 1000
                                yield chunk
                    total_ms = (time.time() - start_time) * 1000
                    metrics_store.record(RequestRecord(
                        request_id=request_id,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        backend_key=backend_key, engine=cfg.get("engine", ""),
                        model_path=cfg.get("model", cfg.get("model_dir", "")),
                        endpoint="anthropic/messages",
                        prompt_tokens=0, completion_tokens=0,
                        ttft_ms=ttft_ms or total_ms, total_latency_ms=total_ms,
                        tokens_per_sec=0.0, status_code=200, error=None,
                    ))
                    logger.info(f"[{backend_key}] anthropic stream done ({total_ms:.0f}ms)")
                except Exception as e:
                    logger.warning(f"[{backend_key}] anthropic stream error: {e}")

            return StreamingResponse(
                anthropic_stream(),
                media_type="text/event-stream",
                headers={"anthropic-version": "2023-06-01"},
            )

        else:
            async with httpx.AsyncClient(timeout=config.proxy.timeout_sec) as client:
                resp = await client.post(target_url, json=oai_payload)

            total_ms = (time.time() - start_time) * 1000

            try:
                oai_body = resp.json()
            except Exception:
                oai_body = {}

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
                status_code=resp.status_code, error=None,
            ))

            logger.info(f"[{backend_key}] anthropic /messages → {resp.status_code} ({total_ms:.0f}ms)")
            return JSONResponse(
                content=anthropic_body,
                status_code=200,
                headers={"anthropic-version": "2023-06-01"},
            )

    finally:
        _semaphore.release()
