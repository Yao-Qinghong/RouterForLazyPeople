"""
router/main.py — FastAPI application factory

Wire all modules together. Start with:
    uvicorn router.main:create_app --factory --host 0.0.0.0 --port 9001

Or via cli.py:
    python cli.py start
"""

import asyncio
import logging
import logging.handlers
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from router.config import load_config, AppConfig
from router.engines import available_engines, clear_engine_cache, ALL_ENGINES, is_engine_available
from router.lifecycle import BackendManager
from router.metrics import MetricsStore
from router.proxy import handle_proxy, handle_anthropic_proxy, init_semaphore
from router.registry import build_backend_registry

logger = logging.getLogger("llm-router")


# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────

def setup_logging(config: AppConfig):
    """Configure rotating file + console logging."""
    log_dir = config.logging.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "router.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.logging.log_max_bytes,
        backupCount=config.logging.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.router.log_level, logging.INFO))
    root.addHandler(file_handler)
    root.addHandler(console_handler)


# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────

def create_app(settings_path: Path | None = None) -> FastAPI:
    """
    FastAPI app factory. Called by uvicorn --factory flag.
    Loads config, wires up all modules, registers routes.
    """
    config = load_config(settings_path)
    setup_logging(config)

    # Ensure data directories exist
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "logs").mkdir(exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # ── Startup ───────────────────────────────────────────
        backends = build_backend_registry(config)
        manager = BackendManager(backends, config)
        metrics_store = MetricsStore(config)

        init_semaphore(config.proxy.max_concurrent_requests)

        engines = available_engines(config)
        manual = sum(1 for v in backends.values() if not v.get("auto_discovered"))
        discovered = sum(1 for v in backends.values() if v.get("auto_discovered"))

        # ── System info at startup ────────────────────────────
        from router.sysinfo import detect_system
        sys_info = detect_system(llama_bin=config.llama_bin)
        app.state.sys_info = sys_info

        gpu  = sys_info.get("gpu", {})
        cuda = sys_info.get("cuda", {})
        cpu  = sys_info.get("cpu", {})

        logger.info(f"LLM Router starting on :{config.router.port}")
        logger.info(f"  OS:  {sys_info.get('platform', {}).get('os')} "
                    f"{sys_info.get('platform', {}).get('arch')}")
        logger.info(f"  CPU: {cpu.get('model') or cpu.get('arch', 'unknown')} "
                    f"({cpu.get('cores')} cores)")
        if gpu.get("available"):
            for i, dev in enumerate(gpu.get("devices", [])):
                logger.info(f"  GPU {i}: {dev['name']} "
                            f"({dev['vram_total_gb']} GB VRAM, {dev['vram_free_gb']} GB free)")
            logger.info(f"  CUDA: {cuda.get('version', 'unknown')} | "
                        f"Driver: {gpu.get('driver_version', 'unknown')}")
        else:
            logger.info("  GPU: none detected (CPU-only mode)")
        logger.info(f"  Available engines: {engines}")
        logger.info(f"  Backends: {manual} manual + {discovered} auto-discovered = {len(backends)} total")

        # Store shared objects on app.state for route handlers
        app.state.manager = manager
        app.state.metrics_store = metrics_store
        app.state.config = config

        # Background tasks
        watchdog_task = asyncio.create_task(manager.idle_watchdog())
        flush_task = asyncio.create_task(metrics_store.flush_loop())

        yield

        # ── Shutdown ──────────────────────────────────────────
        watchdog_task.cancel()
        flush_task.cancel()
        await metrics_store.flush()   # flush any remaining metrics
        manager.stop_all()
        logger.info("LLM Router stopped")

    app = FastAPI(
        title="LLM Router",
        description="Local LLM routing proxy — lazy, efficient, beginner-friendly",
        version="2.0.0",
        lifespan=lifespan,
    )

    # ─────────────────────────────────────────────────────────
    # Routes
    # ─────────────────────────────────────────────────────────

    @app.get("/status", summary="Backend run-state")
    async def status(request: Request):
        return request.app.state.manager.status()

    @app.get("/backends", summary="List all registered backends")
    async def list_backends(request: Request):
        backends = request.app.state.manager.backends
        return {
            k: {
                "engine":          v.get("engine", "llama.cpp"),
                "description":     v.get("description", k),
                "port":            v["port"],
                "auto_discovered": v.get("auto_discovered", False),
                "tier":            v.get("tier"),
                "size_gb":         v.get("size_gb"),
            }
            for k, v in backends.items()
        }

    @app.get("/v1/models", summary="List backends as OpenAI-compatible model objects")
    async def list_models_openai(request: Request):
        """
        OpenAI-compatible GET /v1/models endpoint.

        Required by OpenClaw, LM Studio, Open WebUI, and most OpenAI-compatible
        clients so they can discover what 'models' are available.

        Each backend is listed as a model whose id is the backend key
        (e.g. 'fast', 'mid', 'deep'). Point your client at:
            baseUrl: http://localhost:9001/v1
        and select a model by its backend key.
        """
        import time as _time
        backends = request.app.state.manager.backends
        models = []
        for key, cfg in backends.items():
            models.append({
                "id":       key,
                "object":   "model",
                "created":  int(_time.time()),
                "owned_by": "local",
                "description": cfg.get("description", key),
                "engine":   cfg.get("engine", "llama.cpp"),
                "tier":     cfg.get("tier"),
                "size_gb":  cfg.get("size_gb"),
                "context_window": cfg.get("ctx_size"),
                "auto_discovered": cfg.get("auto_discovered", False),
            })
        return {"object": "list", "data": models}

    @app.get("/v1/models/{model_id}", summary="Get a single model by backend key")
    async def get_model_openai(model_id: str, request: Request):
        """OpenAI-compatible GET /v1/models/{id}."""
        import time as _time
        backends = request.app.state.manager.backends
        if model_id not in backends:
            raise HTTPException(404, f"Model '{model_id}' not found. "
                                     f"Available: {list(backends.keys())}")
        cfg = backends[model_id]
        return {
            "id":       model_id,
            "object":   "model",
            "created":  int(_time.time()),
            "owned_by": "local",
            "description": cfg.get("description", model_id),
            "engine":   cfg.get("engine", "llama.cpp"),
            "tier":     cfg.get("tier"),
            "context_window": cfg.get("ctx_size"),
        }

    @app.get("/engines", summary="Installed engine availability")
    async def list_engines(request: Request):
        cfg = request.app.state.config
        return {e: is_engine_available(e, cfg) for e in ALL_ENGINES}

    @app.get("/sysinfo", summary="Hardware, CUDA, engine versions, and install recommendations")
    async def sysinfo(request: Request):
        """
        Returns detected system info:
          - OS, CPU architecture, core count, RAM
          - GPU names, VRAM (total + free), driver version
          - CUDA version
          - Installed engine versions (llama.cpp, vLLM, SGLang, TRT-LLM, HF)
          - Recommended stable versions + install commands per engine
          - Any processes already occupying LLM ports (conflicts)

        Safe to call before or after backends are loaded.
        Cached from startup; call /rescan to refresh.
        """
        # Return startup-cached value (fast) or re-detect if not cached
        info = getattr(request.app.state, "sys_info", None)
        if info is None:
            from router.sysinfo import detect_system
            info = detect_system(llama_bin=request.app.state.config.llama_bin)
        return info

    @app.get("/metrics", summary="Per-backend performance summary")
    async def get_metrics(request: Request):
        """
        Returns aggregated benchmark stats from the last 1000 requests (in-memory).
        Includes: request count, avg/p50/p95 TTFT, avg latency, tokens/sec, error rate.
        """
        return request.app.state.metrics_store.summary()

    @app.get("/metrics/export", summary="Download metrics as CSV")
    async def export_metrics(request: Request):
        """Export full metrics history (last 365 days of JSONL files) as a CSV download."""
        store = request.app.state.metrics_store
        tmp = Path(tempfile.mktemp(suffix=".csv"))
        store.export_csv(tmp)
        return FileResponse(
            path=tmp,
            filename="llm-router-metrics.csv",
            media_type="text/csv",
        )

    @app.post("/rescan", summary="Re-discover models and reload config")
    async def rescan(request: Request):
        """
        Re-scan model directories and reload backends.yaml.
        Useful after adding a new model file without restarting the router.
        Running backends are left untouched.
        """
        cfg = request.app.state.config
        manager = request.app.state.manager
        clear_engine_cache()
        running_before = {k for k in manager.backends if manager.is_running(k)}
        new_backends = build_backend_registry(cfg)
        manager.update_registry(new_backends)
        discovered = sum(1 for v in new_backends.values() if v.get("auto_discovered"))
        return {
            "total":     len(new_backends),
            "discovered": discovered,
            "engines":   available_engines(cfg),
            "running":   list(running_before),
            "backends":  list(new_backends.keys()),
        }

    @app.post("/start/{key}", summary="Start a backend")
    async def start_backend(key: str, request: Request):
        manager = request.app.state.manager
        if key not in manager.backends:
            raise HTTPException(404, f"Unknown backend '{key}'")
        try:
            await manager.ensure_running(key)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        cfg = manager.backends[key]
        return {"status": "started", "backend": key, "port": cfg["port"]}

    @app.post("/stop/{key}", summary="Stop a backend")
    async def stop_backend(key: str, request: Request):
        manager = request.app.state.manager
        if key not in manager.backends:
            raise HTTPException(404, f"Unknown backend '{key}'")
        manager.stop(key)
        return {"status": "stopped", "backend": key}

    @app.post("/retune/{key}", summary="Force re-tune a TRT-LLM backend")
    async def retune(key: str, request: Request):
        from router.engines import ENGINE_TRTLLM
        from router.trt_tuner import TRTLLMTuner
        manager = request.app.state.manager
        cfg_app = request.app.state.config
        if key not in manager.backends:
            raise HTTPException(404, f"Unknown backend '{key}'")
        if manager.backends[key].get("engine") != ENGINE_TRTLLM:
            raise HTTPException(400, f"'{key}' is not a TRT-LLM backend")
        manager.stop(key)
        TRTLLMTuner(key, cfg_app).clear()
        try:
            await manager.ensure_running(key)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        return {"status": "retuned", "active_config": manager.active_configs.get(key)}

    @app.post("/v1/{path:path}", summary="OpenAI-compatible inference proxy")
    async def proxy(path: str, request: Request):
        """
        Main inference endpoint. Accepts OpenAI-compatible payloads.

        Backend selection (in priority order):
          1. ?backend=key  query parameter
          2. [route:key]   prefix in the first user message
          3. Automatic classification (token count + keywords)
        """
        return await handle_proxy(
            path=path,
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    @app.post("/anthropic/v1/messages", summary="Anthropic Messages API proxy (translates to local backend)")
    @app.post("/v1/messages", summary="Anthropic Messages API proxy (alias)")
    async def anthropic_proxy(request: Request):
        """
        Accepts Anthropic Messages API format and proxies to a local backend.

        Drop-in replacement for https://api.anthropic.com — just change the
        base URL in your Anthropic SDK config:

            import anthropic
            client = anthropic.Anthropic(
                api_key="any-string",
                base_url="http://localhost:9001/anthropic",
            )

        Backend selection (in priority order):
          1. ?backend=key  query parameter
          2. [route:key]   prefix in the first user message
          3. Claude model name → tier mapping (haiku→fast, sonnet→mid, opus→deep)
          4. Automatic classification (token count + keywords)
        """
        return await handle_anthropic_proxy(
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    # ── Global error handler — always return JSON ─────────────
    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled error on {request.method} {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type":  type(exc).__name__,
                "path":  str(request.url.path),
            },
        )

    return app


# Allow direct launch: python -m router.main
app = create_app()
