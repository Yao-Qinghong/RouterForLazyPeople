from __future__ import annotations

"""
router/main.py — FastAPI application factory

Wire all modules together. Start with:
    uvicorn router.main:create_app --factory --host 0.0.0.0 --port 9001

Or via cli.py:
    python cli.py start
"""

import asyncio
import httpx
import json
import logging
import logging.handlers
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from router.config import load_config, AppConfig
from router.engines import available_engines, clear_engine_cache, ALL_ENGINES, is_engine_available
from router.lifecycle import BackendManager
from router.metrics import MetricsStore
from router.proxy import (
    build_model_aliases,
    handle_proxy,
    handle_anthropic_proxy,
    handle_gemini_proxy,
    init_semaphore,
    resolve_requested_model,
)
from router.registry import build_backend_registry

logger = logging.getLogger("llm-router")


# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for log aggregators (ELK, Loki, Datadog)."""
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging(config: AppConfig):
    """Configure rotating file + console logging. Optionally use JSON format."""
    log_dir = config.logging.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "router.log"

    if config.logging.json_format:
        formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        console_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.logging.log_max_bytes,
        backupCount=config.logging.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.router.log_level, logging.INFO))
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Setup audit logger (separate file)
    if config.audit.enabled:
        audit_dir = config.audit.log_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_dir / "audit.jsonl",
            maxBytes=config.logging.log_max_bytes,
            backupCount=config.logging.log_backup_count,
            encoding="utf-8",
        )
        audit_handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger = logging.getLogger("llm-router.audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False


# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────

def _quality_warning(cfg: dict) -> str:
    """
    Return a short warning string if a backend is likely too small for
    code generation or agentic (tool-use) tasks. Empty string = no warning.

    Thresholds (conservative — better to warn than to silently produce bad output):
      Dense model  < 5 GB  → probably < 4-5B params → unreliable for code/agents
      MoE model    < 3B active → same concern
    """
    from router.discovery import _moe_active_params
    desc = cfg.get("description", "")
    size_gb = cfg.get("size_gb")

    active = _moe_active_params(desc)
    if active is not None:
        if active < 3:
            return "too small for reliable code/tool-use (< 3B active params)"
    elif size_gb is not None and size_gb < 5:
        return "too small for reliable code/tool-use (< 5 GB)"
    return ""


def _log_routing_guidance(backends: dict, log) -> None:
    """Log a one-time note explaining how requests are routed to tiers."""
    has_deep = any(v.get("tier") == "deep" for v in backends.values())
    has_mid  = any(v.get("tier") == "mid"  for v in backends.values())

    log.info("  Routing rules:")
    log.info("    tool_use / function_calling  → deep  (agentic tasks always need the best model)")
    log.info("    long prompt / deep keywords  → deep")
    log.info("    code keywords / medium prompt → mid")
    log.info("    everything else              → fast")

    if not has_deep and not has_mid:
        log.warning(
            "  No mid or deep backends available — all requests will go to fast tier.\n"
            "  Code generation and agentic tasks may produce poor results.\n"
            "  Add a larger model to config/backends.yaml or start a larger Ollama model."
        )
    elif not has_deep:
        log.warning(
            "  No deep backend available — tool-use and agentic requests will fall back to mid.\n"
            "  For best agentic results, add a 70B+ model."
        )


def _apply_measured_tiers(backends: dict, bench_results: dict) -> None:
    """Promote backend tiers to match measured TG speed from benchmarks.

    When a backend's measured tier (from bench) differs from its configured
    tier, trust the benchmark — a model that does 49 tok/s belongs in fast
    regardless of its file size.
    """
    for key, result in bench_results.items():
        measured = result.get("tier_measured")
        if measured and key in backends:
            old = backends[key].get("tier")
            if old != measured:
                logger.info(
                    f"[{key}] Tier updated {old} → {measured} "
                    f"(measured {result.get('tg_tok_s', '?')} tok/s)"
                )
                backends[key]["tier"] = measured


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

        # ── Tier table ────────────────────────────────────────
        if backends:
            logger.info(
                f"  Tier ranking: fast < {config.tier_thresholds.fast:.0f} GB "
                f"< mid < {config.tier_thresholds.mid:.0f} GB < deep  "
                f"(set in config/settings.yaml → tier_thresholds_gb)"
            )
            for tier in ("fast", "mid", "deep", None):
                tier_backends = [
                    (k, v) for k, v in backends.items()
                    if v.get("tier") == tier
                ]
                if not tier_backends:
                    continue
                label = tier if tier else "untiered"
                for k, v in tier_backends:
                    desc = v.get("description", k)
                    warn = _quality_warning(v)
                    suffix = f"  ⚠ {warn}" if warn else ""
                    logger.info(f"  [{label:>5}]  {k:<22} {desc}{suffix}")

            _log_routing_guidance(backends, logger)
        else:
            logger.warning(
                "No backends found. Quick options:\n"
                "  • Start LM Studio or Ollama — the router will detect them automatically\n"
                "  • Place GGUF files in ~/models and restart\n"
                "  • Edit config/backends.yaml to add backends manually"
            )

        if config.model_aliases:
            logger.info(f"  Model aliases: {config.model_aliases}")
        if config.auth.enabled:
            logger.info(f"  Auth: enabled ({len(config.auth.api_keys)} keys)")
        if config.rate_limit.enabled:
            logger.warning(
                "Rate limiting is configured but not enforced in this release. "
                "Use an external proxy or gateway for request throttling."
            )

        # Inject cached benchmark results into the router so best-engine
        # selection uses measured TG speed from previous bench runs.
        # Also promote backend tiers to match measured speed.
        from router.benchmark import load_all_results
        from router.routing import set_benchmark_results
        _bench_results = load_all_results(config)
        set_benchmark_results(_bench_results)
        _apply_measured_tiers(backends, _bench_results)
        if _bench_results:
            logger.info(f"  Benchmarks loaded: {len(_bench_results)} backend(s) have measured TG speed")
            for bkey, br in _bench_results.items():
                if br.get("tg_tok_s"):
                    logger.info(
                        f"    {bkey:<22} TG={br['tg_tok_s']:.0f} tok/s  "
                        f"PP={br.get('pp_tok_s', 0):.0f} tok/s  "
                        f"(tier assigned={br.get('tier_assigned')})"
                    )
        else:
            logger.info(
                "  No benchmark data found. Run: python cli.py bench\n"
                "  Without benchmarks, routing picks best engine by capability rank "
                "(trt-llm > trt-llm-docker > vllm > sglang > llama.cpp > openai > ollama)"
            )

        # Store shared objects on app.state for route handlers
        app.state.manager = manager
        app.state.metrics_store = metrics_store
        app.state.config = config

        # Background tasks
        watchdog_task = asyncio.create_task(manager.idle_watchdog())
        flush_task = asyncio.create_task(metrics_store.flush_loop())

        # Preload backends if configured
        if config.preload:
            asyncio.create_task(manager.preload(config.preload))

        yield

        # ── Shutdown ──────────────────────────────────────────
        watchdog_task.cancel()
        flush_task.cancel()
        await metrics_store.flush()   # flush any remaining metrics
        manager.stop_all()
        logger.info("LLM Router stopped")

    app = FastAPI(
        title="LLM Router",
        description="Local LLM routing proxy — lazy, efficient, beginner-friendly. "
                    "Supports OpenAI, Anthropic, and Gemini API formats.",
        version="3.0.0",
        lifespan=lifespan,
    )

    # ─────────────────────────────────────────────────────────
    # Middleware
    # ─────────────────────────────────────────────────────────

    # CORS
    if config.cors.enabled:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors.allow_origins,
            allow_credentials=config.cors.allow_credentials,
            allow_methods=config.cors.allow_methods,
            allow_headers=config.cors.allow_headers,
        )

    # Authentication
    if config.auth.enabled:
        from router.auth import AuthMiddleware
        app.add_middleware(AuthMiddleware, auth_config=config.auth)

    # ─────────────────────────────────────────────────────────
    # Routes — Status & Discovery
    # ─────────────────────────────────────────────────────────

    @app.get("/health", summary="Health check")
    async def health():
        return {"status": "ok"}

    @app.get("/status", summary="Backend run-state")
    async def status(request: Request):
        mgr     = request.app.state.manager
        cfg     = request.app.state.config
        state   = mgr.status()
        from router.benchmark import load_result
        for key, info in state.items():
            bench = load_result(key, cfg)
            if bench:
                info["bench_pp_tok_s"]  = bench.get("pp_tok_s")
                info["bench_tg_tok_s"]  = bench.get("tg_tok_s")
                info["bench_ttft_ms"]   = bench.get("ttft_ms")
                info["bench_tier_measured"] = bench.get("tier_measured")
                info["bench_mismatch"]  = bench.get("tier_mismatch", False)
                info["bench_timestamp"] = bench.get("timestamp")
        return state

    @app.get("/benchmarks", summary="Cached benchmark results for all backends")
    async def benchmarks(request: Request):
        from router.benchmark import load_all_results
        return load_all_results(request.app.state.config)

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
        aliases = build_model_aliases(
            backends,
            request.app.state.config.model_aliases,
            request.app.state.manager,
        )
        models = []
        seen_ids: set[str] = set()
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
            seen_ids.add(key)
        # Also list model aliases as models
        for alias, backend_key in aliases.items():
            if backend_key in backends and alias not in seen_ids:
                cfg = backends[backend_key]
                models.append({
                    "id":       alias,
                    "object":   "model",
                    "created":  int(_time.time()),
                    "owned_by": "local",
                    "description": f"Alias for {backend_key}",
                    "engine":   cfg.get("engine", "llama.cpp"),
                    "tier":     cfg.get("tier"),
                    "alias_for": backend_key,
                })
                seen_ids.add(alias)
        return {"object": "list", "data": models}

    @app.get("/v1/models/{model_id:path}", summary="Get a single model by backend key")
    async def get_model_openai(model_id: str, request: Request):
        """OpenAI-compatible GET /v1/models/{id}."""
        import time as _time
        backends = request.app.state.manager.backends
        resolved = resolve_requested_model(
            model_id,
            backends,
            request.app.state.config.model_aliases,
            request.app.state.manager,
        ) or model_id
        if resolved not in backends:
            raise HTTPException(404, f"Model '{model_id}' not found. "
                                     f"Available: {list(backends.keys())}")
        cfg = backends[resolved]
        return {
            "id":       model_id,
            "object":   "model",
            "created":  int(_time.time()),
            "owned_by": "local",
            "description": cfg.get("description", resolved),
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
          - Installed engine versions (llama.cpp, vLLM, SGLang, TRT-LLM, HF, Ollama)
          - Recommended stable versions + install commands per engine
          - Any processes already occupying LLM ports (conflicts)

        Safe to call before or after backends are loaded.
        Cached from startup; call /rescan to refresh.
        """
        info = getattr(request.app.state, "sys_info", None)
        if info is None:
            from router.sysinfo import detect_system
            info = detect_system(llama_bin=request.app.state.config.llama_bin)
        return info

    # ─────────────────────────────────────────────────────────
    # Routes — Metrics
    # ─────────────────────────────────────────────────────────

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

    @app.get("/metrics/prometheus", summary="Prometheus-compatible metrics")
    async def prometheus_metrics(request: Request):
        """Prometheus text exposition format, scrapeable by Prometheus/Grafana."""
        text = request.app.state.metrics_store.prometheus()
        return PlainTextResponse(content=text, media_type="text/plain; version=0.0.4")

    # ─────────────────────────────────────────────────────────
    # Routes — Control
    # ─────────────────────────────────────────────────────────

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
        # Refresh benchmark data so new/renamed backends get correct routing weights
        from router.benchmark import load_all_results
        from router.routing import set_benchmark_results
        _bench_results = load_all_results(cfg)
        set_benchmark_results(_bench_results)
        _apply_measured_tiers(manager.backends, _bench_results)
        discovered = sum(1 for v in new_backends.values() if v.get("auto_discovered"))
        return {
            "total":     len(new_backends),
            "discovered": discovered,
            "engines":   available_engines(cfg),
            "running":   list(running_before),
            "backends":  list(new_backends.keys()),
        }

    @app.post("/reload-config", summary="Reload settings.yaml without restart")
    async def reload_config(request: Request):
        """Hot-reload settings.yaml. Does NOT restart running backends."""
        try:
            app_config = request.app.state.config
            new_config = load_config(app_config.settings_file, app_config.backends_file)
            # Update mutable config references
            app_config.routing = new_config.routing
            app_config.proxy = new_config.proxy
            app_config.model_aliases = new_config.model_aliases
            app_config.idle_timeouts = new_config.idle_timeouts
            app_config.tier_thresholds = new_config.tier_thresholds
            app_config.rate_limit = new_config.rate_limit
            if new_config.rate_limit.enabled:
                logger.warning(
                    "Rate limiting was enabled in settings.yaml, but enforcement is not "
                    "implemented. Use an external proxy or gateway for throttling."
                )
            logger.info("Config hot-reloaded from settings.yaml")
            return {"status": "reloaded", "model_aliases": new_config.model_aliases}
        except Exception as e:
            raise HTTPException(500, f"Config reload failed: {e}")

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

    @app.post("/restart/{key}", summary="Restart a backend")
    async def restart_backend(key: str, request: Request):
        """Stop and restart a backend gracefully."""
        manager = request.app.state.manager
        if key not in manager.backends:
            raise HTTPException(404, f"Unknown backend '{key}'")
        try:
            await manager.restart(key)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        cfg = manager.backends[key]
        return {"status": "restarted", "backend": key, "port": cfg["port"]}

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

    # ─────────────────────────────────────────────────────────
    # Routes — Inference Proxies
    # ─────────────────────────────────────────────────────────

    @app.post("/v1/chat/completions", summary="OpenAI-compatible chat completions")
    async def chat_completions(request: Request):
        """Explicit chat completions route (also matched by /v1/{path})."""
        return await handle_proxy(
            path="chat/completions",
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    @app.post("/v1/completions", summary="OpenAI-compatible legacy completions")
    async def completions(request: Request):
        """Legacy completions endpoint — proxies to backend's /v1/completions."""
        return await handle_proxy(
            path="completions",
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    @app.post("/v1/embeddings", summary="OpenAI-compatible embeddings")
    async def embeddings(request: Request):
        """Embeddings endpoint — proxies to backend's /v1/embeddings."""
        return await handle_proxy(
            path="embeddings",
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    @app.post("/v1/{path:path}", summary="OpenAI-compatible inference proxy")
    async def proxy(path: str, request: Request):
        """
        Catch-all inference endpoint. Accepts OpenAI-compatible payloads.

        Backend selection (in priority order):
          1. ?backend=key  query parameter
          2. Model alias → backend mapping
          3. [route:key]   prefix in the first user message
          4. Automatic classification (token count + keywords)
        """
        return await handle_proxy(
            path=path,
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    @app.post("/anthropic/v1/messages", summary="Anthropic Messages API proxy")
    @app.post("/v1/messages", summary="Anthropic Messages API proxy (alias)")
    async def anthropic_proxy(request: Request):
        """
        Accepts Anthropic Messages API format and proxies to a local backend.
        Supports tool use / function calling.

        Drop-in replacement for https://api.anthropic.com — just change the
        base URL in your Anthropic SDK config:

            import anthropic
            client = anthropic.Anthropic(
                api_key="any-string",
                base_url="http://localhost:9001/anthropic",
            )

        Backend selection (in priority order):
          1. ?backend=key  query parameter
          2. Model alias → backend mapping
          3. Claude model name → tier mapping (haiku→fast, sonnet→mid, opus→deep)
          4. Automatic classification (token count + keywords)
        """
        return await handle_anthropic_proxy(
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    @app.post("/gemini/v1beta/models/{model}:generateContent",
              summary="Gemini generateContent proxy")
    async def gemini_generate(model: str, request: Request):
        """Google Gemini API compatibility — non-streaming."""
        return await handle_gemini_proxy(
            model=model, is_stream=False,
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    @app.post("/gemini/v1beta/models/{model}:streamGenerateContent",
              summary="Gemini streamGenerateContent proxy")
    async def gemini_stream(model: str, request: Request):
        """Google Gemini API compatibility — streaming."""
        return await handle_gemini_proxy(
            model=model, is_stream=True,
            request=request,
            manager=request.app.state.manager,
            metrics_store=request.app.state.metrics_store,
            config=request.app.state.config,
        )

    # ─────────────────────────────────────────────────────────
    # WebSocket streaming
    # ─────────────────────────────────────────────────────────

    @app.websocket("/v1/chat/completions/ws")
    async def websocket_chat(ws: WebSocket):
        """
        WebSocket endpoint for streaming chat completions.
        Send a JSON payload, receive streamed chunks as JSON messages.
        """
        await ws.accept()
        try:
            while True:
                data = await ws.receive_json()
                data["stream"] = True

                manager = ws.app.state.manager
                config = ws.app.state.config
                from router.routing import classify as ws_classify

                backend_key = data.pop("backend", None) or ws_classify(data, manager.backends, config)
                if backend_key not in manager.backends:
                    await ws.send_json({"error": f"Unknown backend '{backend_key}'"})
                    continue

                try:
                    await manager.ensure_running(backend_key)
                except RuntimeError as e:
                    await ws.send_json({"error": str(e)})
                    continue

                cfg = manager.backends[backend_key]
                manager.last_used[backend_key] = __import__("time").time()
                target_url = f"http://localhost:{cfg['port']}/v1/chat/completions"

                try:
                    async with httpx.AsyncClient(timeout=config.proxy.timeout_sec) as client:
                        async with client.stream("POST", target_url, json=data) as resp:
                            async for chunk in resp.aiter_bytes():
                                for line in chunk.decode("utf-8", errors="replace").splitlines():
                                    line = line.strip()
                                    if line.startswith("data:"):
                                        data_str = line[5:].strip()
                                        if data_str == "[DONE]":
                                            await ws.send_json({"done": True})
                                        else:
                                            try:
                                                await ws.send_json(json.loads(data_str))
                                            except Exception:
                                                pass
                except Exception as e:
                    await ws.send_json({"error": str(e)})

        except WebSocketDisconnect:
            pass

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("router.main:create_app", factory=True, host="0.0.0.0", port=9001)
