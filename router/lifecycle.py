"""
router/lifecycle.py — Backend lifecycle manager

Handles starting, stopping, health-checking, and idle-eviction
of backend inference server processes.
"""

import asyncio
import logging
import subprocess
import time
from typing import TYPE_CHECKING

import httpx

from router.engines import (
    ENGINE_LLAMA, ENGINE_TRTLLM,
    is_engine_available, health_url,
    build_llama_cmd, build_vllm_cmd, build_sglang_cmd,
    build_hf_cmd, build_trtllm_cmd,
)
from router.trt_tuner import TRTLLMTuner

if TYPE_CHECKING:
    from router.config import AppConfig

logger = logging.getLogger("llm-router.lifecycle")


class BackendManager:
    """
    Manages the lifecycle of all backend inference servers.

    Backends are started lazily (on first request) and stopped
    automatically after idle_timeout seconds of inactivity.
    """

    def __init__(self, backends: dict, config: "AppConfig"):
        self.backends = backends
        self.config = config
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_handles: dict[str, object] = {}
        self.last_used: dict[str, float] = {}
        self.active_configs: dict[str, dict] = {}
        self.starting: dict[str, asyncio.Lock] = {k: asyncio.Lock() for k in backends}

    def update_registry(self, new_backends: dict):
        """
        Swap in a new backend registry (called by /rescan).
        Adds locks for newly discovered backends; preserves running processes.
        """
        self.backends = new_backends
        for k in new_backends:
            if k not in self.starting:
                self.starting[k] = asyncio.Lock()

    def is_running(self, key: str) -> bool:
        proc = self.processes.get(key)
        return proc is not None and proc.poll() is None

    async def _wait_healthy(self, key: str) -> bool:
        """Poll the backend's /health endpoint until it responds or times out."""
        cfg = self.backends[key]
        url = health_url(cfg)
        deadline = time.time() + cfg["startup_wait"]
        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                try:
                    r = await client.get(url, timeout=2)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                if not self.is_running(key):
                    return False
                await asyncio.sleep(1)
        return False

    def _open_log(self, key: str):
        """Open (or reopen) the log file for a backend in append mode."""
        cfg = self.backends[key]
        log_path = cfg["log"]
        # Ensure log directory exists
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handle = open(log_path, "a")
        self.log_handles[key] = handle
        return handle

    def _build_cmd(self, key: str, trt_config: dict | None = None) -> list[str]:
        """Build the subprocess command for a backend."""
        cfg = self.backends[key]
        engine = cfg.get("engine", ENGINE_LLAMA)

        if engine == ENGINE_LLAMA:
            return build_llama_cmd(cfg, self.config)
        elif engine == "vllm":
            return build_vllm_cmd(cfg)
        elif engine == "sglang":
            return build_sglang_cmd(cfg)
        elif engine == ENGINE_TRTLLM:
            return build_trtllm_cmd(cfg, trt_config or {})
        elif engine == "huggingface":
            return build_hf_cmd(cfg)
        else:
            raise ValueError(f"Unknown engine '{engine}' for backend '{key}'")

    async def _start_process(self, key: str, trt_config: dict | None = None) -> bool:
        """Spawn the backend subprocess and wait for it to become healthy."""
        cfg = self.backends[key]
        engine = cfg.get("engine", ENGINE_LLAMA)

        if not is_engine_available(engine, self.config):
            logger.error(f"[{key}] Engine '{engine}' is not installed")
            return False

        try:
            cmd = self._build_cmd(key, trt_config)
        except ValueError as e:
            logger.error(f"[{key}] {e}")
            return False

        log_file = self._open_log(key)
        logger.info(f"[{key}] Starting ({engine}) on port {cfg['port']}...")

        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        self.processes[key] = proc
        self.last_used[key] = time.time()

        healthy = await self._wait_healthy(key)
        if not healthy:
            logger.warning(f"[{key}] Failed health check — killing PID {proc.pid}")
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            self.processes.pop(key, None)
            return False

        logger.info(f"[{key}] Ready on port {cfg['port']} (PID {proc.pid})")
        return True

    async def _start_trtllm_with_tuning(self, key: str):
        """Start a TRT-LLM backend, auto-tuning memory config if needed."""
        cfg = self.backends[key]
        tuner = TRTLLMTuner(key, self.config)

        saved = tuner.load_saved()
        if saved:
            ok = await self._start_process(key, trt_config=saved)
            if ok:
                self.active_configs[key] = saved
                return
            tuner.clear()

        base_config = cfg.get("trt_config", {})
        search_space = tuner.get_search_space(base_config)
        logger.info(f"[{key}] Auto-tuning TRT-LLM ({len(search_space)} configs to try)")

        for i, trial in enumerate(search_space, 1):
            logger.info(
                f"[{key}] Attempt {i}/{len(search_space)}: "
                f"input_len={trial.get('max_input_len')}, kv={trial.get('kv_cache_dtype')}"
            )
            ok = await self._start_process(key, trt_config=trial)
            if ok:
                tuner.save(trial)
                self.active_configs[key] = trial
                return
            await asyncio.sleep(3)

        raise RuntimeError(
            f"Backend '{key}' failed all {len(search_space)} tuning attempts. "
            f"Check {cfg['log']}"
        )

    async def ensure_running(self, key: str):
        """
        Ensure a backend is running. Start it if not.
        Uses a per-backend lock to prevent double-starts from concurrent requests.
        Raises RuntimeError if the backend cannot be started.
        """
        if self.is_running(key):
            self.last_used[key] = time.time()
            return

        # Create lock on demand (for backends added by rescan)
        if key not in self.starting:
            self.starting[key] = asyncio.Lock()

        async with self.starting[key]:
            if self.is_running(key):
                self.last_used[key] = time.time()
                return

            cfg = self.backends[key]
            engine = cfg.get("engine", ENGINE_LLAMA)

            if engine == ENGINE_TRTLLM:
                await self._start_trtllm_with_tuning(key)
            else:
                ok = await self._start_process(key)
                if not ok:
                    raise RuntimeError(
                        f"Backend '{key}' ({engine}) failed to start. "
                        f"Check {cfg['log']}"
                    )

    def stop(self, key: str):
        """Gracefully stop a backend (SIGTERM → SIGKILL after 10s)."""
        proc = self.processes.get(key)
        if proc and proc.poll() is None:
            logger.info(f"[{key}] Stopping (PID {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.processes.pop(key, None)
        self.last_used.pop(key, None)
        self.active_configs.pop(key, None)
        if key in self.log_handles:
            try:
                self.log_handles[key].close()
            except Exception:
                pass
            self.log_handles.pop(key, None)

    def stop_all(self):
        for key in list(self.processes.keys()):
            self.stop(key)

    async def idle_watchdog(self):
        """
        Background task: stop backends that have been idle longer
        than their configured idle_timeout.
        Runs every 30 seconds.
        """
        while True:
            await asyncio.sleep(30)
            now = time.time()
            for key, cfg in list(self.backends.items()):
                if not self.is_running(key):
                    continue
                idle_for = now - self.last_used.get(key, now)
                if idle_for > cfg["idle_timeout"]:
                    logger.info(f"[{key}] Idle for {idle_for:.0f}s — auto-stopping")
                    self.stop(key)

    def status(self) -> dict:
        """Return run-state for all registered backends."""
        now = time.time()
        out = {}
        for key, cfg in self.backends.items():
            running = self.is_running(key)
            idle = round(now - self.last_used[key]) if key in self.last_used else None
            entry = {
                "engine":          cfg.get("engine", ENGINE_LLAMA),
                "running":         running,
                "port":            cfg["port"],
                "idle_seconds":    idle,
                "idle_timeout":    cfg["idle_timeout"],
                "description":     cfg.get("description", key),
                "auto_discovered": cfg.get("auto_discovered", False),
                "pid":             self.processes[key].pid if running else None,
                "log":             cfg.get("log", ""),
            }
            if key in self.active_configs:
                entry["trt_active_config"] = self.active_configs[key]
            tune_file = self.config.data_dir / "tuning" / f"{key}.json"
            if tune_file.exists():
                entry["tuning_saved"] = True
            out[key] = entry
        return out
