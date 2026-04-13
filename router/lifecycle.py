from __future__ import annotations

"""
router/lifecycle.py — Backend lifecycle manager

Handles starting, stopping, health-checking, idle-eviction,
restarting, and preloading of backend inference server processes.
"""

import asyncio
import logging
import os
import signal
import socket
import subprocess
import time
from typing import TYPE_CHECKING

import httpx

from router.engines import (
    ENGINE_LLAMA, ENGINE_TRTLLM, ENGINE_TRTLLM_DOCKER, ENGINE_OLLAMA, ENGINE_OPENAI,
    is_engine_available, health_url,
    build_llama_cmd, build_vllm_cmd, build_sglang_cmd,
    build_hf_cmd, build_trtllm_cmd, build_trtllm_docker_cmd, build_ollama_cmd,
    resolve_trtllm_docker_config,
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
    Supports preloading specific backends on startup.
    """

    def __init__(self, backends: dict, config: "AppConfig"):
        self.backends = backends
        self.config = config
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_handles: dict[str, object] = {}
        self.last_used: dict[str, float] = {}
        self.active_configs: dict[str, dict] = {}
        self.starting: dict[str, asyncio.Lock] = {k: asyncio.Lock() for k in backends}
        self._registry_lock: asyncio.Lock | None = None

    async def update_registry(self, new_backends: dict):
        """
        Swap in a new backend registry (called by /rescan).
        Adds locks for newly discovered backends; preserves running processes.
        Acquires _registry_lock to prevent races with in-flight requests
        that snapshot self.backends.
        """
        if self._registry_lock is None:
            self._registry_lock = asyncio.Lock()
        async with self._registry_lock:
            self.backends = new_backends
            for k in new_backends:
                if k not in self.starting:
                    self.starting[k] = asyncio.Lock()

    def snapshot_backends(self) -> dict:
        """Return a reference to the current backends dict.

        Call once at the start of a request handler and use the returned
        dict for all routing/lookup within that request.  Because Python
        dict assignment is atomic (single STORE_ATTR bytecode),
        this is safe without the lock — the caller simply gets a
        consistent view that won't change mid-request even if /rescan
        swaps ``self.backends`` concurrently.
        """
        return self.backends

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

    def _close_log(self, key: str):
        """Close the log file handle for *key* if one is open."""
        handle = self.log_handles.pop(key, None)
        if handle:
            try:
                handle.close()
            except Exception:
                pass

    def _open_log(self, key: str):
        """Open (or reopen) the log file for a backend in append mode."""
        # Close any existing handle first (prevents leak on retry)
        self._close_log(key)
        cfg = self.backends[key]
        log_path = cfg["log"]
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handle = open(log_path, "a")
        self.log_handles[key] = handle
        return handle

    def _build_cmd(self, key: str, trt_config: dict | None = None) -> list[str]:
        """Build the subprocess command for a backend."""
        cfg = self.backends[key]
        engine = cfg.get("engine", ENGINE_LLAMA)

        if engine == ENGINE_OPENAI:
            raise ValueError(f"Engine 'openai' uses an external server — no command to build")

        if engine == ENGINE_LLAMA:
            return build_llama_cmd(cfg, self.config)
        elif engine == "vllm":
            return build_vllm_cmd(cfg)
        elif engine == "sglang":
            return build_sglang_cmd(cfg)
        elif engine == ENGINE_TRTLLM:
            return build_trtllm_cmd(cfg, trt_config or {})
        elif engine == ENGINE_TRTLLM_DOCKER:
            return build_trtllm_docker_cmd(key, cfg, self.config)
        elif engine == "huggingface":
            return build_hf_cmd(cfg)
        elif engine == ENGINE_OLLAMA:
            return build_ollama_cmd(cfg)
        else:
            raise ValueError(f"Unknown engine '{engine}' for backend '{key}'")

    def _docker_config(self, key: str) -> dict:
        return resolve_trtllm_docker_config(key, self.backends[key], self.config)

    def _remove_docker_container(self, container_name: str):
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _append_docker_logs(self, key: str, container_name: str, lines: int = 200):
        log_handle = self.log_handles.get(key) or self._open_log(key)
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), container_name],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            payload = (result.stdout or "") + (result.stderr or "")
            if payload.strip():
                log_handle.write(
                    "\n\n=== docker logs tail "
                    f"({container_name}, last {lines} lines) ===\n"
                )
                log_handle.write(payload)
                if not payload.endswith("\n"):
                    log_handle.write("\n")
                log_handle.flush()
        except Exception as exc:
            log_handle.write(f"\n\n[docker logs unavailable for {container_name}: {exc}]\n")
            log_handle.flush()

    async def _start_process(self, key: str, trt_config: dict | None = None) -> bool:
        """Spawn the backend subprocess and wait for it to become healthy."""
        cfg = self.backends[key]
        engine = cfg.get("engine", ENGINE_LLAMA)

        # External OpenAI-compatible servers are already running — just verify
        if engine == ENGINE_OPENAI:
            return await self._start_external_backend(key)

        # Ollama backends connect to an existing Ollama server
        if engine == ENGINE_OLLAMA:
            return await self._start_ollama_backend(key)

        if engine == ENGINE_TRTLLM_DOCKER:
            if not is_engine_available(engine, self.config):
                logger.error(f"[{key}] Engine '{engine}' is not installed")
                return False
            return await self._start_trtllm_docker_backend(key)

        if not is_engine_available(engine, self.config):
            logger.error(f"[{key}] Engine '{engine}' is not installed")
            return False

        # Clear any process squatting on our port (zombie from previous start, or unrelated)
        self._kill_port(cfg["port"], key)

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
            self._close_log(key)
            return False

        logger.info(f"[{key}] Ready on port {cfg['port']} (PID {proc.pid})")
        return True

    async def _start_trtllm_docker_backend(self, key: str) -> bool:
        """Start a managed Docker-backed TRT-LLM server and wait for /v1/models."""
        cfg = self.backends[key]
        docker_cfg = self._docker_config(key)
        os.makedirs(docker_cfg["hf_cache_dir"], exist_ok=True)
        os.makedirs(docker_cfg["log_dir"], exist_ok=True)

        # Clear any process on the target port (stale container or zombie)
        self._kill_port(cfg["port"], key)

        cmd = self._build_cmd(key)
        log_file = self._open_log(key)
        self._remove_docker_container(docker_cfg["container_name"])
        logger.info(
            f"[{key}] Starting ({cfg.get('engine')}) on port {cfg['port']} "
            f"with container {docker_cfg['container_name']}..."
        )
        result = subprocess.run(cmd, stdout=log_file, stderr=log_file, check=False)
        if result.returncode != 0:
            logger.warning(
                f"[{key}] Docker launcher exited with code {result.returncode}"
            )
            self._append_docker_logs(key, docker_cfg["container_name"])
            self._close_log(key)
            return False

        self.processes[key] = _DockerSentinel(docker_cfg["container_name"])
        self.last_used[key] = time.time()

        healthy = await self._wait_healthy(key)
        if not healthy:
            logger.warning(
                f"[{key}] Failed health check for container "
                f"{docker_cfg['container_name']} on port {cfg['port']}"
            )
            self._append_docker_logs(key, docker_cfg["container_name"])
            self._remove_docker_container(docker_cfg["container_name"])
            self.processes.pop(key, None)
            self.last_used.pop(key, None)
            self._close_log(key)
            return False

        logger.info(
            f"[{key}] Ready on port {cfg['port']} "
            f"(container {docker_cfg['container_name']})"
        )
        return True

    async def _start_external_backend(self, key: str) -> bool:
        """
        Verify that an external OpenAI-compatible server (e.g. LM Studio)
        is reachable. No subprocess is started — the server is already running.
        """
        cfg = self.backends[key]
        port = cfg.get("port", 1234)
        url = f"http://localhost:{port}/v1/models"
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=5)
                if r.status_code == 200:
                    self.processes[key] = _ExternalSentinel(port)
                    self.last_used[key] = time.time()
                    logger.info(f"[{key}] Connected to external server on port {port}")
                    return True
        except Exception as e:
            logger.error(f"[{key}] Cannot reach external server on port {port}: {e}")
        return False

    async def _start_ollama_backend(self, key: str) -> bool:
        """
        Start an Ollama-managed backend. Ollama runs its own server;
        we just need to ensure the model is loaded.
        """
        cfg = self.backends[key]
        model = cfg.get("model", "")
        port = cfg.get("port", 11434)
        logger.info(f"[{key}] Loading Ollama model '{model}' on port {port}...")

        # Check if Ollama is running
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://localhost:{port}/api/tags", timeout=5)
                if r.status_code != 200:
                    logger.error(f"[{key}] Ollama server not running on port {port}")
                    return False
        except Exception:
            logger.error(f"[{key}] Cannot reach Ollama on port {port}")
            return False

        # Pull/load the model via Ollama API
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                r = await client.post(
                    f"http://localhost:{port}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": f"{cfg.get('idle_timeout', 300)}s"},
                )
                if r.status_code == 200:
                    self.last_used[key] = time.time()
                    # Store a sentinel so is_running() returns True
                    self.processes[key] = _OllamaSentinel(port)
                    logger.info(f"[{key}] Ollama model '{model}' loaded")
                    return True
        except Exception as e:
            logger.error(f"[{key}] Failed to load Ollama model: {e}")

        return False

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

    def _kill_port(self, port: int, key: str) -> bool:
        """
        Kill any process currently listening on *port* so we can bind it.
        This clears both port conflicts from unrelated programs and zombie
        processes left over from a previous failed backend start.
        Returns True if something was killed (caller should sleep briefly).
        """
        # Quick check: is the port actually occupied?
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return False  # Port is free — nothing to do

        logger.info(f"[{key}] Port {port} already in use — finding and killing occupying process")
        killed_any = False

        # Try lsof first (macOS + most Linux distributions)
        try:
            r = subprocess.run(
                ["lsof", "-t", "-i", f"TCP:{port}", "-s", "TCP:LISTEN"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            for pid_str in r.stdout.split():
                if pid_str.isdigit():
                    pid = int(pid_str)
                    try:
                        os.kill(pid, signal.SIGKILL)
                        logger.info(f"[{key}] Killed PID {pid} (was on port {port})")
                        killed_any = True
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        logger.warning(
                            f"[{key}] Permission denied killing PID {pid} on port {port} — "
                            f"stop that process manually and retry"
                        )
        except FileNotFoundError:
            # lsof not available — fall back to fuser (Linux)
            try:
                result = subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True, timeout=5, check=False,
                )
                killed_any = result.returncode == 0
                if killed_any:
                    logger.info(f"[{key}] fuser killed process on port {port}")
            except FileNotFoundError:
                logger.warning(
                    f"[{key}] Cannot free port {port}: neither lsof nor fuser is available. "
                    f"Kill the occupying process manually."
                )

        if killed_any:
            time.sleep(1.0)  # Give the OS time to release the port
        return killed_any

    def _evict_for_vram(self, needed_gb: float, exclude_key: str = "") -> bool:
        """
        Evict idle backends until needed_gb of VRAM is free.
        Evicts in last_used ascending order (oldest idle first).
        Returns True if enough space is available, False if not.
        Skips VRAM logic entirely when no GPU info is available.
        """
        from router.sysinfo import query_free_vram
        vram = query_free_vram()
        if vram is None:
            return True  # no GPU info — skip VRAM logic

        free_gb, _total_gb = vram
        if free_gb >= needed_gb:
            return True

        # Collect running backends sorted by last_used (oldest first)
        running = []
        for k in self.backends:
            if k == exclude_key or not self.is_running(k):
                continue
            running.append((self.last_used.get(k, 0), k))
        running.sort()

        for _, k in running:
            if free_gb >= needed_gb:
                return True
            proc = self.processes.get(k)
            if isinstance(proc, _ExternalSentinel):
                logger.warning(
                    f"[{k}] Cannot evict — external server on port "
                    f"{self.backends[k].get('port')} is not managed by the router. "
                    f"Stop it manually to free VRAM."
                )
                continue
            evict_gb = self.backends[k].get("vram_estimate_gb")
            logger.info(f"[{k}] Evicting to free ~{evict_gb or '?'} GB VRAM")
            self.stop(k)
            vram = query_free_vram()
            if vram is not None:
                free_gb = vram[0]
            elif evict_gb:
                free_gb += evict_gb

        return free_gb >= needed_gb

    async def ensure_running(self, key: str):
        """
        Ensure a backend is running. Start it if not.
        Uses a per-backend lock to prevent double-starts from concurrent requests.
        Checks VRAM availability and evicts idle backends if needed.
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

            # VRAM check — evict idle backends if needed to make room
            vram_needed = cfg.get("vram_estimate_gb")
            if vram_needed is not None:
                if not self._evict_for_vram(vram_needed, exclude_key=key):
                    from router.sysinfo import query_free_vram
                    vram = query_free_vram()
                    free = vram[0] if vram else 0
                    raise RuntimeError(
                        f"Not enough VRAM to start '{key}': need ~{vram_needed:.1f} GB, "
                        f"only {free:.1f} GB available after evicting idle backends"
                    )

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
        if proc and isinstance(proc, _ExternalSentinel):
            # External servers are not ours to stop — just disconnect
            pass
        elif proc and isinstance(proc, _DockerSentinel):
            self._remove_docker_container(proc.container_name)
        elif proc and isinstance(proc, _OllamaSentinel):
            # Ollama backends: unload via API
            self._unload_ollama(key)
        elif proc and proc.poll() is None:
            logger.info(f"[{key}] Stopping (PID {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        elif self.backends.get(key, {}).get("engine") == ENGINE_TRTLLM_DOCKER:
            self._remove_docker_container(self._docker_config(key)["container_name"])
        self.processes.pop(key, None)
        self.last_used.pop(key, None)
        self.active_configs.pop(key, None)
        self._close_log(key)

    def _unload_ollama(self, key: str):
        """Tell Ollama to unload a model to free memory."""
        cfg = self.backends.get(key, {})
        model = cfg.get("model", "")
        port = cfg.get("port", 11434)
        try:
            import httpx as _httpx
            _httpx.post(
                f"http://localhost:{port}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=10,
            )
            logger.info(f"[{key}] Ollama model '{model}' unloaded")
        except Exception:
            pass

    def stop_all(self):
        for key in list(self.processes.keys()):
            self.stop(key)

    async def restart(self, key: str):
        """Stop and restart a backend."""
        self.stop(key)
        await asyncio.sleep(1)
        await self.ensure_running(key)

    async def preload(self, keys: list[str]):
        """Start backends in parallel at startup (non-blocking)."""
        tasks = []
        for key in keys:
            if key in self.backends:
                logger.info(f"[{key}] Preloading...")
                tasks.append(self._preload_one(key))
            else:
                logger.warning(f"[{key}] Cannot preload — not in registry")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _preload_one(self, key: str):
        try:
            await self.ensure_running(key)
        except Exception as e:
            logger.error(f"[{key}] Preload failed: {e}")

    async def idle_watchdog(self):
        """
        Background task: stop backends that have been idle longer
        than their configured idle_timeout.
        Also evicts the oldest backend under VRAM pressure (< 20% free).
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

            # VRAM pressure: if free < 20% of total, evict oldest backend
            from router.sysinfo import query_free_vram
            vram = query_free_vram()
            if vram is not None:
                free_gb, total_gb = vram
                if total_gb > 0 and (free_gb / total_gb) < 0.20:
                    oldest_key = None
                    oldest_time = float('inf')
                    for key in self.backends:
                        if not self.is_running(key):
                            continue
                        lu = self.last_used.get(key, 0)
                        if lu < oldest_time:
                            oldest_time = lu
                            oldest_key = key
                    if oldest_key:
                        logger.info(
                            f"[{oldest_key}] VRAM pressure ({free_gb:.1f}/{total_gb:.1f} GB free) "
                            f"— evicting oldest backend"
                        )
                        self.stop(oldest_key)

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
                "tier":            cfg.get("tier"),
                "pid":             self.processes[key].pid if running and not isinstance(self.processes.get(key), (_OllamaSentinel, _ExternalSentinel, _DockerSentinel)) else None,
                "log":             cfg.get("log", ""),
            }
            if key in self.active_configs:
                entry["trt_active_config"] = self.active_configs[key]
            tune_file = self.config.data_dir / "tuning" / f"{key}.json"
            if tune_file.exists():
                entry["tuning_saved"] = True
            out[key] = entry
        return out


class _ExternalSentinel:
    """
    Sentinel for external OpenAI-compatible servers (LM Studio, etc.).
    We do not own this process — poll() checks if it's still reachable.
    """
    def __init__(self, port: int):
        self.port = port
        self.pid = None

    def poll(self):
        """Return None if reachable (alive), 1 if not."""
        try:
            import httpx as _httpx
            r = _httpx.get(f"http://localhost:{self.port}/v1/models", timeout=2)
            return None if r.status_code == 200 else 1
        except Exception:
            return 1

    def kill(self): pass
    def terminate(self): pass
    def wait(self, timeout=None): pass


class _DockerSentinel:
    """Sentinel for managed Docker containers started by the router."""
    def __init__(self, container_name: str):
        self.container_name = container_name
        self.pid = None

    def poll(self):
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return None if result.returncode == 0 and result.stdout.strip() == "true" else 1
        except Exception:
            return 1

    def kill(self):
        pass

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass


class _OllamaSentinel:
    """Sentinel object for Ollama backends — not a real subprocess."""
    def __init__(self, port: int):
        self.port = port
        self.pid = None

    def poll(self):
        """Check if Ollama is still serving. Returns None if running."""
        try:
            import httpx as _httpx
            r = _httpx.get(f"http://localhost:{self.port}/api/tags", timeout=2)
            return None if r.status_code == 200 else 1
        except Exception:
            return 1

    def kill(self):
        pass

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass
