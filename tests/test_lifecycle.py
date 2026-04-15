"""Tests for VRAM-aware lifecycle management in router/lifecycle.py."""

import asyncio
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router.config import BackendConfig
from router.lifecycle import BackendManager, _DockerSentinel


def _make_config():
    """Minimal AppConfig mock for BackendManager."""
    return SimpleNamespace(
        data_dir=Path("/tmp"),
        trtllm_docker=SimpleNamespace(
            image="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7",
            container_port=8000,
            hf_cache_dir=Path("/tmp/hf-cache"),
            log_dir=Path("/tmp/trtllm-logs"),
            env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            serve_defaults={
                "max_seq_len": 65536,
                "max_num_tokens": 16384,
                "max_batch_size": 4,
                "kv_cache_free_gpu_memory_fraction": 0.75,
            },
        ),
    )


def _make_manager(backends: dict) -> BackendManager:
    """Create a BackendManager with fake backends and no real processes."""
    mgr = BackendManager(backends, _make_config())
    return mgr


def _fake_running(mgr: BackendManager, key: str, last_used: float = 0):
    """Mark a backend as running with a fake sentinel process."""
    sentinel = MagicMock()
    sentinel.poll.return_value = None  # poll() == None means "still running"
    mgr.processes[key] = sentinel
    mgr.last_used[key] = last_used


# ── _evict_for_vram ──────────────────────────────────────────


class TestEvictForVram:
    def test_enough_free_vram_no_eviction(self):
        """When free VRAM >= needed, no eviction should happen."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080, vram_estimate_gb=4.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "a", last_used=100)

        with patch("router.sysinfo.query_free_vram", return_value=(20.0, 48.0)):
            result = mgr._evict_for_vram(10.0)

        assert result is True
        assert mgr.is_running("a")  # not evicted

    def test_not_enough_vram_evicts_oldest(self):
        """When VRAM is tight, evict oldest backend to make room."""
        backends = {
            "old": BackendConfig(tier="fast", port=8080, vram_estimate_gb=8.0),
            "new": BackendConfig(tier="deep", port=8081, vram_estimate_gb=20.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "old", last_used=100)
        _fake_running(mgr, "new", last_used=200)

        # First call: 5 GB free. After evicting "old": 13 GB free.
        vram_calls = iter([(5.0, 48.0), (13.0, 48.0)])
        with patch("router.sysinfo.query_free_vram", side_effect=lambda: next(vram_calls)):
            result = mgr._evict_for_vram(12.0)

        assert result is True
        assert not mgr.is_running("old")  # evicted
        assert mgr.is_running("new")  # kept

    def test_still_not_enough_after_all_evictions(self):
        """When evicting all backends still can't free enough, return False."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080, vram_estimate_gb=4.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "a", last_used=100)

        # 2 GB free, after evicting "a" only 6 GB free, but need 20 GB
        vram_calls = iter([(2.0, 48.0), (6.0, 48.0)])
        with patch("router.sysinfo.query_free_vram", side_effect=lambda: next(vram_calls)):
            result = mgr._evict_for_vram(20.0)

        assert result is False
        assert not mgr.is_running("a")  # evicted but still wasn't enough

    def test_no_gpu_info_skips_vram_logic(self):
        """When nvidia-smi is unavailable, skip VRAM logic entirely."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080, vram_estimate_gb=4.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "a", last_used=100)

        with patch("router.sysinfo.query_free_vram", return_value=None):
            result = mgr._evict_for_vram(999.0)

        assert result is True  # always passes when no GPU info
        assert mgr.is_running("a")  # not evicted

    def test_exclude_key_not_evicted(self):
        """The backend being started should not be evicted."""
        backends = {
            "target": BackendConfig(tier="deep", port=8080, vram_estimate_gb=20.0),
            "other": BackendConfig(tier="fast", port=8081, vram_estimate_gb=4.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "target", last_used=50)  # oldest
        _fake_running(mgr, "other", last_used=100)

        vram_calls = iter([(5.0, 48.0), (9.0, 48.0)])
        with patch("router.sysinfo.query_free_vram", side_effect=lambda: next(vram_calls)):
            result = mgr._evict_for_vram(8.0, exclude_key="target")

        assert mgr.is_running("target")  # excluded from eviction
        assert not mgr.is_running("other")  # evicted instead

    def test_eviction_order_oldest_first(self):
        """Backends should be evicted in last_used ascending order."""
        backends = {
            "oldest": BackendConfig(tier="fast", port=8080, vram_estimate_gb=4.0),
            "middle": BackendConfig(tier="mid", port=8081, vram_estimate_gb=8.0),
            "newest": BackendConfig(tier="deep", port=8082, vram_estimate_gb=12.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "oldest", last_used=100)
        _fake_running(mgr, "middle", last_used=200)
        _fake_running(mgr, "newest", last_used=300)

        # Need 15 GB. Free: 2 -> evict oldest -> 6 -> evict middle -> 14 -> evict newest -> 26
        vram_calls = iter([(2.0, 48.0), (6.0, 48.0), (14.0, 48.0), (26.0, 48.0)])
        with patch("router.sysinfo.query_free_vram", side_effect=lambda: next(vram_calls)):
            result = mgr._evict_for_vram(15.0)

        assert result is True
        assert not mgr.is_running("oldest")
        assert not mgr.is_running("middle")
        # newest survives — 14 GB was not enough, but after middle eviction we re-check
        # and the loop checks free_gb >= needed at the top, so newest may or may not be evicted
        # depending on whether 14 >= 15. It's not, so newest also gets evicted.
        assert not mgr.is_running("newest")

    def test_no_running_backends_returns_false_when_insufficient(self):
        """If no backends are running to evict, just return the VRAM check result."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080, vram_estimate_gb=4.0),
        }
        mgr = _make_manager(backends)
        # "a" is NOT running

        with patch("router.sysinfo.query_free_vram", return_value=(2.0, 48.0)):
            result = mgr._evict_for_vram(10.0)

        assert result is False

    def test_backend_without_vram_estimate_still_evicted(self):
        """Backends with vram_estimate_gb=None can still be evicted."""
        backends = {
            "ext": BackendConfig(tier="fast", port=8080, vram_estimate_gb=None),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "ext", last_used=100)

        # 2 GB free, after evicting: re-query says 10 GB
        vram_calls = iter([(2.0, 48.0), (10.0, 48.0)])
        with patch("router.sysinfo.query_free_vram", side_effect=lambda: next(vram_calls)):
            result = mgr._evict_for_vram(8.0)

        assert result is True
        assert not mgr.is_running("ext")


# ── ensure_running VRAM integration ──────────────────────────


class TestEnsureRunningVram:
    @pytest.mark.asyncio
    async def test_vram_check_raises_when_insufficient(self):
        """ensure_running should raise RuntimeError when VRAM can't be freed."""
        backends = {
            "big": BackendConfig(
                engine="llama.cpp", tier="deep", port=8080,
                vram_estimate_gb=40.0, startup_wait=5,
                log="/tmp/test.log",
            ),
        }
        mgr = _make_manager(backends)

        with patch("router.sysinfo.query_free_vram", return_value=(5.0, 48.0)):
            with pytest.raises(RuntimeError, match="Not enough VRAM"):
                await mgr.ensure_running("big")

    @pytest.mark.asyncio
    async def test_no_vram_estimate_skips_check(self):
        """Backends without vram_estimate_gb skip the VRAM check entirely."""
        backends = {
            "ext": BackendConfig(
                engine="openai", tier="fast", port=1234,
                vram_estimate_gb=None, startup_wait=5,
                log="/tmp/test.log",
            ),
        }
        mgr = _make_manager(backends)

        # Mock the external start to succeed
        with patch.object(mgr, "_start_process", return_value=True):
            with patch("router.sysinfo.query_free_vram", return_value=(2.0, 48.0)):
                # Should NOT raise despite only 2 GB free — vram_estimate_gb is None
                await mgr.ensure_running("ext")

    @pytest.mark.asyncio
    async def test_no_gpu_skips_check(self):
        """When no GPU info is available, VRAM check is skipped."""
        backends = {
            "model": BackendConfig(
                engine="llama.cpp", tier="fast", port=8080,
                vram_estimate_gb=20.0, startup_wait=5,
                log="/tmp/test.log",
            ),
        }
        mgr = _make_manager(backends)

        with patch.object(mgr, "_start_process", return_value=True):
            with patch("router.sysinfo.query_free_vram", return_value=None):
                # Should NOT raise despite huge vram_estimate — no GPU detected
                await mgr.ensure_running("model")


class TestManagedTRTLLMDocker:
    @pytest.mark.asyncio
    async def test_start_uses_native_docker_run_command(self, tmp_path):
        backends = {
            "hf-nvidia": BackendConfig(
                engine="trt-llm-docker",
                port=8111,
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
                log=str(tmp_path / "backend.log"),
                docker_config={
                    "hf_cache_dir": str(tmp_path / "hf-cache"),
                    "log_dir": str(tmp_path / "docker-logs"),
                },
            ),
        }
        mgr = _make_manager(backends)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("router.lifecycle.is_engine_available", return_value=True):
            with patch("router.lifecycle.subprocess.run", side_effect=fake_run):
                with patch.object(mgr, "_wait_healthy", AsyncMock(return_value=True)):
                    ok = await mgr._start_process("hf-nvidia")

        assert ok is True
        assert any(cmd[:3] == ["docker", "run", "-d"] for cmd in calls)
        docker_run = next(cmd for cmd in calls if cmd[:3] == ["docker", "run", "-d"])
        assert "--name" in docker_run and docker_run[docker_run.index("--name") + 1] == "llm-router-hf-nvidia"
        assert "-p" in docker_run and docker_run[docker_run.index("-p") + 1] == "8111:8000"

    def test_stop_removes_docker_container(self):
        backends = {
            "hf-nvidia": BackendConfig(
                engine="trt-llm-docker",
                port=8111,
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
                log="/tmp/backend.log",
            ),
        }
        mgr = _make_manager(backends)
        mgr.processes["hf-nvidia"] = _DockerSentinel("llm-router-hf-nvidia")

        with patch("router.lifecycle.subprocess.run") as run_mock:
            mgr.stop("hf-nvidia")

        run_mock.assert_any_call(
            ["docker", "rm", "-f", "llm-router-hf-nvidia"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    @pytest.mark.asyncio
    async def test_launcher_script_bypasses_native_docker_run(self, tmp_path):
        backends = {
            "hf-nvidia": BackendConfig(
                engine="trt-llm-docker",
                port=8111,
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
                log=str(tmp_path / "backend.log"),
                docker_config={"launcher_script": "/tmp/start-trt.sh"},
            ),
        }
        mgr = _make_manager(backends)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("router.lifecycle.is_engine_available", return_value=True):
            with patch("router.lifecycle.subprocess.run", side_effect=fake_run):
                with patch.object(mgr, "_wait_healthy", AsyncMock(return_value=True)):
                    ok = await mgr._start_process("hf-nvidia")

        assert ok is True
        assert ["/tmp/start-trt.sh"] in calls
        assert not any(cmd[:3] == ["docker", "run", "-d"] for cmd in calls)

    @pytest.mark.asyncio
    async def test_failed_health_check_appends_docker_logs(self, tmp_path):
        log_path = tmp_path / "backend.log"
        backends = {
            "hf-nvidia": BackendConfig(
                engine="trt-llm-docker",
                port=8111,
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
                log=str(log_path),
                docker_config={
                    "hf_cache_dir": str(tmp_path / "hf-cache"),
                    "log_dir": str(tmp_path / "docker-logs"),
                },
            ),
        }
        mgr = _make_manager(backends)

        def fake_run(cmd, **kwargs):
            if cmd[:3] == ["docker", "logs", "--tail"]:
                return MagicMock(returncode=0, stdout="container stacktrace\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("router.lifecycle.is_engine_available", return_value=True):
            with patch("router.lifecycle.subprocess.run", side_effect=fake_run):
                with patch.object(mgr, "_wait_healthy", AsyncMock(return_value=False)):
                    ok = await mgr._start_process("hf-nvidia")

        assert ok is False
        assert "docker logs tail" in log_path.read_text()
        assert "container stacktrace" in log_path.read_text()


# ── Request lease & in-flight tracking ──────────────────────


class TestRequestLease:
    @pytest.mark.asyncio
    async def test_lease_increments_and_decrements(self):
        """active_requests counter goes up on enter, down on exit."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080),
        }
        mgr = _make_manager(backends)
        assert mgr.active_requests.get("a", 0) == 0

        async with mgr.request_lease("a"):
            assert mgr.active_requests["a"] == 1
        assert mgr.active_requests["a"] == 0

    @pytest.mark.asyncio
    async def test_lease_updates_last_used(self):
        """last_used is refreshed on both entry and exit."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080),
        }
        mgr = _make_manager(backends)
        mgr.last_used["a"] = 0.0

        async with mgr.request_lease("a"):
            entry_time = mgr.last_used["a"]
            assert entry_time > 0

        exit_time = mgr.last_used["a"]
        assert exit_time >= entry_time

    @pytest.mark.asyncio
    async def test_lease_decrements_on_exception(self):
        """Counter is decremented even if the body raises."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080),
        }
        mgr = _make_manager(backends)

        with pytest.raises(ValueError):
            async with mgr.request_lease("a"):
                assert mgr.active_requests["a"] == 1
                raise ValueError("boom")
        assert mgr.active_requests["a"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_leases_stack(self):
        """Multiple concurrent leases on the same backend add up."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080),
        }
        mgr = _make_manager(backends)

        async with mgr.request_lease("a"):
            assert mgr.active_requests["a"] == 1
            async with mgr.request_lease("a"):
                assert mgr.active_requests["a"] == 2
            assert mgr.active_requests["a"] == 1
        assert mgr.active_requests["a"] == 0


class TestIdleWatchdogSkipsActive:
    @pytest.mark.asyncio
    async def test_watchdog_skips_backend_with_active_requests(self):
        """idle_watchdog must not stop a backend that has active leases."""
        backends = {
            "busy": BackendConfig(tier="fast", port=8080, idle_timeout=1),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "busy", last_used=0)  # ancient last_used
        mgr.active_requests["busy"] = 1  # one active request

        # Run one watchdog cycle — use asyncio.CancelledError to break the loop
        cycle_count = 0

        async def one_cycle(sec):
            nonlocal cycle_count
            cycle_count += 1
            if cycle_count > 1:
                raise asyncio.CancelledError

        with patch("router.lifecycle.asyncio.sleep", side_effect=one_cycle):
            with patch("router.sysinfo.query_free_vram", return_value=None):
                with pytest.raises(asyncio.CancelledError):
                    await mgr.idle_watchdog()

        assert mgr.is_running("busy")  # NOT evicted

    @pytest.mark.asyncio
    async def test_watchdog_stops_idle_backend_without_active_requests(self):
        """idle_watchdog stops backends that are idle AND have no active requests."""
        backends = {
            "idle": BackendConfig(tier="fast", port=8080, idle_timeout=1),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "idle", last_used=0)
        mgr.active_requests["idle"] = 0

        cycle_count = 0

        async def one_cycle(sec):
            nonlocal cycle_count
            cycle_count += 1
            if cycle_count > 1:
                raise asyncio.CancelledError

        with patch("router.lifecycle.asyncio.sleep", side_effect=one_cycle):
            with patch("router.sysinfo.query_free_vram", return_value=None):
                with pytest.raises(asyncio.CancelledError):
                    await mgr.idle_watchdog()

        assert not mgr.is_running("idle")  # evicted


class TestEvictSkipsActive:
    def test_evict_for_vram_skips_backend_with_active_requests(self):
        """_evict_for_vram must skip backends that have active leases."""
        backends = {
            "busy": BackendConfig(tier="fast", port=8080, vram_estimate_gb=8.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "busy", last_used=0)
        mgr.active_requests["busy"] = 1

        with patch("router.sysinfo.query_free_vram", return_value=(2.0, 48.0)):
            result = mgr._evict_for_vram(10.0)

        assert result is False  # can't free enough
        assert mgr.is_running("busy")  # not evicted — has active requests

    def test_evict_for_vram_evicts_idle_backend(self):
        """_evict_for_vram evicts backends with zero active requests."""
        backends = {
            "idle": BackendConfig(tier="fast", port=8080, vram_estimate_gb=8.0),
        }
        mgr = _make_manager(backends)
        _fake_running(mgr, "idle", last_used=0)
        mgr.active_requests["idle"] = 0

        vram_calls = iter([(2.0, 48.0), (10.0, 48.0)])
        with patch("router.sysinfo.query_free_vram", side_effect=lambda: next(vram_calls)):
            result = mgr._evict_for_vram(8.0)

        assert result is True
        assert not mgr.is_running("idle")
