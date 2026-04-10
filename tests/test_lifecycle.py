"""Tests for VRAM-aware lifecycle management in router/lifecycle.py."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from router.config import BackendConfig
from router.lifecycle import BackendManager


def _make_config():
    """Minimal AppConfig mock for BackendManager."""
    return SimpleNamespace(
        data_dir=SimpleNamespace(__truediv__=lambda self, x: f"/tmp/{x}"),
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
