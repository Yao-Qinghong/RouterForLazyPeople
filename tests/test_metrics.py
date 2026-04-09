"""Tests for router/metrics.py — metrics store and Prometheus export."""

import pytest
from unittest.mock import MagicMock
from router.metrics import (
    RequestRecord, MetricsStore, extract_token_counts, _percentile,
)


def _make_config():
    config = MagicMock()
    config.metrics.enabled = True
    config.metrics.persist_dir = MagicMock()
    config.metrics.flush_interval_sec = 60
    return config


def _make_record(**overrides):
    defaults = {
        "request_id": "test123",
        "timestamp_utc": "2026-04-09T12:00:00+00:00",
        "backend_key": "fast",
        "engine": "llama.cpp",
        "model_path": "/models/test.gguf",
        "endpoint": "chat/completions",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "ttft_ms": 200.0,
        "total_latency_ms": 1000.0,
        "tokens_per_sec": 50.0,
        "status_code": 200,
        "error": None,
    }
    defaults.update(overrides)
    return RequestRecord(**defaults)


class TestMetricsStore:
    def test_record_and_summary(self):
        store = MetricsStore(_make_config())
        store.record(_make_record())
        store.record(_make_record(status_code=500, error="boom"))
        summary = store.summary()
        assert "fast" in summary
        assert summary["fast"]["request_count"] == 2
        assert summary["fast"]["error_count"] == 1

    def test_prometheus_format(self):
        store = MetricsStore(_make_config())
        store.record(_make_record())
        prom = store.prometheus()
        assert "llm_router_requests_total 1" in prom
        assert "llm_router_errors_total 0" in prom
        assert "llm_router_tokens_total 150" in prom
        assert 'backend="fast"' in prom

    def test_global_counters(self):
        store = MetricsStore(_make_config())
        store.record(_make_record())
        store.record(_make_record())
        store.record(_make_record(status_code=500, error="err"))
        assert store._total_requests == 3
        assert store._total_errors == 1
        assert store._total_tokens == 450  # 150 * 3

    def test_disabled(self):
        config = _make_config()
        config.metrics.enabled = False
        store = MetricsStore(config)
        store.record(_make_record())
        assert len(store._ring) == 0


class TestExtractTokenCounts:
    def test_from_usage(self):
        body = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        p, c = extract_token_counts(body)
        assert p == 10
        assert c == 20

    def test_estimate_from_text(self):
        body = {
            "choices": [{"message": {"content": "hello world this is a test"}}],
            "usage": {},
        }
        p, c = extract_token_counts(body)
        assert c > 0  # estimated from word count


class TestPercentile:
    def test_p50(self):
        data = [1, 2, 3, 4, 5]
        assert _percentile(data, 50) == 3.0

    def test_p95(self):
        data = list(range(1, 101))
        result = _percentile(data, 95)
        assert 94 <= result <= 96

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([42], 50) == 42.0
