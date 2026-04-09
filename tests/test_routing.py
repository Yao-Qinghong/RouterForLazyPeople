"""Tests for router/routing.py — request classifier and load balancing."""

import pytest
from unittest.mock import MagicMock
from router.routing import classify, _extract_content, _token_estimate, _pick, _backends_for_tier


# ── Fixtures ──────────────────────────────────────────────────

def _make_config(**overrides):
    """Create a mock AppConfig with routing settings."""
    config = MagicMock()
    config.routing.token_threshold_deep = overrides.get("deep_threshold", 4000)
    config.routing.token_threshold_mid = overrides.get("mid_threshold", 500)
    config.routing.deep_keywords = overrides.get("deep_keywords", ["reason", "analyze", "step by step"])
    config.routing.mid_keywords = overrides.get("mid_keywords", ["write", "code", "fix"])
    return config


BACKENDS_BASIC = {
    "fast": {"tier": "fast", "port": 8080},
    "mid": {"tier": "mid", "port": 8081},
    "deep": {"tier": "deep", "port": 8082},
}


# ── _extract_content ──────────────────────────────────────────

class TestExtractContent:
    def test_string_content(self):
        payload = {"messages": [{"content": "Hello World"}]}
        assert _extract_content(payload) == "hello world"

    def test_list_content(self):
        payload = {"messages": [{"content": [{"type": "text", "text": "Hello"}]}]}
        assert _extract_content(payload) == "hello"

    def test_multiple_messages(self):
        payload = {"messages": [
            {"content": "First"},
            {"content": "Second message here"},
        ]}
        result = _extract_content(payload)
        assert "first" in result
        assert "second" in result

    def test_empty_messages(self):
        assert _extract_content({}) == ""
        assert _extract_content({"messages": []}) == ""


# ── _token_estimate ───────────────────────────────────────────

class TestTokenEstimate:
    def test_word_count(self):
        assert _token_estimate("hello world") == 2

    def test_long_text(self):
        text = " ".join(["word"] * 5000)
        assert _token_estimate(text) == 5000

    def test_empty(self):
        assert _token_estimate("") == 0  # "".split() returns []


# ── classify ──────────────────────────────────────────────────

class TestClassify:
    def test_explicit_route_prefix(self):
        payload = {"messages": [{"content": "[route:deep] test message"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "deep"

    def test_explicit_route_unknown_key(self):
        payload = {"messages": [{"content": "[route:nonexistent] test"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        # Falls through to default classification
        assert result in BACKENDS_BASIC

    def test_deep_keyword(self):
        payload = {"messages": [{"content": "Please analyze this code"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "deep"

    def test_mid_keyword(self):
        payload = {"messages": [{"content": "write a function"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

    def test_default_fast(self):
        payload = {"messages": [{"content": "hi"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "fast"

    def test_long_prompt_deep(self):
        text = " ".join(["word"] * 5000)
        payload = {"messages": [{"content": text}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "deep"

    def test_medium_prompt_mid(self):
        text = " ".join(["word"] * 600)
        payload = {"messages": [{"content": text}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

    def test_fallback_when_tier_missing(self):
        backends = {"only-backend": {"tier": "fast", "port": 8080}}
        payload = {"messages": [{"content": "analyze this deeply"}]}
        result = classify(payload, backends, _make_config())
        # Should fall back to the only available backend
        assert result == "only-backend"


# ── _pick (load balancing) ────────────────────────────────────

class TestPick:
    def test_single_backend(self):
        assert _pick(BACKENDS_BASIC, "fast") == "fast"

    def test_fallback_to_first(self):
        backends = {"only": {"tier": "mid", "port": 8080}}
        result = _pick(backends, "fast")
        assert result == "only"

    def test_empty_backends(self):
        result = _pick({}, "fast")
        assert result == "fast"  # returns preferred, caller handles 400

    def test_round_robin_multiple(self):
        backends = {
            "fast-1": {"tier": "fast", "port": 8080},
            "fast-2": {"tier": "fast", "port": 8081},
            "deep": {"tier": "deep", "port": 8082},
        }
        results = [_pick(backends, "fast") for _ in range(4)]
        assert "fast-1" in results
        assert "fast-2" in results


# ── _backends_for_tier ────────────────────────────────────────

class TestBackendsForTier:
    def test_finds_tier(self):
        result = _backends_for_tier(BACKENDS_BASIC, "fast")
        assert result == ["fast"]

    def test_no_match(self):
        result = _backends_for_tier(BACKENDS_BASIC, "nonexistent")
        assert result == []

    def test_multiple_same_tier(self):
        backends = {
            "fast-a": {"tier": "fast"},
            "fast-b": {"tier": "fast"},
            "mid": {"tier": "mid"},
        }
        result = _backends_for_tier(backends, "fast")
        assert set(result) == {"fast-a", "fast-b"}
