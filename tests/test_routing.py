"""Tests for router/routing.py — request classifier and load balancing."""

import pytest
from unittest.mock import MagicMock
from router.config import BackendConfig
from router.routing import (
    classify,
    classify_candidates,
    select_candidates,
    set_benchmark_results,
    _backends_for_tier,
    _extract_content,
    _pick,
    _token_estimate,
)


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
    "fast": BackendConfig(tier="fast", port=8080),
    "mid": BackendConfig(tier="mid", port=8081),
    "deep": BackendConfig(tier="deep", port=8082),
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

    def test_deep_keyword_is_soft_signal(self):
        """Deep keywords push fast→mid, but do NOT force deep alone."""
        payload = {"messages": [{"content": "Please analyze this code"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

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
        backends = {"only-backend": BackendConfig(tier="fast", port=8080)}
        payload = {"messages": [{"content": "analyze this deeply"}]}
        result = classify(payload, backends, _make_config())
        # Should fall back to the only available backend
        assert result == "only-backend"


# ── _pick (load balancing) ────────────────────────────────────

class TestPick:
    def setup_method(self):
        set_benchmark_results({})

    def test_single_backend(self):
        assert _pick(BACKENDS_BASIC, "fast") == "fast"

    def test_fallback_to_first(self):
        backends = {"only": BackendConfig(tier="mid", port=8080)}
        result = _pick(backends, "fast")
        assert result == "only"

    def test_empty_backends(self):
        result = _pick({}, "fast")
        assert result == "fast"  # returns preferred, caller handles 400

    def test_round_robin_multiple(self):
        backends = {
            "fast-1": BackendConfig(tier="fast", port=8080),
            "fast-2": BackendConfig(tier="fast", port=8081),
            "deep": BackendConfig(tier="deep", port=8082),
        }
        results = [_pick(backends, "fast") for _ in range(4)]
        assert "fast-1" in results
        assert "fast-2" in results

    def test_prefers_faster_measured_backend(self):
        backends = {
            "fast-slow": BackendConfig(tier="fast", port=8080),
            "fast-quick": BackendConfig(tier="fast", port=8081),
        }
        set_benchmark_results({
            "fast-slow": {"tg_tok_s": 12.0},
            "fast-quick": {"tg_tok_s": 55.0},
        })

        assert _pick(backends, "fast") == "fast-quick"


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
            "fast-a": BackendConfig(tier="fast"),
            "fast-b": BackendConfig(tier="fast"),
            "mid": BackendConfig(tier="mid"),
        }
        result = _backends_for_tier(backends, "fast")
        assert set(result) == {"fast-a", "fast-b"}


# ── Structural signal classification ─────────────────────────

class TestStructuralSignals:
    def test_tools_routes_to_deep(self):
        payload = {"messages": [{"content": "hi"}], "tools": [{"type": "function"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "deep"

    def test_response_format_json_schema_routes_mid(self):
        payload = {
            "messages": [{"content": "list items"}],
            "response_format": {"type": "json_schema"},
        }
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

    def test_response_format_json_object_routes_mid(self):
        payload = {
            "messages": [{"content": "list items"}],
            "response_format": {"type": "json_object"},
        }
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

    def test_long_system_prompt_routes_mid(self):
        system = " ".join(["instruction"] * 2500)
        payload = {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "do the task"},
        ]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

    def test_many_messages_routes_mid(self):
        msgs = [{"role": "user", "content": "hi"}] * 12
        payload = {"messages": msgs}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

    def test_keyword_alone_does_not_force_deep(self):
        payload = {"messages": [{"content": "analyze this"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"  # keyword pushes fast→mid, not deep

    def test_keyword_mid_from_fast(self):
        payload = {"messages": [{"content": "write a poem"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "mid"

    def test_explicit_route_overrides_everything(self):
        payload = {"messages": [{"content": "[route:fast] analyze deeply"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "fast"

    def test_no_signals_defaults_fast(self):
        payload = {"messages": [{"content": "hello"}]}
        result = classify(payload, BACKENDS_BASIC, _make_config())
        assert result == "fast"


# ── Capability-aware _pick ───────────────────────────────────

class TestCapabilityAwarePick:
    def setup_method(self):
        set_benchmark_results({})

    def test_tools_prefers_capable_backend(self):
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "d1": BackendConfig(tier="deep", port=8080,
                                capabilities=BackendCapabilities(supports_tools=False)),
            "d2": BackendConfig(tier="deep", port=8081,
                                capabilities=BackendCapabilities(supports_tools=True)),
        }
        signals = RequestSignals(has_tools=True)
        assert _pick(backends, "deep", signals) == "d2"

    def test_json_schema_prefers_capable_backend(self):
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "m1": BackendConfig(tier="mid", port=8080,
                                capabilities=BackendCapabilities(supports_json_schema=False)),
            "m2": BackendConfig(tier="mid", port=8081,
                                capabilities=BackendCapabilities(supports_json_schema=True)),
        }
        signals = RequestSignals(needs_json_schema=True)
        assert _pick(backends, "mid", signals) == "m2"

    def test_no_capable_backends_falls_back_to_all(self):
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "d1": BackendConfig(tier="deep", port=8080,
                                capabilities=BackendCapabilities(supports_tools=False)),
            "d2": BackendConfig(tier="deep", port=8081,
                                capabilities=BackendCapabilities(supports_tools=False)),
        }
        signals = RequestSignals(has_tools=True)
        result = _pick(backends, "deep", signals)
        assert result in ("d1", "d2")

    def test_plain_dict_backends_still_work(self):
        """Backward compat: backends without capabilities field."""
        backends = {
            "a": {"tier": "deep", "port": 8080},
            "b": {"tier": "deep", "port": 8081},
        }
        from router.routing import RequestSignals
        signals = RequestSignals(has_tools=True)
        result = _pick(backends, "deep", signals)
        assert result in ("a", "b")


# ── Candidate selection & fallback ──────────────────────────


class TestSelectCandidates:
    def test_returns_ordered_list(self):
        """select_candidates returns a list sorted by engine score."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080, engine="ollama"),
            "b": BackendConfig(tier="fast", port=8081, engine="llama.cpp"),
        }
        result = select_candidates(backends, "fast")
        assert isinstance(result, list)
        assert len(result) == 2
        # llama.cpp (priority 5) beats ollama (priority 8)
        assert result[0] == "b"
        assert result[1] == "a"

    def test_limit_caps_results(self):
        """Only up to limit backends are returned."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080),
            "b": BackendConfig(tier="fast", port=8081),
            "c": BackendConfig(tier="fast", port=8082),
        }
        result = select_candidates(backends, "fast", limit=2)
        assert len(result) == 2

    def test_unhealthy_sorted_last(self):
        """Unhealthy backends are deprioritized to end of list."""
        backends = {
            "a": BackendConfig(tier="fast", port=8080, engine="llama.cpp"),
            "b": BackendConfig(tier="fast", port=8081, engine="llama.cpp"),
        }
        healthy_fn = lambda k: k != "a"
        result = select_candidates(backends, "fast", healthy_fn=healthy_fn)
        assert result[0] == "b"  # healthy first
        assert result[-1] == "a"  # unhealthy last

    def test_fallback_to_any_tier(self):
        """When preferred tier has no backends, fall back to available ones."""
        backends = {
            "only": BackendConfig(tier="mid", port=8080),
        }
        result = select_candidates(backends, "fast")
        # No "fast" tier backends, so falls back to any available
        assert "only" in result


class TestClassifyCandidates:
    def test_returns_list(self):
        """classify_candidates returns a list, not a string."""
        config = _make_config()
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        result = classify_candidates(payload, BACKENDS_BASIC, config)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_explicit_route_returns_single(self):
        """[route:key] prefix returns exactly that backend."""
        config = _make_config()
        payload = {"messages": [{"role": "user", "content": "[route:deep] hello"}]}
        result = classify_candidates(payload, BACKENDS_BASIC, config)
        assert result == ["deep"]

    def test_classify_backward_compat(self):
        """classify() still returns a single string."""
        config = _make_config()
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        result = classify(payload, BACKENDS_BASIC, config)
        assert isinstance(result, str)
        assert result in BACKENDS_BASIC
