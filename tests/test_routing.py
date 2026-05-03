"""Tests for router/routing.py — request classifier and load balancing."""

import pytest
from unittest.mock import MagicMock
from router.config import BackendConfig
from router.routing import (
    InvalidRouteKey,
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
        # Verify the prefix is stripped from the message before forwarding
        assert payload["messages"][0]["content"] == "test message"

    def test_explicit_route_unknown_key_raises(self):
        """`[route:<typo>]` must fail fast, never fall through to the
        classifier — direct selection is validated, like `?backend=`."""
        payload = {"messages": [{"content": "[route:nonexistent] test"}]}
        with pytest.raises(InvalidRouteKey) as excinfo:
            classify(payload, BACKENDS_BASIC, _make_config())
        assert excinfo.value.key == "nonexistent"
        assert set(excinfo.value.valid_backends) == set(BACKENDS_BASIC)
        # The original prefix is preserved (not stripped) so error logs
        # show what the caller actually sent.
        assert payload["messages"][0]["content"].startswith("[route:nonexistent]")

    def test_explicit_route_unknown_key_raises_from_classify_candidates(self):
        payload = {"messages": [{"content": "[route:nope] test"}]}
        with pytest.raises(InvalidRouteKey):
            classify_candidates(payload, BACKENDS_BASIC, _make_config())

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

    def test_classify_returns_empty_when_tier_missing(self):
        """A missing tier must surface as a deterministic empty result rather
        than silently routing to the first registered backend (fail-closed)."""
        backends = {"only-backend": BackendConfig(tier="fast", port=8080)}
        payload = {"messages": [{"content": "analyze this deeply"}]}  # routes to mid
        result = classify(payload, backends, _make_config())
        assert result == ""

    def test_classify_direct_backend_key_match_still_works(self):
        """Explicit [route:key] / ?backend= / alias targeting must keep working
        even when the named key is in a different tier than the classifier."""
        backends = {"only-backend": BackendConfig(tier="fast", port=8080)}
        payload = {"messages": [{"content": "[route:only-backend] anything"}]}
        result = classify(payload, backends, _make_config())
        assert result == "only-backend"


# ── _pick (load balancing) ────────────────────────────────────

class TestPick:
    def setup_method(self):
        set_benchmark_results({})

    def test_single_backend(self):
        assert _pick(BACKENDS_BASIC, "fast") == "fast"

    def test_no_tier_match_returns_empty(self):
        """Fail-closed: no backend in tier means no candidate."""
        backends = {"only": BackendConfig(tier="mid", port=8080)}
        result = _pick(backends, "fast")
        assert result == ""

    def test_direct_key_match_overrides_missing_tier(self):
        """If preferred is a registered backend key, return it directly."""
        backends = {"only": BackendConfig(tier="mid", port=8080)}
        assert _pick(backends, "only") == "only"

    def test_empty_backends_returns_empty(self):
        assert _pick({}, "fast") == ""

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

    def test_tools_with_no_capable_backend_returns_empty(self):
        """Fail-closed: tool-calling requests must not silently route to a
        backend that does not declare supports_tools."""
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "d1": BackendConfig(tier="deep", port=8080,
                                capabilities=BackendCapabilities(supports_tools=False)),
            "d2": BackendConfig(tier="deep", port=8081,
                                capabilities=BackendCapabilities(supports_tools=False)),
        }
        signals = RequestSignals(has_tools=True)
        assert _pick(backends, "deep", signals) == ""

    def test_json_schema_with_no_capable_backend_returns_empty(self):
        """Fail-closed: response_format=json_schema must not silently route to
        a backend that does not declare supports_json_schema."""
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "m1": BackendConfig(tier="mid", port=8080,
                                capabilities=BackendCapabilities(supports_json_schema=False)),
        }
        signals = RequestSignals(needs_json_schema=True)
        assert _pick(backends, "mid", signals) == ""

    def test_plain_dict_backends_with_tools_signal_fail_closed(self):
        """Backends without a capabilities object are treated as not capable —
        fail-closed semantics still apply."""
        backends = {
            "a": {"tier": "deep", "port": 8080},
            "b": {"tier": "deep", "port": 8081},
        }
        from router.routing import RequestSignals
        signals = RequestSignals(has_tools=True)
        assert _pick(backends, "deep", signals) == ""


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

    def test_no_tier_match_returns_empty_list(self):
        """Fail-closed: when the preferred tier has no backends, return []
        rather than silently falling back to a different tier."""
        backends = {
            "only": BackendConfig(tier="mid", port=8080),
        }
        assert select_candidates(backends, "fast") == []

    def test_direct_key_match_returns_single_candidate(self):
        """Direct backend-key match keeps explicit selection working."""
        backends = {
            "only": BackendConfig(tier="mid", port=8080),
        }
        assert select_candidates(backends, "only") == ["only"]

    def test_capability_mismatch_returns_empty_list(self):
        """Fail-closed: tool-calling with no capable backend returns []."""
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "d1": BackendConfig(tier="deep", port=8080,
                                capabilities=BackendCapabilities(supports_tools=False)),
        }
        signals = RequestSignals(has_tools=True)
        assert select_candidates(backends, "deep", signals) == []


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

    def test_classify_candidates_empty_when_capability_unmet(self):
        """Tool-calling routes to deep, but if no deep backend supports
        tools the candidate list is empty (fail-closed)."""
        from router.config import BackendCapabilities
        backends = {
            "deep1": BackendConfig(tier="deep", port=8080,
                                   capabilities=BackendCapabilities(supports_tools=False)),
            "fast1": BackendConfig(tier="fast", port=8081,
                                   capabilities=BackendCapabilities(supports_tools=True)),
        }
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        }
        # Even though fast1 supports tools, the request classifies to deep
        # and must not silently downshift to a different tier.
        assert classify_candidates(payload, backends, _make_config()) == []

    def test_classify_candidates_empty_when_tier_missing(self):
        """No backend in the classified tier and no direct key match → []."""
        backends = {"a": BackendConfig(tier="mid", port=8080)}
        payload = {"messages": [{"role": "user", "content": "hello"}]}  # → fast
        assert classify_candidates(payload, backends, _make_config()) == []


# ── Combined capability filter (tools AND json_schema) ──────

class TestCombinedCapabilityFilter:
    """Hard capability signals must AND together: a request that needs both
    tools and JSON-schema must land on a backend that declares both. The
    earlier `elif` form silently routed to a tools-only backend, which would
    drop the structured-output requirement."""

    def setup_method(self):
        set_benchmark_results({})

    def test_select_candidates_empty_when_no_backend_supports_both(self):
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "tools-only": BackendConfig(
                tier="deep", port=8080,
                capabilities=BackendCapabilities(
                    supports_tools=True, supports_json_schema=False)),
            "schema-only": BackendConfig(
                tier="deep", port=8081,
                capabilities=BackendCapabilities(
                    supports_tools=False, supports_json_schema=True)),
        }
        signals = RequestSignals(has_tools=True, needs_json_schema=True)
        assert select_candidates(backends, "deep", signals) == []

    def test_select_candidates_returns_only_both_capable(self):
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "tools-only": BackendConfig(
                tier="deep", port=8080,
                capabilities=BackendCapabilities(
                    supports_tools=True, supports_json_schema=False)),
            "schema-only": BackendConfig(
                tier="deep", port=8081,
                capabilities=BackendCapabilities(
                    supports_tools=False, supports_json_schema=True)),
            "both": BackendConfig(
                tier="deep", port=8082,
                capabilities=BackendCapabilities(
                    supports_tools=True, supports_json_schema=True)),
        }
        signals = RequestSignals(has_tools=True, needs_json_schema=True)
        result = select_candidates(backends, "deep", signals)
        assert result == ["both"]

    def test_pick_returns_both_capable(self):
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "tools-only": BackendConfig(
                tier="deep", port=8080,
                capabilities=BackendCapabilities(
                    supports_tools=True, supports_json_schema=False)),
            "both": BackendConfig(
                tier="deep", port=8081,
                capabilities=BackendCapabilities(
                    supports_tools=True, supports_json_schema=True)),
        }
        signals = RequestSignals(has_tools=True, needs_json_schema=True)
        assert _pick(backends, "deep", signals) == "both"

    def test_pick_empty_when_none_supports_both(self):
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "tools-only": BackendConfig(
                tier="deep", port=8080,
                capabilities=BackendCapabilities(
                    supports_tools=True, supports_json_schema=False)),
            "schema-only": BackendConfig(
                tier="deep", port=8081,
                capabilities=BackendCapabilities(
                    supports_tools=False, supports_json_schema=True)),
        }
        signals = RequestSignals(has_tools=True, needs_json_schema=True)
        assert _pick(backends, "deep", signals) == ""

    def test_direct_key_bypasses_capability_filter(self):
        """Direct backend selection (`?backend=`, alias, `[route:key]`) must
        return the named backend even when it lacks declared capabilities —
        operators opt in to that target intentionally."""
        from router.config import BackendCapabilities
        from router.routing import RequestSignals
        backends = {
            "named": BackendConfig(
                tier="deep", port=8080,
                capabilities=BackendCapabilities(
                    supports_tools=False, supports_json_schema=False)),
        }
        signals = RequestSignals(has_tools=True, needs_json_schema=True)
        # Direct selection: preferred IS a registered backend key.
        assert _pick(backends, "named", signals) == "named"
        assert select_candidates(backends, "named", signals) == ["named"]

    def test_classify_candidates_empty_for_combined_signals_no_match(self):
        """End-to-end: a payload with both `tools` and
        `response_format={"type": "json_schema"}` returns [] when no deep
        backend supports both, instead of silently dropping the schema
        requirement on a tools-only backend."""
        from router.config import BackendCapabilities
        backends = {
            "tools-only": BackendConfig(
                tier="deep", port=8080,
                capabilities=BackendCapabilities(
                    supports_tools=True, supports_json_schema=False)),
        }
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function",
                       "function": {"name": "noop", "parameters": {}}}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "x",
                                                "schema": {"type": "object"}}},
        }
        assert classify_candidates(payload, backends, _make_config()) == []
