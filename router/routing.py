"""
router/routing.py — Request classifier with best-engine selection

Determines which backend to route a request to based on:
  1. Explicit [route:key] prefix in the first user message
  2. Structural signals (tools, response_format, system prompt, message count)
  3. Token count thresholds
  4. Keywords (soft tiebreaker — can push fast→mid, never force deep alone)
  5. Default: "fast"

When multiple backends share a tier, picks the BEST one rather than
round-robining randomly:
  - Benchmark TG speed (from ./router-start bench) takes priority
  - Falls back to engine capability ranking when no benchmarks exist
  - Capability-aware filtering prefers backends matching request needs
  - Round-robins only among backends with identical measured speed

Engine capability ranking (higher = preferred when speed unknown):
  trt-llm > trt-llm-docker > vllm > sglang > llama.cpp > huggingface > openai > ollama
"""

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig


class InvalidRouteKey(Exception):
    """Raised when ``[route:<key>]`` names a backend that is not registered.

    Direct selection (``?backend=``, ``model_aliases``, ``[route:key]``) is
    validated. The proxy maps this to a surface-specific 400 client error
    rather than silently falling through to automatic classification — a
    typo must not be allowed to land on a different backend.
    """

    def __init__(self, key: str, valid_backends: list[str]):
        self.key = key
        self.valid_backends = valid_backends
        super().__init__(f"Unknown route key '{key}'")

# ─────────────────────────────────────────────────────────────
# Engine priority (lower number = preferred when no benchmark)
# ─────────────────────────────────────────────────────────────
ENGINE_PRIORITY: dict[str, int] = {
    "trt-llm":     1,   # fastest raw throughput, NVIDIA-optimised
    "trt-llm-docker": 2,  # same runtime, but Docker-mediated start/stop
    "vllm":        3,   # best for concurrent users (paged attention)
    "sglang":      4,   # strong concurrency, long-context optimised
    "llama.cpp":   5,   # flexible CPU+GPU, GGUF, great single-user
    "huggingface": 6,   # reference impl, slower
    "openai":      7,   # external server (LM Studio etc.) — speed unknown
    "ollama":      8,   # external server — speed unknown
}

# Benchmark results injected at startup from ~/.llm-router/benchmarks/
# Keyed by backend slug → {tg_tok_s, pp_tok_s, ...}
_bench: dict[str, dict] = {}

# Round-robin cycles — only used when backends have equal priority + speed
_tier_cycles: dict[str, tuple[frozenset, itertools.cycle]] = {}


def set_benchmark_results(results: dict[str, dict]):
    """
    Called once at startup (and after /rescan) with cached benchmark data.
    Allows _pick() to prefer faster backends without file I/O per request.
    """
    global _bench
    _bench = results


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _extract_content(payload: dict) -> str:
    """Flatten all message content into a single lowercase string."""
    messages = payload.get("messages", [])
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
    return " ".join(parts).lower()


def _token_estimate(content: str) -> int:
    """Fast word-count token estimate. No tokenizer needed."""
    return len(content.split())


@dataclass
class RequestSignals:
    """Structural signals extracted from a request payload."""
    has_tools: bool = False
    needs_json_schema: bool = False
    system_token_count: int = 0
    total_messages: int = 0
    token_count: int = 0
    keyword_tier: str = ""  # "mid" or "deep" from keyword scan, or ""


def _extract_signals(payload: dict, config: "AppConfig") -> RequestSignals:
    """Extract all classification signals from a request payload."""
    content = _extract_content(payload)
    messages = payload.get("messages", [])

    # System prompt tokens
    system_tokens = 0
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str):
                system_tokens += _token_estimate(c)

    # response_format
    rf = payload.get("response_format") or {}
    needs_schema = rf.get("type") in ("json_schema", "json_object")

    # Keyword scan (soft signal)
    kw_tier = ""
    if any(kw in content for kw in config.routing.deep_keywords):
        kw_tier = "deep"
    elif any(kw in content for kw in config.routing.mid_keywords):
        kw_tier = "mid"

    return RequestSignals(
        has_tools=bool(payload.get("tools") or payload.get("functions")),
        needs_json_schema=needs_schema,
        system_token_count=system_tokens,
        total_messages=len(messages),
        token_count=_token_estimate(content),
        keyword_tier=kw_tier,
    )


_TIER_ORDER = {"fast": 0, "mid": 1, "deep": 2}


def _max_tier(a: str, b: str) -> str:
    return a if _TIER_ORDER.get(a, 0) >= _TIER_ORDER.get(b, 0) else b


def _backends_for_tier(backends: dict, tier: str) -> list[str]:
    """Return all backend keys that belong to a tier."""
    return [k for k, v in backends.items() if v.get("tier") == tier]


def _engine_score(key: str, backends: dict) -> tuple[float, int]:
    """
    Sorting key for a backend — lower is better.
    (negative_tg_speed, engine_priority)

    Backends with benchmark data are ranked by measured TG tok/s.
    Backends without benchmarks are ranked by engine priority.
    Identical scores → round-robin (load balancing).
    """
    bench = _bench.get(key, {})
    tg = bench.get("tg_tok_s")
    engine = backends[key].get("engine", "llama.cpp")
    priority = ENGINE_PRIORITY.get(engine, 99)

    if tg:
        # Negative so higher tok/s sorts first
        return (-tg, priority)

    # No benchmark: use engine priority, treat speed as 0
    return (0.0, priority)


def _filter_capable(
    backends: dict,
    keys: list[str],
    signals: "RequestSignals | None",
) -> list[str]:
    """Apply hard capability filters in sequence (AND semantics).

    A request that needs both ``tools`` and ``response_format=json_schema``
    must land on a backend that declares **both** capabilities — applying
    the filters with ``elif`` would let a tool-capable but schema-incapable
    backend serve a combined request.

    Returns the (possibly empty) intersection. ``[]`` means no candidate in
    *keys* satisfies every required capability — callers fail closed by
    returning ``""`` / ``[]`` from their public API so the proxy maps it to
    the surface's no-backend error.
    """
    if not signals:
        return keys

    filtered = keys

    if signals.has_tools:
        filtered = [
            k for k in filtered
            if getattr(backends[k].get("capabilities"), "supports_tools", False)
        ]
        if not filtered:
            return []

    if signals.needs_json_schema:
        filtered = [
            k for k in filtered
            if getattr(backends[k].get("capabilities"), "supports_json_schema", False)
        ]
        if not filtered:
            return []

    return filtered


def _pick(backends: dict, preferred: str, signals: RequestSignals = None) -> str:
    """
    Return the best backend for the preferred tier.

    Selection order:
      1. Direct backend-key match (caller named a registered backend key directly).
      2. Filter to the preferred tier.
      3. Capability filter (tools, JSON schema) — fail-closed if signals provided.
      4. Highest measured TG tok/s from benchmarks.
      5. Engine capability ranking (trt-llm > vllm > sglang > llama.cpp …).
      6. Round-robin among backends with identical score (load balancing).

    Returns "" when the preferred tier or required capability cannot be
    satisfied. Callers map "" to a deterministic service-unavailable error
    rather than silently routing to a wrong-tier or incapable backend.
    """
    # Direct key match preserves explicit ?backend=/alias/[route:key] semantics.
    if preferred in backends:
        return preferred

    tier_backends = _backends_for_tier(backends, preferred)
    if not tier_backends:
        return ""

    tier_backends = _filter_capable(backends, tier_backends, signals)
    if not tier_backends:
        return ""

    if len(tier_backends) == 1:
        return tier_backends[0]

    # Sort by score — best first
    ranked = sorted(tier_backends, key=lambda k: _engine_score(k, backends))
    best_score = _engine_score(ranked[0], backends)

    # Collect all backends sharing the top score (for round-robin load balancing)
    top = [k for k in ranked if _engine_score(k, backends) == best_score]

    if len(top) == 1:
        return top[0]

    # Multiple equally-good backends → round-robin among them
    key = preferred
    current = _tier_cycles.get(key)
    top_set = frozenset(top)
    if current is None or current[0] != top_set:
        _tier_cycles[key] = (top_set, itertools.cycle(top))
    return next(_tier_cycles[key][1])


def select_candidates(
    backends: dict,
    preferred: str,
    signals: RequestSignals = None,
    limit: int = 3,
    healthy_fn=None,
) -> list[str]:
    """Return up to *limit* backends ordered by preference for *preferred* tier.

    *healthy_fn*, when provided, is called with a backend key and should
    return False for recently-failed backends.  Unhealthy backends are
    sorted to the end so they are only tried as a last resort.

    Fail-closed semantics: returns ``[]`` when the preferred tier has no
    backends or when a hard capability signal (tools / JSON schema) cannot
    be satisfied by any tier backend. Callers map ``[]`` to a deterministic
    service-unavailable error rather than silently routing to a wrong-tier
    or incapable backend. A direct backend-key match (``preferred in
    backends``) still wins so explicit ``?backend=``, model-alias, and
    ``[route:key]`` selections continue to work.
    """
    if preferred in backends:
        return [preferred]

    tier_backends = _backends_for_tier(backends, preferred)
    if not tier_backends:
        return []

    tier_backends = _filter_capable(backends, tier_backends, signals)
    if not tier_backends:
        return []

    ranked = sorted(tier_backends, key=lambda k: _engine_score(k, backends))

    # Push unhealthy backends to the end
    if healthy_fn:
        healthy = [k for k in ranked if healthy_fn(k)]
        unhealthy = [k for k in ranked if not healthy_fn(k)]
        ranked = healthy + unhealthy

    return ranked[:limit]


# ─────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────

def _strip_route_prefix(text: str, backends: dict) -> "tuple[str, str | None]":
    """Strip a leading ``[route:<key>]`` prefix from *text*.

    Returns ``(stripped_text, key)`` when the prefix is present and the key
    is a registered backend. Returns ``(text, None)`` when no prefix is
    present. Raises :class:`InvalidRouteKey` when the prefix names an
    unregistered backend so the caller can fail fast with a 400.
    """
    if not text.startswith("[route:") or "]" not in text:
        return text, None
    key = text.split("]", 1)[0].replace("[route:", "").strip()
    if key in backends:
        prefix_end = text.index("]") + 1
        return text[prefix_end:].lstrip(), key
    raise InvalidRouteKey(key, list(backends.keys()))


def extract_route_key(payload: dict, backends: dict) -> "str | None":
    """Scan a request payload for a ``[route:<key>]`` prefix.

    Handles the three content shapes the router accepts:

    * OpenAI ``messages[i].content`` as a string
    * Anthropic ``messages[i].content`` as a list of ``{"type": "text", ...}`` blocks
    * Gemini ``contents[i].parts[j].text``

    On success the prefix is stripped from the original payload in place so
    it does not leak to the backend, and the validated backend key is
    returned. Raises :class:`InvalidRouteKey` when a prefix is present but
    names an unregistered backend — proxies map this to a surface-specific
    400 error rather than silently routing to a different backend.
    """
    for m in payload.get("messages", []):
        c = m.get("content", "")
        if isinstance(c, str):
            stripped, key = _strip_route_prefix(c, backends)
            if key is not None:
                m["content"] = stripped
                return key
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and "text" in part:
                    stripped, key = _strip_route_prefix(part.get("text", ""), backends)
                    if key is not None:
                        part["text"] = stripped
                        return key

    for content in payload.get("contents", []) or []:
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and "text" in part:
                stripped, key = _strip_route_prefix(part.get("text", ""), backends)
                if key is not None:
                    part["text"] = stripped
                    return key

    return None


def _classify_tier(payload: dict, backends: dict, config: "AppConfig"):
    """Determine tier and signals, returning (tier, signals, explicit_key_or_None)."""
    messages = payload.get("messages", [])

    # Honor an inline `[route:<key>]` prefix. The helper raises
    # :class:`InvalidRouteKey` for an unregistered key so the proxy can
    # respond with a 400 — never silently fall through to tier classification.
    route_key = extract_route_key(payload, backends)
    if route_key is not None:
        return None, None, route_key

    signals = _extract_signals(payload, config)
    tier = "fast"

    if signals.has_tools:
        tier = "deep"
    elif signals.needs_json_schema:
        tier = _max_tier(tier, "mid")
    elif signals.token_count > config.routing.token_threshold_deep:
        tier = "deep"
    elif signals.system_token_count > 2000:
        tier = _max_tier(tier, "mid")
    elif signals.total_messages > 10:
        tier = _max_tier(tier, "mid")
    elif signals.token_count > config.routing.token_threshold_mid:
        tier = _max_tier(tier, "mid")

    if tier == "fast" and signals.keyword_tier in ("mid", "deep"):
        tier = "mid"
    elif tier == "mid" and signals.keyword_tier == "deep":
        if signals.token_count > config.routing.token_threshold_mid:
            tier = "deep"

    return tier, signals, None


def classify_candidates(
    payload: dict,
    backends: dict,
    config: "AppConfig",
    limit: int = 3,
    healthy_fn=None,
) -> list[str]:
    """Classify a request and return an ordered list of candidate backends."""
    tier, signals, explicit_key = _classify_tier(payload, backends, config)
    if explicit_key:
        return [explicit_key]
    return select_candidates(backends, tier, signals, limit, healthy_fn)


def classify(payload: dict, backends: dict, config: "AppConfig") -> str:
    """
    Classify a request payload and return the single best backend key.

    Returns ``""`` when no backend can satisfy the request's tier/capability
    requirements. Callers map ``""`` to a deterministic service-unavailable
    error rather than silently routing to a wrong-tier or incapable backend.
    For ranked fallback support, use ``classify_candidates()``.
    """
    candidates = classify_candidates(payload, backends, config, limit=1)
    return candidates[0] if candidates else ""
