"""
router/routing.py — Request classifier with load balancing

Determines which backend to route a request to based on:
  1. Explicit [route:key] prefix in the first user message
  2. Token count estimate
  3. Keyword scan (keywords configured in settings.yaml)
  4. Default: "fast"

Supports multiple backends per tier with round-robin load balancing.
"""

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig

# Round-robin state per tier
_tier_cycles: dict[str, tuple[set[str], itertools.cycle]] = {}


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


def _backends_for_tier(backends: dict, tier: str) -> list[str]:
    """Return all backend keys that belong to a tier."""
    return [k for k, v in backends.items() if v.get("tier") == tier]


def _pick(backends: dict, preferred: str) -> str:
    """
    Return a backend from the preferred tier using round-robin
    if multiple backends exist. Falls back to any available backend.
    """
    tier_backends = _backends_for_tier(backends, preferred)

    if not tier_backends:
        # Exact key match (e.g., "fast" is a backend key, not a tier)
        if preferred in backends:
            return preferred
        # Graceful fallback: use any available backend
        if backends:
            return next(iter(backends))
        return preferred  # caller will get a 400 — no backends registered

    if len(tier_backends) == 1:
        return tier_backends[0]

    # Round-robin across backends in this tier
    current = _tier_cycles.get(preferred)
    if current is None or current[0] != set(tier_backends):
        cycle = itertools.cycle(tier_backends)
        _tier_cycles[preferred] = (set(tier_backends), cycle)

    return next(_tier_cycles[preferred][1])


def classify(payload: dict, backends: dict, config: "AppConfig") -> str:
    """
    Classify a request payload and return the backend key to use.

    Backends dict is passed in so routing can fall back gracefully
    when "fast" / "mid" / "deep" aren't defined.
    """
    content = _extract_content(payload)
    messages = payload.get("messages", [])

    # 1. Explicit routing prefix: [route:backend-key] in any message
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str) and c.startswith("[route:"):
            key = c.split("]")[0].replace("[route:", "").strip()
            if key in backends:
                return key

    token_count = _token_estimate(content)

    # 2. Long prompt → deep
    if token_count > config.routing.token_threshold_deep:
        return _pick(backends, "deep")

    # 3. Deep keywords → deep
    if any(kw in content for kw in config.routing.deep_keywords):
        return _pick(backends, "deep")

    # 4. Mid keywords → mid
    if token_count > config.routing.token_threshold_mid:
        return _pick(backends, "mid")
    if any(kw in content for kw in config.routing.mid_keywords):
        return _pick(backends, "mid")

    # 5. Default: fast
    return _pick(backends, "fast")
