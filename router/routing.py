"""
router/routing.py — Request classifier

Determines which backend to route a request to based on:
  1. Explicit [route:key] prefix in the first user message
  2. Token count estimate
  3. Keyword scan (keywords configured in settings.yaml)
  4. Default: "fast"
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig


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


def _pick(backends: dict, preferred: str) -> str:
    """
    Return preferred tier if it exists in backends,
    otherwise fall back to the first available backend.
    """
    if preferred in backends:
        return preferred
    # Graceful fallback: use any available backend
    if backends:
        return next(iter(backends))
    return preferred  # caller will get a 400 — no backends registered
