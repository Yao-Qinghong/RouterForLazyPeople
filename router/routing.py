"""
router/routing.py — Request classifier with best-engine selection

Determines which backend to route a request to based on:
  1. Explicit [route:key] prefix in the first user message
  2. Tool use / function calling → deep (agentic tasks need capable models)
  3. Token count estimate
  4. Keyword scan (keywords configured in settings.yaml)
  5. Default: "fast"

When multiple backends share a tier, picks the BEST one rather than
round-robining randomly:
  - Benchmark TG speed (from ./router-start bench) takes priority
  - Falls back to engine capability ranking when no benchmarks exist
  - Round-robins only among backends with identical measured speed
    (e.g. two vLLM instances of the same model for load balancing)

Engine capability ranking (higher = preferred when speed unknown):
  trt-llm > vllm > sglang > llama.cpp > huggingface > openai > ollama
"""

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig

# ─────────────────────────────────────────────────────────────
# Engine priority (lower number = preferred when no benchmark)
# ─────────────────────────────────────────────────────────────
ENGINE_PRIORITY: dict[str, int] = {
    "trt-llm":     1,   # fastest raw throughput, NVIDIA-optimised
    "vllm":        2,   # best for concurrent users (paged attention)
    "sglang":      3,   # strong concurrency, long-context optimised
    "llama.cpp":   4,   # flexible CPU+GPU, GGUF, great single-user
    "huggingface": 5,   # reference impl, slower
    "openai":      6,   # external server (LM Studio etc.) — speed unknown
    "ollama":      7,   # external server — speed unknown
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


def _pick(backends: dict, preferred: str) -> str:
    """
    Return the best backend for the preferred tier.

    Selection order:
      1. Highest measured TG tok/s from benchmarks
      2. Engine capability ranking (trt-llm > vllm > sglang > llama.cpp …)
      3. Round-robin among backends with identical score (load balancing)

    Falls back gracefully if the preferred tier has no backends.
    """
    tier_backends = _backends_for_tier(backends, preferred)

    if not tier_backends:
        # Exact key match (e.g., user explicitly named a backend key)
        if preferred in backends:
            return preferred
        # Last resort: use any registered backend
        if backends:
            return next(iter(backends))
        return preferred  # caller returns 400

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


# ─────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────

def classify(payload: dict, backends: dict, config: "AppConfig") -> str:
    """
    Classify a request payload and return the backend key to use.

    Priority order:
      1. [route:key] explicit prefix in message
      2. Tool use / function calling → deep  (agentic tasks need capable models)
      3. Long prompt → deep
      4. Deep keywords → deep
      5. Code keywords / medium prompt → mid
      6. Default → fast
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

    # 2. Tool use / function calling → deep
    #    Small models produce unreliable JSON for tool calls and often
    #    fail multi-step agentic tasks entirely.
    if payload.get("tools") or payload.get("functions"):
        return _pick(backends, "deep")

    token_count = _token_estimate(content)

    # 3. Long prompt → deep
    if token_count > config.routing.token_threshold_deep:
        return _pick(backends, "deep")

    # 4. Deep keywords → deep
    if any(kw in content for kw in config.routing.deep_keywords):
        return _pick(backends, "deep")

    # 5. Mid keywords → mid
    if token_count > config.routing.token_threshold_mid:
        return _pick(backends, "mid")
    if any(kw in content for kw in config.routing.mid_keywords):
        return _pick(backends, "mid")

    # 6. Default: fast
    return _pick(backends, "fast")
