from __future__ import annotations

"""
router/benchmark.py — Backend speed benchmarking

Measures prompt-processing (PP) and token-generation (TG) speed for
each backend by sending fixed, reproducible requests and timing the
streaming response.

Results are saved to ~/.llm-router/benchmarks/<key>.json and used to:
  - Validate that a backend actually works
  - Show real tok/s instead of guessed file-size tiers
  - Warn when measured speed contradicts the assigned tier

Called by:
    cli.py  ./router-start bench [--backend key]
    GET /benchmarks  (summary of cached results)
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from router.config import AppConfig

logger = logging.getLogger("llm-router.benchmark")

# ── Fixed benchmark prompts ───────────────────────────────────
# PP prompt: long enough to stress the prefill stage (~400 tokens)
_PP_PROMPT = (
    "The following is a passage about machine learning. "
    "Neural networks are computational models inspired by the structure of the brain. "
    "They consist of layers of interconnected nodes that process information and learn patterns. "
    "Deep learning extends this by using many layers, enabling models to learn hierarchical "
    "representations of data. Transformers, introduced in 2017, revolutionised natural language "
    "processing by replacing recurrent architectures with attention mechanisms. "
    "Large language models trained on vast corpora of text can generate coherent prose, "
    "answer questions, write code, and reason about complex topics. "
) * 5   # repeat to get ~400 tokens

# TG prompt: short input so we measure generation, not prefill
_TG_PROMPT = (
    "Write exactly 80 words about why the ocean is important to life on Earth. "
    "Do not write more or less than 80 words."
)

_THINKING_PROMPT_SUFFIX = {
    "default": "",
    # Qwen-style directives are intentional here: they make benchmark mode
    # explicit when the model/template supports thinking, and are harmlessly
    # treated as plain user text by backends that do not.
    "no_think": "\n\n/no_think\nAnswer directly. Do not include chain-of-thought.",
    "think": "\n\n/think\nUse your reasoning mode before answering.",
}

# Tier boundaries based on measured TG speed (tok/s)
# These reflect practical usability, not arbitrary file-size thresholds:
#   fast  ≥ 30 tok/s  → snappy, suitable for interactive chat and quick tasks
#   mid   ≥ 10 tok/s  → acceptable for code generation (a few seconds per function)
#   deep  < 10 tok/s  → slow; worth the wait only for complex reasoning
TG_TIER_THRESHOLDS = {"fast": 30.0, "mid": 10.0}


# ─────────────────────────────────────────────────────────────
# Core measurement
# ─────────────────────────────────────────────────────────────

async def measure_backend(
    key: str,
    cfg: dict,
    config: "AppConfig",
    thinking_mode: str = "no_think",
) -> dict:
    """
    Run PP and TG benchmarks against a single backend.

    Sends two streaming requests directly to the backend (bypassing the
    router) so results reflect raw model speed, not routing overhead.

    Returns a result dict. On failure, sets 'error' field.
    """
    port  = cfg.get("port")
    engine = cfg.get("engine", "llama.cpp")
    base  = f"http://localhost:{port}/v1"
    if thinking_mode not in _THINKING_PROMPT_SUFFIX:
        raise ValueError(f"Unknown thinking_mode '{thinking_mode}'")

    result = {
        "backend_key":  key,
        "engine":       engine,
        "description":  cfg.get("description", key),
        "tier_assigned": cfg.get("tier", "unknown"),
        "pp_tok_s":     None,
        "tg_tok_s":     None,
        "ttft_ms":      None,
        "validated":    False,
        "error":        None,
        "pp_error":     None,
        "tg_error":     None,
        "pp_validated": False,
        "tier_measured": None,
        "tier_mismatch": False,
        "thinking_mode": thinking_mode,
    }

    # PP phase (independent)
    try:
        pp = await _run_pp(base, thinking_mode, timeout=config.benchmark.pp_timeout_sec)
        result["pp_tok_s"] = pp["pp_tok_s"]
        result["ttft_ms"]  = pp["ttft_ms"]
        result["pp_validated"] = True
    except Exception as e:
        result["pp_error"] = str(e)

    # TG phase (independent)
    # Use benchmark data from similar models to set a realistic timeout
    tg_timeout = _estimate_tg_timeout(
        cfg.get("size_gb"), config,
        fallback=config.benchmark.tg_timeout_sec,
    )
    try:
        tg = await _run_tg(base, thinking_mode, timeout=tg_timeout)
        result["tg_tok_s"] = tg["tg_tok_s"]
    except Exception as e:
        result["tg_error"] = str(e)

    # Summary
    if result["pp_validated"] and result["tg_tok_s"] is not None:
        result["validated"] = True
    if result["tg_tok_s"] is not None:
        result["tier_measured"] = _tier_from_speed(result["tg_tok_s"])
        result["tier_mismatch"] = (
            result["tier_measured"] != result["tier_assigned"]
            and result["tier_assigned"] is not None
        )
    # Legacy "error" field: only set when BOTH phases failed
    if result["pp_error"] and result["tg_error"]:
        result["error"] = f"PP: {result['pp_error']}; TG: {result['tg_error']}"

    return result


def _benchmark_prompt(prompt: str, thinking_mode: str) -> str:
    return prompt + _THINKING_PROMPT_SUFFIX[thinking_mode]


async def _run_pp(base: str, thinking_mode: str = "no_think", timeout: int = 90) -> dict:
    """
    PP benchmark: long prompt, max_tokens=1, streaming.
    Measures time-to-first-token which equals prompt processing time.
    """
    payload = {
        "model":      "benchmark",
        "messages":   [{"role": "user", "content": _benchmark_prompt(_PP_PROMPT, thinking_mode)}],
        "max_tokens": 1,
        "stream":     True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    ttft_ms = None
    prompt_tokens = len(_PP_PROMPT.split())   # rough estimate

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{base}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000

    if ttft_ms is None:
        raise RuntimeError("No tokens received from PP benchmark")

    pp_tok_s = round(prompt_tokens / (ttft_ms / 1000), 1)
    return {"ttft_ms": round(ttft_ms, 1), "pp_tok_s": pp_tok_s}


_TG_MIN_TOKENS = 5    # absolute minimum for any measurement
_TG_ENOUGH_TOKENS = 20  # enough for a reliable estimate — exit early
_TG_TIMEOUT_SAFETY = 5  # multiply estimated time by this for the timeout


def _estimate_tg_timeout(
    size_gb: float | None,
    config: "AppConfig",
    fallback: int = 300,
) -> int:
    """Estimate a TG timeout using benchmarked models of similar size.

    If we already have TG results for other models, extrapolate: larger
    models are slower roughly in proportion to file size.  The estimated
    generation time is multiplied by ``_TG_TIMEOUT_SAFETY`` so we don't
    cut it too close.  Returns the configured fallback when no prior
    data exists.
    """
    if size_gb is None or size_gb <= 0:
        return fallback

    all_results = load_all_results(config)
    # Collect (size_gb, tg_tok_s) pairs from prior benchmarks
    refs: list[tuple[float, float]] = []
    for r in all_results.values():
        tg = r.get("tg_tok_s")
        if tg and tg > 0:
            # Look up this backend's size from the description or
            # from the config registry; fall back to skipping.
            ref_size = _extract_size_from_result(r, config)
            if ref_size and ref_size > 0:
                refs.append((ref_size, tg))

    if not refs:
        return fallback

    # Pick the reference closest in size to the target
    refs.sort(key=lambda pair: abs(pair[0] - size_gb))
    ref_size, ref_tg = refs[0]

    # Estimate: speed scales roughly inversely with size
    estimated_tg = ref_tg * (ref_size / size_gb)
    if estimated_tg <= 0:
        return fallback

    # Time to generate _TG_ENOUGH_TOKENS at estimated speed, with safety margin
    estimated_sec = (_TG_ENOUGH_TOKENS / estimated_tg) * _TG_TIMEOUT_SAFETY
    # Clamp: at least 30s, at most the configured fallback
    timeout = max(30, min(int(estimated_sec), fallback))
    logger.info(
        "TG timeout estimate for %.1f GB model: %.1f tok/s (from %.1f GB ref @ %.1f tok/s) → %ds",
        size_gb, estimated_tg, ref_size, ref_tg, timeout,
    )
    return timeout


def _extract_size_from_result(result: dict, config: "AppConfig") -> float | None:
    """Get the model size_gb for a benchmark result by looking up the backend registry."""
    key = result.get("backend_key")
    if not key:
        return None
    # Try the live backend registry via config's backends file
    try:
        from router.registry import build_backend_registry
        backends = build_backend_registry(config)
        cfg = backends.get(key, {})
        if isinstance(cfg, dict):
            return cfg.get("size_gb")
        return getattr(cfg, "size_gb", None)
    except Exception:
        return None


async def _run_tg(base: str, thinking_mode: str = "no_think", timeout: int = 300) -> dict:
    """
    TG benchmark: short prompt, max_tokens=80, streaming.
    Measures token generation speed after the first token.

    Exits early once _TG_ENOUGH_TOKENS are collected (no need to wait
    for all 80).  On timeout, uses whatever tokens were already received
    as long as there are at least _TG_MIN_TOKENS.
    """
    payload = {
        "model":       "benchmark",
        "messages":    [{"role": "user", "content": _benchmark_prompt(_TG_PROMPT, thinking_mode)}],
        "max_tokens":  80,
        "stream":      True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    ttft_s = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{base}/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:") or "[DONE]" in line:
                        continue
                    try:
                        chunk = json.loads(line[5:])
                        delta = chunk["choices"][0]["delta"]
                        # Count ANY generated token — content, reasoning_content,
                        # or thinking.  For benchmarking we only care about speed,
                        # not which field the model writes to.
                        text = (
                            delta.get("content", "")
                            or delta.get("reasoning_content", "")
                        )
                        if text:
                            if ttft_s is None:
                                ttft_s = time.perf_counter() - t0
                            token_count += 1
                            if token_count >= _TG_ENOUGH_TOKENS:
                                break
                    except Exception as exc:
                        logger.debug("TG chunk parse error: %s — line: %s", exc, line[:120])
                        continue
    except (httpx.TimeoutException, httpx.ReadTimeout):
        if token_count >= _TG_MIN_TOKENS:
            logger.warning("TG benchmark timed out after %d tokens — using partial data", token_count)
        else:
            raise RuntimeError(
                f"TG timed out after {token_count} token(s) — need at least {_TG_MIN_TOKENS}"
            )

    if token_count < _TG_MIN_TOKENS:
        raise RuntimeError(f"Too few tokens generated ({token_count}) — model may have refused")

    total_s = time.perf_counter() - t0
    tg_s = total_s - (ttft_s or 0)
    if tg_s <= 0:
        tg_s = total_s
    tg_tok_s = round(token_count / tg_s, 1)
    return {"tg_tok_s": tg_tok_s, "token_count": token_count}


def _tier_from_speed(tg_tok_s: float) -> str:
    """Assign tier based on measured TG speed."""
    if tg_tok_s >= TG_TIER_THRESHOLDS["fast"]:
        return "fast"
    elif tg_tok_s >= TG_TIER_THRESHOLDS["mid"]:
        return "mid"
    return "deep"


# ─────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────

def save_result(result: dict, config: "AppConfig"):
    bench_dir = config.data_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / f"{result['backend_key']}.json"
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(key: str, config: "AppConfig") -> dict | None:
    path = config.data_dir / "benchmarks" / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_all_results(config: "AppConfig") -> dict[str, dict]:
    bench_dir = config.data_dir / "benchmarks"
    if not bench_dir.exists():
        return {}
    results = {}
    for p in bench_dir.glob("*.json"):
        try:
            with open(p) as f:
                r = json.load(f)
            results[r["backend_key"]] = r
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────────────────────
# Summary formatting (used by CLI)
# ─────────────────────────────────────────────────────────────

def format_results(results: list[dict]) -> str:
    lines = []
    lines.append(
        f"{'Backend':<22} {'Think':<8} {'PP tok/s':>9} {'TG tok/s':>9} "
        f"{'TTFT ms':>8} {'Tier set':>9} {'Measured':>9} {'Status'}"
    )
    lines.append("─" * 100)

    for r in results:
        thinking = r.get("thinking_mode", "—")

        # Total failure: both PP and TG failed (or legacy error with no PP data)
        if r.get("error") and not r.get("pp_tok_s"):
            lines.append(f"{r['backend_key']:<22} {thinking:<8}  ERROR: {r['error'][:55]}")
            continue

        pp   = f"{r['pp_tok_s']:.0f}" if r.get("pp_tok_s") else "—"
        tg   = f"{r['tg_tok_s']:.0f}" if r.get("tg_tok_s") else "—"
        ttft = f"{r['ttft_ms']:.0f}" if r.get("ttft_ms") else "—"
        assigned = r.get("tier_assigned") or "—"
        measured = r.get("tier_measured") or "—"

        if r.get("tg_error"):
            status = f"PP only (TG failed: {r['tg_error'][:40]})"
        elif r.get("tier_mismatch"):
            status = f"⚠  assigned={assigned} but speed suggests {measured}"
        else:
            status = "✓"

        lines.append(
            f"{r['backend_key']:<22} {thinking:<8} {pp:>9} {tg:>9} {ttft:>8} "
            f"{assigned:>9} {measured:>9}  {status}"
        )

    lines.append("")
    lines.append("TG speed tiers:  fast ≥ 30 tok/s  |  mid ≥ 10 tok/s  |  deep < 10 tok/s")
    lines.append("PP = prompt processing speed   TG = token generation speed")
    lines.append("Think = benchmark prompt mode: no_think adds /no_think, think adds /think")
    return "\n".join(lines)
