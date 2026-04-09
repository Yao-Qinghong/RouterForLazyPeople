"""
router/registry.py — Backend registry builder

Merges manual backends (from backends.yaml) with auto-discovered models,
applies user overrides, and saves the discovery cache.
"""

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

from router.config import load_backends, BackendConfig
from router.discovery import (
    detect_running_servers,
    discover_gguf_models,
    discover_hf_models,
    discover_trtllm_engines,
)

if TYPE_CHECKING:
    from router.config import AppConfig

logger = logging.getLogger("llm-router.registry")


def load_user_overrides(config: "AppConfig") -> dict:
    """Load ~/.llm-router/overrides.json; return {} on missing or corrupt file."""
    overrides_path = config.data_dir / "overrides.json"
    if not overrides_path.exists():
        return {}
    try:
        with open(overrides_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load overrides.json: {e}")
        return {}


def save_discovery_cache(discovered: dict, config: "AppConfig"):
    """Persist the auto-discovered subset of the registry to disk."""
    cache_path = config.data_dir / "discovered.json"
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert BackendConfig dataclasses to dicts for JSON serialization
        serializable = {
            slug: asdict(cfg) if isinstance(cfg, BackendConfig) else cfg
            for slug, cfg in discovered.items()
        }
        with open(cache_path, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save discovery cache: {e}")


def build_backend_registry(config: "AppConfig") -> dict:
    """
    Build the full backend registry:
      1. Load manual backends from backends.yaml
      2. Discover GGUF, HF, and TRT-LLM models from scan dirs
      3. Merge (manual entries win on slug or path collision)
      4. Apply user overrides (exclude / patch)
      5. Save discovered-only subset to disk cache

    Returns dict[slug → backend_cfg].
    """
    # ── Manual backends ───────────────────────────────────────
    registry = load_backends(config)
    manual_count = len(registry)

    # ── Detect already-running servers (LM Studio, Ollama, …) ─
    running = detect_running_servers(config)
    for slug, cfg in running.items():
        if slug not in registry:
            registry[slug] = cfg

    # ── Auto-discovery (GGUF / HF / TRT-LLM on disk) ─────────
    port_counter = [config.discovery.port_start]

    gguf = discover_gguf_models(config, port_counter)
    hf   = discover_hf_models(config, port_counter)

    # TRT-LLM gets a separate range starting 200 above GGUF/HF
    trt_port_counter = [config.discovery.port_start + 200]
    trt  = discover_trtllm_engines(config, trt_port_counter)

    # ── Merge: manual wins on slug or path collision ──────────
    manual_paths = {v.get("model", "")     for v in registry.values()}
    manual_paths |= {v.get("model_dir", "") for v in registry.values()}

    discovered_all = {**gguf, **hf, **trt}
    discovered_added = {}

    for slug, cfg in discovered_all.items():
        if slug in registry:
            continue
        model_path = cfg.get("model", cfg.get("model_dir", ""))
        if model_path in manual_paths:
            continue
        registry[slug] = cfg
        discovered_added[slug] = cfg

    # ── User overrides ────────────────────────────────────────
    user = load_user_overrides(config)
    for slug in user.get("exclude", []):
        registry.pop(slug, None)
        discovered_added.pop(slug, None)
    for slug, patches in user.get("overrides", {}).items():
        if slug in registry:
            for attr, value in patches.items():
                if hasattr(registry[slug], attr):
                    setattr(registry[slug], attr, value)

    # ── Save discovery cache ──────────────────────────────────
    save_discovery_cache(discovered_added, config)

    logger.info(
        f"Registry built: {manual_count} manual + {len(discovered_added)} discovered "
        f"= {len(registry)} total backends"
    )
    return registry
