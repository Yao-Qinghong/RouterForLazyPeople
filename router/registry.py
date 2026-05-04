"""
router/registry.py — Backend registry builder

Merges manual backends (from backends.yaml) with auto-discovered models,
applies user overrides, and saves the discovery cache.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from router.config import load_backends, BackendConfig
from router.discovery import (
    detect_running_servers,
    discover_gguf_models,
    discover_hf_models,
    discover_trtllm_engines,
)
from router.engines import (
    ALL_ENGINES,
    ENGINE_LLAMA,
    ENGINE_HF,
    ENGINE_TRTLLM,
    ENGINE_TRTLLM_DOCKER,
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
    enabled = set(getattr(config, "engines_enabled", []) or ALL_ENGINES)
    running = detect_running_servers(config)
    for slug, cfg in running.items():
        if slug not in registry and cfg.get("engine", "") in enabled:
            registry[slug] = cfg

    # ── Auto-discovery (GGUF / HF / TRT-LLM on disk) ─────────
    port_counter = [config.discovery.port_start]

    gguf = discover_gguf_models(config, port_counter) if ENGINE_LLAMA in enabled else {}
    hf   = discover_hf_models(config, port_counter)   if ENGINE_HF in enabled else {}

    # TRT-LLM gets a separate range starting 200 above GGUF/HF
    trt_port_counter = [config.discovery.port_start + 200]
    trt  = discover_trtllm_engines(config, trt_port_counter) if ENGINE_TRTLLM in enabled or ENGINE_TRTLLM_DOCKER in enabled else {}

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


# ─────────────────────────────────────────────────────────────
# Registry validation (used by `bench` to skip stale entries
# without crashing the whole run on a single bad backend).
# `load_backends` already raises ConfigError for fatal issues
# in manual entries; this layer also catches issues that arise
# *after* merging — e.g. a discovered backend that lands on the
# same port as a manual one, or a manual model path that has
# since been deleted.
# ─────────────────────────────────────────────────────────────

_LOCAL_ENGINES_NEED_MODEL = {"llama.cpp", "vllm", "sglang", "huggingface"}
# llama.cpp loads a single GGUF file from disk, so its `model` value must
# always resolve to an existing path. vLLM/SGLang/Hugging Face engines
# accept either a local directory *or* a Hugging Face model ID like
# ``Qwen/Qwen2.5-72B-Instruct-AWQ``; only a value that actually looks
# like a filesystem path is worth checking for existence.
_ALWAYS_LOCAL_PATH_ENGINES = {"llama.cpp"}


def _looks_like_local_path(value: str) -> bool:
    """Return True if ``value`` looks like a filesystem path, not an HF ID.

    Hugging Face IDs are ``<org>/<repo>`` — no leading ``/``, ``~``, ``.``,
    no backslashes. Anything else with those characteristics we treat as
    a local path that should exist on disk.
    """
    if not value:
        return False
    return value.startswith(("/", "~", "./", "../")) or "\\" in value


@dataclass
class RegistryValidation:
    """Structured findings from validating a built backend registry.

    Non-fatal: callers (typically the bench command) decide what to do
    with the findings. Logging and skip behavior are external choices.
    """

    missing_paths: list[tuple[str, str]] = field(default_factory=list)
    port_conflicts: dict[int, list[str]] = field(default_factory=dict)
    invalid_keys: set[str] = field(default_factory=set)

    @property
    def has_issues(self) -> bool:
        return bool(self.missing_paths or self.port_conflicts)

    def summary_lines(self) -> list[str]:
        """Concise human-readable summary; one issue per line."""
        lines: list[str] = []
        for slug, path in self.missing_paths:
            lines.append(f"  invalid: {slug} — model path does not exist: {path}")
        for port, slugs in sorted(self.port_conflicts.items()):
            lines.append(
                f"  invalid: port {port} reused by {len(slugs)} backends: "
                f"{', '.join(sorted(slugs))}"
            )
        return lines


def _backend_field(cfg: object, name: str, default=None):
    """Read a field from either a dict-style or BackendConfig-style entry."""
    if hasattr(cfg, "get"):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def validate_registry(backends: dict) -> RegistryValidation:
    """Inspect a built registry for benchmark-blocking issues.

    Detects, *once* per call:
      • manual backends whose model file/dir no longer exists on disk
        (only meaningful for engines that need a local path)
      • backends sharing the same TCP port (the post-merge case that
        ``load_backends`` cannot see, e.g. discovery colliding with a
        manual entry)

    Does not log and does not raise. The caller decides whether to skip
    the affected keys or surface the report to the user.
    """
    report = RegistryValidation()

    for slug, cfg in backends.items():
        engine = _backend_field(cfg, "engine", "")
        if engine not in _LOCAL_ENGINES_NEED_MODEL:
            continue

        model_dir = _backend_field(cfg, "model_dir")
        model = _backend_field(cfg, "model")

        # Choose what (if anything) to check for existence:
        #   - `model_dir` is always a local directory.
        #   - `model` is local for llama.cpp (GGUF file). For vLLM/SGLang/HF
        #     it may be a HuggingFace ID, in which case we skip the check.
        path_to_check: str | None = None
        if model_dir:
            path_to_check = str(model_dir)
        elif model:
            model_str = str(model)
            if engine in _ALWAYS_LOCAL_PATH_ENGINES or _looks_like_local_path(model_str):
                path_to_check = model_str

        if path_to_check is None:
            # Either no path at all, or an HF model ID that we cannot verify
            # locally. llama.cpp must have a local path; HF-style engines are
            # allowed to use IDs.
            if engine in _ALWAYS_LOCAL_PATH_ENGINES and not (model or model_dir):
                report.missing_paths.append((slug, ""))
                report.invalid_keys.add(slug)
            continue

        if not os.path.exists(os.path.expanduser(path_to_check)):
            report.missing_paths.append((slug, path_to_check))
            report.invalid_keys.add(slug)

    # Port collisions across the merged registry. Skip entries already
    # invalid for other reasons — a stale manual backend that shares a
    # port with a working sibling shouldn't drag the sibling down: the
    # stale one is already excluded, leaving the port free.
    by_port: dict[int, list[str]] = {}
    for slug, cfg in backends.items():
        if slug in report.invalid_keys:
            continue
        port = _backend_field(cfg, "port")
        if port is None:
            continue
        try:
            port_int = int(port)
        except (TypeError, ValueError):
            continue
        by_port.setdefault(port_int, []).append(slug)

    for port, slugs in by_port.items():
        if len(slugs) > 1:
            report.port_conflicts[port] = sorted(slugs)
            report.invalid_keys.update(slugs)

    return report
