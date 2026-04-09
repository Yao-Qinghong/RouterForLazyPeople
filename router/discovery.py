from __future__ import annotations

"""
router/discovery.py — Model auto-discovery

Scans configured directories for:
  - GGUF model files (for llama.cpp)
  - HuggingFace checkpoints (for vLLM / SGLang / HF TGI)
  - TensorRT-LLM engine directories

Also probes for already-running inference servers:
  - LM Studio (port 1234, OpenAI-compatible)
  - Ollama (port 11434)
  - Any other OpenAI-compatible server on user-configured ports

All scan paths and thresholds come from AppConfig — nothing is hardcoded here.
"""

import concurrent.futures
import glob
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from router.engines import (
    ENGINE_LLAMA, ENGINE_VLLM, ENGINE_SGLANG, ENGINE_HF, ENGINE_TRTLLM,
    ENGINE_OLLAMA, ENGINE_OPENAI,
    is_engine_available,
)

if TYPE_CHECKING:
    from router.config import AppConfig

logger = logging.getLogger("llm-router.discovery")


# ─────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────

def _file_size_gb(path: str) -> float:
    """Total size in GB, handling multi-part GGUF files."""
    p = Path(path)
    match = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', p.name)
    if match:
        pattern = re.sub(r'-\d{5}-of-\d{5}\.gguf$', '-*-of-*.gguf', str(p))
        parts = glob.glob(pattern)
        return sum(os.path.getsize(f) for f in parts) / (1024 ** 3)
    return os.path.getsize(path) / (1024 ** 3)


def _dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 ** 3)


def _model_name_from_path(path: str) -> str:
    name = Path(path).stem
    name = re.sub(r'-\d{5}-of-\d{5}$', '', name)
    name = re.sub(r'-(UD|ud)$', '', name)
    return name


def _slug(name: str, path: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if len(slug) > 40:
        slug = slug[:35] + '-' + hashlib.md5(path.encode()).hexdigest()[:4]
    return slug


def _moe_active_params(name: str) -> float | None:
    """
    Extract the active (routed) parameter count from a MoE model name.
    Returns None if the model is not recognisably MoE.

    Patterns handled:
      35B-A3B   → 3.0   (Qwen MoE format: total-Aactive)
      122B-A10B → 10.0
      8x7B      → 7.0   (Mixtral format: experts x params_per_expert)
      8x22B     → 22.0
    """
    # Qwen / DeepSeek style: <total>B-A<active>B
    m = re.search(r'\d+(?:\.\d+)?[Bb]-[Aa](\d+(?:\.\d+)?)[Bb]', name)
    if m:
        return float(m.group(1))
    # Mixtral style: <n>x<size>B  (one expert active per token)
    m = re.search(r'\d+x(\d+(?:\.\d+)?)[Bb]', name)
    if m:
        return float(m.group(1))
    return None


def _classify_tier(size_gb: float, config: "AppConfig", name: str = "") -> str:
    """
    Classify a model into fast / mid / deep.

    MoE models are classified by their *active* parameter count, not total
    file size — a 35B-A3B MoE activates only ~3B params and is as fast as
    a dense 3B model.

    Dense models fall back to file-size thresholds (configurable in
    settings.yaml → tier_thresholds_gb).

    MoE active-param thresholds (not yet configurable — sensible defaults):
      < 7B active  → fast
      7–20B active → mid
      > 20B active → deep
    """
    active = _moe_active_params(name)
    if active is not None:
        if active < 7:
            return "fast"
        elif active < 20:
            return "mid"
        return "deep"

    # Dense model — use file size as proxy for parameter count
    if size_gb < config.tier_thresholds.fast:
        return "fast"
    elif size_gb < config.tier_thresholds.mid:
        return "mid"
    return "deep"


def _estimate_ctx(size_gb: float) -> int:
    if size_gb < 30:  return 131072
    if size_gb < 60:  return 65536
    return 32768


def _estimate_startup(size_gb: float) -> int:
    if size_gb < 15:  return 30
    if size_gb < 40:  return 45
    if size_gb < 80:  return 90
    return 120


def _estimate_idle(tier: str, config: "AppConfig") -> int:
    return {
        "fast": config.idle_timeouts.fast,
        "mid":  config.idle_timeouts.mid,
        "deep": config.idle_timeouts.deep,
    }.get(tier, config.idle_timeouts.fast)


def _has_reasoning_kw(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in ["reason", "r1", "distill", "cot", "think", "122b", "70b"])


def _is_hf_checkpoint(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    has_config = (p / "config.json").exists()
    has_weights = any(
        (p / f).exists() for f in [
            "model.safetensors", "model.safetensors.index.json",
            "pytorch_model.bin", "pytorch_model.bin.index.json",
            "model-00001-of-00001.safetensors",
        ]
    ) or list(p.glob("model-*.safetensors"))
    return has_config and has_weights


def _hf_model_size_gb(path: str) -> float:
    total = 0
    for ext in ["*.safetensors", "*.bin"]:
        for f in Path(path).glob(ext):
            total += f.stat().st_size
    return total / (1024 ** 3)


def _read_hf_config(path: str) -> dict:
    try:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# Discovery functions
# ─────────────────────────────────────────────────────────────

def discover_gguf_models(config: "AppConfig", port_counter: list[int]) -> dict:
    """Scan GGUF scan dirs for .gguf model files."""
    discovered = {}
    seen_paths = set()

    for scan_dir in config.scan_dirs.gguf:
        scan_dir = str(scan_dir)
        if not os.path.isdir(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            gguf_files = [f for f in files if f.endswith('.gguf')]
            if not gguf_files:
                continue

            # Only take the first shard of multi-part files
            primaries = []
            for f in gguf_files:
                m = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', f)
                if m:
                    if m.group(1) == '00001':
                        primaries.append(f)
                else:
                    primaries.append(f)

            for filename in primaries:
                full_path = os.path.join(root, filename)
                if full_path in seen_paths:
                    continue
                seen_paths.add(full_path)

                try:
                    size_gb = _file_size_gb(full_path)
                except OSError:
                    continue
                if size_gb < 0.5:
                    continue
                if port_counter[0] > config.discovery.port_end:
                    break

                name = _model_name_from_path(full_path)
                slug = _slug(name, full_path)
                tier = _classify_tier(size_gb, config, name)
                active = _moe_active_params(name)
                if active is not None:
                    tier_reason = f"MoE, {active:.0f}B active"
                else:
                    tier_reason = f"{size_gb:.1f} GB"

                discovered[slug] = {
                    "engine":        ENGINE_LLAMA,
                    "port":          port_counter[0],
                    "model":         full_path,
                    "log":           str(config.data_dir / "logs" / f"backend-{slug}.log"),
                    "ctx_size":      _estimate_ctx(size_gb),
                    "gpu_layers":    999,
                    "flash_attn":    True,
                    "reasoning":     _has_reasoning_kw(name),
                    "idle_timeout":  _estimate_idle(tier, config),
                    "startup_wait":  _estimate_startup(size_gb),
                    "description":   f"{name} ({tier_reason})",
                    "auto_discovered": True,
                    "tier":          tier,
                    "size_gb":       round(size_gb, 2),
                }
                port_counter[0] += 1

    return discovered


def discover_hf_models(config: "AppConfig", port_counter: list[int]) -> dict:
    """Discover HuggingFace checkpoints and pick the best available engine."""
    # Choose the best available engine for HF models
    if is_engine_available(ENGINE_VLLM, config):
        preferred_engine = ENGINE_VLLM
    elif is_engine_available(ENGINE_SGLANG, config):
        preferred_engine = ENGINE_SGLANG
    elif is_engine_available(ENGINE_HF, config):
        preferred_engine = ENGINE_HF
    else:
        return {}

    discovered = {}
    seen_paths = set()

    for scan_dir in config.scan_dirs.hf:
        scan_dir = str(scan_dir)
        if not os.path.isdir(scan_dir):
            continue

        candidates = []
        for entry in os.scandir(scan_dir):
            if entry.is_dir():
                candidates.append(entry.path)
                try:
                    for sub in os.scandir(entry.path):
                        if sub.is_dir():
                            candidates.append(sub.path)
                except PermissionError:
                    pass

        # HF cache: ~/.cache/huggingface/hub/models--org--name/snapshots/hash/
        if "huggingface" in scan_dir:
            for entry in Path(scan_dir).glob("models--*/snapshots/*"):
                if entry.is_dir():
                    candidates.append(str(entry))

        for path in candidates:
            if path in seen_paths:
                continue
            if not _is_hf_checkpoint(path):
                continue
            seen_paths.add(path)

            size_gb = _hf_model_size_gb(path)
            if size_gb < 0.3:
                continue
            if port_counter[0] > config.discovery.port_end:
                break

            hf_config = _read_hf_config(path)
            arch = (hf_config.get("architectures") or ["Unknown"])[0]
            model_type = hf_config.get("model_type", "unknown")

            dir_name = Path(path).name
            if "snapshots" in path:
                for part in Path(path).parts:
                    if part.startswith("models--"):
                        dir_name = part.replace("models--", "").replace("--", "/")
                        break

            slug = _slug(f"hf-{dir_name}", path)
            tier = _classify_tier(size_gb, config, dir_name)

            discovered[slug] = {
                "engine":              preferred_engine,
                "port":                port_counter[0],
                "model":               path,
                "log":                 str(config.data_dir / "logs" / f"backend-{slug}.log"),
                "ctx_size":            min(_estimate_ctx(size_gb), hf_config.get("max_position_embeddings", 131072)),
                "dtype":               "auto",
                "gpu_memory_fraction": 0.90,
                "trust_remote_code":   True,
                "idle_timeout":        _estimate_idle(tier, config),
                "startup_wait":        _estimate_startup(size_gb) + 30,
                "description":         f"{dir_name} ({size_gb:.1f} GB, {arch}) [{preferred_engine}, auto]",
                "auto_discovered":     True,
                "tier":                tier,
                "size_gb":             round(size_gb, 2),
                "model_type":          model_type,
            }
            port_counter[0] += 1

    return discovered


def detect_running_servers(config: "AppConfig") -> dict:
    """
    Probe for already-running LLM inference servers and register them.

    Checks (in parallel):
      - LM Studio on port 1234
      - Ollama on port 11434
      - llama.cpp / vLLM / SGLang / TRT-LLM on ports in discovery.probe_ports

    Returns a backend dict ready to merge into the registry.
    No subprocess is spawned — these servers are already running.
    """
    backends = {}
    backends.update(_probe_lmstudio(config))
    backends.update(_probe_ollama_models(config))
    backends.update(_probe_openai_servers(config))
    return backends


def _probe_lmstudio(config: "AppConfig") -> dict:
    """Detect LM Studio and register it as a passthrough backend."""
    try:
        import httpx
        r = httpx.get("http://localhost:1234/v1/models", timeout=2)
        if r.status_code != 200:
            return {}
        models = r.json().get("data", [])
        model_id = models[0]["id"] if models else "unknown"
        logger.info(f"Auto-detected: LM Studio on :1234  (model: {model_id})")
        return {
            "lmstudio": {
                "engine":         ENGINE_OPENAI,
                "port":           1234,
                "model":          model_id,
                "tier":           "fast",
                "idle_timeout":   86400,    # never auto-stop a server we don't own
                "startup_wait":   5,
                "auto_discovered": True,
                "description":    f"LM Studio — {model_id}",
                "log":            str(config.data_dir / "logs" / "backend-lmstudio.log"),
            }
        }
    except Exception:
        return {}


def _probe_ollama_models(config: "AppConfig") -> dict:
    """Detect Ollama and register each available model as a backend."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return {}
        models = r.json().get("models", [])
        if not models:
            return {}
        logger.info(f"Auto-detected: Ollama on :11434  ({len(models)} model(s))")
        result = {}
        for m in models:
            name = m["name"]
            slug = "ollama-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            tier = _tier_from_model_name(name)
            result[slug] = {
                "engine":         ENGINE_OLLAMA,
                "port":           11434,
                "model":          name,
                "tier":           tier,
                "idle_timeout":   86400,
                "startup_wait":   30,
                "auto_discovered": True,
                "description":    f"Ollama — {name}",
                "log":            str(config.data_dir / "logs" / f"backend-{slug}.log"),
            }
        return result
    except Exception:
        return {}


def _probe_openai_servers(config: "AppConfig") -> dict:
    """
    Probe discovery.probe_ports for running llama.cpp / vLLM / SGLang / TRT-LLM servers.

    All probes run in parallel with a 1-second timeout each so startup is not slowed down.
    Any port that responds to GET /v1/models is registered as an openai-passthrough backend.
    Ports already claimed by LM Studio (1234) or Ollama (11434) are skipped.
    """
    # Ports already handled by dedicated probes or reserved
    skip_ports: set[int] = {1234, 11434}
    ports = [p for p in config.discovery.probe_ports if p not in skip_ports]
    if not ports:
        return {}

    def _probe_one(port: int) -> dict:
        try:
            import httpx
            r = httpx.get(f"http://localhost:{port}/v1/models", timeout=1.0)
            if r.status_code != 200:
                return {}
            data = r.json().get("data", [])
            model_id = data[0]["id"] if data else "unknown"
            slug = f"local-{port}"
            # Try to guess engine from /health response headers or body
            label = _guess_engine_label(port)
            logger.info(f"Auto-detected: {label} on :{port}  (model: {model_id})")
            return {
                slug: {
                    "engine":         ENGINE_OPENAI,
                    "port":           port,
                    "model":          model_id,
                    "tier":           _tier_from_model_name(model_id),
                    "idle_timeout":   86400,
                    "startup_wait":   5,
                    "auto_discovered": True,
                    "description":    f"{label} :{port} — {model_id}",
                    "log":            str(config.data_dir / "logs" / f"backend-{slug}.log"),
                }
            }
        except Exception:
            return {}

    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as pool:
        for backend in pool.map(_probe_one, ports):
            result.update(backend)
    return result


def _guess_engine_label(port: int) -> str:
    """
    Best-effort label for what engine is likely running on a given port.
    Based on common defaults only — not guaranteed to be correct.
    """
    labels = {
        8080: "llama.cpp",
        8000: "vLLM/SGLang",
        8001: "vLLM/SGLang",
        8002: "TRT-LLM",
        30000: "SGLang",
    }
    return labels.get(port, f"LLM server")


def _tier_from_model_name(name: str) -> str:
    """
    Tier assignment from model name alone (used for Ollama and other
    sources where file size is not available).

    MoE models are detected first via active-param count.
    Dense models fall back to total-param size tokens.
    """
    # MoE: use active params
    active = _moe_active_params(name)
    if active is not None:
        if active < 7:
            return "fast"
        elif active < 20:
            return "mid"
        return "deep"

    # Dense: classify by total param count in name
    n = name.lower()
    for token in ("70b", "72b", "65b", "80b", "110b", "122b", "123b", "178b", "671b"):
        if token in n:
            return "deep"
    for token in ("13b", "14b", "27b", "30b", "32b", "34b", "47b"):
        if token in n:
            return "mid"
    return "fast"


def discover_trtllm_engines(config: "AppConfig", port_counter: list[int]) -> dict:
    """Discover TensorRT-LLM engine directories."""
    discovered = {}

    for scan_dir in config.scan_dirs.trtllm:
        scan_dir = str(scan_dir)
        if not os.path.isdir(scan_dir):
            continue
        for entry in os.scandir(scan_dir):
            if not entry.is_dir():
                continue
            config_path = os.path.join(entry.path, "config.json")
            engine_files = glob.glob(os.path.join(entry.path, "*.engine"))
            if not os.path.exists(config_path) or not engine_files:
                continue
            if port_counter[0] > config.discovery.port_end:
                break

            slug = _slug(f"trt-{entry.name}", entry.path)
            description = entry.name
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                arch = cfg.get("pretrained_config", {}).get("architecture", entry.name)
                description = f"{arch} [trt-llm, auto]"
            except Exception:
                pass

            # Find tokenizer directory
            tokenizer_dir = ""
            for candidate in [
                os.path.join(entry.path, "tokenizer"),
                os.path.join(scan_dir, f"{entry.name}-hf"),
                os.path.join(scan_dir, f"{entry.name}-tokenizer"),
            ]:
                if os.path.isdir(candidate) and (
                    os.path.exists(os.path.join(candidate, "tokenizer.json"))
                    or os.path.exists(os.path.join(candidate, "tokenizer.model"))
                ):
                    tokenizer_dir = candidate
                    break

            discovered[slug] = {
                "engine":        ENGINE_TRTLLM,
                "port":          port_counter[0],
                "model_dir":     entry.path,
                "tokenizer":     tokenizer_dir,
                "log":           str(config.data_dir / "logs" / f"backend-{slug}.log"),
                "idle_timeout":  600,
                "startup_wait":  120,
                "description":   description,
                "auto_discovered": True,
                "trt_config": {
                    "max_batch_size": 1,
                    "max_input_len":  32768,
                    "max_output_len": 4096,
                    "kv_cache_dtype": "int8",
                    "chunked_context": True,
                    "enable_mtp":    False,
                    "max_beam_width": 1,
                    "gpu_memory_fraction": 0.90,
                },
            }
            port_counter[0] += 1

    return discovered
