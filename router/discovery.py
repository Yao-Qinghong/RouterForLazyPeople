"""
router/discovery.py — Model auto-discovery

Scans configured directories for:
  - GGUF model files (for llama.cpp)
  - HuggingFace checkpoints (for vLLM / SGLang / HF TGI)
  - TensorRT-LLM engine directories

All scan paths and thresholds come from AppConfig — nothing is hardcoded here.
"""

import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from router.engines import (
    ENGINE_LLAMA, ENGINE_VLLM, ENGINE_SGLANG, ENGINE_HF, ENGINE_TRTLLM,
    is_engine_available,
)

if TYPE_CHECKING:
    from router.config import AppConfig


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


def _classify_tier(size_gb: float, config: "AppConfig") -> str:
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
                tier = _classify_tier(size_gb, config)

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
                    "description":   f"{name} ({size_gb:.1f} GB) [gguf, auto]",
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
            tier = _classify_tier(size_gb, config)

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
