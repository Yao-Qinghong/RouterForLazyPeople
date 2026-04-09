#!/usr/bin/env python3
"""
LLM Router — port 9001

Multi-engine local LLM router with:
  - llama.cpp, TensorRT-LLM, vLLM, SGLang, HuggingFace Transformers
  - Auto-discovery of local models (GGUF, HF checkpoints, TRT engines)
  - On-demand start/stop with idle timeouts
  - TRT-LLM self-tuning for memory-optimal configs
  - User overrides via ~/.llm-router/overrides.json
"""

import asyncio
import glob
import hashlib
import httpx
import json
import logging
import os
import re
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# ─────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
LLAMA_BIN = f"{HOME}/llama.cpp/build/bin/llama-server"
DATA_DIR = f"{HOME}/.llm-router"
TUNE_DIR = f"{DATA_DIR}/tuning"
DISCOVERY_CACHE = f"{DATA_DIR}/discovered.json"
USER_OVERRIDES = f"{DATA_DIR}/overrides.json"
os.makedirs(TUNE_DIR, exist_ok=True)

# Engine identifiers
ENGINE_LLAMA  = "llama.cpp"
ENGINE_TRTLLM = "trt-llm"
ENGINE_VLLM   = "vllm"
ENGINE_SGLANG = "sglang"
ENGINE_HF     = "huggingface"

ALL_ENGINES = [ENGINE_LLAMA, ENGINE_TRTLLM, ENGINE_VLLM, ENGINE_SGLANG, ENGINE_HF]

# ─────────────────────────────────────────────────────────────
# ENGINE AVAILABILITY — detect what's installed
# ─────────────────────────────────────────────────────────────
_engine_available_cache: dict[str, bool] = {}

def is_engine_available(engine: str) -> bool:
    """Check if an engine's binary/module is installed."""
    if engine in _engine_available_cache:
        return _engine_available_cache[engine]

    checks = {
        ENGINE_LLAMA:  lambda: os.path.isfile(LLAMA_BIN),
        ENGINE_VLLM:   lambda: shutil.which("vllm") is not None or _can_import("vllm"),
        ENGINE_SGLANG: lambda: shutil.which("sglang") is not None or _can_import("sglang"),
        ENGINE_TRTLLM: lambda: _can_import("tensorrt_llm"),
        ENGINE_HF:     lambda: _can_import("transformers"),
    }

    result = checks.get(engine, lambda: False)()
    _engine_available_cache[engine] = result
    return result

def _can_import(module: str) -> bool:
    try:
        subprocess.run(
            ["python3", "-c", f"import {module}"],
            capture_output=True, timeout=10,
        )
        return True
    except Exception:
        return False

def available_engines() -> list[str]:
    return [e for e in ALL_ENGINES if is_engine_available(e)]


# ─────────────────────────────────────────────────────────────
# ENGINE-SPECIFIC COMMAND BUILDERS
# ─────────────────────────────────────────────────────────────

def build_llama_cmd(cfg: dict) -> list[str]:
    cmd = [
        LLAMA_BIN,
        "--model",        cfg["model"],
        "--ctx-size",     str(cfg.get("ctx_size", 32768)),
        "--n-gpu-layers", str(cfg.get("gpu_layers", 999)),
        "--flash-attn",   "on" if cfg.get("flash_attn", True) else "off",
        "--reasoning",    "on" if cfg.get("reasoning", False) else "off",
        "--host",         "0.0.0.0",
        "--port",         str(cfg["port"]),
    ]
    if cfg.get("reasoning_budget"):
        cmd += ["--reasoning-budget", str(cfg["reasoning_budget"])]
    return cmd


def build_vllm_cmd(cfg: dict) -> list[str]:
    """
    vLLM OpenAI-compatible server.
    Works with HF model IDs or local paths.
    """
    model = cfg.get("model") or cfg.get("model_dir", "")
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                model,
        "--host",                 "0.0.0.0",
        "--port",                 str(cfg["port"]),
        "--max-model-len",        str(cfg.get("ctx_size", 32768)),
        "--dtype",                cfg.get("dtype", "auto"),
        "--gpu-memory-utilization", str(cfg.get("gpu_memory_fraction", 0.90)),
    ]
    if cfg.get("tokenizer"):
        cmd += ["--tokenizer", cfg["tokenizer"]]
    if cfg.get("tensor_parallel_size", 1) > 1:
        cmd += ["--tensor-parallel-size", str(cfg["tensor_parallel_size"])]
    if cfg.get("quantization"):
        cmd += ["--quantization", cfg["quantization"]]
    if cfg.get("enforce_eager"):
        cmd.append("--enforce-eager")
    if cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if cfg.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")
    # Extra args passthrough
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)
    return cmd


def build_sglang_cmd(cfg: dict) -> list[str]:
    """
    SGLang OpenAI-compatible server.
    """
    model = cfg.get("model") or cfg.get("model_dir", "")
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path",           model,
        "--host",                 "0.0.0.0",
        "--port",                 str(cfg["port"]),
        "--context-length",       str(cfg.get("ctx_size", 32768)),
        "--mem-fraction-static",  str(cfg.get("gpu_memory_fraction", 0.88)),
    ]
    if cfg.get("tokenizer"):
        cmd += ["--tokenizer-path", cfg["tokenizer"]]
    if cfg.get("tensor_parallel_size", 1) > 1:
        cmd += ["--tp-size", str(cfg["tensor_parallel_size"])]
    if cfg.get("quantization"):
        cmd += ["--quantization", cfg["quantization"]]
    if cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if cfg.get("chunked_prefill"):
        cmd.append("--chunked-prefill-size")
        cmd.append(str(cfg.get("chunked_prefill_size", 8192)))
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)
    return cmd


def build_trtllm_cmd(cfg: dict, trt_config: dict) -> list[str]:
    cmd = [
        "python3", "-m", "tensorrt_llm.commands.serve",
        "--model_dir",          cfg["model_dir"],
        "--tokenizer_dir",      cfg.get("tokenizer", ""),
        "--host",               "0.0.0.0",
        "--port",               str(cfg["port"]),
        "--max_batch_size",     str(trt_config.get("max_batch_size", 1)),
        "--max_input_len",      str(trt_config.get("max_input_len", 32768)),
        "--max_output_len",     str(trt_config.get("max_output_len", 4096)),
        "--max_beam_width",     str(trt_config.get("max_beam_width", 1)),
        "--kv_cache_dtype",     trt_config.get("kv_cache_dtype", "int8"),
    ]
    if trt_config.get("chunked_context"):
        cmd.append("--chunked_context")
    if trt_config.get("enable_mtp"):
        cmd.append("--enable_mtp")
    if trt_config.get("gpu_memory_fraction"):
        cmd += ["--gpu_memory_fraction", str(trt_config["gpu_memory_fraction"])]
    return cmd


def build_hf_cmd(cfg: dict) -> list[str]:
    """
    HuggingFace TGI (text-generation-inference) or a minimal
    transformers-based OpenAI-compatible server.
    Tries TGI first; falls back to a simple wrapper.
    """
    model = cfg.get("model") or cfg.get("model_dir", "")

    # If TGI is installed, prefer it
    if shutil.which("text-generation-launcher"):
        cmd = [
            "text-generation-launcher",
            "--model-id",        model,
            "--hostname",        "0.0.0.0",
            "--port",            str(cfg["port"]),
            "--max-input-tokens",  str(cfg.get("ctx_size", 4096)),
            "--max-total-tokens",  str(cfg.get("ctx_size", 4096) + cfg.get("max_output_len", 2048)),
        ]
        if cfg.get("quantize"):
            cmd += ["--quantize", cfg["quantize"]]
        if cfg.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        return cmd

    # Fallback: use transformers + FastAPI wrapper (user must provide script)
    wrapper = cfg.get("wrapper_script", f"{HOME}/llm-router/hf_serve.py")
    return [
        "python3", wrapper,
        "--model",   model,
        "--host",    "0.0.0.0",
        "--port",    str(cfg["port"]),
    ]


# Dispatch table
CMD_BUILDERS = {
    ENGINE_LLAMA:  lambda cfg, _: build_llama_cmd(cfg),
    ENGINE_VLLM:   lambda cfg, _: build_vllm_cmd(cfg),
    ENGINE_SGLANG: lambda cfg, _: build_sglang_cmd(cfg),
    ENGINE_TRTLLM: lambda cfg, trt: build_trtllm_cmd(cfg, trt or {}),
    ENGINE_HF:     lambda cfg, _: build_hf_cmd(cfg),
}

# Health endpoints per engine (all use /health or /v1/models)
def health_url(cfg: dict) -> str:
    port = cfg["port"]
    engine = cfg.get("engine", ENGINE_LLAMA)
    # SGLang and vLLM expose /health; llama.cpp exposes /health;
    # TGI exposes /health; TRT-LLM exposes /health
    return f"http://localhost:{port}/health"


# ─────────────────────────────────────────────────────────────
# AUTO-DISCOVERY
# ─────────────────────────────────────────────────────────────

GGUF_SCAN_DIRS = [
    f"{HOME}/.lmstudio/models",
    f"{HOME}/models",
    f"{HOME}/llm-models",
    f"{HOME}/.cache/huggingface/hub",
]

HF_SCAN_DIRS = [
    f"{HOME}/models",
    f"{HOME}/llm-models",
    f"{HOME}/.cache/huggingface/hub",
]

TRTLLM_SCAN_DIRS = [
    f"{HOME}/trt-engines",
    f"{HOME}/tensorrt-engines",
    f"{HOME}/models/trt-llm",
]

TIER_THRESHOLDS = {"fast": 15, "mid": 40, "deep": 999}
DISCOVERY_PORT_START = 8100


def _file_size_gb(path: str) -> float:
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


def _classify_tier(size_gb: float) -> str:
    if size_gb < TIER_THRESHOLDS["fast"]:
        return "fast"
    elif size_gb < TIER_THRESHOLDS["mid"]:
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


def _estimate_idle(tier: str) -> int:
    return {"fast": 300, "mid": 180, "deep": 600}.get(tier, 300)


def _has_reasoning_kw(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in ["reason", "r1", "distill", "cot", "think", "122b", "70b"])


def _is_hf_checkpoint(path: str) -> bool:
    """Check if a directory looks like a HF model checkpoint."""
    p = Path(path)
    if not p.is_dir():
        return False
    # Must have config.json + some weights
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
    """Size of safetensors/bin files only."""
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


def discover_gguf_models(port_counter: list[int]) -> dict:
    discovered = {}
    seen_paths = set()

    for scan_dir in GGUF_SCAN_DIRS:
        if not os.path.isdir(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            gguf_files = [f for f in files if f.endswith('.gguf')]
            if not gguf_files:
                continue

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

                name = _model_name_from_path(full_path)
                slug = _slug(name, full_path)
                tier = _classify_tier(size_gb)
                reasoning = _has_reasoning_kw(name)

                discovered[slug] = {
                    "engine":       ENGINE_LLAMA,
                    "port":         port_counter[0],
                    "model":        full_path,
                    "log":          f"{HOME}/llama-{slug}.log",
                    "ctx_size":     _estimate_ctx(size_gb),
                    "gpu_layers":   999,
                    "flash_attn":   True,
                    "reasoning":    reasoning,
                    "idle_timeout": _estimate_idle(tier),
                    "startup_wait": _estimate_startup(size_gb),
                    "description":  f"{name} ({size_gb:.1f} GB) [gguf, auto]",
                    "auto_discovered": True,
                    "tier":         tier,
                    "size_gb":      round(size_gb, 2),
                }
                port_counter[0] += 1

    return discovered


def discover_hf_models(port_counter: list[int]) -> dict:
    """
    Discover HuggingFace checkpoints (safetensors/bin).
    These can be served by vLLM, SGLang, or HF TGI.
    Picks the best available engine automatically.
    """
    discovered = {}
    seen_paths = set()

    # Determine preferred engine for HF models
    if is_engine_available(ENGINE_VLLM):
        preferred_engine = ENGINE_VLLM
    elif is_engine_available(ENGINE_SGLANG):
        preferred_engine = ENGINE_SGLANG
    elif is_engine_available(ENGINE_HF):
        preferred_engine = ENGINE_HF
    else:
        return {}  # No engine available for HF models

    for scan_dir in HF_SCAN_DIRS:
        if not os.path.isdir(scan_dir):
            continue

        # Check immediate subdirs and one level deeper
        candidates = []
        for entry in os.scandir(scan_dir):
            if entry.is_dir():
                candidates.append(entry.path)
                # Also check subdirs (e.g. models/org/model-name)
                try:
                    for sub in os.scandir(entry.path):
                        if sub.is_dir():
                            candidates.append(sub.path)
                except PermissionError:
                    pass

        # Special handling for HF cache structure
        # ~/.cache/huggingface/hub/models--org--name/snapshots/abc123/
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

            hf_config = _read_hf_config(path)
            arch = hf_config.get("architectures", ["Unknown"])[0] if hf_config.get("architectures") else "Unknown"
            model_type = hf_config.get("model_type", "unknown")

            dir_name = Path(path).name
            # For HF cache, extract org/model from parent dirs
            if "snapshots" in path:
                parts = Path(path).parts
                for i, p in enumerate(parts):
                    if p.startswith("models--"):
                        dir_name = p.replace("models--", "").replace("--", "/")
                        break

            slug = _slug(f"hf-{dir_name}", path)
            tier = _classify_tier(size_gb)

            discovered[slug] = {
                "engine":         preferred_engine,
                "port":           port_counter[0],
                "model":          path,
                "log":            f"{HOME}/{preferred_engine}-{slug}.log",
                "ctx_size":       min(_estimate_ctx(size_gb), hf_config.get("max_position_embeddings", 131072)),
                "dtype":          "auto",
                "gpu_memory_fraction": 0.90,
                "trust_remote_code": True,
                "idle_timeout":   _estimate_idle(tier),
                "startup_wait":   _estimate_startup(size_gb) + 30,  # HF models take longer
                "description":    f"{dir_name} ({size_gb:.1f} GB, {arch}) [{preferred_engine}, auto]",
                "auto_discovered": True,
                "tier":           tier,
                "size_gb":        round(size_gb, 2),
                "model_type":     model_type,
            }
            port_counter[0] += 1

    return discovered


def discover_trtllm_engines(port_counter: list[int]) -> dict:
    discovered = {}

    for scan_dir in TRTLLM_SCAN_DIRS:
        if not os.path.isdir(scan_dir):
            continue
        for entry in os.scandir(scan_dir):
            if not entry.is_dir():
                continue
            config_path = os.path.join(entry.path, "config.json")
            engine_files = glob.glob(os.path.join(entry.path, "*.engine"))
            if not os.path.exists(config_path) or not engine_files:
                continue

            slug = _slug(f"trt-{entry.name}", entry.path)
            description = entry.name
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                arch = cfg.get("pretrained_config", {}).get("architecture", entry.name)
                description = f"{arch} [trt-llm, auto]"
            except Exception:
                pass

            # Find tokenizer
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
                "engine":       ENGINE_TRTLLM,
                "port":         port_counter[0],
                "model_dir":    entry.path,
                "tokenizer":    tokenizer_dir,
                "log":          f"{HOME}/trtllm-{slug}.log",
                "idle_timeout": 600,
                "startup_wait": 120,
                "description":  description,
                "auto_discovered": True,
                "trt_config": {
                    "max_batch_size": 1, "max_input_len": 32768,
                    "max_output_len": 4096, "kv_cache_dtype": "int8",
                    "chunked_context": True, "enable_mtp": False,
                    "max_beam_width": 1, "gpu_memory_fraction": 0.90,
                },
            }
            port_counter[0] += 1

    return discovered


def load_user_overrides() -> dict:
    if not os.path.exists(USER_OVERRIDES):
        return {}
    try:
        with open(USER_OVERRIDES) as f:
            return json.load(f)
    except Exception:
        return {}


def save_discovery_cache(discovered: dict):
    with open(DISCOVERY_CACHE, "w") as f:
        json.dump(discovered, f, indent=2)


# ─────────────────────────────────────────────────────────────
# MANUAL BACKEND REGISTRY
# ─────────────────────────────────────────────────────────────
MANUAL_BACKENDS = {
    "fast": {
        "engine":      ENGINE_LLAMA,
        "port":        8080,
        "model":       f"{HOME}/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        "log":         f"{HOME}/llama-fast.log",
        "ctx_size":    131072,
        "gpu_layers":  999,
        "flash_attn":  True,
        "reasoning":   False,
        "idle_timeout": 300,
        "startup_wait": 30,
        "description": "Qwen3.5-35B-A3B Q4_K_XL — fast, no reasoning",
    },
    "mid": {
        "engine":      ENGINE_LLAMA,
        "port":        8082,
        "model":       f"{HOME}/.lmstudio/models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf",
        "log":         f"{HOME}/llama-qwen3.5-27B.log",
        "ctx_size":    131072,
        "gpu_layers":  999,
        "flash_attn":  True,
        "reasoning":   False,
        "idle_timeout": 180,
        "startup_wait": 40,
        "description": "Qwen3.5-27B Q8_0 — balanced quality",
    },
    "deep": {
        "engine":      ENGINE_LLAMA,
        "port":        8081,
        "model":       f"{HOME}/.lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
        "log":         f"{HOME}/llama-reason.log",
        "ctx_size":    131072,
        "gpu_layers":  999,
        "flash_attn":  True,
        "reasoning":   True,
        "reasoning_budget": 2048,
        "idle_timeout": 600,
        "startup_wait": 90,
        "description": "Qwen3.5-122B-A10B Q4_K_XL — deep reasoning",
    },

    # ── vLLM example — uncomment to enable ───────────────────
    # "qwen-vllm": {
    #     "engine":      ENGINE_VLLM,
    #     "port":        8091,
    #     "model":       "Qwen/Qwen2.5-72B-Instruct-AWQ",
    #     "log":         f"{HOME}/vllm-qwen72b.log",
    #     "ctx_size":    32768,
    #     "dtype":       "auto",
    #     "quantization": "awq",
    #     "gpu_memory_fraction": 0.90,
    #     "idle_timeout": 600,
    #     "startup_wait": 120,
    #     "description": "Qwen2.5-72B AWQ via vLLM",
    # },

    # ── SGLang example — uncomment to enable ─────────────────
    # "llama-sglang": {
    #     "engine":      ENGINE_SGLANG,
    #     "port":        8092,
    #     "model":       "meta-llama/Llama-3.1-70B-Instruct",
    #     "log":         f"{HOME}/sglang-llama70b.log",
    #     "ctx_size":    32768,
    #     "gpu_memory_fraction": 0.88,
    #     "idle_timeout": 600,
    #     "startup_wait": 120,
    #     "description": "Llama-3.1-70B via SGLang",
    # },

    # ── HF TGI example — uncomment to enable ────────────────
    # "mistral-hf": {
    #     "engine":      ENGINE_HF,
    #     "port":        8093,
    #     "model":       f"{HOME}/models/Mistral-7B-Instruct-v0.3",
    #     "log":         f"{HOME}/hf-mistral.log",
    #     "ctx_size":    32768,
    #     "idle_timeout": 300,
    #     "startup_wait": 60,
    #     "description": "Mistral-7B via HF TGI",
    # },
}


def build_backend_registry() -> dict:
    registry = dict(MANUAL_BACKENDS)

    # Port counter shared across discovery functions
    port_counter = [DISCOVERY_PORT_START]

    gguf = discover_gguf_models(port_counter)
    hf = discover_hf_models(port_counter)

    port_counter[0] = DISCOVERY_PORT_START + 200
    trt = discover_trtllm_engines(port_counter)

    # Merge discovered — skip slug collisions and path duplicates
    manual_paths = {v.get("model", "") for v in registry.values()}
    manual_paths |= {v.get("model_dir", "") for v in registry.values()}

    for slug, cfg in {**gguf, **hf, **trt}.items():
        if slug in registry:
            continue
        model_path = cfg.get("model", cfg.get("model_dir", ""))
        if model_path in manual_paths:
            continue
        registry[slug] = cfg

    # Apply user overrides
    user = load_user_overrides()
    for slug in user.get("exclude", []):
        registry.pop(slug, None)
    for slug, patches in user.get("overrides", {}).items():
        if slug in registry:
            registry[slug].update(patches)

    # Save discovery cache
    discovered_only = {k: v for k, v in registry.items() if v.get("auto_discovered")}
    save_discovery_cache(discovered_only)

    return registry


BACKENDS = build_backend_registry()


# ─────────────────────────────────────────────────────────────
# ROUTING RULES
# ─────────────────────────────────────────────────────────────
DEEP_KEYWORDS = [
    "reason", "think", "analyze", "analyse", "explain why", "step by step",
    "prove", "derive", "architecture", "design", "complex", "debug",
    "refactor", "optimize", "compare", "evaluate", "strategy",
]
MID_KEYWORDS = [
    "write", "implement", "code", "function", "script", "fix", "review",
    "summarize", "summarise", "translate", "draft", "generate",
]

def classify(payload: dict) -> str:
    messages = payload.get("messages", [])
    content = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str)
        else " ".join(p.get("text", "") for p in m.get("content", []) if isinstance(p, dict))
        for m in messages
    ).lower()
    token_estimate = len(content.split())

    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str) and c.startswith("[route:"):
            key = c.split("]")[0].replace("[route:", "").strip()
            if key in BACKENDS:
                return key

    if token_estimate > 4000:
        return "deep"
    if any(kw in content for kw in DEEP_KEYWORDS):
        return "deep"
    if any(kw in content for kw in MID_KEYWORDS):
        return "mid"
    return "fast"


# ─────────────────────────────────────────────────────────────
# TRT-LLM AUTO-TUNER
# ─────────────────────────────────────────────────────────────
class TRTLLMTuner:
    SEARCH_SPACE = [
        {"max_input_len": 131072, "kv_cache_dtype": "int8", "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.92},
        {"max_input_len": 65536,  "kv_cache_dtype": "int8", "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.92},
        {"max_input_len": 32768,  "kv_cache_dtype": "int8", "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.90},
        {"max_input_len": 16384,  "kv_cache_dtype": "int8", "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.85},
        {"max_input_len": 16384,  "kv_cache_dtype": "fp8",  "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.85},
        {"max_input_len": 8192,   "kv_cache_dtype": "int8", "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.80},
    ]
    OOM_SIGNALS = [
        "out of memory", "oom", "cuda error", "cuda out of memory",
        "cannot allocate", "memory allocation failed", "insufficient memory",
        "std::bad_alloc", "cublas_status_alloc_failed",
    ]

    def __init__(self, key: str):
        self.key = key
        self.tune_file = os.path.join(TUNE_DIR, f"{key}.json")

    def load_saved(self) -> dict | None:
        if os.path.exists(self.tune_file):
            try:
                with open(self.tune_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def save(self, config: dict):
        with open(self.tune_file, "w") as f:
            json.dump(config, f, indent=2)

    def clear(self):
        if os.path.exists(self.tune_file):
            os.remove(self.tune_file)

    def is_oom(self, log_path: str) -> bool:
        try:
            with open(log_path) as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 5000))
                tail = f.read().lower()
            return any(sig in tail for sig in self.OOM_SIGNALS)
        except Exception:
            return False

    def get_search_space(self, base_config: dict) -> list[dict]:
        configs = []
        if base_config:
            configs.append(base_config)
        for t in self.SEARCH_SPACE:
            merged = {**base_config, **t} if base_config else t
            if merged not in configs:
                configs.append(merged)
        return configs


# ─────────────────────────────────────────────────────────────
# BACKEND LIFECYCLE MANAGER
# ─────────────────────────────────────────────────────────────
class BackendManager:
    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_handles: dict[str, object] = {}
        self.last_used: dict[str, float] = {}
        self.starting: dict[str, asyncio.Lock] = {k: asyncio.Lock() for k in BACKENDS}
        self.active_configs: dict[str, dict] = {}

    def refresh_locks(self):
        for k in BACKENDS:
            if k not in self.starting:
                self.starting[k] = asyncio.Lock()

    def is_running(self, key: str) -> bool:
        proc = self.processes.get(key)
        return proc is not None and proc.poll() is None

    async def _wait_healthy(self, key: str) -> bool:
        cfg = BACKENDS[key]
        url = health_url(cfg)
        deadline = time.time() + cfg["startup_wait"]
        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                try:
                    r = await client.get(url, timeout=2)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                if not self.is_running(key):
                    return False
                await asyncio.sleep(1)
        return False

    async def _start_process(self, key: str, trt_config: dict | None = None) -> bool:
        cfg = BACKENDS[key]
        engine = cfg.get("engine", ENGINE_LLAMA)

        # Check engine availability
        if not is_engine_available(engine):
            logger.error(f"[{key}] Engine '{engine}' is not installed")
            return False

        log_file = open(cfg["log"], "a")
        self.log_handles[key] = log_file

        builder = CMD_BUILDERS.get(engine)
        if not builder:
            logger.error(f"[{key}] No command builder for engine '{engine}'")
            return False

        cmd = builder(cfg, trt_config)
        logger.info(f"[{key}] Starting ({engine}) on port {cfg['port']}...")

        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        self.processes[key] = proc
        self.last_used[key] = time.time()

        healthy = await self._wait_healthy(key)
        if not healthy:
            logger.warning(f"[{key}] Failed health check — killing PID {proc.pid}")
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            del self.processes[key]
            return False

        logger.info(f"[{key}] Ready on port {cfg['port']} (PID {proc.pid})")
        return True

    async def ensure_running(self, key: str):
        if self.is_running(key):
            self.last_used[key] = time.time()
            return

        async with self.starting[key]:
            if self.is_running(key):
                self.last_used[key] = time.time()
                return

            cfg = BACKENDS[key]
            engine = cfg.get("engine", ENGINE_LLAMA)

            if engine == ENGINE_TRTLLM:
                await self._start_trtllm_with_tuning(key)
            else:
                ok = await self._start_process(key)
                if not ok:
                    raise RuntimeError(f"Backend '{key}' ({engine}) failed to start. Check {cfg['log']}")

    async def _start_trtllm_with_tuning(self, key: str):
        cfg = BACKENDS[key]
        tuner = TRTLLMTuner(key)

        saved = tuner.load_saved()
        if saved:
            ok = await self._start_process(key, trt_config=saved)
            if ok:
                self.active_configs[key] = saved
                return
            tuner.clear()

        base_config = cfg.get("trt_config", {})
        search_space = tuner.get_search_space(base_config)
        logger.info(f"[{key}] Auto-tuning ({len(search_space)} configs)")

        for i, trial in enumerate(search_space, 1):
            logger.info(f"[{key}] Attempt {i}/{len(search_space)}: input_len={trial.get('max_input_len')}, kv={trial.get('kv_cache_dtype')}")
            ok = await self._start_process(key, trt_config=trial)
            if ok:
                tuner.save(trial)
                self.active_configs[key] = trial
                return
            await asyncio.sleep(3)

        raise RuntimeError(f"Backend '{key}' failed all tuning attempts. Check {cfg['log']}")

    def stop(self, key: str):
        proc = self.processes.get(key)
        if proc and proc.poll() is None:
            logger.info(f"[{key}] Stopping backend (PID {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.processes.pop(key, None)
        self.last_used.pop(key, None)
        self.active_configs.pop(key, None)
        if key in self.log_handles:
            try:
                self.log_handles[key].close()
            except Exception:
                pass
            self.log_handles.pop(key, None)

    def stop_all(self):
        for key in list(self.processes.keys()):
            self.stop(key)

    async def idle_watchdog(self):
        while True:
            await asyncio.sleep(30)
            now = time.time()
            for key, cfg in BACKENDS.items():
                if not self.is_running(key):
                    continue
                idle_for = now - self.last_used.get(key, now)
                if idle_for > cfg["idle_timeout"]:
                    logger.info(f"[{key}] Idle for {idle_for:.0f}s — auto-stopping")
                    self.stop(key)

    def status(self) -> dict:
        now = time.time()
        out = {}
        for key, cfg in BACKENDS.items():
            running = self.is_running(key)
            idle = round(now - self.last_used[key]) if key in self.last_used else None
            entry = {
                "engine":          cfg.get("engine", ENGINE_LLAMA),
                "running":         running,
                "port":            cfg["port"],
                "idle_seconds":    idle,
                "idle_timeout":    cfg["idle_timeout"],
                "description":     cfg["description"],
                "auto_discovered": cfg.get("auto_discovered", False),
                "pid":             self.processes[key].pid if running else None,
            }
            if key in self.active_configs:
                entry["trt_active_config"] = self.active_configs[key]
            if os.path.exists(os.path.join(TUNE_DIR, f"{key}.json")):
                entry["tuning_saved"] = True
            out[key] = entry
        return out


# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm-router")

manager = BackendManager()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engines = available_engines()
    manual = len(MANUAL_BACKENDS)
    discovered = sum(1 for v in BACKENDS.values() if v.get("auto_discovered"))
    logger.info(f"LLM Router :9001 — engines: {engines}")
    logger.info(f"  {manual} manual + {discovered} discovered = {len(BACKENDS)} backends")
    task = asyncio.create_task(manager.idle_watchdog())
    yield
    task.cancel()
    manager.stop_all()

app = FastAPI(title="LLM Router", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return manager.status()

@app.get("/backends")
async def list_backends():
    return {
        k: {
            "engine":          v.get("engine", ENGINE_LLAMA),
            "description":     v["description"],
            "port":            v["port"],
            "auto_discovered": v.get("auto_discovered", False),
            "tier":            v.get("tier"),
            "size_gb":         v.get("size_gb"),
        }
        for k, v in BACKENDS.items()
    }

@app.get("/engines")
async def list_engines():
    """Show which engines are installed."""
    return {e: is_engine_available(e) for e in ALL_ENGINES}

@app.post("/rescan")
async def rescan():
    global BACKENDS
    _engine_available_cache.clear()  # re-check engines too
    running = {k for k in BACKENDS if manager.is_running(k)}
    BACKENDS = build_backend_registry()
    manager.refresh_locks()
    discovered = sum(1 for v in BACKENDS.values() if v.get("auto_discovered"))
    return {
        "total": len(BACKENDS),
        "discovered": discovered,
        "engines": available_engines(),
        "running": list(running),
        "backends": list(BACKENDS.keys()),
    }

@app.post("/retune/{key}")
async def retune(key: str):
    if key not in BACKENDS:
        raise HTTPException(404, f"Unknown backend '{key}'")
    if BACKENDS[key].get("engine") != ENGINE_TRTLLM:
        raise HTTPException(400, f"'{key}' is not a TRT-LLM backend")
    manager.stop(key)
    TRTLLMTuner(key).clear()
    try:
        await manager.ensure_running(key)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"status": "retuned", "active_config": manager.active_configs.get(key)}

@app.post("/stop/{key}")
async def stop_backend(key: str):
    if key not in BACKENDS:
        raise HTTPException(404, f"Unknown backend '{key}'")
    manager.stop(key)
    return {"status": "stopped", "backend": key}

@app.post("/start/{key}")
async def start_backend(key: str):
    if key not in BACKENDS:
        raise HTTPException(404, f"Unknown backend '{key}'")
    try:
        await manager.ensure_running(key)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"status": "started", "backend": key, "port": BACKENDS[key]["port"]}

@app.post("/v1/{path:path}")
async def proxy(path: str, request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    backend_key = request.query_params.get("backend") or classify(payload)

    if backend_key not in BACKENDS:
        raise HTTPException(400, f"Unknown backend '{backend_key}'. Valid: {list(BACKENDS.keys())}")

    try:
        await manager.ensure_running(backend_key)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    manager.last_used[backend_key] = time.time()
    cfg = BACKENDS[backend_key]
    target_url = f"http://localhost:{cfg['port']}/v1/{path}"

    is_stream = payload.get("stream", False)

    if is_stream:
        async def stream_response() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", target_url, json=payload) as resp:
                    logger.info(f"[{backend_key}] streaming /{path}")
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(target_url, json=payload)
        logger.info(f"[{backend_key}] /{path} → {resp.status_code}")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
