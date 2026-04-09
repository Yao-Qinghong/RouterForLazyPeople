"""
router/engines.py — Engine availability detection and command builders

Supports: llama.cpp, vLLM, SGLang, TensorRT-LLM, HuggingFace TGI, Ollama
"""

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig

# Engine identifiers — use these constants everywhere
ENGINE_LLAMA  = "llama.cpp"
ENGINE_TRTLLM = "trt-llm"
ENGINE_VLLM   = "vllm"
ENGINE_SGLANG = "sglang"
ENGINE_HF     = "huggingface"
ENGINE_OLLAMA = "ollama"

ALL_ENGINES = [ENGINE_LLAMA, ENGINE_TRTLLM, ENGINE_VLLM, ENGINE_SGLANG, ENGINE_HF, ENGINE_OLLAMA]

# ─────────────────────────────────────────────────────────────
# Availability detection
# ─────────────────────────────────────────────────────────────
_engine_available_cache: dict[str, bool] = {}


def _can_import(module: str) -> bool:
    try:
        subprocess.run(
            ["python3", "-c", f"import {module}"],
            capture_output=True, timeout=10,
        )
        return True
    except Exception:
        return False


def is_engine_available(engine: str, config: "AppConfig") -> bool:
    """Check whether an engine's binary or Python module is installed."""
    if engine in _engine_available_cache:
        return _engine_available_cache[engine]

    checks = {
        ENGINE_LLAMA:  lambda: os.path.isfile(str(config.llama_bin)),
        ENGINE_VLLM:   lambda: shutil.which("vllm") is not None or _can_import("vllm"),
        ENGINE_SGLANG: lambda: shutil.which("sglang") is not None or _can_import("sglang"),
        ENGINE_TRTLLM: lambda: _can_import("tensorrt_llm"),
        ENGINE_HF:     lambda: _can_import("transformers"),
        ENGINE_OLLAMA: lambda: shutil.which("ollama") is not None or _is_ollama_running(),
    }

    result = checks.get(engine, lambda: False)()
    _engine_available_cache[engine] = result
    return result


def _is_ollama_running() -> bool:
    """Check if Ollama server is already running on its default port."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def available_engines(config: "AppConfig") -> list[str]:
    """Return list of all currently installed engines."""
    return [e for e in ALL_ENGINES if is_engine_available(e, config)]


def clear_engine_cache():
    """Clear the availability cache (called by /rescan)."""
    _engine_available_cache.clear()


# ─────────────────────────────────────────────────────────────
# Command builders
# ─────────────────────────────────────────────────────────────

def build_llama_cmd(cfg: dict, config: "AppConfig") -> list[str]:
    cmd = [
        str(config.llama_bin),
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


def build_vllm_cmd(cfg: dict, config: "AppConfig" = None) -> list[str]:
    model = cfg.get("model") or cfg.get("model_dir", "")
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model,
        "--host",                   "0.0.0.0",
        "--port",                   str(cfg["port"]),
        "--max-model-len",          str(cfg.get("ctx_size", 32768)),
        "--dtype",                  cfg.get("dtype", "auto"),
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
    if cfg.get("enable_prefix_caching", True):
        cmd.append("--enable-prefix-caching")
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)
    return cmd


def build_sglang_cmd(cfg: dict, config: "AppConfig" = None) -> list[str]:
    model = cfg.get("model") or cfg.get("model_dir", "")
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path",          model,
        "--host",                "0.0.0.0",
        "--port",                str(cfg["port"]),
        "--context-length",      str(cfg.get("ctx_size", 32768)),
        "--mem-fraction-static", str(cfg.get("gpu_memory_fraction", 0.88)),
    ]
    if cfg.get("tokenizer"):
        cmd += ["--tokenizer-path", cfg["tokenizer"]]
    if cfg.get("tensor_parallel_size", 1) > 1:
        cmd += ["--tp-size", str(cfg["tensor_parallel_size"])]
    if cfg.get("quantization"):
        cmd += ["--quantization", cfg["quantization"]]
    if cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if cfg.get("enable_prefix_caching", True):
        cmd.append("--enable-prefix-caching")
    if cfg.get("chunked_prefill"):
        cmd += ["--chunked-prefill-size", str(cfg.get("chunked_prefill_size", 8192))]
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)
    return cmd


def build_trtllm_cmd(cfg: dict, trt_config: dict, config: "AppConfig" = None) -> list[str]:
    cmd = [
        "python3", "-m", "tensorrt_llm.commands.serve",
        "--model_dir",      cfg["model_dir"],
        "--tokenizer_dir",  cfg.get("tokenizer", ""),
        "--host",           "0.0.0.0",
        "--port",           str(cfg["port"]),
        "--max_batch_size", str(trt_config.get("max_batch_size", 1)),
        "--max_input_len",  str(trt_config.get("max_input_len", 32768)),
        "--max_output_len", str(trt_config.get("max_output_len", 4096)),
        "--max_beam_width", str(trt_config.get("max_beam_width", 1)),
        "--kv_cache_dtype", trt_config.get("kv_cache_dtype", "int8"),
    ]
    if trt_config.get("chunked_context"):
        cmd.append("--chunked_context")
    if trt_config.get("enable_mtp"):
        cmd.append("--enable_mtp")
    if trt_config.get("gpu_memory_fraction"):
        cmd += ["--gpu_memory_fraction", str(trt_config["gpu_memory_fraction"])]
    return cmd


def build_hf_cmd(cfg: dict, config: "AppConfig" = None) -> list[str]:
    model = cfg.get("model") or cfg.get("model_dir", "")

    if shutil.which("text-generation-launcher"):
        cmd = [
            "text-generation-launcher",
            "--model-id",        model,
            "--hostname",        "0.0.0.0",
            "--port",            str(cfg["port"]),
            "--max-input-tokens", str(cfg.get("ctx_size", 4096)),
            "--max-total-tokens", str(cfg.get("ctx_size", 4096) + cfg.get("max_output_len", 2048)),
        ]
        if cfg.get("quantize"):
            cmd += ["--quantize", cfg["quantize"]]
        if cfg.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        return cmd

    # Fallback: transformers wrapper script
    home = os.path.expanduser("~")
    wrapper = cfg.get("wrapper_script", f"{home}/llm-router/hf_serve.py")
    return [
        "python3", wrapper,
        "--model", model,
        "--host",  "0.0.0.0",
        "--port",  str(cfg["port"]),
    ]


def build_ollama_cmd(cfg: dict, config: "AppConfig" = None) -> list[str]:
    """
    Build command to start Ollama serving a specific model.
    Ollama manages its own model lifecycle, so we use 'ollama serve'
    and pull the model if needed.
    """
    # Ollama is typically already running as a service.
    # We use 'ollama run' in the background which pulls + loads the model.
    model = cfg.get("model", "")
    return [
        "ollama", "run", model,
        "--keepalive", str(cfg.get("idle_timeout", 300)),
    ]


def health_url(cfg: dict) -> str:
    """Return the health-check URL for a backend config."""
    engine = cfg.get("engine", ENGINE_LLAMA)
    if engine == ENGINE_OLLAMA:
        return f"http://localhost:{cfg['port']}/api/tags"
    return f"http://localhost:{cfg['port']}/health"


def get_cmd_builder(engine: str):
    """Return the command builder function for an engine."""
    builders = {
        ENGINE_LLAMA:  build_llama_cmd,
        ENGINE_VLLM:   build_vllm_cmd,
        ENGINE_SGLANG: build_sglang_cmd,
        ENGINE_TRTLLM: None,  # handled specially — needs trt_config
        ENGINE_HF:     build_hf_cmd,
        ENGINE_OLLAMA: build_ollama_cmd,
    }
    return builders.get(engine)
