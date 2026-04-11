"""
router/engines.py — Engine availability detection and command builders

Supports: llama.cpp, vLLM, SGLang, TensorRT-LLM, HuggingFace TGI, Ollama,
          and any already-running OpenAI-compatible server (LM Studio, etc.)
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig

# Engine identifiers — use these constants everywhere
ENGINE_LLAMA  = "llama.cpp"
ENGINE_TRTLLM = "trt-llm"
ENGINE_TRTLLM_DOCKER = "trt-llm-docker"
ENGINE_VLLM   = "vllm"
ENGINE_SGLANG = "sglang"
ENGINE_HF     = "huggingface"
ENGINE_OLLAMA = "ollama"
ENGINE_OPENAI = "openai"   # passthrough to any running OpenAI-compatible server

ALL_ENGINES = [
    ENGINE_LLAMA,
    ENGINE_TRTLLM,
    ENGINE_TRTLLM_DOCKER,
    ENGINE_VLLM,
    ENGINE_SGLANG,
    ENGINE_HF,
    ENGINE_OLLAMA,
    ENGINE_OPENAI,
]

# ─────────────────────────────────────────────────────────────
# Availability detection
# ─────────────────────────────────────────────────────────────
_engine_available_cache: dict[str, bool] = {}


def _can_import(module: str) -> bool:
    try:
        result = subprocess.run(
            [_python_bin(), "-c", f"import {module}"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _docker_available() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        result = subprocess.run(
            [docker, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
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
        ENGINE_TRTLLM_DOCKER: lambda: getattr(config.trtllm_docker, "enabled", True) and _docker_available(),
        ENGINE_HF:     lambda: _can_import("transformers"),
        ENGINE_OLLAMA: lambda: shutil.which("ollama") is not None or _is_ollama_running(),
        ENGINE_OPENAI: lambda: True,   # no local binary required; server already running
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


def _python_bin() -> str:
    """
    Use the same interpreter that launched the router so managed Python
    backends inherit the project venv instead of falling back to /usr/bin/python3.
    """
    return sys.executable or "python3"


def _safe_container_name(key: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", key.lower()).strip("-")
    return safe or "trtllm"


def resolve_trtllm_docker_config(key: str, cfg: dict, config: "AppConfig") -> dict:
    """Merge app defaults with per-backend docker overrides."""
    app_defaults = getattr(config, "trtllm_docker")
    backend_cfg = dict(cfg.get("docker_config") or {})
    merged_env = dict(getattr(app_defaults, "env", {}) or {})
    merged_env.update(dict(backend_cfg.get("env") or {}))
    merged_serve_args = dict(getattr(app_defaults, "serve_defaults", {}) or {})
    merged_serve_args.update(dict(backend_cfg.get("serve_args") or {}))

    return {
        "image": str(backend_cfg.get("image") or app_defaults.image),
        "container_name": str(
            backend_cfg.get("container_name")
            or f"llm-router-{_safe_container_name(key)}"
        ),
        "container_port": int(backend_cfg.get("container_port") or app_defaults.container_port),
        "hf_cache_dir": os.path.expanduser(str(backend_cfg.get("hf_cache_dir") or app_defaults.hf_cache_dir)),
        "log_dir": os.path.expanduser(str(backend_cfg.get("log_dir") or app_defaults.log_dir)),
        "env": merged_env,
        "serve_args": merged_serve_args,
        "launcher_script": os.path.expanduser(str(backend_cfg.get("launcher_script") or "")).strip(),
    }


def _append_cli_arg(parts: list[str], flag: str, value) -> None:
    if value is None or value is False:
        return
    if value is True:
        parts.append(flag)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _append_cli_arg(parts, flag, item)
        return
    parts.extend([flag, str(value)])


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
        _python_bin(), "-m", "vllm.entrypoints.openai.api_server",
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
        _python_bin(), "-m", "sglang.launch_server",
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
    model_dir = cfg.get("model_dir") or cfg.get("model", "")
    cmd = [
        _python_bin(), "-m", "tensorrt_llm.commands.serve",
        "--model_dir",      model_dir,
        "--host",           "0.0.0.0",
        "--port",           str(cfg["port"]),
        "--max_batch_size", str(trt_config.get("max_batch_size", 1)),
        "--max_input_len",  str(trt_config.get("max_input_len", 32768)),
        "--max_output_len", str(trt_config.get("max_output_len", 4096)),
        "--max_beam_width", str(trt_config.get("max_beam_width", 1)),
        "--kv_cache_dtype", trt_config.get("kv_cache_dtype", "int8"),
    ]
    if cfg.get("tokenizer"):
        cmd += ["--tokenizer_dir", cfg["tokenizer"]]
    if trt_config.get("chunked_context"):
        cmd.append("--chunked_context")
    if trt_config.get("enable_mtp"):
        cmd.append("--enable_mtp")
    if trt_config.get("gpu_memory_fraction"):
        cmd += ["--gpu_memory_fraction", str(trt_config["gpu_memory_fraction"])]
    if cfg.get("trust_remote_code"):
        cmd.append("--trust_remote_code")
    return cmd


def build_trtllm_docker_cmd(key: str, cfg: dict, config: "AppConfig") -> list[str]:
    docker_cfg = resolve_trtllm_docker_config(key, cfg, config)
    launcher_script = docker_cfg.get("launcher_script")
    if launcher_script:
        return [launcher_script]

    model = cfg.get("model") or cfg.get("model_dir", "")
    serve_parts = [
        "trtllm-serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(docker_cfg["container_port"]),
    ]
    for arg_name, arg_value in docker_cfg["serve_args"].items():
        _append_cli_arg(serve_parts, f"--{arg_name}", arg_value)

    log_file = f"/logs/{docker_cfg['container_name']}.log"
    inner_cmd = ""
    if docker_cfg["env"]:
        exports = " ".join(
            f"{name}={shlex.quote(str(value))}"
            for name, value in docker_cfg["env"].items()
        )
        inner_cmd += f"export {exports}\n"
    inner_cmd += f"{' '.join(shlex.quote(part) for part in serve_parts)} 2>&1 | tee {shlex.quote(log_file)}"

    cmd = [
        "docker", "run", "-d",
        "--name",    docker_cfg["container_name"],
        "--restart", "unless-stopped",
        "--ipc",     "host",
        "--gpus",    "all",
        "--ulimit",  "memlock=-1",
        "--ulimit",  "stack=67108864",
        "-p", f"{cfg['port']}:{docker_cfg['container_port']}",
        "-v", f"{docker_cfg['hf_cache_dir']}:/root/.cache/huggingface",
        "-v", f"{docker_cfg['log_dir']}:/logs",
    ]
    cmd += [
        docker_cfg["image"],
        "bash",
        "-lc",
        inner_cmd,
    ]
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
        _python_bin(), wrapper,
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
    if engine in (ENGINE_OPENAI, ENGINE_TRTLLM_DOCKER):
        return f"http://localhost:{cfg['port']}/v1/models"
    return f"http://localhost:{cfg['port']}/health"


def get_cmd_builder(engine: str):
    """Return the command builder function for an engine."""
    builders = {
        ENGINE_LLAMA:  build_llama_cmd,
        ENGINE_VLLM:   build_vllm_cmd,
        ENGINE_SGLANG: build_sglang_cmd,
        ENGINE_TRTLLM: None,  # handled specially — needs trt_config
        ENGINE_TRTLLM_DOCKER: None,  # handled specially — needs backend key + merged docker config
        ENGINE_HF:     build_hf_cmd,
        ENGINE_OLLAMA: build_ollama_cmd,
    }
    return builders.get(engine)
