from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from router import engines as engines_module
from router.config import BackendConfig
from router.engines import (
    build_sglang_cmd,
    build_trtllm_cmd,
    build_trtllm_docker_cmd,
    build_vllm_cmd,
)
from router.proxy import build_model_aliases, resolve_requested_model


class _ManagerStub:
    def __init__(self, running: set[str] | None = None):
        self._running = running or set()

    def is_running(self, key: str) -> bool:
        return key in self._running


def _docker_app_config():
    return SimpleNamespace(
        trtllm_docker=SimpleNamespace(
            image="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7",
            container_port=8000,
            hf_cache_dir="/home/test/.cache/huggingface",
            log_dir="/home/test/.llm-router/trtllm-logs",
            env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            serve_defaults={
                "max_seq_len": 65536,
                "max_num_tokens": 16384,
                "max_batch_size": 4,
                "kv_cache_free_gpu_memory_fraction": 0.75,
            },
        )
    )


def test_python_backends_use_router_interpreter():
    vllm = build_vllm_cmd({"model": "org/model", "port": 8000})
    sglang = build_sglang_cmd({"model": "org/model", "port": 8001})
    trt = build_trtllm_cmd({"model_dir": "/tmp/model", "port": 8002}, {})

    assert vllm[0] == sys.executable
    assert sglang[0] == sys.executable
    assert trt[0] == sys.executable


def test_can_import_requires_zero_exit_status(monkeypatch):
    monkeypatch.setattr(
        engines_module.subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(returncode=1),
    )

    assert engines_module._can_import("missing_mod") is False


def test_model_aliases_prefer_external_backend_for_same_model():
    model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    backends = {
        "local-8000": BackendConfig(engine="openai", port=8000, model=model_id, auto_discovered=True),
        "hf-nvidia": BackendConfig(engine="vllm", port=8111, model=model_id, auto_discovered=True),
    }

    aliases = build_model_aliases(backends, {}, _ManagerStub())

    assert aliases[model_id] == "local-8000"


def test_resolve_requested_model_prefers_exact_backend_key_over_alias():
    backends = {
        "fast": BackendConfig(engine="llama.cpp", port=8080, model="/models/fast.gguf"),
    }

    resolved = resolve_requested_model("fast", backends, {"fast": "missing"})

    assert resolved == "fast"


def test_local_paths_do_not_become_auto_aliases():
    backends = {
        "fast": BackendConfig(engine="llama.cpp", port=8080, model="/models/fast.gguf"),
    }

    aliases = build_model_aliases(backends, {})

    assert aliases == {}


def test_trtllm_cmd_uses_model_dir_and_optional_flags():
    cmd = build_trtllm_cmd(
        {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "model_dir": "/models/nvfp4",
            "tokenizer": "/models/nvfp4",
            "port": 8000,
            "trust_remote_code": True,
        },
        {"gpu_memory_fraction": 0.95},
    )

    assert cmd[:3] == [sys.executable, "-m", "tensorrt_llm.commands.serve"]
    assert "--model_dir" in cmd and cmd[cmd.index("--model_dir") + 1] == "/models/nvfp4"
    assert "--tokenizer_dir" in cmd and cmd[cmd.index("--tokenizer_dir") + 1] == "/models/nvfp4"
    assert "--gpu_memory_fraction" in cmd
    assert "--trust_remote_code" in cmd


def test_trtllm_docker_cmd_uses_native_managed_defaults():
    cmd = build_trtllm_docker_cmd(
        "hf-nvidia",
        {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "port": 8111,
            "docker_config": {},
        },
        _docker_app_config(),
    )

    assert cmd[:3] == ["docker", "run", "-d"]
    assert "--name" in cmd and cmd[cmd.index("--name") + 1] == "llm-router-hf-nvidia"
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "8111:8000"
    assert "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7" in cmd
    inner = cmd[-1]
    assert "trtllm-serve" in inner
    assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" in inner
    assert "--max_seq_len" in inner
    assert "--kv_cache_free_gpu_memory_fraction" in inner


def test_trtllm_docker_cmd_honors_launcher_script():
    cmd = build_trtllm_docker_cmd(
        "hf-nvidia",
        {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "port": 8111,
            "docker_config": {"launcher_script": "/tmp/start-trt.sh"},
        },
        _docker_app_config(),
    )

    assert cmd == ["/tmp/start-trt.sh"]
