from __future__ import annotations

import sys

from router.config import BackendConfig
from router.engines import build_sglang_cmd, build_trtllm_cmd, build_vllm_cmd
from router.proxy import build_model_aliases, resolve_requested_model


class _ManagerStub:
    def __init__(self, running: set[str] | None = None):
        self._running = running or set()

    def is_running(self, key: str) -> bool:
        return key in self._running


def test_python_backends_use_router_interpreter():
    vllm = build_vllm_cmd({"model": "org/model", "port": 8000})
    sglang = build_sglang_cmd({"model": "org/model", "port": 8001})
    trt = build_trtllm_cmd({"model_dir": "/tmp/model", "port": 8002}, {})

    assert vllm[0] == sys.executable
    assert sglang[0] == sys.executable
    assert trt[0] == sys.executable


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
