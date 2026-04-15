"""
router/provider.py — Engine provider protocol and registry

Each engine has a provider that knows how to:
  - Build the subprocess command to start the server
  - Return the health-check URL
  - Rewrite outbound model names for provider-specific APIs

Adding a new engine:
  1. Create a provider class implementing EngineProvider
  2. Register it with register_provider()
  3. That's it — lifecycle.py and proxy.py pick it up automatically
"""
from __future__ import annotations

from typing import Protocol, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig


class EngineProvider(Protocol):
    """Protocol that each engine provider must satisfy."""

    engine: str

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        """Build the subprocess command to start this engine."""
        ...

    def health_url(self, cfg: dict) -> str:
        """Return the URL to probe for health/readiness."""
        ...

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        """Rewrite the outbound model name for this engine's API.

        Default: return model unchanged.
        """
        ...


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────
_providers: dict[str, EngineProvider] = {}


def register_provider(provider: EngineProvider):
    """Register an engine provider."""
    _providers[provider.engine] = provider


def get_provider(engine: str) -> Optional[EngineProvider]:
    """Look up a registered provider by engine name."""
    return _providers.get(engine)


# ─────────────────────────────────────────────────────────────
# Concrete providers
# ─────────────────────────────────────────────────────────────

class LlamaProvider:
    engine = "llama.cpp"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        from router.engines import build_llama_cmd
        return build_llama_cmd(cfg, config)

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/health"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return model


class VllmProvider:
    engine = "vllm"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        from router.engines import build_vllm_cmd
        return build_vllm_cmd(cfg)

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/health"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return cfg.get("model") or cfg.get("model_dir", "") or model


class SglangProvider:
    engine = "sglang"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        from router.engines import build_sglang_cmd
        return build_sglang_cmd(cfg)

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/health"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return cfg.get("model") or cfg.get("model_dir", "") or model


class TrtLlmProvider:
    engine = "trt-llm"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        from router.engines import build_trtllm_cmd
        trt_config = kwargs.get("trt_config", {})
        return build_trtllm_cmd(cfg, trt_config)

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/health"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return model


class TrtLlmDockerProvider:
    engine = "trt-llm-docker"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        from router.engines import build_trtllm_docker_cmd
        key = kwargs.get("key", "")
        return build_trtllm_docker_cmd(key, cfg, config)

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/v1/models"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return model


class HuggingFaceProvider:
    engine = "huggingface"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        from router.engines import build_hf_cmd
        return build_hf_cmd(cfg)

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/health"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return cfg.get("model") or cfg.get("model_dir", "") or model


class OllamaProvider:
    engine = "ollama"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        from router.engines import build_ollama_cmd
        return build_ollama_cmd(cfg)

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/api/tags"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return cfg.get("model", "") or model


class OpenAIProvider:
    engine = "openai"

    def build_cmd(self, cfg: dict, config: AppConfig, **kwargs) -> list[str]:
        raise ValueError("Engine 'openai' uses an external server — no command to build")

    def health_url(self, cfg: dict) -> str:
        return f"http://localhost:{cfg['port']}/v1/models"

    def rewrite_model_name(self, cfg: dict, model: str) -> str:
        return model


# ─────────────────────────────────────────────────────────────
# Auto-register all built-in providers
# ─────────────────────────────────────────────────────────────
for _cls in (
    LlamaProvider, VllmProvider, SglangProvider,
    TrtLlmProvider, TrtLlmDockerProvider,
    HuggingFaceProvider, OllamaProvider, OpenAIProvider,
):
    register_provider(_cls())
