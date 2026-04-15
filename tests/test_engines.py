"""Tests for router/engines.py — engine flag detection and command builders."""

from unittest.mock import MagicMock, patch

import pytest

from router.engines import (
    build_llama_cmd,
    clear_llama_flag_cache,
    _detect_llama_flags,
)
import router.engines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(llama_bin: str = "/usr/local/bin/llama-server"):
    config = MagicMock()
    config.llama_bin = llama_bin
    return config


HELP_WITHOUT_REASONING = (
    "--model MODEL\n"
    "--port PORT\n"
    "--ctx-size N\n"
    "--n-gpu-layers N\n"
)

HELP_WITH_ALL_FLAGS = (
    "--model MODEL\n"
    "--port PORT\n"
    "--ctx-size N\n"
    "--n-gpu-layers N\n"
    "--flash-attn on|off\n"
    "--reasoning on|off\n"
    "--reasoning-budget N\n"
)


# ---------------------------------------------------------------------------
# Flag detection
# ---------------------------------------------------------------------------

class TestDetectLlamaFlags:
    def setup_method(self):
        clear_llama_flag_cache()

    def teardown_method(self):
        clear_llama_flag_cache()

    def test_parses_double_dash_flags(self):
        mock_result = MagicMock(stdout=HELP_WITH_ALL_FLAGS, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            flags = _detect_llama_flags("/usr/local/bin/llama-server")
        assert "--model" in flags
        assert "--reasoning" in flags
        assert "--flash-attn" in flags
        assert "--reasoning-budget" in flags

    def test_parses_short_long_combo_format(self):
        clear_llama_flag_cache()
        help_text = "-fa, --flash-attn on|off\n-m, --model MODEL\n"
        mock_result = MagicMock(stdout=help_text, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            flags = _detect_llama_flags("/usr/local/bin/llama-server")
        assert "--flash-attn" in flags
        assert "--model" in flags

    def test_returns_empty_on_error(self):
        clear_llama_flag_cache()
        with patch("router.engines.subprocess.run", side_effect=FileNotFoundError):
            flags = _detect_llama_flags("/no/such/binary")
        assert flags == set()

    def test_caches_result(self):
        clear_llama_flag_cache()
        mock_result = MagicMock(stdout="--model MODEL\n", stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result) as mock_run:
            _detect_llama_flags("/usr/local/bin/llama-server")
            _detect_llama_flags("/usr/local/bin/llama-server")
        # subprocess.run should only be called once due to caching
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------

class TestClearLlamaFlagCache:
    def test_clears_cache(self):
        # Manually set cache to a known value
        router.engines._llama_supported_flags = {"--model"}
        clear_llama_flag_cache()
        assert router.engines._llama_supported_flags is None

    def test_clears_from_none(self):
        """Clearing when already None is a no-op."""
        router.engines._llama_supported_flags = None
        clear_llama_flag_cache()
        assert router.engines._llama_supported_flags is None


# ---------------------------------------------------------------------------
# build_llama_cmd — conditional flags
# ---------------------------------------------------------------------------

class TestBuildLlamaCmdFlagDetection:
    def setup_method(self):
        clear_llama_flag_cache()

    def teardown_method(self):
        clear_llama_flag_cache()

    def test_reasoning_omitted_when_unsupported(self):
        """--reasoning should not appear in cmd when llama-server lacks it."""
        mock_result = MagicMock(stdout=HELP_WITHOUT_REASONING, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080}
            cmd = build_llama_cmd(cfg, _make_config())
        assert "--reasoning" not in cmd

    def test_reasoning_included_when_supported(self):
        """--reasoning should appear in cmd when llama-server supports it."""
        mock_result = MagicMock(stdout=HELP_WITH_ALL_FLAGS, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080, "reasoning": True}
            cmd = build_llama_cmd(cfg, _make_config())
        idx = cmd.index("--reasoning")
        assert cmd[idx + 1] == "on"

    def test_reasoning_off_by_default_when_supported(self):
        """--reasoning defaults to 'off' when flag is supported but cfg has no reasoning key."""
        mock_result = MagicMock(stdout=HELP_WITH_ALL_FLAGS, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080}
            cmd = build_llama_cmd(cfg, _make_config())
        idx = cmd.index("--reasoning")
        assert cmd[idx + 1] == "off"

    def test_flash_attn_omitted_when_unsupported(self):
        """--flash-attn should not appear in cmd when unsupported."""
        mock_result = MagicMock(stdout=HELP_WITHOUT_REASONING, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080}
            cmd = build_llama_cmd(cfg, _make_config())
        assert "--flash-attn" not in cmd

    def test_flash_attn_included_when_supported(self):
        """--flash-attn should appear in cmd when supported."""
        mock_result = MagicMock(stdout=HELP_WITH_ALL_FLAGS, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080}
            cmd = build_llama_cmd(cfg, _make_config())
        idx = cmd.index("--flash-attn")
        assert cmd[idx + 1] == "on"

    def test_reasoning_budget_omitted_when_unsupported(self):
        """--reasoning-budget should not appear even if cfg has the value, when unsupported."""
        mock_result = MagicMock(stdout=HELP_WITHOUT_REASONING, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080, "reasoning_budget": 1024}
            cmd = build_llama_cmd(cfg, _make_config())
        assert "--reasoning-budget" not in cmd

    def test_reasoning_budget_included_when_supported_and_set(self):
        """--reasoning-budget should appear when flag is supported AND cfg has the value."""
        mock_result = MagicMock(stdout=HELP_WITH_ALL_FLAGS, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080, "reasoning_budget": 1024}
            cmd = build_llama_cmd(cfg, _make_config())
        idx = cmd.index("--reasoning-budget")
        assert cmd[idx + 1] == "1024"

    def test_reasoning_budget_omitted_when_supported_but_not_set(self):
        """--reasoning-budget should not appear when flag is supported but cfg has no value."""
        mock_result = MagicMock(stdout=HELP_WITH_ALL_FLAGS, stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080}
            cmd = build_llama_cmd(cfg, _make_config())
        assert "--reasoning-budget" not in cmd

    def test_core_flags_always_present(self):
        """--model, --ctx-size, --n-gpu-layers, --host, --port are always present."""
        mock_result = MagicMock(stdout="", stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            cfg = {"model": "/tmp/test.gguf", "port": 8080}
            cmd = build_llama_cmd(cfg, _make_config())
        assert "--model" in cmd
        assert "--ctx-size" in cmd
        assert "--n-gpu-layers" in cmd
        assert "--host" in cmd
        assert "--port" in cmd


# ── Provider registry ────────────────────────────────────────

class TestProviderRegistry:
    """Tests for router/provider.py — provider protocol and registry."""

    def test_all_engines_have_providers(self):
        from router.engines import ALL_ENGINES
        from router.provider import get_provider
        for engine in ALL_ENGINES:
            assert get_provider(engine) is not None, f"No provider for {engine}"

    def test_provider_cmd_matches_legacy(self):
        """LlamaProvider.build_cmd() produces the same result as build_llama_cmd()."""
        from router.provider import get_provider
        mock_result = MagicMock(stdout="--flash-attn\n--reasoning\n", stderr="")
        with patch("router.engines.subprocess.run", return_value=mock_result):
            clear_llama_flag_cache()
            cfg = {"model": "/tmp/test.gguf", "port": 8080}
            config = _make_config()
            direct = build_llama_cmd(cfg, config)
            clear_llama_flag_cache()
            via_provider = get_provider("llama.cpp").build_cmd(cfg, config)
        assert direct == via_provider

    def test_health_url_via_provider(self):
        from router.provider import get_provider
        cases = [
            ("llama.cpp", {"port": 8080, "engine": "llama.cpp"}, "/health"),
            ("ollama", {"port": 11434, "engine": "ollama"}, "/api/tags"),
            ("openai", {"port": 1234, "engine": "openai"}, "/v1/models"),
            ("trt-llm-docker", {"port": 9000, "engine": "trt-llm-docker"}, "/v1/models"),
        ]
        for engine, cfg, expected_suffix in cases:
            provider = get_provider(engine)
            url = provider.health_url(cfg)
            assert url.endswith(expected_suffix), f"{engine}: {url}"

    def test_openai_provider_raises_on_build_cmd(self):
        from router.provider import get_provider
        provider = get_provider("openai")
        with pytest.raises(ValueError, match="external server"):
            provider.build_cmd({"port": 1234}, MagicMock())

    def test_model_name_rewriting(self):
        from router.provider import get_provider
        # vllm rewrites to cfg model path
        vllm = get_provider("vllm")
        cfg = {"model": "meta-llama/Llama-3-8B", "port": 8000}
        assert vllm.rewrite_model_name(cfg, "gpt-4") == "meta-llama/Llama-3-8B"
        # llama.cpp passes through
        llama = get_provider("llama.cpp")
        assert llama.rewrite_model_name({}, "gpt-4") == "gpt-4"
