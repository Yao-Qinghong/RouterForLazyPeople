"""Tests for router/config.py — configuration loading."""

import pytest
import tempfile
from pathlib import Path

from router.config import (
    load_config, load_backends, AppConfig, BackendConfig,
    BackendCapabilities, ConfigError, _infer_capabilities,
)


class TestLoadConfig:
    def test_loads_default_config(self):
        """Test that the project's settings.yaml loads without error."""
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if not config_path.exists():
            pytest.skip("settings.yaml not found")
        config = load_config(config_path)
        assert isinstance(config, AppConfig)
        assert config.router.port == 9001
        assert config.router.host == "0.0.0.0"

    def test_default_values(self):
        """Verify default values from a minimal config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("router:\n  port: 9001\n")
            sf.flush()

            # Need a backends.yaml too
            backends_path = Path(sf.name).parent / "backends.yaml"
            backends_path.write_text("backends: {}\n")

            try:
                config = load_config(Path(sf.name), backends_path)
                assert config.proxy.timeout_sec == 300
                assert config.proxy.max_concurrent_requests == 20
                assert config.proxy.retry_attempts == 2
                assert config.auth.enabled is False
                assert config.cors.enabled is True
                assert config.audit.enabled is False
                assert config.rate_limit.enabled is False
                assert config.model_aliases == {}
                assert config.preload == []
            finally:
                backends_path.unlink(missing_ok=True)

    def test_auth_config(self):
        """Test auth config parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("""
router:
  port: 9001
auth:
  enabled: true
  api_keys:
    - key: "sk-test"
      name: "test"
      scope: "all"
""")
            sf.flush()
            backends_path = Path(sf.name).parent / "backends.yaml"
            backends_path.write_text("backends: {}\n")
            try:
                config = load_config(Path(sf.name), backends_path)
                assert config.auth.enabled is True
                assert len(config.auth.api_keys) == 1
                assert config.auth.api_keys[0]["key"] == "sk-test"
            finally:
                backends_path.unlink(missing_ok=True)

    def test_model_aliases(self):
        """Test model alias parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("""
router:
  port: 9001
model_aliases:
  gpt-4: deep
  gpt-3.5-turbo: fast
""")
            sf.flush()
            backends_path = Path(sf.name).parent / "backends.yaml"
            backends_path.write_text("backends: {}\n")
            try:
                config = load_config(Path(sf.name), backends_path)
                assert config.model_aliases["gpt-4"] == "deep"
                assert config.model_aliases["gpt-3.5-turbo"] == "fast"
            finally:
                backends_path.unlink(missing_ok=True)

    def test_retry_config(self):
        """Test proxy retry config parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("""
router:
  port: 9001
proxy:
  retry_attempts: 5
  retry_backoff_sec: 2.0
  retry_on_status: [500, 502, 503]
""")
            sf.flush()
            backends_path = Path(sf.name).parent / "backends.yaml"
            backends_path.write_text("backends: {}\n")
            try:
                config = load_config(Path(sf.name), backends_path)
                assert config.proxy.retry_attempts == 5
                assert config.proxy.retry_backoff_sec == 2.0
                assert config.proxy.retry_on_status == [500, 502, 503]
            finally:
                backends_path.unlink(missing_ok=True)


class TestBackendCapabilities:
    def test_explicit_capabilities_override_inferred(self):
        """YAML capabilities block should override heuristic inference."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("router:\n  port: 9001\n")
            sf.flush()
            backends_path = Path(sf.name).parent / "backends.yaml"
            # Small model would normally infer supports_tools=False
            backends_path.write_text("""
backends:
  my-model:
    engine: llama.cpp
    port: 8080
    model: /tmp/small-3b.gguf
    size_gb: 2.0
    capabilities:
      supports_tools: true
      supports_json_schema: true
      code_quality: strong
""")
            try:
                config = load_config(Path(sf.name), backends_path)
                result = load_backends(config)
                backend = result["my-model"]
                assert isinstance(backend, BackendConfig)
                assert backend.capabilities.supports_tools is True
                assert backend.capabilities.supports_json_schema is True
                assert backend.capabilities.code_quality == "strong"
            finally:
                backends_path.unlink(missing_ok=True)

    def test_partial_capabilities_merges_with_inferred(self):
        """Partial YAML capabilities should merge with inferred defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("router:\n  port: 9001\n")
            sf.flush()
            backends_path = Path(sf.name).parent / "backends.yaml"
            backends_path.write_text("""
backends:
  my-model:
    engine: llama.cpp
    port: 8080
    model: /tmp/small-3b.gguf
    size_gb: 2.0
    capabilities:
      supports_tools: true
""")
            try:
                config = load_config(Path(sf.name), backends_path)
                result = load_backends(config)
                backend = result["my-model"]
                # supports_tools overridden to True
                assert backend.capabilities.supports_tools is True
                # Other fields come from inference (small model = weak defaults)
                assert backend.capabilities.code_quality == "weak"
            finally:
                backends_path.unlink(missing_ok=True)

    def test_no_capabilities_block_uses_inference(self):
        """Without capabilities in YAML, heuristics are used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("router:\n  port: 9001\n")
            sf.flush()
            backends_path = Path(sf.name).parent / "backends.yaml"
            backends_path.write_text("""
backends:
  big-model:
    engine: llama.cpp
    port: 8080
    model: /tmp/big-70b.gguf
    size_gb: 40.0
    description: "big-70b model"
""")
            try:
                config = load_config(Path(sf.name), backends_path)
                result = load_backends(config)
                backend = result["big-model"]
                # 40 GB + "70b" in description → strong capabilities inferred
                assert backend.capabilities.supports_tools is True
                assert backend.capabilities.code_quality == "strong"
            finally:
                backends_path.unlink(missing_ok=True)

    def test_vram_estimate_populated_from_size(self):
        """vram_estimate_gb should be calculated from size_gb."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as sf:
            sf.write("router:\n  port: 9001\n")
            sf.flush()
            backends_path = Path(sf.name).parent / "backends.yaml"
            backends_path.write_text("""
backends:
  model-a:
    engine: llama.cpp
    port: 8080
    model: /tmp/model.gguf
    size_gb: 10.0
""")
            try:
                config = load_config(Path(sf.name), backends_path)
                result = load_backends(config)
                backend = result["model-a"]
                # llama.cpp: 10.0 * 1.15 = 11.5
                assert backend.vram_estimate_gb == 11.5
            finally:
                backends_path.unlink(missing_ok=True)
