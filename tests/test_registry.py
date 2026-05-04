"""Tests for router/registry.py — registry validation."""

from router.registry import RegistryValidation, validate_registry


def _backend(engine="llama.cpp", model=None, model_dir=None, port=None):
    cfg = {"engine": engine}
    if model is not None:
        cfg["model"] = model
    if model_dir is not None:
        cfg["model_dir"] = model_dir
    if port is not None:
        cfg["port"] = port
    return cfg


class TestValidateRegistry:
    def test_clean_registry_has_no_issues(self, tmp_path):
        real = tmp_path / "model.gguf"
        real.write_bytes(b"")
        backends = {
            "ok-1": _backend(model=str(real), port=8100),
            "ok-2": _backend(engine="vllm", model_dir=str(tmp_path), port=8101),
        }
        report = validate_registry(backends)
        assert isinstance(report, RegistryValidation)
        assert not report.has_issues
        assert report.invalid_keys == set()

    def test_detects_missing_model_path(self, tmp_path):
        backends = {
            "stale": _backend(model=str(tmp_path / "does-not-exist.gguf"), port=8100),
        }
        report = validate_registry(backends)
        assert report.has_issues
        assert "stale" in report.invalid_keys
        assert any(slug == "stale" for slug, _ in report.missing_paths)

    def test_detects_duplicate_ports(self, tmp_path):
        real = tmp_path / "m.gguf"
        real.write_bytes(b"")
        backends = {
            "a": _backend(model=str(real), port=8101),
            "b": _backend(model=str(real), port=8101),
        }
        report = validate_registry(backends)
        assert 8101 in report.port_conflicts
        assert sorted(report.port_conflicts[8101]) == ["a", "b"]
        assert report.invalid_keys == {"a", "b"}

    def test_skips_external_engines(self):
        # openai/ollama backends do not need a local model path.
        backends = {
            "lmstudio": {"engine": "openai", "port": 1234},
            "ollama-llama": {"engine": "ollama", "port": 11434},
        }
        report = validate_registry(backends)
        assert not report.has_issues
        assert report.invalid_keys == set()

    def test_handles_backendconfig_dataclass(self, tmp_path):
        # Manual entries arrive as BackendConfig dataclasses, not dicts.
        from router.config import BackendConfig
        cfg = BackendConfig(
            engine="llama.cpp",
            model=str(tmp_path / "missing.gguf"),
            port=8200,
        )
        report = validate_registry({"manual": cfg})
        assert "manual" in report.invalid_keys
        assert any(slug == "manual" for slug, _ in report.missing_paths)

    def test_hf_model_id_is_not_treated_as_missing_path(self, tmp_path):
        # vLLM/SGLang/HF backends commonly use Hugging Face model IDs
        # (e.g. "Qwen/Qwen2.5-72B-Instruct-AWQ") which are not filesystem
        # paths. They must not be flagged invalid.
        backends = {
            "vllm-hf": {
                "engine": "vllm",
                "port": 8200,
                "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
            },
            "sglang-hf": {
                "engine": "sglang",
                "port": 8201,
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            },
            "hf-hf": {
                "engine": "huggingface",
                "port": 8202,
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
            },
        }
        report = validate_registry(backends)
        assert not report.has_issues
        assert report.invalid_keys == set()

    def test_vllm_local_path_that_does_not_exist_is_flagged(self, tmp_path):
        # If a vLLM backend points at an absolute path on disk that no
        # longer exists, that *should* be flagged (it's not an HF ID).
        backends = {
            "vllm-stale": {
                "engine": "vllm",
                "port": 8210,
                "model": str(tmp_path / "missing-model"),
            },
        }
        report = validate_registry(backends)
        assert "vllm-stale" in report.invalid_keys

    def test_vllm_existing_model_dir_is_ok(self, tmp_path):
        backends = {
            "vllm-local": {
                "engine": "vllm",
                "port": 8211,
                "model_dir": str(tmp_path),
            },
        }
        report = validate_registry(backends)
        assert not report.has_issues

    def test_port_conflict_with_stale_sibling_does_not_skip_valid(self, tmp_path):
        # Mirrors the DGX scenario: a stale manual backend and a valid
        # backend share a port. The stale one is already excluded for a
        # missing path; the valid one must remain benchmarkable.
        real = tmp_path / "ok.gguf"
        real.write_bytes(b"")
        backends = {
            "stale": {
                "engine": "llama.cpp",
                "port": 8101,
                "model": str(tmp_path / "missing.gguf"),
            },
            "ok": {
                "engine": "llama.cpp",
                "port": 8101,
                "model": str(real),
            },
        }
        report = validate_registry(backends)
        # Stale is invalid for the missing path; "ok" is preserved.
        assert "stale" in report.invalid_keys
        assert "ok" not in report.invalid_keys
        assert 8101 not in report.port_conflicts

    def test_two_valid_backends_on_same_port_are_both_flagged(self, tmp_path):
        real = tmp_path / "m.gguf"
        real.write_bytes(b"")
        backends = {
            "a": {"engine": "llama.cpp", "port": 8300, "model": str(real)},
            "b": {"engine": "llama.cpp", "port": 8300, "model": str(real)},
        }
        report = validate_registry(backends)
        assert 8300 in report.port_conflicts
        assert report.invalid_keys == {"a", "b"}

    def test_summary_lines_format(self, tmp_path):
        backends = {
            "stale": _backend(model=str(tmp_path / "nope.gguf"), port=8100),
            "dup-a": _backend(model=str(tmp_path), port=8200),
            "dup-b": _backend(model=str(tmp_path), port=8200),
        }
        report = validate_registry(backends)
        lines = report.summary_lines()
        assert any("stale" in line and "model path does not exist" in line for line in lines)
        assert any("8200" in line and "dup-a" in line and "dup-b" in line for line in lines)
