"""Tests for router/registry.py — registry validation and port-conflict avoidance."""

from types import SimpleNamespace

from router import registry as reg
from router.registry import RegistryValidation, build_backend_registry, validate_registry


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

    def test_trimmed_registry_view_without_paths_is_not_flagged(self):
        # The `/backends` HTTP endpoint exposes a trimmed view that may
        # have `model: null` (or omit it entirely) for valid backends.
        # validate_registry must not treat that as a stale-path failure.
        backends = {
            "llama-no-path": {
                "engine": "llama.cpp",
                "port": 8400,
                # no `model` key at all
            },
            "llama-null-path": {
                "engine": "llama.cpp",
                "port": 8401,
                "model": None,
                "model_dir": None,
            },
            "vllm-id-only": {
                "engine": "vllm",
                "port": 8402,
                "model": None,
            },
        }
        report = validate_registry(backends)
        assert not report.has_issues
        assert report.invalid_keys == set()

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

class TestBuildBackendRegistryPortCollisions:
    def _config(self, tmp_path):
        return SimpleNamespace(
            engines_enabled=["llama.cpp"],
            data_dir=tmp_path,
            discovery=SimpleNamespace(port_start=8100, port_end=8200),
        )

    def test_discovered_port_colliding_with_manual_is_reassigned(
        self, tmp_path, monkeypatch
    ):
        # Manual nemotron sits on 8101. Discovery would assign 8100, 8101,
        # 8102 sequentially without knowing 8101 is taken — the gemma
        # entry must get bumped to the next free port (8103) instead of
        # silently colliding.
        manual = {
            "nemotron": {"engine": "llama.cpp", "port": 8101, "model": "/m/nemo.gguf"},
        }
        gguf = {
            "first":  {"engine": "llama.cpp", "port": 8100, "model": "/m/a.gguf"},
            "gemma":  {"engine": "llama.cpp", "port": 8101, "model": "/m/g.gguf"},
            "third":  {"engine": "llama.cpp", "port": 8102, "model": "/m/c.gguf"},
        }
        monkeypatch.setattr(reg, "load_backends", lambda config: dict(manual))
        monkeypatch.setattr(reg, "detect_running_servers", lambda config: {})
        monkeypatch.setattr(reg, "discover_gguf_models", lambda config, c: gguf)
        monkeypatch.setattr(reg, "discover_hf_models", lambda config, c: {})
        monkeypatch.setattr(reg, "discover_trtllm_engines", lambda config, c: {})
        monkeypatch.setattr(reg, "save_discovery_cache", lambda discovered, config: None)
        monkeypatch.setattr(reg, "load_user_overrides", lambda config: {})

        registry = build_backend_registry(self._config(tmp_path))

        # No port appears twice across the merged registry.
        ports = [v["port"] for v in registry.values() if isinstance(v, dict)]
        assert len(ports) == len(set(ports)), f"duplicate ports: {ports}"
        # Manual nemotron keeps 8101.
        assert registry["nemotron"]["port"] == 8101
        # The discovered "gemma" entry that originally collided with the
        # manual port no longer does.
        assert registry["gemma"]["port"] != 8101

    def test_discovery_with_no_collision_keeps_assigned_ports(
        self, tmp_path, monkeypatch
    ):
        manual = {"manual-a": {"engine": "llama.cpp", "port": 8500, "model": "/m/x.gguf"}}
        gguf = {
            "disc-1": {"engine": "llama.cpp", "port": 8100, "model": "/m/y.gguf"},
            "disc-2": {"engine": "llama.cpp", "port": 8101, "model": "/m/z.gguf"},
        }
        monkeypatch.setattr(reg, "load_backends", lambda config: dict(manual))
        monkeypatch.setattr(reg, "detect_running_servers", lambda config: {})
        monkeypatch.setattr(reg, "discover_gguf_models", lambda config, c: gguf)
        monkeypatch.setattr(reg, "discover_hf_models", lambda config, c: {})
        monkeypatch.setattr(reg, "discover_trtllm_engines", lambda config, c: {})
        monkeypatch.setattr(reg, "save_discovery_cache", lambda discovered, config: None)
        monkeypatch.setattr(reg, "load_user_overrides", lambda config: {})

        registry = build_backend_registry(self._config(tmp_path))
        assert registry["disc-1"]["port"] == 8100
        assert registry["disc-2"]["port"] == 8101
        assert registry["manual-a"]["port"] == 8500
