from __future__ import annotations

"""Documentation consistency checks."""

from pathlib import Path

import yaml

import router.main as main_module


REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
DOCS_INDEX_PATH = REPO_ROOT / "docs" / "README.md"
API_SPEC_PATH = REPO_ROOT / "docs" / "API_SPEC.md"
ARCHITECTURE_PATH = REPO_ROOT / "docs" / "ARCHITECTURE.md"
OPERATIONS_PATH = REPO_ROOT / "docs" / "OPERATIONS.md"
INTERNAL_DOC_ROUTES = {
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
}


def _write_minimal_settings(tmp_path: Path) -> Path:
    settings = {
        "router": {"host": "0.0.0.0", "port": 9001, "log_level": "INFO"},
        "logging": {"log_dir": str(tmp_path / "logs")},
        "llama_bin": str(tmp_path / "llama-server"),
        "data_dir": str(tmp_path / "data"),
        "scan_dirs": {"gguf": [], "hf": [], "trtllm": []},
        "auth": {"enabled": False, "api_keys": []},
        "model_aliases": {},
    }
    settings_path = tmp_path / "settings.yaml"
    backends_path = tmp_path / "backends.yaml"
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    backends_path.write_text("backends: {}\n")
    return settings_path


def _normalized_route_paths(app) -> set[str]:
    paths = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or path in INTERNAL_DOC_ROUTES:
            continue
        if path == "/v1/{path:path}":
            path = "/v1/{path}"
        paths.add(path)
    return paths


class TestDocs:
    def test_api_spec_mentions_all_public_routes(self, monkeypatch, tmp_path):
        settings_path = _write_minimal_settings(tmp_path)
        monkeypatch.setattr(main_module, "setup_logging", lambda config: None)

        app = main_module.create_app(settings_path)
        api_spec = API_SPEC_PATH.read_text()

        missing = sorted(path for path in _normalized_route_paths(app) if path not in api_spec)
        assert not missing, f"API spec is missing public routes: {missing}"

    def test_docs_are_split_into_index_api_architecture_and_operations(self):
        readme = README_PATH.read_text()
        docs_index = DOCS_INDEX_PATH.read_text().lower()
        architecture = ARCHITECTURE_PATH.read_text().lower()
        operations = OPERATIONS_PATH.read_text().lower()

        for path in [DOCS_INDEX_PATH, API_SPEC_PATH, ARCHITECTURE_PATH, OPERATIONS_PATH]:
            assert path.exists()

        for marker in [
            "docs/api_spec.md",
            "docs/architecture.md",
            "docs/operations.md",
        ]:
            assert marker in readme.lower()

        for marker in [
            "api_spec.md",
            "architecture.md",
            "operations.md",
        ]:
            assert marker in docs_index

        for section in [
            "request routing",
            "backend lifecycle",
            "config precedence",
            "runtime boundaries",
            "non-goals",
        ]:
            assert section in architecture

        for section in [
            "supported platforms and engines",
            "startup and shutdown",
            "update behavior",
            "reload and rescan behavior",
            "diagnostics and monitoring",
            "operational limits",
        ]:
            assert section in operations
