from __future__ import annotations

"""API smoke tests for router.main routes and middleware."""

import json
from pathlib import Path

import yaml
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import router.main as main_module
import router.sysinfo as sysinfo_module


def _sample_system_info() -> dict:
    return {
        "platform": {"os": "Linux", "arch": "x86_64", "os_version": "6.8.0", "python": "3.11.9"},
        "cpu": {"model": "Test CPU", "cores": 8},
        "ram": {"total_gb": 32.0},
        "gpu": {"available": False, "driver_version": None, "devices": []},
        "cuda": {"available": False, "version": None},
        "engine_versions": {},
        "recommendations": {},
        "conflict_processes": [],
    }


def _backend_registry(tmp_path: Path) -> dict:
    return {
        "fast": {
            "engine": "llama.cpp",
            "port": 18080,
            "idle_timeout": 300,
            "startup_wait": 30,
            "description": "Fast backend",
            "tier": "fast",
            "ctx_size": 8192,
            "log": str(tmp_path / "fast.log"),
        }
    }


def _write_config(tmp_path: Path, *, auth_enabled: bool = False,
                  api_keys: list[dict] | None = None,
                  model_aliases: dict[str, str] | None = None) -> Path:
    settings = {
        "router": {"host": "0.0.0.0", "port": 9001, "log_level": "INFO"},
        "logging": {"log_dir": str(tmp_path / "logs")},
        "llama_bin": str(tmp_path / "llama-server"),
        "data_dir": str(tmp_path / "data"),
        "scan_dirs": {"gguf": [], "hf": [], "trtllm": []},
        "discovery": {"port_start": 8100, "port_end": 8299},
        "routing": {
            "token_threshold_deep": 4000,
            "token_threshold_mid": 500,
            "deep_keywords": ["reason"],
            "mid_keywords": ["write"],
        },
        "proxy": {"timeout_sec": 5, "max_concurrent_requests": 2, "queue_timeout_sec": 5},
        "auth": {"enabled": auth_enabled, "api_keys": api_keys or []},
        "model_aliases": model_aliases or {},
    }
    settings_path = tmp_path / "settings.yaml"
    backends_path = tmp_path / "backends.yaml"
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    backends_path.write_text("backends: {}\n")
    return settings_path


def _make_app(tmp_path: Path, monkeypatch, *, auth_enabled: bool = False,
              api_keys: list[dict] | None = None,
              model_aliases: dict[str, str] | None = None,
              registry: dict | None = None):
    settings_path = _write_config(
        tmp_path,
        auth_enabled=auth_enabled,
        api_keys=api_keys,
        model_aliases=model_aliases,
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda config: None)
    monkeypatch.setattr(main_module, "build_backend_registry", lambda config: dict(registry or _backend_registry(tmp_path)))
    monkeypatch.setattr(main_module, "available_engines", lambda config: ["llama.cpp"])
    monkeypatch.setattr(sysinfo_module, "detect_system", lambda llama_bin=None: _sample_system_info())
    return settings_path, main_module.create_app(settings_path)


class TestApiRoutes:
    def test_reload_config_uses_original_settings_file(self, monkeypatch, tmp_path):
        settings_path, app = _make_app(tmp_path, monkeypatch)

        with TestClient(app) as client:
            settings = yaml.safe_load(settings_path.read_text())
            settings["model_aliases"] = {"gpt-4": "fast"}
            settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))

            response = client.post("/reload-config")

            assert response.status_code == 200
            assert response.json()["model_aliases"] == {"gpt-4": "fast"}
            assert client.app.state.config.model_aliases == {"gpt-4": "fast"}
            assert client.app.state.config.settings_file == settings_path

    def test_gemini_routes_are_wired(self, monkeypatch, tmp_path):
        _, app = _make_app(tmp_path, monkeypatch)

        async def fake_gemini_proxy(*, model, is_stream, **kwargs):
            return JSONResponse({"model": model, "stream": is_stream})

        monkeypatch.setattr(main_module, "handle_gemini_proxy", fake_gemini_proxy)

        with TestClient(app) as client:
            resp = client.post("/gemini/v1beta/models/gemini-2.0-flash-latest:generateContent", json={})
            stream_resp = client.post("/gemini/v1beta/models/gemini-2.0-flash-latest:streamGenerateContent", json={})

            assert resp.status_code == 200
            assert resp.json() == {"model": "gemini-2.0-flash-latest", "stream": False}
            assert stream_resp.status_code == 200
            assert stream_resp.json() == {"model": "gemini-2.0-flash-latest", "stream": True}

    def test_auth_boundaries_cover_public_inference_and_admin_paths(self, monkeypatch, tmp_path):
        api_keys = [
            {"key": "sk-infer", "name": "infer", "scope": "inference"},
            {"key": "sk-admin", "name": "admin", "scope": "admin"},
        ]
        _, app = _make_app(tmp_path, monkeypatch, auth_enabled=True, api_keys=api_keys)

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            assert client.get("/status").status_code == 200
            assert client.get("/v1/models").status_code == 200
            assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 401
            assert client.post("/start/fast", headers={"Authorization": "Bearer sk-infer"}).status_code == 403

            admin_resp = client.post("/start/fast", headers={"x-api-key": "sk-admin"})
            assert admin_resp.status_code == 200
            assert admin_resp.json()["status"] == "started"

    def test_websocket_streaming_route_streams_chunks(self, monkeypatch, tmp_path):
        _, app = _make_app(tmp_path, monkeypatch)

        class FakeStreamResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self):
                for chunk in [
                    b'data: {"id":"chunk-1","object":"chat.completion.chunk"}\n\n',
                    b"data: [DONE]\n\n",
                ]:
                    yield chunk

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return FakeStreamResponse()

        monkeypatch.setattr(main_module.httpx, "AsyncClient", FakeAsyncClient)

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            with client.websocket_connect("/v1/chat/completions/ws") as ws:
                ws.send_json({"messages": [{"role": "user", "content": "hello"}]})
                assert ws.receive_json() == {"id": "chunk-1", "object": "chat.completion.chunk"}
                assert ws.receive_json() == {"done": True}
