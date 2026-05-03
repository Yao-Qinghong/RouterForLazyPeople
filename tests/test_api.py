from __future__ import annotations

"""API smoke tests for router.main routes and middleware."""

import json
from pathlib import Path

import pytest
import yaml
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import router.main as main_module
import router.proxy as proxy_module
import router.sysinfo as sysinfo_module
from router.config import BackendCapabilities


def _install_fake_ws_httpx(monkeypatch) -> None:
    """Install an httpx.AsyncClient stub used by the WebSocket route."""

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
    chosen = registry if registry is not None else _backend_registry(tmp_path)
    monkeypatch.setattr(main_module, "build_backend_registry", lambda config: dict(chosen))
    monkeypatch.setattr(main_module, "available_engines", lambda config: ["llama.cpp"])
    monkeypatch.setattr(sysinfo_module, "detect_system", lambda llama_bin=None: _sample_system_info())
    return settings_path, main_module.create_app(settings_path)


class TestApiRoutes:
    def test_openai_models_lists_auto_model_id_alias(self, monkeypatch, tmp_path):
        model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
        registry = {
            "local-8000": {
                "engine": "openai",
                "port": 8000,
                "model": model_id,
                "idle_timeout": 86400,
                "startup_wait": 5,
                "description": "External Nemotron",
                "tier": "fast",
                "log": str(tmp_path / "local-8000.log"),
            },
            "hf-nvidia": {
                "engine": "vllm",
                "port": 8111,
                "model": model_id,
                "idle_timeout": 600,
                "startup_wait": 120,
                "description": "Managed duplicate",
                "tier": "fast",
                "log": str(tmp_path / "hf-nvidia.log"),
            },
        }
        _, app = _make_app(tmp_path, monkeypatch, registry=registry)

        with TestClient(app) as client:
            response = client.get("/v1/models")

            assert response.status_code == 200
            ids = {item["id"]: item for item in response.json()["data"]}
            assert ids[model_id]["alias_for"] == "local-8000"

            single = client.get(f"/v1/models/{model_id}")
            assert single.status_code == 200
            assert single.json()["id"] == model_id

    def test_chat_completions_resolves_exact_model_id_without_manual_backend(self, monkeypatch, tmp_path):
        model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
        registry = {
            "local-8000": {
                "engine": "openai",
                "port": 8000,
                "model": model_id,
                "idle_timeout": 86400,
                "startup_wait": 5,
                "description": "External Nemotron",
                "tier": "fast",
                "log": str(tmp_path / "local-8000.log"),
            },
            "hf-nvidia": {
                "engine": "vllm",
                "port": 8111,
                "model": model_id,
                "idle_timeout": 600,
                "startup_wait": 120,
                "description": "Managed duplicate",
                "tier": "fast",
                "log": str(tmp_path / "hf-nvidia.log"),
            },
        }
        _, app = _make_app(tmp_path, monkeypatch, registry=registry)
        observed = {"ensure": None, "url": None}

        async def fake_ensure_running(key: str):
            observed["ensure"] = key

        class FakeResponse:
            status_code = 200
            text = '{"id":"ok"}'

            def json(self):
                return {"id": "ok", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json):
                observed["url"] = url
                return FakeResponse()

        monkeypatch.setattr(proxy_module.httpx, "AsyncClient", FakeAsyncClient)

        with TestClient(app) as client:
            client.app.state.manager.ensure_running = fake_ensure_running

            response = client.post(
                "/v1/chat/completions",
                json={"model": model_id, "messages": [{"role": "user", "content": "hello"}]},
            )

            assert response.status_code == 200
            assert observed["ensure"] == "local-8000"
            assert observed["url"] == "http://localhost:8000/v1/chat/completions"

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
        _install_fake_ws_httpx(monkeypatch)

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            with client.websocket_connect("/v1/chat/completions/ws") as ws:
                ws.send_json({"messages": [{"role": "user", "content": "hello"}]})
                assert ws.receive_json() == {"id": "chunk-1", "object": "chat.completion.chunk"}
                assert ws.receive_json() == {"done": True}

    def test_websocket_unauthenticated_is_rejected_when_auth_enabled(self, monkeypatch, tmp_path):
        api_keys = [{"key": "sk-infer", "name": "infer", "scope": "inference"}]
        _, app = _make_app(tmp_path, monkeypatch, auth_enabled=True, api_keys=api_keys)
        _install_fake_ws_httpx(monkeypatch)

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            from starlette.websockets import WebSocketDisconnect

            with pytest.raises(WebSocketDisconnect) as excinfo:
                with client.websocket_connect("/v1/chat/completions/ws") as ws:
                    ws.receive_json()
            assert excinfo.value.code == 1008

    def test_websocket_authenticated_with_inference_scope_is_accepted(self, monkeypatch, tmp_path):
        api_keys = [{"key": "sk-infer", "name": "infer", "scope": "inference"}]
        _, app = _make_app(tmp_path, monkeypatch, auth_enabled=True, api_keys=api_keys)
        _install_fake_ws_httpx(monkeypatch)

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            with client.websocket_connect(
                "/v1/chat/completions/ws",
                headers={"Authorization": "Bearer sk-infer"},
            ) as ws:
                ws.send_json({"messages": [{"role": "user", "content": "hi"}]})
                assert ws.receive_json() == {"id": "chunk-1", "object": "chat.completion.chunk"}
                assert ws.receive_json() == {"done": True}

    def test_websocket_admin_only_scope_is_rejected(self, monkeypatch, tmp_path):
        """An admin-scoped key must not be allowed to call the inference WS."""
        api_keys = [{"key": "sk-admin", "name": "admin", "scope": "admin"}]
        _, app = _make_app(tmp_path, monkeypatch, auth_enabled=True, api_keys=api_keys)
        _install_fake_ws_httpx(monkeypatch)

        from starlette.websockets import WebSocketDisconnect

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            with pytest.raises(WebSocketDisconnect) as excinfo:
                with client.websocket_connect(
                    "/v1/chat/completions/ws",
                    headers={"x-api-key": "sk-admin"},
                ) as ws:
                    ws.receive_json()
            assert excinfo.value.code == 1008

    def test_websocket_unknown_explicit_backend_is_invalid_request(self, monkeypatch, tmp_path):
        """An explicit top-level `backend` typo is the WS analog of a
        `?backend=` typo on HTTP — a permanent client error, not a retryable
        outage. The connection stays open so the client can fix the payload."""
        _, app = _make_app(tmp_path, monkeypatch)
        _install_fake_ws_httpx(monkeypatch)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/chat/completions/ws") as ws:
                ws.send_json({
                    "messages": [{"role": "user", "content": "hi"}],
                    "backend": "does-not-exist",
                })
                msg = ws.receive_json()
                assert msg["error"]["type"] == "invalid_request_error"
                assert msg["error"]["param"] == "backend"
                assert "fast" in msg["error"]["valid_backends"]

    def test_websocket_no_backend_available_returns_service_unavailable(self, monkeypatch, tmp_path):
        """The `service_unavailable` envelope is reserved for the case where
        no explicit selection was made and the classifier cannot satisfy the
        request — i.e. the server has nothing to route to right now."""
        _, app = _make_app(tmp_path, monkeypatch, registry={})
        _install_fake_ws_httpx(monkeypatch)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/chat/completions/ws") as ws:
                ws.send_json({
                    "messages": [{"role": "user", "content": "hi"}],
                })
                msg = ws.receive_json()
                assert msg["error"]["type"] == "service_unavailable"

    def test_openai_no_backend_returns_503_with_retry_after(self, monkeypatch, tmp_path):
        """OpenAI surface: no usable backend → 503 + Retry-After (per docs)."""
        _, app = _make_app(tmp_path, monkeypatch, registry={})

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

            assert response.status_code == 503
            body = response.json()
            assert body["type"] == "service_unavailable"
            assert "retry_after" in body
            assert response.headers.get("Retry-After")

    def test_openai_capability_mismatch_returns_503(self, monkeypatch, tmp_path):
        """A tool-calling request with no tool-capable backend must be a clean
        503 — never silently routed to an incapable backend."""
        registry = {
            "fast": {
                "engine": "llama.cpp",
                "port": 18080,
                "idle_timeout": 300,
                "startup_wait": 30,
                "description": "Fast no-tools",
                "tier": "fast",
                "ctx_size": 8192,
                "log": str(tmp_path / "fast.log"),
                # No capabilities → treated as not tool-capable.
            }
        }
        _, app = _make_app(tmp_path, monkeypatch, registry=registry)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function",
                               "function": {"name": "noop", "parameters": {}}}],
                },
            )
            assert response.status_code == 503
            assert response.json()["type"] == "service_unavailable"

    def test_openai_combined_tools_and_json_schema_returns_503_when_deep_is_tool_only(
        self, monkeypatch, tmp_path
    ):
        """A request that combines ``tools`` and ``response_format=json_schema``
        must land on a backend that declares **both** capabilities.  When the
        only deep backend is tool-capable but schema-incapable, the classifier
        must fail closed (503) rather than silently routing — the previous
        ``elif`` filter would have let this through."""
        registry = {
            "fast": {
                "engine": "llama.cpp",
                "port": 18080,
                "idle_timeout": 300,
                "startup_wait": 30,
                "description": "Fast",
                "tier": "fast",
                "ctx_size": 8192,
                "log": str(tmp_path / "fast.log"),
                "capabilities": BackendCapabilities(
                    supports_tools=False, supports_json_schema=False
                ),
            },
            "deep-tool-only": {
                "engine": "llama.cpp",
                "port": 18081,
                "idle_timeout": 300,
                "startup_wait": 30,
                "description": "Deep tool-only",
                "tier": "deep",
                "ctx_size": 8192,
                "log": str(tmp_path / "deep.log"),
                "capabilities": BackendCapabilities(
                    supports_tools=True, supports_json_schema=False
                ),
            },
        }
        _, app = _make_app(tmp_path, monkeypatch, registry=registry)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function",
                               "function": {"name": "noop", "parameters": {}}}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "x",
                            "schema": {"type": "object"},
                        },
                    },
                },
            )

            assert response.status_code == 503
            assert response.json()["type"] == "service_unavailable"

    def test_anthropic_no_backend_returns_503(self, monkeypatch, tmp_path):
        """Anthropic surface: no usable backend → 503 with api_error envelope."""
        _, app = _make_app(tmp_path, monkeypatch, registry={})

        with TestClient(app) as client:
            response = client.post(
                "/anthropic/v1/messages",
                json={"model": "claude-3-haiku",
                      "messages": [{"role": "user", "content": "hi"}]},
            )

            assert response.status_code == 503
            body = response.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "api_error"
            assert response.headers.get("Retry-After")

    def test_gemini_no_backend_keeps_400(self, monkeypatch, tmp_path):
        """Gemini surface keeps 400 per docs (different convention)."""
        _, app = _make_app(tmp_path, monkeypatch, registry={})

        async def fake_handle(*, model, is_stream, request, manager, metrics_store, config):
            from router.proxy import handle_gemini_proxy
            return await handle_gemini_proxy(
                model=model, is_stream=is_stream, request=request,
                manager=manager, metrics_store=metrics_store, config=config,
            )

        with TestClient(app) as client:
            response = client.post(
                "/gemini/v1beta/models/gemini-pro:generateContent",
                json={"contents": [{"role": "user",
                                    "parts": [{"text": "hi"}]}]},
            )
            assert response.status_code == 400

    def test_openai_invalid_explicit_backend_returns_400(self, monkeypatch, tmp_path):
        """An explicit ?backend= typo is a client error (400), not a 503.
        Returning 503 would make SDKs retry typos as if the server were
        temporarily down."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions?backend=does-not-exist",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

            assert response.status_code == 400
            body = response.json()
            assert body["type"] == "invalid_request_error"
            assert body["param"] == "backend"
            assert "fast" in body["valid_backends"]
            # No Retry-After: this is permanent until the client fixes the request.
            assert response.headers.get("Retry-After") is None

    def test_anthropic_invalid_explicit_backend_returns_400(self, monkeypatch, tmp_path):
        """Anthropic surface: invalid ?backend= → 400 with anthropic envelope."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"

        with TestClient(app) as client:
            response = client.post(
                "/anthropic/v1/messages?backend=does-not-exist",
                json={"model": "claude-3-haiku",
                      "messages": [{"role": "user", "content": "hi"}]},
            )

            assert response.status_code == 400
            body = response.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "invalid_request_error"
            assert "does-not-exist" in body["error"]["message"]

    def test_gemini_invalid_explicit_backend_returns_400(self, monkeypatch, tmp_path):
        """Gemini surface: invalid ?backend= → 400 INVALID_ARGUMENT envelope."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"

        with TestClient(app) as client:
            response = client.post(
                "/gemini/v1beta/models/gemini-pro:generateContent?backend=does-not-exist",
                json={"contents": [{"role": "user",
                                    "parts": [{"text": "hi"}]}]},
            )

            assert response.status_code == 400
            body = response.json()
            assert body["error"]["status"] == "INVALID_ARGUMENT"
            assert "does-not-exist" in body["error"]["message"]

    def test_openai_invalid_route_prefix_returns_400(self, monkeypatch, tmp_path):
        """`[route:<typo>]` parity with `?backend=<typo>`: fail fast 400, not
        silent fallback to the classifier."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user",
                                    "content": "[route:does-not-exist] hello"}]},
            )

            assert response.status_code == 400
            body = response.json()
            assert body["type"] == "invalid_request_error"
            assert body["param"] == "route"
            assert "fast" in body["valid_backends"]

    def test_anthropic_invalid_route_prefix_returns_400(self, monkeypatch, tmp_path):
        """Anthropic surface: invalid `[route:key]` → 400 with anthropic envelope."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"

        with TestClient(app) as client:
            response = client.post(
                "/anthropic/v1/messages",
                json={"model": "claude-3-haiku",
                      "messages": [{"role": "user",
                                    "content": "[route:does-not-exist] hi"}]},
            )

            assert response.status_code == 400
            body = response.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "invalid_request_error"
            assert "does-not-exist" in body["error"]["message"]

    def test_gemini_invalid_route_prefix_returns_400(self, monkeypatch, tmp_path):
        """Gemini surface: invalid `[route:key]` → 400 INVALID_ARGUMENT."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"

        with TestClient(app) as client:
            response = client.post(
                "/gemini/v1beta/models/gemini-pro:generateContent",
                json={"contents": [{"role": "user",
                                    "parts": [{"text": "[route:does-not-exist] hi"}]}]},
            )

            assert response.status_code == 400
            body = response.json()
            assert body["error"]["status"] == "INVALID_ARGUMENT"
            assert "does-not-exist" in body["error"]["message"]

    def test_websocket_invalid_route_prefix_returns_invalid_request(self, monkeypatch, tmp_path):
        """WS parity: `[route:<typo>]` returns an invalid_request_error
        message and keeps the connection open for the next payload."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"
        _install_fake_ws_httpx(monkeypatch)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/chat/completions/ws") as ws:
                ws.send_json({
                    "messages": [{"role": "user",
                                  "content": "[route:does-not-exist] hi"}],
                })
                msg = ws.receive_json()
                assert msg["error"]["type"] == "invalid_request_error"
                assert msg["error"]["param"] == "route"
                assert "fast" in msg["error"]["valid_backends"]

    def test_websocket_invalid_route_prefix_with_top_level_backend_still_400(self, monkeypatch, tmp_path):
        """A `[route:<typo>]` prefix must fail fast even when the WS payload
        also supplies a top-level `backend` field — the prefix is validated
        before the top-level short-circuit, so it cannot leak silently."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"
        _install_fake_ws_httpx(monkeypatch)

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            with client.websocket_connect("/v1/chat/completions/ws") as ws:
                ws.send_json({
                    "backend": "fast",
                    "messages": [{"role": "user",
                                  "content": "[route:does-not-exist] hi"}],
                })
                msg = ws.receive_json()
                assert msg["error"]["type"] == "invalid_request_error"
                assert msg["error"]["param"] == "route"

    def test_websocket_route_prefix_is_stripped_before_forwarding(self, monkeypatch, tmp_path):
        """A valid `[route:<key>]` prefix must be stripped from the payload
        before forwarding to the backend, even when the WS caller also
        supplied a top-level `backend` field."""
        _, app = _make_app(tmp_path, monkeypatch)  # default registry has "fast"

        captured: dict = {}

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
                captured["body"] = json
                return FakeStreamResponse()

        monkeypatch.setattr(main_module.httpx, "AsyncClient", FakeAsyncClient)

        with TestClient(app) as client:
            async def fake_ensure_running(key: str):
                return None

            client.app.state.manager.ensure_running = fake_ensure_running

            with client.websocket_connect("/v1/chat/completions/ws") as ws:
                ws.send_json({
                    "backend": "fast",
                    "messages": [{"role": "user",
                                  "content": "[route:fast] hello"}],
                })
                # Drain the streamed response so the forwarder runs to completion.
                ws.receive_json()
                ws.receive_json()

        forwarded = captured["body"]["messages"][0]["content"]
        assert forwarded == "hello", (
            f"route prefix leaked to backend: {forwarded!r}"
        )
