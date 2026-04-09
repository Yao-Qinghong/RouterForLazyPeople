from __future__ import annotations

"""
router/auth.py — Optional API key authentication middleware

Disabled by default. Enable in settings.yaml:

    auth:
      enabled: true
      api_keys:
        - key: "sk-my-secret-key"
          name: "dev-machine"
          scope: "all"          # "all" | "inference" | "admin"
        - key: "sk-readonly"
          name: "monitoring"
          scope: "inference"

Scopes:
  - "inference": can call /v1/*, /anthropic/*, /gemini/* endpoints
  - "admin": can call /start, /stop, /rescan, /retune, /reload-config, /restart
  - "all": both inference and admin

Unauthenticated routes (always allowed):
  /status, /backends, /v1/models, /engines, /sysinfo, /metrics, /health, /docs, /openapi.json
"""

import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from router.config import AuthConfig

logger = logging.getLogger("llm-router.auth")

# Routes that never require authentication
PUBLIC_ROUTES = {
    "/status", "/backends", "/engines", "/sysinfo",
    "/metrics", "/metrics/export", "/metrics/prometheus",
    "/health", "/docs", "/openapi.json", "/redoc",
}

# Route prefixes that require admin scope
ADMIN_PREFIXES = ("/start/", "/stop/", "/rescan", "/retune/", "/reload-config", "/restart/")

# Route prefixes that require inference scope
INFERENCE_PREFIXES = ("/v1/", "/anthropic/", "/gemini/")


class AuthMiddleware(BaseHTTPMiddleware):
    """API key validation middleware. Checks Authorization header or x-api-key."""

    def __init__(self, app, auth_config: "AuthConfig"):
        super().__init__(app)
        self.enabled = auth_config.enabled
        # Build lookup: key_string → {name, scope}
        self.keys: dict[str, dict] = {}
        for entry in auth_config.api_keys:
            self.keys[entry["key"]] = {
                "name": entry.get("name", "unnamed"),
                "scope": entry.get("scope", "all"),
            }

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        path = request.url.path

        # Public routes skip auth
        if path in PUBLIC_ROUTES:
            return await call_next(request)
        # GET /v1/models is public (for discovery)
        if path.startswith("/v1/models") and request.method == "GET":
            return await call_next(request)

        # Extract API key from headers
        api_key = self._extract_key(request)
        if not api_key or api_key not in self.keys:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid or missing API key",
                    "type": "authentication_error",
                },
            )

        key_info = self.keys[api_key]
        scope = key_info["scope"]

        # Check scope
        required_scope = self._required_scope(path)
        if required_scope and not self._scope_allows(scope, required_scope):
            logger.warning(
                f"Key '{key_info['name']}' (scope={scope}) "
                f"denied access to {path} (requires {required_scope})"
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": f"API key scope '{scope}' insufficient for this endpoint",
                    "type": "permission_error",
                    "required_scope": required_scope,
                },
            )

        # Attach key info to request state for audit logging
        request.state.api_key_name = key_info["name"]
        request.state.api_key_scope = scope
        return await call_next(request)

    @staticmethod
    def _extract_key(request: Request) -> str | None:
        """Extract API key from Authorization header or x-api-key header."""
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        return request.headers.get("x-api-key") or None

    @staticmethod
    def _required_scope(path: str) -> str | None:
        for prefix in ADMIN_PREFIXES:
            if path.startswith(prefix) or path == prefix.rstrip("/"):
                return "admin"
        for prefix in INFERENCE_PREFIXES:
            if path.startswith(prefix):
                return "inference"
        return None

    @staticmethod
    def _scope_allows(has_scope: str, needs_scope: str) -> bool:
        if has_scope == "all":
            return True
        return has_scope == needs_scope
