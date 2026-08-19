"""Production middleware: API key auth, request ID propagation, Prometheus metrics."""

from __future__ import annotations

import logging
import secrets
import time
import uuid

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "sentinel_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "sentinel_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)


# ── API Key Authentication ────────────────────────────────────────────────────

# Paths that bypass auth checks (public endpoints)
_PUBLIC_PATHS = frozenset({"/health", "/metrics", "/docs", "/openapi.json", "/redoc"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce X-API-Key and resolve the calling tenant.

    Enforcement is governed entirely by whether SENTINEL_API_KEY is set —
    identical to pre-tenancy behavior, so existing single-tenant deployments
    are unaffected. Two credential forms are accepted:

    1. The legacy SENTINEL_API_KEY string (fast path, zero DB access) —
       resolves to the default tenant.
    2. Any active per-tenant key issued via `sentinel tenant create-key`
       (hashed lookup against the api_keys table) — resolves to that tenant.
       This form is honored even when SENTINEL_API_KEY is unset, so a
       deployment can run purely on tenant-issued keys.

    When SENTINEL_API_KEY is unset and no key (or an unrecognized key) is
    presented, the request proceeds unauthenticated and resolves to the
    default tenant — matching today's dev-mode behavior exactly.
    """

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        auth_required = bool(self._api_key)

        if auth_required and secrets.compare_digest(provided, self._api_key):
            self._set_default_tenant(request)
            return await call_next(request)

        if provided:
            tenant = await self._lookup_tenant(request, provided)
            if tenant is not None:
                request.state.tenant_id = tenant["tenant_id"]
                request.state.tenant_slug = tenant["tenant_slug"]
                return await call_next(request)

        if not auth_required:
            self._set_default_tenant(request)
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"error": {"type": "unauthorized", "message": "Invalid or missing X-API-Key."}},
        )

    async def _lookup_tenant(self, request: Request, provided: str) -> dict | None:
        db_pool = getattr(request.app.state, "db_pool", None)
        if db_pool is None:
            return None
        from sentinel.tenancy.keys import hash_key  # noqa: PLC0415
        from sentinel.tenancy.queries import get_tenant_by_key_hash  # noqa: PLC0415

        try:
            return await get_tenant_by_key_hash(db_pool, hash_key(provided))
        except Exception:
            logger.exception("tenant key lookup failed")
            return None

    def _set_default_tenant(self, request: Request) -> None:
        default = getattr(request.app.state, "default_tenant", None)
        if default is not None:
            request.state.tenant_id = default["tenant_id"]
            request.state.tenant_slug = default["tenant_slug"]


# ── Request ID ───────────────────────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-ID into every request and response.

    request.state.request_id is always a valid UUID string — a client-
    supplied header is only honored when it parses as one, so downstream
    code (the proxy handler's DB primary key, trace span attributes) can
    rely on it without a separate validation step. This is what lets a
    single ID correlate the HTTP response, the `requests` row, and the
    distributed trace for one call.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        client_supplied = request.headers.get("X-Request-ID")
        request_id = _coerce_uuid(client_supplied) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def _coerce_uuid(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


# ── Prometheus Instrumentation ────────────────────────────────────────────────


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        method = request.method
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        status = str(response.status_code)
        REQUEST_COUNT.labels(method=method, path=path, status_code=status).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
        return response
