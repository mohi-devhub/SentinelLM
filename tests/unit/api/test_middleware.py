"""Unit tests for sentinel.api.middleware.APIKeyMiddleware's tenant resolution.

Uses lightweight fake Request/db-pool stubs — no real ASGI app or Postgres
needed, matching the style already used in tests/unit/api/test_rate_limit.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sentinel.api.middleware import APIKeyMiddleware
from sentinel.tenancy.keys import hash_key

DEFAULT_TENANT = {"tenant_id": "default-id", "tenant_slug": "default"}
ACME_TENANT = {"tenant_id": "acme-id", "tenant_slug": "acme"}


class FakeDbPool:
    """Stands in for asyncpg.Pool — only get_tenant_by_key_hash ever touches it."""


def _make_request(path: str, api_key: str | None, db_pool: object | None = FakeDbPool()):
    headers = {"X-API-Key": api_key} if api_key else {}
    app_state = SimpleNamespace(db_pool=db_pool, default_tenant=DEFAULT_TENANT)
    app = SimpleNamespace(state=app_state)
    url = SimpleNamespace(path=path)
    return SimpleNamespace(url=url, headers=headers, state=SimpleNamespace(), app=app)


async def _call_next(request):
    return SimpleNamespace(status_code=200)


def _patch_tenant_lookup(monkeypatch, result: dict | None):
    async def fake_lookup(pool, key_hash):
        return result

    monkeypatch.setattr("sentinel.tenancy.queries.get_tenant_by_key_hash", fake_lookup)


# ── Public paths bypass auth entirely ───────────────────────────────────────


@pytest.mark.asyncio
async def test_public_path_bypasses_auth():
    mw = APIKeyMiddleware(app=None, api_key="secret")
    request = _make_request("/health", api_key=None)
    response = await mw.dispatch(request, _call_next)
    assert response.status_code == 200
    assert not hasattr(request.state, "tenant_id")


# ── Legacy SENTINEL_API_KEY fast path ───────────────────────────────────────


@pytest.mark.asyncio
async def test_legacy_key_match_resolves_default_tenant():
    mw = APIKeyMiddleware(app=None, api_key="secret")
    request = _make_request("/v1/chat/completions", api_key="secret")
    await mw.dispatch(request, _call_next)
    assert request.state.tenant_id == DEFAULT_TENANT["tenant_id"]
    assert request.state.tenant_slug == DEFAULT_TENANT["tenant_slug"]


@pytest.mark.asyncio
async def test_wrong_key_returns_401_when_auth_required(monkeypatch):
    _patch_tenant_lookup(monkeypatch, None)
    mw = APIKeyMiddleware(app=None, api_key="secret")
    request = _make_request("/v1/chat/completions", api_key="wrong")
    response = await mw.dispatch(request, _call_next)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_missing_key_returns_401_when_auth_required():
    mw = APIKeyMiddleware(app=None, api_key="secret")
    request = _make_request("/v1/chat/completions", api_key=None)
    response = await mw.dispatch(request, _call_next)
    assert response.status_code == 401


# ── Per-tenant DB-backed keys ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_tenant_key_resolves_that_tenant(monkeypatch):
    _patch_tenant_lookup(monkeypatch, ACME_TENANT)
    mw = APIKeyMiddleware(app=None, api_key="secret")
    request = _make_request("/v1/chat/completions", api_key="sk-sentinel-acme-key")
    response = await mw.dispatch(request, _call_next)
    assert response.status_code == 200
    assert request.state.tenant_id == ACME_TENANT["tenant_id"]
    assert request.state.tenant_slug == ACME_TENANT["tenant_slug"]


@pytest.mark.asyncio
async def test_tenant_key_honored_even_when_auth_not_required(monkeypatch):
    """A real per-tenant key is always honored, even with SENTINEL_API_KEY unset."""
    _patch_tenant_lookup(monkeypatch, ACME_TENANT)
    mw = APIKeyMiddleware(app=None, api_key="")
    request = _make_request("/v1/chat/completions", api_key="sk-sentinel-acme-key")
    await mw.dispatch(request, _call_next)
    assert request.state.tenant_id == ACME_TENANT["tenant_id"]


@pytest.mark.asyncio
async def test_lookup_failure_does_not_crash(monkeypatch):
    async def raising_lookup(pool, key_hash):
        raise ConnectionError("db down")

    monkeypatch.setattr("sentinel.tenancy.queries.get_tenant_by_key_hash", raising_lookup)
    mw = APIKeyMiddleware(app=None, api_key="secret")
    request = _make_request("/v1/chat/completions", api_key="sk-sentinel-whatever")
    response = await mw.dispatch(request, _call_next)
    assert response.status_code == 401


# ── Dev mode (no auth configured anywhere) ──────────────────────────────────


@pytest.mark.asyncio
async def test_no_key_configured_resolves_default_tenant_unauthenticated():
    mw = APIKeyMiddleware(app=None, api_key="")
    request = _make_request("/v1/chat/completions", api_key=None)
    response = await mw.dispatch(request, _call_next)
    assert response.status_code == 200
    assert request.state.tenant_id == DEFAULT_TENANT["tenant_id"]


@pytest.mark.asyncio
async def test_unrecognized_key_falls_back_to_default_tenant_when_auth_not_required(
    monkeypatch,
):
    _patch_tenant_lookup(monkeypatch, None)
    mw = APIKeyMiddleware(app=None, api_key="")
    request = _make_request("/v1/chat/completions", api_key="garbage")
    response = await mw.dispatch(request, _call_next)
    assert response.status_code == 200
    assert request.state.tenant_id == DEFAULT_TENANT["tenant_id"]


# ── Sanity: hash_key is what the middleware actually hashes ────────────────


def test_hash_key_is_stable_for_lookup():
    assert hash_key("sk-sentinel-acme-key") == hash_key("sk-sentinel-acme-key")
