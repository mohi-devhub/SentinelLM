"""Unit tests for sentinel.api.rate_limit.

Uses lightweight fake Request/Redis stubs — no real Redis or ASGI app needed,
matching the style already used in tests/unit/cache/test_client.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sentinel.api.rate_limit import enforce_rate_limit

# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeRedis:
    """Minimal in-memory Redis stub for incr/expire/ttl."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._ttls: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        self._ttls[key] = seconds

    async def ttl(self, key: str) -> int:
        return self._ttls.get(key, -1)


def _make_request(
    config: dict, redis: FakeRedis, tenant_id: str | None = None, ip: str = "1.2.3.4"
):
    app = SimpleNamespace(state=SimpleNamespace(config=config, redis=redis))
    client = SimpleNamespace(host=ip)
    return SimpleNamespace(app=app, state=SimpleNamespace(tenant_id=tenant_id), client=client)


ENABLED_CONFIG = {"rate_limit": {"enabled": True, "requests_per_minute": 3}}


# ── Disabled by default ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_rate_limit_section_is_noop():
    """No `rate_limit` key in config at all — must not block."""
    redis = FakeRedis()
    request = _make_request({}, redis)
    for _ in range(10):
        await enforce_rate_limit(request)  # never raises


@pytest.mark.asyncio
async def test_explicitly_disabled_is_noop():
    redis = FakeRedis()
    config = {"rate_limit": {"enabled": False, "requests_per_minute": 1}}
    request = _make_request(config, redis)
    for _ in range(10):
        await enforce_rate_limit(request)  # never raises


# ── Enforcement ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_allows_requests_up_to_the_limit():
    redis = FakeRedis()
    request = _make_request(ENABLED_CONFIG, redis, tenant_id="tenant-a")
    for _ in range(3):  # limit is 3
        await enforce_rate_limit(request)  # no raise


@pytest.mark.asyncio
async def test_blocks_the_request_that_exceeds_the_limit():
    redis = FakeRedis()
    request = _make_request(ENABLED_CONFIG, redis, tenant_id="tenant-a")
    for _ in range(3):
        await enforce_rate_limit(request)

    with pytest.raises(HTTPException) as exc_info:
        await enforce_rate_limit(request)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"]["type"] == "rate_limit_exceeded"


@pytest.mark.asyncio
async def test_retry_after_header_is_positive_integer_string():
    redis = FakeRedis()
    request = _make_request(ENABLED_CONFIG, redis, tenant_id="tenant-a")
    for _ in range(3):
        await enforce_rate_limit(request)

    with pytest.raises(HTTPException) as exc_info:
        await enforce_rate_limit(request)

    retry_after = exc_info.value.headers["Retry-After"]
    assert int(retry_after) >= 1
    assert exc_info.value.detail["error"]["retry_after_seconds"] == int(retry_after)


# ── Keying ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_different_tenants_get_independent_quotas():
    redis = FakeRedis()
    req_a = _make_request(ENABLED_CONFIG, redis, tenant_id="tenant-a")
    req_b = _make_request(ENABLED_CONFIG, redis, tenant_id="tenant-b")

    for _ in range(3):
        await enforce_rate_limit(req_a)

    # tenant-a is now exhausted, but tenant-b has its own independent counter
    await enforce_rate_limit(req_b)  # no raise
    with pytest.raises(HTTPException):
        await enforce_rate_limit(req_a)


@pytest.mark.asyncio
async def test_falls_back_to_client_ip_when_no_tenant_resolved():
    redis = FakeRedis()
    req_ip1 = _make_request(ENABLED_CONFIG, redis, tenant_id=None, ip="10.0.0.1")
    req_ip2 = _make_request(ENABLED_CONFIG, redis, tenant_id=None, ip="10.0.0.2")

    for _ in range(3):
        await enforce_rate_limit(req_ip1)

    await enforce_rate_limit(req_ip2)  # different IP, independent quota
    with pytest.raises(HTTPException):
        await enforce_rate_limit(req_ip1)


@pytest.mark.asyncio
async def test_default_limit_is_60_per_minute_when_unset():
    redis = FakeRedis()
    config = {"rate_limit": {"enabled": True}}  # requests_per_minute omitted
    request = _make_request(config, redis, tenant_id="tenant-a")
    for _ in range(60):
        await enforce_rate_limit(request)
    with pytest.raises(HTTPException):
        await enforce_rate_limit(request)
