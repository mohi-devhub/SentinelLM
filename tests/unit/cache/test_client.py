"""Unit tests for sentinel.cache.client.

All tests use a fake redis stub — no real Redis connection needed.
"""

from __future__ import annotations

import pytest

from sentinel.cache.client import cache_key, get_cached_results, set_cached_results

_TENANT = "tenant-abc"

# ── Helpers ──────────────────────────────────────────────────────────────────


class FakeRedis:
    """Minimal in-memory Redis stub for testing hset/hgetall/expire."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}
        self._ttls: dict[str, int] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._store.setdefault(key, {}).update(mapping)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        data = self._store.get(key, {})
        return {k.encode(): v.encode() for k, v in data.items()}

    async def expire(self, key: str, seconds: int) -> None:
        self._ttls[key] = seconds


# ── cache_key ────────────────────────────────────────────────────────────────


def test_cache_key_has_prefix():
    key = cache_key("hello", "1", _TENANT)
    assert key.startswith(f"sentinel:{_TENANT}:scores:")


def test_cache_key_is_deterministic():
    assert cache_key("hello world", "1", _TENANT) == cache_key("hello world", "1", _TENANT)


def test_cache_key_differs_by_input():
    assert cache_key("input A", "1", _TENANT) != cache_key("input B", "1", _TENANT)


def test_cache_key_differs_by_config_version():
    assert cache_key("same input", "1", _TENANT) != cache_key("same input", "2", _TENANT)


def test_cache_key_differs_by_tenant():
    assert cache_key("same input", "1", "tenant-a") != cache_key("same input", "1", "tenant-b")


def test_cache_key_length():
    prefix = f"sentinel:{_TENANT}:scores:"
    key = cache_key("test", "1", _TENANT)
    assert len(key) == len(prefix) + 64


# ── get_cached_results — miss ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    redis = FakeRedis()
    result = await get_cached_results(redis, f"sentinel:{_TENANT}:scores:nonexistent")
    assert result is None


# ── set / get round-trip ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_round_trip_score_and_metadata():
    redis = FakeRedis()
    key = cache_key("some input", "1", _TENANT)
    results = {
        "pii": {"score": 0.12, "metadata": {"entities": [], "action": "redact"}},
        "prompt_injection": {"score": 0.55, "metadata": None},
    }

    await set_cached_results(redis, key, results, ttl_seconds=3600)
    result = await get_cached_results(redis, key)

    assert result == results


@pytest.mark.asyncio
async def test_round_trip_none_score():
    """A score of None (evaluator skipped / errored) survives the round-trip."""
    redis = FakeRedis()
    key = cache_key("input", "1", _TENANT)
    results = {
        "pii": {"score": None, "metadata": None},
        "prompt_injection": {"score": 0.3, "metadata": None},
    }

    await set_cached_results(redis, key, results, ttl_seconds=60)
    result = await get_cached_results(redis, key)

    assert result is not None
    assert result["pii"]["score"] is None
    assert result["prompt_injection"]["score"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_metadata_round_trips_for_pii_redaction():
    """PII's redacted_text must survive the cache — the proxy relies on it."""
    redis = FakeRedis()
    key = cache_key("call me at 555-1234", "1", _TENANT)
    results = {
        "pii": {
            "score": 0.9,
            "metadata": {
                "entities": [{"type": "PHONE_NUMBER", "score": 0.9, "start": 11, "end": 19}],
                "action": "redact",
                "redacted_text": "call me at <PHONE_NUMBER>",
            },
        }
    }

    await set_cached_results(redis, key, results)
    cached = await get_cached_results(redis, key)

    assert cached is not None
    assert cached["pii"]["metadata"]["redacted_text"] == "call me at <PHONE_NUMBER>"
    assert cached["pii"]["metadata"]["action"] == "redact"


@pytest.mark.asyncio
async def test_ttl_is_set_correctly():
    redis = FakeRedis()
    key = cache_key("input", "1", _TENANT)
    await set_cached_results(
        redis, key, {"pii": {"score": 0.1, "metadata": None}}, ttl_seconds=7200
    )
    assert redis._ttls[key] == 7200


@pytest.mark.asyncio
async def test_set_empty_results_does_nothing():
    """Calling set_cached_results with an empty dict should be a no-op."""
    redis = FakeRedis()
    key = cache_key("input", "1", _TENANT)
    await set_cached_results(redis, key, {})
    result = await get_cached_results(redis, key)
    assert result is None


@pytest.mark.asyncio
async def test_default_ttl_is_one_hour():
    redis = FakeRedis()
    key = cache_key("input", "1", _TENANT)
    await set_cached_results(redis, key, {"pii": {"score": 0.0, "metadata": None}})
    assert redis._ttls[key] == 3600
