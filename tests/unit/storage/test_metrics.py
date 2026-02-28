"""Unit tests for storage.queries.metrics.

All tests use asyncpg MagicMocks — no real database required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from sentinel.storage.queries.metrics import (
    _BUCKET_TRUNC,
    _WINDOW_INTERVAL,
    _round,
    get_aggregate_metrics,
    get_summary_metrics,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_pool():
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


def _fake_bucket_row(**overrides) -> MagicMock:
    now = datetime(2026, 2, 28, 12, 0, 0, tzinfo=UTC)
    defaults = {
        "bucket_start": now,
        "request_count": 100,
        "blocked_count": 5,
        "avg_toxicity": 0.12,
        "avg_relevance": 0.85,
        "avg_hallucination": 0.08,
        "flag_rate_pii": 0.02,
        "flag_rate_prompt_injection": 0.01,
        "flag_rate_topic_guardrail": 0.03,
        "flag_rate_toxicity": 0.05,
        "flag_rate_relevance": 0.04,
        "flag_rate_hallucination": 0.02,
        "flag_rate_faithfulness": 0.01,
        "p95_latency_total": 450,
    }
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__ = lambda self, key: defaults[key]
    # Make bucket_start.isoformat() work
    row.__getitem__("bucket_start")
    return row


# ── _round ────────────────────────────────────────────────────────────────────


def test_round_float():
    assert _round(0.123456789) == pytest.approx(0.1235)


def test_round_none():
    assert _round(None) is None


def test_round_integer():
    assert _round(1) == pytest.approx(1.0)


# ── Whitelist maps ────────────────────────────────────────────────────────────


def test_window_interval_keys():
    assert "1h" in _WINDOW_INTERVAL
    assert "24h" in _WINDOW_INTERVAL
    assert "7d" in _WINDOW_INTERVAL


def test_bucket_trunc_keys():
    assert "5m" in _BUCKET_TRUNC
    assert "1h" in _BUCKET_TRUNC
    assert "1d" in _BUCKET_TRUNC


# ── get_aggregate_metrics ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_aggregate_metrics_empty():
    pool, conn = _fake_pool()
    conn.fetch = AsyncMock(return_value=[])

    result = await get_aggregate_metrics(pool)

    assert result == []
    conn.fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_aggregate_metrics_returns_list():
    pool, conn = _fake_pool()
    row = _fake_bucket_row()
    conn.fetch = AsyncMock(return_value=[row])

    result = await get_aggregate_metrics(pool, window="24h", bucket_size="1h")

    assert len(result) == 1
    entry = result[0]
    assert "bucket_start" in entry
    assert "request_count" in entry
    assert "blocked_count" in entry
    assert "avg_toxicity" in entry
    assert "flag_rate_pii" in entry
    assert entry["request_count"] == 100
    assert entry["blocked_count"] == 5
    assert entry["p95_latency_total"] == 450


@pytest.mark.asyncio
async def test_get_aggregate_metrics_null_scores():
    """avg_* fields can be None (no requests in bucket); _round handles None."""
    pool, conn = _fake_pool()
    row = _fake_bucket_row(avg_toxicity=None, avg_relevance=None, avg_hallucination=None)
    conn.fetch = AsyncMock(return_value=[row])

    result = await get_aggregate_metrics(pool)

    assert result[0]["avg_toxicity"] is None
    assert result[0]["avg_relevance"] is None


@pytest.mark.asyncio
async def test_get_aggregate_metrics_unknown_window_defaults():
    """An unknown window/bucket_size falls back to defaults (24 hours / hour)."""
    pool, conn = _fake_pool()
    conn.fetch = AsyncMock(return_value=[])

    await get_aggregate_metrics(pool, window="INVALID", bucket_size="INVALID")

    # Should not raise; fetch should still be called
    conn.fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_aggregate_metrics_multiple_windows():
    for window in _WINDOW_INTERVAL:
        pool, conn = _fake_pool()
        conn.fetch = AsyncMock(return_value=[])
        await get_aggregate_metrics(pool, window=window)
        conn.fetch.assert_awaited_once()


# ── get_summary_metrics ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_summary_metrics_returns_dict():
    pool, conn = _fake_pool()

    summary_row = MagicMock()
    summary_row.__getitem__ = lambda self, key: {
        "total_requests_24h": 500,
        "blocked_requests_24h": 20,
        "block_rate_24h": 0.04,
        "avg_latency_24h_ms": 320,
        "unreviewed_flags": 7,
    }[key]

    top_flag_row = MagicMock()
    top_flag_row.__getitem__ = lambda self, key: {"unnested_flag": "toxicity"}[key]

    conn.fetchrow = AsyncMock(side_effect=[summary_row, top_flag_row])

    result = await get_summary_metrics(pool)

    assert result["total_requests_24h"] == 500
    assert result["blocked_requests_24h"] == 20
    assert result["block_rate_24h"] == pytest.approx(0.04)
    assert result["avg_latency_24h_ms"] == 320
    assert result["top_flag_reason"] == "toxicity"
    assert result["unreviewed_flags"] == 7


@pytest.mark.asyncio
async def test_get_summary_metrics_no_top_flag():
    """When no flags exist, top_flag_reason is None."""
    pool, conn = _fake_pool()

    summary_row = MagicMock()
    summary_row.__getitem__ = lambda self, key: {
        "total_requests_24h": 0,
        "blocked_requests_24h": 0,
        "block_rate_24h": 0.0,
        "avg_latency_24h_ms": 0,
        "unreviewed_flags": 0,
    }[key]

    conn.fetchrow = AsyncMock(side_effect=[summary_row, None])

    result = await get_summary_metrics(pool)

    assert result["top_flag_reason"] is None
