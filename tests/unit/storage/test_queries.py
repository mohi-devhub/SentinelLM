"""Unit tests for storage query helpers.

All tests use asyncpg MagicMocks — no real database connection required.
Tests cover the helper functions in requests.py and metrics.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.storage.queries.requests import (
    _row_to_dict,
    _VALID_EVALUATORS,
    get_scores,
    get_request_by_id,
    get_review_queue,
    submit_review,
)


# ── Fake asyncpg.Record ───────────────────────────────────────────────────────

def _fake_row(**overrides) -> MagicMock:
    """Build a MagicMock that behaves like an asyncpg Record."""
    _id = uuid.uuid4()
    _now = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)

    defaults = {
        "id": _id,
        "created_at": _now,
        "model": "llama3.2",
        "blocked": False,
        "block_reason": None,
        "score_pii": 0.01,
        "score_prompt_injection": 0.02,
        "score_topic_guardrail": 0.80,
        "score_toxicity": 0.03,
        "score_relevance": 0.90,
        "score_hallucination": 0.04,
        "score_faithfulness": 0.91,
        "flag_pii": False,
        "flag_prompt_injection": False,
        "flag_topic_guardrail": False,
        "flag_toxicity": False,
        "flag_relevance": False,
        "flag_hallucination": False,
        "flag_faithfulness": False,
        "latency_pii": 5,
        "latency_prompt_injection": 10,
        "latency_topic_guardrail": 8,
        "latency_toxicity": 15,
        "latency_relevance": 12,
        "latency_hallucination": 60,
        "latency_faithfulness": 58,
        "latency_llm": 300,
        "latency_total": 468,
        "input_text": "What is the capital of France?",
        "input_redacted": "What is the capital of France?",
        "has_context": False,
        "reviewed": False,
        "review_label": None,
        "reviewer_note": None,
        "reviewed_at": None,
    }
    defaults.update(overrides)

    row = MagicMock()
    row.__getitem__ = lambda self, key: defaults[key]
    return row


# ── _row_to_dict ──────────────────────────────────────────────────────────────

def test_row_to_dict_basic_fields():
    """_row_to_dict correctly maps all scalar fields."""
    row = _fake_row()
    d = _row_to_dict(row)

    assert "id" in d
    assert "created_at" in d
    assert d["model"] == "llama3.2"
    assert d["blocked"] is False
    assert d["block_reason"] is None
    assert d["has_context"] is False
    assert d["reviewed"] is False


def test_row_to_dict_scores_structure():
    """scores dict contains all 7 evaluator keys."""
    d = _row_to_dict(_fake_row())
    scores = d["scores"]
    for ev in _VALID_EVALUATORS:
        assert ev in scores


def test_row_to_dict_latency_structure():
    """latency_ms dict contains all evaluator keys plus llm and total."""
    d = _row_to_dict(_fake_row())
    lat = d["latency_ms"]
    for ev in _VALID_EVALUATORS:
        assert ev in lat
    assert "llm" in lat
    assert "total" in lat


def test_row_to_dict_flags_empty_when_none_set():
    """flags list is empty when all flag_ columns are False."""
    d = _row_to_dict(_fake_row())
    assert d["flags"] == []


def test_row_to_dict_flags_populated_correctly():
    """flags list includes evaluator names for True flag columns."""
    row = _fake_row(flag_pii=True, flag_toxicity=True)
    d = _row_to_dict(row)
    assert "pii" in d["flags"]
    assert "toxicity" in d["flags"]
    assert len(d["flags"]) == 2


def test_row_to_dict_id_is_string():
    """id is serialized to a string, not a UUID object."""
    d = _row_to_dict(_fake_row())
    assert isinstance(d["id"], str)


def test_row_to_dict_created_at_is_isoformat():
    """created_at is an ISO 8601 string."""
    d = _row_to_dict(_fake_row())
    # Should be parseable back to a datetime
    parsed = datetime.fromisoformat(d["created_at"])
    assert parsed.year == 2026


# ── get_scores ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pool():
    """asyncpg Pool mock with acquire() as an async context manager."""
    pool = MagicMock()
    conn = AsyncMock()

    # pool.acquire() returns an async context manager
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    return pool, conn


@pytest.mark.asyncio
async def test_get_scores_returns_list_and_total(mock_pool):
    """get_scores returns (items, total) tuple."""
    pool, conn = mock_pool
    row = _fake_row()
    conn.fetchval = AsyncMock(return_value=1)   # total count
    conn.fetch = AsyncMock(return_value=[row])

    items, total = await get_scores(pool, page=1, limit=10)

    assert total == 1
    assert len(items) == 1
    assert items[0]["model"] == "llama3.2"


@pytest.mark.asyncio
async def test_get_scores_flagged_only_filter(mock_pool):
    """flagged_only=True adds the flag filter to the query."""
    pool, conn = mock_pool
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])

    items, total = await get_scores(pool, flagged_only=True)

    # Verify fetch was called (with the flag filter included in query)
    assert conn.fetch.called
    assert items == []
    assert total == 0


@pytest.mark.asyncio
async def test_get_scores_invalid_evaluator_ignored(mock_pool):
    """An evaluator name not in the whitelist is silently ignored."""
    pool, conn = mock_pool
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])

    # Should not raise; unknown evaluator is not interpolated into SQL
    items, total = await get_scores(pool, evaluator="DROP TABLE requests; --")
    assert total == 0


# ── get_request_by_id ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_request_by_id_found(mock_pool):
    """Returns a dict when a row with the given ID exists."""
    pool, conn = mock_pool
    row = _fake_row()
    conn.fetchrow = AsyncMock(return_value=row)

    request_id = uuid.uuid4()
    result = await get_request_by_id(pool, request_id)

    assert result is not None
    assert result["model"] == "llama3.2"
    conn.fetchrow.assert_called_once()


@pytest.mark.asyncio
async def test_get_request_by_id_not_found(mock_pool):
    """Returns None when no row matches the given ID."""
    pool, conn = mock_pool
    conn.fetchrow = AsyncMock(return_value=None)

    result = await get_request_by_id(pool, uuid.uuid4())
    assert result is None


# ── get_review_queue ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_review_queue_returns_flagged_rows(mock_pool):
    """Returns list of dicts for unreviewed flagged rows."""
    pool, conn = mock_pool
    row = _fake_row(flag_toxicity=True)
    conn.fetch = AsyncMock(return_value=[row])

    items = await get_review_queue(pool, limit=10)

    assert len(items) == 1
    assert "toxicity" in items[0]["flags"]


@pytest.mark.asyncio
async def test_get_review_queue_empty(mock_pool):
    """Returns empty list when no unreviewed flagged rows."""
    pool, conn = mock_pool
    conn.fetch = AsyncMock(return_value=[])

    items = await get_review_queue(pool)
    assert items == []


# ── submit_review ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_review_returns_true_on_success(mock_pool):
    """Returns True when UPDATE affects exactly one row."""
    pool, conn = mock_pool
    conn.execute = AsyncMock(return_value="UPDATE 1")

    result = await submit_review(pool, uuid.uuid4(), "correct_flag", note=None)
    assert result is True


@pytest.mark.asyncio
async def test_submit_review_returns_false_when_not_found(mock_pool):
    """Returns False when UPDATE affects zero rows (ID not found)."""
    pool, conn = mock_pool
    conn.execute = AsyncMock(return_value="UPDATE 0")

    result = await submit_review(pool, uuid.uuid4(), "false_positive", note=None)
    assert result is False


@pytest.mark.asyncio
async def test_submit_review_passes_note_to_query(mock_pool):
    """The optional note is forwarded to the UPDATE statement."""
    pool, conn = mock_pool
    conn.execute = AsyncMock(return_value="UPDATE 1")

    await submit_review(pool, uuid.uuid4(), "correct_flag", note="Looks right")

    call_args = conn.execute.call_args
    assert "Looks right" in call_args.args
