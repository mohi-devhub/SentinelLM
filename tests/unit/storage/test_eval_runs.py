"""Unit tests for storage.queries.eval_runs.

All tests use asyncpg MagicMocks — no real database required.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from sentinel.storage.models import EvalRunRecord
from sentinel.storage.queries.eval_runs import (
    _row_to_eval_run,
    complete_eval_run,
    fail_eval_run,
    get_eval_run_by_id,
    get_eval_run_by_label,
    insert_eval_result,
    insert_eval_run,
    list_eval_runs,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_pool():
    """Mock asyncpg Pool with acquire() as async context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


def _fake_eval_run_row(**overrides) -> MagicMock:
    """Build a MagicMock that looks like an asyncpg Record for eval_runs."""
    run_id = uuid.uuid4()
    now = datetime(2026, 2, 28, 12, 0, 0, tzinfo=UTC)
    defaults = {
        "id": run_id,
        "created_at": now,
        "completed_at": None,
        "label": "test_run",
        "dataset_path": "evals/test.jsonl",
        "baseline_run_id": None,
        "record_count": 10,
        "status": "complete",
        "summary_json": json.dumps({"pii": {"flag_rate": 0.1}}),
        "regression_json": None,
    }
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__ = lambda self, key: defaults[key]
    return row


# ── _row_to_eval_run ──────────────────────────────────────────────────────────


def test_row_to_eval_run_basic():
    row = _fake_eval_run_row()
    record = _row_to_eval_run(row)
    assert isinstance(record, EvalRunRecord)
    assert record.label == "test_run"
    assert record.status == "complete"
    assert record.record_count == 10


def test_row_to_eval_run_summary_json_parsed():
    """summary_json string is parsed into a dict."""
    row = _fake_eval_run_row(summary_json=json.dumps({"pii": {"flag_rate": 0.05}}))
    record = _row_to_eval_run(row)
    assert isinstance(record.summary_json, dict)
    assert record.summary_json["pii"]["flag_rate"] == pytest.approx(0.05)


def test_row_to_eval_run_summary_json_already_dict():
    """summary_json that is already a dict is not double-parsed."""
    row = _fake_eval_run_row(summary_json={"pii": {"flag_rate": 0.1}})
    record = _row_to_eval_run(row)
    assert record.summary_json["pii"]["flag_rate"] == pytest.approx(0.1)


def test_row_to_eval_run_null_summary():
    row = _fake_eval_run_row(summary_json=None)
    record = _row_to_eval_run(row)
    assert record.summary_json is None


def test_row_to_eval_run_regression_json_parsed():
    row = _fake_eval_run_row(regression_json=json.dumps({"pii": {"regressed": False}}))
    record = _row_to_eval_run(row)
    assert isinstance(record.regression_json, dict)


def test_row_to_eval_run_null_regression():
    row = _fake_eval_run_row(regression_json=None)
    record = _row_to_eval_run(row)
    assert record.regression_json is None


# ── insert_eval_run ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_eval_run_returns_record():
    pool, conn = _fake_pool()
    run_id = uuid.uuid4()
    now = datetime(2026, 2, 28, tzinfo=UTC)
    row = MagicMock()
    row.__getitem__ = lambda self, key: {"id": run_id, "created_at": now}[key]
    conn.fetchrow = AsyncMock(return_value=row)

    record = await insert_eval_run(pool, label="run1", dataset_path="data.jsonl")

    assert record.label == "run1"
    assert record.status == "running"
    assert record.id is not None


@pytest.mark.asyncio
async def test_insert_eval_run_with_baseline():
    pool, conn = _fake_pool()
    baseline_id = uuid.uuid4()
    run_id = uuid.uuid4()
    now = datetime(2026, 2, 28, tzinfo=UTC)
    row = MagicMock()
    row.__getitem__ = lambda self, key: {"id": run_id, "created_at": now}[key]
    conn.fetchrow = AsyncMock(return_value=row)

    record = await insert_eval_run(
        pool, label="run2", dataset_path="data.jsonl", baseline_run_id=baseline_id
    )

    assert record.baseline_run_id == baseline_id


# ── complete_eval_run ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_eval_run_executes_update():
    pool, conn = _fake_pool()
    conn.execute = AsyncMock()
    run_id = uuid.uuid4()

    await complete_eval_run(
        pool=pool,
        run_id=run_id,
        record_count=42,
        summary={"pii": {"flag_rate": 0.1}},
        regression=None,
    )

    conn.execute.assert_awaited_once()
    call_args = conn.execute.call_args.args
    assert run_id in call_args
    assert 42 in call_args


@pytest.mark.asyncio
async def test_complete_eval_run_with_regression():
    pool, conn = _fake_pool()
    conn.execute = AsyncMock()
    run_id = uuid.uuid4()
    regression = {"pii": {"regressed": False, "delta": 0.0}}

    await complete_eval_run(
        pool=pool,
        run_id=run_id,
        record_count=10,
        summary={},
        regression=regression,
    )

    conn.execute.assert_awaited_once()


# ── fail_eval_run ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_eval_run_executes_update():
    pool, conn = _fake_pool()
    conn.execute = AsyncMock()
    run_id = uuid.uuid4()

    await fail_eval_run(pool, run_id)

    conn.execute.assert_awaited_once()
    assert run_id in conn.execute.call_args.args


# ── insert_eval_result ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_eval_result_executes_insert():
    pool, conn = _fake_pool()
    conn.execute = AsyncMock()
    eval_run_id = uuid.uuid4()
    request_id = uuid.uuid4()

    await insert_eval_result(
        pool=pool,
        eval_run_id=eval_run_id,
        request_id=request_id,
        record_index=0,
        input_text="What is 2+2?",
        expected_output="4",
        actual_output="The answer is 4.",
        passed=True,
    )

    conn.execute.assert_awaited_once()
    args = conn.execute.call_args.args
    assert eval_run_id in args
    assert request_id in args


@pytest.mark.asyncio
async def test_insert_eval_result_null_request_id():
    pool, conn = _fake_pool()
    conn.execute = AsyncMock()

    await insert_eval_result(
        pool=pool,
        eval_run_id=uuid.uuid4(),
        request_id=None,
        record_index=1,
        input_text="test",
        expected_output=None,
        actual_output=None,
        passed=False,
    )

    conn.execute.assert_awaited_once()


# ── list_eval_runs ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_eval_runs_returns_records():
    pool, conn = _fake_pool()
    row = _fake_eval_run_row()
    conn.fetch = AsyncMock(return_value=[row, row])

    runs = await list_eval_runs(pool)

    assert len(runs) == 2
    assert all(isinstance(r, EvalRunRecord) for r in runs)


@pytest.mark.asyncio
async def test_list_eval_runs_empty():
    pool, conn = _fake_pool()
    conn.fetch = AsyncMock(return_value=[])

    runs = await list_eval_runs(pool)

    assert runs == []


# ── get_eval_run_by_label ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_eval_run_by_label_found():
    pool, conn = _fake_pool()
    row = _fake_eval_run_row(label="my_run")
    conn.fetchrow = AsyncMock(return_value=row)

    result = await get_eval_run_by_label(pool, "my_run")

    assert result is not None
    assert result.label == "my_run"


@pytest.mark.asyncio
async def test_get_eval_run_by_label_not_found():
    pool, conn = _fake_pool()
    conn.fetchrow = AsyncMock(return_value=None)

    result = await get_eval_run_by_label(pool, "nonexistent")

    assert result is None


# ── get_eval_run_by_id ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_eval_run_by_id_found():
    pool, conn = _fake_pool()
    run_id = uuid.uuid4()
    row = _fake_eval_run_row(id=run_id)
    conn.fetchrow = AsyncMock(return_value=row)

    result = await get_eval_run_by_id(pool, run_id)

    assert result is not None


@pytest.mark.asyncio
async def test_get_eval_run_by_id_not_found():
    pool, conn = _fake_pool()
    conn.fetchrow = AsyncMock(return_value=None)

    result = await get_eval_run_by_id(pool, uuid.uuid4())

    assert result is None
