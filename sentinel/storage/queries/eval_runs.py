"""DB queries for eval_runs and eval_results tables."""

from __future__ import annotations

import json
import uuid as _uuid
from uuid import UUID

import asyncpg

from sentinel.storage.models import EvalRunRecord

# ── eval_runs ────────────────────────────────────────────────────────────────

_INSERT_EVAL_RUN = """
INSERT INTO eval_runs (id, label, dataset_path, baseline_run_id, status)
VALUES ($1, $2, $3, $4, 'running')
RETURNING id, created_at
"""

_COMPLETE_EVAL_RUN = """
UPDATE eval_runs
SET status        = 'complete',
    completed_at  = NOW(),
    record_count  = $2,
    summary_json  = $3,
    regression_json = $4
WHERE id = $1
"""

_FAIL_EVAL_RUN = """
UPDATE eval_runs
SET status = 'failed', completed_at = NOW()
WHERE id = $1
"""

_LIST_EVAL_RUNS = """
SELECT id, created_at, completed_at, label, dataset_path,
       baseline_run_id, record_count, status, summary_json, regression_json
FROM eval_runs
ORDER BY created_at DESC
LIMIT 50
"""

_GET_EVAL_RUN_BY_LABEL = """
SELECT id, created_at, completed_at, label, dataset_path,
       baseline_run_id, record_count, status, summary_json, regression_json
FROM eval_runs
WHERE label = $1
"""

_GET_EVAL_RUN_BY_ID = """
SELECT id, created_at, completed_at, label, dataset_path,
       baseline_run_id, record_count, status, summary_json, regression_json
FROM eval_runs
WHERE id = $1
"""

# ── eval_results ─────────────────────────────────────────────────────────────

_INSERT_EVAL_RESULT = """
INSERT INTO eval_results
    (id, eval_run_id, request_id, record_index, input_text,
     expected_output, actual_output, passed)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""

_INSERT_EVAL_RESULT_WITH_SCORES = """
INSERT INTO eval_results
    (id, eval_run_id, request_id, record_index, input_text,
     expected_output, actual_output, passed, scores_json)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""

_COMPLETE_OFFLINE_EVAL_RUN = """
UPDATE eval_runs
SET status                       = 'complete',
    completed_at                 = NOW(),
    record_count                 = $2,
    summary_json                 = $3,
    regression_json              = $4,
    eval_mode                    = 'offline',
    statistical_regression_json  = $5
WHERE id = $1
"""

_GET_OFFLINE_RUN_SCORES = """
SELECT scores_json
FROM eval_results
WHERE eval_run_id = $1 AND scores_json IS NOT NULL
"""


# ── helpers ──────────────────────────────────────────────────────────────────


def _row_to_eval_run(row: asyncpg.Record) -> EvalRunRecord:
    summary = row["summary_json"]
    regression = row["regression_json"]
    return EvalRunRecord(
        id=row["id"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        label=row["label"],
        dataset_path=row["dataset_path"],
        baseline_run_id=row["baseline_run_id"],
        record_count=row["record_count"],
        status=row["status"],
        summary_json=json.loads(summary) if isinstance(summary, str) else summary,
        regression_json=json.loads(regression) if isinstance(regression, str) else regression,
    )


# ── public API ───────────────────────────────────────────────────────────────


async def insert_eval_run(
    pool: asyncpg.Pool,
    label: str,
    dataset_path: str,
    baseline_run_id: UUID | None = None,
) -> EvalRunRecord:
    """Create a new eval_run row in 'running' state and return it."""
    run_id = _uuid.uuid4()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(_INSERT_EVAL_RUN, run_id, label, dataset_path, baseline_run_id)
    return EvalRunRecord(
        id=row["id"],
        created_at=row["created_at"],
        label=label,
        dataset_path=dataset_path,
        baseline_run_id=baseline_run_id,
        status="running",
    )


async def complete_eval_run(
    pool: asyncpg.Pool,
    run_id: UUID,
    record_count: int,
    summary: dict,
    regression: dict | None,
) -> None:
    """Mark an eval run as complete and persist the scorecard blobs."""
    async with pool.acquire() as conn:
        await conn.execute(
            _COMPLETE_EVAL_RUN,
            run_id,
            record_count,
            json.dumps(summary),
            json.dumps(regression) if regression else None,
        )


async def fail_eval_run(pool: asyncpg.Pool, run_id: UUID) -> None:
    """Mark an eval run as failed."""
    async with pool.acquire() as conn:
        await conn.execute(_FAIL_EVAL_RUN, run_id)


async def insert_eval_result(
    pool: asyncpg.Pool,
    eval_run_id: UUID,
    request_id: UUID | None,
    record_index: int,
    input_text: str,
    expected_output: str | None,
    actual_output: str | None,
    passed: bool,
    scores_json: dict | None = None,
) -> None:
    """Persist a single eval result row.

    Pass scores_json for offline runs where request_id is None, so that
    per-record scores are stored for later statistical comparison.
    """
    result_id = _uuid.uuid4()
    async with pool.acquire() as conn:
        if scores_json is not None:
            await conn.execute(
                _INSERT_EVAL_RESULT_WITH_SCORES,
                result_id,
                eval_run_id,
                request_id,
                record_index,
                input_text,
                expected_output,
                actual_output,
                passed,
                json.dumps(scores_json),
            )
        else:
            await conn.execute(
                _INSERT_EVAL_RESULT,
                result_id,
                eval_run_id,
                request_id,
                record_index,
                input_text,
                expected_output,
                actual_output,
                passed,
            )


async def list_eval_runs(pool: asyncpg.Pool) -> list[EvalRunRecord]:
    """Return up to 50 most recent eval runs."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(_LIST_EVAL_RUNS)
    return [_row_to_eval_run(r) for r in rows]


async def get_eval_run_by_label(pool: asyncpg.Pool, label: str) -> EvalRunRecord | None:
    """Look up an eval run by its human-readable label."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(_GET_EVAL_RUN_BY_LABEL, label)
    return _row_to_eval_run(row) if row else None


async def get_eval_run_by_id(pool: asyncpg.Pool, run_id: UUID) -> EvalRunRecord | None:
    """Look up an eval run by UUID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(_GET_EVAL_RUN_BY_ID, run_id)
    return _row_to_eval_run(row) if row else None


async def complete_offline_eval_run(
    pool: asyncpg.Pool,
    run_id: UUID,
    record_count: int,
    summary: dict,
    regression: dict | None,
    statistical_regression: dict | None,
) -> None:
    """Mark an offline eval run as complete, storing statistical regression data."""
    async with pool.acquire() as conn:
        await conn.execute(
            _COMPLETE_OFFLINE_EVAL_RUN,
            run_id,
            record_count,
            json.dumps(summary),
            json.dumps(regression) if regression else None,
            json.dumps(statistical_regression) if statistical_regression else None,
        )


async def get_offline_run_scores(
    pool: asyncpg.Pool, run_id: UUID
) -> dict[str, list[float]]:
    """Fetch per-evaluator score lists for an offline eval run.

    Returns a dict of evaluator_name → list[float] (non-None scores only).
    Used to fetch baseline scores for statistical regression comparison.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(_GET_OFFLINE_RUN_SCORES, run_id)

    scores_by_ev: dict[str, list[float]] = {}
    for row in rows:
        scores_json = row["scores_json"]
        if isinstance(scores_json, str):
            scores_json = json.loads(scores_json)
        if not isinstance(scores_json, dict):
            continue
        for ev_name, score in scores_json.items():
            if score is not None:
                scores_by_ev.setdefault(ev_name, []).append(float(score))

    return scores_by_ev
