"""GET /v1/sentinel/eval + GET /v1/sentinel/eval/{id} — eval run history."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from sentinel.storage.queries.eval_runs import (
    get_eval_run_by_id,
    list_eval_runs,
)

router = APIRouter()


@router.get("/v1/sentinel/eval")
async def list_runs(request: Request) -> JSONResponse:
    pool = request.app.state.db_pool
    runs = await list_eval_runs(pool)
    return JSONResponse(
        content=[
            {
                "id": str(r.id),
                "label": r.label,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "status": r.status,
                "record_count": r.record_count,
                "dataset_path": r.dataset_path,
            }
            for r in runs
        ]
    )


@router.get("/v1/sentinel/eval/{run_id}")
async def run_detail(request: Request, run_id: UUID) -> JSONResponse:
    pool = request.app.state.db_pool
    run = await get_eval_run_by_id(pool, run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"error": "eval run not found"})
    return JSONResponse(
        content={
            "id": str(run.id),
            "label": run.label,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "status": run.status,
            "record_count": run.record_count,
            "dataset_path": run.dataset_path,
            "scorecard": run.summary_json,
            "regression": run.regression_json,
        }
    )
