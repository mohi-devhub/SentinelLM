"""GET /v1/sentinel/scores — paginated score history and single-record detail."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from sentinel.storage.queries.requests import get_request_by_id, get_scores

router = APIRouter()


@router.get("/v1/sentinel/scores")
async def scores(
    request: Request,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    flagged_only: bool = Query(False),
    evaluator: Optional[str] = Query(None, enum=[
        "pii", "prompt_injection", "topic_guardrail",
        "toxicity", "relevance", "hallucination", "faithfulness",
    ]),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
) -> JSONResponse:
    pool = request.app.state.db_pool
    items, total = await get_scores(
        pool,
        page=page,
        limit=limit,
        flagged_only=flagged_only,
        evaluator=evaluator,
        start_date=start_date,
        end_date=end_date,
    )
    return JSONResponse(content={"total": total, "page": page, "limit": limit, "items": items})


@router.get("/v1/sentinel/scores/{request_id}")
async def score_detail(request: Request, request_id: UUID) -> JSONResponse:
    pool = request.app.state.db_pool
    record = await get_request_by_id(pool, request_id)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return JSONResponse(content=record)
