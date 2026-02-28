"""GET /v1/sentinel/review + PATCH /v1/sentinel/review/{id} — human review queue."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sentinel.storage.queries.requests import get_review_queue, submit_review

router = APIRouter()

_VALID_LABELS = {"correct_flag", "false_positive", "false_negative"}


class ReviewSubmission(BaseModel):
    label: str
    note: Optional[str] = None


@router.get("/v1/sentinel/review")
async def review_queue(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
) -> JSONResponse:
    pool = request.app.state.db_pool
    items = await get_review_queue(pool, limit=limit)
    return JSONResponse(content=items)


@router.patch("/v1/sentinel/review/{request_id}")
async def submit_review_label(
    request: Request,
    request_id: UUID,
    body: ReviewSubmission,
) -> JSONResponse:
    if body.label not in _VALID_LABELS:
        return JSONResponse(
            status_code=422,
            content={"error": f"label must be one of {sorted(_VALID_LABELS)}"},
        )

    pool = request.app.state.db_pool
    updated = await submit_review(pool, request_id, body.label, body.note)
    if not updated:
        return JSONResponse(status_code=404, content={"error": "request not found"})
    return JSONResponse(content={"status": "ok", "request_id": str(request_id)})
