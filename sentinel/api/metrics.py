"""GET /v1/sentinel/metrics/* — aggregate metrics and summary stats."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from sentinel.storage.queries.metrics import get_aggregate_metrics, get_summary_metrics

router = APIRouter()


@router.get("/v1/sentinel/metrics/aggregate")
async def aggregate(
    request: Request,
    window: str = Query("24h", enum=["1h", "6h", "24h", "7d", "30d"]),
    bucket_size: str = Query("1h", enum=["5m", "15m", "1h", "6h", "1d"]),
) -> JSONResponse:
    pool = request.app.state.db_pool
    buckets = await get_aggregate_metrics(pool, window=window, bucket_size=bucket_size)
    return JSONResponse(content={"window": window, "bucket_size": bucket_size, "buckets": buckets})


@router.get("/v1/sentinel/metrics/summary")
async def summary(request: Request) -> JSONResponse:
    pool = request.app.state.db_pool
    data = await get_summary_metrics(pool)
    return JSONResponse(content=data)
