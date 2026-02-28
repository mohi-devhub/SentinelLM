"""Aggregate metrics queries for the dashboard charts and summary cards."""
from __future__ import annotations

from typing import Any

import asyncpg

# Whitelist maps — values are interpolated into SQL, so they MUST come from here.
_WINDOW_INTERVAL: dict[str, str] = {
    "1h": "1 hour",
    "6h": "6 hours",
    "24h": "24 hours",
    "7d": "7 days",
    "30d": "30 days",
}

# Maps bucket_size → PostgreSQL date_trunc precision
_BUCKET_TRUNC: dict[str, str] = {
    "5m": "minute",
    "15m": "minute",
    "1h": "hour",
    "6h": "hour",
    "1d": "day",
}

_FLAG_COLS = (
    "flag_pii OR flag_prompt_injection OR flag_topic_guardrail "
    "OR flag_toxicity OR flag_relevance OR flag_hallucination OR flag_faithfulness"
)


async def get_aggregate_metrics(
    pool: asyncpg.Pool,
    window: str = "24h",
    bucket_size: str = "1h",
) -> list[dict[str, Any]]:
    """Return time-bucketed metrics for the dashboard line/bar charts.

    Values for `window` and `bucket_size` are validated against whitelists
    before interpolation into the query string.
    """
    interval = _WINDOW_INTERVAL.get(window, "24 hours")
    trunc = _BUCKET_TRUNC.get(bucket_size, "hour")

    # Both `interval` and `trunc` come from whitelists — safe to interpolate.
    query = f"""
        SELECT
            date_trunc('{trunc}', created_at)                              AS bucket_start,
            COUNT(*)                                                        AS request_count,
            COUNT(*) FILTER (WHERE blocked)                                 AS blocked_count,
            AVG(score_toxicity)                                             AS avg_toxicity,
            AVG(score_relevance)                                            AS avg_relevance,
            AVG(score_hallucination)                                        AS avg_hallucination,
            COALESCE(
                COUNT(*) FILTER (WHERE flag_pii)::float / NULLIF(COUNT(*), 0), 0
            )                                                               AS flag_rate_pii,
            COALESCE(
                COUNT(*) FILTER (WHERE flag_prompt_injection)::float / NULLIF(COUNT(*), 0), 0
            ) AS flag_rate_prompt_injection,
            COALESCE(
                COUNT(*) FILTER (WHERE flag_topic_guardrail)::float / NULLIF(COUNT(*), 0), 0
            ) AS flag_rate_topic_guardrail,
            COALESCE(
                COUNT(*) FILTER (WHERE flag_toxicity)::float / NULLIF(COUNT(*), 0), 0
            ) AS flag_rate_toxicity,
            COALESCE(
                COUNT(*) FILTER (WHERE flag_relevance)::float / NULLIF(COUNT(*), 0), 0
            ) AS flag_rate_relevance,
            COALESCE(
                COUNT(*) FILTER (WHERE flag_hallucination)::float / NULLIF(COUNT(*), 0), 0
            ) AS flag_rate_hallucination,
            COALESCE(
                COUNT(*) FILTER (WHERE flag_faithfulness)::float / NULLIF(COUNT(*), 0), 0
            ) AS flag_rate_faithfulness,
            COALESCE(
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_total),
                0
            )::int                                                          AS p95_latency_total
        FROM requests
        WHERE created_at >= NOW() - INTERVAL '{interval}'
        GROUP BY bucket_start
        ORDER BY bucket_start ASC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    return [
        {
            "bucket_start": row["bucket_start"].isoformat(),
            "request_count": row["request_count"],
            "blocked_count": row["blocked_count"],
            "avg_toxicity": _round(row["avg_toxicity"]),
            "avg_relevance": _round(row["avg_relevance"]),
            "avg_hallucination": _round(row["avg_hallucination"]),
            "flag_rate_pii": _round(row["flag_rate_pii"]),
            "flag_rate_prompt_injection": _round(row["flag_rate_prompt_injection"]),
            "flag_rate_topic_guardrail": _round(row["flag_rate_topic_guardrail"]),
            "flag_rate_toxicity": _round(row["flag_rate_toxicity"]),
            "flag_rate_relevance": _round(row["flag_rate_relevance"]),
            "flag_rate_hallucination": _round(row["flag_rate_hallucination"]),
            "flag_rate_faithfulness": _round(row["flag_rate_faithfulness"]),
            "p95_latency_total": row["p95_latency_total"],
        }
        for row in rows
    ]


async def get_summary_metrics(pool: asyncpg.Pool) -> dict[str, Any]:
    """Return high-level 24-hour summary stats for the dashboard header cards."""
    query = f"""
        SELECT
            COUNT(*)                                                        AS total_requests_24h,
            COUNT(*) FILTER (WHERE blocked)                                 AS blocked_requests_24h,
            COALESCE(
                COUNT(*) FILTER (WHERE blocked)::float / NULLIF(COUNT(*), 0),
                0
            )                                                               AS block_rate_24h,
            COALESCE(AVG(latency_total), 0)::int                            AS avg_latency_24h_ms,
            COUNT(*) FILTER (
                WHERE NOT reviewed AND ({_FLAG_COLS})
            )                                                               AS unreviewed_flags
        FROM requests
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """

    top_flag_query = """
        SELECT unnested_flag, COUNT(*) AS cnt
        FROM (
            SELECT unnest(ARRAY[
                CASE WHEN flag_pii              THEN 'pii'              END,
                CASE WHEN flag_prompt_injection THEN 'prompt_injection' END,
                CASE WHEN flag_topic_guardrail  THEN 'topic_guardrail'  END,
                CASE WHEN flag_toxicity         THEN 'toxicity'         END,
                CASE WHEN flag_relevance        THEN 'relevance'        END,
                CASE WHEN flag_hallucination    THEN 'hallucination'    END,
                CASE WHEN flag_faithfulness     THEN 'faithfulness'     END
            ]) AS unnested_flag
            FROM requests
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        ) t
        WHERE unnested_flag IS NOT NULL
        GROUP BY unnested_flag
        ORDER BY cnt DESC
        LIMIT 1
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query)
        top_flag_row = await conn.fetchrow(top_flag_query)

    return {
        "total_requests_24h": row["total_requests_24h"],
        "blocked_requests_24h": row["blocked_requests_24h"],
        "block_rate_24h": _round(row["block_rate_24h"]),
        "avg_latency_24h_ms": row["avg_latency_24h_ms"],
        "top_flag_reason": top_flag_row["unnested_flag"] if top_flag_row else None,
        "unreviewed_flags": row["unreviewed_flags"],
    }


def _round(v) -> float | None:
    return round(float(v), 4) if v is not None else None
