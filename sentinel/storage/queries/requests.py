from __future__ import annotations

import asyncpg

from sentinel.storage.models import RequestRecord

_INSERT_SQL = """
INSERT INTO requests (
    model,
    input_hash,
    input_text,
    input_redacted,
    has_context,
    blocked,
    block_reason,
    score_pii,
    score_prompt_injection,
    score_topic_guardrail,
    score_toxicity,
    score_relevance,
    score_hallucination,
    score_faithfulness,
    latency_pii,
    latency_prompt_injection,
    latency_topic_guardrail,
    latency_toxicity,
    latency_relevance,
    latency_hallucination,
    latency_faithfulness,
    latency_llm,
    latency_total,
    flag_pii,
    flag_prompt_injection,
    flag_topic_guardrail,
    flag_toxicity,
    flag_relevance,
    flag_hallucination,
    flag_faithfulness
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
    $21, $22, $23, $24, $25, $26, $27, $28, $29, $30
)
RETURNING id, created_at
"""


async def insert_request(pool: asyncpg.Pool, record: RequestRecord) -> RequestRecord:
    """Persist a request record and return it with id and created_at populated."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            _INSERT_SQL,
            record.model,
            record.input_hash,
            record.input_text,
            record.input_redacted,
            record.has_context,
            record.blocked,
            record.block_reason,
            record.score_pii,
            record.score_prompt_injection,
            record.score_topic_guardrail,
            record.score_toxicity,
            record.score_relevance,
            record.score_hallucination,
            record.score_faithfulness,
            record.latency_pii,
            record.latency_prompt_injection,
            record.latency_topic_guardrail,
            record.latency_toxicity,
            record.latency_relevance,
            record.latency_hallucination,
            record.latency_faithfulness,
            record.latency_llm,
            record.latency_total,
            record.flag_pii,
            record.flag_prompt_injection,
            record.flag_topic_guardrail,
            record.flag_toxicity,
            record.flag_relevance,
            record.flag_hallucination,
            record.flag_faithfulness,
        )
    record.id = row["id"]
    record.created_at = row["created_at"]
    return record
