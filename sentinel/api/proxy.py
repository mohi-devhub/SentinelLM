from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sentinel.chain.aggregator import SentinelResult, assemble_result, build_request_record
from sentinel.chain.runner import run_input_chain, run_output_chain
from sentinel.evaluators.base import EvalPayload
from sentinel.proxy.factory import get_llm_client
from sentinel.storage.queries.requests import insert_request
from sentinel.ws.broadcaster import manager as ws_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Maps evaluator name → openapi.yaml SentinelBlockError.code enum value
_BLOCK_CODE: dict[str, str] = {
    "pii": "pii_detected",
    "prompt_injection": "prompt_injection_detected",
    "topic_guardrail": "off_topic",
}


# ── Pydantic request model ───────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    context_documents: Optional[list[str]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_input_text(messages: list[Message]) -> str:
    """Return the last user-role message content as the evaluator input."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


def _compute_input_hash(messages: list[Message]) -> str:
    """SHA-256 of the canonicalized messages JSON (used for cache keying and dedup)."""
    raw = json.dumps([m.model_dump() for m in messages], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


async def _log_and_broadcast(
    pool,
    record,
    request_id: uuid.UUID,
    sentinel_result: SentinelResult,
) -> None:
    """Background task: persist the request record and push a WebSocket event.

    Errors here are logged but never raised — they must not affect the client.
    """
    try:
        await insert_request(pool, record)
    except Exception:
        logger.exception("failed to insert request record id=%s", request_id)

    try:
        await ws_manager.broadcast({
            "event_type": "request_blocked" if sentinel_result.blocked else "request_passed",
            "request_id": str(request_id),
            "model": record.model,
            "blocked": sentinel_result.blocked,
            "block_reason": sentinel_result.block_reason,
            "flags": sentinel_result.flags,
            "scores": sentinel_result.scores,
            "latency_total": record.latency_total,
        })
    except Exception:
        logger.exception("failed to broadcast ws event id=%s", request_id)


# ── Route ────────────────────────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    total_start = time.monotonic()

    config: dict = http_request.app.state.config
    shadow_mode: bool = config.get("app", {}).get("shadow_mode", False)
    timeout: float = float(config.get("performance", {}).get("evaluator_timeout_seconds", 3))

    input_text = _extract_input_text(body.messages)
    input_hash = _compute_input_hash(body.messages)
    has_context = bool(body.context_documents)

    payload = EvalPayload(
        input_text=input_text,
        context_documents=body.context_documents,
        config=config,
    )

    # Pre-generate UUID so the response includes it before the background INSERT
    request_id = uuid.uuid4()

    # ── Input evaluator chain ────────────────────────────────────────────────
    input_results, blocked_by = await run_input_chain(
        payload, http_request.app.state.input_evaluators, timeout
    )

    if blocked_by and not shadow_mode:
        latency_total = int((time.monotonic() - total_start) * 1000)
        sentinel_result = assemble_result(input_results, [], None, latency_total)

        record = build_request_record(
            sentinel_result=sentinel_result,
            model=body.model,
            input_hash=input_hash,
            input_text=input_text if config.get("storage", {}).get("store_input_text", True) else None,
            input_redacted=input_text,
            has_context=has_context,
        )
        record.id = request_id

        background_tasks.add_task(
            _log_and_broadcast,
            http_request.app.state.db_pool,
            record,
            request_id,
            sentinel_result,
        )

        # Look up the threshold for the blocking evaluator for the error body
        ev_threshold = 0.8
        for ev in http_request.app.state.input_evaluators:
            if ev.name == blocked_by.evaluator_name:
                ev_threshold = ev.threshold()
                break

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "sentinel_block",
                    "code": _BLOCK_CODE.get(
                        blocked_by.evaluator_name,
                        f"{blocked_by.evaluator_name}_detected",
                    ),
                    "score": blocked_by.score,
                    "threshold": ev_threshold,
                    "message": "Request blocked by SentinelLM input guardrail.",
                }
            },
        )

    # ── LLM backend call ─────────────────────────────────────────────────────
    from sentinel.settings import get_settings  # noqa: PLC0415 — avoid module-level import

    settings = get_settings()
    llm_client = get_llm_client(config, settings.openai_api_key, settings.anthropic_api_key)

    # Strip context_documents — it is a SentinelLM extension, not an LLM API field
    request_dict = body.model_dump(exclude={"context_documents"}, exclude_none=True)

    llm_start = time.monotonic()
    try:
        llm_response = await llm_client.chat(request_dict)
    except Exception as exc:
        logger.error("LLM backend error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "llm_backend_error", "message": str(exc)}},
        )
    latency_llm = int((time.monotonic() - llm_start) * 1000)

    # Extract the assistant text for output evaluators
    output_text: Optional[str] = None
    try:
        output_text = llm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        logger.warning("could not extract output_text from LLM response")

    # ── Output evaluator chain ───────────────────────────────────────────────
    payload.output_text = output_text
    output_results = await run_output_chain(
        payload, http_request.app.state.output_evaluators, timeout
    )

    latency_total = int((time.monotonic() - total_start) * 1000)

    # ── Assemble, log, respond ───────────────────────────────────────────────
    sentinel_result = assemble_result(input_results, output_results, latency_llm, latency_total)

    record = build_request_record(
        sentinel_result=sentinel_result,
        model=body.model,
        input_hash=input_hash,
        input_text=input_text if config.get("storage", {}).get("store_input_text", True) else None,
        input_redacted=input_text,  # Phase 1: no redaction (PII evaluator added in Phase 2)
        has_context=has_context,
    )
    record.id = request_id

    background_tasks.add_task(
        _log_and_broadcast,
        http_request.app.state.db_pool,
        record,
        request_id,
        sentinel_result,
    )

    return JSONResponse(
        status_code=200,
        content={
            **llm_response,
            "sentinel": {
                "request_id": str(request_id),
                "scores": sentinel_result.scores,
                "flags": sentinel_result.flags,
                "latency_ms": sentinel_result.latency_ms,
            },
        },
    )
