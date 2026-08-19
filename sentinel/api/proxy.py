from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from pydantic import BaseModel

from sentinel.api.rate_limit import enforce_rate_limit
from sentinel.cache.client import cache_key as build_cache_key
from sentinel.chain.aggregator import SentinelResult, assemble_result, build_request_record
from sentinel.chain.runner import run_input_chain, run_output_chain
from sentinel.evaluators.base import EvalPayload
from sentinel.observability.alerting import window as alert_window
from sentinel.observability.metrics import observe_llm_call
from sentinel.observability.tracing import tracer
from sentinel.proxy.factory import get_llm_client
from sentinel.storage.queries.requests import insert_request
from sentinel.ws.broadcaster import publish_event

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
    context_documents: list[str] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


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
    redis,
    record,
    request_id: uuid.UUID,
    sentinel_result: SentinelResult,
) -> None:
    """Background task: persist the request record and push a WebSocket event.

    Errors here are logged but never raised — they must not affect the client.
    Events are published to Redis (not delivered directly) so every API
    replica's dashboard connections receive them — see sentinel/ws/broadcaster.py.
    """
    alert_window.record_request(sentinel_result.blocked, record.latency_total)

    try:
        await insert_request(pool, record)
    except Exception:
        logger.exception("failed to insert request record id=%s", request_id)

    try:
        await publish_event(
            redis,
            {
                "event_type": "request_blocked" if sentinel_result.blocked else "request_passed",
                "request_id": str(request_id),
                "tenant_id": str(record.tenant_id) if record.tenant_id else None,
                "model": record.model,
                "blocked": sentinel_result.blocked,
                "block_reason": sentinel_result.block_reason,
                "flags": sentinel_result.flags,
                "scores": sentinel_result.scores,
                "latency_total": record.latency_total,
            },
        )
    except Exception:
        logger.exception("failed to broadcast ws event id=%s", request_id)


# ── Route ────────────────────────────────────────────────────────────────────


@router.post(
    "/v1/chat/completions",
    response_model=None,
    dependencies=[Depends(enforce_rate_limit)],
)
async def chat_completions(
    body: ChatCompletionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse | StreamingResponse:
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

    # Reuse RequestIDMiddleware's ID (always a valid UUID — see middleware.py)
    # so the HTTP response, the DB row, and the trace span all share one ID.
    request_id = uuid.UUID(http_request.state.request_id)

    current_span = trace.get_current_span()
    current_span.set_attribute("sentinel.request_id", str(request_id))
    if http_request.state.tenant_id is not None:
        current_span.set_attribute("sentinel.tenant_id", str(http_request.state.tenant_id))

    # ── Input evaluator chain ────────────────────────────────────────────────
    cache_config: dict = config.get("cache", {})
    input_cache_key: str | None = None
    if cache_config.get("enabled", False):
        config_version = str(config.get("app", {}).get("config_version", "1"))
        input_cache_key = build_cache_key(
            input_text, config_version, str(http_request.state.tenant_id)
        )

    input_results, blocked_by = await run_input_chain(
        payload,
        http_request.app.state.input_evaluators,
        timeout,
        redis=http_request.app.state.redis,
        cache_key=input_cache_key,
        cache_ttl=int(cache_config.get("ttl_seconds", 3600)),
    )

    # ── PII redact: flag but do NOT block; swap in the cleaned text ──────────
    # Looked up from input_results (not blocked_by) because run_input_chain
    # never treats a redact-configured PII flag as block-worthy in the first
    # place — see _is_actual_block() in chain/runner.py — so it never becomes
    # blocked_by to begin with. blocked_by is still cleared here defensively
    # in case a future evaluator adds its own non-blocking flag semantics.
    pii_redacted_text: str | None = None
    pii_result = next((r for r in input_results if r.evaluator_name == "pii"), None)
    if pii_result and pii_result.flag and (pii_result.metadata or {}).get("action") == "redact":
        pii_redacted_text = (pii_result.metadata or {}).get("redacted_text", input_text)
        if blocked_by is pii_result:
            blocked_by = None  # clear block — request continues with redacted text

    if blocked_by and not shadow_mode:
        latency_total = int((time.monotonic() - total_start) * 1000)
        sentinel_result = assemble_result(input_results, [], None, latency_total)

        record = build_request_record(
            sentinel_result=sentinel_result,
            model=body.model,
            input_hash=input_hash,
            input_text=(
                input_text if config.get("storage", {}).get("store_input_text", True) else None
            ),
            input_redacted=input_text,
            has_context=has_context,
            tenant_id=http_request.state.tenant_id,
        )
        record.id = request_id

        background_tasks.add_task(
            _log_and_broadcast,
            http_request.app.state.db_pool,
            http_request.app.state.redis,
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
    llm_client = get_llm_client(
        config, settings.openai_api_key, settings.anthropic_api_key, settings.gemini_api_key
    )
    llm_provider = config.get("llm_backend", {}).get("provider", "unknown")

    # Strip context_documents and stream — SentinelLM extensions, not LLM API fields
    request_dict = body.model_dump(exclude={"context_documents", "stream"}, exclude_none=True)

    # Apply PII redaction: substitute sanitised text in the last user message
    if pii_redacted_text:
        msgs = request_dict.get("messages", [])
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i] = {**msgs[i], "content": pii_redacted_text}
                break

    # ── Streaming path ───────────────────────────────────────────────────────
    if body.stream:

        async def _stream_response() -> AsyncGenerator[bytes, None]:
            accumulated_text = ""
            llm_start_inner = time.monotonic()
            llm_span = tracer.start_span("sentinel.llm.call")
            llm_span.set_attribute("sentinel.llm.provider", llm_provider)
            llm_span.set_attribute("sentinel.llm.model", body.model)

            try:
                async for chunk in llm_client.stream_chat(request_dict):
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                    try:
                        accumulated_text += chunk["choices"][0]["delta"].get("content", "")
                    except (KeyError, IndexError):
                        pass
            except Exception as exc:
                latency_llm_error = int((time.monotonic() - llm_start_inner) * 1000)
                observe_llm_call(llm_provider, body.model, latency_llm_error, errored=True)
                alert_window.record_llm_call(errored=True)
                llm_span.record_exception(exc)
                llm_span.set_status(trace.StatusCode.ERROR)
                llm_span.end()
                logger.error("LLM streaming error: %s", exc)
                err = json.dumps({"error": {"type": "llm_backend_error", "message": str(exc)}})
                yield f"data: {err}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            latency_llm_inner = int((time.monotonic() - llm_start_inner) * 1000)
            observe_llm_call(llm_provider, body.model, latency_llm_inner, errored=False)
            alert_window.record_llm_call(errored=False)
            llm_span.end()

            payload.output_text = accumulated_text or None
            output_results = await run_output_chain(
                payload, http_request.app.state.output_evaluators, timeout
            )

            latency_total_inner = int((time.monotonic() - total_start) * 1000)
            sentinel_result = assemble_result(
                input_results, output_results, latency_llm_inner, latency_total_inner
            )

            record = build_request_record(
                sentinel_result=sentinel_result,
                model=body.model,
                input_hash=input_hash,
                input_text=(
                    input_text if config.get("storage", {}).get("store_input_text", True) else None
                ),
                input_redacted=pii_redacted_text or input_text,
                has_context=has_context,
                tenant_id=http_request.state.tenant_id,
            )
            record.id = request_id

            asyncio.ensure_future(
                _log_and_broadcast(
                    http_request.app.state.db_pool,
                    http_request.app.state.redis,
                    record,
                    request_id,
                    sentinel_result,
                )
            )

            sentinel_chunk = {
                "sentinel": {
                    "request_id": str(request_id),
                    "scores": sentinel_result.scores,
                    "flags": sentinel_result.flags,
                    "latency_ms": sentinel_result.latency_ms,
                }
            }
            yield f"data: {json.dumps(sentinel_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            _stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Non-streaming path ───────────────────────────────────────────────────
    llm_start = time.monotonic()
    with tracer.start_as_current_span("sentinel.llm.call") as llm_span:
        llm_span.set_attribute("sentinel.llm.provider", llm_provider)
        llm_span.set_attribute("sentinel.llm.model", body.model)
        try:
            llm_response = await llm_client.chat(request_dict)
        except Exception as exc:
            latency_llm_error = int((time.monotonic() - llm_start) * 1000)
            observe_llm_call(llm_provider, body.model, latency_llm_error, errored=True)
            alert_window.record_llm_call(errored=True)
            llm_span.record_exception(exc)
            llm_span.set_status(trace.StatusCode.ERROR)
            logger.error("LLM backend error: %s", exc)
            return JSONResponse(
                status_code=502,
                content={"error": {"type": "llm_backend_error", "message": str(exc)}},
            )
    latency_llm = int((time.monotonic() - llm_start) * 1000)
    observe_llm_call(llm_provider, body.model, latency_llm, errored=False)
    alert_window.record_llm_call(errored=False)

    # Extract the assistant text for output evaluators
    output_text: str | None = None
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
        input_redacted=pii_redacted_text or input_text,
        has_context=has_context,
        tenant_id=http_request.state.tenant_id,
    )
    record.id = request_id

    background_tasks.add_task(
        _log_and_broadcast,
        http_request.app.state.db_pool,
        http_request.app.state.redis,
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
                "pii_redacted_text": pii_redacted_text,
            },
        },
    )
