from __future__ import annotations

import asyncio
import logging
from typing import Any

from opentelemetry import trace

from sentinel.cache.client import get_cached_results, set_cached_results
from sentinel.evaluators.base import BaseEvaluator, EvalPayload, EvalResult
from sentinel.observability.alerting import window as alert_window
from sentinel.observability.metrics import observe_evaluator
from sentinel.observability.tracing import tracer

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3.0  # seconds; overridden at startup from config


def _is_actual_block(result: EvalResult) -> bool:
    """True if a flagged result should short-circuit and cancel sibling evaluators.

    False for pii's action:redact — it flags for logging/redaction but is
    explicitly non-blocking (proxy.py clears blocked_by for it and continues
    with the redacted text). Without this check, any request containing
    detectable PII would race-cancel every other still-running input
    evaluator — including prompt_injection mid-inference — purely because
    PII happened to finish first, letting even blatant injection attempts
    through unscored whenever PII is also present. Mirrors the exact
    condition proxy.py already applies after the chain returns; this just
    makes the short-circuit decision itself aware of it, instead of only
    the block decision downstream.
    """
    if result.evaluator_name == "pii" and (result.metadata or {}).get("action") == "redact":
        return False
    return True


async def _run_with_timeout(
    ev: BaseEvaluator,
    payload: EvalPayload,
    timeout: float,
) -> EvalResult:
    """Evaluate with a per-evaluator timeout. Returns fail-open result on timeout.

    Wrapped in its own span — a task cancelled by the chain runner's
    short-circuit logic surfaces in the trace as a cancelled/error span,
    since asyncio.Task copies the enclosing context (including the active
    span) at creation time with no extra propagation code required. Note
    asyncio.CancelledError is a BaseException (not Exception) as of Python
    3.8, so OTel's span context manager does NOT mark it as an error by
    default — that's handled explicitly below.
    """
    with tracer.start_as_current_span(f"sentinel.evaluator.{ev.name}") as span:
        span.set_attribute("sentinel.evaluator.name", ev.name)
        try:
            result = await asyncio.wait_for(ev.evaluate(payload), timeout=timeout)
            if result.score is not None:
                span.set_attribute("sentinel.evaluator.score", result.score)
            if result.error is not None:
                span.set_attribute("sentinel.evaluator.error", result.error)
            return result
        except TimeoutError:
            span.set_attribute("sentinel.evaluator.timed_out", True)
            span.set_status(trace.StatusCode.ERROR, description="evaluator timed out")
            logger.warning("evaluator %s timed out after %.1fs", ev.name, timeout)
            return EvalResult(
                evaluator_name=ev.name,
                score=None,
                flag=False,
                latency_ms=int(timeout * 1000),
                error="timeout",
            )
        except asyncio.CancelledError:
            span.set_attribute("sentinel.evaluator.cancelled", True)
            span.set_status(trace.StatusCode.ERROR, description="cancelled by chain short-circuit")
            raise


async def run_input_chain(
    payload: EvalPayload,
    evaluators: list[BaseEvaluator],
    timeout: float = _DEFAULT_TIMEOUT,
    redis: Any | None = None,
    cache_key: str | None = None,
    cache_ttl: int = 3600,
) -> tuple[list[EvalResult], EvalResult | None]:
    """Run all input evaluators concurrently, short-circuiting on the first block.

    Uses asyncio.wait(FIRST_COMPLETED) so we can cancel remaining tasks the
    moment any evaluator returns a flagged result — avoiding wasted inference.

    When `redis` and `cache_key` are both given, evaluators with a cached
    result (same input text + config version + tenant — see
    sentinel/cache/client.py) skip inference entirely; freshly-computed
    results are written back to cache for next time. Passing either as None
    (the default) skips caching entirely, matching pre-caching behavior.

    Returns:
        (results, blocked_by) where blocked_by is the first flagged EvalResult
        or None if all evaluators passed.
    """
    if not evaluators:
        return [], None

    cached: dict[str, dict] = {}
    if redis is not None and cache_key is not None:
        try:
            cached = await get_cached_results(redis, cache_key) or {}
        except Exception:
            logger.exception("evaluator cache lookup failed; proceeding without cache")

    with tracer.start_as_current_span("sentinel.chain.input") as chain_span:
        chain_span.set_attribute("sentinel.chain.evaluator_count", len(evaluators))

        results: list[EvalResult] = []
        blocked_by: EvalResult | None = None
        to_run: list[BaseEvaluator] = []

        # ── Cache hits — no inference, but still counted, traced, and can block ──
        for ev in evaluators:
            entry = cached.get(ev.name)
            if entry is None:
                to_run.append(ev)
                continue

            with tracer.start_as_current_span(f"sentinel.evaluator.{ev.name}") as span:
                span.set_attribute("sentinel.evaluator.name", ev.name)
                span.set_attribute("sentinel.evaluator.cache_hit", True)
                score = entry.get("score")
                result = EvalResult(
                    evaluator_name=ev.name,
                    score=score,
                    flag=ev.is_flagged(score) if score is not None else False,
                    latency_ms=0,
                    metadata=entry.get("metadata"),
                )
                if score is not None:
                    span.set_attribute("sentinel.evaluator.score", score)

            observe_evaluator(ev.name, ev.runs_on, 0, result.flag, False)
            alert_window.record_evaluator(errored=False)
            results.append(result)

            if result.flag and blocked_by is None and _is_actual_block(result):
                blocked_by = result
                chain_span.set_attribute("sentinel.chain.blocked_by", result.evaluator_name)

        if blocked_by is not None or not to_run:
            return results, blocked_by

        # ── Cache misses — run for real, then cache successful results ──────────
        task_to_ev: dict[asyncio.Task, BaseEvaluator] = {
            asyncio.create_task(_run_with_timeout(ev, payload, timeout)): ev for ev in to_run
        }

        pending: set[asyncio.Task] = set(task_to_ev.keys())
        to_cache: dict[str, dict] = {}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    result = task.result()
                except Exception as exc:
                    # evaluate() already catches all exceptions, but be defensive
                    ev = task_to_ev[task]
                    result = EvalResult(
                        evaluator_name=ev.name, score=None, flag=False, error=str(exc)
                    )

                ev = task_to_ev[task]

                # Apply threshold logic — evaluators return flag=False; runner sets it
                if result.score is not None:
                    result.flag = ev.is_flagged(result.score)

                observe_evaluator(
                    ev.name, ev.runs_on, result.latency_ms, result.flag, result.error is not None
                )
                alert_window.record_evaluator(errored=result.error is not None)

                results.append(result)

                if result.error is None and result.score is not None:
                    to_cache[ev.name] = {"score": result.score, "metadata": result.metadata}

                if result.flag and blocked_by is None and _is_actual_block(result):
                    blocked_by = result
                    chain_span.set_attribute("sentinel.chain.blocked_by", result.evaluator_name)
                    # Cancel remaining tasks — no point running further checks
                    for p in pending:
                        p.cancel()
                    pending = set()
                    break

        if to_cache and redis is not None and cache_key is not None:
            try:
                await set_cached_results(redis, cache_key, to_cache, cache_ttl)
            except Exception:
                logger.exception("evaluator cache write failed")

        return results, blocked_by


async def run_output_chain(
    payload: EvalPayload,
    evaluators: list[BaseEvaluator],
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[EvalResult]:
    """Run all output evaluators concurrently. All always run — no short-circuit.

    Output evaluators provide the full quality picture even when some scores
    are high; everything is logged for analysis.
    """
    if not evaluators:
        return []

    async def _one(ev: BaseEvaluator) -> EvalResult:
        result = await _run_with_timeout(ev, payload, timeout)
        if result.score is not None:
            result.flag = ev.is_flagged(result.score)
        observe_evaluator(
            ev.name, ev.runs_on, result.latency_ms, result.flag, result.error is not None
        )
        alert_window.record_evaluator(errored=result.error is not None)
        return result

    with tracer.start_as_current_span("sentinel.chain.output") as chain_span:
        chain_span.set_attribute("sentinel.chain.evaluator_count", len(evaluators))
        return list(await asyncio.gather(*[_one(ev) for ev in evaluators]))
