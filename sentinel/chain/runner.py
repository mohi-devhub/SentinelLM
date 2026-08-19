from __future__ import annotations

import asyncio
import logging

from opentelemetry import trace

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, EvalResult
from sentinel.observability.alerting import window as alert_window
from sentinel.observability.metrics import observe_evaluator
from sentinel.observability.tracing import tracer

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3.0  # seconds; overridden at startup from config


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
) -> tuple[list[EvalResult], EvalResult | None]:
    """Run all input evaluators concurrently, short-circuiting on the first block.

    Uses asyncio.wait(FIRST_COMPLETED) so we can cancel remaining tasks the
    moment any evaluator returns a flagged result — avoiding wasted inference.

    Returns:
        (results, blocked_by) where blocked_by is the first flagged EvalResult
        or None if all evaluators passed.
    """
    if not evaluators:
        return [], None

    with tracer.start_as_current_span("sentinel.chain.input") as chain_span:
        chain_span.set_attribute("sentinel.chain.evaluator_count", len(evaluators))

        # Map task → evaluator so we can call is_flagged() when each completes
        task_to_ev: dict[asyncio.Task, BaseEvaluator] = {
            asyncio.create_task(_run_with_timeout(ev, payload, timeout)): ev for ev in evaluators
        }

        pending: set[asyncio.Task] = set(task_to_ev.keys())
        results: list[EvalResult] = []
        blocked_by: EvalResult | None = None

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

                if result.flag and blocked_by is None:
                    blocked_by = result
                    chain_span.set_attribute("sentinel.chain.blocked_by", result.evaluator_name)
                    # Cancel remaining tasks — no point running further checks
                    for p in pending:
                        p.cancel()
                    pending = set()
                    break

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
