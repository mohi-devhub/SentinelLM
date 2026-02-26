from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, EvalResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3.0  # seconds; overridden at startup from config


async def _run_with_timeout(
    ev: BaseEvaluator,
    payload: EvalPayload,
    timeout: float,
) -> EvalResult:
    """Evaluate with a per-evaluator timeout. Returns fail-open result on timeout."""
    try:
        return await asyncio.wait_for(ev.evaluate(payload), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("evaluator %s timed out after %.1fs", ev.name, timeout)
        return EvalResult(
            evaluator_name=ev.name,
            score=None,
            flag=False,
            latency_ms=int(timeout * 1000),
            error="timeout",
        )


async def run_input_chain(
    payload: EvalPayload,
    evaluators: list[BaseEvaluator],
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[list[EvalResult], Optional[EvalResult]]:
    """Run all input evaluators concurrently, short-circuiting on the first block.

    Uses asyncio.wait(FIRST_COMPLETED) so we can cancel remaining tasks the
    moment any evaluator returns a flagged result — avoiding wasted inference.

    Returns:
        (results, blocked_by) where blocked_by is the first flagged EvalResult
        or None if all evaluators passed.
    """
    if not evaluators:
        return [], None

    # Map task → evaluator so we can call is_flagged() when each completes
    task_to_ev: dict[asyncio.Task, BaseEvaluator] = {
        asyncio.create_task(_run_with_timeout(ev, payload, timeout)): ev
        for ev in evaluators
    }

    pending: set[asyncio.Task] = set(task_to_ev.keys())
    results: list[EvalResult] = []
    blocked_by: Optional[EvalResult] = None

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

            results.append(result)

            if result.flag and blocked_by is None:
                blocked_by = result
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
        return result

    return list(await asyncio.gather(*[_one(ev) for ev in evaluators]))
