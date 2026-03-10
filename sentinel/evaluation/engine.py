"""Offline evaluation engine for SentinelLM v2.

Scores (input, output) pairs through all enabled evaluators without calling
any LLM backend or HTTP proxy. Reuses sentinel/chain/runner.py directly so
the same evaluation logic runs in both runtime and CI offline modes.

Typical usage:
    config = yaml.safe_load(open("config.yaml"))
    engine = OfflineEvaluationEngine(config)
    results = await engine.evaluate_dataset(records, concurrency=8)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from sentinel.chain.runner import run_input_chain, run_output_chain
from sentinel.evaluators.base import EvalPayload, EvalResult
from sentinel.evaluators.registry import load_evaluators
from sentinel.evaluation.dataset import OfflineDatasetRecord


@dataclass
class OfflineEvalResult:
    """Aggregated evaluator results for a single (input, output) pair."""

    input_results: list[EvalResult]
    output_results: list[EvalResult]
    blocked_by: EvalResult | None = None


@dataclass
class OfflineRunResult:
    """Outcome of evaluating one OfflineDatasetRecord through the engine."""

    record: OfflineDatasetRecord
    eval_result: OfflineEvalResult
    error: str | None = None

    @property
    def scores(self) -> dict[str, float | None]:
        """Flat dict of evaluator_name → score (None if skipped/errored)."""
        all_results = self.eval_result.input_results + self.eval_result.output_results
        return {r.evaluator_name: r.score for r in all_results}

    @property
    def flags(self) -> list[str]:
        """Evaluator names whose score crossed the configured threshold."""
        all_results = self.eval_result.input_results + self.eval_result.output_results
        return [r.evaluator_name for r in all_results if r.flag]

    @property
    def blocked(self) -> bool:
        return self.eval_result.blocked_by is not None


class OfflineEvaluationEngine:
    """Score (input, output) pairs through all enabled evaluators.

    No LLM backend required. Evaluators are loaded directly from the config
    dict at instantiation time, reusing the same registry and BaseEvaluator
    implementations as the runtime proxy.

    Higher default concurrency than live mode (8 vs 4) because there is no
    network bottleneck — only local CPU-bound inference bounded by the thread
    pool executor.
    """

    def __init__(self, config: dict) -> None:
        evaluators = load_evaluators(config)
        self._input_evaluators = [e for e in evaluators if e.runs_on == "input"]
        self._output_evaluators = [e for e in evaluators if e.runs_on == "output"]
        self._timeout = float(
            config.get("performance", {}).get("evaluator_timeout_seconds", 3)
        )

    async def evaluate_record(
        self,
        input_text: str,
        output_text: str | None,
        context_documents: list[str] | None = None,
    ) -> OfflineEvalResult:
        """Score a single (input, output) pair through all enabled evaluators."""
        payload = EvalPayload(
            input_text=input_text,
            output_text=output_text,
            context_documents=context_documents or [],
            config={},
        )
        input_results, blocked_by = await run_input_chain(
            payload, self._input_evaluators, self._timeout
        )
        output_results = await run_output_chain(
            payload, self._output_evaluators, self._timeout
        )
        return OfflineEvalResult(
            input_results=input_results,
            output_results=output_results,
            blocked_by=blocked_by,
        )

    async def evaluate_dataset(
        self,
        records: list[OfflineDatasetRecord],
        concurrency: int = 8,
        on_progress: Callable[[int, int, OfflineRunResult], None] | None = None,
    ) -> list[OfflineRunResult]:
        """Evaluate all records concurrently (bounded by semaphore).

        Returns results sorted by record_index, matching dataset order.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(record: OfflineDatasetRecord) -> OfflineRunResult:
            async with semaphore:
                try:
                    eval_result = await self.evaluate_record(
                        record.input_text,
                        record.output_text,
                        record.context_documents or None,
                    )
                    return OfflineRunResult(record=record, eval_result=eval_result)
                except Exception as exc:
                    return OfflineRunResult(
                        record=record,
                        eval_result=OfflineEvalResult(input_results=[], output_results=[]),
                        error=str(exc),
                    )

        tasks = [asyncio.create_task(_one(r)) for r in records]
        results: list[OfflineRunResult] = []
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            if on_progress:
                on_progress(len(results), len(tasks), res)

        results.sort(key=lambda r: r.record.record_index)
        return results
