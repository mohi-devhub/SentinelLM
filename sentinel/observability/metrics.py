"""Prometheus metrics for per-evaluator and per-LLM-call performance.

Extends the whole-HTTP-request metrics in sentinel.api.middleware with
finer-grained visibility into the evaluator chain and the LLM backend call.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

EVALUATOR_LATENCY = Histogram(
    "sentinel_evaluator_duration_seconds",
    "Per-evaluator inference latency",
    ["evaluator", "chain"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

EVALUATOR_FLAG_TOTAL = Counter(
    "sentinel_evaluator_flags_total",
    "Total times an evaluator's score tripped its configured threshold",
    ["evaluator"],
)

EVALUATOR_ERROR_TOTAL = Counter(
    "sentinel_evaluator_errors_total",
    "Total evaluator failures (exception or timeout) — always fail-open",
    ["evaluator"],
)

LLM_CALL_LATENCY = Histogram(
    "sentinel_llm_call_duration_seconds",
    "LLM backend call latency",
    ["provider", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

LLM_CALL_ERROR_TOTAL = Counter(
    "sentinel_llm_call_errors_total",
    "Total LLM backend call failures",
    ["provider"],
)


def observe_evaluator(
    evaluator_name: str,
    chain: str,
    latency_ms: int,
    flagged: bool,
    errored: bool,
) -> None:
    EVALUATOR_LATENCY.labels(evaluator=evaluator_name, chain=chain).observe(latency_ms / 1000)
    if flagged:
        EVALUATOR_FLAG_TOTAL.labels(evaluator=evaluator_name).inc()
    if errored:
        EVALUATOR_ERROR_TOTAL.labels(evaluator=evaluator_name).inc()


def observe_llm_call(provider: str, model: str, latency_ms: int, errored: bool) -> None:
    LLM_CALL_LATENCY.labels(provider=provider, model=model).observe(latency_ms / 1000)
    if errored:
        LLM_CALL_ERROR_TOTAL.labels(provider=provider).inc()
