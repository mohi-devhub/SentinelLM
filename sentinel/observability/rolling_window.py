"""In-memory rolling window of recent request outcomes, used by the webhook
alerter for cheap threshold math without a time-series database.

Deliberately process-local, not persisted or shared across replicas — each
API replica alerts independently on its own traffic. Fine for the intended
use (catching a broken LLM backend or a runaway block rate), and avoids
adding Redis/Postgres load or a new dependency just for alerting.

Global across tenants, not per-tenant — alerting is an operator/ops concern
("is my deployment healthy") distinct from the per-tenant data isolation
the rest of the multi-tenancy work provides.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

_WINDOW_SECONDS = 900.0  # 15 minutes


@dataclass
class _Event:
    ts: float
    kind: str  # "request" | "llm_call" | "evaluator"
    blocked: bool = False
    errored: bool = False
    latency_ms: int | None = None


class RollingWindow:
    def __init__(self, window_seconds: float = _WINDOW_SECONDS) -> None:
        self._window_seconds = window_seconds
        self._events: deque[_Event] = deque()

    def record_request(self, blocked: bool, latency_ms: int) -> None:
        self._add(_Event(time.monotonic(), "request", blocked=blocked, latency_ms=latency_ms))

    def record_llm_call(self, errored: bool) -> None:
        self._add(_Event(time.monotonic(), "llm_call", errored=errored))

    def record_evaluator(self, errored: bool) -> None:
        self._add(_Event(time.monotonic(), "evaluator", errored=errored))

    def _add(self, event: _Event) -> None:
        self._events.append(event)
        self._prune()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window_seconds
        while self._events and self._events[0].ts < cutoff:
            self._events.popleft()

    def snapshot(self) -> dict:
        """Return current counts/rates over the rolling window."""
        self._prune()
        requests = [e for e in self._events if e.kind == "request"]
        llm_calls = [e for e in self._events if e.kind == "llm_call"]
        evaluator_calls = [e for e in self._events if e.kind == "evaluator"]

        def _fraction(items: list[_Event], predicate) -> float:
            return (sum(1 for i in items if predicate(i)) / len(items)) if items else 0.0

        latencies = sorted(e.latency_ms for e in requests if e.latency_ms is not None)
        p95_latency_ms = latencies[int(len(latencies) * 0.95)] if latencies else 0

        return {
            "request_count": len(requests),
            "block_rate": _fraction(requests, lambda e: e.blocked),
            "llm_call_count": len(llm_calls),
            "llm_error_rate": _fraction(llm_calls, lambda e: e.errored),
            "evaluator_call_count": len(evaluator_calls),
            "evaluator_failure_rate": _fraction(evaluator_calls, lambda e: e.errored),
            "p95_latency_ms": p95_latency_ms,
        }
