"""Unit tests for sentinel.observability.rolling_window.RollingWindow."""

from __future__ import annotations

from sentinel.observability.rolling_window import RollingWindow


def test_empty_window_snapshot():
    w = RollingWindow()
    snap = w.snapshot()
    assert snap == {
        "request_count": 0,
        "block_rate": 0.0,
        "llm_call_count": 0,
        "llm_error_rate": 0.0,
        "evaluator_call_count": 0,
        "evaluator_failure_rate": 0.0,
        "p95_latency_ms": 0,
    }


def test_block_rate_computed_from_requests():
    w = RollingWindow()
    w.record_request(blocked=True, latency_ms=10)
    w.record_request(blocked=False, latency_ms=10)
    w.record_request(blocked=False, latency_ms=10)
    w.record_request(blocked=False, latency_ms=10)
    snap = w.snapshot()
    assert snap["request_count"] == 4
    assert snap["block_rate"] == 0.25


def test_llm_error_rate_independent_of_requests():
    w = RollingWindow()
    w.record_request(blocked=False, latency_ms=5)
    w.record_llm_call(errored=True)
    w.record_llm_call(errored=False)
    snap = w.snapshot()
    assert snap["llm_call_count"] == 2
    assert snap["llm_error_rate"] == 0.5
    # A blocked/errored LLM call must never leak into block_rate
    assert snap["block_rate"] == 0.0


def test_evaluator_failure_rate():
    w = RollingWindow()
    for _ in range(3):
        w.record_evaluator(errored=False)
    w.record_evaluator(errored=True)
    snap = w.snapshot()
    assert snap["evaluator_call_count"] == 4
    assert snap["evaluator_failure_rate"] == 0.25


def test_p95_latency_uses_sorted_index():
    w = RollingWindow()
    for latency in range(1, 101):  # 1..100 ms
        w.record_request(blocked=False, latency_ms=latency)
    snap = w.snapshot()
    # sorted list index int(100 * 0.95) == 95 -> value 96 (0-indexed)
    assert snap["p95_latency_ms"] == 96


def test_old_events_are_pruned():
    w = RollingWindow(window_seconds=0.01)
    w.record_request(blocked=True, latency_ms=1)
    import time

    time.sleep(0.05)
    w.record_request(blocked=False, latency_ms=1)
    snap = w.snapshot()
    # Only the second (recent) request should remain
    assert snap["request_count"] == 1
    assert snap["block_rate"] == 0.0
