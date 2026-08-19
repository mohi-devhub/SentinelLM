"""Verifies the chain runner produces the expected span tree, including a
short-circuited evaluator surfacing as a cancelled span — the core claim
behind the tracing work (see sentinel/chain/runner.py).

Uses a local TracerProvider + InMemorySpanExporter (never the real global
provider) so this never touches process-global OTel state shared with other
tests.
"""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from sentinel.evaluators.base import BaseEvaluator, EvalPayload


class _FakeEvaluator(BaseEvaluator):
    """Minimal evaluator: returns a fixed score after an optional delay."""

    runs_on = "input"
    flag_direction = "above"

    def __init__(self, name: str, score: float, threshold: float = 0.8, delay: float = 0.0):
        self.name = name
        self._score = score
        self._threshold = threshold
        self._delay = delay
        super().__init__(config={})

    def _load_model(self) -> None:
        pass

    async def _run_inference(self, payload: EvalPayload):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._score, None

    def threshold(self) -> float:
        return self._threshold


@pytest.fixture
def exporter(monkeypatch):
    """Point sentinel.chain.runner's tracer at a local, in-memory exporter."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    test_tracer = provider.get_tracer("test")

    monkeypatch.setattr("sentinel.chain.runner.tracer", test_tracer)
    return exporter


@pytest.mark.asyncio
async def test_input_chain_creates_a_span_per_evaluator(exporter):
    from sentinel.chain.runner import run_input_chain

    evaluators = [
        _FakeEvaluator("pii", score=0.1),
        _FakeEvaluator("prompt_injection", score=0.2),
    ]
    payload = EvalPayload(input_text="hello")

    await run_input_chain(payload, evaluators, timeout=1.0)

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert "sentinel.chain.input" in names
    assert "sentinel.evaluator.pii" in names
    assert "sentinel.evaluator.prompt_injection" in names


@pytest.mark.asyncio
async def test_evaluator_spans_are_children_of_the_chain_span(exporter):
    from sentinel.chain.runner import run_input_chain

    evaluators = [_FakeEvaluator("pii", score=0.1)]
    payload = EvalPayload(input_text="hello")

    await run_input_chain(payload, evaluators, timeout=1.0)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    chain_span = spans["sentinel.chain.input"]
    evaluator_span = spans["sentinel.evaluator.pii"]
    assert evaluator_span.parent.span_id == chain_span.context.span_id


@pytest.mark.asyncio
async def test_short_circuited_evaluator_span_reflects_cancellation(exporter):
    """A slow evaluator cancelled by another's flag must show up as an error/cancelled span —
    not silently vanish from the trace.
    """
    from sentinel.chain.runner import run_input_chain

    slow = _FakeEvaluator("prompt_injection", score=0.1, delay=5.0)  # never finishes in time
    fast_flagged = _FakeEvaluator("pii", score=0.99, threshold=0.8, delay=0.0)
    payload = EvalPayload(input_text="hello")

    results, blocked_by = await run_input_chain(payload, [slow, fast_flagged], timeout=5.0)

    assert blocked_by is not None
    assert blocked_by.evaluator_name == "pii"

    # run_input_chain fires task.cancel() but never awaits the cancelled task —
    # its CancelledError unwind (and span close) happens on a later event-loop
    # tick. Give it one before asserting on exported spans.
    for _ in range(5):
        await asyncio.sleep(0)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    cancelled_span = spans["sentinel.evaluator.prompt_injection"]
    assert cancelled_span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_output_chain_creates_a_span_per_evaluator(exporter):
    from sentinel.chain.runner import run_output_chain

    class _FakeOutputEvaluator(_FakeEvaluator):
        runs_on = "output"

    evaluators = [_FakeOutputEvaluator("toxicity", score=0.05)]
    payload = EvalPayload(input_text="hello", output_text="world")

    await run_output_chain(payload, evaluators, timeout=1.0)

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert "sentinel.chain.output" in names
    assert "sentinel.evaluator.toxicity" in names
