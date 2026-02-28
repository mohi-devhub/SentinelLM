"""Unit tests for the chain aggregator.

Tests SentinelResult assembly and RequestRecord construction with
pre-built EvalResult lists — no evaluators or models involved.
"""

from __future__ import annotations

import pytest

from sentinel.chain.aggregator import assemble_result, build_request_record
from sentinel.evaluators.base import EvalResult


def _result(name: str, score: float, flag: bool = False, latency_ms: int = 10) -> EvalResult:
    return EvalResult(evaluator_name=name, score=score, flag=flag, latency_ms=latency_ms)


# ── assemble_result ──────────────────────────────────────────────────────────


def test_assemble_result_no_flags():
    """Clean results produce empty flags list and blocked=False."""
    input_r = [_result("pii", 0.01), _result("prompt_injection", 0.02)]
    output_r = [_result("toxicity", 0.03)]

    sr = assemble_result(input_r, output_r, latency_llm=100, latency_total=200)

    assert sr.flags == []
    assert sr.blocked is False
    assert sr.block_reason is None


def test_assemble_result_output_flag_does_not_block():
    """A flagged output evaluator appears in flags but does not set blocked=True."""
    input_r = [_result("pii", 0.01)]
    output_r = [_result("toxicity", 0.95, flag=True)]

    sr = assemble_result(input_r, output_r, latency_llm=50, latency_total=120)

    assert "toxicity" in sr.flags
    assert sr.blocked is False  # output flags never block


def test_assemble_result_input_flag_sets_blocked():
    """A flagged input evaluator sets blocked=True and block_reason."""
    input_r = [_result("prompt_injection", 0.92, flag=True)]
    output_r = []

    sr = assemble_result(input_r, output_r, latency_llm=None, latency_total=15)

    assert "prompt_injection" in sr.flags
    assert sr.blocked is True
    assert sr.block_reason is not None


def test_assemble_result_scores_populated():
    """All evaluator scores appear in sentinel_result.scores."""
    input_r = [_result("pii", 0.05), _result("prompt_injection", 0.10)]
    output_r = [_result("toxicity", 0.03), _result("relevance", 0.88)]

    sr = assemble_result(input_r, output_r, latency_llm=80, latency_total=180)

    assert sr.scores["pii"] == pytest.approx(0.05)
    assert sr.scores["prompt_injection"] == pytest.approx(0.10)
    assert sr.scores["toxicity"] == pytest.approx(0.03)
    assert sr.scores["relevance"] == pytest.approx(0.88)
    # Evaluators not run → None
    assert sr.scores["hallucination"] is None
    assert sr.scores["faithfulness"] is None


def test_assemble_result_latencies_populated():
    """Latency dict includes per-evaluator ms, llm ms, and total."""
    input_r = [_result("pii", 0.01, latency_ms=8)]
    output_r = [_result("toxicity", 0.02, latency_ms=22)]

    sr = assemble_result(input_r, output_r, latency_llm=150, latency_total=200)

    assert sr.latency_ms["pii"] == 8
    assert sr.latency_ms["toxicity"] == 22
    assert sr.latency_ms["llm"] == 150
    assert sr.latency_ms["total"] == 200


def test_assemble_result_errored_evaluator_score_is_none():
    """Evaluators that errored (score=None) appear as None in scores, not omitted."""
    input_r = []
    output_r = [EvalResult(evaluator_name="toxicity", score=None, flag=False, error="timeout")]

    sr = assemble_result(input_r, output_r, latency_llm=50, latency_total=60)

    assert sr.scores["toxicity"] is None
    assert "toxicity" not in sr.flags


# ── build_request_record ─────────────────────────────────────────────────────


def test_build_request_record_maps_scores_and_flags():
    """RequestRecord receives correct scores and flag columns from SentinelResult."""
    input_r = [_result("pii", 0.05), _result("prompt_injection", 0.92, flag=True)]
    output_r = [_result("toxicity", 0.80, flag=True), _result("relevance", 0.90)]
    sr = assemble_result(input_r, output_r, latency_llm=100, latency_total=250)

    record = build_request_record(
        sentinel_result=sr,
        model="llama3.2",
        input_hash="abc123",
        input_text="What is up?",
        input_redacted="What is up?",
        has_context=False,
    )

    assert record.model == "llama3.2"
    assert record.score_pii == pytest.approx(0.05)
    assert record.score_prompt_injection == pytest.approx(0.92)
    assert record.score_toxicity == pytest.approx(0.80)
    assert record.score_relevance == pytest.approx(0.90)
    assert record.score_hallucination is None

    assert record.flag_prompt_injection is True
    assert record.flag_toxicity is True
    assert record.flag_pii is False
    assert record.flag_relevance is False

    assert record.blocked is True
    assert record.latency_total == 250


def test_build_request_record_blocked_false_when_no_input_flags():
    """blocked=False when no input evaluator flagged (only output flags)."""
    output_r = [_result("toxicity", 0.95, flag=True)]
    sr = assemble_result([], output_r, latency_llm=50, latency_total=80)

    record = build_request_record(
        sentinel_result=sr,
        model="gpt-4o",
        input_hash="xyz",
        input_text="hi",
        input_redacted="hi",
        has_context=False,
    )

    assert record.blocked is False
