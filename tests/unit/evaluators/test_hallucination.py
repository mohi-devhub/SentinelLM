"""Unit tests for HallucinationEvaluator and FaithfulnessEvaluator.

Both use the same cross-encoder NLI model; they differ only in which label
index they extract (contradiction vs entailment) and their flag_direction.
All tests mock the CrossEncoder — no real model is loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sentinel.evaluators.base import EvalPayload

MOCK_CONFIG_HALLUCINATION = {
    "evaluators": {
        "hallucination": {
            "enabled": True,
            "threshold": 0.70,
            "model": "cross-encoder/nli-deberta-v3-base",
            "device": "cpu",
        }
    }
}

MOCK_CONFIG_FAITHFULNESS = {
    "evaluators": {
        "faithfulness": {
            "enabled": True,
            "threshold": 0.70,
            "model": "cross-encoder/nli-deberta-v3-base",
            "device": "cpu",
        }
    }
}

# NLI label order used by nli-deberta-v3-base: 0=contradiction, 1=entailment, 2=neutral
_MOCK_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def _mock_cross_encoder(scores_per_pair: list[list[float]]) -> MagicMock:
    """Build a mock CrossEncoder whose predict() returns given softmax scores."""

    mock_model_config = MagicMock()
    mock_model_config.id2label = _MOCK_ID2LABEL

    mock_inner = MagicMock()
    mock_inner.config = mock_model_config

    mock_ce = MagicMock()
    mock_ce.model = mock_inner
    mock_ce.predict.return_value = [[s for s in row] for row in scores_per_pair]
    return mock_ce


# ── HallucinationEvaluator ────────────────────────────────────────────────────


@pytest.fixture
def hallucination_evaluator():
    """HallucinationEvaluator with CrossEncoder mocked out."""
    mock_ce = _mock_cross_encoder([[0.05, 0.90, 0.05]])  # default: low contradiction

    with patch("sentence_transformers.cross_encoder.CrossEncoder", return_value=mock_ce):
        from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

        ev = HallucinationEvaluator(config=MOCK_CONFIG_HALLUCINATION)
        ev._model = mock_ce
        ev._contradiction_idx = 0  # index of "contradiction" label
        yield ev


@pytest.mark.asyncio
async def test_hallucination_low_contradiction_below_threshold(hallucination_evaluator):
    """Output that agrees with context → low contradiction score → no flag."""
    with patch("sentinel.evaluators.output.hallucination.run_in_executor") as mock_exec:
        mock_exec.return_value = (0.05, [0.05])

        payload = EvalPayload(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            context_documents=["Paris is the capital and most populous city of France."],
        )
        result = await hallucination_evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score < hallucination_evaluator.threshold()
    assert result.flag is False
    assert result.metadata is not None
    assert "per_doc_contradiction" in result.metadata
    assert result.metadata["num_docs"] == 1


@pytest.mark.asyncio
async def test_hallucination_high_contradiction_above_threshold(hallucination_evaluator):
    """Output contradicting context → high contradiction score → above threshold."""
    with patch("sentinel.evaluators.output.hallucination.run_in_executor") as mock_exec:
        mock_exec.return_value = (0.92, [0.92])

        payload = EvalPayload(
            input_text="What is the capital of France?",
            output_text="Berlin is the capital of France.",
            context_documents=["Paris is the capital of France."],
        )
        result = await hallucination_evaluator.evaluate(payload)

    assert result.score is not None
    assert result.score > hallucination_evaluator.threshold()
    assert result.flag is False  # runner sets flag, not evaluate()


@pytest.mark.asyncio
async def test_hallucination_uses_max_across_docs(hallucination_evaluator):
    """When multiple context docs are provided, the max contradiction score is used."""
    with patch("sentinel.evaluators.output.hallucination.run_in_executor") as mock_exec:
        # Two docs: low contradiction on first, high on second
        mock_exec.return_value = (0.88, [0.12, 0.88])

        payload = EvalPayload(
            input_text="q",
            output_text="The answer is wrong.",
            context_documents=["Doc A is consistent.", "Doc B is contradicted."],
        )
        result = await hallucination_evaluator.evaluate(payload)

    assert result.score == pytest.approx(0.88)
    assert result.metadata["num_docs"] == 2
    assert len(result.metadata["per_doc_contradiction"]) == 2


@pytest.mark.asyncio
async def test_hallucination_skipped_without_context(hallucination_evaluator):
    """Hallucination evaluator skips when no context_documents are provided."""
    payload = EvalPayload(
        input_text="What is 2+2?",
        output_text="4.",
        context_documents=None,
    )
    result = await hallucination_evaluator.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert result.metadata == {"skipped": "no context_documents"}


@pytest.mark.asyncio
async def test_hallucination_skipped_when_no_output(hallucination_evaluator):
    """Skipped when output_text is None."""
    payload = EvalPayload(input_text="q", output_text=None)
    result = await hallucination_evaluator.evaluate(payload)

    assert result.score is None
    assert result.metadata == {"skipped": "no output_text"}


@pytest.mark.asyncio
async def test_hallucination_fail_open_on_model_error(hallucination_evaluator):
    """Model crash → score=None, flag=False (fail-open)."""
    with patch("sentinel.evaluators.output.hallucination.run_in_executor") as mock_exec:
        mock_exec.side_effect = RuntimeError("NLI model crashed")

        payload = EvalPayload(
            input_text="q",
            output_text="a",
            context_documents=["context"],
        )
        result = await hallucination_evaluator.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert "NLI model crashed" in result.error


# ── FaithfulnessEvaluator ─────────────────────────────────────────────────────


@pytest.fixture
def faithfulness_evaluator():
    """FaithfulnessEvaluator with CrossEncoder mocked out."""
    mock_ce = _mock_cross_encoder([[0.05, 0.90, 0.05]])  # default: high entailment

    with patch("sentence_transformers.cross_encoder.CrossEncoder", return_value=mock_ce):
        from sentinel.evaluators.output.faithfulness import FaithfulnessEvaluator  # noqa: PLC0415

        ev = FaithfulnessEvaluator(config=MOCK_CONFIG_FAITHFULNESS)
        ev._model = mock_ce
        ev._entailment_idx = 1  # index of "entailment" label
        yield ev


@pytest.mark.asyncio
async def test_faithfulness_high_entailment_above_threshold(faithfulness_evaluator):
    """Output supported by context → high entailment → above threshold → no flag."""
    with patch("sentinel.evaluators.output.faithfulness.run_in_executor") as mock_exec:
        mock_exec.return_value = (0.92, [0.92])

        payload = EvalPayload(
            input_text="Who founded Apple?",
            output_text="Apple was founded by Steve Jobs.",
            context_documents=[
                "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."
            ],
        )
        result = await faithfulness_evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score > faithfulness_evaluator.threshold()
    assert result.flag is False


@pytest.mark.asyncio
async def test_faithfulness_low_entailment_below_threshold(faithfulness_evaluator):
    """Output not supported by context → low entailment → below threshold."""
    with patch("sentinel.evaluators.output.faithfulness.run_in_executor") as mock_exec:
        mock_exec.return_value = (0.08, [0.08])

        payload = EvalPayload(
            input_text="What did the report say?",
            output_text="The report concluded sales increased by 200%.",
            context_documents=["The report showed a modest 5% growth in Q3."],
        )
        result = await faithfulness_evaluator.evaluate(payload)

    assert result.score < faithfulness_evaluator.threshold()


@pytest.mark.asyncio
async def test_faithfulness_uses_max_entailment_across_docs(faithfulness_evaluator):
    """Max entailment across multiple docs is the final score."""
    with patch("sentinel.evaluators.output.faithfulness.run_in_executor") as mock_exec:
        mock_exec.return_value = (0.87, [0.15, 0.87])

        payload = EvalPayload(
            input_text="q",
            output_text="supported claim",
            context_documents=["Doc A is irrelevant.", "Doc B supports the claim."],
        )
        result = await faithfulness_evaluator.evaluate(payload)

    assert result.score == pytest.approx(0.87)
    assert result.metadata["num_docs"] == 2
    assert "per_doc_entailment" in result.metadata


@pytest.mark.asyncio
async def test_faithfulness_skipped_without_context(faithfulness_evaluator):
    """Faithfulness evaluator skips when no context_documents are provided."""
    payload = EvalPayload(input_text="q", output_text="a", context_documents=None)
    result = await faithfulness_evaluator.evaluate(payload)

    assert result.score is None
    assert result.metadata == {"skipped": "no context_documents"}


@pytest.mark.asyncio
async def test_faithfulness_fail_open_on_model_error(faithfulness_evaluator):
    """Model crash → score=None, flag=False (fail-open)."""
    with patch("sentinel.evaluators.output.faithfulness.run_in_executor") as mock_exec:
        mock_exec.side_effect = ValueError("tensor shape mismatch")

        payload = EvalPayload(input_text="q", output_text="a", context_documents=["ctx"])
        result = await faithfulness_evaluator.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert "tensor shape mismatch" in result.error


# ── Class-level attribute checks ──────────────────────────────────────────────


def test_hallucination_class_attributes():
    from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

    assert HallucinationEvaluator.name == "hallucination"
    assert HallucinationEvaluator.runs_on == "output"
    assert HallucinationEvaluator.flag_direction == "above"


def test_faithfulness_class_attributes():
    from sentinel.evaluators.output.faithfulness import FaithfulnessEvaluator  # noqa: PLC0415

    assert FaithfulnessEvaluator.name == "faithfulness"
    assert FaithfulnessEvaluator.runs_on == "output"
    assert FaithfulnessEvaluator.flag_direction == "below"
