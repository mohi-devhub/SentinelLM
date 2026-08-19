"""Unit tests for the 'consistency' backend (default) shared by
HallucinationEvaluator and FaithfulnessEvaluator — see
sentinel/evaluators/output/_consistency_model.py.

All tests mock load_consistency_model/score_consistency — no real model
(and no trust_remote_code execution) happens in these tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sentinel.evaluators.base import EvalPayload

MOCK_CONFIG_HALLUCINATION_DEFAULT = {
    "evaluators": {"hallucination": {"enabled": True, "threshold": 0.50}}
}
MOCK_CONFIG_FAITHFULNESS_DEFAULT = {
    "evaluators": {"faithfulness": {"enabled": True, "threshold": 0.50}}
}


def _make_evaluator(cls, config):
    """Build an evaluator with the consistency backend's model load mocked out."""
    with patch(
        "sentinel.evaluators.output._consistency_model.load_consistency_model",
        return_value=MagicMock(),
    ) as mock_load:
        ev = cls(config=config)
    assert mock_load.called
    return ev


# ── Backend defaults to 'consistency' when unset ────────────────────────────


def test_hallucination_defaults_to_consistency_backend():
    from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

    ev = _make_evaluator(HallucinationEvaluator, MOCK_CONFIG_HALLUCINATION_DEFAULT)
    assert ev._backend == "consistency"


def test_faithfulness_defaults_to_consistency_backend():
    from sentinel.evaluators.output.faithfulness import FaithfulnessEvaluator  # noqa: PLC0415

    ev = _make_evaluator(FaithfulnessEvaluator, MOCK_CONFIG_FAITHFULNESS_DEFAULT)
    assert ev._backend == "consistency"


# ── Score polarity: hallucination = 1 - consistency ─────────────────────────


@pytest.mark.asyncio
async def test_hallucination_score_is_inverse_of_consistency():
    from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

    ev = _make_evaluator(HallucinationEvaluator, MOCK_CONFIG_HALLUCINATION_DEFAULT)

    with patch(
        "sentinel.evaluators.output._consistency_model.score_consistency", return_value=[0.9]
    ):
        payload = EvalPayload(
            input_text="q",
            output_text="grounded answer",
            context_documents=["supporting context"],
        )
        result = await ev.evaluate(payload)

    # High consistency (0.9) -> low hallucination score (0.1)
    assert result.score == pytest.approx(0.1)
    assert result.metadata["backend"] == "consistency"


@pytest.mark.asyncio
async def test_hallucination_score_high_when_consistency_low():
    from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

    ev = _make_evaluator(HallucinationEvaluator, MOCK_CONFIG_HALLUCINATION_DEFAULT)

    with patch(
        "sentinel.evaluators.output._consistency_model.score_consistency", return_value=[0.05]
    ):
        payload = EvalPayload(
            input_text="q", output_text="ungrounded answer", context_documents=["context"]
        )
        result = await ev.evaluate(payload)

    assert result.score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_hallucination_uses_max_contradiction_across_docs():
    from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

    ev = _make_evaluator(HallucinationEvaluator, MOCK_CONFIG_HALLUCINATION_DEFAULT)

    # consistency scores [0.9, 0.2] -> contradiction scores [0.1, 0.8] -> max 0.8
    with patch(
        "sentinel.evaluators.output._consistency_model.score_consistency",
        return_value=[0.9, 0.2],
    ):
        payload = EvalPayload(
            input_text="q",
            output_text="output",
            context_documents=["consistent doc", "contradicted doc"],
        )
        result = await ev.evaluate(payload)

    assert result.score == pytest.approx(0.8)
    assert result.metadata["num_docs"] == 2


# ── Score polarity: faithfulness = consistency directly ─────────────────────


@pytest.mark.asyncio
async def test_faithfulness_score_equals_consistency():
    from sentinel.evaluators.output.faithfulness import FaithfulnessEvaluator  # noqa: PLC0415

    ev = _make_evaluator(FaithfulnessEvaluator, MOCK_CONFIG_FAITHFULNESS_DEFAULT)

    with patch(
        "sentinel.evaluators.output._consistency_model.score_consistency", return_value=[0.85]
    ):
        payload = EvalPayload(
            input_text="q", output_text="grounded answer", context_documents=["context"]
        )
        result = await ev.evaluate(payload)

    assert result.score == pytest.approx(0.85)
    assert result.metadata["backend"] == "consistency"


@pytest.mark.asyncio
async def test_faithfulness_uses_max_consistency_across_docs():
    from sentinel.evaluators.output.faithfulness import FaithfulnessEvaluator  # noqa: PLC0415

    ev = _make_evaluator(FaithfulnessEvaluator, MOCK_CONFIG_FAITHFULNESS_DEFAULT)

    with patch(
        "sentinel.evaluators.output._consistency_model.score_consistency",
        return_value=[0.15, 0.87],
    ):
        payload = EvalPayload(
            input_text="q",
            output_text="supported claim",
            context_documents=["irrelevant doc", "supporting doc"],
        )
        result = await ev.evaluate(payload)

    assert result.score == pytest.approx(0.87)


# ── Fail-open still holds under the consistency backend ─────────────────────


@pytest.mark.asyncio
async def test_hallucination_fail_open_on_consistency_model_error():
    from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

    ev = _make_evaluator(HallucinationEvaluator, MOCK_CONFIG_HALLUCINATION_DEFAULT)

    with patch(
        "sentinel.evaluators.output._consistency_model.score_consistency",
        side_effect=RuntimeError("consistency model crashed"),
    ):
        payload = EvalPayload(input_text="q", output_text="a", context_documents=["ctx"])
        result = await ev.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert "consistency model crashed" in result.error


# ── backend: nli explicitly opts out of the consistency default ─────────────


def test_backend_nli_does_not_load_consistency_model():
    from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

    config = {"evaluators": {"hallucination": {"enabled": True, "backend": "nli", "device": "cpu"}}}
    with (
        patch("sentinel.evaluators.output._consistency_model.load_consistency_model") as mock_load,
        patch("sentence_transformers.cross_encoder.CrossEncoder", return_value=MagicMock()),
    ):
        ev = HallucinationEvaluator(config=config)

    mock_load.assert_not_called()
    assert ev._backend == "nli"
