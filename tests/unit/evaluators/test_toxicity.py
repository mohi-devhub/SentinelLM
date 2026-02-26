"""Unit tests for ToxicityEvaluator.

All tests mock Detoxify — no real model is loaded or called.
"""
from __future__ import annotations

import sys

import pytest
from unittest.mock import MagicMock

from sentinel.evaluators.base import EvalPayload

MOCK_CONFIG = {
    "evaluators": {
        "toxicity": {
            "enabled": True,
            "threshold": 0.70,
            "dimensions": ["toxicity", "threat"],
            "flag_on": "any",
        }
    }
}


@pytest.fixture
def evaluator():
    """ToxicityEvaluator with Detoxify replaced by a MagicMock.

    Detoxify is imported inside _load_model() (deferred import), so we patch
    sys.modules["detoxify"] before instantiation. This works whether or not
    the detoxify package is installed in the current environment.
    """
    # Build a fake detoxify module whose Detoxify class returns a mock instance
    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)
    fake_detoxify_module = MagicMock()
    fake_detoxify_module.Detoxify = mock_cls

    original = sys.modules.get("detoxify")
    sys.modules["detoxify"] = fake_detoxify_module
    try:
        # Import after the sys.modules patch so _load_model() picks up the mock
        from sentinel.evaluators.output.toxicity import ToxicityEvaluator  # noqa: PLC0415

        ev = ToxicityEvaluator(config=MOCK_CONFIG)
        # Expose the mock instance directly for test assertions
        ev._model = mock_instance
        yield ev
    finally:
        if original is None:
            sys.modules.pop("detoxify", None)
        else:
            sys.modules["detoxify"] = original


@pytest.mark.asyncio
async def test_clean_output_not_flagged(evaluator):
    """Low toxicity scores produce a score below threshold; flag stays False."""
    evaluator._model.predict.return_value = {"toxicity": 0.01, "threat": 0.02}

    payload = EvalPayload(input_text="Hello", output_text="Paris is the capital of France.")
    result = await evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score < evaluator.threshold()
    # The chain runner sets flag — evaluate() always returns flag=False
    assert result.flag is False


@pytest.mark.asyncio
async def test_toxic_output_scores_above_threshold(evaluator):
    """High toxicity scores produce a score >= threshold."""
    evaluator._model.predict.return_value = {"toxicity": 0.95, "threat": 0.88}

    payload = EvalPayload(input_text="Hello", output_text="[toxic content]")
    result = await evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score >= evaluator.threshold()
    assert result.metadata is not None
    assert "dimensions" in result.metadata


@pytest.mark.asyncio
async def test_model_error_returns_fail_open(evaluator):
    """If the model raises, evaluate() returns score=None, flag=False (fail-open)."""
    evaluator._model.predict.side_effect = RuntimeError("model crashed")

    payload = EvalPayload(input_text="Hello", output_text="Some output")
    result = await evaluator.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert result.error is not None
    assert "model crashed" in result.error


@pytest.mark.asyncio
async def test_skipped_when_no_output_text(evaluator):
    """Output evaluators return a skipped result when output_text is None."""
    payload = EvalPayload(input_text="Hello", output_text=None)
    result = await evaluator.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert result.metadata == {"skipped": "no output_text"}
    # Model should not have been called
    evaluator._model.predict.assert_not_called()


@pytest.mark.asyncio
async def test_flag_on_all_uses_min_score(evaluator):
    """flag_on=all aggregates via min (only flag if ALL dimensions are high)."""
    evaluator.config["flag_on"] = "all"
    evaluator._model.predict.return_value = {"toxicity": 0.95, "threat": 0.10}

    payload = EvalPayload(input_text="Hello", output_text="text")
    result = await evaluator.evaluate(payload)

    # min(0.95, 0.10) = 0.10 — below threshold, would not flag
    assert result.score == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_flag_on_any_uses_max_score(evaluator):
    """flag_on=any aggregates via max (flag if ANY dimension is high)."""
    evaluator.config["flag_on"] = "any"
    evaluator._model.predict.return_value = {"toxicity": 0.95, "threat": 0.10}

    payload = EvalPayload(input_text="Hello", output_text="text")
    result = await evaluator.evaluate(payload)

    # max(0.95, 0.10) = 0.95 — above threshold, would flag
    assert result.score == pytest.approx(0.95)
