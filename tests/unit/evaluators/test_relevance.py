"""Unit tests for RelevanceEvaluator.

All tests mock sentence_transformers — no real model is loaded or called.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from sentinel.evaluators.base import EvalPayload

MOCK_CONFIG = {
    "evaluators": {
        "relevance": {
            "enabled": True,
            "threshold": 0.30,
        }
    }
}


def _make_mock_st(cosine_value: float):
    """Return a fake sentence_transformers module whose cos_sim returns cosine_value."""
    mock_tensor = MagicMock()
    mock_tensor.item.return_value = cosine_value

    mock_util = MagicMock()
    mock_util.cos_sim.return_value = mock_tensor

    mock_model = MagicMock()
    # encode() returns a mock tensor; the actual value doesn't matter since
    # cos_sim is also mocked.
    mock_model.encode.return_value = MagicMock()

    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model
    mock_st_module.util = mock_util

    return mock_st_module, mock_model, mock_util


@pytest.fixture
def evaluator():
    """RelevanceEvaluator with sentence_transformers replaced by a MagicMock."""
    mock_st, mock_model, _ = _make_mock_st(cosine_value=0.9)

    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_st
    try:
        from sentinel.evaluators.output.relevance import RelevanceEvaluator  # noqa: PLC0415

        ev = RelevanceEvaluator(config=MOCK_CONFIG)
        ev._model = mock_model
        yield ev
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original


# ── Basic score tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_relevant_output_scores_above_threshold(evaluator):
    """High cosine similarity → score above threshold → no flag."""
    with patch("sentinel.evaluators.output.relevance.run_in_executor") as mock_exec:
        mock_exec.return_value = 0.85

        payload = EvalPayload(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
        )
        result = await evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score > evaluator.threshold()
    assert result.flag is False  # runner sets flag; evaluate() always returns False
    assert result.metadata == {"cosine_similarity": result.score}


@pytest.mark.asyncio
async def test_irrelevant_output_scores_below_threshold(evaluator):
    """Low cosine similarity → score below threshold."""
    with patch("sentinel.evaluators.output.relevance.run_in_executor") as mock_exec:
        mock_exec.return_value = 0.05

        payload = EvalPayload(
            input_text="What is the capital of France?",
            output_text="I like pizza on Fridays.",
        )
        result = await evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score < evaluator.threshold()


@pytest.mark.asyncio
async def test_score_clamped_to_zero_for_negative_cosine(evaluator):
    """Negative cosine similarity is clamped to 0.0 (never returns negative)."""
    with patch("sentinel.evaluators.output.relevance.run_in_executor") as mock_exec:
        mock_exec.return_value = 0.0  # clamped from negative

        payload = EvalPayload(input_text="hello", output_text="completely unrelated")
        result = await evaluator.evaluate(payload)

    assert result.score == pytest.approx(0.0)
    assert result.error is None


# ── Edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skipped_when_no_output_text(evaluator):
    """Output evaluators skip when output_text is None."""
    payload = EvalPayload(input_text="Hello", output_text=None)
    result = await evaluator.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert result.metadata == {"skipped": "no output_text"}


@pytest.mark.asyncio
async def test_model_error_returns_fail_open(evaluator):
    """If run_in_executor raises, evaluate() returns score=None, flag=False."""
    with patch("sentinel.evaluators.output.relevance.run_in_executor") as mock_exec:
        mock_exec.side_effect = RuntimeError("encoder crashed")

        payload = EvalPayload(input_text="hello", output_text="world")
        result = await evaluator.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert result.error is not None
    assert "encoder crashed" in result.error


@pytest.mark.asyncio
async def test_metadata_contains_cosine_similarity(evaluator):
    """Result metadata always includes cosine_similarity key."""
    with patch("sentinel.evaluators.output.relevance.run_in_executor") as mock_exec:
        mock_exec.return_value = 0.72

        payload = EvalPayload(input_text="q", output_text="a")
        result = await evaluator.evaluate(payload)

    assert result.metadata is not None
    assert "cosine_similarity" in result.metadata
    assert result.metadata["cosine_similarity"] == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_evaluator_name_and_runs_on():
    """Sanity check on class-level attributes."""
    from sentinel.evaluators.output.relevance import RelevanceEvaluator  # noqa: PLC0415

    assert RelevanceEvaluator.name == "relevance"
    assert RelevanceEvaluator.runs_on == "output"
    assert RelevanceEvaluator.flag_direction == "below"


# ── Inner _score function coverage (no run_in_executor mock) ──────────────────


@pytest.mark.asyncio
async def test_relevance_inner_score_executes(evaluator):
    """Let run_in_executor actually call _score to exercise the inner closure.

    We patch sentence_transformers in sys.modules so the deferred import in
    _score picks up our mock.  The mock cos_sim returns 0.82.
    """
    import sys

    mock_cos_tensor = MagicMock()
    mock_cos_tensor.item.return_value = 0.82

    mock_util = MagicMock()
    mock_util.cos_sim.return_value = mock_cos_tensor

    mock_st = MagicMock()
    mock_st.util = mock_util

    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_st

    evaluator._model.encode.return_value = [MagicMock(), MagicMock()]

    try:
        payload = EvalPayload(
            input_text="What is 2+2?",
            output_text="The answer is 4.",
        )
        result = await evaluator.evaluate(payload)
        assert result.error is None
        assert result.score is not None
        assert result.score >= 0.0
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original


@pytest.mark.asyncio
async def test_relevance_inner_score_clamps_negative(evaluator):
    """Negative cosine similarity is clamped to 0.0 in the inner _score function."""
    import sys

    mock_cos_tensor = MagicMock()
    mock_cos_tensor.item.return_value = -0.3  # negative → clamped to 0.0

    mock_util = MagicMock()
    mock_util.cos_sim.return_value = mock_cos_tensor

    mock_st = MagicMock()
    mock_st.util = mock_util

    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_st

    evaluator._model.encode.return_value = [MagicMock(), MagicMock()]

    try:
        payload = EvalPayload(input_text="irrelevant question", output_text="unrelated answer")
        result = await evaluator.evaluate(payload)
        assert result.error is None
        assert result.score is not None
        assert result.score == pytest.approx(0.0)  # clamped from -0.3
    finally:
        if original is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = original
