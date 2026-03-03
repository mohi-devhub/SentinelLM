"""Unit tests for OnnxNliCrossEncoder and the use_onnx evaluator paths.

All tests mock out ORTModelForSequenceClassification and AutoTokenizer — no real
models or the optimum package are required.  The key techniques used:

* OnnxNliCrossEncoder.predict() tests: bypass __init__ via object.__new__() and
  set the required attributes directly on the instance.
* Evaluator _load_model() tests: patch the class at the _nli_onnx module level so
  the local `from ... import OnnxNliCrossEncoder` inside _load_model() picks up
  the mock without needing optimum to be installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import sentinel.evaluators.output._nli_onnx as _nli_onnx_mod
from sentinel.evaluators.base import EvalPayload

# ── Shared helpers ─────────────────────────────────────────────────────────────

# NLI label mapping used by nli-deberta-v3-base
_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}

_ONNX_CONFIG_HALLUCINATION = {
    "evaluators": {
        "hallucination": {
            "enabled": True,
            "threshold": 0.70,
            "model": "cross-encoder/nli-deberta-v3-base",
            "use_onnx": True,
        }
    }
}

_ONNX_CONFIG_FAITHFULNESS = {
    "evaluators": {
        "faithfulness": {
            "enabled": True,
            "threshold": 0.70,
            "model": "cross-encoder/nli-deberta-v3-base",
            "use_onnx": True,
        }
    }
}


def _make_mock_ort_model(logits: torch.Tensor) -> MagicMock:
    """Return a mock ORT model whose __call__ produces the given logits."""
    mock_output = MagicMock()
    mock_output.logits = logits

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.config = MagicMock()
    mock_model.config.id2label = _ID2LABEL
    return mock_model


def _make_mock_tokenizer() -> MagicMock:
    """Return a mock tokenizer that produces dummy PT tensors."""
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    return mock_tok


def _make_onnx_encoder(logits: torch.Tensor) -> _nli_onnx_mod.OnnxNliCrossEncoder:
    """Instantiate OnnxNliCrossEncoder WITHOUT calling __init__.

    Uses object.__new__ so optimum does not need to be installed.
    Sets the two attributes that predict() uses: .model and ._tokenizer.
    """
    enc: _nli_onnx_mod.OnnxNliCrossEncoder = object.__new__(_nli_onnx_mod.OnnxNliCrossEncoder)
    enc.model = _make_mock_ort_model(logits)
    enc._tokenizer = _make_mock_tokenizer()
    return enc


# ── OnnxNliCrossEncoder.predict() unit tests ───────────────────────────────────


def test_onnx_encoder_predict_apply_softmax():
    """predict(apply_softmax=True) returns probabilities that sum to 1."""
    enc = _make_onnx_encoder(torch.tensor([[2.0, 0.5, -1.0]]))

    result = enc.predict([("premise", "hypothesis")], apply_softmax=True)

    assert len(result) == 1
    assert len(result[0]) == 3
    assert abs(sum(result[0]) - 1.0) < 1e-5


def test_onnx_encoder_predict_no_softmax():
    """predict(apply_softmax=False) returns raw logit values unchanged."""
    enc = _make_onnx_encoder(torch.tensor([[1.5, -0.5, 0.3]]))

    result = enc.predict([("premise", "hypothesis")], apply_softmax=False)

    assert len(result) == 1
    assert abs(result[0][0] - 1.5) < 1e-4
    assert abs(result[0][1] - (-0.5)) < 1e-4


def test_onnx_encoder_predict_multiple_pairs():
    """predict() handles a batch of multiple pairs, all rows sum to 1."""
    enc = _make_onnx_encoder(
        torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
            ]
        )
    )

    result = enc.predict(
        [("doc1", "output"), ("doc2", "output")],
        apply_softmax=True,
    )

    assert len(result) == 2
    for row in result:
        assert abs(sum(row) - 1.0) < 1e-5


def test_onnx_encoder_model_config_id2label_accessible():
    """model.config.id2label is reachable via _get_label_index's access pattern."""
    enc = _make_onnx_encoder(torch.tensor([[0.5, 0.3, 0.2]]))

    # _get_label_index accesses model.model.config.id2label;
    # for OnnxNliCrossEncoder, 'model' is the ORTModel so .model.config.id2label works.
    assert enc.model.config.id2label[0] == "contradiction"
    assert enc.model.config.id2label[1] == "entailment"


# ── HallucinationEvaluator with use_onnx=True ─────────────────────────────────


def _make_onnx_encoder_mock() -> tuple[MagicMock, MagicMock]:
    """Return (encoder_instance_mock, ort_model_mock) for evaluator fixtures."""
    mock_ort = _make_mock_ort_model(torch.tensor([[0.05, 0.90, 0.05]]))
    mock_encoder = MagicMock(spec=_nli_onnx_mod.OnnxNliCrossEncoder)
    mock_encoder.model = mock_ort
    mock_encoder._tokenizer = _make_mock_tokenizer()
    return mock_encoder, mock_ort


@pytest.fixture
def hallucination_onnx_evaluator():
    """HallucinationEvaluator with ONNX backend replaced by a MagicMock."""
    mock_encoder, mock_ort = _make_onnx_encoder_mock()

    # Patch OnnxNliCrossEncoder at the _nli_onnx module level.
    # _load_model does `from sentinel.evaluators.output._nli_onnx import OnnxNliCrossEncoder`
    # which reads the attribute from the already-imported module object — so patching
    # the attribute on the module IS sufficient without any sys.modules manipulation.
    with patch.object(_nli_onnx_mod, "OnnxNliCrossEncoder", return_value=mock_encoder):
        from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

        ev = HallucinationEvaluator(config=_ONNX_CONFIG_HALLUCINATION)
        yield ev


@pytest.mark.asyncio
async def test_hallucination_onnx_low_contradiction(hallucination_onnx_evaluator):
    """ONNX-backed hallucination evaluator scores correctly and returns metadata."""
    with patch("sentinel.evaluators.output.hallucination.run_in_executor") as mock_exec:
        mock_exec.return_value = (0.04, [0.04])

        payload = EvalPayload(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            context_documents=["Paris is the capital of France."],
        )
        result = await hallucination_onnx_evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score < hallucination_onnx_evaluator.threshold()
    assert result.metadata is not None
    assert "per_doc_contradiction" in result.metadata


def test_hallucination_onnx_fallback_on_init_error():
    """When OnnxNliCrossEncoder raises, _load_model falls back to CrossEncoder."""
    mock_ce = MagicMock()
    mock_ce.model = MagicMock()
    mock_ce.model.config = MagicMock()
    mock_ce.model.config.id2label = _ID2LABEL

    with (
        patch.object(
            _nli_onnx_mod.OnnxNliCrossEncoder,
            "__init__",
            side_effect=RuntimeError("ort unavailable"),
        ),
        patch("sentence_transformers.cross_encoder.CrossEncoder", return_value=mock_ce),
    ):
        from sentinel.evaluators.output.hallucination import HallucinationEvaluator  # noqa: PLC0415

        ev = HallucinationEvaluator(config=_ONNX_CONFIG_HALLUCINATION)

    # The model should be the CrossEncoder mock, not an OnnxNliCrossEncoder
    assert not isinstance(ev._model, _nli_onnx_mod.OnnxNliCrossEncoder)


# ── FaithfulnessEvaluator with use_onnx=True ──────────────────────────────────


@pytest.fixture
def faithfulness_onnx_evaluator():
    """FaithfulnessEvaluator with ONNX backend replaced by a MagicMock."""
    mock_encoder, mock_ort = _make_onnx_encoder_mock()

    with patch.object(_nli_onnx_mod, "OnnxNliCrossEncoder", return_value=mock_encoder):
        from sentinel.evaluators.output.faithfulness import FaithfulnessEvaluator  # noqa: PLC0415

        ev = FaithfulnessEvaluator(config=_ONNX_CONFIG_FAITHFULNESS)
        yield ev


@pytest.mark.asyncio
async def test_faithfulness_onnx_high_entailment(faithfulness_onnx_evaluator):
    """ONNX-backed faithfulness evaluator returns entailment score correctly."""
    with patch("sentinel.evaluators.output.faithfulness.run_in_executor") as mock_exec:
        mock_exec.return_value = (0.90, [0.90])

        payload = EvalPayload(
            input_text="Who founded Apple?",
            output_text="Apple was founded by Steve Jobs.",
            context_documents=["Apple was co-founded by Steve Jobs in 1976."],
        )
        result = await faithfulness_onnx_evaluator.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score > faithfulness_onnx_evaluator.threshold()
    assert result.metadata is not None
    assert "per_doc_entailment" in result.metadata
