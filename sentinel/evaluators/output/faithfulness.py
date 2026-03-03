"""Faithfulness evaluator using a cross-encoder NLI model.

Measures whether the LLM's response is supported (entailed) by the provided
context documents. For each (context_doc, output) pair, the NLI model assigns
an entailment probability. The best-case (maximum) entailment score across all
documents is used as the final score.

Score interpretation:
    1.0 → output is well-supported by context (faithful)
    0.0 → output is not supported by any context document (unfaithful)

flag_direction = 'below': flag when score < threshold (output not grounded in context).

This evaluator is skipped (score = None) when no context_documents are provided.

Note: uses the same model as the hallucination evaluator but scores the
entailment dimension instead of contradiction. Both evaluators load separate
model instances.

Device / backend note:
    MPS backend is not fully supported for cross-encoder inference. The device
    setting is respected but 'auto' always resolves to 'cpu' here.

    When use_onnx: true is set in config, the evaluator uses ONNX Runtime
    instead of PyTorch for ~3–5x faster CPU inference. Falls back to
    CrossEncoder if ONNX loading fails.
"""

from __future__ import annotations

import logging

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor
from sentinel.evaluators.output.hallucination import _get_label_index

logger = logging.getLogger(__name__)


class FaithfulnessEvaluator(BaseEvaluator):
    """Scores how well the LLM output is supported by context documents.

    Uses a cross-encoder NLI model (premise=context, hypothesis=output) to
    compute the probability that the context entails the output. A low score
    means the output makes claims not grounded in the provided context.

    Config keys (under evaluators.faithfulness in config.yaml):
        threshold (float): Entailment score below which output is flagged. Default 0.70.
        model (str):       HuggingFace cross-encoder model ID.
                           Default 'cross-encoder/nli-deberta-v3-base'.
        device (str):      'auto' | 'cpu'. Default 'auto' (resolves to 'cpu').
                           Ignored when use_onnx is true (ONNX always runs on CPU).
        use_onnx (bool):   Use ONNX Runtime for inference (~3–5x faster). Default False.
                           Requires onnxruntime and optimum to be installed.
    """

    name = "faithfulness"
    runs_on = "output"
    flag_direction = "below"

    def _load_model(self) -> None:
        model_id: str = self.config.get("model", "cross-encoder/nli-deberta-v3-base")
        use_onnx: bool = self.config.get("use_onnx", False)

        if use_onnx:
            try:
                from sentinel.evaluators.output._nli_onnx import (
                    OnnxNliCrossEncoder,  # noqa: PLC0415
                )

                self._model = OnnxNliCrossEncoder(model_id)
            except Exception:
                logger.warning(
                    "ONNX load failed for %s; falling back to CrossEncoder", model_id, exc_info=True
                )
                use_onnx = False

        if not use_onnx:
            from sentence_transformers.cross_encoder import CrossEncoder  # noqa: PLC0415

            # MPS is not fully supported for cross-encoder inference; always use CPU.
            device: str = self.config.get("device", "auto")
            if device == "auto":
                device = "cpu"
            self._model = CrossEncoder(model_id, device=device)

        self._entailment_idx: int = _get_label_index(self._model, "entailment", fallback=1)

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        output = payload.output_text  # guaranteed non-None by BaseEvaluator.evaluate()
        docs = payload.context_documents
        assert docs is not None  # guaranteed by BaseEvaluator.evaluate()

        def _score(out: str, context_docs: list[str]) -> tuple[float, list[float]]:
            # Pairs: (premise=context_doc, hypothesis=LLM_output)
            pairs = [(doc, out) for doc in context_docs]
            raw_scores = self._model.predict(pairs, apply_softmax=True)
            # raw_scores shape: (n_docs, n_labels)
            # Take the best-case entailment across all docs: if any doc supports
            # the output, the output is considered faithful to that extent.
            entailment_scores = [float(row[self._entailment_idx]) for row in raw_scores]
            return max(entailment_scores), entailment_scores

        max_score, per_doc_scores = await run_in_executor(_score, output, docs)
        return max_score, {
            "per_doc_entailment": per_doc_scores,
            "num_docs": len(docs),
        }
