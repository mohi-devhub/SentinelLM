"""Hallucination evaluator using a cross-encoder NLI model.

Detects when the LLM's response contradicts the provided context documents.
For each (context_doc, output) pair, the NLI model assigns a contradiction
probability. The worst-case (maximum) contradiction score across all documents
is used as the final score.

Score interpretation:
    1.0 → output strongly contradicts context (likely hallucination)
    0.0 → output does not contradict context

flag_direction = 'above': flag when score > threshold.

This evaluator is skipped (score = None) when no context_documents are provided,
since contradiction cannot be assessed without a ground-truth context.

Device note:
    MPS backend is not fully supported for cross-encoder inference. The device
    setting is respected but 'auto' always resolves to 'cpu' here.
"""
from __future__ import annotations

from typing import Optional

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor


def _get_label_index(model, label_name: str, fallback: int) -> int:
    """Look up the output index for a given NLI label from the model's config."""
    id2label: dict = getattr(model.model.config, "id2label", {})
    for idx, label in id2label.items():
        if label.lower() == label_name.lower():
            return int(idx)
    return fallback


class HallucinationEvaluator(BaseEvaluator):
    """Detects hallucinations by scoring contradiction against context documents.

    Uses a cross-encoder NLI model (premise=context, hypothesis=output) to
    compute contradiction probability. A high score means the output contradicts
    the provided context.

    Config keys (under evaluators.hallucination in config.yaml):
        threshold (float): Contradiction score above which output is flagged. Default 0.70.
        model (str):       HuggingFace cross-encoder model ID.
                           Default 'cross-encoder/nli-deberta-v3-base'.
        device (str):      'auto' | 'cpu'. Default 'auto' (resolves to 'cpu').
    """

    name = "hallucination"
    runs_on = "output"
    flag_direction = "above"

    def _load_model(self) -> None:
        from sentence_transformers.cross_encoder import CrossEncoder  # noqa: PLC0415

        model_id: str = self.config.get("model", "cross-encoder/nli-deberta-v3-base")

        # MPS is not fully supported for cross-encoder inference; always use CPU.
        device: str = self.config.get("device", "auto")
        if device == "auto":
            device = "cpu"

        self._model = CrossEncoder(model_id, device=device)
        self._contradiction_idx: int = _get_label_index(self._model, "contradiction", fallback=0)

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, Optional[dict]]:
        output = payload.output_text  # guaranteed non-None by BaseEvaluator.evaluate()
        docs = payload.context_documents  # guaranteed non-None by BaseEvaluator.evaluate()

        def _score(out: str, context_docs: list[str]) -> tuple[float, list[float]]:
            # Pairs: (premise=context_doc, hypothesis=LLM_output)
            pairs = [(doc, out) for doc in context_docs]
            raw_scores = self._model.predict(pairs, apply_softmax=True)
            # raw_scores shape: (n_docs, n_labels)
            contradiction_scores = [float(row[self._contradiction_idx]) for row in raw_scores]
            return max(contradiction_scores), contradiction_scores

        max_score, per_doc_scores = await run_in_executor(_score, output, docs)
        return max_score, {
            "per_doc_contradiction": per_doc_scores,
            "num_docs": len(docs),
        }
