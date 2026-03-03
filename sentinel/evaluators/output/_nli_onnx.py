"""ONNX Runtime-accelerated NLI cross-encoder inference.

Replaces sentence_transformers.CrossEncoder with ONNX Runtime inference for
~3–5x speedup on CPU, targeting the PRD's p95 hallucination eval < 80ms goal.

The first call to OnnxNliCrossEncoder() exports the model to ONNX (via
optimum's export=True flag) and caches it in the HuggingFace cache dir.
Subsequent runs load the cached ONNX graph, skipping export entirely.

Requires: onnxruntime, optimum[exporters] (see requirements.txt).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OnnxNliCrossEncoder:
    """ONNX Runtime wrapper for cross-encoder NLI sequence classification.

    Provides the same `.predict(pairs, apply_softmax)` interface as
    ``sentence_transformers.CrossEncoder``, and exposes a ``.model`` attribute
    whose ``.config.id2label`` dict is compatible with the shared
    ``_get_label_index()`` helper used by both NLI evaluators.

    Args:
        model_id: HuggingFace model ID (e.g. 'cross-encoder/nli-deberta-v3-base').
    """

    def __init__(self, model_id: str) -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        logger.info("Loading NLI model via ONNX Runtime: %s", model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # export=True: auto-export PyTorch weights → ONNX on first run, then cache.
        # CPUExecutionProvider: MPS is not supported by onnxruntime on macOS.
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_id,
            export=True,
            provider="CPUExecutionProvider",
        )

        logger.info("ONNX NLI model ready: %s", model_id)

    def predict(
        self,
        pairs: list[tuple[str, str]],
        apply_softmax: bool = True,
    ) -> list[list[float]]:
        """Score NLI pairs and return per-label probabilities.

        Args:
            pairs: List of (premise, hypothesis) string tuples.
            apply_softmax: If True, apply softmax over the raw logits.

        Returns:
            List of score lists, one per pair. Length equals the number of
            NLI labels in the model (typically 3: contradiction/entailment/neutral).
        """
        import torch.nn.functional as F  # noqa: PLC0415

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]

        inputs = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        outputs = self.model(**inputs)
        logits = outputs.logits  # shape: (n_pairs, n_labels)

        if apply_softmax:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = logits

        return probs.detach().tolist()
