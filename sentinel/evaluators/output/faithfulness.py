"""Faithfulness evaluator — scores support (entailment) from context documents.

Two backends, selected by `evaluators.faithfulness.backend`:

  consistency (default) — vectara/hallucination_evaluation_model, the same
    purpose-built factual-consistency model used by hallucination.py. Its
    raw output IS the faithfulness score directly (high = grounded). See
    hallucination.py's module docstring and README § Evaluator Accuracy for
    the benchmarked numbers and the trust_remote_code trade-off.

  nli — a generic cross-encoder NLI model (default
    cross-encoder/nli-deberta-v3-base). For each (context_doc, output) pair,
    scores entailment probability. Kept available for environments where
    running vendor remote code is not acceptable.

Either way, the best-case (maximum) entailment/consistency score across all
context documents is used as the final score.

Score interpretation:
    1.0 → output is well-supported by context (faithful)
    0.0 → output is not supported by any context document (unfaithful)

flag_direction = 'below': flag when score < threshold (output not grounded in context).

This evaluator is skipped (score = None) when no context_documents are provided.

Note: uses the same model as the hallucination evaluator when backend is
'consistency' — the two evaluators just read opposite ends of the same
signal. With backend 'nli' they load separate model instances.

Device / backend note (nli only):
    MPS backend is not fully supported for cross-encoder inference. The device
    setting is respected but 'auto' always resolves to 'cpu' here.

    When use_onnx: true is set in config, the evaluator uses ONNX Runtime
    instead of PyTorch for ~3–5x faster CPU inference. Falls back to
    CrossEncoder if ONNX loading fails. Only applies to the nli backend.
"""

from __future__ import annotations

import logging

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor
from sentinel.evaluators.output.hallucination import _get_label_index

logger = logging.getLogger(__name__)

_DEFAULT_CONSISTENCY_MODEL = "vectara/hallucination_evaluation_model"
_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


class FaithfulnessEvaluator(BaseEvaluator):
    """Scores how well the LLM output is supported by context documents.

    Config keys (under evaluators.faithfulness in config.yaml):
        threshold (float): Score below which output is flagged. Default 0.50
                           (consistency backend) — see module docstring.
        backend (str):     'consistency' | 'nli'. Default 'consistency'.
        model (str):       HuggingFace model ID. Defaults to the backend's
                           benchmarked default model — only override if you've
                           measured the alternative yourself.
        device (str):      nli backend only. 'auto' | 'cpu'. Default 'auto'
                           (resolves to 'cpu'). Ignored when use_onnx is true.
        use_onnx (bool):   nli backend only. Use ONNX Runtime for inference
                           (~3–5x faster). Default False. Requires onnxruntime
                           and optimum to be installed.
    """

    name = "faithfulness"
    runs_on = "output"
    flag_direction = "below"

    def _load_model(self) -> None:
        self._backend: str = self.config.get("backend", "consistency")

        if self._backend == "consistency":
            from sentinel.evaluators.output._consistency_model import (  # noqa: PLC0415
                load_consistency_model,
            )

            model_id: str = self.config.get("model", _DEFAULT_CONSISTENCY_MODEL)
            self._model = load_consistency_model(model_id)
            return

        # ── nli backend ──────────────────────────────────────────────────────
        model_id = self.config.get("model", _DEFAULT_NLI_MODEL)
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
            self._model = CrossEncoder(
                model_id, device=device, automodel_args={"low_cpu_mem_usage": False}
            )

        self._entailment_idx: int = _get_label_index(self._model, "entailment", fallback=1)

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        output = payload.output_text  # guaranteed non-None by BaseEvaluator.evaluate()
        docs = payload.context_documents
        assert docs is not None  # guaranteed by BaseEvaluator.evaluate()

        if self._backend == "consistency":
            from sentinel.evaluators.output._consistency_model import (  # noqa: PLC0415
                score_consistency,
            )

            def _score_consistency(out: str, context_docs: list[str]) -> tuple[float, list[float]]:
                consistency_scores = score_consistency(self._model, context_docs, out)
                return max(consistency_scores), consistency_scores

            max_score, per_doc_scores = await run_in_executor(_score_consistency, output, docs)
            return max_score, {
                "per_doc_entailment": per_doc_scores,
                "num_docs": len(docs),
                "backend": "consistency",
            }

        def _score_nli(out: str, context_docs: list[str]) -> tuple[float, list[float]]:
            # Pairs: (premise=context_doc, hypothesis=LLM_output)
            pairs = [(doc, out) for doc in context_docs]
            raw_scores = self._model.predict(pairs, apply_softmax=True)
            # raw_scores shape: (n_docs, n_labels)
            # Take the best-case entailment across all docs: if any doc supports
            # the output, the output is considered faithful to that extent.
            entailment_scores = [float(row[self._entailment_idx]) for row in raw_scores]
            return max(entailment_scores), entailment_scores

        max_score, per_doc_scores = await run_in_executor(_score_nli, output, docs)
        return max_score, {
            "per_doc_entailment": per_doc_scores,
            "num_docs": len(docs),
            "backend": "nli",
        }
