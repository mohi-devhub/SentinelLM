"""Hallucination evaluator — scores contradiction against context documents.

Two backends, selected by `evaluators.hallucination.backend`:

  consistency (default) — vectara/hallucination_evaluation_model, a model
    purpose-built for factual-consistency detection. Benchmarked on this
    project against HaluEval (see README § Evaluator Accuracy): AUC 0.78,
    ~77% accuracy at threshold 0.50 — a large, measured improvement over the
    nli backend below. Requires trust_remote_code=True (see
    _consistency_model.py for what that means and why it's still the
    default).

  nli — a generic cross-encoder NLI model (default
    cross-encoder/nli-deberta-v3-base). For each (context_doc, output) pair,
    scores contradiction probability. Benchmarked at AUC 0.59 — real but
    weak signal; kept available for environments where running vendor
    remote code is not acceptable.

Either way, the worst-case (maximum) contradiction score across all context
documents is used as the final score.

Score interpretation:
    1.0 → output strongly contradicts / is unsupported by context (likely hallucination)
    0.0 → output does not contradict context

flag_direction = 'above': flag when score > threshold.

This evaluator is skipped (score = None) when no context_documents are provided,
since contradiction cannot be assessed without a ground-truth context.

Device / backend note (nli only):
    MPS backend is not fully supported for cross-encoder inference. The device
    setting is respected but 'auto' always resolves to 'cpu' here.

    When use_onnx: true is set in config, the evaluator uses ONNX Runtime
    instead of PyTorch for ~3–5x faster CPU inference. On first startup the
    model is auto-exported to ONNX and cached; subsequent starts load the
    cached graph. Falls back to CrossEncoder if ONNX loading fails. Only
    applies to the nli backend — the consistency backend's custom model
    class isn't ONNX-exportable through this path.
"""

from __future__ import annotations

import logging

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor

logger = logging.getLogger(__name__)

_DEFAULT_CONSISTENCY_MODEL = "vectara/hallucination_evaluation_model"
_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def _get_label_index(model, label_name: str, fallback: int) -> int:
    """Look up the output index for a given NLI label from the model's config."""
    id2label: dict = getattr(model.model.config, "id2label", {})
    for idx, label in id2label.items():
        if label.lower() == label_name.lower():
            return int(idx)
    return fallback


class HallucinationEvaluator(BaseEvaluator):
    """Detects hallucinations by scoring contradiction against context documents.

    Config keys (under evaluators.hallucination in config.yaml):
        threshold (float): Score above which output is flagged. Default 0.50
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

    name = "hallucination"
    runs_on = "output"
    flag_direction = "above"

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
            # low_cpu_mem_usage=False: newer transformers defaults to loading
            # weights onto a meta device first, which some sentence-transformers
            # CrossEncoder versions then fail to materialize ("Cannot copy out
            # of meta tensor; no data!") when .to(device) runs. Forcing eager
            # loading avoids it — confirmed against transformers 4.57/torch 2.10.
            self._model = CrossEncoder(
                model_id, device=device, automodel_args={"low_cpu_mem_usage": False}
            )

        self._contradiction_idx: int = _get_label_index(self._model, "contradiction", fallback=0)

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
                # Hallucination score is the inverse of consistency: low
                # consistency (unsupported/contradicted) => high hallucination score.
                contradiction_scores = [1.0 - c for c in consistency_scores]
                return max(contradiction_scores), contradiction_scores

            max_score, per_doc_scores = await run_in_executor(_score_consistency, output, docs)
            return max_score, {
                "per_doc_contradiction": per_doc_scores,
                "num_docs": len(docs),
                "backend": "consistency",
            }

        def _score_nli(out: str, context_docs: list[str]) -> tuple[float, list[float]]:
            # Pairs: (premise=context_doc, hypothesis=LLM_output)
            pairs = [(doc, out) for doc in context_docs]
            raw_scores = self._model.predict(pairs, apply_softmax=True)
            # raw_scores shape: (n_docs, n_labels)
            contradiction_scores = [float(row[self._contradiction_idx]) for row in raw_scores]
            return max(contradiction_scores), contradiction_scores

        max_score, per_doc_scores = await run_in_executor(_score_nli, output, docs)
        return max_score, {
            "per_doc_contradiction": per_doc_scores,
            "num_docs": len(docs),
            "backend": "nli",
        }
