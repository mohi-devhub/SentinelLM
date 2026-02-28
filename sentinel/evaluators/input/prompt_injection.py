"""Prompt injection evaluator using a HuggingFace text-classification pipeline.

Scores user input for prompt injection attempts using the
deepset/deberta-v3-base-injection model (binary classifier: INJECTION vs LEGITIMATE).

Score = probability of the INJECTION class (0.0–1.0).
A score above the configured threshold triggers a block.

Device selection:
  device: auto  — MPS if available on Apple Silicon, else CPU
  device: cpu   — force CPU
  device: mps   — force MPS (not recommended; use auto)
"""

from __future__ import annotations

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor


def _resolve_device(device_cfg: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch  # noqa: PLC0415

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class PromptInjectionEvaluator(BaseEvaluator):
    """Detects prompt injection attacks in user input.

    Uses a HuggingFace text-classification pipeline trained on injection examples.
    The score is the model's predicted probability for the INJECTION class.

    Config keys (under evaluators.prompt_injection in config.yaml):
        threshold (float): Injection probability above which the request is blocked. Default 0.80.
        model (str):       HuggingFace model ID. Default 'deepset/deberta-v3-base-injection'.
        device (str):      'auto' | 'cpu' | 'mps'. Default 'auto'.
    """

    name = "prompt_injection"
    runs_on = "input"
    flag_direction = "above"

    def _load_model(self) -> None:
        from transformers import pipeline  # noqa: PLC0415

        model_id: str = self.config.get("model", "deepset/deberta-v3-base-injection")
        device: str = _resolve_device(self.config.get("device", "auto"))

        # top_k=None returns scores for all labels so we can always find INJECTION
        self._model = pipeline("text-classification", model=model_id, device=device, top_k=None)

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        text = payload.input_text

        def _predict(t: str) -> list:
            return self._model(t)

        raw = await run_in_executor(_predict, text)

        # pipeline(str, top_k=None) → list[dict] for a single string
        # pipeline([str], top_k=None) → list[list[dict]] for a batch
        # Normalise to a flat list of {"label": ..., "score": ...} dicts
        if raw and isinstance(raw[0], list):
            label_scores: list[dict] = raw[0]
        else:
            label_scores = raw

        # Extract the INJECTION class probability
        injection_score = 0.0
        for item in label_scores:
            if item["label"].upper() == "INJECTION":
                injection_score = float(item["score"])
                break

        return injection_score, {"labels": label_scores}
