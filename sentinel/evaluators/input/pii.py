"""PII evaluator using Microsoft Presidio + spaCy.

Detects personally identifiable information in user input. Supports two actions:
  redact — strips PII entities from the text; the proxy substitutes the
           redacted version before forwarding to the LLM.
  block  — returns a high score so the chain runner rejects the request.

Score = max Presidio confidence across all entities that meet the threshold.
A score of 0.0 means no PII was detected above the confidence threshold.

spaCy model must be downloaded before first use:
    python -m spacy download en_core_web_lg
"""
from __future__ import annotations

from typing import Optional

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor


class PIIEvaluator(BaseEvaluator):
    """Detects PII in user input using Microsoft Presidio + spaCy en_core_web_lg.

    Config keys (under evaluators.pii in config.yaml):
        threshold (float): Minimum Presidio confidence to count an entity. Default 0.5.
        action (str):      'redact' | 'block'. Default 'block'.
    """

    name = "pii"
    runs_on = "input"
    flag_direction = "above"

    def _load_model(self) -> None:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
        from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        # Satisfy BaseEvaluator._model convention (used by tests)
        self._model = self._analyzer

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, Optional[dict]]:
        text = payload.input_text
        threshold: float = float(self.config.get("threshold", 0.5))
        action: str = self.config.get("action", "block")

        def _analyze(t: str) -> list:
            return self._analyzer.analyze(text=t, language="en")

        results = await run_in_executor(_analyze, text)

        # Keep only entities that meet the configured confidence threshold
        detected = [r for r in results if r.score >= threshold]

        if not detected:
            return 0.0, {"entities": [], "action": action}

        score = max(r.score for r in detected)
        entities = [
            {
                "type": r.entity_type,
                "score": r.score,
                "start": r.start,
                "end": r.end,
            }
            for r in detected
        ]

        metadata: dict = {"entities": entities, "action": action}

        if action == "redact":
            def _anonymize(t: str, res: list) -> str:
                return self._anonymizer.anonymize(text=t, analyzer_results=res).text

            metadata["redacted_text"] = await run_in_executor(_anonymize, text, detected)

        return score, metadata
