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

    # Entity types considered genuinely sensitive PII.
    # Excludes LOCATION (country/city names are not private) and other
    # non-sensitive NER categories that Presidio detects by default.
    _SENSITIVE_ENTITY_TYPES: frozenset[str] = frozenset({
        "PERSON",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "US_SSN",
        "US_BANK_NUMBER",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "IP_ADDRESS",
        "IBAN_CODE",
        "MEDICAL_LICENSE",
        "DATE_TIME",
        "NRP",
        "URL",
    })

    def _load_model(self) -> None:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
        from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        # Satisfy BaseEvaluator._model convention (used by tests)
        self._model = self._analyzer

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        text = payload.input_text
        threshold: float = float(self.config.get("threshold", 0.5))
        action: str = self.config.get("action", "block")
        # Allow config to override the default sensitive entity type list
        entity_types: list[str] | None = self.config.get("entity_types") or None

        def _analyze(t: str) -> list:
            return self._analyzer.analyze(
                text=t,
                language="en",
                entities=entity_types,  # None → Presidio uses all registered
            )

        results = await run_in_executor(_analyze, text)

        # Restrict to the sensitive types whitelist when no override is configured
        if entity_types is None:
            results = [r for r in results if r.entity_type in self._SENSITIVE_ENTITY_TYPES]

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
