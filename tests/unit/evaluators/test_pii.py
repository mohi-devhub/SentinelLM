"""Unit tests for PIIEvaluator.

All tests mock presidio_analyzer (including its nlp_engine submodule) and
presidio_anonymizer — no real models loaded.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.evaluators.base import EvalPayload

# ── Presidio stub types ───────────────────────────────────────────────────────


@dataclass
class FakeRecognizerResult:
    """Minimal stand-in for presidio_analyzer.RecognizerResult."""

    entity_type: str
    score: float
    start: int
    end: int


@dataclass
class FakeAnonymizerResult:
    text: str


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_CONFIG = {
    "evaluators": {
        "pii": {
            "enabled": True,
            "threshold": 0.5,
            "action": "block",
        }
    }
}


def _make_evaluator(config: dict, analyzer_results: list[Any], anonymized_text: str = "") -> Any:
    """Build a PIIEvaluator with Presidio fully mocked via sys.modules."""
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = analyzer_results

    mock_anonymizer = MagicMock()
    anonymizer_result = FakeAnonymizerResult(text=anonymized_text)
    mock_anonymizer.anonymize.return_value = anonymizer_result

    mock_analyzer_module = MagicMock()
    mock_analyzer_module.AnalyzerEngine.return_value = mock_analyzer

    mock_anonymizer_module = MagicMock()
    mock_anonymizer_module.AnonymizerEngine.return_value = mock_anonymizer

    # NlpEngineProvider(nlp_configuration=...).create_engine() must resolve too —
    # _load_model builds an explicit NLP engine rather than relying on Presidio's
    # own default (which is en_core_web_lg, not what the Docker image installs).
    mock_nlp_engine_module = MagicMock()
    mock_nlp_engine_provider_cls = mock_nlp_engine_module.NlpEngineProvider
    mock_nlp_engine_provider_cls.return_value.create_engine.return_value = MagicMock(
        name="fake_nlp_engine"
    )

    original_pa = sys.modules.get("presidio_analyzer")
    original_pa_nlp = sys.modules.get("presidio_analyzer.nlp_engine")
    original_pan = sys.modules.get("presidio_anonymizer")
    sys.modules["presidio_analyzer"] = mock_analyzer_module
    sys.modules["presidio_analyzer.nlp_engine"] = mock_nlp_engine_module
    sys.modules["presidio_anonymizer"] = mock_anonymizer_module
    try:
        # Force re-import with mocked modules
        if "sentinel.evaluators.input.pii" in sys.modules:
            del sys.modules["sentinel.evaluators.input.pii"]
        from sentinel.evaluators.input.pii import PIIEvaluator  # noqa: PLC0415

        ev = PIIEvaluator(config=config)
        # Expose the mocks so tests can inspect calls
        ev._analyzer = mock_analyzer
        ev._anonymizer = mock_anonymizer
        ev._nlp_engine_provider_cls = mock_nlp_engine_provider_cls
        return ev
    finally:
        if original_pa is None:
            sys.modules.pop("presidio_analyzer", None)
        else:
            sys.modules["presidio_analyzer"] = original_pa
        if original_pa_nlp is None:
            sys.modules.pop("presidio_analyzer.nlp_engine", None)
        else:
            sys.modules["presidio_analyzer.nlp_engine"] = original_pa_nlp
        if original_pan is None:
            sys.modules.pop("presidio_anonymizer", None)
        else:
            sys.modules["presidio_anonymizer"] = original_pan


# ── Tests: metadata and class attributes ─────────────────────────────────────


def test_evaluator_class_attributes():
    ev = _make_evaluator(MOCK_CONFIG, [])
    assert ev.name == "pii"
    assert ev.runs_on == "input"
    assert ev.flag_direction == "above"


# ── Tests: spaCy model wiring ─────────────────────────────────────────────────
#
# Regression coverage for a bug that shipped silently: PIIEvaluator used to call
# AnalyzerEngine() with no arguments, which makes Presidio fall back to its own
# default NLP config (en_core_web_lg) — a model docker/Dockerfile.api never
# installs (it installs en_core_web_sm). The evaluator now builds an explicit
# NlpEngineProvider pinned to config['spacy_model'] so the loaded model always
# matches what's actually on disk, and these tests pin that contract down so a
# future refactor can't silently reintroduce the mismatch.


def test_defaults_to_docker_installed_spacy_model():
    """With no spacy_model configured, the NLP engine must target en_core_web_lg —
    the model actually installed by docker/Dockerfile.api. Explicitly pinned
    rather than relying on Presidio's own matching default, so a deployment
    that configures a smaller model doesn't silently load the wrong one."""
    ev = _make_evaluator(MOCK_CONFIG, [])

    ev._nlp_engine_provider_cls.assert_called_once_with(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )


def test_respects_configured_spacy_model():
    """A deployment that has installed en_core_web_lg can opt into it via config."""
    config = {
        "evaluators": {
            "pii": {
                "enabled": True,
                "threshold": 0.5,
                "action": "block",
                "spacy_model": "en_core_web_lg",
            }
        }
    }
    ev = _make_evaluator(config, [])

    ev._nlp_engine_provider_cls.assert_called_once_with(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )


# ── Tests: no PII detected ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clean_input_score_is_zero():
    """Input with no PII should return score 0.0 and no entities."""
    ev = _make_evaluator(MOCK_CONFIG, [])

    payload = EvalPayload(input_text="What is the capital of France?")
    result = await ev.evaluate(payload)

    assert result.error is None
    assert result.score == 0.0
    assert result.flag is False
    assert result.metadata is not None
    assert result.metadata["entities"] == []


# ── Tests: PII detected — action: block ──────────────────────────────────────


@pytest.mark.asyncio
async def test_detected_pii_score_is_max_confidence():
    """Score equals the highest confidence among detected entities."""
    entities = [
        FakeRecognizerResult("PERSON", 0.85, 0, 10),
        FakeRecognizerResult("EMAIL_ADDRESS", 0.92, 15, 35),
    ]
    ev = _make_evaluator(MOCK_CONFIG, entities)

    payload = EvalPayload(input_text="John Smith john@example.com")
    result = await ev.evaluate(payload)

    assert result.error is None
    assert result.score == pytest.approx(0.92)
    assert result.flag is False  # chain runner sets this, not the evaluator
    assert result.metadata is not None
    assert len(result.metadata["entities"]) == 2


@pytest.mark.asyncio
async def test_block_action_has_no_redacted_text():
    """action: block should not include redacted_text in metadata."""
    entities = [FakeRecognizerResult("PHONE_NUMBER", 0.80, 0, 12)]
    ev = _make_evaluator(MOCK_CONFIG, entities)

    payload = EvalPayload(input_text="555-867-5309")
    result = await ev.evaluate(payload)

    assert result.metadata is not None
    assert result.metadata["action"] == "block"
    assert "redacted_text" not in result.metadata


# ── Tests: action: redact ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_redact_action_includes_redacted_text():
    """action: redact should run the anonymizer and include redacted_text."""
    entities = [FakeRecognizerResult("PERSON", 0.90, 0, 8)]
    redact_config = {"evaluators": {"pii": {"enabled": True, "threshold": 0.5, "action": "redact"}}}
    ev = _make_evaluator(redact_config, entities, anonymized_text="<PERSON> visited Paris.")

    payload = EvalPayload(input_text="John Doe visited Paris.")
    result = await ev.evaluate(payload)

    assert result.error is None
    assert result.score == pytest.approx(0.90)
    assert result.metadata is not None
    assert result.metadata["action"] == "redact"
    assert result.metadata["redacted_text"] == "<PERSON> visited Paris."


# ── Tests: sensitive-entity-type whitelist ──────────────────────────────────


@pytest.mark.asyncio
async def test_non_sensitive_entity_types_are_excluded_regardless_of_score():
    """LOCATION and DATE_TIME are excluded even at a high confidence score.

    Both were measured directly against Presidio to score as confidently as
    genuine PII (DATE_TIME on ordinary phrases like "year-over-year" scores
    the same 0.85 as a real PERSON match) — the whitelist exists precisely
    because no threshold can separate them from real detections.
    """
    entities = [
        FakeRecognizerResult("LOCATION", 0.99, 0, 5),
        FakeRecognizerResult("DATE_TIME", 0.85, 10, 24),
    ]
    ev = _make_evaluator(MOCK_CONFIG, entities)

    payload = EvalPayload(input_text="Paris grew year-over-year")
    result = await ev.evaluate(payload)

    assert result.score == 0.0
    assert result.metadata is not None
    assert result.metadata["entities"] == []


@pytest.mark.asyncio
async def test_sensitive_entity_type_still_detected_alongside_excluded_ones():
    """A genuinely sensitive entity is still caught even mixed with excluded types."""
    entities = [
        FakeRecognizerResult("DATE_TIME", 0.85, 0, 14),
        FakeRecognizerResult("PERSON", 0.85, 20, 30),
    ]
    ev = _make_evaluator(MOCK_CONFIG, entities)

    payload = EvalPayload(input_text="year-over-year, John Smith")
    result = await ev.evaluate(payload)

    assert result.score == pytest.approx(0.85)
    assert result.metadata is not None
    assert [e["type"] for e in result.metadata["entities"]] == ["PERSON"]


# ── Tests: threshold filtering ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_entities_below_threshold_are_filtered():
    """Entities with score < threshold are ignored; score is 0.0 when all filtered."""
    # Both entities are below the default threshold of 0.5
    entities = [
        FakeRecognizerResult("LOCATION", 0.30, 0, 5),
        FakeRecognizerResult("PERSON", 0.45, 10, 18),
    ]
    ev = _make_evaluator(MOCK_CONFIG, entities)

    payload = EvalPayload(input_text="Paris and John Doe")
    result = await ev.evaluate(payload)

    assert result.score == 0.0
    assert result.metadata is not None
    assert result.metadata["entities"] == []


@pytest.mark.asyncio
async def test_only_entities_above_threshold_counted():
    """Only entities meeting or exceeding threshold contribute to the score."""
    entities = [
        FakeRecognizerResult("PERSON", 0.40, 0, 8),  # below threshold → ignored
        FakeRecognizerResult("EMAIL_ADDRESS", 0.75, 9, 30),  # above threshold → counted
    ]
    ev = _make_evaluator(MOCK_CONFIG, entities)

    payload = EvalPayload(input_text="John Doe j@test.com")
    result = await ev.evaluate(payload)

    assert result.score == pytest.approx(0.75)
    assert result.metadata is not None
    assert len(result.metadata["entities"]) == 1
    assert result.metadata["entities"][0]["type"] == "EMAIL_ADDRESS"


# ── Tests: fail-open ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyzer_exception_returns_fail_open():
    """If the analyzer raises, evaluate() returns score=None, flag=False."""
    ev = _make_evaluator(MOCK_CONFIG, [])
    ev._analyzer.analyze.side_effect = RuntimeError("spaCy model not found")

    payload = EvalPayload(input_text="Some input text")
    result = await ev.evaluate(payload)

    assert result.score is None
    assert result.flag is False
    assert result.error is not None
    assert "spaCy model not found" in result.error
