"""Boots real (unmocked) evaluators against the actual test config.

Regression coverage for a bug that shipped silently: PIIEvaluator used to build
Presidio's AnalyzerEngine() with no explicit NLP config, so it fell back to
Presidio's own default model (en_core_web_lg) instead of the one actually
installed in the Docker image (en_core_web_sm) — a mismatch that crashed the
app at startup. tests/unit/evaluators/test_pii.py mocks presidio_analyzer
entirely, so it could not, and cannot, catch this class of bug; something has
to actually import the real library and load the real model.

Requires the spaCy model configured in tests/test_config.yaml (en_core_web_sm)
to be installed — skips cleanly when it isn't, so this stays optional for
contributors who haven't run `python -m spacy download en_core_web_sm` locally,
while CI (which does install it) always exercises the real path.
"""

from __future__ import annotations

import pytest
import yaml

spacy = pytest.importorskip("spacy")

_CONFIG_PATH = "tests/test_config.yaml"


def _spacy_model_installed(model_name: str) -> bool:
    return spacy.util.is_package(model_name)


with open(_CONFIG_PATH) as f:
    _TEST_CONFIG = yaml.safe_load(f)

_SPACY_MODEL = _TEST_CONFIG["evaluators"]["pii"].get("spacy_model", "en_core_web_sm")

pytestmark = pytest.mark.skipif(
    not _spacy_model_installed(_SPACY_MODEL),
    reason=f"spaCy model '{_SPACY_MODEL}' not installed — run "
    f"`python -m spacy download {_SPACY_MODEL}` to enable this test",
)


def test_pii_evaluator_loads_real_model_and_scores_pii():
    """PIIEvaluator._load_model() must succeed against the model the runtime
    environment actually has installed — not silently require a bigger one."""
    from sentinel.evaluators.input.pii import PIIEvaluator

    ev = PIIEvaluator(config=_TEST_CONFIG)
    assert ev._analyzer is not None


@pytest.mark.asyncio
async def test_pii_evaluator_detects_email_end_to_end():
    from sentinel.evaluators.base import EvalPayload
    from sentinel.evaluators.input.pii import PIIEvaluator

    ev = PIIEvaluator(config=_TEST_CONFIG)
    payload = EvalPayload(input_text="Reach me at jane.doe@example.com anytime.")
    result = await ev.evaluate(payload)

    assert result.error is None
    assert result.score is not None
    assert result.score > 0.0
    assert result.metadata is not None
    assert any(e["type"] == "EMAIL_ADDRESS" for e in result.metadata["entities"])


def test_pii_loads_via_registry_without_error():
    """The exact path FastAPI's lifespan uses at startup (sentinel/main.py calls
    load_evaluators) — if PII instantiation raises here, the real app
    crash-loops in production exactly the way it did before this fix.

    Scoped to just pii (not the full evaluator set) so this stays a fast,
    targeted regression test rather than pulling in every heavyweight model.
    """
    from sentinel.evaluators.registry import load_evaluators

    config = {"evaluators": {"pii": _TEST_CONFIG["evaluators"]["pii"]}}
    evaluators = load_evaluators(config)

    assert [ev.name for ev in evaluators] == ["pii"]
