"""Unit tests for eval_pipeline.runner.

Covers DatasetRecord, RunResult, load_dataset, and _run_one (via mocked httpx).
run_eval is tested with a simple mock that returns predefined results.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from sentinel.eval_pipeline.runner import (
    _BLOCK_CODE_TO_EVALUATOR,
    EVALUATOR_NAMES,
    DatasetRecord,
    RunResult,
    _run_one,
    load_dataset,
)

# ── DatasetRecord ──────────────────────────────────────────────────────────────


def test_dataset_record_defaults():
    rec = DatasetRecord(record_index=0, input="hello")
    assert rec.model == "llama3.2"
    assert rec.context_documents == []
    assert rec.expected_output is None
    assert rec.expected_blocked is False


def test_dataset_record_custom_fields():
    rec = DatasetRecord(
        record_index=3,
        input="query",
        model="gpt-4o",
        context_documents=["doc1", "doc2"],
        expected_output="answer",
        expected_blocked=True,
    )
    assert rec.record_index == 3
    assert rec.model == "gpt-4o"
    assert len(rec.context_documents) == 2
    assert rec.expected_blocked is True


# ── RunResult ──────────────────────────────────────────────────────────────────


def test_run_result_passed_when_no_flags():
    rec = DatasetRecord(record_index=0, input="q")
    result = RunResult(record=rec, scores={}, flags=[], blocked=False)
    assert result.passed is True


def test_run_result_not_passed_when_flagged():
    rec = DatasetRecord(record_index=0, input="q")
    result = RunResult(record=rec, flags=["toxicity"], blocked=False)
    assert result.passed is False


def test_run_result_not_passed_when_blocked():
    rec = DatasetRecord(record_index=0, input="q")
    result = RunResult(record=rec, flags=[], blocked=True)
    assert result.passed is False


def test_run_result_not_passed_when_blocked_and_flagged():
    rec = DatasetRecord(record_index=0, input="q")
    result = RunResult(record=rec, flags=["pii"], blocked=True)
    assert result.passed is False


# ── load_dataset ────────────────────────────────────────────────────────────────


def test_load_dataset_basic(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('{"input": "What is 2+2?"}\n{"input": "Who was Einstein?"}\n')
    records = load_dataset(jsonl)
    assert len(records) == 2
    assert records[0].input == "What is 2+2?"
    assert records[0].record_index == 0
    assert records[1].record_index == 1


def test_load_dataset_skips_blank_lines(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('{"input": "q1"}\n\n{"input": "q2"}\n')
    records = load_dataset(jsonl)
    assert len(records) == 2


def test_load_dataset_skips_comment_lines(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('# This is a comment\n{"input": "q1"}\n# Another comment\n{"input": "q2"}\n')
    records = load_dataset(jsonl)
    assert len(records) == 2


def test_load_dataset_full_record(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    record = {
        "input": "Summarise the document",
        "model": "gpt-4o",
        "context_documents": ["doc1", "doc2"],
        "expected_output": "Summary here.",
        "expected_blocked": False,
    }
    jsonl.write_text(json.dumps(record) + "\n")
    records = load_dataset(jsonl)
    assert len(records) == 1
    r = records[0]
    assert r.model == "gpt-4o"
    assert r.context_documents == ["doc1", "doc2"]
    assert r.expected_output == "Summary here."
    assert r.expected_blocked is False


def test_load_dataset_empty_file(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    records = load_dataset(jsonl)
    assert records == []


def test_load_dataset_default_model(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('{"input": "question"}\n')
    records = load_dataset(jsonl)
    assert records[0].model == "llama3.2"


# ── _run_one ──────────────────────────────────────────────────────────────────


def _make_record(index: int = 0) -> DatasetRecord:
    return DatasetRecord(record_index=index, input="test question")


@pytest.mark.asyncio
async def test_run_one_success_200():
    """200 response: scores, flags, and output extracted correctly."""
    import asyncio

    record = _make_record()
    req_id = str(uuid.uuid4())
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "The answer is 42."}}],
        "sentinel": {
            "request_id": req_id,
            "scores": {ev: 0.1 for ev in EVALUATOR_NAMES},
            "flags": [],
            "latency_ms": {"total": 100},
        },
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    semaphore = asyncio.Semaphore(1)

    result = await _run_one(mock_client, "http://localhost:8000", record, semaphore)

    assert result.error is None
    assert result.blocked is False
    assert result.actual_output == "The answer is 42."
    assert result.request_id is not None
    assert result.flags == []


@pytest.mark.asyncio
async def test_run_one_blocked_400():
    """400 response: blocked=True, block_reason set, evaluator in flags."""
    import asyncio

    record = _make_record()
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "error": {
            "code": "prompt_injection_detected",
            "score": 0.93,
            "message": "Prompt injection detected",
        }
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    semaphore = asyncio.Semaphore(1)

    result = await _run_one(mock_client, "http://localhost:8000", record, semaphore)

    assert result.blocked is True
    assert result.block_reason == "prompt_injection_detected"
    assert "prompt_injection" in result.flags


@pytest.mark.asyncio
async def test_run_one_pii_block():
    import asyncio

    record = _make_record()
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"error": {"code": "pii_detected", "score": 0.88}}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    semaphore = asyncio.Semaphore(1)

    result = await _run_one(mock_client, "http://localhost:8000", record, semaphore)

    assert result.blocked is True
    assert "pii" in result.flags
    assert result.scores["pii"] == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_run_one_unexpected_status():
    """Non-200/400 status code returns an error result."""
    import asyncio

    record = _make_record()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    semaphore = asyncio.Semaphore(1)

    result = await _run_one(mock_client, "http://localhost:8000", record, semaphore)

    assert result.error is not None
    assert "500" in result.error


@pytest.mark.asyncio
async def test_run_one_network_error():
    """Network exception returns error result (fail-open)."""
    import asyncio

    record = _make_record()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=ConnectionError("Connection refused"))
    semaphore = asyncio.Semaphore(1)

    result = await _run_one(mock_client, "http://localhost:8000", record, semaphore)

    assert result.error is not None
    assert "Connection refused" in result.error
    assert result.blocked is False


@pytest.mark.asyncio
async def test_run_one_context_documents_included():
    """context_documents are forwarded in the request body."""
    import asyncio

    record = DatasetRecord(record_index=0, input="summarise", context_documents=["doc1", "doc2"])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Summary."}}],
        "sentinel": {"scores": {}, "flags": [], "latency_ms": {}},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    semaphore = asyncio.Semaphore(1)

    await _run_one(mock_client, "http://localhost:8000", record, semaphore)

    call_kwargs = mock_client.post.call_args.kwargs
    assert "json" in call_kwargs
    assert call_kwargs["json"]["context_documents"] == ["doc1", "doc2"]


# ── EVALUATOR_NAMES and _BLOCK_CODE_TO_EVALUATOR ─────────────────────────────


def test_evaluator_names_contains_all_seven():
    expected = {
        "pii",
        "prompt_injection",
        "topic_guardrail",
        "toxicity",
        "relevance",
        "hallucination",
        "faithfulness",
    }
    assert set(EVALUATOR_NAMES) == expected


def test_block_code_mapping():
    assert _BLOCK_CODE_TO_EVALUATOR["pii_detected"] == "pii"
    assert _BLOCK_CODE_TO_EVALUATOR["prompt_injection_detected"] == "prompt_injection"
    assert _BLOCK_CODE_TO_EVALUATOR["off_topic"] == "topic_guardrail"
