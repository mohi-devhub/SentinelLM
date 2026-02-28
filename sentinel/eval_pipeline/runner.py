"""Async runner for the eval pipeline.

Loads a JSONL dataset and sends each record to the running SentinelLM proxy.
Results are collected and returned for scorecard computation.

Dataset format (JSONL, one record per line):
    {
        "input": "What is the capital of France?",
        "model": "llama3.2",                        # optional, uses server default
        "context_documents": ["Paris is ..."],       # optional, enables RAG evaluators
        "expected_output": "Paris.",                 # optional, stored for reference
        "expected_blocked": false                    # optional, for injection test records
    }
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import httpx

# Canonical evaluator ordering matches chain/aggregator.py
EVALUATOR_NAMES = (
    "pii",
    "prompt_injection",
    "topic_guardrail",
    "toxicity",
    "relevance",
    "hallucination",
    "faithfulness",
)

# Maps 400 block codes → evaluator name
_BLOCK_CODE_TO_EVALUATOR: dict[str, str] = {
    "pii_detected": "pii",
    "prompt_injection_detected": "prompt_injection",
    "off_topic": "topic_guardrail",
}


@dataclass
class DatasetRecord:
    """A single record loaded from the golden JSONL dataset."""

    record_index: int
    input: str
    model: str = "llama3.2"
    context_documents: list[str] = field(default_factory=list)
    expected_output: str | None = None
    expected_blocked: bool = False


@dataclass
class RunResult:
    """Outcome of sending one DatasetRecord through the SentinelLM proxy."""

    record: DatasetRecord

    # Sentinel scores keyed by evaluator name (None = not run or errored)
    scores: dict[str, float | None] = field(default_factory=dict)

    # Evaluator names whose score crossed the configured threshold
    flags: list[str] = field(default_factory=list)

    # True if an input evaluator blocked the request
    blocked: bool = False
    block_reason: str | None = None

    # LLM response text (None for blocked requests)
    actual_output: str | None = None

    # UUID returned in the sentinel response body (used to link to eval_results)
    request_id: UUID | None = None

    # Per-evaluator and total latency in ms from the sentinel header
    latency_ms: dict[str, int | None] = field(default_factory=dict)

    # Non-None when the HTTP call itself failed (network error, server crash, etc.)
    error: str | None = None

    @property
    def passed(self) -> bool:
        """True when no evaluator flagged the response."""
        return not self.flags and not self.blocked


def load_dataset(path: Path) -> list[DatasetRecord]:
    """Parse a JSONL file into DatasetRecord objects.

    Blank lines and lines starting with '#' are ignored so the file can
    include comments and visual section separators.
    """
    records: list[DatasetRecord] = []
    with open(path, encoding="utf-8") as fh:
        for raw_idx, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            records.append(
                DatasetRecord(
                    record_index=len(records),
                    input=data["input"],
                    model=data.get("model", "llama3.2"),
                    context_documents=data.get("context_documents", []),
                    expected_output=data.get("expected_output"),
                    expected_blocked=data.get("expected_blocked", False),
                )
            )
    return records


async def _run_one(
    client: httpx.AsyncClient,
    server_url: str,
    record: DatasetRecord,
    semaphore: asyncio.Semaphore,
) -> RunResult:
    """Send one record to the proxy and parse the response."""
    body: dict = {
        "model": record.model,
        "messages": [{"role": "user", "content": record.input}],
    }
    if record.context_documents:
        body["context_documents"] = record.context_documents

    async with semaphore:
        try:
            resp = await client.post(
                f"{server_url}/v1/chat/completions",
                json=body,
                timeout=120.0,
            )
        except Exception as exc:
            return RunResult(
                record=record,
                scores={k: None for k in EVALUATOR_NAMES},
                error=str(exc),
            )

    if resp.status_code == 200:
        data = resp.json()
        sentinel = data.get("sentinel", {})

        output_text: str | None = None
        try:
            output_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            pass

        request_id: UUID | None = None
        try:
            request_id = UUID(sentinel["request_id"])
        except (KeyError, ValueError):
            pass

        return RunResult(
            record=record,
            scores=sentinel.get("scores", {k: None for k in EVALUATOR_NAMES}),
            flags=sentinel.get("flags", []),
            blocked=False,
            actual_output=output_text,
            request_id=request_id,
            latency_ms=sentinel.get("latency_ms", {}),
        )

    if resp.status_code == 400:
        data = resp.json()
        err = data.get("error", {})
        block_code = err.get("code", "")
        blocking_ev = _BLOCK_CODE_TO_EVALUATOR.get(block_code, "unknown")

        scores: dict[str, float | None] = {k: None for k in EVALUATOR_NAMES}
        if blocking_ev in scores:
            scores[blocking_ev] = err.get("score")

        return RunResult(
            record=record,
            scores=scores,
            flags=[blocking_ev] if blocking_ev != "unknown" else [],
            blocked=True,
            block_reason=block_code,
        )

    # Unexpected status code
    return RunResult(
        record=record,
        scores={k: None for k in EVALUATOR_NAMES},
        error=f"HTTP {resp.status_code}: {resp.text[:300]}",
    )


async def run_eval(
    records: list[DatasetRecord],
    server_url: str,
    concurrency: int = 4,
    on_progress: Callable[[int, int, RunResult], None] | None = None,
) -> list[RunResult]:
    """Send all records to the proxy concurrently (bounded by semaphore).

    Args:
        records:      Dataset records to evaluate.
        server_url:   Base URL of the running SentinelLM server.
        concurrency:  Max number of in-flight requests at once.
        on_progress:  Optional callback(completed, total, latest_result).

    Returns:
        List of RunResult objects sorted by record_index.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RunResult] = []

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(_run_one(client, server_url, rec, semaphore)) for rec in records
        ]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if on_progress:
                on_progress(len(results), len(tasks), result)

    # Restore dataset order (as_completed returns in completion order)
    results.sort(key=lambda r: r.record.record_index)
    return results
