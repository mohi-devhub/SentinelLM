from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass
class RequestRecord:
    """Python representation of a row in the `requests` table.

    All score and latency fields are Optional because they may be NULL when
    an evaluator was not run (e.g. output scores on a blocked request) or
    failed gracefully.
    """

    # Set by DB default — leave as None before insert
    id: UUID | None = None
    created_at: datetime | None = None

    # Request metadata
    model: str = ""
    input_hash: str = ""
    input_text: str | None = None
    input_redacted: str | None = None
    has_context: bool = False

    # Block status
    blocked: bool = False
    block_reason: str | None = None

    # Input evaluator scores
    score_pii: float | None = None
    score_prompt_injection: float | None = None
    score_topic_guardrail: float | None = None

    # Output evaluator scores
    score_toxicity: float | None = None
    score_relevance: float | None = None
    score_hallucination: float | None = None
    score_faithfulness: float | None = None

    # Latencies (ms)
    latency_pii: int | None = None
    latency_prompt_injection: int | None = None
    latency_topic_guardrail: int | None = None
    latency_toxicity: int | None = None
    latency_relevance: int | None = None
    latency_hallucination: int | None = None
    latency_faithfulness: int | None = None
    latency_llm: int | None = None
    latency_total: int = 0

    # Flags
    flag_pii: bool = False
    flag_prompt_injection: bool = False
    flag_topic_guardrail: bool = False
    flag_toxicity: bool = False
    flag_relevance: bool = False
    flag_hallucination: bool = False
    flag_faithfulness: bool = False

    # Human review (not set at insert time)
    reviewed: bool = False
    review_label: str | None = None
    reviewed_at: datetime | None = None
    reviewer_note: str | None = None


@dataclass
class EvalRunRecord:
    """Python representation of a row in the `eval_runs` table."""

    id: UUID | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    label: str = ""
    dataset_path: str = ""
    baseline_run_id: UUID | None = None

    record_count: int = 0
    status: str = "running"  # running | complete | failed

    # Computed on completion; stored as JSON blobs in the DB
    summary_json: dict | None = None
    regression_json: dict | None = None


@dataclass
class EvalResultRecord:
    """Python representation of a row in the `eval_results` table."""

    id: UUID | None = None
    eval_run_id: UUID | None = None
    request_id: UUID | None = None

    record_index: int = 0
    input_text: str = ""
    expected_output: str | None = None
    actual_output: str | None = None

    passed: bool = True
