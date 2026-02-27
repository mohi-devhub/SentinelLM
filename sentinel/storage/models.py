from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass
class RequestRecord:
    """Python representation of a row in the `requests` table.

    All score and latency fields are Optional because they may be NULL when
    an evaluator was not run (e.g. output scores on a blocked request) or
    failed gracefully.
    """

    # Set by DB default — leave as None before insert
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None

    # Request metadata
    model: str = ""
    input_hash: str = ""
    input_text: Optional[str] = None
    input_redacted: Optional[str] = None
    has_context: bool = False

    # Block status
    blocked: bool = False
    block_reason: Optional[str] = None

    # Input evaluator scores
    score_pii: Optional[float] = None
    score_prompt_injection: Optional[float] = None
    score_topic_guardrail: Optional[float] = None

    # Output evaluator scores
    score_toxicity: Optional[float] = None
    score_relevance: Optional[float] = None
    score_hallucination: Optional[float] = None
    score_faithfulness: Optional[float] = None

    # Latencies (ms)
    latency_pii: Optional[int] = None
    latency_prompt_injection: Optional[int] = None
    latency_topic_guardrail: Optional[int] = None
    latency_toxicity: Optional[int] = None
    latency_relevance: Optional[int] = None
    latency_hallucination: Optional[int] = None
    latency_faithfulness: Optional[int] = None
    latency_llm: Optional[int] = None
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
    review_label: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reviewer_note: Optional[str] = None


@dataclass
class EvalRunRecord:
    """Python representation of a row in the `eval_runs` table."""

    id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    label: str = ""
    dataset_path: str = ""
    baseline_run_id: Optional[UUID] = None

    record_count: int = 0
    status: str = "running"  # running | complete | failed

    # Computed on completion; stored as JSON blobs in the DB
    summary_json: Optional[dict] = None
    regression_json: Optional[dict] = None


@dataclass
class EvalResultRecord:
    """Python representation of a row in the `eval_results` table."""

    id: Optional[UUID] = None
    eval_run_id: Optional[UUID] = None
    request_id: Optional[UUID] = None

    record_index: int = 0
    input_text: str = ""
    expected_output: Optional[str] = None
    actual_output: Optional[str] = None

    passed: bool = True
