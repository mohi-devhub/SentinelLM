from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from sentinel.evaluators.base import EvalResult
from sentinel.storage.models import RequestRecord

# All evaluator names in a fixed order for consistent key iteration
_EVALUATOR_NAMES = (
    "pii",
    "prompt_injection",
    "topic_guardrail",
    "toxicity",
    "relevance",
    "hallucination",
    "faithfulness",
)


@dataclass
class SentinelResult:
    """Assembled evaluation outcome for a single proxied request.

    Matches the SentinelResult schema in openapi.yaml and is appended to
    every successful LLM response under the `sentinel` key.
    """

    # Set after DB insert; None until then
    request_id: UUID | None = None

    # Per-evaluator scores (None = not run or errored)
    scores: dict[str, float | None] = field(default_factory=dict)

    # Names of evaluators whose score tripped the threshold
    flags: list[str] = field(default_factory=list)

    # Per-evaluator latencies in ms + total
    latency_ms: dict[str, int | None] = field(default_factory=dict)

    # True if any input evaluator blocked the request
    blocked: bool = False
    block_reason: str | None = None


def assemble_result(
    input_results: list[EvalResult],
    output_results: list[EvalResult],
    latency_llm: int | None,
    latency_total: int,
) -> SentinelResult:
    """Build a SentinelResult from all evaluator outputs.

    Merges input and output EvalResult lists, extracts scores, flags, and
    latencies into the flat structure expected by the API response.
    """
    all_results = input_results + output_results
    result_by_name = {r.evaluator_name: r for r in all_results}

    scores: dict[str, float | None] = {}
    latencies: dict[str, int | None] = {}
    flags: list[str] = []

    for name in _EVALUATOR_NAMES:
        ev_result = result_by_name.get(name)
        scores[name] = ev_result.score if ev_result else None
        latencies[name] = ev_result.latency_ms if ev_result else None
        if ev_result and ev_result.flag:
            flags.append(name)

    latencies["llm"] = latency_llm
    latencies["total"] = latency_total

    blocked = bool(flags and any(
        r.flag for r in input_results if r.evaluator_name in result_by_name
    ))
    block_reason: str | None = None
    if blocked:
        # First flagged input evaluator determines the block reason
        for r in input_results:
            if r.flag:
                block_reason = f"{r.evaluator_name}_detected"
                break

    return SentinelResult(
        scores=scores,
        flags=flags,
        latency_ms=latencies,
        blocked=blocked,
        block_reason=block_reason,
    )


def build_request_record(
    sentinel_result: SentinelResult,
    model: str,
    input_hash: str,
    input_text: str | None,
    input_redacted: str | None,
    has_context: bool,
) -> RequestRecord:
    """Populate a RequestRecord from a SentinelResult for DB persistence."""
    s = sentinel_result.scores
    lat = sentinel_result.latency_ms

    return RequestRecord(
        model=model,
        input_hash=input_hash,
        input_text=input_text,
        input_redacted=input_redacted,
        has_context=has_context,
        blocked=sentinel_result.blocked,
        block_reason=sentinel_result.block_reason,
        # Scores
        score_pii=s.get("pii"),
        score_prompt_injection=s.get("prompt_injection"),
        score_topic_guardrail=s.get("topic_guardrail"),
        score_toxicity=s.get("toxicity"),
        score_relevance=s.get("relevance"),
        score_hallucination=s.get("hallucination"),
        score_faithfulness=s.get("faithfulness"),
        # Latencies
        latency_pii=lat.get("pii"),
        latency_prompt_injection=lat.get("prompt_injection"),
        latency_topic_guardrail=lat.get("topic_guardrail"),
        latency_toxicity=lat.get("toxicity"),
        latency_relevance=lat.get("relevance"),
        latency_hallucination=lat.get("hallucination"),
        latency_faithfulness=lat.get("faithfulness"),
        latency_llm=lat.get("llm"),
        latency_total=lat.get("total") or 0,
        # Flags
        flag_pii="pii" in sentinel_result.flags,
        flag_prompt_injection="prompt_injection" in sentinel_result.flags,
        flag_topic_guardrail="topic_guardrail" in sentinel_result.flags,
        flag_toxicity="toxicity" in sentinel_result.flags,
        flag_relevance="relevance" in sentinel_result.flags,
        flag_hallucination="hallucination" in sentinel_result.flags,
        flag_faithfulness="faithfulness" in sentinel_result.flags,
    )
