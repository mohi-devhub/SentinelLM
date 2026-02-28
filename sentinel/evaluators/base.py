from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

T = TypeVar("T")

# Shared thread pool for all CPU-bound model inference.
# Initialized once at module load; shared across all evaluator instances.
# Size is overridden at startup from config performance.thread_pool_workers.
_executor = ThreadPoolExecutor(max_workers=4)


def set_executor_workers(n: int) -> None:
    """Replace the shared executor with one sized to n workers.

    Called once at app startup after config is loaded.
    """
    global _executor
    _executor = ThreadPoolExecutor(max_workers=n)


async def run_in_executor(fn: Callable[..., T], *args) -> T:
    """Run a synchronous function in the shared thread pool.

    All CPU-bound model inference must go through this helper so the
    asyncio event loop stays unblocked for other in-flight requests.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


@dataclass
class EvalPayload:
    """The input passed to every evaluator.

    Constructed once per request in the proxy handler and shared across
    all evaluators in the chain.
    """

    # User's input message (after PII redaction when applicable)
    input_text: str

    # LLM response text — None for input-layer evaluators
    output_text: str | None = None

    # Source documents supplied for RAG grounding checks.
    # None for non-RAG requests.
    context_documents: list[str] | None = None

    # Full parsed config.yaml dict.
    # Evaluators read their own section: config["evaluators"][self.name]
    config: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """The output of every evaluator.

    SCORING CONVENTION:
    - Risk evaluators (pii, prompt_injection, toxicity, hallucination):
        score → 1.0 means more dangerous; flag when score EXCEEDS threshold.
    - Quality evaluators (relevance, faithfulness, topic_guardrail):
        score → 1.0 means better quality; flag when score FALLS BELOW threshold.

    The chain runner applies threshold logic via BaseEvaluator.is_flagged()
    after receiving the raw score. Evaluators must NOT set flag themselves.
    """

    evaluator_name: str

    # Normalized score 0.0–1.0. None when evaluator was skipped or errored.
    score: float | None

    # Set by the chain runner after applying threshold — evaluators leave this False.
    flag: bool = False

    # Wall-clock inference time in milliseconds
    latency_ms: int = 0

    # Evaluator-specific detail (e.g. detected PII entities, toxicity dimensions).
    # Used for dashboard display and debug logging.
    metadata: dict | None = None

    # Non-None when the evaluator raised an exception or timed out.
    # Always paired with score=None and flag=False (fail-open).
    error: str | None = None


class BaseEvaluator(ABC):
    """Abstract base class that every evaluator must subclass.

    Subclasses must:
    1. Set `name` as a class-level string matching the config.yaml key.
    2. Set `runs_on` to 'input' or 'output'.
    3. Set `flag_direction` to 'above' or 'below'.
    4. Implement `_load_model()` for one-time heavyweight initialization.
    5. Implement `_run_inference()` for the actual scoring logic.
    6. NOT override `evaluate()` — it is a template method.
    """

    # ── Class-level attributes (must be defined by every subclass) ──────────

    name: str
    runs_on: Literal["input", "output"]
    flag_direction: Literal["above", "below"]

    # ── Initialization ───────────────────────────────────────────────────────

    def __init__(self, config: dict) -> None:
        """Load evaluator-specific config section and initialize the model.

        `config` is the full parsed config.yaml dict. Called once at startup.
        """
        self.config = config.get("evaluators", {}).get(self.name, {})
        self._model: Any = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load and warm up the model. Runs synchronously at startup — that is fine."""
        ...

    # ── Inference ────────────────────────────────────────────────────────────

    @abstractmethod
    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        """Score the payload.

        Returns:
            (score, metadata) where score is in [0.0, 1.0] and metadata is an
            optional dict with evaluator-specific detail, or None.

        CPU-bound work must be offloaded via run_in_executor:
            score = await run_in_executor(self._sync_fn, payload.input_text)
        """
        ...

    # ── Public interface (do not override) ───────────────────────────────────

    async def evaluate(self, payload: EvalPayload) -> EvalResult:
        """Template method called by the chain runner.

        Wraps _run_inference with timing, skip logic, and fail-open error
        handling. The chain runner calls this; subclasses must not override it.
        """
        start = time.monotonic()
        try:
            # Output evaluators have nothing to score without a response
            if self.runs_on == "output" and not payload.output_text:
                return EvalResult(
                    evaluator_name=self.name,
                    score=None,
                    flag=False,
                    latency_ms=0,
                    metadata={"skipped": "no output_text"},
                )

            # Hallucination and faithfulness require grounding documents
            if self.name in ("hallucination", "faithfulness") and not payload.context_documents:
                return EvalResult(
                    evaluator_name=self.name,
                    score=None,
                    flag=False,
                    latency_ms=0,
                    metadata={"skipped": "no context_documents"},
                )

            score, metadata = await self._run_inference(payload)
            latency_ms = int((time.monotonic() - start) * 1000)

            return EvalResult(
                evaluator_name=self.name,
                score=score,
                flag=False,  # chain runner sets this after applying threshold
                latency_ms=latency_ms,
                metadata=metadata,
            )

        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            return EvalResult(
                evaluator_name=self.name,
                score=None,
                flag=False,  # fail-open: never block on evaluator crash
                latency_ms=latency_ms,
                error=str(e),
            )

    def threshold(self) -> float:
        """Return the configured threshold for this evaluator."""
        return float(self.config.get("threshold", 0.8))

    def is_flagged(self, score: float) -> bool:
        """Apply threshold logic based on flag_direction."""
        if self.flag_direction == "above":
            return score >= self.threshold()
        return score <= self.threshold()
