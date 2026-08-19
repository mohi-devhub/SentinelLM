from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from sentinel.evaluators.input.pii import PIIEvaluator
from sentinel.evaluators.input.prompt_injection import PromptInjectionEvaluator
from sentinel.evaluators.input.topic_guardrail import TopicGuardrailEvaluator
from sentinel.evaluators.output.faithfulness import FaithfulnessEvaluator
from sentinel.evaluators.output.hallucination import HallucinationEvaluator
from sentinel.evaluators.output.relevance import RelevanceEvaluator
from sentinel.evaluators.output.toxicity import ToxicityEvaluator

if TYPE_CHECKING:
    from sentinel.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

# Maps config.yaml evaluator keys to their implementation classes.
# Order determines the order evaluators are instantiated and run within each layer.
EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    # Input evaluators (run before the LLM call, concurrent with short-circuit on block)
    "pii": PIIEvaluator,
    "prompt_injection": PromptInjectionEvaluator,
    "topic_guardrail": TopicGuardrailEvaluator,
    # Output evaluators (run after the LLM response, all always run)
    "toxicity": ToxicityEvaluator,
    "relevance": RelevanceEvaluator,
    "hallucination": HallucinationEvaluator,
    "faithfulness": FaithfulnessEvaluator,
}


def _prewarm_imports() -> None:
    """Import heavy ML libraries in the main thread before parallel loading.

    Python's import lock is per-module. If multiple threads race to import the
    same library for the first time, they hit a circular-import deadlock.
    Importing here populates sys.modules so threads skip the import entirely.
    """
    try:
        import transformers  # noqa: F401
    except Exception:
        pass
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        pass
    try:
        import detoxify  # noqa: F401
    except Exception:
        pass
    try:
        import presidio_analyzer  # noqa: F401
    except Exception:
        pass


def load_evaluators(config: dict) -> list[BaseEvaluator]:
    """Instantiate all enabled evaluators in parallel, return in registry order."""
    evaluator_cfg: dict = config.get("evaluators", {})

    enabled: list[tuple[str, type[BaseEvaluator]]] = [
        (name, cls)
        for name, cls in EVALUATOR_REGISTRY.items()
        if evaluator_cfg.get(name, {}).get("enabled", False)
    ]

    if not enabled:
        return []

    _prewarm_imports()

    results: dict[str, BaseEvaluator] = {}

    def _load(name: str, cls: type[BaseEvaluator]) -> tuple[str, BaseEvaluator]:
        logger.info("loading evaluator: %s", name)
        return name, cls(config)

    with ThreadPoolExecutor(max_workers=len(enabled)) as pool:
        futures = {pool.submit(_load, name, cls): name for name, cls in enabled}
        for future in as_completed(futures):
            name, ev = future.result()
            results[name] = ev

    # Return in original registry order
    return [results[name] for name, _ in enabled]
