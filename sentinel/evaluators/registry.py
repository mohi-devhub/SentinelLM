from __future__ import annotations

import logging
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


def load_evaluators(config: dict) -> list[BaseEvaluator]:
    """Instantiate and return all enabled evaluators in registry order.

    Only evaluators listed as enabled: true in config.yaml are instantiated.
    Unknown registry keys for disabled evaluators are silently skipped.
    """
    evaluators: list[BaseEvaluator] = []
    evaluator_cfg: dict = config.get("evaluators", {})

    for name, cls in EVALUATOR_REGISTRY.items():
        ev_config = evaluator_cfg.get(name, {})
        if not ev_config.get("enabled", False):
            logger.debug("evaluator %s disabled — skipping", name)
            continue
        logger.info("loading evaluator: %s", name)
        evaluators.append(cls(config))

    return evaluators
