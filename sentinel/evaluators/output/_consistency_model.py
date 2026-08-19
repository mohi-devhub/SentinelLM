"""Shared loader/scorer for vectara-style factual-consistency models.

Used by both hallucination.py and faithfulness.py when their `backend`
config is `consistency` (the default) — a single purpose-built model whose
output is one consistency score per (context, output) pair: high = grounded,
low = hallucinated/unsupported. Each evaluator derives its own polarity from
this one score (see each evaluator's docstring for the exact mapping).

Loads via transformers.AutoModelForSequenceClassification with
trust_remote_code=True — the model ships custom inference code from
HuggingFace, not just weights, which means this executes that vendor's
Python code in this process. This is a deliberate, disclosed trade-off: on
SentinelLM's own benchmark against HaluEval (see README § Evaluator
Accuracy), this model measured AUC 0.78 versus 0.59 for the previous default
generic NLI classifier — a large, real accuracy difference, not a marginal
one. Review the model card before deploying if remote code execution is a
concern for your environment, or set `backend: nli` on both evaluators to
use the original CrossEncoder path instead (weaker measured accuracy, no
remote code execution).
"""

from __future__ import annotations

from typing import Any


def load_consistency_model(model_id: str) -> Any:
    from transformers import AutoModelForSequenceClassification  # noqa: PLC0415

    # low_cpu_mem_usage=False: same meta-tensor fix as the nli backend (see
    # hallucination.py) — newer transformers defaults to meta-device lazy
    # loading, which crashes on .to(device)/.forward() without this. Matters
    # more here than for a single evaluator: hallucination and faithfulness
    # both load this model concurrently (see registry.py's parallel loading),
    # which is exactly when this surfaced during testing.
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=True, low_cpu_mem_usage=False
    )


def score_consistency(model: Any, context_docs: list[str], output: str) -> list[float]:
    """Return one consistency score per context doc — high = grounded, low = hallucinated."""
    pairs = [(doc, output) for doc in context_docs]
    scores = model.predict(pairs)
    return [float(s) for s in scores]
