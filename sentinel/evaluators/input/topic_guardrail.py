"""Topic guardrail evaluator using sentence-transformers cosine similarity.

Embeds the user's input and computes cosine similarity against a set of
allowed topics defined in config.yaml. If the maximum similarity across all
allowed topics falls below the configured threshold, the request is blocked
as off-topic.

Score interpretation:
    1.0 → input is highly similar to at least one allowed topic (on-topic)
    0.0 → input has no semantic overlap with any allowed topic (off-topic)

flag_direction = 'below': flag when score < threshold (off-topic).

Topic embeddings are pre-computed once at startup and reused for every request,
so per-request cost is just one forward pass for the user's input.
"""
from __future__ import annotations

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor


class TopicGuardrailEvaluator(BaseEvaluator):
    """Rejects off-topic user input using semantic similarity to allowed topics.

    Config keys (under evaluators.topic_guardrail in config.yaml):
        threshold (float):         Min cosine similarity to any allowed topic. Default 0.30.
        allowed_topics (list[str]): Topics that define the app's domain.
        embedding_model (str):     HuggingFace sentence-transformer model ID.
                                   Default 'sentence-transformers/all-MiniLM-L6-v2'.
    """

    name = "topic_guardrail"
    runs_on = "input"
    flag_direction = "below"

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model_id: str = self.config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = SentenceTransformer(model_id)

        allowed_topics: list[str] = self.config.get("allowed_topics", [])
        if allowed_topics:
            # Pre-encode topics once at startup; shape (n_topics, embedding_dim)
            self._topic_embeddings = self._model.encode(
                allowed_topics, convert_to_tensor=True, show_progress_bar=False
            )
        else:
            self._topic_embeddings = None

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        if self._topic_embeddings is None:
            # No topics configured — pass everything through (score = 1.0)
            return 1.0, {"warning": "no allowed_topics configured; all requests pass"}

        text = payload.input_text
        allowed_topics: list[str] = self.config.get("allowed_topics", [])

        def _score(t: str) -> float:
            from sentence_transformers import util  # noqa: PLC0415

            input_emb = self._model.encode(t, convert_to_tensor=True, show_progress_bar=False)
            # cos_sim returns shape (1, n_topics); take the row's max
            sims = util.cos_sim(input_emb, self._topic_embeddings)[0]
            return float(max(0.0, sims.max().item()))

        score = await run_in_executor(_score, text)
        return score, {"allowed_topics": allowed_topics, "max_similarity": score}
