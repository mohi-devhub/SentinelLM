"""Relevance evaluator using sentence-transformers cosine similarity.

Measures how semantically similar the LLM's response is to the original user
input. A low score indicates the model produced an off-topic or unrelated
response.

Score interpretation:
    1.0 → response is highly relevant to the user's question
    0.0 → response is completely unrelated to the user's question

flag_direction = 'below': flag when score < threshold (irrelevant response).
"""
from __future__ import annotations

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor


class RelevanceEvaluator(BaseEvaluator):
    """Scores LLM output for relevance to the original user input.

    Uses a sentence-transformer to embed both texts and returns their cosine
    similarity. Scores below the configured threshold trigger a flag.

    Config keys (under evaluators.relevance in config.yaml):
        threshold (float):     Cosine similarity below which output is flagged. Default 0.30.
        embedding_model (str): HuggingFace sentence-transformer model ID.
                               Default 'sentence-transformers/all-MiniLM-L6-v2'.
    """

    name = "relevance"
    runs_on = "output"
    flag_direction = "below"

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model_id: str = self.config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = SentenceTransformer(model_id)

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        input_text = payload.input_text
        output_text = payload.output_text  # guaranteed non-None by BaseEvaluator.evaluate()

        def _score(inp: str, out: str) -> float:
            from sentence_transformers import util  # noqa: PLC0415

            embs = self._model.encode([inp, out], convert_to_tensor=True, show_progress_bar=False)
            sim = util.cos_sim(embs[0], embs[1]).item()
            # Cosine similarity can be slightly negative for opposing vectors; clamp to [0, 1]
            return float(max(0.0, min(1.0, sim)))

        score = await run_in_executor(_score, input_text, output_text)
        return score, {"cosine_similarity": score}
