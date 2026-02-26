from __future__ import annotations

from typing import Optional

from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor


class ToxicityEvaluator(BaseEvaluator):
    """Scores LLM output for toxicity using the Detoxify library.

    Checks the configured dimensions (toxicity, severe_toxicity, threat,
    insult, identity_attack, obscenity) and returns the worst-case score
    across checked dimensions. flag_on: any means flag if ANY dimension
    exceeds threshold; all means ALL must exceed.
    """

    name = "toxicity"
    runs_on = "output"
    flag_direction = "above"

    def _load_model(self) -> None:
        from detoxify import Detoxify  # noqa: PLC0415

        # 'original' model is the lightest; weights ~500 MB, runs well on CPU/MPS
        self._model = Detoxify("original")

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, Optional[dict]]:
        text = payload.output_text
        dimensions: list[str] = self.config.get(
            "dimensions", ["toxicity", "severe_toxicity", "threat", "insult", "identity_attack"]
        )
        flag_on: str = self.config.get("flag_on", "any")

        def _predict(t: str) -> dict:
            return self._model.predict(t)

        raw: dict = await run_in_executor(_predict, text)

        # Restrict to configured dimensions only; ignore unconfigured ones
        scores = {dim: float(raw[dim]) for dim in dimensions if dim in raw}

        if not scores:
            return 0.0, {"dimensions": raw, "checked": dimensions}

        if flag_on == "all":
            # Worst case only matters when every dimension is high; use min
            aggregate = min(scores.values())
        else:
            # flag_on == "any": single worst dimension drives the result
            aggregate = max(scores.values())

        return aggregate, {"dimensions": scores}

    def _sync_inference(self, text: str) -> dict:
        """Synchronous wrapper used by run_in_executor."""
        return self._model.predict(text)
