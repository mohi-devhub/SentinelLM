from __future__ import annotations

from openai import AsyncOpenAI

from sentinel.proxy.base import LLMClient


class OpenAIClient(LLMClient):
    """LLM client for the OpenAI cloud API.

    Uses the official openai SDK. The response is already in OpenAI format
    so only a minimal dict conversion is needed.
    """

    def __init__(self, model: str, api_key: str, timeout: float = 60.0) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def chat(self, request: dict) -> dict:
        response = await self._client.chat.completions.create(
            model=request.get("model", self._model),
            messages=request["messages"],
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens"),
        )

        return response.model_dump()
