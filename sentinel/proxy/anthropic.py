from __future__ import annotations

import time
import uuid

from anthropic import AsyncAnthropic

from sentinel.proxy.base import LLMClient


class AnthropicClient(LLMClient):
    """LLM client for the Anthropic cloud API.

    Translates OpenAI-format messages to Anthropic format (system message
    is a separate parameter) and normalises the response back to OpenAI
    format for the proxy handler.
    """

    def __init__(self, model: str, api_key: str, timeout: float = 60.0) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def chat(self, request: dict) -> dict:
        messages = request.get("messages", [])

        # Anthropic requires system message as a top-level param, not in messages
        system: str | None = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        kwargs: dict = {
            "model": request.get("model", self._model),
            "messages": user_messages,
            "max_tokens": request.get("max_tokens", 1024),
        }
        if system:
            kwargs["system"] = system
        if "temperature" in request:
            kwargs["temperature"] = request["temperature"]

        response = await self._client.messages.create(**kwargs)

        # Normalise Anthropic response → OpenAI format
        content = response.content[0].text if response.content else ""

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }
