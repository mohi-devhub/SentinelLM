from __future__ import annotations

import asyncio
import time
import uuid

from google import genai
from google.genai import types

from sentinel.proxy.base import LLMClient


class GeminiClient(LLMClient):
    """LLM client for the Google Gemini API (google-genai SDK).

    Translates OpenAI-format messages to Gemini format (role 'assistant' →
    'model', system message → system_instruction) and normalises the response
    back to OpenAI format for the proxy handler.
    """

    def __init__(self, model: str, api_key: str, timeout: float = 60.0) -> None:
        self._model_name = model
        self._timeout = timeout
        self._client = genai.Client(api_key=api_key)

    async def chat(self, request: dict) -> dict:
        messages = request.get("messages", [])

        # Gemini takes system message as a model-level param, not in the history
        system_instruction: str | None = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                chat_messages.append(msg)

        # Convert OpenAI role names → Gemini role names
        gemini_contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in chat_messages
        ]

        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction,
        )
        if "temperature" in request:
            cfg.temperature = request["temperature"]
        if "max_tokens" in request:
            cfg.max_output_tokens = request["max_tokens"]

        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model_name,
                contents=gemini_contents,  # type: ignore[arg-type]
                config=cfg,
            ),
            timeout=self._timeout,
        )

        content = response.text if response.text else ""
        usage = response.usage_metadata

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_token_count if usage else 0,
                "completion_tokens": usage.candidates_token_count if usage else 0,
                "total_tokens": usage.total_token_count if usage else 0,
            },
        }
