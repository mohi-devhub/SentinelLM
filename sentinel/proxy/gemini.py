from __future__ import annotations

import time
import uuid

import google.generativeai as genai

from sentinel.proxy.base import LLMClient


class GeminiClient(LLMClient):
    """LLM client for the Google Gemini API.

    Translates OpenAI-format messages to Gemini format (role 'assistant' →
    'model', system message → system_instruction) and normalises the response
    back to OpenAI format for the proxy handler.
    """

    def __init__(self, model: str, api_key: str, timeout: float = 60.0) -> None:
        self._model_name = model
        self._timeout = timeout
        genai.configure(api_key=api_key)

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
        gemini_history = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [m["content"]],
            }
            for m in chat_messages
        ]

        generation_config: dict = {}
        if "temperature" in request:
            generation_config["temperature"] = request["temperature"]
        if "max_tokens" in request:
            generation_config["max_output_tokens"] = request["max_tokens"]

        model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_instruction,
        )

        response = await model.generate_content_async(
            gemini_history,
            generation_config=generation_config or None,
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
