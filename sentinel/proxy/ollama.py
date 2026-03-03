from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator

import httpx

from sentinel.proxy.base import LLMClient


class OllamaClient(LLMClient):
    """LLM client for a locally running Ollama server.

    Translates OpenAI-format requests to Ollama's /api/chat wire format and
    normalises the response back to OpenAI format so the proxy handler stays
    provider-agnostic.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def chat(self, request: dict) -> dict:
        messages = request.get("messages", [])

        ollama_body = {
            "model": request.get("model", self._model),
            "messages": messages,
            "stream": False,
            "options": {},
        }

        if "temperature" in request:
            ollama_body["options"]["temperature"] = request["temperature"]
        if "max_tokens" in request:
            ollama_body["options"]["num_predict"] = request["max_tokens"]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=ollama_body,
            )
            response.raise_for_status()
            data = response.json()

        # Normalise Ollama response → OpenAI format
        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", self._model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def stream_chat(self, request: dict) -> AsyncGenerator[dict, None]:
        """Stream tokens from Ollama using its NDJSON streaming API."""
        messages = request.get("messages", [])
        model = request.get("model", self._model)

        ollama_body: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {},
        }
        if "temperature" in request:
            ollama_body["options"]["temperature"] = request["temperature"]
        if "max_tokens" in request:
            ollama_body["options"]["num_predict"] = request["max_tokens"]

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/chat", json=ollama_body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)

                    yield {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": data.get("model", model),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content} if content else {},
                                "finish_reason": "stop" if done else None,
                            }
                        ],
                    }
