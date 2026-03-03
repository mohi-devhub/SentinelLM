from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class LLMClient(ABC):
    """Common interface for all LLM backend clients.

    Each implementation translates between the OpenAI-format request/response
    contract used internally and the wire format of the target provider.
    """

    @abstractmethod
    async def chat(self, request: dict) -> dict:
        """Send a chat completion request and return an OpenAI-format response dict.

        Args:
            request: Parsed ChatCompletionRequest body (OpenAI format).
                     The `context_documents` key is a SentinelLM extension
                     and must be stripped before forwarding to the backend.

        Returns:
            OpenAI-format chat completion dict with at minimum:
                id, object, created, model, choices, usage
        """
        ...

    async def stream_chat(self, request: dict) -> AsyncGenerator[dict, None]:
        """Yield OpenAI-format chunk dicts for streaming responses.

        Default implementation falls back to the non-streaming chat() method,
        emitting the entire response as a single chunk. Subclasses should
        override this with real token-by-token streaming.
        """
        response = await self.chat(request)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        yield {
            "id": response.get("id", ""),
            "object": "chat.completion.chunk",
            "created": response.get("created", 0),
            "model": response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
