from __future__ import annotations

from abc import ABC, abstractmethod


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
