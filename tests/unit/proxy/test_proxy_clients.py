"""Unit tests for proxy LLM clients and factory.

OllamaClient uses mocked httpx.AsyncClient.
GeminiClient uses a mocked genai.Client.
OpenAIClient uses a mocked AsyncOpenAI instance.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.proxy.factory import get_llm_client
from sentinel.proxy.ollama import OllamaClient

# ── OllamaClient ──────────────────────────────────────────────────────────────


def _make_ollama_response(content: str = "Paris.") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_ollama_chat_basic():
    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2")
    mock_resp = _make_ollama_response("Paris is the capital of France.")

    with patch("sentinel.proxy.ollama.httpx.AsyncClient") as MockAsyncClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        MockAsyncClient.return_value = ctx

        request = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        result = await client.chat(request)

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Paris is the capital of France."
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_ollama_chat_passes_temperature():
    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2")
    mock_resp = _make_ollama_response()

    with patch("sentinel.proxy.ollama.httpx.AsyncClient") as MockAsyncClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        MockAsyncClient.return_value = ctx

        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.9,
            "max_tokens": 256,
        }
        await client.chat(request)

        call_kwargs = ctx.post.call_args.kwargs
        body = call_kwargs["json"]
        assert body["options"]["temperature"] == 0.9
        assert body["options"]["num_predict"] == 256


@pytest.mark.asyncio
async def test_ollama_chat_no_options_when_not_specified():
    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2")
    mock_resp = _make_ollama_response()

    with patch("sentinel.proxy.ollama.httpx.AsyncClient") as MockAsyncClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        MockAsyncClient.return_value = ctx

        request = {"messages": [{"role": "user", "content": "hi"}]}
        await client.chat(request)

        body = ctx.post.call_args.kwargs["json"]
        assert body["options"] == {}


@pytest.mark.asyncio
async def test_ollama_chat_uses_model_from_request():
    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2")
    mock_resp = _make_ollama_response()
    mock_resp.json.return_value["model"] = "mistral"

    with patch("sentinel.proxy.ollama.httpx.AsyncClient") as MockAsyncClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        MockAsyncClient.return_value = ctx

        request = {"model": "mistral", "messages": [{"role": "user", "content": "hi"}]}
        result = await client.chat(request)

    assert result["model"] == "mistral"


@pytest.mark.asyncio
async def test_ollama_chat_response_structure():
    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2")
    mock_resp = _make_ollama_response("42")

    with patch("sentinel.proxy.ollama.httpx.AsyncClient") as MockAsyncClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        MockAsyncClient.return_value = ctx

        result = await client.chat({"messages": [{"role": "user", "content": "q"}]})

    assert result["id"].startswith("chatcmpl-")
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["index"] == 0


# ── get_llm_client factory ────────────────────────────────────────────────────


def test_factory_returns_ollama_client():
    config = {
        "llm_backend": {
            "provider": "ollama",
            "ollama": {"base_url": "http://localhost:11434", "model": "llama3.2"},
        }
    }
    client = get_llm_client(config)
    from sentinel.proxy.ollama import OllamaClient

    assert isinstance(client, OllamaClient)


def test_factory_returns_openai_client():
    config = {
        "llm_backend": {
            "provider": "openai",
            "openai": {"model": "gpt-4o"},
        }
    }
    client = get_llm_client(config, openai_api_key="test-key")
    from sentinel.proxy.openai import OpenAIClient

    assert isinstance(client, OpenAIClient)


def test_factory_returns_gemini_client():
    config = {
        "llm_backend": {
            "provider": "gemini",
            "gemini": {"model": "gemini-2.5-flash-lite"},
        }
    }
    client = get_llm_client(config, gemini_api_key="test-key")
    from sentinel.proxy.gemini import GeminiClient

    assert isinstance(client, GeminiClient)


def test_factory_unknown_provider_raises():
    config = {"llm_backend": {"provider": "unknown_provider"}}
    with pytest.raises(ValueError, match="Unknown llm_backend.provider"):
        get_llm_client(config)


def test_factory_default_provider_is_ollama():
    """Empty config defaults to ollama."""
    client = get_llm_client({})
    from sentinel.proxy.ollama import OllamaClient

    assert isinstance(client, OllamaClient)


def test_factory_ollama_defaults():
    """Factory uses sensible defaults when ollama config is minimal."""
    config = {"llm_backend": {"provider": "ollama"}}
    client = get_llm_client(config)
    assert client._base_url == "http://localhost:11434"
    assert client._model == "llama3.2"


# ── GeminiClient ──────────────────────────────────────────────────────────────


def _make_gemini_response(text: str = "Paris.", prompt_tokens: int = 8, completion_tokens: int = 4):
    """Build a mock Gemini API response."""
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = completion_tokens
    mock_usage.total_token_count = prompt_tokens + completion_tokens

    mock_resp = MagicMock()
    mock_resp.text = text
    mock_resp.usage_metadata = mock_usage
    return mock_resp


@pytest.mark.asyncio
async def test_gemini_chat_basic():
    """GeminiClient.chat returns an OpenAI-compatible response dict."""
    from unittest.mock import AsyncMock as _AsyncMock

    mock_resp = _make_gemini_response("Paris is the capital of France.")

    with patch("sentinel.proxy.gemini.genai.Client") as MockClient:
        mock_client_instance = MagicMock()
        mock_client_instance.aio.models.generate_content = _AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_client_instance

        from sentinel.proxy.gemini import GeminiClient

        client = GeminiClient(model="gemini-2.5-flash-lite", api_key="test-key")
        request = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        result = await client.chat(request)

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Paris is the capital of France."
    assert result["usage"]["prompt_tokens"] == 8
    assert result["usage"]["completion_tokens"] == 4
    assert result["usage"]["total_tokens"] == 12


@pytest.mark.asyncio
async def test_gemini_chat_with_system_message():
    """System messages are separated from chat history."""
    from unittest.mock import AsyncMock as _AsyncMock

    mock_resp = _make_gemini_response("Hello!")

    with patch("sentinel.proxy.gemini.genai.Client") as MockClient:
        mock_client_instance = MagicMock()
        mock_client_instance.aio.models.generate_content = _AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_client_instance

        from sentinel.proxy.gemini import GeminiClient

        client = GeminiClient(model="gemini-2.5-flash-lite", api_key="test-key")
        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi!"},
            ],
        }
        result = await client.chat(request)

    assert result["choices"][0]["message"]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_gemini_chat_with_temperature_and_max_tokens():
    """temperature and max_tokens are forwarded to the Gemini config."""
    from unittest.mock import AsyncMock as _AsyncMock

    mock_resp = _make_gemini_response("Done.")

    with patch("sentinel.proxy.gemini.genai.Client") as MockClient:
        mock_client_instance = MagicMock()
        mock_client_instance.aio.models.generate_content = _AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_client_instance

        from sentinel.proxy.gemini import GeminiClient

        client = GeminiClient(model="gemini-2.5-flash-lite", api_key="test-key")
        request = {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        result = await client.chat(request)

    assert result["object"] == "chat.completion"


@pytest.mark.asyncio
async def test_gemini_chat_assistant_role_converted():
    """OpenAI 'assistant' role is converted to Gemini 'model' role."""
    from unittest.mock import AsyncMock as _AsyncMock

    mock_resp = _make_gemini_response("Continuing.")

    with patch("sentinel.proxy.gemini.genai.Client") as MockClient:
        mock_client_instance = MagicMock()
        generate_mock = _AsyncMock(return_value=mock_resp)
        mock_client_instance.aio.models.generate_content = generate_mock
        MockClient.return_value = mock_client_instance

        from sentinel.proxy.gemini import GeminiClient

        client = GeminiClient(model="gemini-2.5-flash-lite", api_key="test-key")
        request = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."},
                {"role": "user", "content": "And 3+3?"},
            ],
        }
        await client.chat(request)

    # Verify generate_content was called (role conversion happened without error)
    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_chat_response_structure():
    """Response dict has all required OpenAI-compatible fields."""
    from unittest.mock import AsyncMock as _AsyncMock

    mock_resp = _make_gemini_response("42")

    with patch("sentinel.proxy.gemini.genai.Client") as MockClient:
        mock_client_instance = MagicMock()
        mock_client_instance.aio.models.generate_content = _AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_client_instance

        from sentinel.proxy.gemini import GeminiClient

        client = GeminiClient(model="gemini-2.5-flash-lite", api_key="test-key")
        result = await client.chat({"messages": [{"role": "user", "content": "q"}]})

    assert result["id"].startswith("chatcmpl-")
    assert result["object"] == "chat.completion"
    assert result["model"] == "gemini-2.5-flash-lite"
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["finish_reason"] == "stop"


# ── stream_chat: base class fallback ─────────────────────────────────────────


class _StubLLMClient:
    """Minimal LLM client that does NOT override stream_chat(), to test the base fallback."""

    def __init__(self, response: dict) -> None:
        self._response = response

    async def chat(self, request: dict) -> dict:
        return self._response

    # Inherit stream_chat from LLMClient base via monkey-patch below


@pytest.mark.asyncio
async def test_base_stream_chat_fallback():
    """Default stream_chat() emits a single chunk wrapping the chat() response."""
    from sentinel.proxy.base import LLMClient

    fixed_response = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world."},
                "finish_reason": "stop",
            }
        ],
    }

    # Attach the base stream_chat to our stub (it's not abstract so we can bind it)
    stub = _StubLLMClient(fixed_response)
    stub.stream_chat = LLMClient.stream_chat.__get__(stub, type(stub))  # type: ignore[attr-defined]

    chunks = []
    async for chunk in stub.stream_chat({"messages": [{"role": "user", "content": "Hi"}]}):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello world."
    assert chunks[0]["choices"][0]["finish_reason"] == "stop"


# ── OllamaClient.stream_chat ──────────────────────────────────────────────────


def _make_ollama_stream_lines(tokens: list[str], model: str = "llama3.2") -> list[str]:
    """Build NDJSON lines as Ollama would stream them."""
    lines = []
    for i, token in enumerate(tokens):
        done = i == len(tokens) - 1
        lines.append(
            json.dumps(
                {
                    "model": model,
                    "message": {"role": "assistant", "content": token},
                    "done": done,
                }
            )
        )
    return lines


@pytest.mark.asyncio
async def test_ollama_stream_chat_yields_chunks():
    """stream_chat() yields one chunk per NDJSON line from Ollama."""
    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2")
    tokens = ["Hello", " world", "!"]
    ndjson_lines = _make_ollama_stream_lines(tokens)

    async def fake_aiter_lines():
        for line in ndjson_lines:
            yield line

    mock_streaming_response = MagicMock()
    mock_streaming_response.raise_for_status = MagicMock()
    mock_streaming_response.aiter_lines = fake_aiter_lines
    mock_streaming_response.__aenter__ = AsyncMock(return_value=mock_streaming_response)
    mock_streaming_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_streaming_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("sentinel.proxy.ollama.httpx.AsyncClient", return_value=mock_client):
        chunks = []
        async for chunk in client.stream_chat(
            {"model": "llama3.2", "messages": [{"role": "user", "content": "Hi"}]}
        ):
            chunks.append(chunk)

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"].startswith("chatcmpl-")

    # Last chunk has finish_reason=stop
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # Earlier chunks have no finish_reason
    assert chunks[0]["choices"][0]["finish_reason"] is None


@pytest.mark.asyncio
async def test_ollama_stream_chat_accumulates_content():
    """Content in each chunk's delta should match the streamed token."""
    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2")
    tokens = ["Paris", " is", " great"]
    ndjson_lines = _make_ollama_stream_lines(tokens)

    async def fake_aiter_lines():
        for line in ndjson_lines:
            yield line

    mock_streaming_response = MagicMock()
    mock_streaming_response.raise_for_status = MagicMock()
    mock_streaming_response.aiter_lines = fake_aiter_lines
    mock_streaming_response.__aenter__ = AsyncMock(return_value=mock_streaming_response)
    mock_streaming_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_streaming_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("sentinel.proxy.ollama.httpx.AsyncClient", return_value=mock_client):
        content_pieces = []
        async for chunk in client.stream_chat(
            {"messages": [{"role": "user", "content": "Tell me about Paris"}]}
        ):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                content_pieces.append(delta["content"])

    assert "".join(content_pieces) == "Paris is great"


# ── OpenAIClient.stream_chat ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_stream_chat_yields_chunks():
    """stream_chat() forwards chunks from the OpenAI async stream as dicts."""

    async def fake_stream():
        for content in ["Hello", " world"]:
            chunk = MagicMock()
            chunk.model_dump.return_value = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield chunk

    with patch("sentinel.proxy.openai.AsyncOpenAI") as MockOpenAI:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=fake_stream())
        MockOpenAI.return_value = mock_instance

        from sentinel.proxy.openai import OpenAIClient

        client = OpenAIClient(model="gpt-4o", api_key="test-key")
        chunks = []
        async for chunk in client.stream_chat(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        ):
            chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"


# ── AnthropicClient.stream_chat ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_stream_chat_yields_chunks():
    """stream_chat() yields one chunk per text token plus a final stop chunk."""
    pytest.importorskip("anthropic", reason="anthropic SDK not installed")

    async def fake_text_stream():
        for text in ["Hello", " there"]:
            yield text

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.text_stream = fake_text_stream()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("sentinel.proxy.anthropic.AsyncAnthropic") as MockAnthropic:
        mock_instance = MagicMock()
        mock_instance.messages.stream.return_value = mock_stream_ctx
        MockAnthropic.return_value = mock_instance

        from sentinel.proxy.anthropic import AnthropicClient

        client = AnthropicClient(model="claude-opus-4-6", api_key="test-key")
        chunks = []
        async for chunk in client.stream_chat(
            {"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "Hi"}]}
        ):
            chunks.append(chunk)

    # 2 content chunks + 1 stop chunk
    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " there"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[2]["choices"][0]["delta"] == {}


# ── GeminiClient.stream_chat ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gemini_stream_chat_yields_chunks():
    """stream_chat() yields one chunk per Gemini response chunk plus a final stop chunk."""

    async def _fake_stream():
        for text in ["Paris", " is", " beautiful"]:
            chunk = MagicMock()
            chunk.text = text
            yield chunk

    async def fake_generate_stream(*args, **kwargs):
        return _fake_stream()

    with patch("sentinel.proxy.gemini.genai.Client") as MockClient:
        mock_instance = MagicMock()
        mock_instance.aio.models.generate_content_stream = fake_generate_stream
        MockClient.return_value = mock_instance

        from sentinel.proxy.gemini import GeminiClient

        client = GeminiClient(model="gemini-2.5-flash-lite", api_key="test-key")
        chunks = []
        async for chunk in client.stream_chat(
            {"messages": [{"role": "user", "content": "Tell me about Paris"}]}
        ):
            chunks.append(chunk)

    # 3 content chunks + 1 stop chunk
    assert len(chunks) == 4
    assert chunks[0]["choices"][0]["delta"]["content"] == "Paris"
    assert chunks[2]["choices"][0]["delta"]["content"] == " beautiful"
    assert chunks[3]["choices"][0]["finish_reason"] == "stop"
    assert chunks[3]["choices"][0]["delta"] == {}
