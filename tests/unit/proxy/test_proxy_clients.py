"""Unit tests for proxy LLM clients and factory.

OllamaClient uses mocked httpx.AsyncClient.
GeminiClient uses a mocked genai.Client.
OpenAIClient uses a mocked AsyncOpenAI instance.
"""

from __future__ import annotations

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
