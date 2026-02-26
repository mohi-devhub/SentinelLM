from __future__ import annotations

from sentinel.proxy.base import LLMClient


def get_llm_client(config: dict, openai_api_key: str = "", anthropic_api_key: str = "") -> LLMClient:
    """Return the correct LLM client for the configured provider.

    Reads llm_backend.provider from config and instantiates the matching
    client. Switching providers requires only a config.yaml change.

    Args:
        config: Full parsed config.yaml dict.
        openai_api_key: Value of OPENAI_API_KEY env var (passed from settings).
        anthropic_api_key: Value of ANTHROPIC_API_KEY env var.
    """
    # Imports are deferred so unused provider SDKs don't need to be installed
    backend = config.get("llm_backend", {})
    provider = backend.get("provider", "ollama")

    if provider == "ollama":
        from sentinel.proxy.ollama import OllamaClient  # noqa: PLC0415

        cfg = backend.get("ollama", {})
        return OllamaClient(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            model=cfg.get("model", "llama3.2"),
            timeout=float(cfg.get("timeout_seconds", 60)),
        )

    if provider == "openai":
        from sentinel.proxy.openai import OpenAIClient  # noqa: PLC0415

        cfg = backend.get("openai", {})
        return OpenAIClient(
            model=cfg.get("model", "gpt-4o"),
            api_key=openai_api_key,
            timeout=float(cfg.get("timeout_seconds", 60)),
        )

    if provider == "anthropic":
        from sentinel.proxy.anthropic import AnthropicClient  # noqa: PLC0415

        cfg = backend.get("anthropic", {})
        return AnthropicClient(
            model=cfg.get("model", "claude-sonnet-4-6"),
            api_key=anthropic_api_key,
            timeout=float(cfg.get("timeout_seconds", 60)),
        )

    raise ValueError(f"Unknown llm_backend.provider: {provider!r}. Expected ollama, openai, or anthropic.")
