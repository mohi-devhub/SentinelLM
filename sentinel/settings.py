from __future__ import annotations

from functools import lru_cache

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Infrastructure — conventional names (no prefix)
    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(alias="REDIS_URL")
    # App config — SENTINEL_ prefixed
    config_path: str = Field("config.yaml", alias="SENTINEL_CONFIG_PATH")
    env: str = Field("development", alias="SENTINEL_ENV")
    # Security — SENTINEL_ prefixed
    # Comma-separated list of allowed CORS origins. Empty = allow none (use * only in dev).
    cors_origins: str = Field(
        "http://localhost:3000,http://127.0.0.1:3000",
        alias="SENTINEL_CORS_ORIGINS",
    )
    # Optional API key. When set, all requests must include X-API-Key header.
    # Leave empty to disable auth (development / local-only deployments).
    api_key: str = Field("", alias="SENTINEL_API_KEY")
    # LLM API keys — provider-conventional names (no prefix)
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")

    model_config = {"env_file": ".env", "populate_by_name": True}

    @property
    def config(self) -> dict:
        """Load and return the parsed config.yaml dict."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings singleton. Call inside functions — never at module level."""
    return Settings()
