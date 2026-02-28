from __future__ import annotations

from functools import lru_cache

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    redis_url: str
    config_path: str = "config.yaml"
    env: str = "development"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    model_config = {"env_file": ".env", "env_prefix": "SENTINEL_"}

    @property
    def config(self) -> dict:
        """Load and return the parsed config.yaml dict."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings singleton. Call inside functions — never at module level."""
    return Settings()
