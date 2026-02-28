"""Shared pytest fixtures for SentinelLM tests.

Unit tests use mock helpers directly and never need the app/db/redis fixtures.
Integration tests require Docker services: postgres (sentinellm_test) + redis.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

# ── Test environment variables ───────────────────────────────────────────────
# Set before any module that calls get_settings() is imported.
os.environ.setdefault(
    "DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinellm_test"
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("SENTINEL_CONFIG_PATH", "tests/test_config.yaml")


@pytest.fixture(scope="session", autouse=True)
def clear_settings_cache():
    """Clear the lru_cache on get_settings() so test env vars take effect."""
    from sentinel.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── App / client fixtures ────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client():
    """Async httpx ASGI client with mocked Detoxify and real test DB/Redis.

    The patch must wrap both app creation AND the client context manager so
    it is active during the FastAPI lifespan startup (where models are loaded).
    """
    from unittest.mock import MagicMock, patch

    from httpx import ASGITransport, AsyncClient

    from sentinel.main import create_app

    with patch("sentinel.evaluators.output.toxicity.Detoxify") as mock_det:
        instance = MagicMock()
        instance.predict.return_value = {
            "toxicity": 0.01,
            "severe_toxicity": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_attack": 0.0,
        }
        mock_det.return_value = instance

        application = create_app(config_path="tests/test_config.yaml")
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as ac:
            yield ac


# ── Database fixture ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_pool():
    """Real asyncpg pool against sentinellm_test database.

    Truncates all tables after each test to leave a clean slate.
    Requires PostgreSQL running locally (via docker compose).
    """
    import asyncpg

    pool = await asyncpg.create_pool(
        "postgresql://sentinel:sentinel@localhost:5432/sentinellm_test"
    )
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE requests, eval_runs, eval_results RESTART IDENTITY CASCADE")
    await pool.close()


# ── Redis fixture ────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def redis_client():
    """Redis client on DB 1 (isolated from dev DB 0). Flushed after each test."""
    import redis.asyncio as aioredis

    r = aioredis.from_url("redis://localhost:6379/1")
    yield r
    await r.flushdb()
    await r.aclose()


# ── LLM response factory ─────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_response():
    """Factory that builds an OpenAI-format chat completion dict."""

    def _factory(content: str = "Paris is the capital of France.") -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    return _factory


# ── Evaluator result helper ──────────────────────────────────────────────────


def make_eval_result(
    name: str,
    score: float = 0.1,
    flag: bool = False,
    latency_ms: int = 10,
):
    """Build a clean EvalResult for use in chain/aggregator tests."""
    from sentinel.evaluators.base import EvalResult

    return EvalResult(
        evaluator_name=name,
        score=score,
        flag=flag,
        latency_ms=latency_ms,
    )
