from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
import yaml
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_app(config_path: str | None = None) -> FastAPI:
    """FastAPI application factory.

    Args:
        config_path: Override path to config.yaml. Defaults to the path in
                     SENTINEL_CONFIG_PATH env var (via settings). Used by tests
                     to inject a minimal test config without touching the env.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # ── Import inside lifespan — never at module level (gotcha #17) ─────
        from sentinel.evaluators.base import set_executor_workers
        from sentinel.evaluators.registry import load_evaluators
        from sentinel.settings import get_settings
        from sentinel.storage.database import create_pool

        settings = get_settings()

        # Load config — use override path when provided (test support)
        if config_path is not None:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = settings.config

        app.state.config = config
        app.state.start_time = time.monotonic()

        # Configure the shared thread pool for model inference
        workers: int = config.get("performance", {}).get("thread_pool_workers", 4)
        set_executor_workers(workers)
        logger.info("thread pool configured with %d workers", workers)

        # Load evaluators — only enabled ones are instantiated
        evaluators = load_evaluators(config)
        app.state.evaluators = evaluators
        app.state.input_evaluators = [ev for ev in evaluators if ev.runs_on == "input"]
        app.state.output_evaluators = [ev for ev in evaluators if ev.runs_on == "output"]
        logger.info(
            "loaded %d evaluator(s): %s",
            len(evaluators),
            [ev.name for ev in evaluators],
        )

        # asyncpg connection pool — created once, stored on app.state (gotcha #11)
        app.state.db_pool = await create_pool(settings.database_url)
        logger.info("database pool connected")

        # Redis async client
        app.state.redis = aioredis.from_url(settings.redis_url)
        logger.info("redis connected")

        yield

        # ── Shutdown ─────────────────────────────────────────────────────────
        await app.state.db_pool.close()
        await app.state.redis.aclose()
        logger.info("SentinelLM shutdown complete")

    app = FastAPI(
        title="SentinelLM",
        version="1.0.0",
        description="LLM Guardrails & Evaluation Middleware",
        lifespan=lifespan,
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    from sentinel.api import health, proxy, websocket  # noqa: PLC0415

    app.include_router(proxy.router)
    app.include_router(health.router)
    app.include_router(websocket.router)

    return app


# Module-level app instance for uvicorn: `uvicorn sentinel.main:app`
app = create_app()
