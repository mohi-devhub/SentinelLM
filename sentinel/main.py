from __future__ import annotations

import logging
import logging.config
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

logger = logging.getLogger(__name__)


def _configure_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging for production."""
    import sys

    try:
        from pythonjsonlogger import jsonlogger

        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    except ImportError:
        # Fall back to plain text logging if python-json-logger not installed
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )


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

        # Configure logging early (uses log_level from config.yaml)
        log_level = config.get("app", {}).get("log_level", "INFO")
        _configure_logging(log_level)

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

    from sentinel.settings import get_settings  # noqa: PLC0415

    settings = get_settings()

    app = FastAPI(
        title="SentinelLM",
        version="1.0.0",
        description="LLM Guardrails & Evaluation Middleware",
        lifespan=lifespan,
    )

    # ── Prometheus metrics sub-application ───────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Middleware stack (applied bottom-up — first added = outermost) ────────
    from sentinel.api.middleware import (  # noqa: PLC0415
        APIKeyMiddleware,
        PrometheusMiddleware,
        RequestIDMiddleware,
    )

    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)

    # ── CORS — origins from env var (comma-separated) ─────────────────────────
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*", "X-API-Key"],
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    from sentinel.api import (  # noqa: PLC0415
        eval,
        health,
        metrics,
        proxy,
        review,
        scores,
        websocket,
    )

    app.include_router(proxy.router)
    app.include_router(health.router)
    app.include_router(websocket.router)
    app.include_router(scores.router)
    app.include_router(metrics.router)
    app.include_router(review.router)
    app.include_router(eval.router)

    return app


# Module-level app instance for uvicorn: `uvicorn sentinel.main:app`
app = create_app()
