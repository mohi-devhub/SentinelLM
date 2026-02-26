from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_db(pool) -> bool:
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False


async def _check_redis(redis) -> bool:
    try:
        return await redis.ping()
    except Exception:
        return False


async def _check_llm_backend(config: dict) -> bool:
    """Ping the LLM backend. Cloud providers are assumed reachable (no API call made)."""
    provider = config.get("llm_backend", {}).get("provider", "ollama")
    if provider != "ollama":
        return True  # assume cloud APIs are up; checking would cost tokens
    base_url = config.get("llm_backend", {}).get("ollama", {}).get("base_url", "http://ollama:11434")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return service health, loaded evaluators, and dependency connectivity."""
    app = request.app
    config: dict = app.state.config

    uptime = int(time.monotonic() - app.state.start_time)
    evaluator_names = [ev.name for ev in app.state.evaluators]

    db_ok, redis_ok, llm_ok = (
        await _check_db(app.state.db_pool),
        await _check_redis(app.state.redis),
        await _check_llm_backend(config),
    )

    status = "healthy" if (db_ok and redis_ok) else "degraded"

    return JSONResponse(
        status_code=200 if status == "healthy" else 503,
        content={
            "status": status,
            "uptime_seconds": uptime,
            "evaluators_loaded": evaluator_names,
            "db_connected": db_ok,
            "redis_connected": redis_ok,
            "llm_backend_reachable": llm_ok,
        },
    )


@router.get("/v1/sentinel/config")
async def active_config(request: Request) -> JSONResponse:
    """Return the active evaluator configuration (no secrets)."""
    config: dict = request.app.state.config
    app_cfg = config.get("app", {})
    backend = config.get("llm_backend", {})
    provider = backend.get("provider", "ollama")
    model = backend.get(provider, {}).get("model", "")

    evaluators_out = {}
    for name, ev_cfg in config.get("evaluators", {}).items():
        evaluators_out[name] = {
            "enabled": ev_cfg.get("enabled", False),
            "threshold": ev_cfg.get("threshold"),
        }

    return JSONResponse(content={
        "config_version": app_cfg.get("config_version", ""),
        "llm_provider": provider,
        "llm_model": model,
        "evaluators": evaluators_out,
    })
