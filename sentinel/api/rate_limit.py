"""Redis-backed rate limiting for the LLM proxy route.

Fixed-window counter keyed by tenant (resolved by APIKeyMiddleware) or client
IP when no tenant was resolved. Only guards /v1/chat/completions — the one
route that forwards to a metered LLM backend and therefore the one that turns
unlimited traffic into unlimited spend. Every other endpoint (health, metrics,
dashboard polling) stays free.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


def _rate_limit_key(request: Request) -> str:
    tenant_id = getattr(request.state, "tenant_id", None)
    identity = tenant_id or (request.client.host if request.client else "unknown")
    return f"sentinel:ratelimit:{identity}"


async def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: raise 429 once the caller exceeds its quota.

    Fails open on Redis errors — a broken rate limiter must never block the
    core proxy route, matching this codebase's fail-open philosophy for
    infrastructure failures (evaluators, WS fanout) elsewhere.

    Config (under `rate_limit` in config.yaml):
        enabled (bool):             Default False — opt-in.
        requests_per_minute (int):  Per-identity quota. Default 60.
    """
    config: dict = request.app.state.config
    rl_config = config.get("rate_limit", {})
    if not rl_config.get("enabled", False):
        return

    limit = int(rl_config.get("requests_per_minute", 60))
    window_seconds = 60

    key = _rate_limit_key(request)
    redis = request.app.state.redis

    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window_seconds)
    except Exception:
        logger.exception("rate limit check failed; failing open")
        return

    if count > limit:
        try:
            ttl = await redis.ttl(key)
        except Exception:
            logger.exception("rate limit ttl lookup failed")
            ttl = window_seconds
        retry_after = max(int(ttl), 1)
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "type": "rate_limit_exceeded",
                    "message": f"Rate limit of {limit} requests/minute exceeded.",
                    "retry_after_seconds": retry_after,
                }
            },
            headers={"Retry-After": str(retry_after)},
        )
