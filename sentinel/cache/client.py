"""Redis cache helpers for input evaluator scores.

Cache strategy:
- Only input evaluator scores are cached (not LLM responses).
- Key = SHA-256(input_text + "|" + config_version) so any config change
  automatically invalidates all existing entries.
- TTL is taken from config.cache.ttl_seconds (default 3600).
- Scores are stored as a Redis hash where each field is an evaluator name
  and each value is a JSON-encoded float or the string "null".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_key(input_text: str, config_version: str) -> str:
    """Return the SHA-256 cache key for this (input, config) pair.

    Args:
        input_text: Raw user input string.
        config_version: ``app.config_version`` from config.yaml.

    Returns:
        64-character lowercase hex digest prefixed with ``sentinel:scores:``.
    """
    raw = f"{input_text}|{config_version}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"sentinel:scores:{digest}"


async def get_cached_scores(
    redis: Any,
    key: str,
) -> dict[str, float | None] | None:
    """Look up cached input evaluator scores.

    Args:
        redis: An active ``redis.asyncio`` client.
        key: Cache key produced by :func:`cache_key`.

    Returns:
        A mapping of evaluator name → score (float or None) on a cache hit,
        or ``None`` when the key does not exist.
    """
    raw: dict[bytes, bytes] = await redis.hgetall(key)
    if not raw:
        return None
    return {field.decode(): json.loads(value.decode()) for field, value in raw.items()}


async def set_cached_scores(
    redis: Any,
    key: str,
    scores: dict[str, float | None],
    ttl_seconds: int = 3600,
) -> None:
    """Store input evaluator scores and set expiry.

    Args:
        redis: An active ``redis.asyncio`` client.
        key: Cache key produced by :func:`cache_key`.
        scores: Mapping of evaluator name → score (float or None).
        ttl_seconds: Time-to-live in seconds (default 3600).
    """
    if not scores:
        return
    mapping = {name: json.dumps(score) for name, score in scores.items()}
    await redis.hset(key, mapping=mapping)
    await redis.expire(key, ttl_seconds)
