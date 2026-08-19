"""Redis cache helpers for input evaluator results.

Cache strategy:
- Only input evaluator results are cached (not LLM responses, not output
  evaluator scores — those depend on output_text, which varies per LLM call
  even for identical input, so caching them wouldn't hit).
- Key = SHA-256(input_text + "|" + config_version), scoped per tenant, so any
  config change (bump app.config_version in config.yaml) or tenant boundary
  invalidates cleanly.
- TTL is taken from config.cache.ttl_seconds (default 3600).
- Results are stored as a Redis hash where each field is an evaluator name
  and each value is a JSON-encoded {"score": float|None, "metadata": dict|None}
  — metadata is cached too (not just the score) because some evaluators
  encode behavior-affecting data there, e.g. the PII evaluator's
  redacted_text, which the proxy needs even on a cache hit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_key(input_text: str, config_version: str, tenant_id: object) -> str:
    """Return the SHA-256 cache key for this (tenant, input, config) tuple.

    Args:
        input_text: Raw user input string.
        config_version: ``app.config_version`` from config.yaml.
        tenant_id: Owning tenant's id — scopes the key so one tenant can
            never read another's cached evaluator results.

    Returns:
        64-character lowercase hex digest prefixed with ``sentinel:{tenant_id}:scores:``.
    """
    raw = f"{input_text}|{config_version}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"sentinel:{tenant_id}:scores:{digest}"


async def get_cached_results(
    redis: Any,
    key: str,
) -> dict[str, dict[str, Any]] | None:
    """Look up cached input evaluator results.

    Args:
        redis: An active ``redis.asyncio`` client.
        key: Cache key produced by :func:`cache_key`.

    Returns:
        A mapping of evaluator name → {"score": float|None, "metadata": dict|None}
        on a cache hit, or ``None`` when the key does not exist.
    """
    raw: dict[bytes, bytes] = await redis.hgetall(key)
    if not raw:
        return None
    return {field.decode(): json.loads(value.decode()) for field, value in raw.items()}


async def set_cached_results(
    redis: Any,
    key: str,
    results: dict[str, dict[str, Any]],
    ttl_seconds: int = 3600,
) -> None:
    """Store input evaluator results and set expiry.

    Args:
        redis: An active ``redis.asyncio`` client.
        key: Cache key produced by :func:`cache_key`.
        results: Mapping of evaluator name → {"score": ..., "metadata": ...}.
        ttl_seconds: Time-to-live in seconds (default 3600).
    """
    if not results:
        return
    mapping = {name: json.dumps(result) for name, result in results.items()}
    await redis.hset(key, mapping=mapping)
    await redis.expire(key, ttl_seconds)
