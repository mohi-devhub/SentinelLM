"""Tenancy startup bootstrap — resolves the default tenant and, when a legacy
SENTINEL_API_KEY is configured, attaches it to that tenant so existing
single-tenant deployments keep working unchanged after upgrading.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from sentinel.tenancy.keys import hash_key, key_prefix
from sentinel.tenancy.queries import get_default_tenant, upsert_default_tenant_key

logger = logging.getLogger(__name__)


async def bootstrap_tenancy(pool: asyncpg.Pool, legacy_api_key: str) -> dict[str, Any]:
    """Return the default tenant dict for caching on app.state.default_tenant."""
    default_tenant = await get_default_tenant(pool)
    if legacy_api_key:
        await upsert_default_tenant_key(pool, hash_key(legacy_api_key), key_prefix(legacy_api_key))
        logger.info("legacy SENTINEL_API_KEY attached to default tenant")
    return default_tenant
