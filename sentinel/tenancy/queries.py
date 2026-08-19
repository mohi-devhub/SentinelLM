"""DB queries for tenants and api_keys."""

from __future__ import annotations

import uuid as _uuid
from typing import Any
from uuid import UUID

import asyncpg


async def get_tenant_by_key_hash(pool: asyncpg.Pool, key_hash: str) -> dict[str, Any] | None:
    """Look up the tenant owning an active (non-revoked) key hash.

    Best-effort touches last_used_at — a failure there must never break auth.
    """
    query = """
        SELECT t.id AS tenant_id, t.slug AS tenant_slug, ak.id AS api_key_id
        FROM api_keys ak
        JOIN tenants t ON t.id = ak.tenant_id
        WHERE ak.key_hash = $1 AND ak.revoked_at IS NULL
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, key_hash)
        if row is None:
            return None
        try:
            await conn.execute(
                "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1", row["api_key_id"]
            )
        except Exception:
            pass
    return {"tenant_id": row["tenant_id"], "tenant_slug": row["tenant_slug"]}


async def get_default_tenant(pool: asyncpg.Pool) -> dict[str, Any]:
    """Return the seeded default tenant's id/slug (always present after migration)."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, slug FROM tenants WHERE is_default = TRUE LIMIT 1")
    return {"tenant_id": row["id"], "tenant_slug": row["slug"]}


async def upsert_default_tenant_key(pool: asyncpg.Pool, key_hash: str, prefix: str) -> None:
    """Idempotently attach a hashed key to the default tenant (backward compat).

    Called at every app startup when SENTINEL_API_KEY is set. ON CONFLICT
    makes this a no-op after the first run.
    """
    query = """
        INSERT INTO api_keys (tenant_id, key_hash, key_prefix, label)
        SELECT id, $1, $2, 'legacy SENTINEL_API_KEY'
        FROM tenants WHERE is_default = TRUE
        ON CONFLICT (key_hash) DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.execute(query, key_hash, prefix)


async def create_tenant(pool: asyncpg.Pool, name: str, slug: str) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO tenants (name, slug) VALUES ($1, $2)
            RETURNING id, created_at, name, slug
            """,
            name,
            slug,
        )
    return dict(row)


async def get_tenant_by_slug(pool: asyncpg.Pool, slug: str) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, slug, is_default FROM tenants WHERE slug = $1", slug
        )
    return dict(row) if row else None


async def list_tenants(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, created_at, name, slug, is_default FROM tenants ORDER BY created_at"
        )
    return [dict(r) for r in rows]


async def create_api_key(
    pool: asyncpg.Pool,
    tenant_id: UUID,
    key_hash: str,
    prefix: str,
    label: str | None,
) -> dict[str, Any]:
    key_id = _uuid.uuid4()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO api_keys (id, tenant_id, key_hash, key_prefix, label)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, created_at
            """,
            key_id,
            tenant_id,
            key_hash,
            prefix,
            label,
        )
    return {"id": row["id"], "created_at": row["created_at"]}


async def list_api_keys(pool: asyncpg.Pool, tenant_id: UUID | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT ak.id, ak.tenant_id, t.slug AS tenant_slug, ak.key_prefix, ak.label,
               ak.created_at, ak.last_used_at, ak.revoked_at
        FROM api_keys ak
        JOIN tenants t ON t.id = ak.tenant_id
    """
    params: list[Any] = []
    if tenant_id is not None:
        query += " WHERE ak.tenant_id = $1"
        params.append(tenant_id)
    query += " ORDER BY ak.created_at DESC"
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [dict(r) for r in rows]


async def revoke_api_key(pool: asyncpg.Pool, key_id: UUID) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE api_keys SET revoked_at = NOW() WHERE id = $1 AND revoked_at IS NULL",
            key_id,
        )
    return result == "UPDATE 1"
