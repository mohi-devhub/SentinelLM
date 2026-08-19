from __future__ import annotations

import logging
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

_SCHEMA_DIR = Path(__file__).parent
_MIGRATIONS_DIR = _SCHEMA_DIR / "migrations"


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Apply schema.sql then every migrations/v*.sql file, in filename order.

    Every statement in schema.sql and each migration uses IF NOT EXISTS /
    IF EXISTS / ON CONFLICT guards, so this is safe to run on every startup —
    a fresh database gets the full schema, an already-migrated one is a no-op.
    """
    files = [_SCHEMA_DIR / "schema.sql", *sorted(_MIGRATIONS_DIR.glob("v*.sql"))]
    async with pool.acquire() as conn:
        for path in files:
            await conn.execute(path.read_text())
            logger.info("applied %s", path.name)
