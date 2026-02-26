from __future__ import annotations

import asyncpg


async def create_pool(database_url: str) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

    Called once at app startup in the FastAPI lifespan and stored on app.state.
    Never call this per-request — pool creation is expensive.
    """
    return await asyncpg.create_pool(database_url)


def get_pool(app_state) -> asyncpg.Pool:
    """Retrieve the connection pool from FastAPI app state."""
    return app_state.db_pool
