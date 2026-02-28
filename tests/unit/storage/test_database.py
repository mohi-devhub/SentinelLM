"""Unit tests for storage.database.

asyncpg.create_pool is mocked so no real database connection is needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_create_pool_calls_asyncpg_create_pool():
    """create_pool awaits asyncpg.create_pool and returns the pool."""
    mock_pool = MagicMock()
    with patch(
        "sentinel.storage.database.asyncpg.create_pool",
        new=AsyncMock(return_value=mock_pool),
    ):
        from sentinel.storage.database import create_pool

        result = await create_pool("postgresql://user:pass@localhost/testdb")

    assert result is mock_pool


@pytest.mark.asyncio
async def test_create_pool_passes_url_through():
    """create_pool forwards the database URL to asyncpg.create_pool."""
    url = "postgresql://sentinel:secret@db:5432/sentinel"
    mock_pool = MagicMock()
    with patch(
        "sentinel.storage.database.asyncpg.create_pool",
        new=AsyncMock(return_value=mock_pool),
    ) as mock_create:
        from sentinel.storage.database import create_pool

        await create_pool(url)

    mock_create.assert_awaited_once_with(url)


def test_get_pool_returns_db_pool_from_app_state():
    """get_pool reads .db_pool from app state and returns it."""
    from sentinel.storage.database import get_pool

    mock_pool = MagicMock()
    mock_state = MagicMock()
    mock_state.db_pool = mock_pool

    result = get_pool(mock_state)

    assert result is mock_pool
