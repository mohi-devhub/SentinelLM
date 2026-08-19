"""Unit tests for sentinel.ws.broadcaster.ConnectionManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sentinel.ws.broadcaster import ConnectionManager, manager

# ── ConnectionManager ─────────────────────────────────────────────────────────


@pytest.fixture
def cm() -> ConnectionManager:
    return ConnectionManager()


@pytest.fixture
def mock_ws() -> MagicMock:
    ws = AsyncMock()
    return ws


def test_initial_connection_count(cm):
    assert cm.connection_count == 0


@pytest.mark.asyncio
async def test_connect_accepts_websocket(cm, mock_ws):
    await cm.connect(mock_ws, "tenant-a")
    mock_ws.accept.assert_awaited_once()
    assert cm.connection_count == 1


@pytest.mark.asyncio
async def test_connect_multiple_clients(cm):
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    await cm.connect(ws1, "tenant-a")
    await cm.connect(ws2, "tenant-b")
    assert cm.connection_count == 2


def test_disconnect_removes_client(cm, mock_ws):
    cm._connections[mock_ws] = "tenant-a"
    assert cm.connection_count == 1
    cm.disconnect(mock_ws)
    assert cm.connection_count == 0


def test_disconnect_unknown_client_is_noop(cm, mock_ws):
    """Disconnecting a client that was never connected is a no-op."""
    cm.disconnect(mock_ws)  # should not raise
    assert cm.connection_count == 0


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_clients_when_event_has_no_tenant():
    cm = ConnectionManager()
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    cm._connections[ws1] = "tenant-a"
    cm._connections[ws2] = "tenant-b"

    event = {"type": "score", "data": {"pii": 0.1}}
    await cm.broadcast(event)

    ws1.send_json.assert_awaited_once_with(event)
    ws2.send_json.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_broadcast_only_reaches_the_owning_tenant(cm):
    ws_a = AsyncMock()
    ws_b = AsyncMock()
    cm._connections[ws_a] = "tenant-a"
    cm._connections[ws_b] = "tenant-b"

    event = {"type": "score", "tenant_id": "tenant-a"}
    await cm.broadcast(event)

    ws_a.send_json.assert_awaited_once_with(event)
    ws_b.send_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections(cm):
    """A WebSocket that raises during send_json is removed silently."""
    live_ws = AsyncMock()
    dead_ws = AsyncMock()
    dead_ws.send_json = AsyncMock(side_effect=RuntimeError("connection closed"))

    cm._connections[live_ws] = "tenant-a"
    cm._connections[dead_ws] = "tenant-a"

    await cm.broadcast({"type": "ping"})

    # Dead connection is removed; live connection is untouched
    assert dead_ws not in cm._connections
    assert live_ws in cm._connections
    live_ws.send_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_broadcast_empty_connections(cm):
    """Broadcast with no connected clients should not raise."""
    await cm.broadcast({"type": "empty"})
    assert cm.connection_count == 0


@pytest.mark.asyncio
async def test_connect_then_disconnect_full_cycle(cm):
    ws = AsyncMock()
    await cm.connect(ws, "tenant-a")
    assert cm.connection_count == 1
    cm.disconnect(ws)
    assert cm.connection_count == 0


# ── Global singleton ──────────────────────────────────────────────────────────


def test_global_manager_instance():
    """The module-level `manager` singleton should be a ConnectionManager."""
    assert isinstance(manager, ConnectionManager)
