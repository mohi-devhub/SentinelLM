from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Redis pub/sub channel used to fan events out across API replicas — see
# publish_event() and run_pubsub_listener() below.
CHANNEL = "sentinel:ws:feed"


class ConnectionManager:
    """Manages active WebSocket connections for the real-time dashboard feed.

    Singleton — import `manager` directly; do not instantiate elsewhere.
    """

    def __init__(self) -> None:
        self._connections: dict[WebSocket, str] = {}  # ws -> tenant_id (str)

    async def connect(self, websocket: WebSocket, tenant_id: object) -> None:
        await websocket.accept()
        self._connections[websocket] = str(tenant_id)
        logger.debug("ws client connected; total=%d", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.pop(websocket, None)
        logger.debug("ws client disconnected; total=%d", len(self._connections))

    async def broadcast(self, event: dict) -> None:
        """Push a JSON event to every connection belonging to the event's tenant.

        Events with no tenant_id (pre-tenancy or malformed) are delivered to
        every connection — dead connections (closed browser, lost network)
        are removed silently.
        """
        event_tenant_id = event.get("tenant_id")
        dead: set[WebSocket] = set()
        for ws, tenant_id in self._connections.items():
            if event_tenant_id is not None and tenant_id != event_tenant_id:
                continue
            try:
                await ws.send_json(event)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections.pop(ws, None)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Global singleton — imported by proxy handler and websocket route
manager = ConnectionManager()


# ── Cross-replica fanout ──────────────────────────────────────────────────────
#
# ConnectionManager.broadcast() only reaches WebSocket clients connected to
# *this* process. Behind more than one API replica, a request scored on
# replica B never reaches a dashboard client connected to replica A. These
# two functions fix that: publish_event() publishes to Redis instead of
# delivering locally, and run_pubsub_listener() — one instance per replica,
# started in the FastAPI lifespan — receives every published event and hands
# it to this replica's own ConnectionManager.broadcast() for local delivery.
# A single-instance deployment still works correctly: it publishes to itself
# and its own listener relays the message straight back to its own clients.


async def publish_event(redis: Any, event: dict) -> None:
    """Publish a scored-request event for delivery to every API replica.

    Falls back to direct local delivery when Redis is unavailable (no redis
    client configured, or the publish call itself fails) so a Redis outage
    degrades the feed to single-replica-only rather than losing events.
    """
    if redis is None:
        await manager.broadcast(event)
        return
    try:
        await redis.publish(CHANNEL, json.dumps(event))
    except Exception:
        logger.exception("failed to publish ws event to redis; broadcasting locally instead")
        await manager.broadcast(event)


async def run_pubsub_listener(redis: Any) -> None:
    """Background task: relay every event published on CHANNEL to local clients.

    Runs for the lifetime of the app (started and cancelled in the FastAPI
    lifespan). Malformed messages are logged and dropped rather than crashing
    the listener — one bad payload should never take down the feed.
    """
    pubsub = redis.pubsub()
    await pubsub.subscribe(CHANNEL)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                event = json.loads(message["data"])
            except (TypeError, ValueError):
                logger.warning("dropping malformed ws pubsub message")
                continue
            await manager.broadcast(event)
    finally:
        await pubsub.unsubscribe(CHANNEL)
        await pubsub.aclose()
