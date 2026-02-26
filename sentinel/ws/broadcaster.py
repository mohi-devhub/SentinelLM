from __future__ import annotations

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for the real-time dashboard feed.

    Singleton — import `manager` directly; do not instantiate elsewhere.
    """

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        logger.debug("ws client connected; total=%d", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)
        logger.debug("ws client disconnected; total=%d", len(self._connections))

    async def broadcast(self, event: dict) -> None:
        """Push a JSON event to all connected clients.

        Dead connections (closed browser, lost network) are removed silently.
        """
        dead: set[WebSocket] = set()
        for ws in self._connections:
            try:
                await ws.send_json(event)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Global singleton — imported by proxy handler and websocket route
manager = ConnectionManager()
