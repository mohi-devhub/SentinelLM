from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sentinel.ws.broadcaster import manager

router = APIRouter()


@router.websocket("/ws/feed")
async def ws_feed(websocket: WebSocket) -> None:
    """Real-time scored request feed.

    Pushes a SentinelEvent JSON message for every completed request (pass or
    block). Used by the Next.js dashboard LiveFeed component.
    """
    await manager.connect(websocket)
    try:
        # Hold the connection open; all traffic flows server → client via broadcast()
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
