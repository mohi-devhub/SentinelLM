from __future__ import annotations

import logging
import secrets

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from sentinel.ws.broadcaster import manager

logger = logging.getLogger(__name__)

router = APIRouter()


async def _resolve_tenant(websocket: WebSocket, api_key: str | None) -> dict | None:
    """Resolve the tenant for a WS connection, mirroring APIKeyMiddleware's logic.

    Browsers cannot set custom headers on a WebSocket handshake, so the key
    is passed as a query parameter (`?api_key=...`) instead of X-API-Key.
    Returns None when auth is required and no valid key was presented.
    """
    from sentinel.settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    default_tenant = getattr(websocket.app.state, "default_tenant", None)
    auth_required = bool(settings.api_key)

    if auth_required and api_key is not None and secrets.compare_digest(api_key, settings.api_key):
        return default_tenant

    if api_key:
        from sentinel.tenancy.keys import hash_key  # noqa: PLC0415
        from sentinel.tenancy.queries import get_tenant_by_key_hash  # noqa: PLC0415

        try:
            tenant = await get_tenant_by_key_hash(websocket.app.state.db_pool, hash_key(api_key))
        except Exception:
            logger.exception("ws tenant key lookup failed")
            tenant = None
        if tenant is not None:
            return tenant

    if not auth_required:
        return default_tenant

    return None


@router.websocket("/ws/feed")
async def ws_feed(websocket: WebSocket, api_key: str | None = Query(default=None)) -> None:
    """Real-time scored request feed, scoped to the connecting tenant.

    Pushes a SentinelEvent JSON message for every completed request (pass or
    block) belonging to the same tenant as the connection's API key. Used by
    the Next.js dashboard LiveFeed component.
    """
    tenant = await _resolve_tenant(websocket, api_key)
    if tenant is None:
        await websocket.close(code=4401)
        return

    await manager.connect(websocket, tenant["tenant_id"])
    try:
        # Hold the connection open; all traffic flows server → client via broadcast()
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
