"""Generic JSON webhook sender for alert notifications.

Sends a Slack-compatible `{"text": ...}` payload — Slack incoming webhooks
consume it directly, and it's readable JSON for any other endpoint (a
generic logging/notification sink, a custom handler, etc).
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


async def send_alert(url: str, title: str, lines: list[str]) -> None:
    """POST an alert to a webhook URL.

    Logs and swallows failures — a broken alert channel must never affect
    request handling.
    """
    text = f"*{title}*\n" + "\n".join(f"- {line}" for line in lines)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, json={"text": text})
            response.raise_for_status()
    except Exception:
        logger.exception("failed to deliver alert webhook")
