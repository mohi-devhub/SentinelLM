"""Unit tests for the cross-replica WebSocket fanout in sentinel.ws.broadcaster.

publish_event() and run_pubsub_listener() are what make the dashboard feed
correct behind more than one API replica — a request scored on replica B must
still reach a dashboard client connected to replica A. These tests use fake
Redis/pubsub stubs (no real Redis needed), matching the style already used in
tests/unit/cache/test_client.py and tests/unit/ws/test_broadcaster.py.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sentinel.ws.broadcaster import CHANNEL, ConnectionManager, publish_event, run_pubsub_listener

# ── Fakes ────────────────────────────────────────────────────────────────────


class FakePubSub:
    """Replays a pre-seeded list of pub/sub messages, then stops."""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages
        self.subscribed_channel: str | None = None
        self.unsubscribed_channel: str | None = None
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        self.subscribed_channel = channel

    async def listen(self):
        for message in self._messages:
            yield message

    async def unsubscribe(self, channel: str) -> None:
        self.unsubscribed_channel = channel

    async def aclose(self) -> None:
        self.closed = True


class FakeRedis:
    def __init__(self, pubsub_messages: list[dict] | None = None) -> None:
        self.published: list[tuple[str, str]] = []
        self._pubsub = FakePubSub(pubsub_messages or [])

    async def publish(self, channel: str, data: str) -> None:
        self.published.append((channel, data))

    def pubsub(self) -> FakePubSub:
        return self._pubsub


# ── publish_event ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_event_publishes_to_the_fanout_channel():
    redis = FakeRedis()
    event = {"event_type": "request_passed", "request_id": "abc"}

    await publish_event(redis, event)

    assert len(redis.published) == 1
    channel, payload = redis.published[0]
    assert channel == CHANNEL
    assert json.loads(payload) == event


@pytest.mark.asyncio
async def test_publish_event_falls_back_to_local_broadcast_when_redis_is_none(monkeypatch):
    broadcast_calls = []

    async def fake_broadcast(event):
        broadcast_calls.append(event)

    monkeypatch.setattr("sentinel.ws.broadcaster.manager.broadcast", fake_broadcast)

    event = {"event_type": "request_passed"}
    await publish_event(None, event)

    assert broadcast_calls == [event]


@pytest.mark.asyncio
async def test_publish_event_falls_back_to_local_broadcast_on_redis_error(monkeypatch):
    broadcast_calls = []

    async def fake_broadcast(event):
        broadcast_calls.append(event)

    monkeypatch.setattr("sentinel.ws.broadcaster.manager.broadcast", fake_broadcast)

    redis = AsyncMock()
    redis.publish.side_effect = ConnectionError("redis down")

    event = {"event_type": "request_passed"}
    await publish_event(redis, event)  # must not raise

    assert broadcast_calls == [event]


# ── run_pubsub_listener ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_listener_subscribes_to_the_fanout_channel():
    redis = FakeRedis(pubsub_messages=[])
    await run_pubsub_listener(redis)
    assert redis._pubsub.subscribed_channel == CHANNEL


@pytest.mark.asyncio
async def test_listener_relays_published_events_to_local_connections():
    event = {"event_type": "request_blocked", "request_id": "xyz"}
    messages = [
        {"type": "subscribe", "data": 1},  # subscription confirmation — must be ignored
        {"type": "message", "data": json.dumps(event)},
    ]
    redis = FakeRedis(pubsub_messages=messages)

    cm = ConnectionManager()
    ws = AsyncMock()
    cm._connections[ws] = "tenant-a"

    import sentinel.ws.broadcaster as broadcaster_module

    original_manager = broadcaster_module.manager
    broadcaster_module.manager = cm
    try:
        await run_pubsub_listener(redis)
    finally:
        broadcaster_module.manager = original_manager

    ws.send_json.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_listener_drops_malformed_messages_without_crashing():
    messages = [{"type": "message", "data": "not valid json"}]
    redis = FakeRedis(pubsub_messages=messages)

    # Must complete without raising — a bad payload shouldn't kill the listener.
    await run_pubsub_listener(redis)


@pytest.mark.asyncio
async def test_listener_unsubscribes_and_closes_pubsub_when_done():
    redis = FakeRedis(pubsub_messages=[])
    await run_pubsub_listener(redis)

    assert redis._pubsub.unsubscribed_channel == CHANNEL
    assert redis._pubsub.closed is True
