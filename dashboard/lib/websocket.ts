"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getApiKey } from "./auth";
import type { SentinelEvent } from "./types";

const WS_BASE =
  (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/^http/, "ws") +
  "/ws/feed";

const MAX_EVENTS = 100;
const RECONNECT_DELAY_MS = 3000;

// Browsers cannot set custom headers on a WebSocket handshake, so the key
// is passed as a query param instead of X-API-Key — the server mirrors this
// (see sentinel/api/websocket.py's _resolve_tenant).
function wsUrl(): string {
  const apiKey = getApiKey();
  return apiKey ? `${WS_BASE}?api_key=${encodeURIComponent(apiKey)}` : WS_BASE;
}

export function useWebSocketFeed() {
  const [events, setEvents] = useState<SentinelEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmounted = useRef(false);

  const connect = useCallback(() => {
    if (unmounted.current) return;

    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (ev) => {
      try {
        const event: SentinelEvent = JSON.parse(ev.data as string);
        setEvents((prev) => [event, ...prev].slice(0, MAX_EVENTS));
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (!unmounted.current) {
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    unmounted.current = false;
    connect();
    return () => {
      unmounted.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { events, connected };
}
