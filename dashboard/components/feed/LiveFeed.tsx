"use client";

import { useWebSocketFeed } from "@/lib/websocket";
import { FeedItem } from "./FeedItem";

export function LiveFeed() {
  const { events, connected } = useWebSocketFeed();

  return (
    <div className="border border-[#1a1a1a] rounded-lg overflow-hidden flex flex-col" style={{ height: "calc(100vh - 148px)" }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1a1a1a] shrink-0">
        <h2 className="text-[10px] text-zinc-500 uppercase tracking-widest">Real-time Request Feed</h2>
        <div className="flex items-center gap-2">
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-white animate-pulse" : "bg-zinc-700"}`} />
          <span className="text-[10px] text-zinc-600">{connected ? "connected" : "reconnecting…"}</span>
          <span className="text-[10px] text-zinc-800 ml-1">{events.length}</span>
        </div>
      </div>

      {/* Feed */}
      <div className="overflow-y-auto flex-1 bg-black">
        {events.length === 0 ? (
          <div className="flex items-center justify-center h-full text-zinc-700 text-xs">
            {connected ? "waiting for requests…" : "connecting to server…"}
          </div>
        ) : (
          events.map((event) => (
            <FeedItem key={`${event.request_id}-${event.latency_total}`} event={event} />
          ))
        )}
      </div>
    </div>
  );
}
