import type { SentinelEvent } from "@/lib/types";

function ScorePill({ name, value }: { name: string; value: number | null }) {
  if (value === null) return null;
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-zinc-600">
      <span className="capitalize">{name.replace(/_/g, " ")}:</span>
      <span className="text-zinc-400">{value.toFixed(3)}</span>
    </span>
  );
}

export function FeedItem({ event }: { event: SentinelEvent }) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  const statusLabel = event.blocked ? "blocked" : event.flags.length ? "flagged" : "pass";
  const statusClass = event.blocked
    ? "bg-white text-black"
    : event.flags.length
    ? "border border-zinc-600 text-zinc-300"
    : "text-zinc-600";

  const leftBorder = event.blocked
    ? "border-l border-l-white"
    : event.flags.length
    ? "border-l border-l-zinc-600"
    : "";

  return (
    <div className={`border-b border-[#1a1a1a] px-4 py-3 hover:bg-[#0d0d0d] transition-colors ${leftBorder}`}>
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 min-w-0">
          <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${statusClass}`}>
            {statusLabel}
          </span>
          <span className="text-[10px] text-zinc-600 font-mono truncate">{event.request_id.slice(0, 8)}</span>
          <span className="text-[10px] text-zinc-700">{event.model}</span>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <span className="text-[10px] text-zinc-700">{event.latency_total}ms</span>
          <span className="text-[10px] text-zinc-700">{time}</span>
        </div>
      </div>

      {/* Flags */}
      {event.flags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {event.flags.map((flag) => (
            <span key={flag} className="text-[10px] px-1.5 py-0.5 rounded border border-[#2a2a2a] text-zinc-500">
              {flag.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}

      {/* Scores */}
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1.5">
        {(Object.entries(event.scores) as [string, number | null][]).map(([k, v]) =>
          v !== null ? <ScorePill key={k} name={k} value={v} /> : null
        )}
      </div>
    </div>
  );
}
