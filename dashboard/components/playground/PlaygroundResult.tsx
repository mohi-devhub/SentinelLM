import { SentinelBlockError } from "@/lib/api";
import type { PlaygroundSuccessResponse } from "@/lib/types";

// Unlike FeedItem's ScorePill (which hides null scores — correct for a dense
// feed), the playground shows *why* an evaluator didn't run: seeing
// "hallucination: skipped" when no context documents were given is the point.
function ResultScorePill({ name, value }: { name: string; value: number | null }) {
  const label = name.replace(/_/g, " ");
  if (value === null) {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] text-zinc-800">
        <span className="capitalize">{label}:</span> skipped
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-zinc-600">
      <span className="capitalize">{label}:</span>
      <span className="text-zinc-400">{value.toFixed(3)}</span>
    </span>
  );
}

export function PlaygroundResult({
  data,
  error,
  isPending,
  hasSubmitted,
  sentMessage,
}: {
  data?: PlaygroundSuccessResponse;
  error?: unknown;
  isPending: boolean;
  hasSubmitted: boolean;
  sentMessage: string;
}) {
  if (!hasSubmitted) {
    return <p className="text-xs text-zinc-700">Results will appear here.</p>;
  }

  if (isPending) {
    return <p className="text-xs text-zinc-600">scoring…</p>;
  }

  if (error instanceof SentinelBlockError) {
    const { block } = error;
    return (
      <div className="border border-[#1a1a1a] rounded-lg p-5 bg-[#0d0d0d]">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-white text-black">
            blocked
          </span>
          <span className="text-[10px] text-zinc-600">{block.code}</span>
        </div>
        <p className="text-xs text-zinc-400 mb-2">{block.message}</p>
        <p className="text-[10px] text-zinc-600">
          score <span className="text-zinc-400">{block.score.toFixed(3)}</span> · threshold{" "}
          <span className="text-zinc-400">{block.threshold.toFixed(3)}</span>
        </p>
      </div>
    );
  }

  if (error) {
    return <p className="text-xs text-zinc-500">Request failed. Check the API is running.</p>;
  }

  if (!data) return null;

  const { sentinel } = data;
  const statusLabel = sentinel.flags.length ? "flagged" : "pass";
  const statusClass = sentinel.flags.length
    ? "border border-zinc-600 text-zinc-300"
    : "text-zinc-600";

  const showRedactionDiff =
    sentinel.pii_redacted_text !== null && sentinel.pii_redacted_text !== sentMessage;

  return (
    <div className="border border-[#1a1a1a] rounded-lg p-5 bg-[#0d0d0d] space-y-4">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${statusClass}`}>
          {statusLabel}
        </span>
        {sentinel.flags.map((f) => (
          <span
            key={f}
            className="text-[10px] px-1.5 py-0.5 rounded border border-[#2a2a2a] text-zinc-500"
          >
            {f.replace(/_/g, " ")}
          </span>
        ))}
      </div>

      <div>
        <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Response</p>
        <p className="text-xs text-zinc-400 bg-black border border-[#1a1a1a] rounded px-3 py-2 whitespace-pre-wrap">
          {data.choices[0]?.message.content ?? "(no content)"}
        </p>
      </div>

      {showRedactionDiff && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Original</p>
            <p className="text-xs text-zinc-400 bg-black border border-[#1a1a1a] rounded px-3 py-2 whitespace-pre-wrap">
              {sentMessage}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Redacted</p>
            <p className="text-xs text-zinc-400 bg-black border border-[#1a1a1a] rounded px-3 py-2 whitespace-pre-wrap">
              {sentinel.pii_redacted_text}
            </p>
          </div>
        </div>
      )}

      <div>
        <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Evaluators</p>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {(Object.entries(sentinel.scores) as [string, number | null][]).map(([k, v]) => (
            <ResultScorePill key={k} name={k} value={v} />
          ))}
        </div>
      </div>

      <p className="text-[10px] text-zinc-700">{sentinel.latency_ms.total}ms total</p>
    </div>
  );
}
