"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { submitReview } from "@/lib/api";
import type { RequestDetail, ReviewLabel } from "@/lib/types";

const LABEL_OPTIONS: { value: ReviewLabel; label: string }[] = [
  { value: "correct_flag",   label: "Correct Flag" },
  { value: "false_positive", label: "False Positive" },
  { value: "false_negative", label: "False Negative" },
];

export function ReviewCard({ request }: { request: RequestDetail }) {
  const qc = useQueryClient();
  const [selectedLabel, setSelectedLabel] = useState<ReviewLabel | null>(null);
  const [note, setNote] = useState("");

  const mutation = useMutation({
    mutationFn: () => submitReview(request.id, selectedLabel!, note || undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["review-queue"] });
    },
  });

  const statusLabel = request.blocked ? "blocked" : "flagged";

  return (
    <div className="border border-[#1a1a1a] rounded-lg p-5 bg-[#0d0d0d]">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${
            request.blocked ? "bg-white text-black" : "border border-zinc-700 text-zinc-400"
          }`}>
            {statusLabel}
          </span>
          {request.flags.map((f) => (
            <span key={f} className="text-[10px] px-1.5 py-0.5 rounded border border-[#2a2a2a] text-zinc-600">
              {f.replace(/_/g, " ")}
            </span>
          ))}
        </div>
        <div className="text-right shrink-0">
          <span className="text-[10px] text-zinc-700 font-mono">{request.id.slice(0, 8)}</span>
          <div className="text-[10px] text-zinc-700 mt-0.5">
            {new Date(request.created_at).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Input */}
      {request.input_redacted && (
        <div className="mb-4">
          <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Input</p>
          <p className="text-xs text-zinc-400 bg-black border border-[#1a1a1a] rounded px-3 py-2 whitespace-pre-wrap line-clamp-4">
            {request.input_redacted}
          </p>
        </div>
      )}

      {/* Scores */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mb-4">
        {(Object.entries(request.scores) as [string, number | null][]).map(([k, v]) =>
          v !== null ? (
            <span key={k} className="text-[10px] text-zinc-600">
              <span className="capitalize">{k.replace(/_/g, " ")}: </span>
              <span className="text-zinc-400">{v.toFixed(3)}</span>
            </span>
          ) : null
        )}
      </div>

      {/* Review form / already reviewed */}
      {request.reviewed ? (
        <div className="border-t border-[#1a1a1a] pt-3 text-[10px] text-zinc-600">
          reviewed as <span className="text-zinc-400">{request.review_label?.replace(/_/g, " ")}</span>
          {request.reviewer_note && <span> — {request.reviewer_note}</span>}
        </div>
      ) : (
        <div className="border-t border-[#1a1a1a] pt-4 space-y-3">
          <p className="text-[10px] text-zinc-600 uppercase tracking-widest">Submit Review</p>
          <div className="flex gap-2">
            {LABEL_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setSelectedLabel(opt.value)}
                className={`flex-1 text-[10px] py-1.5 rounded border transition-colors ${
                  selectedLabel === opt.value
                    ? "bg-white text-black border-white"
                    : "border-[#2a2a2a] text-zinc-600 hover:border-zinc-600 hover:text-zinc-400"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <input
            type="text"
            placeholder="Optional note…"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            className="w-full text-xs bg-black border border-[#1a1a1a] rounded px-3 py-2 text-zinc-400 placeholder-zinc-800 focus:outline-none focus:border-zinc-700 font-mono"
          />
          <button
            disabled={!selectedLabel || mutation.isPending}
            onClick={() => mutation.mutate()}
            className="w-full text-xs py-2 rounded border border-zinc-700 text-zinc-400 hover:bg-white hover:text-black hover:border-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            {mutation.isPending ? "submitting…" : "Submit"}
          </button>
          {mutation.isError && (
            <p className="text-[10px] text-zinc-500">Failed to submit. Try again.</p>
          )}
        </div>
      )}
    </div>
  );
}
