"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchEvalRuns, fetchEvalRun } from "@/lib/api";
import type { EvalRunSummary, EvalRunDetail } from "@/lib/types";
import { ScorecardTable } from "@/components/eval/ScorecardTable";

function RunRow({
  run,
  selected,
  onClick,
}: {
  run: EvalRunSummary;
  selected: boolean;
  onClick: () => void;
}) {
  const statusDot =
    run.status === "complete" ? "bg-white" :
    run.status === "running"  ? "bg-zinc-500 animate-pulse" :
    "bg-zinc-700";

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-4 py-3 border-b border-[#111] transition-colors ${
        selected ? "bg-[#111]" : "hover:bg-[#0a0a0a]"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-zinc-300 truncate">{run.label}</span>
        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${statusDot}`} />
      </div>
      <div className="flex gap-3 mt-1 text-[10px] text-zinc-700">
        <span>{run.record_count} records</span>
        <span>{run.created_at ? new Date(run.created_at).toLocaleDateString() : "—"}</span>
      </div>
    </button>
  );
}

function RunDetail({ runId }: { runId: string }) {
  const { data, isLoading } = useQuery<EvalRunDetail>({
    queryKey: ["eval-run", runId],
    queryFn: () => fetchEvalRun(runId),
  });

  if (isLoading) return <div className="text-xs text-zinc-700 p-6">loading…</div>;
  if (!data) return null;

  const hasRegression = data.regression
    ? Object.values(data.regression).some((r) => r.regressed)
    : false;

  return (
    <div className="p-6">
      <div className="flex items-center gap-3 mb-6">
        <h2 className="text-sm font-semibold text-white">{data.label}</h2>
        {data.regression && (
          <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${
            hasRegression
              ? "border-zinc-600 text-zinc-300"
              : "border-[#2a2a2a] text-zinc-600"
          }`}>
            {hasRegression ? "regression" : "no regression"}
          </span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3 mb-6">
        <div className="border border-[#1a1a1a] rounded-lg p-4 bg-[#0d0d0d]">
          <p className="text-[10px] text-zinc-700 uppercase tracking-widest">Records</p>
          <p className="text-2xl font-semibold text-white mt-1 tabular-nums">{data.record_count}</p>
        </div>
        <div className="border border-[#1a1a1a] rounded-lg p-4 bg-[#0d0d0d]">
          <p className="text-[10px] text-zinc-700 uppercase tracking-widest">Dataset</p>
          <p className="text-xs text-zinc-400 mt-1 truncate">{data.dataset_path}</p>
        </div>
        <div className="border border-[#1a1a1a] rounded-lg p-4 bg-[#0d0d0d]">
          <p className="text-[10px] text-zinc-700 uppercase tracking-widest">Completed</p>
          <p className="text-xs text-zinc-400 mt-1">
            {data.completed_at ? new Date(data.completed_at).toLocaleString() : "—"}
          </p>
        </div>
      </div>

      <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-3">Scorecard</p>
      <div className="border border-[#1a1a1a] rounded-lg p-4 bg-[#0d0d0d]">
        <ScorecardTable run={data} />
      </div>
    </div>
  );
}

export default function EvalPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { data: runs, isLoading } = useQuery<EvalRunSummary[]>({
    queryKey: ["eval-runs"],
    queryFn: fetchEvalRuns,
    refetchInterval: 15_000,
  });

  return (
    <div className="flex h-full" style={{ height: "calc(100vh - 0px)" }}>
      {/* List */}
      <div className="w-64 shrink-0 border-r border-[#1a1a1a] overflow-y-auto bg-[#050505]">
        <div className="px-4 py-3 border-b border-[#1a1a1a]">
          <h1 className="text-[10px] text-zinc-600 uppercase tracking-widest">Eval Runs</h1>
        </div>
        {isLoading ? (
          <div className="text-xs text-zinc-700 p-4">loading…</div>
        ) : !runs || runs.length === 0 ? (
          <div className="text-xs text-zinc-700 p-4">no eval runs yet.</div>
        ) : (
          runs.map((r) => (
            <RunRow
              key={r.id}
              run={r}
              selected={selectedId === r.id}
              onClick={() => setSelectedId(r.id)}
            />
          ))
        )}
      </div>

      {/* Detail */}
      <div className="flex-1 overflow-y-auto">
        {selectedId ? (
          <RunDetail runId={selectedId} />
        ) : (
          <div className="flex items-center justify-center h-full text-xs text-zinc-700">
            select an eval run to view its scorecard
          </div>
        )}
      </div>
    </div>
  );
}
