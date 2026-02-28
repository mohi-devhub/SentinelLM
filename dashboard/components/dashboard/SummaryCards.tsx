"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchSummaryMetrics } from "@/lib/api";
import type { SummaryMetrics } from "@/lib/types";

function Card({ title, value, sub }: { title: string; value: string; sub?: string }) {
  return (
    <div className="border border-[#1a1a1a] rounded-lg p-5 bg-[#0d0d0d]">
      <p className="text-[10px] text-zinc-600 uppercase tracking-widest">{title}</p>
      <p className="text-3xl font-semibold text-white mt-2 tabular-nums">{value}</p>
      {sub && <p className="text-xs text-zinc-600 mt-1">{sub}</p>}
    </div>
  );
}

export function SummaryCards() {
  const { data } = useQuery<SummaryMetrics>({
    queryKey: ["summary-metrics"],
    queryFn: fetchSummaryMetrics,
    refetchInterval: 30_000,
  });

  const total     = data?.total_requests_24h ?? "—";
  const blocked   = data?.blocked_requests_24h ?? "—";
  const blockRate = data != null ? `${(data.block_rate_24h * 100).toFixed(1)}%` : "—";
  const latency   = data != null ? `${data.avg_latency_24h_ms}ms` : "—";
  const unreviewed = data?.unreviewed_flags ?? "—";
  const topFlag   = data?.top_flag_reason?.replace(/_/g, " ") ?? "—";

  return (
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
      <Card title="Requests (24h)"   value={String(total)}      sub={`${blocked} blocked`} />
      <Card title="Block Rate"       value={blockRate}           sub={`top flag: ${topFlag}`} />
      <Card title="Avg Latency"      value={latency}             sub="middleware overhead" />
      <Card title="Unreviewed Flags" value={String(unreviewed)}  sub="awaiting review" />
    </div>
  );
}
