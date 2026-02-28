"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchAggregateMetrics } from "@/lib/api";
import type { AggregateMetricsResponse } from "@/lib/types";

const WINDOWS = ["1h", "6h", "24h", "7d", "30d"] as const;

const LINES = [
  { key: "requests",  label: "Requests",    color: "#e5e5e5", width: 1.5, dash: undefined },
  { key: "blocked",   label: "Blocked",     color: "#888888", width: 1.5, dash: undefined },
  { key: "toxicity",  label: "Toxicity %",  color: "#666666", width: 1,   dash: "4 3" },
  { key: "injection", label: "Injection %", color: "#444444", width: 1,   dash: "4 3" },
] as const;

function fmt(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function MetricsChart() {
  const [window, setWindow] = useState<string>("24h");

  const { data, isLoading } = useQuery<AggregateMetricsResponse>({
    queryKey: ["aggregate-metrics", window],
    queryFn: () => fetchAggregateMetrics(window, "1h"),
    refetchInterval: 60_000,
  });

  const chartData = (data?.buckets ?? []).map((b) => ({
    time:      fmt(b.bucket_start),
    requests:  b.request_count,
    blocked:   b.blocked_count,
    toxicity:  b.flag_rate_toxicity         != null ? +(b.flag_rate_toxicity         * 100).toFixed(1) : null,
    injection: b.flag_rate_prompt_injection != null ? +(b.flag_rate_prompt_injection * 100).toFixed(1) : null,
  }));

  return (
    <div className="border border-[#1a1a1a] rounded-lg p-5 bg-[#0d0d0d]">
      {/* Header row */}
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-[10px] text-zinc-600 uppercase tracking-widest">
          Request Volume & Flag Rates
        </h2>
        <div className="flex gap-1">
          {WINDOWS.map((w) => (
            <button
              key={w}
              onClick={() => setWindow(w)}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                window === w
                  ? "bg-white text-black font-medium"
                  : "text-zinc-600 hover:text-white"
              }`}
            >
              {w}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="h-[260px] flex items-center justify-center text-zinc-700 text-xs">loading…</div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fill: "#555", fontSize: 10, fontFamily: "var(--font-mono)" }}
                tickLine={false}
                axisLine={{ stroke: "#1a1a1a" }}
              />
              <YAxis
                tick={{ fill: "#555", fontSize: 10, fontFamily: "var(--font-mono)" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                cursor={{ stroke: "#2a2a2a", strokeWidth: 1 }}
                contentStyle={{
                  background: "#000",
                  border: "1px solid #1a1a1a",
                  borderRadius: 4,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "#aaa",
                }}
                labelStyle={{ color: "#666", marginBottom: 4 }}
              />
              {LINES.map(({ key, color, width, dash }) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={color}
                  strokeWidth={width}
                  strokeDasharray={dash}
                  dot={false}
                  legendType="none"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>

          {/* Manual legend — consistent sizing and color matching */}
          <div className="flex items-center gap-5 mt-3 px-1">
            {LINES.map(({ key, label, color, dash }) => (
              <div key={key} className="flex items-center gap-1.5">
                <svg width="18" height="8">
                  <line
                    x1="0" y1="4" x2="18" y2="4"
                    stroke={color}
                    strokeWidth="1.5"
                    strokeDasharray={dash}
                  />
                </svg>
                <span className="text-[10px]" style={{ color }}>{label}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
