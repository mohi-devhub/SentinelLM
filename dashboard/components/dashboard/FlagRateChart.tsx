"use client";

import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchAggregateMetrics } from "@/lib/api";
import type { AggregateMetricsResponse, MetricBucket } from "@/lib/types";

const EVALUATORS: { key: keyof MetricBucket; label: string }[] = [
  { key: "flag_rate_pii",              label: "PII" },
  { key: "flag_rate_prompt_injection", label: "Injection" },
  { key: "flag_rate_topic_guardrail",  label: "Topic" },
  { key: "flag_rate_toxicity",         label: "Toxicity" },
  { key: "flag_rate_relevance",        label: "Relevance" },
  { key: "flag_rate_hallucination",    label: "Hallucination" },
  { key: "flag_rate_faithfulness",     label: "Faithfulness" },
];

// Grayscale range — stops at #444 so the darkest bar stays visible on #0d0d0d bg
const BAR_GRAYS = ["#e5e5e5", "#c2c2c2", "#9a9a9a", "#777777", "#5a5a5a", "#484848", "#444444"];

export function FlagRateChart() {
  const { data, isLoading } = useQuery<AggregateMetricsResponse>({
    queryKey: ["aggregate-metrics", "24h"],
    queryFn: () => fetchAggregateMetrics("24h", "1h"),
    refetchInterval: 60_000,
  });

  const chartData = EVALUATORS.map(({ key, label }) => {
    const buckets = data?.buckets ?? [];
    const vals = buckets.map((b) => b[key] as number).filter((v) => v != null);
    const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    return { label, rate: +(avg * 100).toFixed(2) };
  });

  return (
    <div className="border border-[#1a1a1a] rounded-lg p-5 bg-[#0d0d0d]">
      <h2 className="text-[10px] text-zinc-600 uppercase tracking-widest mb-5">
        Avg Flag Rate by Evaluator (24h)
      </h2>

      {isLoading ? (
        <div className="h-[260px] flex items-center justify-center text-zinc-700 text-xs">loading…</div>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" vertical={false} />
            <XAxis
              dataKey="label"
              interval={0}
              angle={-35}
              textAnchor="end"
              height={60}
              tick={{ fill: "#555", fontSize: 10, fontFamily: "var(--font-mono)" }}
              tickLine={false}
              axisLine={{ stroke: "#1a1a1a" }}
            />
            <YAxis
              unit="%"
              tick={{ fill: "#555", fontSize: 10, fontFamily: "var(--font-mono)" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              cursor={{ fill: "#ffffff08" }}
              contentStyle={{
                background: "#000",
                border: "1px solid #1a1a1a",
                borderRadius: 4,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "#aaa",
              }}
              formatter={(val) => [`${val ?? 0}%`, "flag rate"]}
            />
            <Bar dataKey="rate" radius={[2, 2, 0, 0]} maxBarSize={40}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={BAR_GRAYS[i % BAR_GRAYS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
