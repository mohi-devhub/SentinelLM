import type { EvalRunDetail } from "@/lib/types";
import { RegressionBadge } from "./RegressionBadge";

function pct(v: number | null): string {
  return v !== null ? `${(v * 100).toFixed(1)}%` : "—";
}

function num(v: number | null): string {
  return v !== null ? v.toFixed(3) : "—";
}

export function ScorecardTable({ run }: { run: EvalRunDetail }) {
  const { scorecard, regression } = run;

  if (!scorecard || Object.keys(scorecard).length === 0) {
    return <p className="text-xs text-zinc-700">No scorecard data available.</p>;
  }

  const evaluators = Object.keys(scorecard);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[10px]">
        <thead>
          <tr className="text-zinc-700 border-b border-[#1a1a1a]">
            <th className="text-left py-2 pr-4 font-medium uppercase tracking-widest">Evaluator</th>
            <th className="text-right py-2 pr-4 font-medium">N</th>
            <th className="text-right py-2 pr-4 font-medium">Mean</th>
            <th className="text-right py-2 pr-4 font-medium">p50</th>
            <th className="text-right py-2 pr-4 font-medium">p95</th>
            <th className="text-right py-2 pr-4 font-medium">Flag Rate</th>
            <th className="text-left py-2 font-medium">Δ Baseline</th>
          </tr>
        </thead>
        <tbody>
          {evaluators.map((ev) => {
            const s = scorecard[ev];
            const reg = regression?.[ev] ?? null;
            return (
              <tr key={ev} className="border-b border-[#111] hover:bg-[#0d0d0d]">
                <td className="py-2 pr-4 text-zinc-400 capitalize">{ev.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4 text-right text-zinc-600">{s.n}</td>
                <td className="py-2 pr-4 text-right text-zinc-500 font-mono">{num(s.mean)}</td>
                <td className="py-2 pr-4 text-right text-zinc-500 font-mono">{num(s.p50)}</td>
                <td className="py-2 pr-4 text-right text-zinc-500 font-mono">{num(s.p95)}</td>
                <td className="py-2 pr-4 text-right font-mono">
                  <span className={s.flag_rate > 0.1 ? "text-white" : "text-zinc-500"}>
                    {pct(s.flag_rate)}
                  </span>
                  <span className="text-zinc-800 ml-1">({s.flag_count})</span>
                </td>
                <td className="py-2">
                  {reg ? <RegressionBadge entry={reg} /> : <span className="text-zinc-800">—</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
