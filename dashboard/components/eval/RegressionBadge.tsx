import type { RegressionEntry } from "@/lib/types";

export function RegressionBadge({ entry }: { entry: RegressionEntry }) {
  if (entry.regressed) {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border border-zinc-600 text-zinc-300">
        ↑ {(entry.delta * 100).toFixed(1)}pp
      </span>
    );
  }
  if (entry.delta < -0.01) {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border border-[#2a2a2a] text-zinc-600">
        ↓ {(Math.abs(entry.delta) * 100).toFixed(1)}pp
      </span>
    );
  }
  return (
    <span className="text-[10px] text-zinc-700">stable</span>
  );
}
