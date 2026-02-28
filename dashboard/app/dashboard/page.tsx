import { FlagRateChart } from "@/components/dashboard/FlagRateChart";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { SummaryCards } from "@/components/dashboard/SummaryCards";

export default function DashboardPage() {
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-sm font-semibold text-white uppercase tracking-widest">Overview</h1>
        <p className="text-xs text-zinc-600 mt-1">Last 24 hours · auto-refreshes every 30s</p>
      </div>

      <SummaryCards />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <MetricsChart />
        <FlagRateChart />
      </div>
    </div>
  );
}
