"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart2,
  Eye,
  FlaskConical,
  KeyRound,
  LayoutDashboard,
  Radio,
  Shield,
  Terminal,
} from "lucide-react";
import { clearApiKey } from "@/lib/auth";

const NAV = [
  { href: "/dashboard",            label: "Overview",   icon: LayoutDashboard },
  { href: "/dashboard/playground", label: "Playground", icon: Terminal },
  { href: "/dashboard/feed",       label: "Live Feed",  icon: Radio },
  { href: "/dashboard/review",     label: "Review",     icon: Eye },
  { href: "/dashboard/eval",       label: "Eval Runs",  icon: FlaskConical },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-52 shrink-0 flex flex-col border-r border-[#1a1a1a] min-h-screen bg-black">
      {/* Logo */}
      <div className="px-5 py-5 flex items-center gap-2.5 border-b border-[#1a1a1a]">
        <Shield size={16} className="text-white" />
        <span className="text-sm font-semibold text-white tracking-tight">SentinelLM</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 px-2 space-y-0.5">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2 rounded text-xs transition-colors ${
                active
                  ? "bg-white text-black font-medium"
                  : "text-zinc-500 hover:text-white hover:bg-[#111]"
              }`}
            >
              <Icon size={14} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-[#1a1a1a] flex items-center justify-between text-xs text-zinc-700">
        <span>v1.0.0</span>
        <button
          onClick={() => {
            clearApiKey();
            window.location.reload();
          }}
          title="Change API key"
          className="text-zinc-700 hover:text-white transition-colors"
        >
          <KeyRound size={13} />
        </button>
      </div>
    </aside>
  );
}
