"use client";

import { useEffect, useState } from "react";
import { KeyRound } from "lucide-react";
import { getApiKey, getSkippedKey, setApiKey, setSkippedKey } from "@/lib/auth";

/**
 * Gates the dashboard behind a locally-stored API key.
 *
 * Not a login system — there's no session or server-side check here, just a
 * convenience so the browser sends X-API-Key on every request once it knows
 * one. A deployment with no tenant/API-key auth configured works exactly the
 * same either way — "Continue without a key" skips this entirely.
 */
export function ApiKeyGate({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [hasKey, setHasKey] = useState(false);
  const [input, setInput] = useState("");

  useEffect(() => {
    setHasKey(getApiKey() !== null || getSkippedKey());
    setReady(true);
  }, []);

  if (!ready) return null;
  if (hasKey) return <>{children}</>;

  const submit = (skip: boolean) => {
    if (skip) {
      setSkippedKey();
    } else if (input.trim()) {
      setApiKey(input.trim());
    }
    setHasKey(true);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black">
      <div className="w-full max-w-sm px-6">
        <div className="flex items-center gap-2.5 mb-6">
          <KeyRound size={16} className="text-white" />
          <span className="text-sm font-semibold text-white tracking-tight">
            SentinelLM API Key
          </span>
        </div>
        <p className="text-xs text-zinc-500 mb-4">
          Stored only in this browser and sent as{" "}
          <code className="text-zinc-400">X-API-Key</code> on every request. Leave blank if this
          deployment doesn&apos;t require auth.
        </p>
        <input
          type="password"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit(false)}
          placeholder="sk-sentinel-..."
          className="w-full bg-[#0a0a0a] border border-[#1a1a1a] rounded px-3 py-2 text-xs text-white placeholder:text-zinc-700 focus:outline-none focus:border-zinc-600 mb-3"
          autoFocus
        />
        <div className="flex gap-2">
          <button
            onClick={() => submit(false)}
            disabled={!input.trim()}
            className="flex-1 bg-white text-black text-xs font-medium rounded px-3 py-2 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-zinc-200 transition-colors"
          >
            Save key
          </button>
          <button
            onClick={() => submit(true)}
            className="flex-1 text-zinc-500 text-xs rounded px-3 py-2 hover:text-white hover:bg-[#111] transition-colors"
          >
            Continue without a key
          </button>
        </div>
      </div>
    </div>
  );
}
