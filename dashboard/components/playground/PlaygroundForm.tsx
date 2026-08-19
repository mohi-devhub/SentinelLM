export interface Preset {
  label: string;
  message: string;
  contextDocs?: string;
  model?: string;
}

export function PlaygroundForm({
  message,
  setMessage,
  contextDocs,
  setContextDocs,
  model,
  setModel,
  presets,
  onPreset,
  onSubmit,
  isPending,
}: {
  message: string;
  setMessage: (v: string) => void;
  contextDocs: string;
  setContextDocs: (v: string) => void;
  model: string;
  setModel: (v: string) => void;
  presets: Preset[];
  onPreset: (preset: Preset) => void;
  onSubmit: () => void;
  isPending: boolean;
}) {
  const canSubmit = message.trim().length > 0 && model.trim().length > 0 && !isPending;

  return (
    <div className="border border-[#1a1a1a] rounded-lg p-5 bg-[#0d0d0d] space-y-3">
      <div>
        <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Examples</p>
        <div className="flex flex-wrap gap-1.5">
          {presets.map((p) => (
            <button
              key={p.label}
              onClick={() => onPreset(p)}
              className="text-[10px] py-1.5 px-2.5 rounded border border-[#2a2a2a] text-zinc-600 hover:border-zinc-600 hover:text-zinc-400 transition-colors"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Message</p>
        <textarea
          rows={4}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message to send through the proxy…"
          className="w-full text-xs bg-black border border-[#1a1a1a] rounded px-3 py-2 text-zinc-400 placeholder-zinc-800 focus:outline-none focus:border-zinc-700 font-mono resize-none"
        />
      </div>

      <div>
        <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">
          Context documents
        </p>
        <textarea
          rows={3}
          value={contextDocs}
          onChange={(e) => setContextDocs(e.target.value)}
          placeholder="One document per line (optional — enables hallucination/faithfulness scoring)"
          className="w-full text-xs bg-black border border-[#1a1a1a] rounded px-3 py-2 text-zinc-400 placeholder-zinc-800 focus:outline-none focus:border-zinc-700 font-mono resize-none"
        />
      </div>

      <div>
        <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-1.5">Model</p>
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="e.g. claude-haiku-4-5-20251001"
          className="w-full text-xs bg-black border border-[#1a1a1a] rounded px-3 py-2 text-zinc-400 placeholder-zinc-800 focus:outline-none focus:border-zinc-700 font-mono"
        />
        <p className="text-[10px] text-zinc-700 mt-1">
          Must match the model your backend's configured provider serves.
        </p>
      </div>

      <button
        disabled={!canSubmit}
        onClick={onSubmit}
        className="w-full text-xs py-2 rounded border border-zinc-700 text-zinc-400 hover:bg-white hover:text-black hover:border-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      >
        {isPending ? "scoring…" : "Send"}
      </button>
    </div>
  );
}
