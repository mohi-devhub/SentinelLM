import { Playground } from "@/components/playground/Playground";

export default function PlaygroundPage() {
  return (
    <div className="p-6">
      <div className="mb-4">
        <h1 className="text-sm font-semibold text-white uppercase tracking-widest">Playground</h1>
        <p className="text-xs text-zinc-600 mt-1">
          Send a live request through the SentinelLM proxy and watch it get scored
        </p>
      </div>
      <Playground />
    </div>
  );
}
