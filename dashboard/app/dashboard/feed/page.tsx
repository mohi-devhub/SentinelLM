import { LiveFeed } from "@/components/feed/LiveFeed";

export default function FeedPage() {
  return (
    <div className="p-6">
      <div className="mb-4">
        <h1 className="text-sm font-semibold text-white uppercase tracking-widest">Live Feed</h1>
        <p className="text-xs text-zinc-600 mt-1">Real-time stream of proxied requests</p>
      </div>
      <LiveFeed />
    </div>
  );
}
