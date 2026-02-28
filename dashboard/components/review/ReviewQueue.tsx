"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchReviewQueue } from "@/lib/api";
import type { RequestDetail } from "@/lib/types";
import { ReviewCard } from "./ReviewCard";

export function ReviewQueue() {
  const { data, isLoading, isError } = useQuery<RequestDetail[]>({
    queryKey: ["review-queue"],
    queryFn: () => fetchReviewQueue(),
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return <div className="py-20 text-center text-xs text-zinc-700">loading review queue…</div>;
  }

  if (isError) {
    return <div className="py-20 text-center text-xs text-zinc-700">failed to load review queue.</div>;
  }

  if (!data || data.length === 0) {
    return (
      <div className="py-20 text-center text-xs text-zinc-700">
        no items pending review
      </div>
    );
  }

  const unreviewed = data.filter((r) => !r.reviewed);
  const reviewed   = data.filter((r) => r.reviewed);

  return (
    <div className="space-y-6">
      {unreviewed.length > 0 && (
        <section>
          <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-3">
            Pending ({unreviewed.length})
          </p>
          <div className="space-y-3">
            {unreviewed.map((r) => <ReviewCard key={r.id} request={r} />)}
          </div>
        </section>
      )}

      {reviewed.length > 0 && (
        <section>
          <p className="text-[10px] text-zinc-700 uppercase tracking-widest mb-3">
            Reviewed ({reviewed.length})
          </p>
          <div className="space-y-3">
            {reviewed.map((r) => <ReviewCard key={r.id} request={r} />)}
          </div>
        </section>
      )}
    </div>
  );
}
