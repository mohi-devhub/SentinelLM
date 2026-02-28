import { ReviewQueue } from "@/components/review/ReviewQueue";

export default function ReviewPage() {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-sm font-semibold text-white uppercase tracking-widest">Human Review</h1>
        <p className="text-xs text-zinc-600 mt-1">
          correct_flag · false_positive · false_negative
        </p>
      </div>
      <div className="max-w-2xl">
        <ReviewQueue />
      </div>
    </div>
  );
}
