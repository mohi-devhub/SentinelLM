import type {
  AggregateMetricsResponse,
  EvalRunDetail,
  EvalRunSummary,
  RequestDetail,
  ReviewLabel,
  ScoreHistoryResponse,
  SummaryMetrics,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(BASE + path);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    });
  }
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`PATCH ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ── Scores ────────────────────────────────────────────────────────────────────

export interface ScoresParams {
  page?: number;
  limit?: number;
  flagged_only?: boolean;
  evaluator?: string;
}

export function fetchScores(params: ScoresParams = {}): Promise<ScoreHistoryResponse> {
  return get("/v1/sentinel/scores", {
    page: String(params.page ?? 1),
    limit: String(params.limit ?? 50),
    flagged_only: params.flagged_only ? "true" : "false",
    evaluator: params.evaluator ?? "",
  });
}

export function fetchScore(id: string): Promise<RequestDetail> {
  return get(`/v1/sentinel/scores/${id}`);
}

// ── Metrics ───────────────────────────────────────────────────────────────────

export function fetchSummaryMetrics(): Promise<SummaryMetrics> {
  return get("/v1/sentinel/metrics/summary");
}

export function fetchAggregateMetrics(
  window = "24h",
  bucket_size = "1h"
): Promise<AggregateMetricsResponse> {
  return get("/v1/sentinel/metrics/aggregate", { window, bucket_size });
}

// ── Review ────────────────────────────────────────────────────────────────────

export function fetchReviewQueue(limit = 20): Promise<RequestDetail[]> {
  return get("/v1/sentinel/review", { limit: String(limit) });
}

export function submitReview(
  requestId: string,
  label: ReviewLabel,
  note?: string
): Promise<{ status: string }> {
  return patch(`/v1/sentinel/review/${requestId}`, { label, note });
}

// ── Eval ──────────────────────────────────────────────────────────────────────

export function fetchEvalRuns(): Promise<EvalRunSummary[]> {
  return get("/v1/sentinel/eval");
}

export function fetchEvalRun(runId: string): Promise<EvalRunDetail> {
  return get(`/v1/sentinel/eval/${runId}`);
}
