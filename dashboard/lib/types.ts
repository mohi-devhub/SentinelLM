// TypeScript types derived from the OpenAI spec (docs/openapi.yaml)

export interface Scores {
  pii: number | null;
  prompt_injection: number | null;
  topic_guardrail: number | null;
  toxicity: number | null;
  relevance: number | null;
  hallucination: number | null;
  faithfulness: number | null;
}

export interface Latencies {
  pii: number | null;
  prompt_injection: number | null;
  topic_guardrail: number | null;
  toxicity: number | null;
  relevance: number | null;
  hallucination: number | null;
  faithfulness: number | null;
  llm: number | null;
  total: number;
}

export interface RequestSummary {
  id: string;
  created_at: string;
  model: string;
  blocked: boolean;
  block_reason: string | null;
  flags: string[];
  scores: Scores;
  latency_total: number;
}

export interface RequestDetail extends RequestSummary {
  input_text: string | null;
  input_redacted: string | null;
  has_context: boolean;
  latency_ms: Latencies;
  reviewed: boolean;
  review_label: string | null;
  reviewer_note: string | null;
}

export interface ScoreHistoryResponse {
  total: number;
  page: number;
  limit: number;
  items: RequestDetail[];
}

export interface MetricBucket {
  bucket_start: string;
  request_count: number;
  blocked_count: number;
  avg_toxicity: number | null;
  avg_relevance: number | null;
  avg_hallucination: number | null;
  flag_rate_pii: number;
  flag_rate_prompt_injection: number;
  flag_rate_topic_guardrail: number;
  flag_rate_toxicity: number;
  flag_rate_relevance: number;
  flag_rate_hallucination: number;
  flag_rate_faithfulness: number;
  p95_latency_total: number;
}

export interface AggregateMetricsResponse {
  window: string;
  bucket_size: string;
  buckets: MetricBucket[];
}

export interface SummaryMetrics {
  total_requests_24h: number;
  blocked_requests_24h: number;
  block_rate_24h: number;
  avg_latency_24h_ms: number;
  top_flag_reason: string | null;
  unreviewed_flags: number;
}

export interface EvalRunSummary {
  id: string;
  label: string;
  created_at: string | null;
  completed_at: string | null;
  status: "running" | "complete" | "failed";
  record_count: number;
  dataset_path: string;
}

export interface EvaluatorStats {
  mean: number | null;
  p50: number | null;
  p95: number | null;
  flag_rate: number;
  flag_count: number;
  n: number;
  n_scored: number;
}

export interface RegressionEntry {
  baseline_flag_rate: number;
  current_flag_rate: number;
  delta: number;
  regressed: boolean;
}

export interface EvalRunDetail extends EvalRunSummary {
  scorecard: Record<string, EvaluatorStats> | null;
  regression: Record<string, RegressionEntry> | null;
}

export type ReviewLabel = "correct_flag" | "false_positive" | "false_negative";

export interface SentinelEvent {
  event_type: "request_passed" | "request_blocked";
  request_id: string;
  model: string;
  blocked: boolean;
  block_reason: string | null;
  flags: string[];
  scores: Scores;
  latency_total: number;
}
