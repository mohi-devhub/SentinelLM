-- SentinelLM Database Schema
-- Version: 1.0
-- Apply with: psql -U sentinel -d sentinellm -f schema.sql
-- This file is the single source of truth. Never modify the DB directly.

-- ─────────────────────────────────────────────────────────────────────────────
-- EXTENSIONS
-- ─────────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- for gen_random_uuid()


-- ─────────────────────────────────────────────────────────────────────────────
-- REQUESTS
-- Every proxied request is logged here, pass or block.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS requests (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Request metadata
    model           TEXT NOT NULL,
    input_hash      TEXT NOT NULL,          -- SHA-256(raw_input_text)
    input_text      TEXT,                   -- NULL if PII was detected (privacy)
    input_redacted  TEXT,                   -- PII-redacted version (always stored)
    has_context     BOOLEAN NOT NULL DEFAULT FALSE,

    -- Block status
    blocked         BOOLEAN NOT NULL DEFAULT FALSE,
    block_reason    TEXT,                   -- 'prompt_injection_detected' | 'pii_detected' | 'off_topic'

    -- Input evaluator scores (0.0–1.0; NULL = evaluator not run or errored)
    score_pii               FLOAT CHECK (score_pii BETWEEN 0 AND 1),
    score_prompt_injection  FLOAT CHECK (score_prompt_injection BETWEEN 0 AND 1),
    score_topic_guardrail   FLOAT CHECK (score_topic_guardrail BETWEEN 0 AND 1),

    -- Output evaluator scores (NULL if request was blocked before LLM call)
    score_toxicity          FLOAT CHECK (score_toxicity BETWEEN 0 AND 1),
    score_relevance         FLOAT CHECK (score_relevance BETWEEN 0 AND 1),
    score_hallucination     FLOAT CHECK (score_hallucination BETWEEN 0 AND 1),
    score_faithfulness      FLOAT CHECK (score_faithfulness BETWEEN 0 AND 1),

    -- Per-evaluator latency in milliseconds (NULL = not run)
    latency_pii              INT,
    latency_prompt_injection INT,
    latency_topic_guardrail  INT,
    latency_toxicity         INT,
    latency_relevance        INT,
    latency_hallucination    INT,
    latency_faithfulness     INT,
    latency_llm              INT,
    latency_total            INT NOT NULL DEFAULT 0,

    -- Flag columns (true = score exceeded configured threshold)
    flag_pii                BOOLEAN NOT NULL DEFAULT FALSE,
    flag_prompt_injection   BOOLEAN NOT NULL DEFAULT FALSE,
    flag_topic_guardrail    BOOLEAN NOT NULL DEFAULT FALSE,
    flag_toxicity           BOOLEAN NOT NULL DEFAULT FALSE,
    flag_relevance          BOOLEAN NOT NULL DEFAULT FALSE,
    flag_hallucination      BOOLEAN NOT NULL DEFAULT FALSE,
    flag_faithfulness       BOOLEAN NOT NULL DEFAULT FALSE,

    -- Human review fields
    reviewed                BOOLEAN NOT NULL DEFAULT FALSE,
    review_label            TEXT CHECK (review_label IN ('correct_flag', 'false_positive', 'false_negative')),
    reviewed_at             TIMESTAMPTZ,
    reviewer_note           TEXT
);

-- Primary time-ordered lookup (most dashboard queries use this)
CREATE INDEX IF NOT EXISTS idx_requests_created_at
    ON requests (created_at DESC);

-- Fast filter for blocked requests
CREATE INDEX IF NOT EXISTS idx_requests_blocked
    ON requests (created_at DESC)
    WHERE blocked = TRUE;

-- Fast filter for any flagged request (used in review queue + feed)
CREATE INDEX IF NOT EXISTS idx_requests_any_flag
    ON requests (created_at DESC)
    WHERE flag_pii OR flag_prompt_injection OR flag_topic_guardrail
       OR flag_toxicity OR flag_relevance OR flag_hallucination OR flag_faithfulness;

-- Cache invalidation + deduplication lookup
CREATE INDEX IF NOT EXISTS idx_requests_input_hash
    ON requests (input_hash);

-- Review queue: unreviewed flagged requests
CREATE INDEX IF NOT EXISTS idx_requests_review_queue
    ON requests (created_at ASC)
    WHERE reviewed = FALSE
      AND (flag_pii OR flag_prompt_injection OR flag_topic_guardrail
           OR flag_toxicity OR flag_relevance OR flag_hallucination OR flag_faithfulness);


-- ─────────────────────────────────────────────────────────────────────────────
-- EVAL RUNS
-- Metadata for each execution of the eval pipeline CLI.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,

    label           TEXT NOT NULL UNIQUE,   -- human-readable run name: 'run_20260226'
    dataset_path    TEXT NOT NULL,
    baseline_run_id UUID REFERENCES eval_runs(id) ON DELETE SET NULL,

    record_count    INT NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'complete', 'failed')),

    -- Cached scorecard JSON: { evaluator_name: { mean, p50, p95, flag_rate } }
    -- Populated on completion; recomputed if NULL
    summary_json    JSONB,

    -- Regression summary vs baseline: { evaluator_name: { delta_flag_rate, regressed } }
    regression_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created_at
    ON eval_runs (created_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- EVAL RESULTS
-- Per-record outcomes for each eval run.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS eval_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id     UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    request_id      UUID NOT NULL REFERENCES requests(id) ON DELETE CASCADE,

    record_index    INT NOT NULL,           -- position in the source JSONL file
    input_text      TEXT NOT NULL,
    expected_output TEXT,                   -- from golden dataset (nullable for open-ended)
    actual_output   TEXT,                   -- LLM response text

    passed          BOOLEAN NOT NULL        -- TRUE if no evaluator tripped threshold
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run_id
    ON eval_results (eval_run_id);

CREATE INDEX IF NOT EXISTS idx_eval_results_failed
    ON eval_results (eval_run_id)
    WHERE passed = FALSE;


-- ─────────────────────────────────────────────────────────────────────────────
-- HELPER VIEW: flagged_requests
-- Convenience view used by the dashboard review queue and feed.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW flagged_requests AS
SELECT
    id,
    created_at,
    model,
    input_redacted,
    blocked,
    block_reason,
    score_pii,
    score_prompt_injection,
    score_topic_guardrail,
    score_toxicity,
    score_relevance,
    score_hallucination,
    score_faithfulness,
    flag_pii,
    flag_prompt_injection,
    flag_topic_guardrail,
    flag_toxicity,
    flag_relevance,
    flag_hallucination,
    flag_faithfulness,
    latency_total,
    reviewed,
    review_label
FROM requests
WHERE
    flag_pii OR flag_prompt_injection OR flag_topic_guardrail
    OR flag_toxicity OR flag_relevance OR flag_hallucination OR flag_faithfulness
    OR blocked = TRUE
ORDER BY created_at DESC;
