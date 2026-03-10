-- sentinel/storage/migrations/v002_versions.sql
--
-- Additive migration: prompt/model version registry + offline eval support.
-- Apply after schema.sql. Safe to re-run (IF NOT EXISTS / IF NOT EXISTS guards).

-- ── Prompt version registry ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS prompt_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    name            TEXT NOT NULL,           -- e.g. 'customer-support'
    version_tag     TEXT NOT NULL,           -- e.g. 'v3.2.1'
    template        TEXT NOT NULL,           -- full system prompt text
    description     TEXT,
    metadata        JSONB,
    UNIQUE (name, version_tag)
);

-- ── Model version registry ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider        TEXT NOT NULL,           -- 'openai' | 'anthropic' | 'gemini' | 'ollama'
    model_id        TEXT NOT NULL,           -- e.g. 'gpt-4o'
    version_tag     TEXT NOT NULL,           -- user tag: 'prod-2026-03'
    config_snapshot JSONB NOT NULL,          -- full llm_backend config at eval time
    UNIQUE (provider, model_id, version_tag)
);

-- ── Extend eval_runs: version tracking + offline eval mode ───────────────────

ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS prompt_version_id           UUID REFERENCES prompt_versions(id),
    ADD COLUMN IF NOT EXISTS model_version_id            UUID REFERENCES model_versions(id),
    ADD COLUMN IF NOT EXISTS eval_mode                   TEXT NOT NULL DEFAULT 'live'
                                         CHECK (eval_mode IN ('live', 'offline', 'replay')),
    ADD COLUMN IF NOT EXISTS config_snapshot             JSONB,
    ADD COLUMN IF NOT EXISTS tags                        TEXT[] DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS statistical_regression_json JSONB;
-- statistical_regression_json schema per evaluator:
-- { "toxicity": { mann_whitney_u, p_value, cohens_d, effect_size,
--                 significant, direction, flag_rate_delta, n_current, n_baseline } }

CREATE INDEX IF NOT EXISTS idx_eval_runs_model_version ON eval_runs (model_version_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_tags ON eval_runs USING GIN (tags);

-- ── Extend eval_results: offline eval scores (no live proxy request) ─────────

ALTER TABLE eval_results
    ADD COLUMN IF NOT EXISTS scores_json JSONB;
-- scores_json stores evaluator scores for offline runs where request_id is NULL.

-- Make request_id nullable to support offline eval runs that bypass the proxy.
ALTER TABLE eval_results ALTER COLUMN request_id DROP NOT NULL;
