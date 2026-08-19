-- sentinel/storage/migrations/v004_exfiltration.sql
--
-- Additive migration: adds columns for the new `exfiltration` output
-- evaluator (detects/strips markdown image tags with external URLs — see
-- sentinel/evaluators/output/exfiltration.py). Apply after v003_tenants.sql.
-- Safe to re-run (IF NOT EXISTS / CREATE OR REPLACE guards).

ALTER TABLE requests ADD COLUMN IF NOT EXISTS
    score_exfiltration FLOAT CHECK (score_exfiltration BETWEEN 0 AND 1);
ALTER TABLE requests ADD COLUMN IF NOT EXISTS
    latency_exfiltration INT;
ALTER TABLE requests ADD COLUMN IF NOT EXISTS
    flag_exfiltration BOOLEAN NOT NULL DEFAULT FALSE;

-- Recreate the flag-dependent indexes and view to include the new column —
-- schema.sql's CREATE INDEX/VIEW IF NOT EXISTS won't touch these on an
-- already-migrated database, since the objects already exist.

DROP INDEX IF EXISTS idx_requests_any_flag;
CREATE INDEX idx_requests_any_flag
    ON requests (created_at DESC)
    WHERE flag_pii OR flag_prompt_injection OR flag_topic_guardrail
       OR flag_toxicity OR flag_exfiltration OR flag_relevance
       OR flag_hallucination OR flag_faithfulness;

DROP INDEX IF EXISTS idx_requests_review_queue;
CREATE INDEX idx_requests_review_queue
    ON requests (created_at ASC)
    WHERE reviewed = FALSE
      AND (flag_pii OR flag_prompt_injection OR flag_topic_guardrail
           OR flag_toxicity OR flag_exfiltration OR flag_relevance
           OR flag_hallucination OR flag_faithfulness);

-- CREATE OR REPLACE VIEW can only append columns at the end of the existing
-- list, not insert one in the middle (Postgres rejects that as a column
-- rename) — drop and recreate instead, matching the indexes above.
DROP VIEW IF EXISTS flagged_requests;
CREATE VIEW flagged_requests AS
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
    score_exfiltration,
    score_relevance,
    score_hallucination,
    score_faithfulness,
    flag_pii,
    flag_prompt_injection,
    flag_topic_guardrail,
    flag_toxicity,
    flag_exfiltration,
    flag_relevance,
    flag_hallucination,
    flag_faithfulness,
    latency_total,
    reviewed,
    review_label
FROM requests
WHERE
    flag_pii OR flag_prompt_injection OR flag_topic_guardrail
    OR flag_toxicity OR flag_exfiltration OR flag_relevance
    OR flag_hallucination OR flag_faithfulness
    OR blocked = TRUE
ORDER BY created_at DESC;
