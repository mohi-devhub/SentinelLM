-- sentinel/storage/migrations/v003_tenants.sql
--
-- Additive migration: multi-tenancy — tenants + per-tenant hashed API keys.
-- Apply after v002_versions.sql. Safe to re-run (IF NOT EXISTS / ON CONFLICT guards).

-- ── Tenants ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tenants (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL UNIQUE,
    is_default      BOOLEAN NOT NULL DEFAULT FALSE
);

-- Guarantees exactly one default tenant — without this, nothing stops a
-- second is_default=TRUE row, which would make get_default_tenant()'s
-- "LIMIT 1" pick order-dependent instead of deterministic.
CREATE UNIQUE INDEX IF NOT EXISTS idx_tenants_single_default
    ON tenants (is_default) WHERE is_default;

-- ── API keys ───────────────────────────────────────────────────────────────
-- Keys are stored hashed (SHA-256 of the presented secret) — the plaintext
-- key is shown once at creation time (`sentinel tenant create-key`) and never
-- persisted. key_prefix is the first 12 chars of the plaintext, kept only for
-- display/lookup convenience in `sentinel tenant list-keys`.

CREATE TABLE IF NOT EXISTS api_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash        TEXT NOT NULL UNIQUE,
    key_prefix      TEXT NOT NULL,
    label           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at    TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys (tenant_id);

-- One default tenant so a legacy single-key deployment (SENTINEL_API_KEY env
-- var, no tenant CLI ever run) has somewhere to attribute its data. The app
-- lifespan upserts a hashed api_keys row for this tenant on every startup
-- when SENTINEL_API_KEY is set — see sentinel/tenancy/bootstrap.py.
INSERT INTO tenants (name, slug, is_default)
VALUES ('Default', 'default', true)
ON CONFLICT (slug) DO NOTHING;

-- ── Tenant scoping on existing tables ─────────────────────────────────────
-- Added nullable first so the ALTER is instant on a large existing table,
-- then backfilled to the default tenant and locked to NOT NULL below — every
-- row (historical and new) ends up with an owner, so query code can always
-- filter with a plain `WHERE tenant_id = $1` and never has to special-case
-- NULL as "unscoped".

ALTER TABLE requests
    ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);

ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);

UPDATE requests SET tenant_id = (SELECT id FROM tenants WHERE is_default = TRUE)
    WHERE tenant_id IS NULL;

UPDATE eval_runs SET tenant_id = (SELECT id FROM tenants WHERE is_default = TRUE)
    WHERE tenant_id IS NULL;

-- A bare `ALTER COLUMN ... SET NOT NULL` forces a full-table scan under an
-- ACCESS EXCLUSIVE lock — on `requests` (this app's highest-write table)
-- that's a real outage, not just a slow migration, on any table with
-- production-sized data. Instead: add the constraint NOT VALID (instant,
-- metadata-only), VALIDATE it separately (does the same scan, but only
-- takes a SHARE UPDATE EXCLUSIVE lock, which still allows concurrent reads
-- and writes), then SET NOT NULL — Postgres skips its own scan because it
-- recognizes the already-validated CHECK constraint covers it (PG 12+) —
-- and finally drop the now-redundant CHECK. Each block is a no-op on
-- replay once the column is already NOT NULL, so this doesn't re-scan on
-- every app startup (migrate.py applies every migration file unconditionally).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'requests'::regclass AND attname = 'tenant_id' AND attnotnull
    ) THEN
        ALTER TABLE requests ADD CONSTRAINT requests_tenant_id_not_null
            CHECK (tenant_id IS NOT NULL) NOT VALID;
        ALTER TABLE requests VALIDATE CONSTRAINT requests_tenant_id_not_null;
        ALTER TABLE requests ALTER COLUMN tenant_id SET NOT NULL;
        ALTER TABLE requests DROP CONSTRAINT requests_tenant_id_not_null;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'eval_runs'::regclass AND attname = 'tenant_id' AND attnotnull
    ) THEN
        ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_tenant_id_not_null
            CHECK (tenant_id IS NOT NULL) NOT VALID;
        ALTER TABLE eval_runs VALIDATE CONSTRAINT eval_runs_tenant_id_not_null;
        ALTER TABLE eval_runs ALTER COLUMN tenant_id SET NOT NULL;
        ALTER TABLE eval_runs DROP CONSTRAINT eval_runs_tenant_id_not_null;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_requests_tenant_created
    ON requests (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_eval_runs_tenant
    ON eval_runs (tenant_id);

-- eval_runs.label was globally UNIQUE (schema.sql) — relax to unique-per-tenant
-- so two tenants can each use a label like 'v1.0-baseline' independently.
-- PostgreSQL has no `ADD CONSTRAINT IF NOT EXISTS`, so guard via pg_constraint.
ALTER TABLE eval_runs DROP CONSTRAINT IF EXISTS eval_runs_label_key;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'eval_runs_tenant_label_key'
    ) THEN
        ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_tenant_label_key UNIQUE (tenant_id, label);
    END IF;
END $$;
