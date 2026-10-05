-- Domain model. Names follow the blueprint's schema; additions are marked (+).
-- Every table is tenant-scoped: each firm works on its own copy of the (synthetic) public record, and its uploaded
-- briefs never leave its tenant.

CREATE TABLE judges (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, court text NOT NULL, appointed_at date, metadata jsonb NOT NULL DEFAULT '{}'::jsonb, ordinal int NOT NULL);
SELECT enable_tenant_rls('judges');

CREATE TABLE cases (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, court text NOT NULL, docket_number text NOT NULL, filed_at date NOT NULL, closed_at date, nature_of_suit text NOT NULL,
  judge_id uuid NOT NULL REFERENCES judges, status text NOT NULL,
  facts text NOT NULL, features jsonb NOT NULL, ordinal int NOT NULL,                                                         -- (+)
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', facts)) STORED, UNIQUE (tenant_id, docket_number));
CREATE INDEX cases_judge ON cases (judge_id);
CREATE INDEX cases_filed ON cases (tenant_id, filed_at DESC, id);
CREATE INDEX cases_tsv ON cases USING gin (tsv);
SELECT enable_tenant_rls('cases');

CREATE TABLE filings (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, case_id uuid NOT NULL REFERENCES cases ON DELETE CASCADE, filing_type text NOT NULL, filed_at timestamptz NOT NULL, document_uri text, text_hash text);
CREATE INDEX filings_case ON filings (case_id);
SELECT enable_tenant_rls('filings');

CREATE TABLE citations (source_filing uuid REFERENCES filings ON DELETE CASCADE, cited_case uuid REFERENCES cases ON DELETE CASCADE, context text NOT NULL, treatment text NOT NULL, PRIMARY KEY (source_filing, cited_case),
  confidence real NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants);                                                       -- (+)
CREATE INDEX citations_cited ON citations (cited_case);
SELECT enable_tenant_rls('citations');

CREATE TABLE motions (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, case_id uuid NOT NULL REFERENCES cases ON DELETE CASCADE, motion_type text NOT NULL, filed_at date NOT NULL, outcome text NOT NULL CHECK (outcome IN ('granted','denied')),
  decided_at date NOT NULL, order_filing uuid REFERENCES filings);                                                             -- (+)
CREATE INDEX motions_case ON motions (case_id);
SELECT enable_tenant_rls('motions');

CREATE TABLE indexes (tenant_id uuid PRIMARY KEY REFERENCES tenants, artifact bytea NOT NULL, metrics jsonb NOT NULL, version text NOT NULL, fitted_at timestamptz NOT NULL DEFAULT now());   -- (+) this tenant's similarity index and outcome model
SELECT enable_tenant_rls('indexes');

CREATE TABLE analyses (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), request jsonb NOT NULL, result jsonb NOT NULL);   -- (+) uploaded briefs and their memos: confidential to the firm
SELECT enable_tenant_rls('analyses');
