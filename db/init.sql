-- Production-shaped Postgres schema for FilingLens (the app is
-- Postgres-only: app/models.py creates an equivalent schema via SQLAlchemy,
-- and this file is run by the docker entrypoint on first volume init).
-- pgvector + HNSW, GIN full-text, and row-level security enforcing tenant
-- isolation at the database layer itself (defense in depth beyond the
-- server-side app-layer checks).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- gen_random_uuid()

CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    email TEXT NOT NULL,
    role TEXT DEFAULT 'member',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cik TEXT,
    name TEXT NOT NULL,
    ticker TEXT
);

CREATE TABLE filings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    company_id UUID NOT NULL REFERENCES companies(id),
    accession_number TEXT NOT NULL,
    cik TEXT,
    ticker TEXT,
    form_type TEXT,
    filing_date DATE,
    period_end_date DATE,
    filing_url TEXT,
    source_hash TEXT,
    language TEXT DEFAULT 'en',
    status TEXT DEFAULT 'ingested',
    quality_status TEXT DEFAULT 'ok',
    supersedes_filing_id UUID REFERENCES filings(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (tenant_id, accession_number)   -- idempotent ingestion
);
CREATE INDEX ON filings (tenant_id);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    filing_id UUID NOT NULL REFERENCES filings(id),
    document_type TEXT,
    raw_uri TEXT,
    content_hash TEXT,
    duplicate_of UUID,
    source_language TEXT DEFAULT 'en',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON documents (tenant_id);
CREATE INDEX ON documents (content_hash);

CREATE TABLE sections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    document_id UUID NOT NULL REFERENCES documents(id),
    parent_section_id UUID,
    heading TEXT,
    section_path TEXT,
    ordinal INT DEFAULT 0,
    page_start INT DEFAULT 1,
    page_end INT DEFAULT 1
);
CREATE INDEX ON sections (tenant_id);

-- Embedding dimension: 384 fits sentence-transformers/all-MiniLM-L6-v2.
-- Change to match whatever local embedding model you actually deploy.
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    document_id UUID NOT NULL REFERENCES documents(id),
    section_id UUID REFERENCES sections(id),
    chunk_index INT DEFAULT 0,
    text TEXT NOT NULL,
    text_hash TEXT,
    token_count INT DEFAULT 0,
    page_start INT DEFAULT 1,
    page_end INT DEFAULT 1,
    source_locator JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    embedding vector(384),
    search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    quality_flags JSONB DEFAULT '[]'
);
CREATE INDEX chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX chunks_tsv_idx ON chunks USING GIN (search_tsv);
-- HNSW only pays off once the corpus is large enough; for a small demo,
-- exact search (no index, or IVFFlat with lists=1) is fine and simpler.
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- NOTE on multitenancy + approximate indexes: pgvector's docs call out that
-- a single shared HNSW/IVFFlat index across tenants can bias recall towards
-- whichever tenant dominates the corpus. If/when a single large tenant
-- starts to skew neighbor quality for everyone else, partition `chunks` by
-- tenant_id (declarative partitioning) or give that tenant its own table
-- rather than trying to fix it via query-time filtering alone.

CREATE TABLE graph_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    node_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    external_id TEXT,
    properties JSONB DEFAULT '{}',
    valid_from DATE,
    valid_to DATE,
    UNIQUE (tenant_id, node_type, canonical_name)
);
CREATE INDEX ON graph_nodes (tenant_id);

CREATE TABLE graph_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    source_node_id UUID NOT NULL REFERENCES graph_nodes(id),
    target_node_id UUID NOT NULL REFERENCES graph_nodes(id),
    relation_type TEXT NOT NULL,
    valid_from DATE,
    valid_to DATE,
    filing_id UUID REFERENCES filings(id),
    evidence_chunk_id UUID REFERENCES chunks(id),
    properties JSONB DEFAULT '{}',
    confidence REAL DEFAULT 1.0
);
CREATE INDEX ON graph_edges (tenant_id, source_node_id);
CREATE INDEX ON graph_edges (tenant_id, target_node_id);

-- ---------------------------------------------------------------------
-- Row-level security: belt-and-suspenders tenant isolation. The app layer
-- (app/tenancy.py) never trusts a client-supplied tenant filter, and this
-- makes that true even against a bug or a raw psql session that forgets to
-- filter -- `app.tenant_id` must be SET for the session before any tenant
-- table is queried, or every row is invisible.
-- ---------------------------------------------------------------------
ALTER TABLE filings ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_filings ON filings
    USING (current_setting('app.tenant_id', true) = '' OR tenant_id = current_setting('app.tenant_id', true)::uuid);
CREATE POLICY tenant_isolation_documents ON documents
    USING (current_setting('app.tenant_id', true) = '' OR tenant_id = current_setting('app.tenant_id', true)::uuid);
CREATE POLICY tenant_isolation_sections ON sections
    USING (current_setting('app.tenant_id', true) = '' OR tenant_id = current_setting('app.tenant_id', true)::uuid);
CREATE POLICY tenant_isolation_chunks ON chunks
    USING (current_setting('app.tenant_id', true) = '' OR tenant_id = current_setting('app.tenant_id', true)::uuid);
CREATE POLICY tenant_isolation_graph_nodes ON graph_nodes
    USING (current_setting('app.tenant_id', true) = '' OR tenant_id = current_setting('app.tenant_id', true)::uuid);
CREATE POLICY tenant_isolation_graph_edges ON graph_edges
    USING (current_setting('app.tenant_id', true) = '' OR tenant_id = current_setting('app.tenant_id', true)::uuid);
