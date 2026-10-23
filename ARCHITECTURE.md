# FilingLens — Complete Repo Knowledge Base

This document explains every file in the repo (excluding `tests/`), what it does, how it works, and how the pieces connect. It is written for interview preparation — understand the *why* as well as the *what*.

---

# 0. HIGH-LEVEL ARCHITECTURE

FilingLens is a **multi-tenant hybrid search + GraphRAG engine over SEC filings**. It answers questions that pure vector search cannot (e.g. "trace how Apple's voting share structure changed across its last three filings relative to its peer group").

## Two modes

| Mode | Entry point | What happens |
|------|-------------|--------------|
| **Ingest (write path)** | `scripts/ingest_sec.py` | Downloads SEC filings → parses HTML → token-budgeted chunks → MiniLM embeddings → evidence-backed graph edges → Postgres + pgvector |
| **Serve (read path)** | `app/main:app` (FastAPI) | Question → plan → lexical + vector retrieval → RRF fusion → cross-encoder rerank → graph traversal → deterministic answer (+ optional Gemini prose) |

## The core principle: evidence-backed, never hallucinated

> "The graph MUST be evidence-backed. Do not create a relationship only because an LLM inferred it."

Every graph edge carries `evidence_chunk_id` pointing to an exact span of filing text. Numbers are never corrected by LLMs (`l0%` stays `l0%` with a quality flag). The system works with the LLM turned off — retrieval, graph traversal, and citation-backed answers all function deterministically.

## Data model (tables in `app/models.py`)

```
tenants          → tenants (slug: 'sec', 'emerging', ...)
  └── filings      → SEC filings (accession_number, form_type, company)
    └── documents  → raw filing document
      └── sections → Item/Part sections with page locators
        └── chunks → token-budgeted text chunks + 384-d embedding vector
  
  └── graph_nodes  → Company, Filing, SecurityClass, Person, Executive, BoardMember, GovernanceEvent
    └── graph_edges→ evidence-backed relations (DESCRIBES, FILED, HOLDS_ROLE, PEER_OF, etc.)
```

Tenant isolation is enforced twice: app-side filter on `tenant_id` + Postgres Row Level Security (RLS).

---

# 1. ROOT CONFIGURATION & PACKAGING

## `requirements.txt`
Python dependencies pinned with minimums:

| Package | Role |
|---------|------|
| `fastapi>=0.110` | Web framework; defines all HTTP routes, request validation, lifespan events |
| `uvicorn[standard]>=0.29` | ASGI server that runs the FastAPI app |
| `sqlalchemy>=2.0` | ORM + connection pool; defines models, queries, transactions |
| `pydantic>=2.6` | Request/response schema validation (FastAPI uses this under the hood) |
| `numpy>=1.26` | Vector math for cross-encoder duplicate suppression (`select_final`) |
| `requests>=2.31` | HTTP client for SEC EDGAR downloads and Gemini API calls |
| `pytest>=8.0` | Test runner |
| `psycopg2-binary>=2.9` | Postgres driver |
| `pgvector>=0.2` | pgvector SQLAlchemy type + HNSW index support |
| `sentence-transformers>=2.6` | Local embeddings (all-MiniLM-L6-v2) and cross-encoder reranker |

All free / open source. No paid APIs required.

## `Dockerfile`
Multi-stage-free single-stage build:
- Base: `python:3.12-slim`
- Installs `requirements.txt` only (psycopg2-binary and pgvector are transitively installed via requirements now)
- Copies source into `/srv`
- Exposes port 8000
- Command: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}` — honors `$PORT` (Render/HF/etc.) but defaults to 8000 locally

## `docker-compose.yml`
Defines two services:

### `db` (Postgres 17 + pgvector)
- Image: `pgvector/pgvector:pg17`
- Environment: `POSTGRES_DB=graphrag`, `POSTGRES_USER=graphrag`, `POSTGRES_PASSWORD=graphrag_dev_only`
- Volumes: `pgdata` (persists across restarts), `./db/init.sql` mounted as read-only entrypoint script (creates extensions/tables on first boot)
- Port: `5432:5432`
- Healthcheck: `pg_isready -U graphrag`

### `api` (FastAPI)
- Build: local context, Dockerfile
- Depends on: `db` (condition: `service_healthy`)
- Environment from `.env` via Compose interpolation: `DATABASE_URL`, `EMBED_DIM`, `EMBEDDER`, `SEC_USER_AGENT`, `GEMINI_API_KEY`, `GEMINI_MODEL`
- Port: `8000:8000`
- Volumes: `hf-cache` (persist MiniLM weights across restarts), bind mounts `./app` and `./frontend` so code edits are live without rebuilding
- Restart: `unless-stopped`

Named volumes: `pgdata`, `hf-cache`

## `.env` (gitignored)
Your private runtime values. Never committed:
```
DATABASE_URL=postgresql+psycopg2://graphrag:graphrag_dev_only@localhost:5432/graphrag
EMBEDDER=minilm
EMBED_DIM=384

# 2. APP PACKAGE (`app/`)

## `app/__init__.py`
Empty. Makes `app` a Python package.

## `app/config.py`
Single source of truth for runtime configuration. Loads `.env` via `_load_dotenv()`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `DATABASE_URL` | "" | Postgres connection; must start with "postgres" |
| `SEC_USER_AGENT` | "FilingLens demo research@example.com" | Required by SEC |
| `SEC_MAX_REQUESTS_PER_SECOND` | 10 | SEC fair-access ceiling |
| `RRF_K` | 60 | RRF damping constant |
| `LEXICAL_WEIGHT` / `VECTOR_WEIGHT` | 1.0 | Fusion weights |
| `TOP_K_CANDIDATES` | 40 | Fusion pool size |
| `TOP_K_RETURNED` | 8 | Final chunks per answer |
| `EMBED_TOKENIZER` | sentence-transformers/all-MiniLM-L6-v2 | WordPiece tokenizer |
| `RERANK_ENABLED` | true | Cross-encoder toggle |
| `RERANK_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model |
| `EMBED_DIM` | 384 | Must match chunks.embedding vector(384) |
| `MAX_GRAPH_HOPS` | 3 | Graph traversal depth |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | "" / gemini-3.5-flash-lite | Optional LLM synthesis |

**Key interview point:** `EMBED_DIM` must match `Vector(EMBED_DIM)` in `models.py` and MiniLM's 384-d output. Change model → change both → re-ingest.

## `app/db.py`
Database engine and bootstrap:

- `engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)` — connection pool with pre-ping
- `SessionLocal` — session factory, `autoflush=False, autocommit=False`
- `init_db()` — called at API startup:

## `app/tenancy.py`
Server-side tenant context:

- `TenantContext(tenant_id, slug)` — simple dataclass
- `get_or_create_tenant(slug, name)` — lookup by slug; create if missing. Used by `ingest_sec.py` to create `sec`/`emerging`
- `resolve_tenant_by_slug(slug)` — called by every tenant-scoped API endpoint. Raises 404 if unknown

**Key interview point:** Tenant slug comes from server-side auth context, never client-supplied. In production this would be a JWT/OAuth claim. Swap `resolve_tenant` for real auth without touching callers.

## `app/models.py`
SQLAlchemy 2.0 declarative models. All tables carry `tenant_id` for isolation.

| Table | Key columns | Purpose |
|-------|-------------|---------|
| `tenants` | id, name, slug | Multi-tenant boundary |
| `companies` | id, cik, name, ticker | Public companies |
| `filings` | id, tenant_id, company_id, accession_number, form_type | SEC filings; idempotent on (tenant_id, accession_number) |
| `documents` | id, tenant_id, filing_id, raw_uri, content_hash | Raw filing document |
| `sections` | id, tenant_id, document_id, heading, section_path, page_start/end | Item/Part sections |
| `chunks` | id, tenant_id, document_id, text, embedding (Vector(384)), search_tsv | Atomic text units; HNSW + GIN indexes |
| `graph_nodes` | id, tenant_id, node_type, canonical_name | Company, Filing, SecurityClass, Person, Executive, BoardMember, GovernanceEvent |
| `graph_edges` | id, tenant_id, source_node_id, target_node_id, relation_type, evidence_chunk_id, confidence | Evidence-backed relations |

`chunks` indexes:
- `chunks_embedding_idx`: HNSW over `embedding` using `vector_cosine_ops` — ANN vector search
- `chunks_tsv_idx`: GIN over persisted `search_tsv` — full-text search

**Key interview point:** `search_tsv` is a PostgreSQL computed column: `to_tsvector('english', coalesce(text,''))`. Persisted + GIN-indexed → server-side lexical search without app-side full-text logic.

## `app/embeddings.py`
Local dense embedding layer:


## `app/ingestion/sec_client.py`
Thin client for SEC's public, key-free APIs (`data.sec.gov`, `www.sec.gov`):

- **Throttling:** Enforces SEC's 10 req/s ceiling
- **User-Agent:** Required by SEC; from `SEC_USER_AGENT` config
- **Disk caching:** Every response cached to `.sec_cache/` by SHA-256 of URL. Re-runs don't re-hit SEC
- Functions:
  - `resolve_cik(ticker)` — ticker → zero-padded 10-digit CIK
  - `list_all_filings(cik, form_types, limit, ...)` — walks `filings.recent` + paginated archives; returns newest-first capped at limit
  - `get_text(url)` — cached HTML getter
  - `filing_document_url(cik, accession, doc)` — constructs EDGAR archive URL
  - `get_companyfacts(cik)` — XBRL structured facts, MB-sized, cached per CIK

**Key interview point:** SEC blocks scripts without User-Agent and rate-limits at 10 req/s. Both handled here; disk caching makes ingestion idempotent and fast on re-runs.

## `app/ingestion/parser.py`
Turns raw filing documents into sections:

- `ParsedSection(dataclass)`: heading, ordinal, text, tables_markdown, page_start/end
- `_FilingHTMLParser` (stdlib `html.parser`):
  - Skips `<script>`, `<style>`, `<head>`, `ix:hidden`/`ix:header` (raw XBRL junk)
  - Detects `h1`-`h4` as section boundaries
  - Converts HTML tables to Markdown

## `app/ingestion/quality.py`
Rule-based quality flags — **never silently fixes numbers**:

- `sha256_text(text)` — content hash for dedup (filings, documents, chunks)
- `detect_quality_flags(text)` → flag dicts:
  - `ocr_numeric_suspect` — `l0%` (L instead of 1), `1O,000` (O instead of 0), `S1,200` (S instead of $). `high` severity, kept verbatim in text
  - `encoding_artifact` — control/replacement chars, `medium`
  - `low_content` — `< 20 chars`, `low`

**Key interview point:** Financial meaning preserved exactly. `l0%` stays `l0%` with a quality flag — an LLM never "fixes" financial/legal meaning.

## `app/ingestion/chunker.py`
Token-aware chunking bounded by MiniLM's real WordPiece tokenizer:

Budget math (WordPiece tokens):
- Contextual header (`company | form | date | section`) ≤ `CONTEXT_CAP` (48)
- Chunk text ≤ `HARD_CAP` (206)
- `[CLS]/[SEP]` = 2
- Total = 256 (MiniLM `max_seq_length`)

Constants:
- `TARGET_TOKENS = 200` — normal chunk budget
- `OVERLAP_TOKENS = 40` — repeated at boundaries so facts split across chunks aren't lost
- `HARD_CAP = 206` — never emit a chunk the embedder would truncate

## `app/ingestion/ingest.py`
Ingestion orchestrator — heart of the write path:

- `embed_text(text, metadata)` — prepends contextual header (`company | form | filed date | section`) to chunk text at embed time. Header stored in `metadata_json.context` at ingest time, prepended only at embed time — so `text` (and lexical index over it) stays clean for display while embedder sees where chunk came from
- `get_or_create_company(db, name, cik, ticker)` — idempotent company lookup/creation
- `get_or_create_node(db, tenant_id, node_type, canonical_name, properties)` — idempotent graph node lookup/creation
- `ingest_filing(tenant_id, filing_meta, raw_text, is_html, skip_embedding_rebuild)`:
  1. Idempotency check on `(tenant_id, accession_number)` — returns existing id if already ingested
  2. Creates company; finds prior same-form filing for `supersedes_filing_id` (temporal lineage)
  3. Parses document (`parser.parse_document`)
  4. Creates `Document`, then per `Section`:
     - Chunks section text + tables (`chunker`)
     - Per chunk: quality flags, `sha256` hash, creates `Chunk` row
     - Embeds inline (batched, 64 at a time) unless `skip_embedding_rebuild=True`
     - Graph extraction (`graph.extract.extract_facts`) — creates `GraphEdge` rows with `evidence_chunk_id` → exact chunk span
  5. Always creates `Company —FILED→ Filing` edge
  6. Creates `Filing —SUPERSEDES→ Filing` edge if temporal lineage exists
- `rebuild_embeddings(only_missing, batch_size)` — (re-)embeds chunks with NULL embedding. Streams IDs in batches so 150k+ chunk corpus never fits in memory. Commits per batch (giant executemany at ~700 MB gets Postgres-killed and rolls back everything). Returns count embedded.

## `app/graph/extract.py`
Rule-based entity/relation extraction over chunk text. Regex/pattern-based against governance-disclosure language (formulaic in 10-Ks/S-1s). Every extracted fact carries the exact matched span as evidence. **No generative model writes to the graph.**

Patterns:
- `_VOTES_PER_SHARE_RE` — "Class A common stock ... entitled to 10 votes" → `HAS_VOTING_RIGHTS` (SecurityClass → GovernanceEvent)
- `_SECURITY_CLASS_RE` — "Class A Common Stock" → `DESCRIBES` (Company → SecurityClass)
- `_ROLE_RE` — "John Smith, our Chief Executive Officer" → `HOLDS_ROLE` (Person → Executive/BoardMember)
- `_OWNERSHIP_RE` — "Jane Doe beneficially owns approximately 42% of our Class B common stock" → `OWNS` (Shareholder → SecurityClass)
- `_PEER_RE` — "our peer group includes Beta Industries..." → `PEER_OF` (Company → Company)

`extract_facts(text, self_company_name)` returns `ExtractedFact` (relation_type, source, source_type, target, target_type, evidence_span, properties).

`_clean_peer_name(raw)` — strips clause prefixes like "the following companies:" so peer lists store clean company names.

**Key interview point:** Every fact carries `evidence_span`. Graph stores companies you never ingested (mentioned in peer groups), flagged `in_corpus=False` so UI can distinguish them from your 16 ingested companies.

## `app/retrieval/__init__.py`
Empty. Makes `app.retrieval` a package.

## `app/retrieval/lexical.py`
Lexical retrieval (Postgres-only full-text search):

- `LexicalHit(chunk_id, score)`
- `lexical_search(db, tenant_id, query, top_k, filters)` — `plainto_tsquery('english', :q)` against persisted `search_tsv` TSVECTOR column, `ts_rank` scoring, GIN-indexed. Optional filters on company/form/date via joins. Tenant-scoped via `WHERE chunks.tenant_id = :tid`

**Key interview point:** Server-side full-text search. No Elasticsearch — PostgreSQL GIN + TSVECTOR handles it.

## `app/retrieval/vector.py`
Vector retrieval (pgvector HNSW ANN):

- `VectorHit(chunk_id, score)` where `score = 1.0 / (1.0 + distance)`
- `vector_search(db, tenant_id, query_embedding, top_k, filters)` — `chunks.embedding <=> CAST(:q AS vector)` cosine distance, HNSW index, tenant-scoped, `embedding IS NOT NULL` filter
- `merge_hits(*hit_lists)` — best-score merge across multiple vector searches

**Key interview point:** pgvector HNSW gives approximate nearest neighbor search with `vector_cosine_ops`.

## `app/retrieval/fusion.py`
Weighted Reciprocal Rank Fusion (RRF):

- `FusedHit(chunk_id, fused_score, lexical_rank, vector_rank)`
- `fuse(lexical_hits, vector_hits, top_k)` — RRF only needs each list's *rank order*, sidestepping incomparable scales (BM25 vs cosine). Score = `LEXICAL_WEIGHT * (1/(RRF_K + lex_rank + 1)) + VECTOR_WEIGHT * (1/(RRF_K + vec_rank + 1))`

**Key interview point:** RRF used instead of averaging raw scores because BM25 and cosine live on incomparable scales. RRF only needs rank ordering.

## `app/retrieval/rerank.py`
Post-fusion candidate selection: cross-encoder reranking + duplicate suppression:

- `MAX_SIM = 0.92` — cosine similarity above which a candidate counts as near-duplicate
- `_get_cross_encoder()` — lazily loads `CrossEncoder(RERANK_MODEL)`; falls back to RRF order if can't load
- `warm_up()` — background thread at API startup so first query doesn't pay model load cost
- `rerank(query, hits, texts)` — cross-encoder predicts relevance per (query, chunk) pair; reorders hits
- `select_final(db, tenant_id, query, pool, top_k)` — fetches texts/embeddings in one query, reranks, greedily picks while staying under `MAX_SIM` against already-picked chunks

**Key interview point:** Cross-encoder reads each (query, chunk) pair jointly — more accurate than bi-encoder cosine but slower. Applied only to top-40 fusion pool, not whole corpus.

## `app/graph/__init__.py`
Empty.


## `app/main.py`
FastAPI application — entry point for the read path:

- CORS middleware (`allow_origins=["*"]`)
- Startup event: `init_db()` (creates extensions/tables/RLS) + background thread `warm_up()` (cross-encoder)
- Endpoints:
  - `GET /health` — status ok
  - `GET /tenants` — list tenants for UI picker
  - `GET /companies?tenant_slug=` — companies in tenant (for dropdown)
  - `GET /companies/{company_name}/filings?tenant_slug=` — company's filings
  - `GET /compare?company_a=&company_b=&section=` — side-by-side sections, companion to PEER_OF edges
  - `GET /facts?ticker=&tag=&unit=&tenant_slug=` — XBRL company facts from SEC companyfacts API (disk-cached), no retrieval guesswork
  - `POST /query` — main Q&A: plan → hybrid retrieval → fuse → rerank → graph → answer builder → JSON with `answer`, `key_points`, `citations`, `debug` (per-stage timings)
  - `GET /graph?tenant_slug=&node_type=&relation=&company=&limit=` — nodes/edges for Graph view; parallel edges collapsed into weighted arcs; Company nodes flagged `in_corpus`
  - `GET /stats?tenant_slug=` — corpus overview (filings, chunks, embedded, graph nodes/edges, token stats, latest filing date)
- Static UI mounted at `/` via `StaticFiles(directory=frontend, html=True)` — serves `index.html`, `app.js`, `styles.css`

# 3. FRONTEND (`frontend/`)

## `frontend/index.html`
Minimal static UI served at `/` by the API. Two tabs:

- **Ask**: textarea for question, corpus/tenant dropdown, company/optional form filters, Ask button, answer output, citation cards, Scout evidence section, debug expander
- **Graph**: corpus/type/relation/focus/node-limit controls, Draw button, canvas area, legend (node types + relation types with counts), node detail panel

No build step — vanilla HTML/CSS/JS. Frontend calls relative API paths (`API = location.protocol === "file:" ? "http://localhost:8000" : ""`), so it works whether opened directly or served by the API.

## `frontend/app.js`
Vanilla JavaScript (~24 KB), no build step:

- `API` — base URL; empty string when served by API (same-origin), `http://localhost:8000` when opened as file
- `CHIPS` — example questions per corpus (`sec`, `emerging`)
- `esc(s)` — HTML escape to prevent injection from filing snippets
- `fmtNum(n)` — human-readable numbers (K/M/B suffixes)
- Tab switching — hides/shows `.view` sections
- `loadTenants()` — fetches `GET /tenants`, populates corpus dropdowns, defaults to `sec`
- `renderChips()` — example question pills
- `ask()` — POSTs to `/query`, renders answer, key points, citations, debug info, Scout evidence
- `loadGraph()` — fetches `GET /graph`, populates type/relation filter options from counts, initializes canvas state
- Graph rendering (hand-rolled force layout on `<canvas>`, no library):
  - `G` state object: nodes, edges, byId, scale, tx/ty, pan/zoom, selected/hover/drag
  - `gSeed()` — circular initial positions
  - `gFit()` — fits graph to canvas with padding
  - `gStep()` — force simulation: repulsion between all node pairs, springs along edges, center pull, velocity damping
  - `gLoop()` — requestAnimationFrame loop, auto re-fit when settled
  - `gDraw()` — draws edges as weighted arcs (line width ∝ log(count)), arrowheads, nodes colored by type (TYPE_COLORS), labels shown for companies/selected/hovered or when zoomed in
  - `gPick(x, y)` — hit-test nodes under cursor
  - Interaction: drag nodes, pan (drag background), scroll to zoom, click to select (shows detail panel with properties + edge table)

# 4. SCRIPTS (`scripts/`)

## `scripts/ingest_sec.py`
Bulk SEC EDGAR ingestion — main write-path entry point:

- CLI args: `--cik`, `--ticker`, `--tickers-file`, `--tenant` (default `sec`), `--forms` (default 10-K/10-Q/8-K), `--limit` (per company), `--start-date`, `--end-date`, `--delay` (default 0.15s, under SEC 10 req/s ceiling)
- Calls `init_db()`, `get_or_create_tenant()`
- For each target: resolve CIK, list filings, for each filing: download HTML (cached), `ingest_filing(..., skip_embedding_rebuild=True)`, print progress
- After all filings: `rebuild_embeddings(only_missing=True)` — embeds NULL chunks in batches
- Idempotent: same accession number = no-op; re-runs with `--start-date` pick up new filings incrementally

## `scripts/rebuild_corpus.py`
Wipe + re-ingest + re-embed after pipeline changes:

- `TRUNCATE TABLE graph_edges, graph_nodes, chunks, sections, documents, filings RESTART IDENTITY CASCADE`
- `--yes` to skip confirmation
- Same args as `ingest_sec.py`; `--skip-ingest` only verifies + reindexes; `--sanity-floor` fails if fewer filings than threshold

## `scripts/embed_missing.py`
Re-embed chunks whose `embedding` is NULL. Use after a crashed/rolled-back embed transaction. Safe to re-run (only NULL chunks touched). Calls `rebuild_embeddings(only_missing=True)`.

## `scripts/check_counts.py`
Corpus health check:

- Prints filings, chunks, chunks with embedding, graph nodes/edges, sample chunk length/tokens/embedding dim, sample text head
- With `query` argument, POSTs a test question to `http://localhost:8000/query` and prints answer + citations

## `scripts/eval_rag.py`
Golden-question retrieval evaluation:

- 20 golden questions with expected companies and keyword stems
- `evaluate(query_fn, golden)` — checks if expected company appears in top-8 citations with matching keyword; computes `company_hit@8`, `mrr`, `avg_citations`
- Env-tunable floors: `EVAL_MIN_HIT` / `EVAL_MIN_MRR`

## `scripts/eval_final.json`
Evaluation results artifact (0.90 hit@8, 0.90 MRR).

## `scripts/tickers_top16.txt`
16-company S&P corpus list (default data set).

## `scripts/tickers_emerging.txt`
Emerging names (Snowflake, Datadog, Coinbase) — used to create the `emerging` tenant.


# 5. DATA FLOW SUMMARY (Interview Cheat Sheet)

## Write Path (Ingestion)

```
SEC EDGAR (data.sec.gov)
    │
    ▼
sec_client.py (throttle 10 req/s, User-Agent, disk cache .sec_cache/)
    │
    ▼
ingest_sec.py (CLI: tickers, forms, limit, tenant)
    │
    ▼
ingest.py::ingest_filing()
    ├── parser.py       → sections (Item headings, tables→markdown)
    ├── chunker.py      → token-budgeted chunks (206 tokens max, 40 overlap)
    ├── quality.py      → sha256 hash + OCR flags (never rewrites numbers)
    ├── embeddings.py   → MiniLM 384-d vectors (normalized)
    └── graph/extract.py → regex facts (DESCRIBES, FILED, HOLDS_ROLE, PEER_OF, HAS_VOTING_RIGHTS)
           every edge has evidence_chunk_id → exact chunk span
    │
    ▼
Postgres 17 + pgvector
  tables: tenants, filings, documents, sections, chunks, graph_nodes, graph_edges
  indexes: HNSW (vector cosine), GIN (full-text TSVECTOR)
  RLS: tenant_id isolation
```

## Read Path (Query)

```
User question
    │
    ▼
main.py::POST /query
    │
    ├── planner.py      → QueryPlan (companies, temporal, peer, graph_hop intent)
    ├── lexical.py      → ts_rank + GIN (full-text)
    ├── vector.py       → MiniLM embed question → HNSW cosine ANN
    ├── fusion.py       → RRF fuse (top-40 pool)
    ├── rerank.py       → cross-encoder re-score + 0.92 cosine dedup → final 8
    └── graph_retriever.py → BFS up to 3 hops, evidence_chunk_ids attached
    │
    ▼
answer/builder.py → citations + graph facts + deterministic summary
    │
    ├── Gemini API (optional, structured JSON, fallback on failure)
    └── Ollama (optional polish, off by default)
    │
    ▼
JSON: answer, key_points, citations, debug timings, generation_mode
    │
    ▼
Frontend (vanilla JS + canvas force layout graph)
```

## Key Design Decisions (Interview Talking Points)

1. **One Postgres, not three systems:** pgvector gives vector search, PostgreSQL gives full-text search, relational tables give the graph. One transactional store instead of Neo4j + Pinecone + Elasticsearch.

2. **Evidence-backed graph, not LLM-vibes:** Every edge carries `evidence_chunk_id` and `confidence`. Nothing is written because a model "inferred" it.

3. **Tenant isolation enforced twice:** App filter + Postgres RLS on `tenant_id`.

4. **Numbers never LLM-corrected:** `l0%` stays `l0%` with a quality flag. LLM doesn't fix financial/legal meaning.

5. **Works with LLM turned off:** Retrieval, graph traversal, and citation-backed answers all function without any generation model. Gemini/Ollama are optional prose-polishing layers.

6. **Deterministic answers first:** The citation-backed answer is built first and always returned. Generation is polish, not source of truth.

7. **Token-exact chunking:** WordPiece tokenizer budget (206 + 48 + 2 = 256) prevents silent truncation of embedding tails.

8. **Idempotent ingestion:** Same accession number = no-op. Re-runs with `--start-date` pick up new filings incrementally.

9. **No paid APIs, no paid infra:** All free/open source. Local MiniLM embeddings, SEC EDGAR is free, Postgres + pgvector is free.

---
## `scripts/tickers_sp100.txt`
S&P 100 ticker list for bulk ingests.

## `scripts/*.log`, `scripts/*.err`
Run artifacts (gitignored): uvicorn logs, ingest logs, rebuild logs, embed logs, etc.

## `db/init.sql`
Full Postgres + pgvector + RLS schema. Run by Docker entrypoint on first volume init. Contains:
- `CREATE EXTENSION IF NOT EXISTS vector;`
- `CREATE EXTENSION IF NOT EXISTS pgcrypto;`
- All table definitions matching `app/models.py`
- HNSW index on `chunks.embedding`
- GIN index on `chunks.search_tsv`
- RLS policies

**Note:** `app/db.py` `init_db()` also runs this logic idempotently at startup, so the SQL file is a mirror/belt-and-suspenders.

---
- `neighboursOf(node)` — neighborhood highlight when a node is selected
- `renderNodeDetail(node)` — shows degree, valid dates, properties, connected edges table (relation, endpoint, count, evidence presence)

**Key interview point:** No frontend framework, no build step, nothing to install. Canvas force layout is ~400 lines of hand-rolled JavaScript.

## `frontend/styles.css`
CSS with CSS variables for light/dark mode (`color-scheme: light dark`). Styles for hero, tabs, cards, forms, answers, citations, graph canvas, legend, detail table, debug grid, compare grid, facts table, corpus bars. Responsive (`@media (max-width: 640px)`).

---

**Key interview point:** All endpoints are tenant-scoped. `/query` returns `debug` timing per stage (plan+filters, retrieval, fuse+rerank+dedup, graph, answer) for observability.

---
## `app/answer/__init__.py`
Empty.

## `app/answer/planner.py`
Rule-based query decomposition:

- `QueryPlan(dataclass)`: raw_question, mentioned_companies, wants_temporal, wants_peer_comparison, wants_graph_hop
- `build_plan(db, tenant_id, question)` — keyword matching against known graph `Company` nodes. Detects temporal markers ("changed", "across", "over time") and comparison markers ("peer", "compare", "versus"). Sets `wants_graph_hop` if temporal OR peer OR companies mentioned.

**Key interview point:** No LLM needed to decide *what kind* of question this is — keyword matching against known graph nodes is deterministic and inspectable.

## `app/answer/builder.py`
Evidence-first answer assembly:

- `Citation(dataclass)`: filing_accession, company, form_type, filing_date, section, page, snippet, filing_url
- `AnswerPayload(dataclass)`: question, summary, graph_facts, citations, generation_mode, key_points, citation_ids
- `build_answer(db, tenant_id, question, fused_hits, graph_paths)`:
  1. Builds citations from fused hits (hit → chunk → document → filing → company)
  2. Describes graph paths as `A --REL--> B` lines
  3. Builds summary: graph facts first, then supporting evidence
  4. If `GEMINI_API_KEY` set: calls Gemini REST API with structured JSON schema. On failure → deterministic answer
  5. Validates `citation_ids` are within range and deduplicated
- `_try_ollama_polish()` — optional local generation (off by default). Only rephrases existing evidence

**Key interview point:** Deterministic citation-backed answer built first and ALWAYS returned. If generation fails, deterministic answer is what caller gets — system works with LLM turned off.

---

**Key interview point:** `skip_embedding_rebuild=True` defers embedding for bulk loads (fast), then `rebuild_embeddings(only_missing=True)` embeds everything NULL once at end. Commits per batch prevent memory blowup and transaction rollback disasters.

---

Functions:
- `count_tokens(text)` — exact WordPiece count via `AutoTokenizer`
- `chunk_section_text(text, page_start, page_end)` — binary search over prefix sums of per-word token counts; defensive shrinking for pathological words; drops unembeddable noise rather than silently truncating
- `chunk_table_markdown(table_md)` — tables stay whole when they fit; oversized tables split into row groups repeating header rows

**Key interview point:** Old word-based budget (~220 words ≈ 280-320 WordPiece tokens) meant the tail 10-15% of every chunk never reached its embedding. Real-token counting makes the budget exact.

---
  - Tracks page via `<hr>` / `pagebreak`
- `_split_on_item_headings()` — Splits section text on lines like "Item 1A. Risk Factors" or "PART II". EDGAR HTML rarely uses real heading tags — headings exist as text lines. Heading line kept in body so full-text search matches
- `parse_document(raw, is_html=None)` — auto-detects HTML vs plain text

**Key interview point:** Dependency-free (stdlib `html.parser` + `re`) rather than BeautifulSoup — the transform is simple enough to justify a short dependency list.

---
- `MiniLMEmbedder`:
  - `MODEL_NAME = "all-MiniLM-L6-v2"` (384-d, CPU, no API key, ~90 MB first download)
  - `_ensure()` — lazy-loads `SentenceTransformer` once (process singleton)
  - `embed(texts)` — `model.encode(texts, normalize_embeddings=True)` → normalized 384-d vectors
- `get_embedder()` — module-level singleton. Loads once per process (~1s saved per request vs loading every time).

**Key interview point:** Embeddings are normalized (cosine = dot product). Same model at ingest (chunk embedding) and query (question embedding) → vector space matches.

## `app/ingestion/__init__.py`
Empty. Makes `app.ingestion` a package.

---
  1. Imports `app.models` to register SQLAlchemy metadata
  2. Runs `CREATE EXTENSION IF NOT EXISTS vector` and `pgcrypto`
  3. `models.Base.metadata.create_all(engine)` — creates all tables/indexes from `models.py`
  4. `_ensure_row_level_security()` — enables RLS on 6 tables + idempotent `tenant_isolation_<table>` policies
- `session_scope(tenant_id=None)` — yields a session; when `tenant_id` given, executes `SET LOCAL app.tenant_id = :tid` so Postgres RLS enforces tenant scope

**Key interview point:** Tenant isolation enforced **twice** — app-side filter + Postgres RLS. Prevents cross-tenant leakage even with app bugs.

---
SEC_USER_AGENT="Prakash prakash.mahatra18@gmail.com"
LEXICAL_WEIGHT=1.0
VECTOR_WEIGHT=1.0
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
USE_OLLAMA=false
...
```

## `.env.template`
Public template (committed to Git). Contains placeholder values like `GEMINI_API_KEY=your-key-here`. A collaborator copies it to `.env` and fills in their own key.

## `.dockerignore`
Prevents secrets and artifacts from entering the Docker image:
- `.env` (never bake secrets into the image)
- `__pycache__/`, `*.pyc`
- `.sec_cache/` (2 GB of SEC HTML cache — too large, regenerated on ingest)
- `.pytest_cache/`
- `scripts/*.log`, `scripts/*.err`

## `.gitignore`
Same exclusions for Git: `__pycache__/`, `*.pyc`, `.sec_cache/`, `.pytest_cache/`, `.env`, `scripts/*.log`, `scripts/*.err`

## `README.md`
User-facing documentation: two ways to run (real SEC data via Docker; production-shaped path), CLI flags for `ingest_sec.py`, available API endpoints (`GET /stats`, `GET /graph`, `GET /compare`, `GET /facts`, `POST /query`), engineering decisions (one Postgres instead of Neo4j+Pinecone+Elasticsearch, evidence-backed graph, tenant isolation twice, numbers never LLM-corrected, works with LLM turned off), repo layout, test commands.

## `DEPLOY.md`
Free deployment guide: Option A (Cloudflare Tunnel from your PC, temporary link), Option B (Oracle Cloud Always Free ARM VM, permanent, requires credit card verification but $0 usage). Includes instructions to push repo to GitHub without secrets, move data via `pg_dump`/`pg_restore` instead of re-ingesting 841 filings, and optional Caddy reverse proxy for HTTPS.

---