# FilingLens

A multi-tenant **hybrid search + GraphRAG** engine over SEC filings (10-K, S-1,
proxy statements, etc.), built to answer questions plain vector search can't:

> "Trace how the voting share structure of Acme Holdings changed across its
> last three filings relative to its peer group."

Combines **lexical search + vector search + knowledge-graph traversal +
temporal reasoning**, with every answer backed by a filing/section/page
citation. **Zero paid APIs, zero paid infra.**

## Why this exists

Most "RAG demo" repos are a vector store and a prompt. This one is closer to
how a real financial-data platform is built: idempotent ingestion, quality
flags on OCR'd numbers, an evidence-backed knowledge graph (not
LLM-hallucinated edges), tenant isolation enforced server-side, and answers
that are deterministic and citable rather than free-form LLM prose.

## Two ways to run it

### 1. Real SEC data (bulk, free, no API key)

Needs outbound internet to `data.sec.gov` / `www.sec.gov` and a declared
`SEC_USER_AGENT` in `.env` (SEC requirement). **Postgres-only** — start the
database with Docker first; embeddings are local `all-MiniLM-L6-v2`
(384-d, CPU, no API key).

```bash
docker compose up -d            # pgvector/pg16 Postgres (+ API), first
pip install -r requirements.txt
cp .env.example .env   # then set SEC_USER_AGENT="Your Name your-email@example.com"

# Option A: 16-company S&P corpus (default data set, ~168k chunks)
python scripts/ingest_sec.py --tickers-file scripts/tickers_top16.txt \
    --forms 10-K 10-Q 8-K --limit 100 --tenant sec

# Option B: Emerging names (Snowflake/Datadog/Coinbase) -> 'emerging' tenant (~21k chunks)
python scripts/ingest_sec.py --tickers-file scripts/tickers_emerging.txt \
    --forms 10-K 10-Q 8-K --limit 30 --start-date 2023-10-01 --tenant emerging

# Option C: Single or comma-separated tickers
python scripts/ingest_sec.py --ticker AAPL,MSFT,NVDA --forms 10-K 10-Q 8-K --limit 50 --tenant sec

uvicorn app.main:app --reload
open http://127.0.0.1:8000/            # the API serves the UI; or open frontend/index.html directly
```

> To put the site on the public internet for free (share link or permanent
> VM), see **[DEPLOY.md](DEPLOY.md)**.

Try:
```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"tenant_slug":"sec","question":"What voting rights do Class B shares have?"}' | python -m json.tool
```

Flags: `--cik`, `--ticker AAPL,MSFT`, `--tickers-file scripts/tickers_top16.txt` (or `scripts/tickers_emerging.txt`),
`--forms 10-K 10-Q 8-K S-1 DEF 14A`, `--limit N` (per company),
`--start-date/--end-date YYYY-MM-DD`, `--tenant`, `--delay`.
Re-runs are idempotent (same accession number = no-op), so re-running with
`--start-date` picks up new filings incrementally.

Also available: `GET /stats` (corpus overview), `GET /graph?tenant_slug=sec`
(nodes/edges for the Graph view — filter with `node_type`, `relation`,
`company`, `limit`; parallel edges are collapsed into one weighted arc),
`GET /compare?company_a=A&company_b=B`
(shared sections side-by-side), `GET /facts?ticker=AAPL&tag=<us-gaap-tag>`
(XBRL structured facts from SEC's companyfacts API, disk-cached). `/query`
accepts optional `company`, `form_type`, `date_from`, `date_to` filters and
`debug=true` (per-stage timings + retrieval detail).

### 2. Production-shaped path (Postgres + pgvector, still free/local)

```bash
docker compose up -d            # pgvector/pgvector:pg16 + API with EMBEDDER=minilm
# first boot runs db/init.sql (vector/GIN/RLS schema); app startup backfills
# extensions + HNSW/GIN indexes + RLS idempotently via init_db()
python scripts/ingest_sec.py --ticker AAPL,MSFT --forms 10-K 10-Q --limit 20 --tenant sec
```

`chunks.embedding` is a real `vector(384)` pgvector column queried with
HNSW ANN (`<=>` cosine) in `app/retrieval/vector.py`; lexical is `ts_rank`
+ GIN in `app/retrieval/lexical.py`; embeddings come from local
`all-MiniLM-L6-v2` (384-d, CPU, no API key) via `app/embeddings.py`;
tenant isolation is enforced twice (app filter + Postgres RLS on
`app.tenant_id`).

`db/init.sql` has the full Postgres schema: `pgvector` HNSW index, GIN
full-text index, tenant_id + row-level-security policies, graph tables. The
app layer (`app/db.py:init_db()`) applies the same extensions + indexes +
RLS idempotently on startup, so `docker compose up` works even on an
existing volume.

**Render's free tier is for the demo API only** — 512MB RAM, spins down after
15 min idle, free Postgres expires after 30 days. This is explicitly not
presented as an SLA-backed production service.

## Architecture

```
Web UI ──HTTP──> FastAPI (auth/tenant ctx)
                   │
      ┌────────────┼─────────────┐
 Query Planner   Hybrid       Graph
 (decomposer)   Retriever    Retriever
      │        (lex+vec+RRF)     │
      └────────────┬─────────────┘
              Evidence / Rerank
                     │
              Answer Builder (citations)
                     │
        Postgres+pgvector (HNSW · ts_rank · RLS)
        chunks · lexical index · graph nodes/edges
        filings + metadata · tenants + ACL
```

## What's actually implemented

| Milestone | Status |
|---|---|
| 1. DB foundation (tenants/filings/documents/chunks/graph tables) | ✅ |
| 2. SEC ingestion (CIK resolution, EDGAR submissions API, rate-limited, cached, idempotent) | ✅ |
| 3. Document pipeline (section splitting, table→markdown, hashing, OCR quality flags, chunking) | ✅ |
| 4. Hybrid retrieval (Postgres-FTS-style lexical + vector + weighted RRF) | ✅ |
| 5. GraphRAG (rule-based, **evidence-linked** entity/relation extraction; temporal edges; multi-hop BFS traversal) | ✅ |
| 6. Answer engine (query decomposition, deterministic evidence assembly, citations; optional local Ollama for prose synthesis) | ✅ |
| 7. Multi-tenancy (tenant_id everywhere, server-side-only scoping, isolation tests) | ✅ |
| 8. Public demo (FastAPI + static frontend + real SEC corpus) | ✅ |
| 9. Retrieval quality loop (token-aware chunks, contextual headers, cross-encoder rerank, golden-set eval) | ✅ |
**On the corpus:** this repo is SEC-only now — `fixtures/` + synthetic seeding
were removed. `scripts/ingest_sec.py` hits `data.sec.gov` with a declared
User-Agent and a 10 req/s ceiling, walks `filings.recent` + paginated
`filings.files[]` archives for bulk `--limit` values, and loads filings into
Postgres (via docker compose). Point it at tickers/CIKs on your own
machine.

## Retrieval quality

Four mechanisms keep answers accurate, each measurable with the eval:

- **Token-aware chunking** — chunks are sized by the embedding model's real
  WordPiece tokenizer (target 200, hard cap 206, + 48-token contextual header
  + 2 specials = MiniLM's 256-token window exactly). The old word-based
  budget silently truncated ~10–15% of every chunk out of its embedding;
  oversized financial tables (largest was 25,439 tokens!) are now split into
  header-repeating row groups instead of being >99% truncated.
- **Contextual headers** — every chunk embeds with
  `company | form | filed date | section` prepended (stored in
  `metadata_json.context`; display text and the lexical index stay clean).
- **Cross-encoder rerank + diversity** — the RRF pool (40) is re-scored by
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, then greedily picked with a 0.92
  cosine duplicate threshold down to the final 8.
- **Optional structured LLM answers** — set `GEMINI_API_KEY` in `.env`
  (free key from Google AI Studio; plain REST, no extra packages) and
  `/query` returns `{answer, key_points, citation_ids}` synthesized ONLY
  from the numbered evidence, with citation ids validated server-side.
  No key (or any API failure) → the deterministic evidence answer stands.
- **Golden-set eval** — `pytest tests/test_eval.py` (in-process) or
  `python scripts/eval_rag.py --base-url http://127.0.0.1:8000` scores 20
  real questions as `company_hit@8` + MRR (floors tunable via
  `EVAL_MIN_HIT` / `EVAL_MIN_MRR`). Current 16-company / 3-year corpus
  (scripts/tickers_top16.txt): **0.90 hit@8 / 0.90 MRR** — saved in
  `scripts/eval_final.json`.

## Scripts: `rebuild_corpus.py` and `embed_missing.py`

### `scripts/rebuild_corpus.py` — Full wipe + re-ingest + re-embed

Use this when you change the chunker, embedding model, embedding input format,
or want a clean corpus from scratch. It:

1. **Truncates** `filings`, `documents`, `sections`, `chunks`, `graph_nodes`,
   `graph_edges` (full wipe, confirmed with `--yes`)
2. **Re-ingests** all filings from `.sec_cache` (no re-download from SEC if
   cached)
3. **Re-embeds** all chunks with MiniLM (commits per batch so a 150k+ chunk
   corpus never holds one giant transaction)
4. **Verifies** zero over-budget chunks
5. **Reindexes** HNSW vector index

Example — rebuild the 16-company S&P corpus:

```bash
python scripts/rebuild_corpus.py --yes --tickers-file scripts/tickers_top16.txt \
    --forms 10-K 10-Q 8-K --limit 100 --start-date 2023-10-01
```

Example — rebuild the emerging names corpus:

```bash
python scripts/rebuild_corpus.py --yes --tickers-file scripts/tickers_emerging.txt \
    --forms 10-K 10-Q 8-K --limit 30 --start-date 2023-10-01 --tenant emerging
```

### `scripts/embed_missing.py` — Embed only NULL-embedding chunks

Use this when a previous ingestion or embedding run crashed partway through,
leaving some chunks with `NULL` embeddings. It:

- Finds chunks where `embedding IS NULL`
- Embeds only those (skips already-embedded chunks)
- Is idempotent and safe to re-run

```bash
python scripts/embed_missing.py
```

**When to use which:**

| Situation | Command |
|-----------|---------|
| Changed chunker/embedding logic | `rebuild_corpus.py --yes` |
| Changed the ticker list or date range | `rebuild_corpus.py --yes` |
| Embedding run crashed midway | `embed_missing.py` |
| Just want to verify corpus health | `scripts/check_counts.py` |

Both scripts are **essential for day-to-day operation** — keep them in the repo.
Any user who ingests their own tickers will need `rebuild_corpus.py` if they
change pipeline settings, and `embed_missing.py` if a network hiccup leaves
embeddings incomplete.

## Key engineering decisions (and why)

- **PostgreSQL, not three separate systems.** `pgvector` gives exact + HNSW
  vector search, PostgreSQL gives full-text search, and plain relational
  tables give you a graph via adjacency — one transactional store instead of
  Neo4j + Pinecone + Elasticsearch on day one.
- **The graph is evidence-backed, not LLM-vibes.** Every edge carries
  `evidence_chunk_id` and `confidence`; nothing is written to the graph
  because a model "inferred" it without a source span.
- **Tenant scope is never client-supplied.** It comes from server-side auth
  context; every persistent row carries `tenant_id`; isolation has automated
  tests (`tests/test_tenant_isolation.py`).
- **Numbers are never LLM-corrected.** `l0%` stays `l0%` with a quality flag
  — an LLM does not get to "fix" financial or legal meaning.
- **The system works with the LLM turned off.** Retrieval, graph traversal,
  and citation-backed answers all function without any generation model;
  a local Ollama model is an optional prose-polishing layer, not the source
  of truth.

## Repo layout

```
app/
  models.py            SQLAlchemy models (tenants, filings, documents,
                        sections, chunks, graph_nodes, graph_edges)
  db.py                 engine/session + RLS/index bootstrap
  tenancy.py             server-side tenant context + guard
  embeddings.py            MiniLM local embedder (all-MiniLM-L6-v2, 384-d)
  ingestion/
    sec_client.py        EDGAR API client (rate-limited, cached, idempotent)
    parser.py             HTML section splitter + table→markdown
    quality.py             hashing + OCR/quality flags
    chunker.py              token-aware chunking with page/section locators
    ingest.py                 orchestrator
  graph/extract.py       rule-based entity/relation extraction, evidence-linked
  retrieval/
    lexical.py            FTS-style lexical scoring
    vector.py               cosine similarity vector search
    fusion.py                 weighted Reciprocal Rank Fusion
    rerank.py                  cross-encoder rerank + duplicate suppression
    graph_retriever.py         multi-hop BFS, temporal-aware
  answer/
    planner.py            rule-based query decomposition
    builder.py              evidence-first answer assembly + citations
  main.py                FastAPI app
scripts/ingest_sec.py    bulk SEC EDGAR ingestion (tickers/CIKs, forms, limits)
scripts/embed_missing.py re-embed chunks left NULL by a failed run
scripts/rebuild_corpus.py wipe + re-ingest + re-embed after pipeline changes
scripts/eval_rag.py      golden-question retrieval eval (baseline vs after)
scripts/check_counts.py  corpus health check (filings/chunks/embeddings/graph
                         counts) + a one-shot /query smoke test
scripts/tickers_top16.txt    16-company S&P corpus (default data set)
scripts/tickers_emerging.txt Snowflake/Datadog/Coinbase (`emerging` tenant)
scripts/tickers_sp100.txt    S&P 100 list (batch ingests via rebuild/check scripts)
frontend/index.html      minimal static UI: Ask (Q&A) + Graph (force-directed
                         knowledge-graph view) — served at / by the API
tests/                   tenant isolation, fusion, graph extraction, chunker
db/init.sql              full Postgres+pgvector+RLS schema
docker-compose.yml
```

## Tests

```bash
docker compose up -d db        # tests require the local Postgres
pytest tests/ -v
```

Covers: cross-tenant retrieval/graph isolation (0 leakage), RRF fusion
ordering, evidence-linked graph extraction, token-aware chunker budgets +
table splitting, rerank duplicate suppression, and a 20-question retrieval
quality eval (`tests/test_eval.py`) with env-tunable floors
(`EVAL_MIN_HIT` / `EVAL_MIN_MRR`).
