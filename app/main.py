from __future__ import annotations
import re
import threading
import time
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, or_, select, text

from app.db import init_db, session_scope
from app.models import Chunk, Company, Document, Filing, GraphEdge, GraphNode, Section, Tenant
from app.tenancy import resolve_tenant_by_slug
from app.embeddings import get_embedder
from app.retrieval.lexical import lexical_search
from app.retrieval.vector import vector_search, merge_hits
from app.retrieval.fusion import fuse
from app.retrieval.rerank import select_final, warm_up
from app.retrieval.graph_retriever import find_node, traverse
from app.answer.planner import build_plan
from app.answer.builder import build_answer
from app.ingestion.chunker import HARD_CAP as CHUNK_TOKEN_BUDGET
from app.ingestion.sec_client import get_companyfacts, resolve_cik
from app.config import (BASE_DIR, RERANK_ENABLED, RERANK_POOL,
                        TOP_K_CANDIDATES, TOP_K_RETURNED)

app = FastAPI(title="FilingLens", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@app.on_event("startup")
def _startup():
    init_db()
    # Load the cross-encoder reranker in the background so the first query
    # doesn't pay the model load/download cost.
    threading.Thread(target=warm_up, daemon=True).start()


class QueryRequest(BaseModel):
    tenant_slug: str
    question: str
    # optional scoping / tooling (all have safe defaults)
    company: str | None = None        # company name, ticker, or CIK
    form_type: str | None = None      # e.g. "10-K"
    date_from: str | None = None      # YYYY-MM-DD inclusive
    date_to: str | None = None        # YYYY-MM-DD inclusive
    debug: bool = False               # include per-stage timings + retrieval detail


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tenants")
def list_tenants():
    """All tenants for the UI picker (slugs only -- no auth in this project)."""
    with session_scope() as db:
        rows = db.execute(select(Tenant).order_by(Tenant.slug)).scalars().all()
        return [{"slug": r.slug, "name": r.name} for r in rows]


@app.get("/companies")
def list_companies(tenant_slug: str):
    ctx = resolve_tenant_by_slug(tenant_slug)
    with session_scope() as db:
        rows = db.execute(
            select(Company)
            .join(Filing, Filing.company_id == Company.id)
            .where(Filing.tenant_id == ctx.tenant_id)
            .distinct()
            .order_by(Company.name)
        ).scalars().all()
        return [{"name": r.name, "cik": r.cik, "ticker": r.ticker} for r in rows]


@app.get("/companies/{company_name}/filings")
def company_filings(company_name: str, tenant_slug: str):
    ctx = resolve_tenant_by_slug(tenant_slug)
    with session_scope() as db:
        company = db.execute(select(Company).where(Company.name == company_name)).scalar_one_or_none()
        if not company:
            raise HTTPException(404, "unknown company")
        filings = db.execute(
            select(Filing).where(Filing.tenant_id == ctx.tenant_id, Filing.company_id == company.id)
            .order_by(Filing.filing_date.desc())
        ).scalars().all()
        return [
            {
                "accession_number": f.accession_number, "form_type": f.form_type,
                "filing_date": f.filing_date, "supersedes_filing_id": f.supersedes_filing_id,
            }
            for f in filings
        ]


def _lookup_company(db, value: str):
    """Resolve a filter value (name, ticker, or CIK) to a Company row."""
    q = value.strip()
    conds = [func.lower(Company.name) == q.casefold(),
             func.upper(Company.ticker) == q.upper()]
    if q.isdigit():
        conds.append(Company.cik == q.zfill(10))
    row = db.execute(select(Company).where(or_(*conds))).scalars().first()
    if not row:
        raise HTTPException(404, f"unknown company '{value}'")
    return row


def _resolve_filters(db, req: QueryRequest) -> dict:
    filters: dict = {}
    if req.company:
        filters["company_id"] = _lookup_company(db, req.company).id
    if req.form_type:
        filters["form_type"] = req.form_type
    for name, val in (("date_from", req.date_from), ("date_to", req.date_to)):
        if val:
            if not _ISO_DATE.match(val):
                raise HTTPException(422, f"{name} must be YYYY-MM-DD")
            filters[name] = val
    return filters


@app.post("/query")
def query(req: QueryRequest):
    """
    Hybrid retrieval (lexical + vector -> RRF -> cross-encoder rerank ->
    diversity pick) fused with graph traversal over entities implied in the
    question, tenant-scoped end-to-end. Deterministic, citation-backed
    answer. Optional company/form/date filters scope both retrievers;
    debug=true returns per-stage timings and retrieval detail.
    """
    ctx = resolve_tenant_by_slug(req.tenant_slug)
    tenant_id = ctx.tenant_id
    t0 = time.perf_counter()

    with session_scope() as db:
        plan = build_plan(db, tenant_id, req.question)
        filters = _resolve_filters(db, req)
        t_plan = time.perf_counter()

        embedder = get_embedder()
        try:
            q_vec = embedder.embed_one(req.question)
        except RuntimeError:
            q_vec = None

        lex_hits = lexical_search(db, tenant_id, req.question, TOP_K_CANDIDATES, filters=filters)
        vec_hits = vector_search(db, tenant_id, q_vec, TOP_K_CANDIDATES, filters=filters) if q_vec else []

        # Multi-query expansion: when the question names companies, also run
        # a company-anchored variant and keep each chunk's best score.
        variant_used = False
        if q_vec and plan.mentioned_companies:
            variant = " ".join(plan.mentioned_companies[:2]) + " " + req.question
            vq = embedder.embed_one(variant)
            vec_hits = merge_hits(
                vec_hits,
                vector_search(db, tenant_id, vq, TOP_K_CANDIDATES, filters=filters),
            )
            variant_used = True
        t_retrieve = time.perf_counter()

        pool = fuse(lex_hits, vec_hits, RERANK_POOL if RERANK_ENABLED else TOP_K_RETURNED)
        fused = select_final(db, tenant_id, req.question, pool, TOP_K_RETURNED)
        t_rerank = time.perf_counter()

        graph_paths = []
        if plan.wants_graph_hop:
            # Graph hops seed from companies explicitly named in the question.
            seed_names = plan.mentioned_companies
            for name in seed_names:
                node = find_node(db, tenant_id, "Company", name)
                if node:
                    relation_filter = None
                    if plan.wants_peer_comparison:
                        relation_filter = ["PEER_OF", "DESCRIBES", "HAS_VOTING_RIGHTS"]
                    elif plan.wants_temporal:
                        relation_filter = ["FILED", "SUPERSEDES", "DESCRIBES", "HAS_VOTING_RIGHTS"]
                    graph_paths.extend(traverse(db, tenant_id, node.id, relation_types=relation_filter))
        t_graph = time.perf_counter()

        answer = build_answer(db, tenant_id, req.question, fused, graph_paths)

        payload = {
            "question": req.question,
            "plan": asdict(plan),
            "answer": answer.summary,
            "key_points": answer.key_points,
            "citation_ids": answer.citation_ids,
            "graph_facts": answer.graph_facts,
            "citations": [asdict(c) for c in answer.citations],
            "generation_mode": answer.generation_mode,
        }
        if req.debug:
            ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
            payload["debug"] = {
                "timings_ms": {
                    "plan+filters": ms(t0, t_plan),
                    "retrieval": ms(t_plan, t_retrieve),
                    "fuse+rerank+dedup": ms(t_retrieve, t_rerank),
                    "graph": ms(t_rerank, t_graph),
                    "answer": ms(t_graph, time.perf_counter()),
                    "total": ms(t0, time.perf_counter()),
                },
                "filters": {k: str(v) for k, v in filters.items()},
                "rerank_enabled": RERANK_ENABLED,
                "variant_query_used": variant_used,
                "lexical_hits": len(lex_hits),
                "vector_hits": len(vec_hits),
                "pool_size": len(pool),
                "selected": [h.chunk_id for h in fused],
                "graph_paths": len(graph_paths),
            }
        return payload


@app.get("/stats")
def stats(tenant_slug: str):
    """Corpus size / freshness / embedding coverage at a glance."""
    ctx = resolve_tenant_by_slug(tenant_slug)
    with session_scope() as db:
        tid = ctx.tenant_id
        by_form = dict(db.execute(
            text("SELECT form_type, COUNT(*) FROM filings WHERE tenant_id = :tid GROUP BY form_type"),
            {"tid": tid},
        ).all())
        chunks_total, chunks_embedded, tok_max, tok_avg = db.execute(
            text("SELECT COUNT(*), COUNT(embedding), COALESCE(MAX(token_count), 0), "
                 "COALESCE(ROUND(AVG(token_count))::int, 0) "
                 "FROM chunks WHERE tenant_id = :tid"),
            {"tid": tid},
        ).one()
        nodes = db.execute(select(func.count(GraphNode.id)).where(GraphNode.tenant_id == tid)).scalar()
        edges = db.execute(select(func.count(GraphEdge.id)).where(GraphEdge.tenant_id == tid)).scalar()
        companies = db.execute(
            select(func.count(func.distinct(Filing.company_id))).where(Filing.tenant_id == tid)
        ).scalar()
        latest = db.execute(select(func.max(Filing.filing_date)).where(Filing.tenant_id == tid)).scalar()
        return {
            "tenant": tenant_slug,
            "filings": sum(by_form.values()),
            "filings_by_form": by_form,
            "companies": companies,
            "chunks": chunks_total,
            "chunks_embedded": chunks_embedded,
            "token_count": {"max": tok_max, "avg": tok_avg, "budget": CHUNK_TOKEN_BUDGET},
            "graph": {"nodes": nodes, "edges": edges},
            "latest_filing_date": latest,
        }


@app.get("/graph")
def graph_view(
    tenant_slug: str,
    node_type: str | None = None,   # e.g. "Company", "Filing"
    relation: str | None = None,    # e.g. "DESCRIBES", "FILED"
    company: str | None = None,     # seed node label (or substring) to focus on
    limit: int = 300,               # max nodes returned, ranked by degree
):
    """Node/edge payload for the UI's Graph view.

    Tenant-scoped end-to-end. The whole tenant graph is small (~10^3 rows),
    so filtering and degree ranking happen in Python rather than in SQL --
    it keeps the logic readable and the response is one round trip.

    Filtering order: relation -> company seed expansion -> node_type ->
    top-N by degree (seed nodes are always kept so the view stays anchored)
    -> edges induced on the surviving nodes.
    """
    ctx = resolve_tenant_by_slug(tenant_slug)
    limit = max(10, min(limit, 1000))

    with session_scope() as db:
        tid = ctx.tenant_id
        # Company nodes are created from *text mentions* too (e.g. a filing's
        # peer group), so the graph can hold companies you never ingested.
        # Flag which ones are actually in the corpus so the UI can tell them
        # apart instead of reporting "Company: 18" when you ingested 16.
        corpus_names = {
            nm.casefold() for nm in db.execute(
                select(Company.name)
                .join(Filing, Filing.company_id == Company.id)
                .where(Filing.tenant_id == tid)
            ).scalars().all()
        }
        nodes = [
            {"id": str(n.id), "type": n.node_type, "label": n.canonical_name,
             "properties": n.properties or {}, "valid_from": n.valid_from,
             "valid_to": n.valid_to}
            for n in db.execute(
                select(GraphNode).where(GraphNode.tenant_id == tid)
            ).scalars().all()
        ]
        for n in nodes:
            if n["type"] == "Company":
                n["in_corpus"] = n["label"].casefold() in corpus_names
            else:
                n["in_corpus"] = True
        edges = [
            {"id": str(e.id), "source": str(e.source_node_id),
             "target": str(e.target_node_id), "relation": e.relation_type,
             "valid_from": e.valid_from, "valid_to": e.valid_to,
             "confidence": e.confidence,
             "evidence": e.evidence_chunk_id is not None}
            for e in db.execute(
                select(GraphEdge).where(GraphEdge.tenant_id == tid)
            ).scalars().all()
        ]

    # Filter options reflect what this corpus actually contains.
    node_type_counts: dict[str, int] = {}
    for n in nodes:
        node_type_counts[n["type"]] = node_type_counts.get(n["type"], 0) + 1
    relation_counts: dict[str, int] = {}
    for e in edges:
        relation_counts[e["relation"]] = relation_counts.get(e["relation"], 0) + 1

    if relation:
        edges = [e for e in edges if e["relation"] == relation]

    # Adjacency over the relation-filtered edge list (undirected: the UI draws
    # both Company--FILED-->Filing and Person--HOLDS_ROLE-->Executive).
    adjacency: dict[str, set[str]] = {}
    for e in edges:
        adjacency.setdefault(e["source"], set()).add(e["target"])
        adjacency.setdefault(e["target"], set()).add(e["source"])

    seeds: set[str] = set()
    if company:
        q = company.strip().casefold()
        seeds = {n["id"] for n in nodes if q and q in n["label"].casefold()}
        # Seed + its 2-hop neighbourhood, so "Apple" shows that company, its
        # filings, its share classes and its officers -- not the whole corpus.
        region, frontier = set(seeds), set(seeds)
        for _ in range(2):
            frontier = {nb for s in frontier for nb in adjacency.get(s, ())}
            region |= frontier
        nodes = [n for n in nodes if n["id"] in region]
    elif relation:
        # A relation filter on its own should show that relation's shape, not
        # 300 arbitrary nodes with two lonely edges between them.
        nodes = [n for n in nodes if n["id"] in adjacency]

    if node_type:
        visible = {n["id"] for n in nodes if n["type"] == node_type}
        # Seeds anchor the view even when they fall outside the type filter
        # (otherwise "company=Apple & type=Filing" would show orphaned filings).
        visible |= seeds
        nodes = [n for n in nodes if n["id"] in visible]

    # Induce the edge list on the visible nodes, then collapse parallel edges.
    # The same assertion is stored once per filing that states it (4,351
    # Company--DESCRIBES-->SecurityClass rows across ~72 real pairs), so the
    # renderer should see one weighted arc, not thousands of duplicates.
    def _induce_and_collapse(edges: list[dict], ids: set[str]) -> list[dict]:
        merged: dict[tuple[str, str, str], dict] = {}
        for e in edges:
            if e["source"] not in ids or e["target"] not in ids:
                continue
            key = (e["source"], e["target"], e["relation"])
            hit = merged.get(key)
            if hit is None:
                merged[key] = {**e, "count": 1}
            else:
                hit["count"] += 1
                hit["evidence"] = hit["evidence"] or e["evidence"]
                hit["confidence"] = max(hit["confidence"] or 0.0, e["confidence"] or 0.0)
        return list(merged.values())

    live_ids = {n["id"] for n in nodes}
    visible_edges = _induce_and_collapse(edges, live_ids)

    degree = {i: 0 for i in live_ids}
    for e in visible_edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    if len(nodes) > limit:
        # Keep seeds + Company anchors, then fill the rest by degree.
        keep = set(seeds) | {n["id"] for n in nodes if n["type"] == "Company"}
        ranked = sorted((n for n in nodes if n["id"] not in keep),
                        key=lambda n: (-degree[n["id"]], n["label"]))
        keep |= {n["id"] for n in ranked[: max(0, limit - len(keep))]}
        nodes = [n for n in nodes if n["id"] in keep]
        live_ids = keep
        visible_edges = _induce_and_collapse(edges, live_ids)
        degree = {i: 0 for i in live_ids}
        for e in visible_edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

    for n in nodes:
        n["degree"] = degree[n["id"]]

    return {
        "tenant": tenant_slug,
        "filters": {"node_type": node_type, "relation": relation, "company": company},
        "limit": limit,
        "totals": {"nodes": sum(node_type_counts.values()),
                   "edges": sum(relation_counts.values())},
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "returned": {"nodes": len(nodes), "edges": len(visible_edges)},
        "nodes": nodes,
        "edges": visible_edges,
    }


@app.get("/compare")
def compare(company_a: str, company_b: str, tenant_slug: str,
            form_type: str = "10-K", section: str | None = None, limit: int = 20):
    """Side-by-side sections from each company's latest filing of form_type
    (shared headings only) -- companion to the graph's PEER_OF edges for
    peer-anchored questions. Snippets are the first chunk of each section."""
    ctx = resolve_tenant_by_slug(tenant_slug)
    with session_scope() as db:
        tid = ctx.tenant_id

        def latest_sections(company_id):
            filing = db.execute(
                select(Filing).where(
                    Filing.tenant_id == tid, Filing.company_id == company_id,
                    Filing.form_type == form_type,
                ).order_by(Filing.filing_date.desc())
            ).scalars().first()
            if not filing:
                return None, {}
            doc = db.execute(select(Document).where(Document.filing_id == filing.id)).scalars().first()
            if not doc:
                return filing, {}
            rows = db.execute(
                select(Section.heading, Chunk.text)
                .join(Section, Section.id == Chunk.section_id)
                .where(Chunk.document_id == doc.id, Chunk.chunk_index == 0)
                .order_by(Section.ordinal)
            ).all()
            secs: dict[str, tuple[str, str]] = {}
            for heading, body in rows:
                # Normalize so "Item 1A.\xa0\xa0\xa0\xa0Risk Factors" (Apple,
                # non-breaking spaces) matches "Item 1A. Risk Factors" (peer).
                key = re.sub(r"\s+", " ", heading or "Document").strip().rstrip(".").casefold()
                secs.setdefault(key, (heading or "Document", body))
            return filing, secs

        a = _lookup_company(db, company_a)
        b = _lookup_company(db, company_b)
        filing_a, secs_a = latest_sections(a.id)
        filing_b, secs_b = latest_sections(b.id)
        if not filing_a:
            raise HTTPException(404, f"no {form_type} filing for '{company_a}'")
        if not filing_b:
            raise HTTPException(404, f"no {form_type} filing for '{company_b}'")

        shared = sorted(set(secs_a) & set(secs_b))
        if section:
            shared = [h for h in shared if section.casefold() in h]
        pairs = [
            {"section": secs_a[h][0],
             company_a: secs_a[h][1][:600],
             company_b: secs_b[h][1][:600]}
            for h in shared[:limit]
        ]
        return {
            "form_type": form_type,
            "filings": {
                company_a: {"accession": filing_a.accession_number, "date": filing_a.filing_date},
                company_b: {"accession": filing_b.accession_number, "date": filing_b.filing_date},
            },
            "sections_shared": len(shared),
            "pairs": pairs,
        }


@app.get("/facts")
def company_facts(ticker: str, tag: str | None = None, unit: str | None = None,
                  tenant_slug: str = "sec"):
    """Structured XBRL facts from SEC's companyfacts API (cached in
    .sec_cache). tag=None lists available us-gaap tags; with a tag, returns
    the most recent reported values -- exact numbers for the questions that
    prose retrieval can only approximate."""
    resolve_tenant_by_slug(tenant_slug)
    try:
        cik = resolve_cik(ticker)
    except ValueError as e:
        raise HTTPException(404, str(e))
    try:
        facts = get_companyfacts(cik)
    except Exception as e:
        raise HTTPException(502, f"SEC companyfacts fetch failed: {e}")
    gaap = facts.get("facts", {}).get("us-gaap", {})
    entity = facts.get("entityName")
    if not tag:
        return {"ticker": ticker.upper(), "cik": cik, "entity_name": entity,
                "tags_available": len(gaap), "sample_tags": sorted(gaap)[:50]}
    entry = gaap.get(tag)
    if entry is None:
        raise HTTPException(404, f"tag '{tag}' not present for {entity}")
    units = entry.get("units", {})
    if not units:
        raise HTTPException(404, f"tag '{tag}' has no units for {entity}")
    u = unit or next(iter(units))
    if u not in units:
        raise HTTPException(404, f"unit '{u}' not present for tag '{tag}'")
    recent = units[u][-10:]
    return {"ticker": ticker.upper(), "cik": cik, "entity_name": entity,
            "tag": tag, "label": entry.get("label"), "unit": u,
            "values": [{"start": r.get("start"), "end": r.get("end"),
                        "fy": r.get("fy"), "fp": r.get("fp"),
                        "form": r.get("form"), "val": r.get("val")} for r in recent]}


# Static UI, mounted last so it never shadows the API routes above.
# "/" serves frontend/index.html (html=True) -> open http://127.0.0.1:8000/.
app.mount("/", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="frontend")
