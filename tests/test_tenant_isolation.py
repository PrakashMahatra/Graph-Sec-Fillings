"""
Cross-tenant isolation is the single most important correctness property
of this system. These tests ingest the same filing content under two
different tenants and assert zero leakage across: chunk retrieval (lexical,
vector, fused), and graph nodes/edges/traversal.

Postgres-only: requires the local Docker database (docker compose up -d db)
and DATABASE_URL (defaults to the compose dev URL).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://graphrag:graphrag_dev_only@localhost:5432/graphrag",
)

from app.db import init_db, session_scope  # noqa: E402
from app.tenancy import get_or_create_tenant  # noqa: E402
from app.ingestion.ingest import ingest_filing  # noqa: E402
from app.retrieval.lexical import lexical_search  # noqa: E402
from app.retrieval.vector import vector_search  # noqa: E402
from app.retrieval.graph_retriever import find_node, traverse  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from sqlalchemy import select, text  # noqa: E402
from app.models import Chunk, GraphNode, GraphEdge  # noqa: E402


FILING_META_TEMPLATE = {
    "company_name": "Acme Holdings",
    "cik": "0001111111",
    "ticker": "ACME",
    "form_type": "10-K",
    "filing_date": "2024-11-08",
    "period_end_date": "2024-09-30",
    "filing_url": "https://example.com/filing.htm",
}

RAW_TEXT = (
    "CAPITAL STOCK AND VOTING RIGHTS\n\n"
    "Each share of our Class A Common Stock is entitled to 1 vote per share. "
    "Each share of our Class B Common Stock is entitled to 10 votes per share.\n\n"
    "EXECUTIVE OFFICERS AND DIRECTORS\n\n"
    "John Carter, our Chief Executive Officer, founded the company in 2015.\n\n"
    "COMPETITION AND PEER GROUP\n\n"
    "Our peer group includes Beta Industries and Gamma Corp."
)


def setup_module(module):
    init_db()
    module.tenant_a = get_or_create_tenant("tenant-a")
    module.tenant_b = get_or_create_tenant("tenant-b")

    meta_a = {**FILING_META_TEMPLATE, "accession_number": "0001-A"}
    meta_b = {**FILING_META_TEMPLATE, "accession_number": "0001-B"}

    ingest_filing(module.tenant_a, meta_a, RAW_TEXT, is_html=False)
    ingest_filing(module.tenant_b, meta_b, RAW_TEXT, is_html=False)


def teardown_module(module):
    """Remove the fixtures again. Without this, every `pytest` run leaves
    'tenant-a'/'tenant-b' in the tenants table, where the demo UI's corpus
    picker would show them as if they were real data."""
    tid_a, tid_b = getattr(module, "tenant_a", None), getattr(module, "tenant_b", None)
    if not tid_a or not tid_b:
        return
    with session_scope() as db:
        for tid in (tid_a, tid_b):
            for tbl in ("graph_edges", "graph_nodes", "chunks", "sections",
                        "documents", "filings", "users"):
                db.execute(text(f"DELETE FROM {tbl} WHERE tenant_id = :t"), {"t": tid})
            db.execute(text("DELETE FROM tenants WHERE id = :t"), {"t": tid})


def test_chunks_are_tenant_scoped():
    with session_scope() as db:
        chunks_a = db.execute(select(Chunk).where(Chunk.tenant_id == tenant_a)).scalars().all()
        chunks_b = db.execute(select(Chunk).where(Chunk.tenant_id == tenant_b)).scalars().all()
        assert len(chunks_a) > 0 and len(chunks_b) > 0
        ids_a = {c.id for c in chunks_a}
        ids_b = {c.id for c in chunks_b}
        assert ids_a.isdisjoint(ids_b), "chunk ids leaked across tenants"


def test_lexical_search_does_not_cross_tenants():
    with session_scope() as db:
        hits_a = lexical_search(db, tenant_a, "voting rights Class B", top_k=10)
        chunk_ids = {c.id for c in db.execute(select(Chunk).where(Chunk.tenant_id == tenant_a)).scalars()}
        assert all(h.chunk_id in chunk_ids for h in hits_a)


def test_vector_search_does_not_cross_tenants():
    with session_scope() as db:
        embedder = get_embedder()
        q_vec = embedder.embed_one("voting rights")
        hits_a = vector_search(db, tenant_a, q_vec, top_k=10)
        chunk_ids_a = {c.id for c in db.execute(select(Chunk).where(Chunk.tenant_id == tenant_a)).scalars()}
        chunk_ids_b = {c.id for c in db.execute(select(Chunk).where(Chunk.tenant_id == tenant_b)).scalars()}
        assert all(h.chunk_id in chunk_ids_a for h in hits_a)
        assert not any(h.chunk_id in chunk_ids_b for h in hits_a)


def test_graph_nodes_are_tenant_scoped():
    with session_scope() as db:
        nodes_a = db.execute(select(GraphNode).where(GraphNode.tenant_id == tenant_a)).scalars().all()
        nodes_b = db.execute(select(GraphNode).where(GraphNode.tenant_id == tenant_b)).scalars().all()
        ids_a = {n.id for n in nodes_a}
        ids_b = {n.id for n in nodes_b}
        assert ids_a.isdisjoint(ids_b)


def test_graph_traversal_never_crosses_tenants():
    with session_scope() as db:
        node_a = find_node(db, tenant_a, "Company", "Acme Holdings")
        assert node_a is not None
        paths = traverse(db, tenant_a, node_a.id)
        # every node visited in every path must belong to tenant_a's graph
        tenant_a_names = {
            n.canonical_name
            for n in db.execute(select(GraphNode).where(GraphNode.tenant_id == tenant_a)).scalars()
        }
        for p in paths:
            for name in p.node_names:
                assert name in tenant_a_names


def test_cross_tenant_edge_ids_disjoint():
    with session_scope() as db:
        edges_a = db.execute(select(GraphEdge).where(GraphEdge.tenant_id == tenant_a)).scalars().all()
        edges_b = db.execute(select(GraphEdge).where(GraphEdge.tenant_id == tenant_b)).scalars().all()
        ids_a = {e.id for e in edges_a}
        ids_b = {e.id for e in edges_b}
        assert ids_a.isdisjoint(ids_b)
        assert len(edges_a) > 0 and len(edges_b) > 0
