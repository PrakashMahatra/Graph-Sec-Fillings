"""
Vector retrieval layer (Postgres-only).

pgvector HNSW ANN search (`embedding <=> :q` cosine distance) directly in SQL
-- tenant-scoped, top_k limited server-side.
"""
from __future__ import annotations
from dataclasses import dataclass

from sqlalchemy import text


@dataclass
class VectorHit:
    chunk_id: str
    score: float


def vector_search(db, tenant_id: str, query_embedding: list[float] | None, top_k: int,
                  filters: dict | None = None) -> list[VectorHit]:
    if not query_embedding:
        return []
    joins, conds, params = "", "", {
        "q": "[" + ",".join(map(str, query_embedding)) + "]",
        "tid": tenant_id, "k": top_k,
    }
    if filters:
        joins = (" JOIN documents d ON d.id = chunks.document_id"
                 " JOIN filings f ON f.id = d.filing_id")
        if filters.get("company_id"):
            conds += " AND f.company_id = :cid"
            params["cid"] = filters["company_id"]
        if filters.get("form_type"):
            conds += " AND f.form_type = :form"
            params["form"] = filters["form_type"]
        if filters.get("date_from"):
            conds += " AND f.filing_date >= :dfrom"
            params["dfrom"] = filters["date_from"]
        if filters.get("date_to"):
            conds += " AND f.filing_date <= :dto"
            params["dto"] = filters["date_to"]
    rows = db.execute(
        text(
            "SELECT chunks.id, (chunks.embedding <=> CAST(:q AS vector)) AS dist "
            f"FROM chunks{joins} WHERE chunks.tenant_id = :tid "
            "AND chunks.embedding IS NOT NULL " + conds +
            " ORDER BY chunks.embedding <=> CAST(:q AS vector) LIMIT :k"
        ),
        params,
    ).all()
    return [VectorHit(chunk_id=r[0], score=1.0 / (1.0 + float(r[1]))) for r in rows]


def merge_hits(*hit_lists: list[VectorHit]) -> list[VectorHit]:
    """Best-score merge across multiple vector searches (multi-query
    expansion): keep each chunk's highest score, order by score desc."""
    best: dict[str, float] = {}
    for lst in hit_lists:
        for h in lst:
            if h.chunk_id not in best or h.score > best[h.chunk_id]:
                best[h.chunk_id] = h.score
    return [VectorHit(cid, s) for cid, s in sorted(best.items(), key=lambda kv: -kv[1])]
