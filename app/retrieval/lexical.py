"""
Lexical retrieval layer (Postgres-only).

Real full-text search -- `search_tsv @@ plainto_tsquery` + `ts_rank`,
GIN-indexed (see db/init.sql). Exact, fast, server-side, tenant-scoped.
"""
from __future__ import annotations
from dataclasses import dataclass

from sqlalchemy import text


@dataclass
class LexicalHit:
    chunk_id: str
    score: float


def lexical_search(db, tenant_id: str, query: str, top_k: int,
                   filters: dict | None = None) -> list[LexicalHit]:
    """Optional `filters` = {company_id, form_type, date_from, date_to}
    (all None-tolerant) -> scoped to specific filings via documents/join."""
    joins, conds, params = "", "", {"q": query, "tid": tenant_id, "k": top_k}
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
            "SELECT chunks.id, ts_rank(chunks.search_tsv, plainto_tsquery('english', :q)) AS rank "
            f"FROM chunks{joins} WHERE chunks.tenant_id = :tid "
            "AND chunks.search_tsv @@ plainto_tsquery('english', :q) " + conds +
            " ORDER BY rank DESC LIMIT :k"
        ),
        params,
    ).all()
    return [LexicalHit(chunk_id=r[0], score=float(r[1])) for r in rows]
