from __future__ import annotations
from sqlalchemy import select

from app.db import session_scope
from app.models import Company, Filing, Document, Section, Chunk, GraphNode, GraphEdge
from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_section_text, chunk_table_markdown, fit_context
from app.ingestion.quality import sha256_text, detect_quality_flags
from app.graph.extract import extract_facts
from app.embeddings import get_embedder

EMBED_BATCH = 64  # MiniLM batch size for bulk loads (CPU-friendly)


def embed_text(text: str, metadata: dict | None) -> str:
    """Embedding input = contextual header + chunk text. The header (company |
    form | filed date | section) is stored per-chunk in metadata_json at
    ingest time, prepended only at embed time -- so `text` (and the lexical
    index over it) stays clean for display while the embedder sees where the
    chunk came from (Anthropic-style contextual headers)."""
    ctx = (metadata or {}).get("context")
    return f"{ctx}\n\n{text}" if ctx else text


def get_or_create_company(db, name: str, cik: str | None, ticker: str | None) -> Company:
    row = db.execute(select(Company).where(Company.name == name)).scalar_one_or_none()
    if row:
        return row
    c = Company(name=name, cik=cik, ticker=ticker)
    db.add(c)
    db.flush()
    return c


def get_or_create_node(db, tenant_id: str, node_type: str, canonical_name: str, properties: dict | None = None) -> GraphNode:
    row = db.execute(
        select(GraphNode).where(
            GraphNode.tenant_id == tenant_id,
            GraphNode.node_type == node_type,
            GraphNode.canonical_name == canonical_name,
        )
    ).scalar_one_or_none()
    if row:
        return row
    node = GraphNode(tenant_id=tenant_id, node_type=node_type, canonical_name=canonical_name,
                      properties=properties or {})
    db.add(node)
    db.flush()
    return node


def ingest_filing(tenant_id: str, filing_meta: dict, raw_text: str, is_html: bool | None = None,
                skip_embedding_rebuild: bool = False) -> str:
    """
    filing_meta: {company_name, cik, ticker, form_type, accession_number,
                   filing_date, period_end_date, filing_url}
    Idempotent: (tenant_id, accession_number) is unique -- re-ingesting the
    same accession number is a no-op that returns the existing filing id.
    skip_embedding_rebuild=True defers embedding this filing's chunks (leave
    NULL) -- use for bulk loads, then call rebuild_embeddings(only_missing=True)
    once at the end.
    """
    with session_scope() as db:
        existing = db.execute(
            select(Filing).where(
                Filing.tenant_id == tenant_id,
                Filing.accession_number == filing_meta["accession_number"],
            )
        ).scalar_one_or_none()
        if existing:
            return existing.id

        company = get_or_create_company(
            db, filing_meta["company_name"], filing_meta.get("cik"), filing_meta.get("ticker")
        )

        # temporal lineage: does this filing supersede an earlier one of the
        # same form_type for the same company/tenant?
        prior = db.execute(
            select(Filing)
            .where(Filing.tenant_id == tenant_id, Filing.company_id == company.id,
                   Filing.form_type == filing_meta["form_type"])
            .order_by(Filing.filing_date.desc())
        ).scalars().first()

        filing = Filing(
            tenant_id=tenant_id,
            company_id=company.id,
            accession_number=filing_meta["accession_number"],
            cik=filing_meta.get("cik"),
            ticker=filing_meta.get("ticker"),
            form_type=filing_meta["form_type"],
            filing_date=filing_meta["filing_date"],
            period_end_date=filing_meta.get("period_end_date"),
            filing_url=filing_meta.get("filing_url"),
            source_hash=sha256_text(raw_text),
            supersedes_filing_id=prior.id if prior and prior.filing_date < filing_meta["filing_date"] else None,
        )
        db.add(filing)
        db.flush()

        document = Document(
            tenant_id=tenant_id, filing_id=filing.id, document_type=filing_meta["form_type"],
            raw_uri=filing_meta.get("filing_url", ""), content_hash=sha256_text(raw_text),
        )
        db.add(document)
        db.flush()

        sections = parse_document(raw_text, is_html=is_html)
        all_chunk_rows: list[Chunk] = []

        for sec in sections:
            section_row = Section(
                tenant_id=tenant_id, document_id=document.id, heading=sec.heading,
                section_path=sec.heading, ordinal=sec.ordinal,
                page_start=sec.page_start, page_end=sec.page_end,
            )
            db.add(section_row)
            db.flush()

            context = fit_context(
                f"{filing_meta['company_name']} | {filing_meta['form_type']} | "
                f"filed {filing_meta['filing_date']} | section: {sec.heading}"
            )
            candidates = chunk_section_text(sec.text, sec.page_start, sec.page_end)
            for table_md in sec.tables_markdown:
                candidates.extend(chunk_table_markdown(table_md))

            for i, cand in enumerate(candidates):
                flags = detect_quality_flags(cand.text)
                chunk = Chunk(
                    tenant_id=tenant_id, document_id=document.id, section_id=section_row.id,
                    chunk_index=i, text=cand.text, text_hash=sha256_text(cand.text),
                    token_count=cand.token_count, page_start=cand.page_start, page_end=cand.page_end,
                    source_locator={"heading": sec.heading, "page_start": cand.page_start},
                    metadata_json={"context": context},
                    quality_flags=flags,
                )
                db.add(chunk)
                all_chunk_rows.append(chunk)

        db.flush()

        # --- embeddings: local MiniLM (pretrained, 384-d), batched ---
        # For bulk loads pass skip_embedding_rebuild=True and call
        # rebuild_embeddings(only_missing=True) once at the end; otherwise
        # embed inline (batched, cheap) so progress survives a restart.
        if not skip_embedding_rebuild:
            embedder = get_embedder()
            embedder.fit([])  # no-op: ensures model loads
            for i in range(0, len(all_chunk_rows), EMBED_BATCH):
                batch_rows = all_chunk_rows[i:i + EMBED_BATCH]
                vectors = embedder.embed([embed_text(c.text, c.metadata_json) for c in batch_rows])
                for chunk, vec in zip(batch_rows, vectors):
                    chunk.embedding = vec

        # --- evidence-linked graph extraction ---
        for chunk in all_chunk_rows:
            facts = extract_facts(chunk.text, self_company_name=filing_meta["company_name"])
            for fact in facts:
                src_node = get_or_create_node(db, tenant_id, fact.source_type, fact.source)
                tgt_node = get_or_create_node(db, tenant_id, fact.target_type, fact.target, fact.properties)
                edge = GraphEdge(
                    tenant_id=tenant_id, source_node_id=src_node.id, target_node_id=tgt_node.id,
                    relation_type=fact.relation_type, filing_id=filing.id, evidence_chunk_id=chunk.id,
                    properties={**fact.properties, "evidence_span": fact.evidence_span},
                    valid_from=filing_meta["filing_date"], confidence=1.0,
                )
                db.add(edge)

        # Company --FILED--> Filing edge (always, evidence = the filing itself)
        company_node = get_or_create_node(db, tenant_id, "Company", filing_meta["company_name"])
        filing_node = get_or_create_node(db, tenant_id, "Filing", filing_meta["accession_number"],
                                          {"form_type": filing_meta["form_type"], "filing_date": filing_meta["filing_date"]})
        db.add(GraphEdge(
            tenant_id=tenant_id, source_node_id=company_node.id, target_node_id=filing_node.id,
            relation_type="FILED", filing_id=filing.id, valid_from=filing_meta["filing_date"], confidence=1.0,
        ))

        if filing.supersedes_filing_id:
            prior_filing_node = get_or_create_node(db, tenant_id, "Filing", prior.accession_number)
            db.add(GraphEdge(
                tenant_id=tenant_id, source_node_id=filing_node.id, target_node_id=prior_filing_node.id,
                relation_type="SUPERSEDES", filing_id=filing.id, valid_from=filing_meta["filing_date"], confidence=1.0,
            ))

        return filing.id


def rebuild_embeddings(only_missing: bool = False, batch_size: int = 2000) -> int:
    """(Re-)embed chunks with MiniLM. only_missing=True embeds only chunks
    whose embedding is NULL (after bulk loads with skip_embedding_rebuild);
    only_missing=False re-embeds everything (after chunker/context changes).
    Streams ids in batches so a 150k+ chunk corpus never has to fit in
    memory, and prints progress. Returns number of chunks embedded."""
    with session_scope() as db:
        ids_q = select(Chunk.id)
        if only_missing:
            ids_q = ids_q.where(Chunk.embedding.is_(None))
        ids = [r[0] for r in db.execute(ids_q.order_by(Chunk.id)).all()]
        if not ids:
            return 0
        embedder = get_embedder()
        embedder.fit([])  # no-op: ensures model loads once
        done = 0
        for i in range(0, len(ids), batch_size):
            rows = db.execute(
                select(Chunk).where(Chunk.id.in_(ids[i:i + batch_size]))
            ).scalars().all()
            vectors = embedder.embed([embed_text(c.text, c.metadata_json) for c in rows])
            for chunk, vec in zip(rows, vectors):
                chunk.embedding = vec
            # Commit per batch: sessions run with autoflush=False, so without
            # this every update piles up until ONE giant executemany at the
            # end (~700MB for 150k rows) -- which gets the connection killed
            # by Postgres and rolls back ALL embeddings.
            db.commit()
            done += len(rows)
            print(f"  embedded {done}/{len(ids)} chunks", flush=True)
        return done
