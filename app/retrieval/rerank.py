"""Post-fusion candidate selection: cross-encoder reranking + diversity.

RRF only sees rank *order* from two weak scorers (ts_rank and cosine). A
cross-encoder reads each (query, chunk) pair jointly and re-scores it -- the
standard quality step for hybrid RAG. `ms-marco-MiniLM-L-6-v2` (~80MB) ships
inside the existing sentence-transformers dependency, runs on GPU/CPU, and
loads lazily; if it can't load (offline/disk) retrieval keeps RRF order.

After reranking, `select_final` greedily drops near-duplicate candidates
(cosine >= MAX_SIM to an already-picked chunk) so sibling chunks overlapping
by 40 tokens don't burn multiple top-k slots on the same content.
"""
from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select

from app.config import RERANK_ENABLED, RERANK_MODEL
from app.models import Chunk

logger = logging.getLogger(__name__)

MAX_SIM = 0.92  # cosine similarity above which a candidate counts as a dup

_ce = None
_ce_failed = False


def _get_cross_encoder():
    global _ce, _ce_failed
    if _ce is None and not _ce_failed:
        try:
            from sentence_transformers import CrossEncoder
            _ce = CrossEncoder(RERANK_MODEL)
            logger.info("reranker loaded: %s", RERANK_MODEL)
        except Exception as e:
            _ce_failed = True
            logger.warning("reranker unavailable (%s); keeping RRF order", e)
    return _ce


def warm_up() -> None:
    """Load the cross-encoder in the background at startup so the first query
    doesn't pay the model load/download cost."""
    if RERANK_ENABLED:
        _get_cross_encoder()


def rerank(query: str, hits: list, texts: dict[str, str]) -> list:
    """Reorder hits by cross-encoder relevance (input order breaks ties).
    Returns input order unchanged if disabled/unavailable/predict fails."""
    if not RERANK_ENABLED or len(hits) < 2:
        return hits
    ce = _get_cross_encoder()
    if ce is None:
        return hits
    try:
        scores = ce.predict([(query, texts.get(h.chunk_id, "")) for h in hits])
    except Exception as e:
        logger.warning("rerank predict failed (%s); keeping RRF order", e)
        return hits
    order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
    return [hits[i] for i in order]


def select_final(db, tenant_id: str, query: str, pool: list, top_k: int) -> list:
    """Final top-k from the fused pool: fetch chunk texts/embeddings in one
    query, rerank by cross-encoder, then greedily pick while staying under
    MAX_SIM against already-picked chunks. Returns FusedHit list."""
    if not pool:
        return []
    rows = db.execute(
        select(Chunk.id, Chunk.text, Chunk.embedding)
        .where(Chunk.tenant_id == tenant_id, Chunk.id.in_([h.chunk_id for h in pool]))
    ).all()
    texts = {r[0]: r[1] for r in rows}
    vecs = {r[0]: np.asarray(r[2], dtype=np.float32) for r in rows if r[2] is not None}

    ranked = rerank(query, pool, texts)

    picked, picked_vecs = [], []
    for h in ranked:
        v = vecs.get(h.chunk_id)
        if v is not None and picked_vecs:
            if max(float(v @ u) for u in picked_vecs) >= MAX_SIM:
                continue  # near-duplicate of an already-picked chunk
        picked.append(h)
        if v is not None:
            picked_vecs.append(v)
        if len(picked) >= top_k:
            break
    return picked