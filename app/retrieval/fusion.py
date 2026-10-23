"""Weighted Reciprocal Rank Fusion (RRF) of lexical + vector rankings.

RRF is used (rather than e.g. averaging raw scores) because BM25 scores and
cosine similarities live on incomparable scales -- RRF only needs each
list's *rank order*, which sidesteps that problem entirely.
"""
from __future__ import annotations
from dataclasses import dataclass

from app.config import RRF_K, LEXICAL_WEIGHT, VECTOR_WEIGHT
from app.retrieval.lexical import LexicalHit
from app.retrieval.vector import VectorHit


@dataclass
class FusedHit:
    chunk_id: str
    fused_score: float
    lexical_rank: int | None
    vector_rank: int | None


def fuse(lexical_hits: list[LexicalHit], vector_hits: list[VectorHit], top_k: int) -> list[FusedHit]:
    lex_rank = {h.chunk_id: i for i, h in enumerate(lexical_hits)}
    vec_rank = {h.chunk_id: i for i, h in enumerate(vector_hits)}

    all_ids = set(lex_rank) | set(vec_rank)
    fused: list[FusedHit] = []
    for cid in all_ids:
        score = 0.0
        lr = lex_rank.get(cid)
        vr = vec_rank.get(cid)
        if lr is not None:
            score += LEXICAL_WEIGHT * (1.0 / (RRF_K + lr + 1))
        if vr is not None:
            score += VECTOR_WEIGHT * (1.0 / (RRF_K + vr + 1))
        fused.append(FusedHit(chunk_id=cid, fused_score=score, lexical_rank=lr, vector_rank=vr))

    fused.sort(key=lambda h: h.fused_score, reverse=True)
    return fused[:top_k]
