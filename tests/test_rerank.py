"""Unit tests for post-fusion selection: rerank short-circuit + duplicate
suppression. The cross-encoder itself is exercised live in test_eval.py;
here we keep it out (no model download) by patching the module globals."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import rerank as rr
from app.retrieval.fusion import FusedHit


class _FakeDB:
    """Stands in for the SQLAlchemy session: returns pre-built rows for the
    (id, text, embedding) lookup inside select_final."""
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows


def _hit(cid, score):
    return FusedHit(chunk_id=cid, fused_score=score, lexical_rank=0, vector_rank=0)


def test_rerank_disabled_returns_input_order(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_ENABLED", False)
    hits = [_hit("a", 0.3), _hit("b", 0.2)]
    assert rr.rerank("q", hits, {}) == hits


def test_select_final_drops_near_duplicates(monkeypatch):
    monkeypatch.setattr(rr, "rerank", lambda q, hits, texts: hits)  # no model
    rows = [
        ("a", "first chunk", np.array([1.0, 0.0], dtype=np.float32)),
        ("b", "near-duplicate of a", np.array([0.999, 0.044], dtype=np.float32)),
        ("c", "different content", np.array([0.0, 1.0], dtype=np.float32)),
    ]
    db = _FakeDB(rows)
    pool = [_hit("a", 0.3), _hit("b", 0.2), _hit("c", 0.1)]
    picked = rr.select_final(db, "tenant", "q", pool, top_k=2)
    assert [h.chunk_id for h in picked] == ["a", "c"]  # b suppressed as dup


def test_select_final_empty_pool(monkeypatch):
    monkeypatch.setattr(rr, "rerank", lambda q, hits, texts: hits)
    assert rr.select_final(_FakeDB([]), "tenant", "q", [], top_k=8) == []