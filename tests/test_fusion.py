import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.fusion import fuse
from app.retrieval.lexical import LexicalHit
from app.retrieval.vector import VectorHit


def test_fuse_ranks_items_in_both_lists_highest():
    lex = [LexicalHit("c1", 5.0), LexicalHit("c2", 3.0), LexicalHit("c3", 1.0)]
    vec = [VectorHit("c2", 0.9), VectorHit("c1", 0.8), VectorHit("c4", 0.5)]
    fused = fuse(lex, vec, top_k=10)
    ids = [f.chunk_id for f in fused]
    # c1 and c2 appear in both lists near the top -> should outrank c3/c4
    assert ids.index("c1") < ids.index("c3")
    assert ids.index("c2") < ids.index("c4")


def test_fuse_handles_disjoint_lists():
    lex = [LexicalHit("only_lex", 2.0)]
    vec = [VectorHit("only_vec", 0.7)]
    fused = fuse(lex, vec, top_k=10)
    ids = {f.chunk_id for f in fused}
    assert ids == {"only_lex", "only_vec"}


def test_fuse_respects_top_k():
    lex = [LexicalHit(f"c{i}", float(10 - i)) for i in range(10)]
    fused = fuse(lex, [], top_k=3)
    assert len(fused) == 3
