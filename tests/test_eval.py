"""Retrieval-quality regression test over the golden question set.

Runs the full in-process pipeline (planner -> hybrid retrieval -> rerank ->
answer builder) against the live Postgres corpus and asserts hit-rate/MRR
floors. Thresholds are env-tunable so the same suite can report a baseline
(EVAL_MIN_HIT=0 EVAL_MIN_MRR=0) or enforce the production floor.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://graphrag:graphrag_dev_only@localhost:5432/graphrag",
)

from eval_rag import evaluate  # noqa: E402

MIN_HIT = float(os.environ.get("EVAL_MIN_HIT", "0.80"))
MIN_MRR = float(os.environ.get("EVAL_MIN_MRR", "0.70"))


def test_retrieval_quality():
    from fastapi.testclient import TestClient
    from app.main import app

    def query_fn(question: str) -> dict:
        with TestClient(app) as client:  # fresh client per query: no state sharing
            r = client.post("/query", json={"tenant_slug": "sec", "question": question})
            assert r.status_code == 200, r.text
            return r.json()

    m = evaluate(query_fn, verbose=True)
    print("\nEVAL METRICS:", {k: v for k, v in m.items() if k != "details"})
    assert m["company_hit@8"] >= MIN_HIT, (
        f"company_hit@8={m['company_hit@8']} < {MIN_HIT}")
    assert m["mrr"] >= MIN_MRR, f"mrr={m['mrr']} < {MIN_MRR}"
