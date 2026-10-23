"""Full corpus rebuild: wipe content tables, re-ingest from .sec_cache with
the CURRENT pipeline (token-aware chunker + contextual headers + fixed EDGAR
parser), re-embed everything, verify the token budget, refresh the HNSW
index.

Run this whenever chunk sizing, embedding input, or the parser changes --
old embeddings were produced under the old rules and are incompatible.

Usage:
    python scripts/rebuild_corpus.py --yes
    python scripts/rebuild_corpus.py --yes --skip-ingest   # verify-only

Keeps tenants/companies/users; wipes filings/documents/sections/chunks/
graph_nodes/graph_edges for ALL tenants (single-tenant demo). SEC traffic
served from .sec_cache where warm; only cache misses hit sec.gov (10 req/s
ceiling enforced by sec_client).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://graphrag:graphrag_dev_only@localhost:5432/graphrag",
)
os.environ.setdefault("EMBEDDER", "minilm")
os.environ.setdefault("EMBED_DIM", "384")

from sqlalchemy import create_engine, text  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402
from app.ingestion.chunker import HARD_CAP  # noqa: E402

WIPE_SQL = """
TRUNCATE TABLE graph_edges, graph_nodes, chunks, sections, documents, filings
RESTART IDENTITY CASCADE
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--tickers-file", default="scripts/tickers_sp100.txt")
    ap.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    ap.add_argument("--limit", type=int, default=5,
                    help="filings per company (5 = original 486-filing corpus)")
    ap.add_argument("--tenant", default="sec")
    ap.add_argument("--start-date", default=None,
                    help="YYYY-MM-DD; only filings on/after this date")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="only verify + reindex existing data")
    ap.add_argument("--sanity-floor", type=int, default=400,
                    help="fail if fewer filings than this end up ingested")
    args = ap.parse_args()

    if not args.yes:
        print("This TRUNCATES filings/documents/sections/chunks/graph tables.")
        print("Re-run with --yes to proceed.")
        return 1

    engine = create_engine(DATABASE_URL)
    t0 = time.time()

    print("[1/4] wiping content tables (tenants/companies kept) ...", flush=True)
    with engine.begin() as conn:
        conn.execute(text(WIPE_SQL))

    if not args.skip_ingest:
        print("[2/4] re-ingest + re-embed via scripts/ingest_sec.py ...", flush=True)
        cmd = [sys.executable, "scripts/ingest_sec.py",
               "--tickers-file", args.tickers_file,
               "--forms", *args.forms,
               "--limit", str(args.limit),
               "--tenant", args.tenant,
               "--delay", "0.05"]
        if args.start_date:
            cmd += ["--start-date", args.start_date]
        if args.end_date:
            cmd += ["--end-date", args.end_date]
        print("      + " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=BASE)
        if proc.returncode != 0:
            print(f"FAILED: ingest exited {proc.returncode}")
            return proc.returncode

    print("[3/4] verifying token budget + embedding coverage ...", flush=True)
    with engine.begin() as conn:
        total, embedded, over, mx, avg = conn.execute(text(
            "SELECT COUNT(*), COUNT(embedding), "
            "COUNT(*) FILTER (WHERE token_count > :cap), "
            "COALESCE(MAX(token_count), 0), COALESCE(ROUND(AVG(token_count)), 0) "
            "FROM chunks"
        ), {"cap": HARD_CAP}).one()
        filings = conn.execute(text("SELECT COUNT(*) FROM filings")).scalar()
        sections = conn.execute(text("SELECT COUNT(*) FROM sections")).scalar()
        distinct_headings = conn.execute(
            text("SELECT COUNT(DISTINCT heading) FROM sections")).scalar()
        contexts = conn.execute(text(
            "SELECT COUNT(*) FROM chunks WHERE metadata_json->>'context' IS NOT NULL"
        )).scalar()
        print(f"      filings={filings} sections={sections} "
              f"distinct_headings={distinct_headings}")
        print(f"      chunks={total} embedded={embedded} with_context={contexts}")
        print(f"      token_count: max={mx} avg={avg} budget={HARD_CAP} "
              f"over_budget={over}")

        problems = []
        if over:
            problems.append(f"{over} chunks exceed the {HARD_CAP}-token budget")
        if total and embedded != total:
            problems.append(f"{total - embedded} chunks missing embeddings")
        if filings < args.sanity_floor:
            problems.append(f"only {filings} filings (< {args.sanity_floor})")
        if problems:
            print("VERIFICATION FAILED: " + "; ".join(problems))
            return 2

        print("[4/4] reindexing HNSW ...", flush=True)
        conn.execute(text("REINDEX INDEX chunks_embedding_idx"))

    print(f"rebuild OK in {time.time() - t0:.0f}s "
          f"({filings} filings, {total} chunks, max {mx} tokens)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())