"""Re-run embedding for chunks whose embedding is NULL.

Use after a crashed or rolled-back embed transaction (rebuild_embeddings
commits per batch now, but an older run may have lost everything). Safe to
re-run: only NULL-embedding chunks are touched.

    python scripts/embed_missing.py
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://graphrag:graphrag_dev_only@localhost:5432/graphrag",
)

from app.ingestion.ingest import rebuild_embeddings  # noqa: E402

if __name__ == "__main__":
    n = rebuild_embeddings(only_missing=True)
    print(f"EMBEDDED_ALL={n}")
    sys.exit(0 if n >= 0 else 1)