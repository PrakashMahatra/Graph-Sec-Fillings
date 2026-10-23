import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv():
    """Load .env (KEY=VALUE lines) into os.environ without overwriting real
    env vars and without any dependency. Lets `python scripts/...` and
    `uvicorn` pick up DATABASE_URL/EMBEDDER/SEC_USER_AGENT from .env."""
    p = BASE_DIR / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

# Postgres-only. DATABASE_URL must be set, e.g.:
# DATABASE_URL=postgresql+psycopg2://graphrag:graphrag_dev_only@localhost:5432/graphrag
# (docker compose provides this automatically for the api service).
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL.startswith("postgres"):
    raise RuntimeError(
        "DATABASE_URL must be a Postgres URL, e.g. "
        "postgresql+psycopg2://graphrag:graphrag_dev_only@localhost:5432/graphrag"
    )

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "FilingLens demo research@example.com"
)
SEC_MAX_REQUESTS_PER_SECOND = 10  # SEC's documented fair-access ceiling

# Retrieval tuning
RRF_K = 60  # standard RRF damping constant
LEXICAL_WEIGHT = float(os.environ.get("LEXICAL_WEIGHT", 1.0))
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", 1.0))
TOP_K_CANDIDATES = 40
TOP_K_RETURNED = 8

# Embedding token budget -- shared by the chunker (sizing) and the embedder.
EMBED_TOKENIZER = os.environ.get("EMBED_TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")

# Cross-encoder reranker over the fused candidate pool (ON by default).
# If the model can't load (offline/disk), retrieval falls back to RRF order.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_POOL = int(os.environ.get("RERANK_POOL", str(TOP_K_CANDIDATES)))

# Embeddings: local MiniLM-L6-v2, 384-d, CPU, no API key.
EMBEDDER = os.environ.get("EMBEDDER", "minilm")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))  # must match chunks.embedding vector(N)

# Graph traversal
MAX_GRAPH_HOPS = 3

# Optional local generation (OFF by default -- system works without it)
USE_OLLAMA = os.environ.get("USE_OLLAMA", "false").lower() == "true"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

# Optional structured answer synthesis via Gemini (plain REST, no SDK).
# Leave the key empty and the deterministic evidence builder answers instead.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
