import sys, json, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import select, func
from app.db import init_db, session_scope
from app.models import Filing, Chunk, GraphNode, GraphEdge

if len(sys.argv) > 1 and sys.argv[1] == "query":
    q = sys.argv[2] if len(sys.argv) > 2 else "What are Apple risk factors?"
    req = json.dumps({"tenant_slug": "sec", "question": q}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://localhost:8000/query", data=req,
        headers={"Content-Type": "application/json"}), timeout=120)
    d = json.loads(r.read())
    print("ANSWER:", (d.get("answer") or "")[:800])
    print("CITATIONS:", len(d.get("citations", [])))
    for c in d.get("citations", [])[:5]:
        print(" -", c.get("form_type"), c.get("filing_date"), "|", (c.get("section_heading") or "")[:60], "| p.", c.get("page_start"))
    sys.exit(0)

init_db()
with session_scope() as db:
    print("filings:", db.execute(select(func.count()).select_from(Filing)).scalar())
    print("chunks:", db.execute(select(func.count()).select_from(Chunk)).scalar())
    print("chunks w/ embedding:", db.execute(select(func.count()).select_from(Chunk).where(Chunk.embedding.isnot(None))).scalar())
    print("nodes:", db.execute(select(func.count()).select_from(GraphNode)).scalar())
    print("edges:", db.execute(select(func.count()).select_from(GraphEdge)).scalar())
    c = db.execute(select(Chunk).limit(1)).scalar_one()
    emb = c.embedding
    print("sample chunk len:", len(c.text), "| tokens:", c.token_count, "| emb dim:", len(emb) if emb else None)
    print("sample text head:", c.text[:200].replace("\n", " "))
