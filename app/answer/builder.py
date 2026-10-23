"""
Evidence-first answer assembly.

The deterministic citation-backed answer is built first and is ALWAYS
returned as-is, whether or not a local LLM is configured. If `USE_OLLAMA`
is on, a local model is given that same evidence and asked only to
smooth the prose -- it cannot introduce a claim that isn't already in the
`facts`/`citations` payload, and if Ollama is unreachable the deterministic
answer is what the caller gets anyway. This is the "system stays useful
with the LLM turned off" requirement from the spec.
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field

import requests
from sqlalchemy import select

from app.models import Chunk, Document, Filing, GraphNode
from app.retrieval.fusion import FusedHit
from app.retrieval.graph_retriever import GraphPath
from app.config import (USE_OLLAMA, OLLAMA_MODEL, OLLAMA_URL,
                        GEMINI_API_KEY, GEMINI_MODEL)

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    filing_accession: str
    company: str
    form_type: str
    filing_date: str
    section: str
    page: int
    snippet: str
    filing_url: str = ""  # deep link to the source document on sec.gov


@dataclass
class AnswerPayload:
    question: str
    summary: str
    graph_facts: list[str]
    citations: list[Citation] = field(default_factory=list)
    generation_mode: str = "deterministic"
    key_points: list[str] = field(default_factory=list)
    citation_ids: list[int] = field(default_factory=list)  # 1-based, refs `citations`


def _citation_for_chunk(db, tenant_id: str, chunk_id: str) -> Citation | None:
    chunk = db.get(Chunk, chunk_id)
    if not chunk or chunk.tenant_id != tenant_id:
        return None
    doc = db.get(Document, chunk.document_id)
    filing = db.get(Filing, doc.filing_id) if doc else None
    if not filing:
        return None
    company_name = filing.company.name if filing.company else "Unknown"
    return Citation(
        filing_accession=filing.accession_number,
        company=company_name,
        form_type=filing.form_type,
        filing_date=filing.filing_date,
        section=(chunk.source_locator or {}).get("heading", "Unknown section"),
        page=chunk.page_start,
        snippet=(chunk.text[:240] + ("…" if len(chunk.text) > 240 else "")),
        filing_url=filing.filing_url or "",
    )


def _describe_path(path: GraphPath) -> str:
    parts = [path.node_names[0]]
    for rel, name in zip(path.relations, path.node_names[1:]):
        parts.append(f"--{rel}-->")
        parts.append(name)
    return " ".join(parts)


def build_answer(db, tenant_id: str, question: str, fused_hits: list[FusedHit],
                  graph_paths: list[GraphPath]) -> AnswerPayload:
    citations = []
    for hit in fused_hits:
        c = _citation_for_chunk(db, tenant_id, hit.chunk_id)
        if c:
            citations.append(c)

    graph_facts = [_describe_path(p) for p in graph_paths]

    summary_lines = []
    if graph_facts:
        summary_lines.append(
            f"Graph traversal found {len(graph_facts)} relevant path(s) connecting the "
            f"entities in your question:"
        )
        summary_lines.extend(f"  • {f}" for f in graph_facts[:8])
    if citations:
        summary_lines.append("")
        summary_lines.append(
            f"Supporting evidence from {len(citations)} filing section(s) "
            f"(see citations for exact filing/section/page):"
        )
        for c in citations[:5]:
            summary_lines.append(
                f"  • [{c.company} {c.form_type} {c.filing_date}, \"{c.section}\", p.{c.page}] "
                f"{c.snippet}"
            )
    if not summary_lines:
        summary_lines.append("No matching evidence was found in the indexed corpus for this tenant.")

    payload = AnswerPayload(
        question=question,
        summary="\n".join(summary_lines),
        graph_facts=graph_facts,
        citations=citations,
    )

    if GEMINI_API_KEY:
        structured = _try_gemini_structured(payload)
        if structured:
            payload.summary = structured["summary"]
            payload.key_points = structured["key_points"]
            payload.citation_ids = structured["citation_ids"]
            payload.generation_mode = f"gemini:{GEMINI_MODEL}"
    if payload.generation_mode == "deterministic" and USE_OLLAMA:
        polished = _try_ollama_polish(payload)
        if polished:
            payload.summary = polished
            payload.generation_mode = "ollama_polished_deterministic_evidence"

    return payload


def _try_gemini_structured(payload: AnswerPayload) -> dict | None:
    """Optional Gemini answer synthesis via plain REST (no SDK).

    Returns {summary, key_points, citation_ids} grounded ONLY in the
    numbered evidence below, or None on ANY failure (bad key, quota,
    network, malformed JSON) -- the deterministic answer then stands, so
    the API can never break because of the LLM.
    """
    evidence = []
    for i, c in enumerate(payload.citations, 1):
        evidence.append(
            f"[{i}] {c.company} {c.form_type} ({c.filing_date}), "
            f"section \"{c.section}\", p.{c.page}: {c.snippet}"
        )
    if payload.graph_facts:
        evidence.append("Graph facts:\n" + "\n".join(f"- {g}" for g in payload.graph_facts))
    if not evidence:
        return None  # nothing grounded to synthesize from
    prompt = (
        "You are a financial analyst. Answer the question using ONLY the "
        "numbered evidence below. Never invent numbers or facts. Cite "
        "evidence inline as [n]. If the evidence is insufficient, say so.\n\n"
        "Evidence:\n" + "\n".join(evidence) +
        f"\n\nQuestion: {payload.question}\n\n"
        'Return JSON exactly: {"summary": "2-4 sentence answer with [n] '
        'citations", "key_points": ["3-6 short bullets"], '
        '"citation_ids": [1-based integers of evidence actually used]}'
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "key_points": {"type": "ARRAY", "items": {"type": "STRING"}},
            "citation_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        },
        "required": ["summary", "key_points", "citation_ids"],
    }
    try:
        resp = None
        for attempt in range(3):
            resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent",
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "responseMimeType": "application/json",
                        "responseSchema": schema,
                    },
                },
                timeout=30,
            )
            if resp.status_code in (429, 500, 503) and attempt < 2:
                # Gemini free tier intermittently returns 503 under demand
                time.sleep(2.0 * (attempt + 1))
                continue
            break
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
        summary = str(data.get("summary") or "").strip()
        if not summary:
            return None
        n = len(payload.citations)
        ids = []
        for i in data.get("citation_ids") or []:
            try:
                v = int(i)
            except (TypeError, ValueError):
                continue
            if 1 <= v <= n and v not in ids:
                ids.append(v)
        return {
            "summary": summary,
            "key_points": [str(k) for k in (data.get("key_points") or [])][:8],
            "citation_ids": ids,
        }
    except Exception as e:
        logger.warning("gemini synthesis failed (%s); using deterministic answer", e)
        return None


def _try_ollama_polish(payload: AnswerPayload) -> str | None:
    """Optional. Local-only, zero-cost. Never trusted as the source of any
    fact -- it's only allowed to rephrase what's already in `payload`."""
    prompt = (
        "Rewrite the following evidence-backed findings as clear prose for "
        "a financial analyst. Do NOT add any fact, number, or claim that is "
        "not already present below. Keep every citation bracket exactly as "
        "given.\n\n" + payload.summary
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("response")
    except Exception:
        return None
