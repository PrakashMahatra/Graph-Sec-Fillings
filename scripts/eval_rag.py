"""Retrieval-quality evaluation over a golden question set.

Corpus: the 16-company lineup (S&P100 top-10 + AMD/INTC/PLTR/ORCL/MU +
S&P Global) ingested from scripts/tickers_top16.txt -- see GOLDEN below.
Each entry = question, companies whose filings SHOULD be cited (ground
truth), and lowercase keyword stems that must appear in the evidence
(section heading, citation snippet, or answer text).

Metrics:
  company_hit@8 -- fraction of questions where a citation from an expected
                   company (with a matching keyword) appears in the top-8
  mrr           -- mean reciprocal rank of that first correct citation
  avg_citations -- mean citations returned (coverage sanity check)

Two ways to run:
  1. Standalone against a running server:
       python scripts/eval_rag.py --base-url http://127.0.0.1:8000
  2. In-process via tests/test_eval.py (pytest, asserts thresholds).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLDEN: list[dict] = [
    {"q": "What voting rights do Class B shares have?",
     "companies": ["Alphabet Inc.", "Meta Platforms, Inc.", "BERKSHIRE HATHAWAY INC"],
     "keywords": ["class b", "votes", "voting"]},
    {"q": "How many votes does a Class B share carry at Meta?",
     "companies": ["Meta Platforms, Inc."],
     "keywords": ["class b", "vote"]},
    {"q": "What are the risk factors disclosed by Apple?",
     "companies": ["Apple Inc."],
     "keywords": ["risk factor"]},
    {"q": "How did Microsoft's cloud revenue grow year over year?",
     "companies": ["MICROSOFT CORP"],
     "keywords": ["cloud", "azure", "revenue"]},
    {"q": "Who is the chief executive officer of JPMorgan Chase?",
     "companies": ["JPMORGAN CHASE & CO"],
     "keywords": ["jamie dimon", "chief executive", "ceo"]},
    {"q": "Describe NVIDIA's data center segment revenue.",
     "companies": ["NVIDIA CORP"],
     "keywords": ["data center"]},
    {"q": "What is Amazon's operating income?",
     "companies": ["AMAZON COM INC"],
     "keywords": ["operating income"]},
    {"q": "How much cash did Apple return to shareholders through repurchases?",
     "companies": ["Apple Inc."],
     "keywords": ["repurchas", "buyback"]},
    {"q": "What insurance underwriting results did Berkshire Hathaway report?",
     "companies": ["BERKSHIRE HATHAWAY INC"],
     "keywords": ["insurance", "underwrit"]},
    {"q": "What stock split did NVIDIA implement?",
     "companies": ["NVIDIA CORP"],
     "keywords": ["split"]},
    {"q": "What share repurchases has JPMorgan announced?",
     "companies": ["JPMORGAN CHASE & CO"],
     "keywords": ["repurchas", "buyback"]},
    {"q": "How does Alphabet describe its Google Services segment?",
     "companies": ["Alphabet Inc."],
     "keywords": ["google services"]},
    {"q": "What legal proceedings and litigation are discussed in Meta's filings?",
     "companies": ["Meta Platforms, Inc."],
     "keywords": ["legal proceeding", "litigation", "lawsuit"]},
    {"q": "What does AMD say about data center GPU revenue?",
     "companies": ["ADVANCED MICRO DEVICES INC"],
     "keywords": ["data center"]},
    {"q": "What foundry strategy does Intel describe?",
     "companies": ["INTEL CORP"],
     "keywords": ["foundry"]},
    {"q": "How does Palantir describe its AI platform?",
     "companies": ["Palantir Technologies Inc."],
     "keywords": ["artificial intelligence", "aip"]},
    {"q": "What cloud infrastructure does Oracle offer?",
     "companies": ["ORACLE CORP"],
     "keywords": ["cloud"]},
    {"q": "What high-bandwidth memory does Micron produce?",
     "companies": ["MICRON TECHNOLOGY INC"],
     "keywords": ["hbm", "high bandwidth memory"]},
    {"q": "How does S&P Global describe its Ratings segment?",
     "companies": ["S&P Global Inc."],
     "keywords": ["ratings"]},
    {"q": "What does Eli Lilly say about Mounjaro?",
     "companies": ["ELI LILLY & Co"],
     "keywords": ["mounjaro", "tirzepatide"]},
]


def evaluate(query_fn, golden: list[dict] | None = None, verbose: bool = True) -> dict:
    """query_fn(question:str) -> /query response dict."""
    golden = golden or GOLDEN
    details, hits, rrs = [], 0, 0.0
    for g in golden:
        resp = query_fn(g["q"])
        citations = resp.get("citations", []) or []
        answer_text = (resp.get("answer") or "").casefold()
        rank = None
        for i, c in enumerate(citations):
            company_ok = (c.get("company") or "").strip().casefold() in {
                x.casefold() for x in g["companies"]}
            if not company_ok:
                continue
            evidence = " ".join([c.get("section") or "", c.get("snippet") or ""]).casefold()
            if any(k in evidence or k in answer_text for k in g["keywords"]):
                rank = i + 1
                break
        ok = rank is not None and rank <= 8
        hits += int(ok)
        rrs += (1.0 / rank) if ok else 0.0
        details.append({"q": g["q"], "rank": rank, "ok": ok,
                        "n_citations": len(citations),
                        "companies_cited": [c.get("company") for c in citations]})
        if verbose:
            mark = "HIT " if ok else "MISS"
            print(f"  [{mark}] rank={rank!s:>4}  {g['q']}", flush=True)
    n = len(golden)
    return {
        "n": n,
        "company_hit@8": round(hits / n, 4),
        "mrr": round(rrs / n, 4),
        "avg_citations": round(sum(d["n_citations"] for d in details) / n, 2),
        "details": details,
    }


def _http_query(base_url: str, tenant: str):
    import requests

    def q(question: str) -> dict:
        r = requests.post(f"{base_url.rstrip('/')}/query",
                          json={"tenant_slug": tenant, "question": question},
                          timeout=180)
        r.raise_for_status()
        return r.json()
    return q


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tenant", default="sec")
    ap.add_argument("--json-out", default=None, help="write full metrics JSON here")
    args = ap.parse_args()

    print(f"Evaluating {len(GOLDEN)} golden questions against {args.base_url} ...")
    m = evaluate(_http_query(args.base_url, args.tenant))
    print(json.dumps({k: v for k, v in m.items() if k != "details"}, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(m, indent=2))
    floor = float(os.environ.get("EVAL_MIN_HIT", "0"))
    return 0 if m["company_hit@8"] >= floor else 1


if __name__ == "__main__":
    raise SystemExit(main())