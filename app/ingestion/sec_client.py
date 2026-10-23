"""
Thin client for SEC's public, key-free APIs (data.sec.gov / www.sec.gov).

SEC documents a 10 requests/second fair-access ceiling and requires a
declared User-Agent for scripted access -- both are enforced here.
Responses are cached to disk so re-running ingestion doesn't re-hit SEC and
ingestion stays idempotent.

This module needs outbound internet to sec.gov, which the sandbox this
repo may have been generated in does not have -- run it on your own
machine (`python scripts/ingest_sec.py --cik <CIK>`) to pull real filings.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import requests

from app.config import BASE_DIR, SEC_USER_AGENT, SEC_MAX_REQUESTS_PER_SECOND

CACHE_DIR = BASE_DIR / ".sec_cache"
CACHE_DIR.mkdir(exist_ok=True)

_MIN_INTERVAL = 1.0 / SEC_MAX_REQUESTS_PER_SECOND
_last_request_ts = 0.0


def _throttle():
    global _last_request_ts
    elapsed = time.monotonic() - _last_request_ts
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_ts = time.monotonic()


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def get_json(url: str, force: bool = False) -> dict:
    cache_file = _cache_path(url)
    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text())

    _throttle()
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data))
    return data


def get_text(url: str, force: bool = False) -> str:
    cache_file = _cache_path(url + "::text")
    if cache_file.exists() and not force:
        return cache_file.read_text()

    _throttle()
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    resp.raise_for_status()
    text = resp.text
    cache_file.write_text(text)
    return text


def resolve_cik(ticker_or_cik: str) -> str:
    """Resolve a ticker to a zero-padded 10-digit CIK via SEC's ticker map."""
    if ticker_or_cik.isdigit():
        return ticker_or_cik.zfill(10)
    mapping = get_json("https://www.sec.gov/files/company_tickers.json")
    for _, row in mapping.items():
        if row.get("ticker", "").upper() == ticker_or_cik.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Could not resolve ticker '{ticker_or_cik}' to a CIK")


def get_submissions(cik: str) -> dict:
    cik = cik.zfill(10)
    return get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")


def get_companyfacts(cik: str) -> dict:
    """XBRL structured facts (us-gaap tags with values per period). Payloads
    are large (MBs) but cached like everything else -- first call per CIK
    hits sec.gov, afterwards offline."""
    cik = cik.zfill(10)
    return get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")


def _collect_from_block(cik: str, block: dict, company_name, ticker, form_types, start_date, end_date) -> list[dict]:
    out = []
    forms = block.get("form", [])
    n = len(forms)
    accs = block.get("accessionNumber", [])
    dates = block.get("filingDate", [])
    reports = block.get("reportDate", [])
    docs = block.get("primaryDocument", [])
    for i in range(n):
        form = forms[i] if i < len(forms) else None
        if form not in form_types:
            continue
        filing_date = dates[i] if i < len(dates) else None
        if start_date and filing_date and filing_date < start_date:
            continue
        if end_date and filing_date and filing_date > end_date:
            continue
        out.append({
            "cik": cik,
            "form_type": form,
            "accession_number": accs[i] if i < len(accs) else "",
            "filing_date": filing_date,
            "period_end_date": reports[i] if i < len(reports) else None,
            "primary_document": docs[i] if i < len(docs) else "",
            "company_name": company_name,
            "ticker": ticker,
        })
    return out


def list_all_filings(
    cik: str,
    form_types: tuple[str, ...] = ("10-K", "10-Q", "8-K", "S-1", "DEF 14A"),
    limit: int = 100,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Bulk-aware listing: walks `filings.recent` + paginated `filings.files[]`
    archives (oldest submissions) so '--limit 500' actually returns 500.
    Dates are 'YYYY-MM-DD' strings; comparison is lexicographic (safe ISO format).
    Returns newest-first, capped at `limit`."""
    cik = cik.zfill(10)
    sub = get_submissions(cik)
    company_name = sub.get("name")
    ticker = (sub.get("tickers") or [None])[0]
    filings = sub.get("filings", {})
    out: list[dict] = []
    recent = filings.get("recent", {})
    if recent:
        out.extend(_collect_from_block(cik, recent, company_name, ticker, set(form_types), start_date, end_date))
        if len(out) >= limit:
            return out[:limit]
    # Paginated archives, e.g. CIK0000320193-submissions-001.json
    for f in filings.get("files", []):
        if len(out) >= limit:
            break
        name = f.get("name", "")
        # name already like "CIK0000320193-submissions-001.json"
        url = f"https://data.sec.gov/submissions/{name}"
        try:
            archived = get_json(url)
        except Exception:
            continue
        # archived files have varying shapes: either the same `recent`-style
        # arrays at top level or nested under filings. Handle both.
        block = archived
        if "filings" in archived and isinstance(archived["filings"], dict):
            # some archives nest under filings.recent
            block = archived["filings"].get("recent", archived)
        out.extend(_collect_from_block(cik, block, company_name, ticker, set(form_types), start_date, end_date))
    return out[:limit]


def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    acc_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary_document}"
