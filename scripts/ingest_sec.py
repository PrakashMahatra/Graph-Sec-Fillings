"""
Real bulk ingestion from SEC EDGAR (free, key-free).
Needs outbound internet to data.sec.gov / www.sec.gov.

Usage (lots of data):
    python scripts/ingest_sec.py --ticker AAPL,MSFT,NVDA --forms 10-K 10-Q 8-K --limit 50
    python scripts/ingest_sec.py --cik 0000320193 --forms 10-K 10-Q 8-K S-1 DEF 14A --limit 100
    python scripts/ingest_sec.py --tickers-file tickers.txt --forms 10-K --limit 20 --tenant sec
    python scripts/ingest_sec.py --ticker AAPL --forms 10-K --limit 50 --start-date 2020-01-01
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import init_db
from app.tenancy import get_or_create_tenant
from app.ingestion import sec_client
from app.ingestion.ingest import ingest_filing, rebuild_embeddings


def parse_targets(args) -> list[str]:
    targets: list[str] = []
    if args.cik:
        targets.append(args.cik)
    if args.ticker:
        targets.extend([t.strip() for t in args.ticker.split(",") if t.strip()])
    if args.tickers_file:
        p = Path(args.tickers_file)
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                targets.append(line.split()[0])  # allow "AAPL # comment" lines
    # de-dup preserving order
    seen, out = set(), []
    for t in targets:
        k = t.upper()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description="Bulk SEC EDGAR ingestion (free, no API key).")
    ap.add_argument("--cik", help="10-digit CIK, e.g. 0000320193")
    ap.add_argument("--ticker", help="One ticker or comma-separated list, e.g. AAPL or AAPL,MSFT,NVDA")
    ap.add_argument("--tickers-file", help="Text file with one ticker/CIK per line for bulk loads")
    ap.add_argument("--tenant", default="sec")
    ap.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"],
                    help="Form types, e.g. 10-K 10-Q 8-K S-1 DEF 14A")
    ap.add_argument("--limit", type=int, default=50,
                    help="max filings to ingest PER COMPANY (use high values for lots of data)")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--delay", type=float, default=0.15,
                    help="seconds between filing downloads (SEC ceiling is 10 req/s)")
    args = ap.parse_args()

    targets = parse_targets(args)
    if not targets:
        ap.error("provide --cik and/or --ticker and/or --tickers-file")

    init_db()
    tenant_id = get_or_create_tenant(args.tenant, args.tenant)
    print(f"Tenant '{args.tenant}' ({tenant_id}) | forms={args.forms} | limit/company={args.limit}")

    total_ok, total_skip, total_fail = 0, 0, 0
    for target in targets:
        try:
            cik = sec_client.resolve_cik(target)
        except Exception as e:
            print(f"[{target}] SKIP: cannot resolve to CIK: {e}")
            total_fail += 1
            continue
        try:
            filings = sec_client.list_all_filings(
                cik, tuple(args.forms), limit=args.limit,
                start_date=args.start_date, end_date=args.end_date,
            )
        except Exception as e:
            print(f"[{target} -> {cik}] SKIP: submissions fetch failed: {e}")
            total_fail += 1
            continue
        if not filings:
            print(f"[{target} -> {cik}] no filings of type {args.forms} found")
            continue
        print(f"[{target} -> {cik}] {len(filings)} filing(s) to ingest")
        for f in filings:
            url = sec_client.filing_document_url(cik, f["accession_number"], f["primary_document"])
            try:
                raw_html = sec_client.get_text(url)
            except Exception as e:
                print(f"  FAIL {f['form_type']} {f['filing_date']} {f['accession_number']}: download: {e}")
                total_fail += 1
                continue
            meta = {
                "company_name": f["company_name"],
                "cik": cik,
                "ticker": f.get("ticker"),
                "form_type": f["form_type"],
                "accession_number": f["accession_number"],
                "filing_date": f["filing_date"],
                "period_end_date": f.get("period_end_date"),
                "filing_url": url,
            }
            try:
                # skip_embedding_rebuild=True -> one rebuild at the end (fast for bulk)
                filing_id = ingest_filing(tenant_id, meta, raw_html, is_html=True,
                                          skip_embedding_rebuild=True)
                print(f"  OK {f['form_type']} {f['filing_date']} {f['accession_number']} -> {filing_id}")
                total_ok += 1
            except Exception as e:
                # idempotent re-ingest returns existing id (not an error); real errors land here
                if "already" in str(e).lower():
                    total_skip += 1
                else:
                    print(f"  FAIL {f['form_type']} {f['filing_date']}: {e}")
                    total_fail += 1
            time.sleep(args.delay)

    if total_ok:
        print(f"\nEmbedding {total_ok} new filing(s) (only chunks with NULL embedding)...")
        n_emb = rebuild_embeddings(only_missing=True)
        print(f"Embedded {n_emb} chunk(s).")
    print(f"\nDone. ok={total_ok} failed={total_fail} (idempotent re-runs are no-ops).")
    print("Query with tenant_slug='%s', e.g.:" % args.tenant)
    print('  uvicorn app.main:app --reload')


if __name__ == "__main__":
    main()
