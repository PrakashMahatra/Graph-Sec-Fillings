"""
Quality flags for ingested text.

Rule: never silently "fix" a number. `l0%` (lowercase L, likely OCR
mis-read of `10%`) stays `l0%` verbatim in `chunks.text`; we only ever
*flag* it in `quality_flags` so a human/consumer can decide, and we treat
any chunk carrying a numeric flag as high-risk evidence for financial
claims.
"""
from __future__ import annotations
import hashlib
import re

# Patterns that look like a digit got OCR'd into a lookalike letter next to
# other digits/%/$ context -- e.g. "l0%", "1O,000", "S1,200" for "$1,200".
_OCR_SUSPECT_PATTERNS = [
    re.compile(r"\bl\d[\d,.]*%"),        # l0%  (l instead of 1)
    re.compile(r"\b\d*[Oo]\d[\d,.]*\b"), # 1O,000 (O instead of 0)
    re.compile(r"\bS\d[\d,]*\b"),        # S1,200 (S instead of $)
]

_MOJIBAKE_PATTERN = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_quality_flags(text: str) -> list[dict]:
    flags: list[dict] = []

    for pattern in _OCR_SUSPECT_PATTERNS:
        for m in pattern.finditer(text):
            flags.append({
                "type": "ocr_numeric_suspect",
                "span": m.group(0),
                "start": m.start(),
                "end": m.end(),
                "severity": "high",
                "note": "Possible OCR misread near a numeric/currency value; "
                        "kept verbatim, treat as high-risk evidence.",
            })

    if _MOJIBAKE_PATTERN.search(text):
        flags.append({
            "type": "encoding_artifact",
            "severity": "medium",
            "note": "Control/replacement characters detected; source encoding "
                    "may be corrupted.",
        })

    if len(text.strip()) < 20:
        flags.append({
            "type": "low_content",
            "severity": "low",
            "note": "Very short chunk; may be a parsing artifact rather than "
                    "meaningful content.",
        })

    return flags
