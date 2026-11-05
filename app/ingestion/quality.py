# Quality flags keep suspicious OCR-like numeric text verbatim while marking it as risky evidence.
from __future__ import annotations
import hashlib
import re

# OCR can flip digits into lookalike letters in numeric contexts, e.g. l0%, 1O,000, or S1,200.
_OCR_SUSPECT_PATTERNS = [
    re.compile(r"\bl\d[\d,.]*%"),        # l0%: l instead of 1.
    re.compile(r"\b\d*[Oo]\d[\d,.]*\b"), # 1O,000: O instead of 0.
    re.compile(r"\bS\d[\d,]*\b"),        # S1,200: S instead of $.
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
