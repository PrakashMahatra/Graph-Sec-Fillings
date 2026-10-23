import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import (
    chunk_section_text, chunk_table_markdown, count_tokens, fit_context,
    TARGET_TOKENS, HARD_CAP, CONTEXT_CAP,
)
from app.ingestion.quality import sha256_text, detect_quality_flags


def test_chunking_is_deterministic():
    text = " ".join(f"word{i}" for i in range(1000))
    c1 = chunk_section_text(text)
    c2 = chunk_section_text(text)
    assert [c.text for c in c1] == [c.text for c in c2]


def test_no_chunk_exceeds_target_by_much():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_section_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= TARGET_TOKENS + 5


def test_chunks_fit_real_wordpiece_budget():
    """The actual truncation bug: chunks must fit MiniLM's 256-token window
    when measured with the REAL tokenizer, not whitespace words."""
    text = " ".join(f"word{i}" for i in range(800))
    for c in chunk_section_text(text):
        assert count_tokens(c.text) <= HARD_CAP


def test_context_plus_chunk_fits_model_window():
    """Contextual header + chunk text + [CLS]/[SEP] must fit in 256."""
    ctx = fit_context(
        "Apple Inc. | 10-K | filed 2024-02-02 | section: "
        "Item 1A. Risk Factors and Uncertainties Overview"
    )
    assert count_tokens(ctx) <= CONTEXT_CAP
    chunk = chunk_section_text(" ".join(f"token{i}" for i in range(500)))[0]
    assert count_tokens(ctx) + count_tokens(chunk.text) + 2 <= 256


def test_oversized_table_split_with_header_repeated():
    rows = "\n".join(f"| 20{ i % 10 } | ${i},000 | {i % 100}% |" for i in range(300))
    table = "| Year | Revenue | Growth |\n|---|---|---|\n" + rows
    chunks = chunk_table_markdown(table)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= HARD_CAP
        assert c.text.startswith("| Year | Revenue | Growth |")
    joined = "\n".join(c.text for c in chunks)
    for i in range(0, 300, 37):
        assert f"| ${i},000 |" in joined


def test_small_table_stays_whole():
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert len(chunk_table_markdown(table)) == 1


def test_chunking_never_infinite_loops_on_tiny_text():
    chunks = chunk_section_text("just a few words here")
    assert len(chunks) == 1


def test_hash_is_stable():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")


def test_ocr_suspect_flagged_but_not_altered():
    text = "Revenue grew by l0% year over year."
    flags = detect_quality_flags(text)
    assert any(f["type"] == "ocr_numeric_suspect" for f in flags)
    # never rewritten
    assert "l0%" in text
