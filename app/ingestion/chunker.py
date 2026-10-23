"""
Splits filing text into overlapping chunks bounded by the *embedding model's*
real tokenizer (WordPiece, all-MiniLM-L6-v2) -- not whitespace words.

Why: MiniLM silently truncates every input at max_seq_length=256 WordPiece
tokens (incl. [CLS]/[SEP]). The old word-based budget (220 whitespace words
~= 280-320 WordPiece tokens) meant the tail ~10-15% of every chunk never
reached its embedding, and unbounded table chunks (largest: 25,439 words)
were >99% truncated. Counting real tokens makes the budget exact.

Budget math (WordPiece tokens):
    contextual header (company | form | date | section) <= CONTEXT_CAP (48)
  + chunk text                                          <= HARD_CAP   (206)
  + [CLS]/[SEP]                                          = 2
  = 256 (embedder max_seq_length)
"""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass

from app.config import EMBED_TOKENIZER

logger = logging.getLogger(__name__)

TARGET_TOKENS = 200     # normal chunk budget (leaves context headroom)
OVERLAP_TOKENS = 40     # tokens repeated at each boundary
HARD_CAP = 206          # never emit a chunk the embedder would truncate
CONTEXT_CAP = 48        # max tokens for the contextual header prefix

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_TOKENIZER)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Exact WordPiece token count (no special tokens)."""
    return len(_get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def _word_token_counts(words: list[str]) -> list[int]:
    """Per-word WordPiece counts. WordPiece never merges across whitespace,
    so summing these equals tokenizing the joined text -- while letting us
    pick boundaries by binary search instead of re-tokenizing repeatedly."""
    enc = _get_tokenizer()(words, is_split_into_words=True, add_special_tokens=False)
    counts = [0] * len(words)
    for w in enc.word_ids():
        counts[w] += 1
    return [max(1, c) for c in counts]


def fit_context(ctx: str, cap: int = CONTEXT_CAP) -> str:
    """Trim a contextual header to <= cap tokens (drops trailing words)."""
    words = ctx.split()
    if not words:
        return ""
    counts = _word_token_counts(words)
    if sum(counts) <= cap:
        return ctx
    keep, acc = 0, 0
    for i, c in enumerate(counts):
        if acc + c > cap:
            break
        acc += c
        keep = i + 1
    return " ".join(words[:keep])


@dataclass
class ChunkCandidate:
    text: str
    token_count: int
    page_start: int
    page_end: int


def chunk_section_text(text: str, page_start: int = 1, page_end: int = 1) -> list[ChunkCandidate]:
    words = text.split()
    if not words:
        return []
    counts = _word_token_counts(words)
    prefix = [0]
    for c in counts:
        prefix.append(prefix[-1] + c)
    n = len(words)
    chunks: list[ChunkCandidate] = []
    start = 0
    while start < n:
        # widest end whose WordPiece total stays within TARGET_TOKENS
        cap = prefix[start] + TARGET_TOKENS
        end = bisect.bisect_right(prefix, cap, lo=start + 1, hi=n + 1) - 1
        if end <= start:
            end = start + 1
        piece = " ".join(words[start:end])
        tok_count = count_tokens(piece)
        # Defensive: per-word sums == joined counts for WordPiece, so this
        # should never fire -- but never emit an over-budget chunk.
        while tok_count > HARD_CAP and end - start > 1:
            end = start + max(1, (end - start) * 3 // 4)
            piece = " ".join(words[start:end])
            tok_count = count_tokens(piece)
        if tok_count > HARD_CAP:
            # A single pathological "word" (underscore rules, raw URLs,
            # XBRL ids) that can't be shrunk by word count -- unembeddable
            # noise, so drop it rather than silently truncate its embedding.
            logger.warning("dropping unembeddable chunk (%d tokens, %d chars)",
                           tok_count, len(piece))
        else:
            chunks.append(ChunkCandidate(
                text=piece, token_count=tok_count,
                page_start=page_start, page_end=page_end,
            ))
        if end == n:
            break
        # next window starts ~OVERLAP_TOKENS back so a fact split across the
        # boundary isn't lost to either chunk
        back = bisect.bisect_left(prefix, prefix[end] - OVERLAP_TOKENS, lo=start + 1)
        start = back if back > start else start + 1
    return chunks


_SEP_OK = set("-|: \t")


def _is_separator_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and "-" in s and all(ch in _SEP_OK for ch in s)


def chunk_table_markdown(table_md: str) -> list[ChunkCandidate]:
    """Tables stay whole when they fit the budget. Oversized tables are split
    into row groups that repeat the header rows, so every piece is embeddable
    and still interpretable (previously >99% of a large table was truncated
    out of its embedding)."""
    lines = [l for l in table_md.splitlines() if l.strip()]
    if not lines:
        return []
    total = count_tokens(table_md)
    if total <= TARGET_TOKENS:
        return [ChunkCandidate(text=table_md, token_count=total,
                               page_start=1, page_end=1)]

    n_head = 2 if len(lines) > 1 and _is_separator_line(lines[1]) else 1
    header_lines = lines[:n_head]
    rows = lines[n_head:]
    if not rows:
        return chunk_section_text(table_md)  # degenerate oversized header

    # single batched tokenization for header + all rows
    enc = _get_tokenizer()(header_lines + rows, add_special_tokens=False)
    costs = [len(ids) for ids in enc["input_ids"]]
    header_cost = sum(costs[:n_head])
    row_costs = costs[n_head:]

    out: list[ChunkCandidate] = []
    i = 0
    while i < len(rows):
        group: list[str] = []
        cost = header_cost
        while i < len(rows):
            if group and cost + row_costs[i] > TARGET_TOKENS:
                break
            group.append(rows[i])
            cost += row_costs[i]
            i += 1
        text = "\n".join(header_lines + group)
        tc = count_tokens(text)
        if tc > HARD_CAP:
            # pathological single row wider than the budget: word-split it
            logger.warning("table row group exceeded %d tokens; word-splitting", HARD_CAP)
            out.extend(chunk_section_text(text))
        else:
            out.append(ChunkCandidate(text=text, token_count=tc,
                                      page_start=1, page_end=1))
    return out
