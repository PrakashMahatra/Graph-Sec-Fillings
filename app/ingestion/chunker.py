# Chunk filing text using the embedding model's real WordPiece budget.
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
    """Per-word WordPiece counts let us split on token boundaries without re-tokenizing the whole string."""
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
        # Use the widest boundary that keeps the chunk under TARGET_TOKENS.
        cap = prefix[start] + TARGET_TOKENS
        end = bisect.bisect_right(prefix, cap, lo=start + 1, hi=n + 1) - 1
        if end <= start:
            end = start + 1
        piece = " ".join(words[start:end])
        tok_count = count_tokens(piece)
        # Defensive fallback: if a token count still exceeds the cap, trim until it fits.
        while tok_count > HARD_CAP and end - start > 1:
            end = start + max(1, (end - start) * 3 // 4)
            piece = " ".join(words[start:end])
            tok_count = count_tokens(piece)
        if tok_count > HARD_CAP:
            # Drop pathological single-word chunks that cannot be shrunk without truncating the embedding.
            logger.warning("dropping unembeddable chunk (%d tokens, %d chars)",
                           tok_count, len(piece))
        else:
            chunks.append(ChunkCandidate(
                text=piece, token_count=tok_count,
                page_start=page_start, page_end=page_end,
            ))
        if end == n:
            break
        # Start the next chunk a bit back so a fact split at the boundary is not lost.
        back = bisect.bisect_left(prefix, prefix[end] - OVERLAP_TOKENS, lo=start + 1)
        start = back if back > start else start + 1
    return chunks


_SEP_OK = set("-|: \t")


def _is_separator_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and "-" in s and all(ch in _SEP_OK for ch in s)


def chunk_table_markdown(table_md: str) -> list[ChunkCandidate]:
    """Tables stay whole when they fit the budget; oversized tables are split into row groups that repeat the header."""
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

    # Batch-tokenize the header and row groups together.
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
            # Pathological rows are word-split to keep the chunk embeddable.
            logger.warning("table row group exceeded %d tokens; word-splitting", HARD_CAP)
            out.extend(chunk_section_text(text))
        else:
            out.append(ChunkCandidate(text=text, token_count=tc,
                                      page_start=1, page_end=1))
    return out
