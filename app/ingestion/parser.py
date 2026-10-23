"""
Turns a raw filing document (HTML or already-plain text) into an ordered
list of sections, each with heading + page-ish locators + body text, and
converts any HTML tables into *both* a Markdown rendering and a structured
row/column form -- the spec requires both when comparisons need them (e.g.
diffing a voting-rights table across filings).

Kept dependency-free (stdlib `html.parser` + `re`) rather than pulling in
BeautifulSoup, since the transform needed here (strip tags, keep structure
markers) is simple enough to be worth keeping the dependency list short.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr"}

# Content that must never reach chunk text: scripts/styles, <head> metadata,
# and hidden iXBRL header blocks -- otherwise every filing's first chunks are
# seeded with raw XBRL like "aapl-20250927 false 2025 FY 0000320193 P1Y ...".
_JUNK_TAGS = ("script", "style", "head")


def _is_junk_tag(tag: str) -> bool:
    t = tag.lower()
    return t in _JUNK_TAGS or t.startswith("ix:hidden") or t.startswith("ix:header")


@dataclass
class ParsedSection:
    heading: str
    ordinal: int
    text: str
    tables_markdown: list[str] = field(default_factory=list)
    page_start: int = 1
    page_end: int = 1


class _FilingHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: list[ParsedSection] = []
        self._cur_heading = "Document"
        self._cur_ordinal = 0
        self._buf: list[str] = []
        self._in_table = 0
        self._table_rows: list[list[str]] = []
        self._cur_row: list[str] = []
        self._cur_cell: list[str] = []
        self._tables_for_section: list[str] = []
        self._page = 1
        self._ignore = 0  # >0 while inside script/style/head/ix:hidden

    def _flush_section(self):
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(self._buf)).strip()
        if text or self._tables_for_section:
            self.sections.append(ParsedSection(
                heading=self._cur_heading,
                ordinal=self._cur_ordinal,
                text=text,
                tables_markdown=list(self._tables_for_section),
                page_start=self._page,
                page_end=self._page,
            ))
            self._cur_ordinal += 1
        self._buf = []
        self._tables_for_section = []

    def handle_starttag(self, tag, attrs):
        if _is_junk_tag(tag):
            self._ignore += 1
            return
        if self._ignore:
            return
        if tag in _HEADING_TAGS:
            self._flush_section()
            self._cur_heading = "PENDING_HEADING"
        elif tag == "table":
            self._in_table += 1
            self._table_rows = []
        elif tag == "tr" and self._in_table:
            self._cur_row = []
        elif tag in ("td", "th") and self._in_table:
            self._cur_cell = []
        elif tag == "hr" or tag == "pagebreak":
            self._page += 1

    def handle_endtag(self, tag):
        if self._ignore:
            if _is_junk_tag(tag):
                self._ignore -= 1
            return
        if tag in _HEADING_TAGS:
            heading_text = "".join(self._buf).strip() or "Untitled section"
            self._buf = []
            self._cur_heading = heading_text
        elif tag in ("td", "th") and self._in_table:
            self._cur_row.append("".join(self._cur_cell).strip())
        elif tag == "tr" and self._in_table:
            if any(c for c in self._cur_row):
                self._table_rows.append(self._cur_row)
        elif tag == "table" and self._in_table:
            self._in_table -= 1
            if self._table_rows:
                self._tables_for_section.append(_rows_to_markdown(self._table_rows))

    def handle_data(self, data):
        if self._ignore:
            return
        if self._in_table:
            self._cur_cell.append(data)
        else:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush_section()


def _rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * width) + "|"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# EDGAR HTML almost never uses <h1>-<h4> for the Item/Part structure -- the
# headings exist only as text lines like "Item 1A. Risk Factors" or
# "PART II". Splitting on those lines is what gives sections (and citations,
# and contextual headers) real Item-level granularity instead of one giant
# "Document" section per filing.
_ITEM_LINE = re.compile(r"(?i)^item\s+\d{1,2}[a-z]?\s*[.)\-–—:]?\s*\S")
_PART_LINE = re.compile(r"(?i)^part\s+[ivx]{1,4}\s*[.)\-–—:]?(?:\s+\S.*)?$")


def _is_item_heading_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120 or len(s.split()) > 14:
        return False  # prose lines can start with "Item 2 of the agreement..."
    return bool(_ITEM_LINE.match(s) or _PART_LINE.match(s))


def _split_on_item_headings(section: ParsedSection) -> list[ParsedSection]:
    """Second pass: split one section's text on Item/Part heading lines.
    The heading line is kept IN its section's body as well, so full-text
    search over the heading words ("risk factors") still matches."""
    lines = section.text.splitlines()
    marks = [(i, l.strip()) for i, l in enumerate(lines) if _is_item_heading_line(l)]
    if not marks:
        return [section]
    bounds = [0] + [i for i, _ in marks] + [len(lines)]
    headings = [section.heading] + [h for _, h in marks]
    out: list[ParsedSection] = []
    ordinal = 0
    for k in range(len(bounds) - 1):
        start, end = bounds[k], bounds[k + 1]
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        out.append(ParsedSection(
            heading=headings[k], ordinal=ordinal, text=body,
            # section-level tables attach to the first non-empty piece
            tables_markdown=section.tables_markdown if not out else [],
            page_start=section.page_start, page_end=section.page_end,
        ))
        ordinal += 1
    return out or [section]


def parse_document(raw: str, is_html: bool | None = None) -> list[ParsedSection]:
    if is_html is None:
        is_html = "<" in raw and ">" in raw and re.search(r"<\s*\w+[^>]*>", raw)

    if not is_html:
        # Plain text: split on markdown-style headings or ALL-CAPS lines.
        lines = raw.splitlines()
        sections, cur_heading, cur_body, ordinal = [], "Document", [], 0
        for line in lines:
            stripped = line.strip()
            looks_like_heading = (
                stripped
                and (stripped.startswith("#") or (stripped.isupper() and len(stripped) < 90))
            )
            if looks_like_heading:
                if cur_body:
                    sections.append(ParsedSection(cur_heading, ordinal, "\n".join(cur_body).strip()))
                    ordinal += 1
                cur_heading = stripped.lstrip("#").strip()
                cur_body = []
            else:
                cur_body.append(line)
        if cur_body:
            sections.append(ParsedSection(cur_heading, ordinal, "\n".join(cur_body).strip()))
        return [s for s in sections if s.text]

    parser = _FilingHTMLParser()
    parser.feed(raw)
    parser.close()
    sections: list[ParsedSection] = []
    for s in parser.sections:
        sections.extend(_split_on_item_headings(s))
    for i, s in enumerate(sections):
        s.ordinal = i
    return sections
