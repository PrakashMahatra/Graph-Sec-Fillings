"""
Rule-based entity/relation extraction over chunk text.

This is the part of the spec that matters most: "The graph MUST be
evidence-backed. Do not create a relationship only because an LLM inferred
it." So extraction here is regex/pattern-based against governance-disclosure
language (which is fairly formulaic in 10-Ks/S-1s), and every extracted fact
carries the exact matched span as its evidence. No generative model writes
to the graph in this system.

Patterns cover the relations named in the spec:
  Company --FILED--> Filing
  Filing --DESCRIBES--> SecurityClass
  SecurityClass --HAS_VOTING_RIGHTS--> GovernanceEvent (votes-per-share fact)
  Person --HOLDS_ROLE--> BoardMember / Executive
  Shareholder --OWNS--> SecurityClass
  Company --PEER_OF--> Company
  Filing --SUPERSEDES--> Filing   (handled at ingest time from filing dates,
                                    not text extraction)
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class ExtractedFact:
    relation_type: str
    source: str            # canonical name of source node
    source_type: str
    target: str            # canonical name of target node
    target_type: str
    evidence_span: str
    properties: dict


# "Class A common stock" / "Class B Common Stock" / "Series A Preferred Stock"
_SECURITY_CLASS_RE = re.compile(
    r"\bClass\s+([A-Z])\s+(?:Common|Preferred)\s+Stock\b", re.IGNORECASE
)

# "each share of Class B common stock is entitled to 10 votes"
_VOTES_PER_SHARE_RE = re.compile(
    r"(Class\s+[A-Z]\s+(?:Common|Preferred)\s+Stock)[^.]{0,80}?"
    r"entitled to\s+([\d,]+)\s+votes?",
    re.IGNORECASE,
)

# "John Smith, our Chief Executive Officer" / "Jane Doe serves as a director"
_ROLE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-z]+)\s*,?\s*"
    r"(?:our|the Company's)?\s*"
    r"(Chief Executive Officer|Chief Financial Officer|Chairman(?: of the Board)?|"
    r"President|director|Director|board member)\b"
)

# "Jane Doe beneficially owns approximately 42% of our Class B common stock"
_OWNERSHIP_RE = re.compile(
    r"\b([A-Z][a-zA-Z.&, ]{2,60}?)\s+beneficially owns?\s+(?:approximately\s+)?"
    r"([\d.]+)%\s+of\s+(?:our\s+)?(Class\s+[A-Z]\s+(?:Common|Preferred)\s+Stock)",
    re.IGNORECASE,
)

# "our peer group includes Beta Industries, Gamma Corp and Delta Inc"
_PEER_RE = re.compile(
    r"peer group (?:includes|consists of)\s+([^.]+)\.", re.IGNORECASE
)
_PEER_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+")


def extract_facts(text: str, self_company_name: str) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []

    for m in _VOTES_PER_SHARE_RE.finditer(text):
        klass, votes = m.group(1), m.group(2).replace(",", "")
        facts.append(ExtractedFact(
            relation_type="HAS_VOTING_RIGHTS",
            source=_norm(klass), source_type="SecurityClass",
            target=f"{votes} votes per share", target_type="GovernanceEvent",
            evidence_span=m.group(0),
            properties={"votes_per_share": int(votes)},
        ))

    for m in _SECURITY_CLASS_RE.finditer(text):
        klass = m.group(0)
        facts.append(ExtractedFact(
            relation_type="DESCRIBES",
            source=self_company_name, source_type="Company",
            target=_norm(klass), target_type="SecurityClass",
            evidence_span=m.group(0),
            properties={},
        ))

    for m in _ROLE_RE.finditer(text):
        person, role = m.group(1).strip(), m.group(2).strip()
        node_type = "Executive" if "Chief" in role or "President" in role else "BoardMember"
        facts.append(ExtractedFact(
            relation_type="HOLDS_ROLE",
            source=person, source_type="Person",
            target=role, target_type=node_type,
            evidence_span=m.group(0),
            properties={"company": self_company_name},
        ))

    for m in _OWNERSHIP_RE.finditer(text):
        holder, pct, klass = m.group(1).strip(), float(m.group(2)), _norm(m.group(3))
        facts.append(ExtractedFact(
            relation_type="OWNS",
            source=holder, source_type="Shareholder",
            target=klass, target_type="SecurityClass",
            evidence_span=m.group(0),
            properties={"percent": pct},
        ))

    for m in _PEER_RE.finditer(text):
        names = [n.strip(" .") for n in _PEER_SPLIT_RE.split(m.group(1)) if n.strip(" .")]
        for raw in names:
            name = _clean_peer_name(raw)
            if not name:
                continue
            facts.append(ExtractedFact(
                relation_type="PEER_OF",
                source=self_company_name, source_type="Company",
                target=name, target_type="Company",
                evidence_span=m.group(0),
                properties={},
            ))

    return facts


def _clean_peer_name(raw: str) -> str:
    """Peer lists often come out of a sentence like

        "our peer group includes the following companies: Moody's
         Corporation, CME Group Inc."

    Splitting on commas leaves "the following companies: Moody's Corporation"
    -- a clause, not a company. Keep only the text after the final ':' and
    drop anything implausibly long, so the graph stores a company name.
    """
    name = re.sub(r"\s+", " ", raw).strip(" .,;")
    if ":" in name:
        name = name.rsplit(":", 1)[-1].strip(" .,;")
    if len(name) > 80 or len(name) < 2:
        return ""
    return name


def _norm(security_class_str: str) -> str:
    return re.sub(r"\s+", " ", security_class_str).strip().title()
