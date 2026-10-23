"""
Rule-based query decomposition. No LLM required to decide *what kind* of
question this is -- keyword/entity matching against known graph nodes is
enough to decide whether to add a graph hop (peer comparison, "traced
across filings") on top of hybrid retrieval, and that decision needs to be
deterministic and inspectable, not left to a model's mood.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from sqlalchemy import select
from app.models import GraphNode

_TEMPORAL_MARKERS = ("changed", "across", "over time", "last three", "history", "evolved", "trend")
_COMPARISON_MARKERS = ("peer", "compare", "compared", "versus", "vs", "relative to")


@dataclass
class QueryPlan:
    raw_question: str
    mentioned_companies: list[str] = field(default_factory=list)
    wants_temporal: bool = False
    wants_peer_comparison: bool = False
    wants_graph_hop: bool = False


def build_plan(db, tenant_id: str, question: str) -> QueryPlan:
    q_lower = question.lower()
    companies = db.execute(
        select(GraphNode).where(GraphNode.tenant_id == tenant_id, GraphNode.node_type == "Company")
    ).scalars().all()

    mentioned = [c.canonical_name for c in companies if c.canonical_name.lower() in q_lower]

    wants_temporal = any(m in q_lower for m in _TEMPORAL_MARKERS)
    wants_peer = any(m in q_lower for m in _COMPARISON_MARKERS)

    return QueryPlan(
        raw_question=question,
        mentioned_companies=mentioned,
        wants_temporal=wants_temporal,
        wants_peer_comparison=wants_peer,
        wants_graph_hop=wants_temporal or wants_peer or len(mentioned) > 0,
    )
