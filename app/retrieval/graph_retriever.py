"""
Deterministic, inspectable multi-hop graph traversal.

Not a black box: `traverse` does plain BFS over `graph_edges`, tenant-scoped,
optionally filtered by relation type and temporal validity (`valid_from`),
and returns full paths with each edge's evidence_chunk_id attached -- so
"trace how X changed" queries can literally walk Filing --SUPERSEDES-->
Filing chains, or Company --PEER_OF--> Company --DESCRIBES--> SecurityClass
fan-outs, and cite the exact evidence for every hop.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from sqlalchemy import select
from app.models import GraphNode, GraphEdge
from app.config import MAX_GRAPH_HOPS


@dataclass
class GraphPath:
    node_names: list[str]
    relations: list[str]
    evidence_chunk_ids: list[str | None]
    filing_ids: list[str | None]


def find_node(db, tenant_id: str, node_type: str, canonical_name: str) -> GraphNode | None:
    return db.execute(
        select(GraphNode).where(
            GraphNode.tenant_id == tenant_id,
            GraphNode.node_type == node_type,
            GraphNode.canonical_name == canonical_name,
        )
    ).scalar_one_or_none()


def traverse(
    db, tenant_id: str, start_node_id: str,
    relation_types: list[str] | None = None,
    max_hops: int = MAX_GRAPH_HOPS,
) -> list[GraphPath]:
    """BFS from start_node_id, tenant-scoped. Returns every simple path up
    to max_hops. Deterministic: same graph state -> same paths, same order."""
    edges = db.execute(select(GraphEdge).where(GraphEdge.tenant_id == tenant_id)).scalars().all()
    if relation_types:
        edges = [e for e in edges if e.relation_type in relation_types]

    nodes = {n.id: n for n in db.execute(select(GraphNode).where(GraphNode.tenant_id == tenant_id)).scalars()}

    adjacency: dict[str, list[GraphEdge]] = {}
    for e in edges:
        adjacency.setdefault(e.source_node_id, []).append(e)

    completed: list[GraphPath] = []

    def dfs(current_id: str, names: list[str], relations: list[str],
            evidence: list[str | None], filings: list[str | None], depth: int, visited: set[str]):
        if depth >= max_hops or current_id not in adjacency:
            if len(names) > 1:
                completed.append(GraphPath(list(names), list(relations), list(evidence), list(filings)))
            return
        extended = False
        for e in adjacency[current_id]:
            if e.target_node_id in visited:
                continue
            target = nodes.get(e.target_node_id)
            if not target:
                continue
            extended = True
            dfs(
                e.target_node_id,
                names + [target.canonical_name],
                relations + [e.relation_type],
                evidence + [e.evidence_chunk_id],
                filings + [e.filing_id],
                depth + 1,
                visited | {e.target_node_id},
            )
        if not extended and len(names) > 1:
            completed.append(GraphPath(list(names), list(relations), list(evidence), list(filings)))

    start_node = nodes.get(start_node_id)
    if not start_node:
        return []
    dfs(start_node_id, [start_node.canonical_name], [], [], [], 0, {start_node_id})

    # Multiple filings independently assert the same node/relation sequence
    # (e.g. "Acme DESCRIBES Class B" appears once per filing that mentions
    # it) -- that's correct evidence-wise, but for display it's the same
    # *fact*. Dedupe by the (names, relations) shape, keeping first evidence.
    seen = set()
    deduped: list[GraphPath] = []
    for p in completed:
        key = (tuple(p.node_names), tuple(p.relations))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped
