"""
global_graph.py
----------------
Persistent, incrementally-merged knowledge graph across ALL papers processed by
the fixed-style extractors (fixed / fixed_scinex).

Each paper is a "big node" (`paper:<paper_id>`). Fixed extraction's entities are
"augmented nodes" hanging off it via `mentions` edges. Citation edges become real
`cites` edges directly between big nodes — unlike kg_transe_pipeline.py's ephemeral
build_unified_graph(), which holds citation edges out as eval-only ground truth,
this graph includes them as real structure.

This module does NOT replace kg_transe_pipeline.py or change any existing per-paper
output format. It's an additive, persisted artifact:

    output/global_kg/<extractor>/<model_slug>/graph.graphml
    output/global_kg/<extractor>/<model_slug>/meta.json

Node identity across time: a citation edge references a neighbor paper by its raw
Semantic Scholar id, which may not be in our corpus yet. A stub node
`paper:<s2_id>` (type="paper_stub") is created for it; if/when that paper is later
extracted for real, its stub is relabeled to `paper:<acl_id>` so citation-network
connectivity discovered today is preserved once the referenced paper actually joins
the corpus, regardless of merge order.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from .kg_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

PAPER_PREFIX  = "paper:"
CITES_REL     = "cites"
MENTIONS_REL  = "mentions"

GRAPH_FILENAME = "graph.graphml"
META_FILENAME  = "meta.json"


def _paper_node(pid: str) -> str:
    return PAPER_PREFIX + pid


def graph_dir(root: str | Path, extractor: str, model_slug: str) -> Path:
    return Path(root) / extractor / model_slug


# ── Load / save ───────────────────────────────────────────────────────────────

def load_global_graph(root: str | Path, extractor: str, model_slug: str):
    """Load the persistent global graph + its bookkeeping meta, or start fresh."""
    d = graph_dir(root, extractor, model_slug)
    graph_path = d / GRAPH_FILENAME
    meta_path  = d / META_FILENAME

    if graph_path.exists():
        graph = nx.read_graphml(str(graph_path), force_multigraph=True)
        if not isinstance(graph, nx.MultiDiGraph):
            graph = nx.MultiDiGraph(graph)
    else:
        graph = nx.MultiDiGraph()

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.setdefault("merged_papers", {})
    meta.setdefault("s2_to_paper", {})

    logger.info(
        f"Loaded global graph ({extractor}/{model_slug}): "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
        f"{len(meta['merged_papers'])} papers already merged"
    )
    return graph, meta


def save_global_graph(graph, meta: dict, root: str | Path, extractor: str, model_slug: str):
    d = graph_dir(root, extractor, model_slug)
    d.mkdir(parents=True, exist_ok=True)

    graph_path = d / GRAPH_FILENAME
    meta_path  = d / META_FILENAME

    nx.write_graphml(graph, str(graph_path))

    meta["stats"] = {
        "nodes":         graph.number_of_nodes(),
        "edges":         graph.number_of_edges(),
        "papers_merged": len(meta["merged_papers"]),
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        f"Saved global graph ({extractor}/{model_slug}) → {graph_path} "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
        f"{len(meta['merged_papers'])} papers)"
    )


# ── Incremental merge ─────────────────────────────────────────────────────────

def merge_paper_into_graph(
    graph: nx.MultiDiGraph,
    meta: dict,
    paper_id: str,
    triples: list[dict],
    citation_network: dict | None = None,
    entity_set=None,
    force: bool = False,
) -> dict:
    """
    Fold one paper's fixed-extraction triples (+ its citation edges, if available)
    into the persistent global graph, in place.

    Returns merge stats: {"skipped", "reason", "entities_added", "cites_added",
    "stub_resolved"}.
    """
    n_triples = len(triples)
    prior = meta["merged_papers"].get(paper_id)
    if prior is not None and prior.get("n_triples") == n_triples and not force:
        return {"skipped": True, "reason": "unchanged", "entities_added": 0,
                "cites_added": 0, "stub_resolved": False}

    pnode = _paper_node(paper_id)

    # ── 1. Resolve this paper's own identity BEFORE adding anything, so a
    #      pre-existing citation stub (created earlier by some other paper's
    #      reference to this one) gets merged into the real node cleanly. ──────
    stub_resolved = False
    s2_id = (citation_network or {}).get("paper_id") or None
    if s2_id:
        stub_node = _paper_node(s2_id)
        if stub_node != pnode and stub_node in graph and pnode not in graph:
            graph = nx.relabel_nodes(graph, {stub_node: pnode}, copy=False)
            stub_resolved = True
        meta["s2_to_paper"][s2_id] = paper_id

    graph.add_node(pnode, type="paper")
    if citation_network:
        title = citation_network.get("title") or ""
        year  = citation_network.get("year")
        if title:
            graph.nodes[pnode]["title"] = title
        if year is not None:
            graph.nodes[pnode]["year"] = year

    # ── 2. Augmented (entity) nodes + entity-entity edges — reuse the same
    #      KnowledgeGraphBuilder kg_main.py already uses per paper, so entity
    #      normalisation/canonicalisation is identical to the per-paper KG. ────
    builder = KnowledgeGraphBuilder(entity_set=entity_set)
    builder.add_triples(triples)
    paper_graph = builder.graph

    entities_added = 0
    for node, data in paper_graph.nodes(data=True):
        is_new = node not in graph
        graph.add_node(node, **{**data, "type": "entity"})
        if is_new:
            entities_added += 1
        if not graph.has_edge(pnode, node, key=MENTIONS_REL):
            graph.add_edge(pnode, node, key=MENTIONS_REL, relation=MENTIONS_REL)

    for u, v, data in paper_graph.edges(data=True):
        rel = data.get("relation", "")
        key = f"{rel}|{data.get('paper', '')}|{data.get('section', '')}"
        if not graph.has_edge(u, v, key=key):
            graph.add_edge(u, v, key=key, **data)

    # ── 3. Citation edges → real paper-paper edges ───────────────────────────
    cites_added = 0
    if citation_network:
        nodes_meta = citation_network.get("nodes", {}) or {}

        def resolve(s2: str) -> str:
            acl = meta["s2_to_paper"].get(s2)
            node_id = _paper_node(acl if acl else s2)
            if node_id not in graph:
                attrs = {"type": "paper"} if acl else {"type": "paper_stub"}
                info = nodes_meta.get(s2, {})
                if info.get("title"):
                    attrs["title"] = info["title"]
                if info.get("year") is not None:
                    attrs["year"] = info["year"]
                graph.add_node(node_id, **attrs)
            return node_id

        for edge in citation_network.get("edges", []) or []:
            src_s2 = edge.get("source", "")
            tgt_s2 = edge.get("target", "")
            if not src_s2 or not tgt_s2:
                continue
            u = resolve(src_s2)
            v = resolve(tgt_s2)
            if not graph.has_edge(u, v, key=CITES_REL):
                graph.add_edge(u, v, key=CITES_REL, relation=CITES_REL)
                cites_added += 1

    meta["merged_papers"][paper_id] = {
        "n_triples": n_triples,
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "skipped": False,
        "reason": None,
        "entities_added": entities_added,
        "cites_added": cites_added,
        "stub_resolved": stub_resolved,
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def connectivity_report(graph: nx.MultiDiGraph, paper_ids: list[str]) -> dict:
    """
    For each paper_id, check whether its big node connects to anything else in
    the graph — via a `cites` edge to another paper node, or via a shared
    `mentions`-linked entity with some other paper. Used as a standing health
    check and by expand_and_validate.py to confirm newly-merged papers aren't
    isolated islands.
    """
    report = {}
    for pid in paper_ids:
        pnode = _paper_node(pid)
        if pnode not in graph:
            report[pid] = {"present": False}
            continue

        cites_edges = 0
        entity_neighbors = set()
        for _, v, data in graph.out_edges(pnode, data=True):
            if data.get("relation") == CITES_REL:
                cites_edges += 1
            elif data.get("relation") == MENTIONS_REL:
                entity_neighbors.add(v)
        for _, _, data in graph.in_edges(pnode, data=True):
            if data.get("relation") == CITES_REL:
                cites_edges += 1

        other_papers_via_entities = set()
        for ent in entity_neighbors:
            for u2, _, data2 in graph.in_edges(ent, data=True):
                if (data2.get("relation") == MENTIONS_REL
                        and u2 != pnode and u2.startswith(PAPER_PREFIX)):
                    other_papers_via_entities.add(u2)

        report[pid] = {
            "present":                 True,
            "cites_edges":             cites_edges,
            "shared_entity_neighbors": len(other_papers_via_entities),
            "isolated":                cites_edges == 0 and not other_papers_via_entities,
        }
    return report
