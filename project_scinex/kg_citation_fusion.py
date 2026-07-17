"""
kg_citation_fusion.py
---------------------
Merges an existing KG (from kg_main.py) with a citation network
(citation_network.json produced by the PDF parser pipeline).

Graph structure after fusion:
    Entity nodes   — same as KG (BERT, MLM, BooksCorpus, ...)
    Paper nodes    — one per paper in the citation network
    KG edges       — same as KG (trainedOn, addresses, ...)
    Citation edges — cites (paper → paper)
    Bridge edges   — mentions (paper → entity), meaning the paper's KG
                     entities are linked back to the paper node

The bridge edges let you ask:
    "Which papers mention the same entities?"
    "Which entities are central across citing/cited papers?"

Usage:
    python kg_citation_fusion.py --paper 1810.04805 --extractor fixed/Qwen3-32B
    python kg_citation_fusion.py --paper 1810.04805 --extractor llm/Qwen3-32B

Output:
    output/<paper>/kg/fused/fused_kg.graphml
    output/<paper>/kg/fused/fused_kg_stats.json
    output/<paper>/kg/fused/fused_kg_viz.html
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
KG_SUBDIR  = "kg"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_kg(kg_dir: Path) -> nx.MultiDiGraph:
    """Load existing KG from graphml file."""
    graphml_path = kg_dir / "kg.graphml"
    if not graphml_path.exists():
        raise FileNotFoundError(f"KG not found: {graphml_path}. Run kg_main.py first.")
    graph = nx.read_graphml(str(graphml_path))
    logger.info(f"Loaded KG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def load_citation_network(paper_dir: Path) -> dict:
    """Load citation_network.json from the paper's output directory."""
    # Try common locations
    candidates = [
        paper_dir / "citation_network.json",
        paper_dir / "no-llm" / "citation_network.json",
        paper_dir / "default" / "citation_network.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded citation network from {path}: {len(data.get('nodes', {}))} nodes")
            return data
    raise FileNotFoundError(
        f"citation_network.json not found in {paper_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse(kg: nx.MultiDiGraph, citation_data: dict) -> nx.MultiDiGraph:
    """
    Merge a KG graph with a citation network into one unified graph.

    Node types added:
        paper   — one per paper in the citation network (root + all cited/citing)
    Edge types added:
        cites         — paper → paper (from citation network)
        mentions      — paper → entity (bridge: root paper mentions its KG entities)
    """
    fused = kg.copy()

    root_id    = citation_data["paper_id"]
    root_title = citation_data["title"]
    root_year  = citation_data.get("year", "")

    # ── Add root paper node ───────────────────────────────────────────────────
    root_node = f"paper::{root_id}"
    fused.add_node(
        root_node,
        label=root_title,
        node_type="paper",
        paper_id=root_id,
        year=str(root_year),
        abstract=citation_data.get("abstract", "") or "",
        citation_count=str(citation_data.get("citation_count", 0)),
    )
    logger.info(f"Root paper node: {root_title[:60]}")

    # ── Add cited/citing paper nodes ──────────────────────────────────────────
    nodes_added = 0
    for paper_id, paper_data in citation_data.get("nodes", {}).items():
        node_id = f"paper::{paper_id}"
        fused.add_node(
            node_id,
            label=paper_data.get("title", paper_id),
            node_type="paper",
            paper_id=paper_id,
            year=str(paper_data.get("year", "")),
            abstract=paper_data.get("abstract", "") or "",
            citation_count=str(paper_data.get("citation_count", 0)),
            relation=paper_data.get("relation", ""),
        )
        nodes_added += 1

    logger.info(f"Added {nodes_added} citation paper nodes")

    # ── Add citation edges ────────────────────────────────────────────────────
    edges_added = 0
    for edge in citation_data.get("edges", []):
        src = f"paper::{edge['source']}"
        tgt = f"paper::{edge['target']}"
        if src in fused and tgt in fused:
            fused.add_edge(src, tgt, relation="cites", edge_type="citation")
            edges_added += 1

    logger.info(f"Added {edges_added} citation edges")

    # ── Bridge: root paper → its KG entities (mentions edges) ────────────────
    # Every entity node in the KG is mentioned by the root paper
    kg_entity_nodes = [
        n for n, d in fused.nodes(data=True)
        if not str(n).startswith("paper::")
    ]
    bridge_added = 0
    for entity_node in kg_entity_nodes:
        fused.add_edge(
            root_node,
            entity_node,
            relation="mentions",
            edge_type="bridge",
        )
        bridge_added += 1

    logger.info(f"Added {bridge_added} bridge (mentions) edges from root paper to KG entities")

    return fused


# ── Stats & save ──────────────────────────────────────────────────────────────

def get_fused_stats(graph: nx.MultiDiGraph) -> dict:
    """Compute stats for the fused graph."""
    paper_nodes  = [n for n, d in graph.nodes(data=True) if str(n).startswith("paper::")]
    entity_nodes = [n for n, d in graph.nodes(data=True) if not str(n).startswith("paper::")]

    from collections import Counter
    edge_types = Counter()
    for _, _, d in graph.edges(data=True):
        edge_types[d.get("relation", "unknown")] += 1

    # Most connected entities
    degree_sorted = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in degree_sorted[:10]]

    return {
        "total_nodes":    graph.number_of_nodes(),
        "entity_nodes":   len(entity_nodes),
        "paper_nodes":    len(paper_nodes),
        "total_edges":    graph.number_of_edges(),
        "edge_breakdown": dict(edge_types),
        "top_nodes":      top_nodes,
    }


def save_fused(graph: nx.MultiDiGraph, output_dir: Path, paper_title: str = ""):
    """Save fused graph to graphml, stats json, and html visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # GraphML
    graphml_path = output_dir / "fused_kg.graphml"
    nx.write_graphml(graph, str(graphml_path))
    logger.info(f"Saved fused KG → {graphml_path}")

    # Stats
    stats = get_fused_stats(graph)
    stats_path = output_dir / "fused_kg_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved stats → {stats_path}")
    logger.info(f"  Entity nodes : {stats['entity_nodes']}")
    logger.info(f"  Paper nodes  : {stats['paper_nodes']}")
    logger.info(f"  KG edges     : {stats['edge_breakdown'].get('mentions', 0)} mentions + "
                f"{sum(v for k,v in stats['edge_breakdown'].items() if k not in ('cites','mentions'))} KG relations")
    logger.info(f"  Citation edges: {stats['edge_breakdown'].get('cites', 0)}")

    # Visualization
    try:
        from kg_extraction.visualizer import build_viz
        viz_path = output_dir / "fused_kg_viz.html"
        build_viz(graph=graph, output_path=viz_path, paper_title=f"{paper_title} [FUSED]")
        logger.info(f"Saved viz → {viz_path}")
    except Exception as e:
        logger.warning(f"Visualization skipped: {e}")

    return graphml_path, stats_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse KG triples with citation network into one unified graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kg_citation_fusion.py --paper 1810.04805 --extractor fixed/Qwen3-32B
  python kg_citation_fusion.py --paper 1810.04805 --extractor llm/Qwen3-32B
  python kg_citation_fusion.py --paper 1810.04805 --extractor rebel
        """
    )
    parser.add_argument("--paper", required=True, metavar="PAPER_ID",
        help="Paper ID (e.g. 1810.04805)")
    parser.add_argument("--extractor", required=True, metavar="EXTRACTOR",
        help="Extractor subfolder (e.g. fixed/Qwen3-32B, llm/Qwen3-32B, rebel)")
    return parser.parse_args()


def main():
    args = parse_args()

    paper_dir = OUTPUT_DIR / args.paper
    if not paper_dir.exists():
        logger.error(f"Paper not found: {paper_dir}")
        sys.exit(1)

    # Load KG
    kg_dir = paper_dir / KG_SUBDIR / args.extractor
    kg = load_kg(kg_dir)

    # Load citation network
    citation_data = load_citation_network(paper_dir)

    # Fuse
    logger.info("Fusing KG with citation network...")
    fused = fuse(kg, citation_data)

    # Save
    output_dir = paper_dir / KG_SUBDIR / "fused"
    save_fused(fused, output_dir, paper_title=citation_data.get("title", args.paper))

    logger.info("Done.")


if __name__ == "__main__":
    main()