"""
kg_builder.py
-------------
Builds a Knowledge Graph from extracted triples using NetworkX.

Graph structure:
    Nodes  = entities (subjects + objects), normalized to lowercase canonical form
    Edges  = predicates (directed: subject → object)
    Attrs  = paper source, section, count (how many times the edge appears)

Entity normalisation:
    If an EntitySet is provided, any surface form of a known entity
    (e.g. "BERT", "BERT_BASE", "OpenAI GPT") is collapsed to its canonical
    name (e.g. "Bidirectional Encoder Representations from Transformers").
    This prevents the same real-world entity appearing as multiple nodes.

Saves as:
    triples.json  — raw triples with full metadata
    kg.graphml    — standard graph format (loadable in Gephi, Neo4j, etc.)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


import re as _re

def _normalize(text: str) -> str:
    """Normalize entity names — lowercase, strip, replace underscores with spaces."""
    text = text.strip()
    text = text.replace('_', ' ')       # underscores → spaces
    text = _re.sub(r'\s+', ' ', text)   # collapse whitespace
    return text.lower().strip()


def _is_valid_entity(text: str) -> bool:
    """
    Filter out bad entities before they enter the graph.
    Returns False for math, citations, theorems, vague phrases, too short/long.
    """
    t = text.strip()

    # Too short or too long
    if len(t) < 3:
        return False
    if len(t.split()) > 8:
        return False

    # Math expressions — contain =, +, operators with numbers/symbols
    if _re.search(r'[=+|]', t) and _re.search(r'[0-9{}()\[\]]', t):
        return False

    # Citation keys like [GM21], [CFKM20]
    if _re.match(r'^\[[\w,\s]+\]$', t):
        return False

    # Theorem/Lemma/Proposition/Corollary labels
    if _re.match(r'^(theorem|lemma|proposition|corollary|assumption|definition|remark)\s*[\d\.]+', t, _re.IGNORECASE):
        return False

    # Vague generic phrases
    VAGUE = {
        'this paper', 'the authors', 'the paper', 'our work', 'our approach',
        'other schemes', 'various variants', 'the above', 'the following',
        'additional analysis', 'experimental setup', 'the proposed', 'recent work',
        'prior work', 'baseline', 'the model', 'the method', 'the algorithm',
        'the framework', 'the system', 'the approach', 'the technique',
    }
    if t.lower() in VAGUE:
        return False

    # Must contain at least one real letter (not just symbols/numbers)
    if not _re.search(r'[a-zA-Z]{2,}', t):
        return False

    return True


def _clean_label(text: str) -> str:
    """Clean entity label for display — remove underscores, fix spacing."""
    text = text.replace('_', ' ')
    text = _re.sub(r'\s+', ' ', text).strip()
    # Capitalize first letter
    return text[0].upper() + text[1:] if text else text


class KnowledgeGraphBuilder:
    """
    Builds and serializes a directed knowledge graph from triples.

    Optionally accepts an EntitySet for entity normalisation — any known
    surface form is resolved to its canonical name before entering the graph,
    preventing duplicate nodes for the same real-world entity.
    """

    def __init__(self, entity_set=None):
        """
        Args:
            entity_set: Optional EntitySet from entity_loader.py.
                        When provided, surface forms (e.g. "BERT", "OpenAI GPT",
                        "BERT_BASE") are collapsed to canonical names before
                        nodes are created, merging duplicate nodes.
        """
        self.graph = nx.MultiDiGraph()
        self.triples: list[dict] = []
        self._entity_set = entity_set

    def _resolve_entity(self, raw: str) -> str:
        """
        Resolve a raw entity string to its canonical form.

        Priority:
          1. EntitySet lookup — if the surface form matches a known entity,
             return the canonical name (e.g. "BERT" → "Bidirectional Encoder...")
          2. Fallback — return the lowercased, stripped, underscore-free string
        """
        if self._entity_set is not None:
            canonical = self._entity_set.match(raw)
            if canonical:
                return canonical.lower().strip()
            # Also try after basic normalisation
            canonical = self._entity_set.match(_normalize(raw))
            if canonical:
                return canonical.lower().strip()
        return _normalize(raw)

    def add_triples(self, triples: list[dict]):
        """
        Add a list of triple dicts to the graph.
        Each triple must have: subject, predicate, object.
        Optional metadata: paper, section, source_sentence.
        """
        skipped = 0
        merged  = 0

        for t in triples:
            raw_subj = t["subject"]
            raw_obj  = t["object"]

            # Filter bad entities before they enter the graph
            if not _is_valid_entity(raw_subj) or not _is_valid_entity(raw_obj):
                skipped += 1
                continue

            # Resolve to canonical form — merges surface variants into one node
            subj_prev = _normalize(raw_subj)
            subj = self._resolve_entity(raw_subj)
            obj  = self._resolve_entity(raw_obj)
            pred = t["predicate"].lower().strip().replace("_", " ")

            if subj != subj_prev:
                merged += 1
                logger.debug(f"Entity normalised: '{raw_subj}' → '{subj}'")

            if not subj or not pred or not obj:
                continue

            # Add nodes with cleaned display label
            # For known entities, use their canonical full name as the label
            if subj not in self.graph:
                if self._entity_set and self._entity_set.match(raw_subj):
                    label = _clean_label(self._entity_set.match(raw_subj))
                else:
                    label = _clean_label(raw_subj)
                self.graph.add_node(subj, label=label)

            if obj not in self.graph:
                if self._entity_set and self._entity_set.match(raw_obj):
                    label = _clean_label(self._entity_set.match(raw_obj))
                else:
                    label = _clean_label(raw_obj)
                self.graph.add_node(obj, label=label)

            # Add directed edge with relation and metadata
            self.graph.add_edge(
                subj, obj,
                relation=pred,
                paper=t.get("paper", ""),
                section=t.get("section", ""),
                sentence=t.get("source_sentence", "")
            )

            self.triples.append(t)

        if skipped:
            logger.debug(f"Skipped {skipped} triples with invalid entities")
        if merged:
            logger.info(f"Entity normalisation: {merged} surface forms resolved to canonical names")

        logger.info(
            f"Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def get_stats(self) -> dict:
        """Return basic graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "triples": 0}

        try:
            # Most connected entities
            degree_sorted = sorted(
                self.graph.degree(), key=lambda x: x[1], reverse=True
            )
            top_entities = [node for node, _ in degree_sorted[:10]]
        except Exception:
            top_entities = []

        # Most common relations
        relation_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relation_counts[data.get("relation", "")] += 1
        top_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "nodes":         self.graph.number_of_nodes(),
            "edges":         self.graph.number_of_edges(),
            "triples":       len(self.triples),
            "top_entities":  top_entities,
            "top_relations": top_relations,
        }

    def save(self, output_dir: str | Path):
        """Save triples.json and kg.graphml to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Raw triples JSON
        triples_path = output_dir / "triples.json"
        with open(triples_path, "w", encoding="utf-8") as f:
            json.dump(self.triples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.triples)} triples → {triples_path}")

        # 2. GraphML (interoperable with Gephi, Neo4j, etc.)
        graphml_path = output_dir / "kg.graphml"
        nx.write_graphml(self.graph, str(graphml_path))
        logger.info(f"Saved KG → {graphml_path}")

        # 3. Stats JSON
        stats = self.get_stats()
        stats_path = output_dir / "kg_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return triples_path, graphml_path, stats

    def load_graphml(self, path: str | Path):
        """Load an existing KG from GraphML (for inspection or merging)."""
        self.graph = nx.read_graphml(str(path))
        logger.info(f"Loaded KG from {path}: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")