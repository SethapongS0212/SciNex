"""
ontology_loader.py — load a relation ontology from an OWL/Turtle file for the
fixed extractor.

The original CEO ontology is hardcoded in fixed_extractor.py (CEO_RELATIONS +
_CEO_SCHEMA). This loader lets us plug in an alternative ontology (e.g. the
refined `scinex` OWL) WITHOUT touching the CEO path, so the two can be run and
compared side by side.

Returns:
    relations: list[str]              # object-property local names (camelCase)
    schema:    dict[str, str]         # name -> "Domain → Range  |  <usage hint>"

Usage:
    from kg_extraction.ontology_loader import load_ontology
    relations, schema = load_ontology("scinex_refined_14.owl")
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _local(uri: str) -> str:
    s = str(uri)
    return s.split("#")[-1].split("/")[-1]


def load_ontology(owl_path: str) -> tuple[list[str], dict[str, str]]:
    """Parse an OWL/Turtle ontology → (relations, schema) for the fixed extractor."""
    import rdflib
    from rdflib import RDF, RDFS, OWL
    from rdflib.collection import Collection

    g = rdflib.Graph()
    # Turtle is the serialization used by scinex_refined_*.owl; fall back to xml.
    try:
        g.parse(owl_path, format="turtle")
    except Exception:
        g.parse(owl_path)

    def resolve(node) -> str:
        """Local name of a class node, expanding owl:unionOf blank nodes."""
        if isinstance(node, rdflib.BNode):
            for _, _, lst in g.triples((node, OWL.unionOf, None)):
                members = [_local(x) for x in Collection(g, lst)]
                if members:
                    return "/".join(members)
            return "?"
        return _local(node)

    relations: list[str] = []
    schema: dict[str, str] = {}

    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        name = _local(p)
        if not name:
            continue
        dom = "/".join(resolve(o) for o in g.objects(p, RDFS.domain)) or "?"
        rng = "/".join(resolve(o) for o in g.objects(p, RDFS.range)) or "?"
        comment = ""
        for _, _, c in g.triples((p, RDFS.comment, None)):
            comment = str(c)
            break
        # First sentence of the comment as a concise usage hint (prompt stays lean).
        hint = comment.split(". ")[0].strip()
        if len(hint) > 220:
            hint = hint[:217] + "..."
        schema[name] = f"{dom} → {rng}  |  {hint}" if hint else f"{dom} → {rng}"
        relations.append(name)

    relations.sort()
    if not relations:
        raise ValueError(f"No owl:ObjectProperty found in {owl_path}")
    logger.info(f"Loaded ontology from {owl_path}: {len(relations)} relations")
    return relations, schema
