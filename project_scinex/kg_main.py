#!/usr/bin/env python3
"""
kg_main.py
----------
CLI entry point for triple extraction + knowledge graph construction.
Supports multiple extractors: REBEL/ITER baselines and Qwen3-32B LLM.

Usage examples:
    python kg_main.py --paper 2205.11361 --extractor rebel        # REBEL only
    python kg_main.py --paper 2205.11361 --extractor iter         # ITER SciERC only
    python kg_main.py --paper 2205.11361 --extractor llm          # Qwen3-32B only
    python kg_main.py --paper 2205.11361 --extractor all          # all three extractors
    python kg_main.py --all --extractor all --limit 5             # first 5 papers, all extractors
    python kg_main.py --paper 2205.11361 --extractor rebel --no-viz
    python kg_main.py --paper 2205.11361 --extractor llm --model Qwen/Qwen3-32B

Output per paper:
    output/<paper>/kg/
        rebel/
            triples.json       all (S,P,O) triples extracted by REBEL
            kg.graphml         NetworkX graph (Gephi / Neo4j compatible)
            kg_stats.json      node/edge counts, top entities, top relations
            kg_viz.html        interactive pyvis visualization
        iter/
            triples.json       typed triples from ITER (SciERC)
            kg.graphml
            kg_stats.json
            kg_viz.html
        llm/
            Qwen3-32B/
                triples.json   triples extracted by Qwen LLM
                kg.graphml
                kg_stats.json
                kg_viz.html
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("output")
KG_SUBDIR   = "kg"
VALID_MODES = ("default", "fast", "no-llm")

# Extractors that feed the persistent cross-paper global graph (kg_extraction/global_graph.py).
# Scoped to "fixed extraction" only, per the entity+ontology triple shape it's designed around.
GLOBAL_GRAPH_EXTRACTORS = ("fixed", "fixed_scinex")


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_html(paper_dir: Path, mode: str) -> Path | None:
    """Find output.html for a paper. Tries mode subfolder first, then any available."""
    candidate = paper_dir / mode / "output.html"
    if candidate.exists():
        return candidate
    for m in VALID_MODES:
        candidate = paper_dir / m / "output.html"
        if candidate.exists():
            logger.warning(f"Mode '{mode}' not found for {paper_dir.name}, using '{m}'.")
            return candidate
    candidate = paper_dir / "output.html"
    if candidate.exists():
        return candidate
    return None


def discover_papers() -> list[Path]:
    """Return all paper directories under output/ that contain an output.html."""
    if not OUTPUT_DIR.exists():
        logger.error(f"Output directory '{OUTPUT_DIR}' not found. Run the parser first.")
        sys.exit(1)
    papers = []
    for paper_dir in sorted(OUTPUT_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        if any(paper_dir.rglob("output.html")):
            papers.append(paper_dir)
    return papers


# ── Entity CSV auto-resolution ────────────────────────────────────────────────

def resolve_entity_csv(paper_name: str) -> Path | None:
    """
    Find the entity CSV for a paper from its id alone, so fixed extraction doesn't
    need an explicit --entity-csv. Prefers the enriched ACL CSV, then the title-only
    ACL CSV, then manual CSVs in the project root. Returns the first existing path.
    """
    candidates = [
        OUTPUT_DIR / "acl" / paper_name / f"Entity_{paper_name}_enriched.csv",
        OUTPUT_DIR / "acl" / paper_name / f"Entity_{paper_name}.csv",
        Path(f"Entity_{paper_name}.csv"),
        Path(f"Entity-{paper_name}v2.csv"),
        Path(f"Entity - {paper_name}v2.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ── Global graph helpers ──────────────────────────────────────────────────────
# global_graphs: dict {(extractor_name, model_slug): [graph, meta, papers_since_save]}
# Loaded lazily (per extractor+model key) on first use, mutated in place, and saved
# by the caller (main()) — see kg_extraction/global_graph.py for the merge logic.

def _load_citation_network(paper_dir: Path) -> dict | None:
    cnet_path = paper_dir / "citation_network.json"
    if not cnet_path.exists():
        return None
    try:
        return json.loads(cnet_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  Could not read citation_network.json for {paper_dir.name}: {e}")
        return None


def _merge_into_global_graph(
    paper_dir: Path,
    paper_name: str,
    triples: list,
    extractor_name: str,
    model_slug: str,
    entity_set,
    global_graphs: dict,
    global_graph_dir,
) -> None:
    from kg_extraction import global_graph

    key = (extractor_name, model_slug)
    if key not in global_graphs:
        graph, meta = global_graph.load_global_graph(global_graph_dir, extractor_name, model_slug)
        global_graphs[key] = [graph, meta, 0]
    graph, meta, _ = global_graphs[key]

    cnet = _load_citation_network(paper_dir)
    result = global_graph.merge_paper_into_graph(
        graph, meta, paper_name, triples, citation_network=cnet, entity_set=entity_set,
    )
    if not result["skipped"]:
        global_graphs[key][2] += 1
        stub_note = " [resolved earlier citation stub]" if result["stub_resolved"] else ""
        logger.info(
            f"  Global graph ({extractor_name}): +{result['entities_added']} entities, "
            f"+{result['cites_added']} cite edges{stub_note}"
        )


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_extraction(
    paper_dir: Path,
    html_path: Path,
    extractor,
    extractor_name: str,
    no_viz: bool = False,
    entity_set=None,
    skip_existing: bool = False,
    global_graphs: dict | None = None,
    global_graph_dir="output/global_kg",
) -> dict | None:
    """
    Run triple extraction + KG construction for one paper with one extractor.
    Saves results to output/<paper>/kg/<extractor_name>/

    entity_set: Optional EntitySet for entity normalisation — surface form variants
                (e.g. "BERT", "BERT_BASE", "OpenAI GPT") are collapsed to canonical
                names before nodes are created in the graph.
    skip_existing: If True and triples.json already exists for this paper+extractor,
                   skip re-extraction (resume a batch).
    global_graphs: If not None and extractor_name is in GLOBAL_GRAPH_EXTRACTORS, this
                   paper's triples (+ citation_network.json, if present) are folded
                   into the persistent cross-paper global graph — even when skipped
                   via skip_existing, so a first run over an already-extracted corpus
                   backfills the global graph from disk instead of only covering
                   papers extracted from here on.
    """
    from kg_extraction import (
        parse_html, split_into_sentences,
        KnowledgeGraphBuilder, build_viz
    )

    paper_name = paper_dir.name

    # Resolve output dir up front so we can skip already-extracted papers
    # (llm/fixed get a model-name subfolder; rebel/iter stay flat)
    model_slug = None
    if extractor_name in ('llm', 'fixed', 'fixed_scinex', 'pair') and hasattr(extractor, 'model_name'):
        model_slug = extractor.model_name.split('/')[-1]
        kg_output_dir = paper_dir / KG_SUBDIR / extractor_name / model_slug
    else:
        kg_output_dir = paper_dir / KG_SUBDIR / extractor_name

    if skip_existing and (kg_output_dir / "triples.json").exists():
        logger.info(f"  ⏭  Skipping {extractor_name}: triples.json already exists at {kg_output_dir}")
        if global_graphs is not None and extractor_name in GLOBAL_GRAPH_EXTRACTORS:
            try:
                existing_triples = json.loads((kg_output_dir / "triples.json").read_text(encoding="utf-8"))
                _merge_into_global_graph(
                    paper_dir, paper_name, existing_triples, extractor_name, model_slug,
                    entity_set, global_graphs, global_graph_dir,
                )
            except Exception as e:
                logger.warning(f"  Could not backfill global graph for {paper_name}: {e}")
        return None

    # Parse HTML into sections
    paper_title, sections = parse_html(str(html_path))
    logger.info(f"  Title    : {paper_title}")
    logger.info(f"  Sections : {len(sections)}")

    # Reset LLM entity map between papers
    if hasattr(extractor, "reset_entity_map"):
        extractor.reset_entity_map()

    # Extract triples section by section
    kg_builder = KnowledgeGraphBuilder(entity_set=entity_set)
    total_units = 0

    for sec in sections:
        sentences = split_into_sentences(sec["text"])
        if not sentences:
            continue

        total_units += len(sentences)
        logger.info(f"    [{sec['section'][:45]}] {len(sentences)} sentences")

        triples = extractor.extract_from_sentences(
            sentences,
            source_meta={
                "paper":   paper_name,
                "title":   paper_title,
                "section": sec["section"],
            }
        )
        kg_builder.add_triples(triples)

    logger.info(f"  Units processed : {total_units}")

    # Step 2 — LLM post-processing (deduplicate, normalize, clean full triple list)
    all_triples = kg_builder.triples
    if hasattr(extractor, "postprocess_triples") and all_triples:
        logger.info(f"  Running LLM post-processing on {len(all_triples)} triples...")
        cleaned_triples = extractor.postprocess_triples(all_triples)
        # Rebuild KG from cleaned triples — keep entity_set for normalisation
        kg_builder = KnowledgeGraphBuilder(entity_set=entity_set)
        kg_builder.add_triples(cleaned_triples)
        logger.info(f"  KG after post-processing: {kg_builder.graph.number_of_nodes()} nodes, {kg_builder.graph.number_of_edges()} edges")

    # Save outputs (kg_output_dir resolved at top of function)
    triples_path, graphml_path, stats = kg_builder.save(kg_output_dir)

    logger.info(f"  Nodes    : {stats['nodes']}")
    logger.info(f"  Edges    : {stats['edges']}")
    logger.info(f"  Triples  : {stats['triples']}")

    if stats.get("top_entities"):
        logger.info(f"  Top nodes: {', '.join(stats['top_entities'][:5])}")
    if stats.get("top_relations"):
        top_rels = [f"{r}({c})" for r, c in stats["top_relations"][:5]]
        logger.info(f"  Top rels : {', '.join(top_rels)}")

    # Visualization
    if not no_viz and stats["nodes"] > 0:
        viz_path = kg_output_dir / "kg_viz.html"
        build_viz(
            graph=kg_builder.graph,
            output_path=viz_path,
            paper_title=f"{paper_title} [{extractor_name.upper()}]"
        )
        logger.info(f"  Viz      : {viz_path}")

    if global_graphs is not None and extractor_name in GLOBAL_GRAPH_EXTRACTORS:
        _merge_into_global_graph(
            paper_dir, paper_name, kg_builder.triples, extractor_name, model_slug,
            entity_set, global_graphs, global_graph_dir,
        )

    return {"extractor": extractor_name, **stats}


def process_paper(
    paper_dir: Path,
    mode: str,
    extractors: dict,
    no_viz: bool = False,
    entity_set=None,
    skip_existing: bool = False,
    auto_entity: bool = False,
    global_graphs: dict | None = None,
    global_graph_dir="output/global_kg",
) -> list[dict]:
    """Process one paper with all requested extractors."""
    paper_name = paper_dir.name

    html_path = find_html(paper_dir, mode)
    if html_path is None:
        logger.warning(f"No output.html found for '{paper_name}'. Skipping.")
        return []

    logger.info(f"{'─'*60}")
    logger.info(f"Paper : {paper_name}")
    logger.info(f"HTML  : {html_path}")

    # Auto-resolve this paper's entity CSV for the entity-driven extractors
    # (fixed, pair) when no explicit --entity-csv was given.
    ENTITY_EXTRACTORS = ("fixed", "fixed_scinex", "pair")
    paper_entity_set = entity_set
    entity_unavailable = False
    if auto_entity and any(e in extractors for e in ENTITY_EXTRACTORS):
        from kg_extraction import load_entity_csv
        csv_path = resolve_entity_csv(paper_name)
        if csv_path is None:
            logger.warning(f"  No entity CSV found for '{paper_name}' "
                           f"(looked in output/acl/{paper_name}/ and project root). "
                           f"Skipping entity-based extraction for this paper.")
            entity_unavailable = True
        else:
            paper_entity_set = load_entity_csv(csv_path, tp_only=True)
            for e in ENTITY_EXTRACTORS:
                if e in extractors:
                    extractors[e].entity_set = paper_entity_set
            logger.info(f"  Entity CSV (auto): {csv_path} ({len(paper_entity_set)} entities)")

    results = []
    for name, extractor in extractors.items():
        if name in ENTITY_EXTRACTORS and entity_unavailable:
            continue
        logger.info(f"  ── Extractor: {name.upper()} ──")
        stats = run_extraction(
            paper_dir=paper_dir,
            html_path=html_path,
            extractor=extractor,
            extractor_name=name,
            no_viz=no_viz,
            entity_set=paper_entity_set if name in ENTITY_EXTRACTORS else entity_set,
            skip_existing=skip_existing,
            global_graphs=global_graphs,
            global_graph_dir=global_graph_dir,
        )
        if stats:
            results.append({"paper": paper_name, **stats})

    return results


def _maybe_periodic_save(global_graphs: dict, global_graph_dir, save_every: int) -> None:
    """Checkpoint any (extractor, model) global graph that's accumulated >= save_every
    newly-merged papers since its last save — a crash safety net for long --all batches."""
    from kg_extraction import global_graph
    for key, (graph, meta, since_save) in global_graphs.items():
        if since_save >= save_every:
            extractor_name, model_slug = key
            global_graph.save_global_graph(graph, meta, global_graph_dir, extractor_name, model_slug)
            global_graphs[key][2] = 0


def _save_all_global_graphs(global_graphs: dict, global_graph_dir) -> None:
    """Final save for every (extractor, model) global graph with unsaved merges."""
    from kg_extraction import global_graph
    for key, (graph, meta, since_save) in global_graphs.items():
        if since_save > 0:
            extractor_name, model_slug = key
            global_graph.save_global_graph(graph, meta, global_graph_dir, extractor_name, model_slug)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Triple extraction + KG construction (REBEL vs LLM comparison).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kg_main.py --paper 2205.11361 --extractor rebel
  python kg_main.py --paper 2205.11361 --extractor iter
  python kg_main.py --paper 2205.11361 --extractor llm
  python kg_main.py --paper 2205.11361 --extractor all
  python kg_main.py --all --extractor all --limit 5
  python kg_main.py --paper 2205.11361 --extractor llm --model Qwen/Qwen3-32B
        """
    )

    # Target
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paper", metavar="PAPER_NAME",
        help="Single paper folder under output/ (e.g. 2205.11361)")
    group.add_argument("--all", action="store_true",
        help="Process all papers in output/")

    # Extractor choice
    parser.add_argument("--extractor", choices=["rebel", "iter", "llm", "fixed", "pair", "all"], default="rebel",
        help="Which extractor to use: rebel, iter, llm, fixed (constrained), or all (default: rebel)")

    # Options
    parser.add_argument("--mode", choices=VALID_MODES, default="no-llm",
        help="Which pipeline mode's output.html to use (default: no-llm)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
        help="When using --all, process only first N papers")
    parser.add_argument("--entity-csv", default=None, metavar="CSV_PATH",
        help="Path to entity CSV for fixed extractor. Optional: if omitted, the CSV is "
             "auto-resolved per paper from its id (output/acl/<id>/Entity_<id>_enriched.csv, "
             "then Entity_<id>.csv, then project-root manual CSVs).")
    parser.add_argument("--relations", nargs="+", default=None, metavar="RELATION",
        help="Custom relation list for fixed extractor (default: placeholder set)")
    parser.add_argument("--ontology", choices=["ceo", "scinex"], default="ceo",
        help="Ontology for the fixed extractor: 'ceo' (default, hardcoded Core "
             "Experiment Ontology) or 'scinex' (loaded from --ontology-file). "
             "scinex output is written to kg/fixed_scinex/ so both can be compared.")
    parser.add_argument("--ontology-file", default="scinex_refined_14.owl", metavar="OWL",
        help="OWL/Turtle file to load when --ontology scinex (default: scinex_refined_14.owl)")
    parser.add_argument("--no-viz", action="store_true",
        help="Skip HTML visualization")
    parser.add_argument("--no-gpu", action="store_true",
        help="Force CPU inference (for local debugging)")

    # REBEL options
    parser.add_argument("--batch-size", type=int, default=8,
        help="Batch size for REBEL inference (default: 8)")

    # LLM options
    parser.add_argument("--model", default=None, metavar="HF_MODEL_ID",
        help="HuggingFace model ID for LLM extractor (default: Qwen/Qwen3-32B)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
        help="Max tokens for LLM generation (default: 4096; Qwen3 needs headroom beyond its think block)")
    parser.add_argument("--skip-existing", action="store_true",
        help="Skip papers whose triples.json already exists for this extractor (resume a batch)")

    # Persistent cross-paper global graph (fixed/fixed_scinex only)
    parser.add_argument("--global-graph-dir", default="output/global_kg", metavar="DIR",
        help="Root dir for the persistent cross-paper global KG that fixed/fixed_scinex "
             "extraction incrementally merges into (default: output/global_kg). Each "
             "paper becomes a 'paper:<id>' node linked to its extracted entities via "
             "'mentions' edges and to other papers via real 'cites' edges.")
    parser.add_argument("--no-global-merge", action="store_true",
        help="Don't merge this run's fixed/fixed_scinex extractions into the persistent "
             "global graph (per-paper triples.json/kg.graphml are unaffected either way).")
    parser.add_argument("--global-save-every", type=int, default=25, metavar="N",
        help="Checkpoint the global graph to disk every N newly-merged papers during a "
             "batch run, as a crash safety net (default: 25). Always saved at the end too.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paper list
    if args.all:
        papers = discover_papers()
        if args.limit:
            papers = papers[:args.limit]
        logger.info(f"Found {len(papers)} paper(s).")
    else:
        paper_dir = OUTPUT_DIR / args.paper
        if not paper_dir.exists():
            logger.error(f"Paper directory not found: {paper_dir}")
            sys.exit(1)
        papers = [paper_dir]

    if not papers:
        logger.error("No papers to process.")
        sys.exit(1)

    # Build extractor dict — load once, reuse across papers
    from kg_extraction import TripleExtractor, LLMExtractor, ITERExtractor, FixedTripleExtractor, load_entity_csv

    extractors = {}
    device = "cpu" if args.no_gpu else None

    if args.extractor in ("rebel", "all"):
        logger.info("Initializing REBEL extractor...")
        extractors["rebel"] = TripleExtractor(
            device=device,
            batch_size=args.batch_size
        )

    if args.extractor in ("iter", "all"):
        logger.info("Initializing ITER extractor (SciERC)...")
        extractors["iter"] = ITERExtractor(device=device)

    if args.extractor in ("llm", "all"):
        logger.info("Initializing LLM extractor...")
        extractors["llm"] = LLMExtractor(
            model_name=args.model,       # None = default Qwen3-32B
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

    auto_entity = False
    # fixed (subject∈list, relation∈ontology, object free) and
    # pair  (subject∈list, relation free, object∈list) are both entity-CSV-driven.
    if args.extractor in ("fixed", "pair", "all"):
        from kg_extraction import EntitySet, EntityPairExtractor
        from pathlib import Path as _Path
        if args.entity_csv:
            # Explicit CSV → one entity set for all papers (original behaviour)
            entity_csv = _Path(args.entity_csv)
            if not entity_csv.exists():
                logger.error(f"Entity CSV not found: {entity_csv}")
                sys.exit(1)
            logger.info(f"Loading entity set from {entity_csv.name}...")
            entity_set = load_entity_csv(entity_csv, tp_only=True)
            logger.info(f"Loaded {len(entity_set)} entities (TP=1)")
        else:
            # No CSV given → auto-resolve per paper from the paper id
            auto_entity = True
            entity_set = EntitySet([], {}, {})   # placeholder, swapped per paper
            logger.info("No --entity-csv given → auto-resolving the entity CSV per "
                        "paper from its id (enriched ACL CSV preferred).")
        if args.extractor in ("fixed", "all"):
            # Ontology selection: 'ceo' = hardcoded default; 'scinex' = loaded
            # from an .owl and routed to a separate output dir (kg/fixed_scinex/)
            # so it never overwrites the CEO results — the two are compared.
            fixed_relations = args.relations
            fixed_schema = None
            fixed_name = "fixed"
            if args.ontology == "scinex":
                from kg_extraction.ontology_loader import load_ontology
                fixed_relations, fixed_schema = load_ontology(args.ontology_file)
                fixed_name = "fixed_scinex"
                logger.info(f"Using scinex ontology ({len(fixed_relations)} relations) "
                            f"from {args.ontology_file} → output dir kg/{fixed_name}/")
            extractors[fixed_name] = FixedTripleExtractor(
                entity_set=entity_set,
                relations=fixed_relations,   # None = placeholder set (ceo)
                model_name=args.model,       # None = default Qwen3-14B
                device=device,
                max_new_tokens=args.max_new_tokens,
                schema=fixed_schema,         # None = CEO schema
            )
        if args.extractor in ("pair", "all"):
            extractors["pair"] = EntityPairExtractor(
                entity_set=entity_set,       # subject AND object constrained to this set
                model_name=args.model,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

    # Process all papers
    global_graphs = None if args.no_global_merge else {}
    all_results = []
    try:
        for paper_dir in papers:
            results = process_paper(
                paper_dir=paper_dir,
                mode=args.mode,
                extractors=extractors,
                no_viz=args.no_viz,
                entity_set=entity_set if 'entity_set' in dir() else None,
                skip_existing=args.skip_existing,
                auto_entity=auto_entity,
                global_graphs=global_graphs,
                global_graph_dir=args.global_graph_dir,
            )
            all_results.extend(results)

            if global_graphs:
                _maybe_periodic_save(global_graphs, args.global_graph_dir, args.global_save_every)
    finally:
        if global_graphs:
            _save_all_global_graphs(global_graphs, args.global_graph_dir)

    # Unload all models
    for name, ext in extractors.items():
        ext.unload()

    # Final summary
    logger.info(f"\n{'═'*70}")
    logger.info(f"DONE — {len(papers)} paper(s) × {len(extractors)} extractor(s)")
    logger.info(f"{'─'*70}")
    logger.info(f"  {'Paper':<30} {'Extractor':<8} {'Triples':>8} {'Nodes':>7} {'Edges':>7}")
    logger.info(f"{'─'*70}")
    for r in all_results:
        logger.info(
            f"  {r['paper']:<30} {r['extractor']:<8} "
            f"{r['triples']:>8} {r['nodes']:>7} {r['edges']:>7}"
        )
    logger.info(f"{'═'*70}")


if __name__ == "__main__":
    main()
