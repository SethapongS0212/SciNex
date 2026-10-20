"""
claude_extract.py
-----------------
Run the SAME extraction with Claude in place of the local model.

`kg_main.py` calls an extractor object that owns a prompt, a decoding step and a set of
post-parse guards. Only the middle step is model-specific. This script splits the
extractor in half so a model that has no local weights — Claude, in this session — can be
dropped into the same slot:

  prompts   render every paragraph of a paper into the exact system+user prompt the
            RelationOnlyExtractor / FixedTripleExtractor would have sent, and write them
            to prompts.jsonl (plus system_prompt.txt).

  ingest    take the raw model replies back (one JSON object per paragraph, same
            {"triples":[...]} contract), run them through the SAME
            `_parse_fixed_output()` guards and the SAME KnowledgeGraphBuilder, and write
            triples.json / kg.graphml / kg_stats.json where kg_main would have.

Because the prompt, the guards and the graph builder are imported — not reimplemented —
the only variable between this and a local run is the model. That is what makes the
comparison meaningful.

    python3 claude_extract.py prompts --paper strabismus2026 --extractor relation
    python3 claude_extract.py prompts --paper strabismus2026 --extractor relation --ontology scinex
    python3 claude_extract.py ingest  --paper strabismus2026 --extractor relation \
            --responses output/strabismus2026/kg/relation/claude-opus-5/responses.jsonl

Response file format — one JSON object per line, `para_id` matching prompts.jsonl:
    {"para_id": 0, "raw": "{\"triples\": [ ... ]}"}
"""

import argparse
import json
import logging
from pathlib import Path

from kg_extraction.fixed_extractor import _parse_fixed_output
from kg_extraction.html_parser import parse_html, split_into_sentences
from kg_extraction.kg_builder import KnowledgeGraphBuilder
from kg_extraction.relation_extractor import (
    RelationOnlyExtractor,
    _make_relation_user_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
DEFAULT_MODEL_SLUG = "claude-opus-5"


def build_extractor(kind: str, ontology: str, ontology_file: str,
                    entity_csv: Path | None, model_slug: str = DEFAULT_MODEL_SLUG):
    """Instantiate the real extractor (no weights are loaded until _load(), which we never call)."""
    relations = schema = None
    if ontology == "scinex":
        from kg_extraction.ontology_loader import load_ontology
        relations, schema = load_ontology(ontology_file)

    if kind == "relation":
        ex = RelationOnlyExtractor(relations=relations, schema=schema,
                                   model_name=model_slug)
        return ex, None

    from kg_extraction.entity_loader import load_entity_set
    from kg_extraction.fixed_extractor import FixedTripleExtractor, _make_fixed_user_prompt
    if entity_csv is None:
        raise SystemExit("--entity-csv is required for --extractor fixed")
    entity_set = load_entity_set(str(entity_csv))
    ex = FixedTripleExtractor(entity_set=entity_set, relations=relations, schema=schema,
                              model_name=model_slug)
    return ex, entity_set


def out_dir(paper: str, kind: str, ontology: str, model_slug: str) -> Path:
    name = kind if ontology == "ceo" else f"{kind}_scinex"
    return OUTPUT_DIR / paper / "kg" / name / model_slug


def cmd_prompts(args):
    ex, entity_set = build_extractor(args.extractor, args.ontology,
                                     args.ontology_file, args.entity_csv, args.model_slug)
    html = OUTPUT_DIR / args.paper / "no-llm" / "output.html"
    if not html.exists():
        raise SystemExit(f"No parsed HTML at {html} — run main.py first")

    title, sections = parse_html(str(html))
    dest = out_dir(args.paper, args.extractor, args.ontology, args.model_slug)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "system_prompt.txt").write_text(ex._system_prompt, encoding="utf-8")

    from kg_extraction.fixed_extractor import _is_garbled_section

    n = 0
    with open(dest / "prompts.jsonl", "w", encoding="utf-8") as f:
        for sec in sections:
            section = sec["section"]
            if _is_garbled_section(section):
                continue
            sentences = split_into_sentences(sec["text"])
            if not sentences:
                continue
            for para in ex._sentences_to_paragraphs(sentences):
                if args.extractor == "fixed":
                    present = entity_set.find_in_text(para) if entity_set else []
                    if not present:
                        continue
                    from kg_extraction.fixed_extractor import _make_fixed_user_prompt
                    user = _make_fixed_user_prompt(para, section, present,
                                                   entity_set.abbreviations)
                else:
                    user = _make_relation_user_prompt(para, section)
                f.write(json.dumps({
                    "para_id": n,
                    "section": section,
                    "paragraph": para,
                    "user_prompt": user,
                }, ensure_ascii=False) + "\n")
                n += 1

    meta = {
        "paper": args.paper, "title": title, "extractor": args.extractor,
        "ontology": args.ontology, "model_slug": args.model_slug,
        "n_paragraphs": n, "n_relations": len(ex.relations),
        "relations": ex.relations,
    }
    (dest / "prompt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"{n} paragraph prompts -> {dest/'prompts.jsonl'}")
    logger.info(f"system prompt ({len(ex._system_prompt)} chars) -> {dest/'system_prompt.txt'}")


def cmd_ingest(args):
    ex, entity_set = build_extractor(args.extractor, args.ontology,
                                     args.ontology_file, args.entity_csv, args.model_slug)
    dest = out_dir(args.paper, args.extractor, args.ontology, args.model_slug)
    meta = json.loads((dest / "prompt_meta.json").read_text(encoding="utf-8"))
    prompts = {json.loads(l)["para_id"]: json.loads(l)
               for l in open(dest / "prompts.jsonl", encoding="utf-8")}

    responses = [json.loads(l) for l in open(args.responses, encoding="utf-8") if l.strip()]
    logger.info(f"{len(responses)} responses for {len(prompts)} paragraphs")

    builder = KnowledgeGraphBuilder(entity_set=entity_set)
    kept = dropped = unparsed = empty = 0
    for resp in responses:
        pid = resp["para_id"]
        src = prompts.get(pid)
        if src is None:
            logger.warning(f"  para_id {pid} not in prompts.jsonl — skipped")
            continue
        raw = resp["raw"] if isinstance(resp["raw"], str) else json.dumps(resp["raw"])
        # `before` is only for the "rejected by the guards" tally, so a reply this
        # cheap parse can't read (markdown fences, a <think> block, a truncated
        # string) must not kill the ingest — _parse_fixed_output handles all three.
        try:
            before = len(json.loads(raw).get("triples", []))
        except Exception:
            before = 0
            unparsed += 1
        triples = _parse_fixed_output(raw, entity_set, ex.relations)
        dropped += max(0, before - len(triples))
        if not triples and before == 0:
            empty += 1
        for t in triples:
            if not t.get("source_sentence"):
                t["source_sentence"] = src["paragraph"][:500]
            t["extraction_mode"] = ex.EXTRACTION_MODE
            t.setdefault("paper", args.paper)
            t.setdefault("title", meta.get("title", ""))
            t.setdefault("section", src["section"])
        builder.add_triples(triples)
        kept += len(triples)

    triples_path, graphml_path, stats = builder.save(dest)
    logger.info(f"  kept {kept} triples ({dropped} rejected by the guards)")
    if unparsed or empty:
        logger.info(f"  {unparsed} replies were not plain JSON (fences/think-block/truncation), "
                    f"{empty} yielded nothing")
    logger.info(f"  nodes {stats['nodes']}  edges {stats['edges']}  triples {stats['triples']}")
    if stats.get("top_relations"):
        logger.info("  top rels : " + ", ".join(f"{r}({c})" for r, c in stats["top_relations"][:8]))
    logger.info(f"  wrote {triples_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("prompts", "ingest"):
        p = sub.add_parser(name)
        p.add_argument("--paper", required=True)
        p.add_argument("--extractor", choices=["relation", "fixed"], default="relation")
        p.add_argument("--ontology", choices=["ceo", "scinex"], default="ceo")
        p.add_argument("--ontology-file", default="scinex_refined_14.owl")
        p.add_argument("--entity-csv", type=Path, default=None)
        p.add_argument("--model-slug", default=DEFAULT_MODEL_SLUG,
                       help="Output subfolder name; must not contain ':' (Windows paths)")
        if name == "ingest":
            p.add_argument("--responses", required=True,
                           help="JSONL of {'para_id': int, 'raw': str} model replies")

    args = ap.parse_args()
    if ":" in args.model_slug:
        raise SystemExit("--model-slug must not contain ':' — it becomes a directory name")
    (cmd_prompts if args.cmd == "prompts" else cmd_ingest)(args)


if __name__ == "__main__":
    main()
