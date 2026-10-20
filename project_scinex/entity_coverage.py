#!/usr/bin/env python3
"""
entity_coverage.py
------------------
Decide, per paper, whether `fixed` extraction is viable or the corpus should fall back
to `relation` extraction (relation ∈ ontology, subject free) for that paper.

Why this exists: the fixed extractor's subject pool comes from the CS-NER gazetteer
intersected with the paper's text, and CS-NER is annotated over CS/NLP papers. A
chemistry, materials or clinical paper can intersect it down to a handful of generic
terms, at which point `fixed` produces almost nothing — a coverage failure that looks
like an extraction-quality failure. This script separates the two before any GPU time
is spent, by counting what the entity CSV actually gives each paper.

It reads files only — no model, no GPU.

Usage:
    python3 entity_coverage.py                      # every parsed paper
    python3 entity_coverage.py --papers a b c       # just these
    python3 entity_coverage.py --min-entities 25    # stricter viability bar
    python3 entity_coverage.py --plan output/corpus_plan.json
"""

import argparse
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("output")


def parsed_papers() -> list[str]:
    if not OUTPUT_DIR.exists():
        sys.exit(f"No {OUTPUT_DIR}/ directory — run main.py on the PDFs first.")
    return sorted(
        d.name for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and d.name != "acl" and (d / "no-llm" / "output.html").exists()
    )


def entity_stats(paper_id: str) -> dict:
    """Entities the fixed extractor would actually get for this paper."""
    from kg_main import resolve_entity_csv

    csv_path = resolve_entity_csv(paper_id)
    if csv_path is None:
        return {"csv": None, "n_entities": 0, "n_all_rows": 0}

    from kg_extraction.entity_loader import load_entity_csv
    try:
        # tp_only=True mirrors what kg_main passes to the extractor; the untyped count
        # is reported too, since a CSV can be full of rows that the TP filter removes.
        tp = load_entity_csv(csv_path, tp_only=True)
        n_tp = len(tp)
    except Exception:
        n_tp = 0
    try:
        n_all = len(load_entity_csv(csv_path, tp_only=False))
    except Exception:
        n_all = 0

    return {"csv": str(csv_path), "n_entities": n_tp, "n_all_rows": n_all}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", nargs="+", default=None,
                    help="Paper ids to check (default: every parsed paper)")
    ap.add_argument("--min-entities", type=int, default=15,
                    help="Fewest TP entities for `fixed` to be worth running (default: 15)")
    ap.add_argument("--plan", default=None, metavar="JSON",
                    help="Write the fixed/relation split to this file")
    args = ap.parse_args()

    papers = args.papers or parsed_papers()
    if not papers:
        sys.exit("No parsed papers found under output/.")

    rows = []
    for pid in papers:
        st = entity_stats(pid)
        st["paper"] = pid
        if st["n_entities"] >= args.min_entities:
            st["decision"] = "fixed"
        elif st["n_entities"] > 0:
            st["decision"] = "relation"   # too thin to constrain subjects usefully
        else:
            st["decision"] = "relation"   # no entity list at all
        rows.append(st)

    width = max(len(r["paper"]) for r in rows) + 2
    print(f"\n{'═'*(width + 46)}")
    print(f"  {'Paper':<{width}}{'TP ents':>9}{'all rows':>10}{'decision':>12}   entity CSV")
    print(f"{'─'*(width + 46)}")
    for r in sorted(rows, key=lambda r: (r["decision"], -r["n_entities"])):
        csv_note = Path(r["csv"]).name if r["csv"] else "— none —"
        print(f"  {r['paper']:<{width}}{r['n_entities']:>9}{r['n_all_rows']:>10}"
              f"{r['decision']:>12}   {csv_note}")

    fixed = [r["paper"] for r in rows if r["decision"] == "fixed"]
    relation = [r["paper"] for r in rows if r["decision"] == "relation"]
    print(f"{'─'*(width + 46)}")
    print(f"  {len(fixed)} paper(s) → fixed      {len(relation)} paper(s) → relation "
          f"(threshold: {args.min_entities} entities)")
    print(f"{'═'*(width + 46)}\n")

    if relation:
        print("Papers with no usable entity list — check these before accepting the fallback:")
        print("  1. did enrich_entity_csv.py --source csner actually run for them?")
        print("  2. did the PDF parse (output/<id>/no-llm/output.html non-empty)?")
        print("  Only if both are fine is the gazetteer genuinely not covering the domain.\n")

    if fixed:
        print("# ontology-constrained subjects (both ontologies):")
        print(f"for p in {' '.join(fixed)}; do")
        print("  python3 kg_main.py --paper $p --extractor fixed --model Qwen/Qwen3-14B")
        print("  python3 kg_main.py --paper $p --extractor fixed --ontology scinex --model Qwen/Qwen3-14B")
        print("done\n")
    if relation:
        print("# free subjects, ontology relations (fallback):")
        print(f"for p in {' '.join(relation)}; do")
        print("  python3 kg_main.py --paper $p --extractor relation --model Qwen/Qwen3-14B")
        print("  python3 kg_main.py --paper $p --extractor relation --ontology scinex --model Qwen/Qwen3-14B")
        print("done\n")

    if args.plan:
        plan_path = Path(args.plan)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(
            {"min_entities": args.min_entities, "fixed": fixed,
             "relation": relation, "detail": rows},
            indent=2), encoding="utf-8")
        print(f"Plan → {plan_path}")


if __name__ == "__main__":
    main()
