"""
kg_fixed_compare.py
-------------------
Detailed comparison between fixed-subject triples and free triples.

Fixed  = subjects constrained to ground-truth entity set (TP=1 from CSV)
Free   = subjects chosen freely by LLM

Comparison shows:
  1. Overview stats — triple counts, unique entities, unique relations
  2. Entity coverage — which fixed entities were found vs missed
  3. Relation distribution — what relations each mode uses
  4. Triple overlap — triples that appear in both (subject + object match)
  5. Fixed-only triples — what fixed mode found that free missed
  6. Free-only triples — what free mode found that fixed missed
  7. Per-entity breakdown — for each fixed entity, what did each mode find

Usage:
    python kg_fixed_compare.py --paper BERT --entity-csv Entity_-_BERTv2.csv
    python kg_fixed_compare.py --paper KAN --entity-csv Entity_-_KANv2.csv
    python kg_fixed_compare.py --paper BERT --entity-csv Entity_-_BERTv2.csv --save-report
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
KG_SUBDIR  = "kg"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_triples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_triples_path(paper_dir: Path, mode: str) -> Path | None:
    """
    Find triples.json for a given mode.
    mode: 'fixed' or 'free'

    kg_main.py saves to:
        fixed  →  kg/fixed/<model_name>/triples.json
        llm    →  kg/llm/<model_name>/triples.json
        rebel  →  kg/rebel/triples.json

    This function checks the model-name subdirectory structure first,
    then falls back to the flat path for older runs.
    """
    kg_dir = paper_dir / KG_SUBDIR

    if mode == "fixed":
        fixed_dir = kg_dir / "fixed"
        if fixed_dir.exists():
            # Prefer most-recently-modified model subfolder
            model_dirs = sorted(
                [d for d in fixed_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            for d in model_dirs:
                candidate = d / "triples.json"
                if candidate.exists():
                    return candidate
            # Flat fallback for older runs
            flat = fixed_dir / "triples.json"
            if flat.exists():
                return flat

    elif mode == "free":
        # Try llm/<model_name>/triples.json
        llm_dir = kg_dir / "llm"
        if llm_dir.exists():
            model_dirs = sorted(
                [d for d in llm_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            for d in model_dirs:
                candidate = d / "triples.json"
                if candidate.exists():
                    return candidate
        # Fallback: rebel
        candidate = kg_dir / "rebel" / "triples.json"
        if candidate.exists():
            return candidate

    return None


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _triple_key(t: dict) -> tuple:
    """Normalized key for triple matching."""
    return (
        t.get("subject", "").lower().strip(),
        t.get("object",  "").lower().strip(),
    )


def _triple_full_key(t: dict) -> tuple:
    """Full key including predicate."""
    return (
        t.get("subject",   "").lower().strip(),
        t.get("predicate", "").lower().strip(),
        t.get("object",    "").lower().strip(),
    )


def compute_overlap(fixed: list[dict], free: list[dict]) -> dict:
    """Find triples that appear in both fixed and free (by subject+object)."""
    fixed_keys = {_triple_key(t): t for t in fixed}
    free_keys  = {_triple_key(t): t for t in free}

    overlap_keys   = set(fixed_keys.keys()) & set(free_keys.keys())
    fixed_only_keys = set(fixed_keys.keys()) - set(free_keys.keys())
    free_only_keys  = set(free_keys.keys())  - set(fixed_keys.keys())

    return {
        "overlap":     [fixed_keys[k] for k in overlap_keys],
        "fixed_only":  [fixed_keys[k] for k in fixed_only_keys],
        "free_only":   [free_keys[k]  for k in free_only_keys],
        "overlap_count":     len(overlap_keys),
        "fixed_only_count":  len(fixed_only_keys),
        "free_only_count":   len(free_only_keys),
    }


def entity_coverage(
    fixed_triples: list[dict],
    free_triples: list[dict],
    entity_set_names: list[str],
) -> dict:
    """
    For each entity in the ground truth set, check if it appears
    as a subject in fixed triples and/or free triples.
    """
    fixed_subjects = {t.get("subject", "").lower() for t in fixed_triples}
    free_subjects  = {t.get("subject", "").lower() for t in free_triples}

    coverage = {}
    for entity in entity_set_names:
        entity_lower = entity.lower()
        in_fixed = entity_lower in fixed_subjects
        in_free  = entity_lower in free_subjects
        coverage[entity] = {
            "in_fixed": in_fixed,
            "in_free":  in_free,
            "in_both":  in_fixed and in_free,
            "missed_by_both": not in_fixed and not in_free,
        }

    found_fixed       = sum(1 for v in coverage.values() if v["in_fixed"])
    found_free        = sum(1 for v in coverage.values() if v["in_free"])
    found_both        = sum(1 for v in coverage.values() if v["in_both"])
    missed_both       = sum(1 for v in coverage.values() if v["missed_by_both"])
    fixed_only_count  = sum(1 for v in coverage.values() if v["in_fixed"] and not v["in_free"])
    free_only_count   = sum(1 for v in coverage.values() if v["in_free"]  and not v["in_fixed"])

    return {
        "per_entity":        coverage,
        "total_entities":    len(entity_set_names),
        "found_by_fixed":    found_fixed,
        "found_by_free":     found_free,
        "found_by_both":     found_both,
        "missed_by_both":    missed_both,
        "fixed_only_count":  fixed_only_count,
        "free_only_count":   free_only_count,
    }


def per_entity_triples(triples: list[dict]) -> dict[str, list[dict]]:
    """Group triples by subject entity."""
    result = defaultdict(list)
    for t in triples:
        subj = t.get("subject", "").strip()
        if subj:
            result[subj].append(t)
    return dict(result)


# ── Report printing ───────────────────────────────────────────────────────────

def _bar(value: int, max_val: int, width: int = 25) -> str:
    if max_val == 0:
        return "░" * width
    filled = int((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    paper_name: str,
    fixed: list[dict],
    free: list[dict],
    overlap: dict,
    coverage: dict,
    entity_names: list[str],
):
    sep  = "═" * 72
    sep2 = "─" * 72

    print(f"\n{sep}")
    print(f"  FIXED vs FREE TRIPLE COMPARISON — {paper_name}")
    print(f"{sep}")

    # ── 1. Overview ───────────────────────────────────────────────────────────
    print(f"\n  {'OVERVIEW':<35} {'FIXED':>10} {'FREE':>10}")
    print(sep2)

    fixed_subjs = {t.get("subject","").lower() for t in fixed}
    free_subjs  = {t.get("subject","").lower() for t in free}
    fixed_preds = {t.get("predicate","").lower() for t in fixed}
    free_preds  = {t.get("predicate","").lower() for t in free}
    fixed_objs  = {t.get("object","").lower() for t in fixed}
    free_objs   = {t.get("object","").lower() for t in free}

    rows = [
        ("Total triples",         len(fixed),          len(free)),
        ("Unique subjects",        len(fixed_subjs),    len(free_subjs)),
        ("Unique predicates",      len(fixed_preds),    len(free_preds)),
        ("Unique objects",         len(fixed_objs),     len(free_objs)),
        ("Unique entities total",  len(fixed_subjs | fixed_objs), len(free_subjs | free_objs)),
    ]
    for label, fv, rv in rows:
        print(f"  {label:<35} {fv:>10} {rv:>10}")

    # ── 2. Overlap ────────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  TRIPLE OVERLAP  (matched by subject + object)")
    print(sep2)
    total = max(len(fixed), len(free), 1)
    print(f"  In both        : {overlap['overlap_count']:>4}  {_bar(overlap['overlap_count'], total)}")
    print(f"  Fixed only     : {overlap['fixed_only_count']:>4}  {_bar(overlap['fixed_only_count'], total)}")
    print(f"  Free only      : {overlap['free_only_count']:>4}  {_bar(overlap['free_only_count'], total)}")

    if fixed:
        pct = overlap['overlap_count'] / len(fixed) * 100
        print(f"\n  Fixed triples captured by free mode : {pct:.1f}%")
    if free:
        pct2 = overlap['overlap_count'] / len(free) * 100
        print(f"  Free triples captured by fixed mode : {pct2:.1f}%")

    # ── 3. Entity coverage ────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  ENTITY COVERAGE  (ground truth entities, TP=1)")
    print(sep2)
    c = coverage
    print(f"  Total ground truth entities : {c['total_entities']}")
    print(f"  Found by FIXED              : {c['found_by_fixed']}  ({c['found_by_fixed']/max(c['total_entities'],1)*100:.1f}%)")
    print(f"  Found by FREE               : {c['found_by_free']}   ({c['found_by_free']/max(c['total_entities'],1)*100:.1f}%)")
    print(f"  Found by BOTH               : {c['found_by_both']}")
    print(f"  Fixed only (not in free)    : {c['fixed_only_count']}")
    print(f"  Free only  (not in fixed)   : {c['free_only_count']}")
    print(f"  Missed by both              : {c['missed_by_both']}")

    # Entities missed by both
    missed = [e for e, v in c["per_entity"].items() if v["missed_by_both"]]
    if missed:
        print(f"\n  Entities missed by both extractors:")
        for e in missed[:10]:
            print(f"    • {e}")
        if len(missed) > 10:
            print(f"    ... and {len(missed)-10} more")

    # ── 4. Relation distribution ──────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  RELATION DISTRIBUTION")
    print(sep2)
    fixed_rels = Counter(t.get("predicate","") for t in fixed)
    free_rels  = Counter(t.get("predicate","") for t in free)
    all_rels   = set(fixed_rels) | set(free_rels)
    max_count  = max((max(fixed_rels.values(), default=0), max(free_rels.values(), default=0)), default=1)

    print(f"  {'Relation':<30} {'Fixed':>7} {'Free':>7}")
    print(f"  {'─'*44}")
    for rel in sorted(all_rels, key=lambda r: -(fixed_rels.get(r,0)+free_rels.get(r,0)))[:15]:
        fc = fixed_rels.get(rel, 0)
        rc = free_rels.get(rel, 0)
        print(f"  {rel:<30} {fc:>7} {rc:>7}")

    # ── 5. Sample triples — overlap ───────────────────────────────────────────
    print(f"\n{sep2}")
    print("  SAMPLE TRIPLES IN BOTH (agree on subject + object)")
    print(sep2)
    for t in overlap["overlap"][:5]:
        print(f"  ({t.get('subject','')[:35]}) --[{t.get('predicate','')[:20]}]--> ({t.get('object','')[:35]})")

    # ── 6. Fixed-only samples ─────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  SAMPLE TRIPLES — FIXED ONLY (not found by free)")
    print(sep2)
    if overlap["fixed_only"]:
        for t in overlap["fixed_only"][:5]:
            print(f"  ({t.get('subject','')[:35]}) --[{t.get('predicate','')[:20]}]--> ({t.get('object','')[:35]})")
            if t.get("source_sentence"):
                print(f"    ↳ {t['source_sentence'][:80]}")
    else:
        print("  (none)")

    # ── 7. Free-only samples ──────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  SAMPLE TRIPLES — FREE ONLY (not found by fixed)")
    print(sep2)
    if overlap["free_only"]:
        for t in overlap["free_only"][:5]:
            print(f"  ({t.get('subject','')[:35]}) --[{t.get('predicate','')[:20]}]--> ({t.get('object','')[:35]})")
            if t.get("source_sentence"):
                print(f"    ↳ {t['source_sentence'][:80]}")
    else:
        print("  (none)")

    # ── 8. Per-entity breakdown ───────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  PER-ENTITY BREAKDOWN (ground truth entities)")
    print(sep2)
    fixed_by_entity = per_entity_triples(fixed)
    free_by_entity  = per_entity_triples(free)

    print(f"  {'Entity':<40} {'Fixed':>6} {'Free':>6} {'Status'}")
    print(f"  {'─'*60}")
    for entity in entity_names:
        f_triples = fixed_by_entity.get(entity, [])
        r_triples = free_by_entity.get(entity, [])
        fc = len(f_triples)
        rc = len(r_triples)

        if fc > 0 and rc > 0:
            status = "✓ both"
        elif fc > 0:
            status = "→ fixed only"
        elif rc > 0:
            status = "→ free only"
        else:
            status = "✗ missed"

        print(f"  {entity[:40]:<40} {fc:>6} {rc:>6}  {status}")

        # Show sample triples for each entity
        shown = set()
        for t in (f_triples + r_triples)[:3]:
            key = _triple_full_key(t)
            if key not in shown:
                mode = "F" if t in f_triples else "R"
                print(f"    [{mode}] --[{t.get('predicate','')[:18]}]--> {t.get('object','')[:40]}")
                shown.add(key)

    print(f"\n{sep}\n")


# ── Core ──────────────────────────────────────────────────────────────────────

def run_comparison(
    paper_dir: Path,
    entity_csv: Path,
    save_report: bool = False,
):
    from kg_extraction.entity_loader import load_entity_csv

    paper_name = paper_dir.name

    # Load entity set
    entity_set = load_entity_csv(entity_csv, tp_only=True)
    logger.info(entity_set.summary())

    # Load fixed and free triples
    fixed_path = find_triples_path(paper_dir, "fixed")
    free_path  = find_triples_path(paper_dir, "free")

    if not fixed_path:
        logger.error(
            f"No fixed triples found for '{paper_name}'. "
            f"Run: python kg_main.py --paper {paper_name} --extractor fixed --entity-csv <path>"
        )
        return

    if not free_path:
        logger.error(
            f"No free triples found for '{paper_name}'. "
            f"Run: python kg_main.py --paper {paper_name} --extractor llm"
        )
        return

    fixed = load_triples(fixed_path)
    free  = load_triples(free_path)

    logger.info(f"Fixed triples : {len(fixed)} (from {fixed_path})")
    logger.info(f"Free  triples : {len(free)}  (from {free_path})")

    # Compute analysis
    overlap  = compute_overlap(fixed, free)
    coverage = entity_coverage(fixed, free, entity_set.entities)

    # Print report
    print_report(paper_name, fixed, free, overlap, coverage, entity_set.entities)

    # Save report
    if save_report:
        report = {
            "paper":    paper_name,
            "fixed_count": len(fixed),
            "free_count":  len(free),
            "overlap":  {k: v for k, v in overlap.items() if not isinstance(v, list)},
            "coverage": {
                k: v for k, v in coverage.items() if k != "per_entity"
            },
            "per_entity_coverage": coverage["per_entity"],
            "relation_distribution": {
                "fixed": dict(Counter(t.get("predicate","") for t in fixed)),
                "free":  dict(Counter(t.get("predicate","") for t in free)),
            },
        }
        report_path = paper_dir / KG_SUBDIR / "fixed_vs_free_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved → {report_path}")

    return {"overlap": overlap, "coverage": coverage}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare fixed-subject vs free triple extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kg_fixed_compare.py --paper BERT --entity-csv Entity_-_BERTv2.csv
  python kg_fixed_compare.py --paper KAN  --entity-csv Entity_-_KANv2.csv
  python kg_fixed_compare.py --paper BERT --entity-csv Entity_-_BERTv2.csv --save-report
        """
    )
    parser.add_argument("--paper", required=True,
        help="Paper folder name under output/")
    parser.add_argument("--entity-csv", required=True, metavar="CSV_PATH",
        help="Path to entity CSV file (e.g. Entity_-_BERTv2.csv)")
    parser.add_argument("--save-report", action="store_true",
        help="Save comparison report as JSON")
    return parser.parse_args()


def main():
    args      = parse_args()
    paper_dir = OUTPUT_DIR / args.paper
    entity_csv = Path(args.entity_csv)

    if not paper_dir.exists():
        logger.error(f"Paper not found: {paper_dir}")
        sys.exit(1)

    if not entity_csv.exists():
        logger.error(f"Entity CSV not found: {entity_csv}")
        sys.exit(1)

    run_comparison(
        paper_dir=paper_dir,
        entity_csv=entity_csv,
        save_report=args.save_report,
    )


if __name__ == "__main__":
    main()