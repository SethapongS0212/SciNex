"""
kg_compare.py
-------------
Compare triple extraction results across different extractors (REBEL, ITER, LLM).
Reads from already-generated output directories and produces a side-by-side report.

Usage:
    python kg_compare.py --paper 2205.11361
    python kg_compare.py --paper 2205.11361 --extractors rebel llm
    python kg_compare.py --all
    python kg_compare.py --all --save-report
"""

import argparse
import json
import logging
import sys
from collections import Counter
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

def find_extractor_dirs(paper_dir: Path) -> dict[str, Path]:
    """
    Find all extractor output directories for a paper.
    Handles both flat (kg/rebel/) and nested (kg/llm/ModelName/) layouts.

    Returns: {"rebel": Path(...), "llm": Path(...), ...}
    """
    kg_dir = paper_dir / KG_SUBDIR
    if not kg_dir.exists():
        return {}

    extractors = {}
    for child in sorted(kg_dir.iterdir()):
        if not child.is_dir():
            continue

        # Flat layout: kg/rebel/triples.json
        if (child / "triples.json").exists():
            extractors[child.name] = child

        # Nested layout: kg/llm/ModelName/triples.json
        else:
            for subchild in sorted(child.iterdir()):
                if subchild.is_dir() and (subchild / "triples.json").exists():
                    # Key: "llm/ModelName"
                    extractors[f"{child.name}/{subchild.name}"] = subchild

    return extractors


def load_triples(triples_path: Path) -> list[dict]:
    """Load triples.json file."""
    try:
        with open(triples_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {triples_path}: {e}")
        return []


def load_stats(stats_path: Path) -> dict:
    """Load kg_stats.json file."""
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_triples(triples: list[dict]) -> dict:
    """Compute analysis metrics from a list of triples."""
    if not triples:
        return {
            "count": 0,
            "unique_subjects": 0,
            "unique_objects": 0,
            "unique_predicates": 0,
            "unique_entities": 0,
            "top_predicates": [],
            "top_entities": [],
            "has_entity_types": False,
            "entity_type_counts": {},
            "sections_covered": [],
            "avg_subject_len": 0,
            "avg_object_len": 0,
            "sample_triples": [],
        }

    subjects   = [t.get("subject", "") for t in triples]
    objects    = [t.get("object", "")  for t in triples]
    predicates = [t.get("predicate", "") for t in triples]
    entities   = set(subjects) | set(objects)

    # Entity types (ITER and LLM have these, REBEL doesn't)
    has_types = any("subject_type" in t for t in triples)
    type_counts = {}
    if has_types:
        all_types = (
            [t.get("subject_type", "Unknown") for t in triples] +
            [t.get("object_type",  "Unknown") for t in triples]
        )
        type_counts = dict(Counter(all_types).most_common(10))

    # Sections covered
    sections = list(dict.fromkeys(
        t.get("section", "") for t in triples if t.get("section")
    ))

    # Sample triples — pick diverse ones across sections
    seen_preds = set()
    samples = []
    for t in triples:
        pred = t.get("predicate", "")
        if pred not in seen_preds:
            samples.append(t)
            seen_preds.add(pred)
        if len(samples) >= 5:
            break

    return {
        "count":              len(triples),
        "unique_subjects":    len(set(subjects)),
        "unique_objects":     len(set(objects)),
        "unique_predicates":  len(set(predicates)),
        "unique_entities":    len(entities),
        "top_predicates":     Counter(predicates).most_common(8),
        "top_entities":       Counter(subjects + objects).most_common(8),
        "has_entity_types":   has_types,
        "entity_type_counts": type_counts,
        "sections_covered":   sections[:8],
        "avg_subject_len":    sum(len(s.split()) for s in subjects) / len(subjects),
        "avg_object_len":     sum(len(o.split()) for o in objects)  / len(objects),
        "sample_triples":     samples,
    }


# ── Report printing ───────────────────────────────────────────────────────────

def _bar(value: int, max_val: int, width: int = 20) -> str:
    """Simple ASCII bar chart."""
    if max_val == 0:
        return "░" * width
    filled = int((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)


def print_comparison(paper_name: str, results: dict[str, dict]):
    """Print a formatted side-by-side comparison report."""
    extractors = list(results.keys())
    analyses   = {name: results[name]["analysis"] for name in extractors}

    sep  = "═" * 70
    sep2 = "─" * 70

    print(f"\n{sep}")
    print(f"  KNOWLEDGE GRAPH COMPARISON — {paper_name}")
    print(f"{sep}")

    # ── Overview table ────────────────────────────────────────────────────────
    print(f"\n{'Metric':<28}", end="")
    for name in extractors:
        print(f"{name:>18}", end="")
    print()
    print(sep2)

    metrics = [
        ("Total triples",       "count"),
        ("Unique predicates",   "unique_predicates"),
        ("Unique entities",     "unique_entities"),
        ("Unique subjects",     "unique_subjects"),
        ("Unique objects",      "unique_objects"),
        ("Sections covered",    None),
        ("Avg subject length",  None),
        ("Entity types?",       None),
    ]

    for label, key in metrics:
        print(f"  {label:<26}", end="")
        for name in extractors:
            a = analyses[name]
            if key:
                val = str(a.get(key, 0))
            elif label == "Sections covered":
                val = str(len(a.get("sections_covered", [])))
            elif label == "Avg subject length":
                val = f"{a.get('avg_subject_len', 0):.1f} words"
            elif label == "Entity types?":
                val = "Yes" if a.get("has_entity_types") else "No"
            print(f"{val:>18}", end="")
        print()

    # ── Triple count bar chart ────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  TRIPLE COUNT")
    print(sep2)
    max_count = max((analyses[n]["count"] for n in extractors), default=1)
    for name in extractors:
        count = analyses[name]["count"]
        bar   = _bar(count, max_count)
        print(f"  {name:<20} {bar} {count}")

    # ── Top predicates per extractor ──────────────────────────────────────────
    print(f"\n{sep2}")
    print("  TOP RELATIONS (predicate, count)")
    print(sep2)
    for name in extractors:
        print(f"\n  [{name.upper()}]")
        top = analyses[name]["top_predicates"]
        if not top:
            print("    (no triples)")
        for pred, count in top[:6]:
            print(f"    {pred:<35} {count}")

    # ── Entity types (if available) ───────────────────────────────────────────
    typed = [n for n in extractors if analyses[n]["has_entity_types"]]
    if typed:
        print(f"\n{sep2}")
        print("  ENTITY TYPE DISTRIBUTION")
        print(sep2)
        for name in typed:
            print(f"\n  [{name.upper()}]")
            for etype, count in analyses[name]["entity_type_counts"].items():
                print(f"    {etype:<35} {count}")

    # ── Sample triples ────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  SAMPLE TRIPLES")
    print(sep2)
    for name in extractors:
        print(f"\n  [{name.upper()}]")
        samples = analyses[name]["sample_triples"]
        if not samples:
            print("    (no triples)")
            continue
        for t in samples:
            subj = t.get("subject", "")[:35]
            pred = t.get("predicate", "")[:25]
            obj  = t.get("object", "")[:35]
            sec  = t.get("section", "")[:20]
            print(f"    ({subj})  --[{pred}]-->  ({obj})")
            if sec:
                print(f"      ↳ Section: {sec}")

    print(f"\n{sep}\n")


def build_report_dict(paper_name: str, results: dict) -> dict:
    """Build a JSON-serializable report dictionary."""
    report = {"paper": paper_name, "extractors": {}}
    for name, data in results.items():
        a = data["analysis"]
        report["extractors"][name] = {
            "count":             a["count"],
            "unique_predicates": a["unique_predicates"],
            "unique_entities":   a["unique_entities"],
            "top_predicates":    a["top_predicates"],
            "has_entity_types":  a["has_entity_types"],
            "entity_type_counts": a["entity_type_counts"],
            "sections_covered":  a["sections_covered"],
        }
    return report


# ── Core ──────────────────────────────────────────────────────────────────────

def compare_paper(
    paper_dir: Path,
    filter_extractors: list[str] | None = None,
    save_report: bool = False,
) -> dict | None:
    """Compare all extractor outputs for one paper."""
    paper_name = paper_dir.name
    extractor_dirs = find_extractor_dirs(paper_dir)

    if not extractor_dirs:
        logger.warning(f"No KG outputs found for '{paper_name}'. Run kg_main.py first.")
        return None

    # Filter if requested
    if filter_extractors:
        extractor_dirs = {
            k: v for k, v in extractor_dirs.items()
            if any(f in k for f in filter_extractors)
        }

    if not extractor_dirs:
        logger.warning(f"No matching extractors found for '{paper_name}'.")
        return None

    # Load and analyze
    results = {}
    for name, ext_dir in extractor_dirs.items():
        triples = load_triples(ext_dir / "triples.json")
        stats   = load_stats(ext_dir / "kg_stats.json")
        results[name] = {
            "triples":  triples,
            "stats":    stats,
            "analysis": analyze_triples(triples),
        }

    # Print report
    print_comparison(paper_name, results)

    # Save report
    if save_report:
        report      = build_report_dict(paper_name, results)
        report_path = paper_dir / KG_SUBDIR / "comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved → {report_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare KG extraction results across REBEL, ITER, and LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kg_compare.py --paper 2205.11361
  python kg_compare.py --paper 2205.11361 --extractors rebel llm
  python kg_compare.py --all
  python kg_compare.py --all --save-report
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paper", metavar="PAPER_NAME",
        help="Single paper to compare (e.g. 2205.11361)")
    group.add_argument("--all", action="store_true",
        help="Compare all papers in output/")

    parser.add_argument("--extractors", nargs="+", metavar="NAME",
        help="Filter to specific extractors e.g. --extractors rebel llm")
    parser.add_argument("--limit", type=int, default=None,
        help="When using --all, process only first N papers")
    parser.add_argument("--save-report", action="store_true",
        help="Save comparison_report.json to each paper's kg/ folder")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        if not OUTPUT_DIR.exists():
            logger.error(f"Output directory '{OUTPUT_DIR}' not found.")
            sys.exit(1)
        papers = sorted([
            d for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and (d / KG_SUBDIR).exists()
        ])
        if args.limit:
            papers = papers[:args.limit]
    else:
        paper_dir = OUTPUT_DIR / args.paper
        if not paper_dir.exists():
            logger.error(f"Paper not found: {paper_dir}")
            sys.exit(1)
        papers = [paper_dir]

    if not papers:
        logger.error("No papers with KG outputs found. Run kg_main.py first.")
        sys.exit(1)

    logger.info(f"Comparing {len(papers)} paper(s)...")

    for paper_dir in papers:
        compare_paper(
            paper_dir=paper_dir,
            filter_extractors=args.extractors,
            save_report=args.save_report,
        )


if __name__ == "__main__":
    main()