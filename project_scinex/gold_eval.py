"""
gold_eval.py
------------
Human (and cross-model) evaluation of fixed-extraction triples, and agreement
against the LLM-as-Judge verdicts from kg_evaluate.py.

Three subcommands:

  export   sample triples into a CSV with a blank `verdict` column for a human —
           or another model — to fill in. Each row carries the predicate's ontology
           definition, the same one the judge sees. The judge's own verdict is
           deliberately NOT included, so the labeller is not anchored by it.

  agree    join a filled label CSV back onto the judge verdicts (by triple id) and
           report raw agreement, Cohen's kappa, a confusion matrix, per-predicate
           agreement and every disagreement. `--b` compares two label files instead
           (human vs Claude, 7B judge vs 14B judge, annotator A vs B).

  errors   group the judge's PARTIAL/INCORRECT verdicts by predicate with examples —
           the worklist for the next round of prompt/guard fixes.

Usage:
    python3 gold_eval.py export --extractor fixed fixed_scinex --n 150 --out gold/sample.csv
    python3 gold_eval.py agree  --gold gold/sample_filled.csv --out gold/agreement.json
    python3 gold_eval.py agree  --gold gold/human.csv --b gold/claude.csv
    python3 gold_eval.py errors --extractor fixed --top 15
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kg_evaluate import schema_for, triple_id

OUTPUT_DIR = Path("output")
KG_SUBDIR  = "kg"

VERDICTS = ("CORRECT", "PARTIAL", "INCORRECT")

CSV_FIELDS = [
    "triple_id", "paper", "extractor", "model",
    "subject", "predicate", "object", "object_type",
    "predicate_definition", "section", "source_sentence",
    "verdict", "note",
]

_VERDICT_ALIASES = {
    "C": "CORRECT", "1": "CORRECT", "Y": "CORRECT", "YES": "CORRECT", "OK": "CORRECT",
    "P": "PARTIAL", "0.5": "PARTIAL", "HALF": "PARTIAL",
    "I": "INCORRECT", "0": "INCORRECT", "N": "INCORRECT", "NO": "INCORRECT", "X": "INCORRECT",
}


def normalise_verdict(raw: str) -> str | None:
    """Accept CORRECT/PARTIAL/INCORRECT plus the shorthands a human actually types."""
    if raw is None:
        return None
    v = str(raw).strip().upper()
    if not v:
        return None
    v = _VERDICT_ALIASES.get(v, v)
    return v if v in VERDICTS else None


# ── Loading triples ───────────────────────────────────────────────────────────

def iter_extraction_dirs(extractors: list[str], papers: list[str] | None):
    """Yield (paper, extractor_family, model_slug, dir) for every extraction on disk."""
    if not OUTPUT_DIR.exists():
        sys.exit(f"Output directory not found: {OUTPUT_DIR.resolve()}")

    for paper_dir in sorted(OUTPUT_DIR.iterdir()):
        if not paper_dir.is_dir() or paper_dir.name == "acl":
            continue
        if papers and paper_dir.name not in papers:
            continue
        for ext in extractors:
            ext_dir = paper_dir / KG_SUBDIR / ext
            if not ext_dir.is_dir():
                continue
            if (ext_dir / "triples.json").exists():
                yield paper_dir.name, ext, "", ext_dir
            for model_dir in sorted(ext_dir.iterdir()):
                if model_dir.is_dir() and (model_dir / "triples.json").exists():
                    yield paper_dir.name, ext, model_dir.name, model_dir


def load_triples(extractors: list[str], papers: list[str] | None,
                 model: str | None, ontology_file: str = "scinex_refined_14.owl") -> list[dict]:
    rows = []
    for paper, ext, model_slug, ext_dir in iter_extraction_dirs(extractors, papers):
        if model and model_slug and model_slug != model:
            continue
        try:
            triples = json.loads((ext_dir / "triples.json").read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! could not read {ext_dir / 'triples.json'}: {e}")
            continue
        # The labeller must see the same predicate definition the judge sees, or the
        # two are answering different questions and agreement means nothing.
        schema = schema_for(ext, ontology_file) or {}
        definitions = {k.lower(): v for k, v in schema.items()}
        for t in triples:
            t.setdefault("paper", paper)
            rows.append({
                "triple_id":       triple_id(t, ext),
                "paper":           paper,
                "extractor":       ext,
                "model":           model_slug,
                "subject":         t.get("subject", ""),
                "predicate":       t.get("predicate", ""),
                "object":          t.get("object", ""),
                "object_type":     t.get("object_type", ""),
                "predicate_definition": definitions.get(
                    str(t.get("predicate", "")).strip().lower(), ""),
                "section":         t.get("section", ""),
                "source_sentence": t.get("source_sentence", ""),
            })
    return rows


# ── export ────────────────────────────────────────────────────────────────────

def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin over (extractor, predicate) buckets.

    Proportional sampling would spend the whole budget on `uses`/`achieves` and never
    show the rare predicates, which are exactly where the extractor's guards are
    weakest — so every bucket contributes before any bucket contributes twice.
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        buckets[(r["extractor"], r["predicate"])].append(r)
    for b in buckets.values():
        rng.shuffle(b)

    keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    picked, cursor = [], 0
    while len(picked) < n and any(buckets.values()):
        key = keys[cursor % len(keys)]
        cursor += 1
        if buckets[key]:
            picked.append(buckets[key].pop())
    return picked


def cmd_export(args):
    rows = load_triples(args.extractor, args.papers, args.model, args.ontology_file)
    if not rows:
        sys.exit("No triples found. Run kg_main.py --extractor fixed first.")

    by_ext = Counter(r["extractor"] for r in rows)
    print(f"Found {len(rows)} triples: " + ", ".join(f"{k}={v}" for k, v in by_ext.items()))

    rng = random.Random(args.seed)
    sample = rows if args.n >= len(rows) else stratified_sample(rows, args.n, rng)
    sample.sort(key=lambda r: (r["paper"], r["extractor"], r["predicate"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in sample:
            w.writerow({**r, "verdict": "", "note": ""})

    covered = Counter((r["extractor"], r["predicate"]) for r in sample)
    print(f"\nWrote {len(sample)} rows → {out_path}")
    print(f"Predicate coverage ({len(covered)} extractor/predicate buckets):")
    for (ext, pred), c in covered.most_common():
        print(f"   {ext:<14} {pred:<18} {c}")
    print("""
Fill the `verdict` column with CORRECT / PARTIAL / INCORRECT (C / P / I also work).
Use the SAME rubric the judge is given, or agreement measures the rubric gap, not quality:
  CORRECT   — the source sentence explicitly states the fact AND the predicate fits its
              `predicate_definition` (domain → range)
  PARTIAL   — the fact is implied rather than stated, or the predicate is a loose fit
              (e.g. the object is not the type the range calls for)
  INCORRECT — unsupported by the sentence, hallucinated, or subject/object swapped
              relative to the predicate's domain → range direction
Judge only from `source_sentence` — not from what you know about the paper.""")
    print(f"\nThen: python3 gold_eval.py agree --gold {out_path}")


# ── agreement ─────────────────────────────────────────────────────────────────

def load_labels(csv_path: Path) -> dict[str, dict]:
    """Read a filled label CSV → {triple_id: row}, keeping only labelled rows."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and "triple_id" not in rows[0]:
        sys.exit(f"{csv_path} has no `triple_id` column — labels can't be joined. "
                 f"Columns found: {list(rows[0])}")
    out = {}
    for r in rows:
        v = normalise_verdict(r.get("verdict"))
        if v:
            out[r["triple_id"]] = {**r, "verdict": v}
    if not out:
        sys.exit(f"No usable verdicts in {csv_path} — the `verdict` column is empty.")
    print(f"{csv_path}: {len(out)}/{len(rows)} rows labelled")
    return out


def load_judge_verdicts(rows: list[dict]) -> dict[str, str]:
    """Judge verdicts for the labelled triples, read from each evaluation.json."""
    wanted_dirs = {(r["paper"], r["extractor"], r.get("model", "")) for r in rows}
    verdicts: dict[str, str] = {}
    for paper, ext, model_slug in sorted(wanted_dirs):
        ext_dir = OUTPUT_DIR / paper / KG_SUBDIR / ext / model_slug if model_slug \
            else OUTPUT_DIR / paper / KG_SUBDIR / ext
        path = ext_dir / "evaluation.json"
        if not path.exists():
            continue
        try:
            for t in json.loads(path.read_text(encoding="utf-8")):
                if t.get("triple_id") and t.get("verdict"):
                    verdicts[t["triple_id"]] = t["verdict"]
        except Exception as e:
            print(f"  ! could not read {path}: {e}")
    return verdicts


def cohens_kappa(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> float:
    """Chance-corrected agreement. 0 = chance, 1 = perfect, <0 = worse than chance."""
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in pairs if a == b) / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected = sum((a_counts[l] / n) * (b_counts[l] / n) for l in labels)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def agreement_report(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> dict:
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        matrix[a][b] += 1
    return {
        "n": n,
        "agreement": round(agree / n, 3) if n else 0,
        "kappa": round(cohens_kappa(pairs, labels), 3),
        "confusion": matrix,
    }


def print_matrix(matrix: dict, labels: tuple[str, ...], a_name: str, b_name: str):
    width = max(len(l) for l in labels) + 2
    print(f"    {'':<{width}}" + "".join(f"{l[:9]:>11}" for l in labels) + f"   ← {b_name}")
    for a in labels:
        row = "".join(f"{matrix[a][b]:>11}" for b in labels)
        print(f"    {a:<{width}}{row}")
    print(f"    ↑ {a_name}")


def cmd_agree(args):
    gold_path = Path(args.gold)
    a_labels = load_labels(gold_path)
    a_name = args.a_name or gold_path.stem

    if args.b:
        b_path = Path(args.b)
        b_raw = load_labels(b_path)
        b_labels = {k: v["verdict"] for k, v in b_raw.items()}
        b_name = args.b_name or b_path.stem
        if b_name == a_name:
            b_name = f"{b_name} (b)"
    else:
        b_labels = load_judge_verdicts(list(a_labels.values()))
        b_name = args.b_name or "llm-judge"
        if not b_labels:
            sys.exit("No judge verdicts found. Run kg_evaluate.py first, "
                     "or pass --b <other-labels.csv>.")

    # UNVERIFIABLE (judge could not read the triple) is not a verdict on quality —
    # scoring it against a human verdict would corrupt both kappa and the matrix.
    unscorable = sum(1 for tid in a_labels
                     if tid in b_labels and b_labels[tid] not in VERDICTS)
    common = [tid for tid in a_labels
              if tid in b_labels and b_labels[tid] in VERDICTS]
    missing = len(a_labels) - len(common) - unscorable
    if not common:
        sys.exit("No overlapping triple ids — were the labels made from a different "
                 "extraction run? Re-export after re-extracting.")

    pairs = [(a_labels[t]["verdict"], b_labels[t]) for t in common]
    overall = agreement_report(pairs, VERDICTS)

    # Binary collapse: the number that matters for "is this triple usable".
    def binary(v: str) -> str:
        return "CORRECT" if v == "CORRECT" else "NOT_CORRECT"
    bin_pairs = [(binary(a), binary(b)) for a, b in pairs]
    binary_report = agreement_report(bin_pairs, ("CORRECT", "NOT_CORRECT"))

    a_prec = sum(1 for a, _ in pairs if a == "CORRECT") / len(pairs)
    b_prec = sum(1 for _, b in pairs if b == "CORRECT") / len(pairs)

    print(f"\n{'═'*70}")
    print(f"  AGREEMENT — {a_name} vs {b_name}")
    print(f"{'═'*70}")
    print(f"  Compared            : {len(common)} triples"
          + (f"  ({missing} labelled but not found in {b_name})" if missing else "")
          + (f"  ({unscorable} UNVERIFIABLE, excluded)" if unscorable else ""))
    print(f"  Raw agreement       : {overall['agreement']*100:.1f}%")
    print(f"  Cohen's kappa (3)   : {overall['kappa']:.3f}")
    print(f"  Binary agreement    : {binary_report['agreement']*100:.1f}%   "
          f"(kappa {binary_report['kappa']:.3f})")
    print(f"  Precision — {a_name[:14]:<14}: {a_prec*100:.1f}%")
    print(f"  Precision — {b_name[:14]:<14}: {b_prec*100:.1f}%")
    print(f"{'─'*70}")
    print("  Confusion matrix:")
    print_matrix(overall["confusion"], VERDICTS, a_name, b_name)

    # Per-predicate and per-extractor breakdowns
    by_pred: dict[str, list] = defaultdict(list)
    by_ext: dict[str, list] = defaultdict(list)
    for tid in common:
        row = a_labels[tid]
        pair = (row["verdict"], b_labels[tid])
        by_pred[row.get("predicate", "")].append(pair)
        by_ext[row.get("extractor", "")].append(pair)

    print(f"{'─'*70}")
    print(f"  {'Extractor':<18}{'n':>5}{'agree':>9}{'kappa':>8}"
          f"{a_name[:10]:>12}{b_name[:10]:>12}   (precision)")
    for ext, ps in sorted(by_ext.items()):
        rep = agreement_report(ps, VERDICTS)
        pa = sum(1 for a, _ in ps if a == "CORRECT") / len(ps)
        pb = sum(1 for _, b in ps if b == "CORRECT") / len(ps)
        print(f"  {ext:<18}{len(ps):>5}{rep['agreement']*100:>8.1f}%{rep['kappa']:>8.2f}"
              f"{pa*100:>11.1f}%{pb*100:>11.1f}%")

    print(f"{'─'*70}")
    print(f"  {'Predicate':<20}{'n':>5}{'agree':>9}{a_name[:10]:>12}{b_name[:10]:>12}")
    for pred, ps in sorted(by_pred.items(), key=lambda kv: -len(kv[1])):
        rep = agreement_report(ps, VERDICTS)
        pa = sum(1 for a, _ in ps if a == "CORRECT") / len(ps)
        pb = sum(1 for _, b in ps if b == "CORRECT") / len(ps)
        print(f"  {pred:<20}{len(ps):>5}{rep['agreement']*100:>8.1f}%"
              f"{pa*100:>11.1f}%{pb*100:>11.1f}%")

    disagreements = [
        {
            "triple_id": tid,
            "paper":     a_labels[tid].get("paper", ""),
            "extractor": a_labels[tid].get("extractor", ""),
            "subject":   a_labels[tid].get("subject", ""),
            "predicate": a_labels[tid].get("predicate", ""),
            "object":    a_labels[tid].get("object", ""),
            "source_sentence": a_labels[tid].get("source_sentence", ""),
            a_name:      a_labels[tid]["verdict"],
            b_name:      b_labels[tid],
            "note":      a_labels[tid].get("note", ""),
        }
        for tid in common if a_labels[tid]["verdict"] != b_labels[tid]
    ]

    print(f"{'─'*70}")
    print(f"  DISAGREEMENTS ({len(disagreements)}) — showing {min(args.show, len(disagreements))}")
    for d in disagreements[:args.show]:
        print(f"\n  [{d['extractor']}] ({d['subject']}) --[{d['predicate']}]--> ({d['object']})")
        print(f"    {a_name}: {d[a_name]:<10}  {b_name}: {d[b_name]}")
        print(f"    src: {d['source_sentence'][:150]}")
        if d["note"]:
            print(f"    note: {d['note']}")
    print(f"\n{'═'*70}\n")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "a": a_name,
            "b": b_name,
            "compared": len(common),
            "unmatched": missing,
            "unscorable": unscorable,
            "overall": overall,
            "binary": binary_report,
            "precision": {a_name: round(a_prec, 3), b_name: round(b_prec, 3)},
            "by_extractor": {
                e: {**agreement_report(ps, VERDICTS),
                    "precision_a": round(sum(1 for a, _ in ps if a == "CORRECT") / len(ps), 3),
                    "precision_b": round(sum(1 for _, b in ps if b == "CORRECT") / len(ps), 3)}
                for e, ps in by_ext.items()
            },
            "by_predicate": {
                p: {**agreement_report(ps, VERDICTS),
                    "precision_a": round(sum(1 for a, _ in ps if a == "CORRECT") / len(ps), 3),
                    "precision_b": round(sum(1 for _, b in ps if b == "CORRECT") / len(ps), 3)}
                for p, ps in by_pred.items()
            },
            "disagreements": disagreements,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved → {out_path}")


# ── errors ────────────────────────────────────────────────────────────────────

def cmd_errors(args):
    """Group the judge's non-CORRECT verdicts by predicate — the prompt-fix worklist."""
    buckets: dict[tuple, list] = defaultdict(list)
    totals: Counter = Counter()

    for paper, ext, model_slug, ext_dir in iter_extraction_dirs(args.extractor, args.papers):
        path = ext_dir / "evaluation.json"
        if not path.exists():
            continue
        try:
            evaluated = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! could not read {path}: {e}")
            continue
        for t in evaluated:
            pred = str(t.get("predicate", "")).lower()
            totals[(ext, pred)] += 1
            if t.get("verdict") in ("PARTIAL", "INCORRECT"):
                buckets[(ext, pred, t["verdict"])].append({**t, "paper": paper})

    if not totals:
        sys.exit("No evaluation.json files found. Run kg_evaluate.py first.")

    ranked = sorted(
        {(e, p) for e, p, _ in buckets},
        key=lambda k: -(len(buckets.get((*k, "INCORRECT"), []))
                        + len(buckets.get((*k, "PARTIAL"), []))),
    )

    print(f"\n{'═'*70}")
    print("  JUDGE ERROR PROFILE — worst predicates first")
    print(f"{'═'*70}")
    print(f"  {'Extractor':<15}{'Predicate':<20}{'n':>5}{'INCORRECT':>11}{'PARTIAL':>9}{'err%':>8}")
    for ext, pred in ranked[:args.top]:
        n = totals[(ext, pred)]
        inc = len(buckets.get((ext, pred, "INCORRECT"), []))
        par = len(buckets.get((ext, pred, "PARTIAL"), []))
        print(f"  {ext:<15}{pred:<20}{n:>5}{inc:>11}{par:>9}{(inc + par) / n * 100:>7.1f}%")

    for ext, pred in ranked[:args.examples_for]:
        print(f"\n{'─'*70}")
        print(f"  {ext} / {pred} — examples")
        for verdict in ("INCORRECT", "PARTIAL"):
            for t in buckets.get((ext, pred, verdict), [])[:args.examples]:
                print(f"\n    [{verdict}] {t.get('paper')}: "
                      f"({t.get('subject')}) --[{pred}]--> ({t.get('object')})")
                print(f"      src   : {str(t.get('source_sentence', ''))[:150]}")
                print(f"      reason: {str(t.get('reason', ''))[:150]}")
    print(f"\n{'═'*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Human/cross-model evaluation of extracted triples and agreement "
                    "with the LLM judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="sample triples into a CSV for labelling")
    ex.add_argument("--extractor", nargs="+", default=["fixed", "fixed_scinex"],
                    help="Extractor families to sample from (default: fixed fixed_scinex)")
    ex.add_argument("--papers", nargs="+", default=None,
                    help="Restrict to these paper ids (default: all in output/)")
    ex.add_argument("--model", default=None,
                    help="Restrict to one model slug, e.g. Qwen3-14B")
    ex.add_argument("--n", type=int, default=150, help="Sample size (default: 150)")
    ex.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    ex.add_argument("--ontology-file", default="scinex_refined_14.owl",
                    help="Ontology file supplying scinex predicate definitions "
                         "(CEO definitions are built in)")
    ex.add_argument("--out", default="gold/sample.csv", help="Output CSV path")
    ex.set_defaults(func=cmd_export)

    ag = sub.add_parser("agree", help="compare labels against the judge (or another labeller)")
    ag.add_argument("--gold", required=True, help="Filled label CSV (side A)")
    ag.add_argument("--b", default=None,
                    help="Second label CSV (side B). Omit to compare against the "
                         "judge verdicts in evaluation.json")
    ag.add_argument("--a-name", default=None, help="Display name for side A")
    ag.add_argument("--b-name", default=None, help="Display name for side B")
    ag.add_argument("--show", type=int, default=15, help="Disagreements to print")
    ag.add_argument("--out", default=None, help="Write the full report to this JSON")
    ag.set_defaults(func=cmd_agree)

    er = sub.add_parser("errors", help="judge error profile by predicate")
    er.add_argument("--extractor", nargs="+", default=["fixed", "fixed_scinex"])
    er.add_argument("--papers", nargs="+", default=None)
    er.add_argument("--top", type=int, default=15, help="Predicates in the table")
    er.add_argument("--examples-for", type=int, default=5,
                    help="Show examples for the N worst predicates")
    er.add_argument("--examples", type=int, default=3, help="Examples per verdict")
    er.set_defaults(func=cmd_errors)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
