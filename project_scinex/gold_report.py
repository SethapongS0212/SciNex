"""
gold_report.py
--------------
Turn one or more filled gold-label CSVs (from `gold_eval.py export`, verdict column
completed by a human or by a model acting as judge) into the numbers the paper quotes:

  * sampled precision       — precision over the stratified sample as drawn
  * corpus-weighted precision — per-predicate precision re-weighted by how often each
                              predicate actually occurs in the corpus
  * per-predicate table     — the guard worklist
  * per-paper table         — which papers the extractor handles well

**Why two precisions.** `gold_eval.py export` samples round-robin over
extractor x predicate buckets, so a predicate contributing 1% of the corpus gets the same
number of labels as one contributing 25%. That is right for estimating *per-predicate*
accuracy and wrong for estimating *corpus* accuracy. Quote both, and say which is which.

Usage:
    python3 gold_report.py --labels gold/sample_claude.csv gold/sample_round2_claude.csv
    python3 gold_report.py --labels gold/*.csv --out gold/report.json
"""

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_DIR = Path("output")

STRICT = {"CORRECT"}
LENIENT = {"CORRECT", "PARTIAL"}


def load_labels(paths):
    rows, seen = [], {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            v = (r.get("verdict") or "").strip().upper()
            if v not in ("CORRECT", "PARTIAL", "INCORRECT"):
                continue
            tid = r["triple_id"]
            if tid in seen:          # same triple labelled twice — keep the first
                continue
            seen[tid] = p
            r["verdict"] = v
            r["source_csv"] = p
            rows.append(r)
    return rows


def corpus_predicate_counts(model_slug):
    """{extractor: Counter(predicate -> n)} over every triples.json on disk."""
    counts = defaultdict(Counter)
    for tf in OUTPUT_DIR.glob(f"*/kg/*/{model_slug}/triples.json"):
        extractor = tf.parent.parent.name
        try:
            data = json.loads(tf.read_text(encoding="utf-8"))
        except Exception:
            continue
        triples = data.get("triples", data) if isinstance(data, dict) else data
        for t in triples:
            p = (t.get("predicate") or "").strip().lower()
            if p:
                counts[extractor][p] += 1
    return counts


def precision(rows, accept):
    if not rows:
        return None
    return sum(r["verdict"] in accept for r in rows) / len(rows)


def pct(x):
    return "  n/a " if x is None else f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Filled gold CSVs (globs allowed). Duplicate triple_ids keep the first.")
    ap.add_argument("--model", default="qwen3-8b", help="Model slug on disk (default: qwen3-8b)")
    ap.add_argument("--out", default=None, help="Write the full report as JSON here")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Bootstrap resamples for a 95%% CI on the weighted numbers "
                         "(resamples labels WITHIN each predicate bucket, so the corpus "
                         "weights stay fixed and only the per-predicate estimates move)")
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap seed")
    args = ap.parse_args()

    paths = [p for pat in args.labels for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit("No label files matched.")
    rows = load_labels(paths)
    if not rows:
        raise SystemExit("No usable verdicts found.")

    corpus = corpus_predicate_counts(args.model)
    extractors = sorted({r["extractor"] for r in rows})

    print(f"Label files : {', '.join(paths)}")
    print(f"Labelled    : {len(rows)} triples over {len(extractors)} extractors\n")

    report = {"label_files": paths, "n_labelled": len(rows), "extractors": {}}

    # ── headline table ────────────────────────────────────────────────────────
    print(f"{'extractor':17s} {'n':>4s}  {'sampled':>8s} {'lenient':>8s}   "
          f"{'WEIGHTED':>8s} {'lenient':>8s}   {'cover':>6s}")
    for ex in extractors:
        ex_rows = [r for r in rows if r["extractor"] == ex]
        by_pred = defaultdict(list)
        for r in ex_rows:
            by_pred[r["predicate"].strip().lower()].append(r)

        dist = corpus[ex]
        total = sum(dist.values())
        covered = sum(n for p, n in dist.items() if p in by_pred)
        w_strict = w_lenient = 0.0
        for p, n in dist.items():
            if p not in by_pred:
                continue
            w = n / covered if covered else 0
            w_strict += w * precision(by_pred[p], STRICT)
            w_lenient += w * precision(by_pred[p], LENIENT)

        print(f"{ex:17s} {len(ex_rows):4d}  {pct(precision(ex_rows, STRICT))} "
              f"{pct(precision(ex_rows, LENIENT))}   {pct(w_strict)} {pct(w_lenient)}   "
              f"{pct(covered / total if total else None)}")

        report["extractors"][ex] = {
            "n_labelled": len(ex_rows),
            "corpus_triples": total,
            "sampled_strict": precision(ex_rows, STRICT),
            "sampled_lenient": precision(ex_rows, LENIENT),
            "weighted_strict": w_strict,
            "weighted_lenient": w_lenient,
            "weight_coverage": covered / total if total else None,
            "by_predicate": {
                p: {
                    "n": len(rs),
                    "correct": sum(r["verdict"] == "CORRECT" for r in rs),
                    "partial": sum(r["verdict"] == "PARTIAL" for r in rs),
                    "incorrect": sum(r["verdict"] == "INCORRECT" for r in rs),
                    "corpus_n": dist.get(p, 0),
                    "corpus_share": dist.get(p, 0) / total if total else None,
                    "strict": precision(rs, STRICT),
                    "lenient": precision(rs, LENIENT),
                }
                for p, rs in sorted(by_pred.items())
            },
        }

    # ── per-predicate detail ──────────────────────────────────────────────────
    for ex in extractors:
        d = report["extractors"][ex]["by_predicate"]
        print(f"\n{ex} — per predicate (sorted by corpus share)")
        print(f"  {'predicate':18s} {'n':>3s} {'C':>3s} {'P':>3s} {'I':>3s}  "
              f"{'strict':>7s} {'lenient':>7s}  {'corpus':>7s}")
        for p, s in sorted(d.items(), key=lambda kv: -(kv[1]["corpus_share"] or 0)):
            print(f"  {p:18s} {s['n']:3d} {s['correct']:3d} {s['partial']:3d} "
                  f"{s['incorrect']:3d}  {pct(s['strict'])} {pct(s['lenient'])}  "
                  f"{pct(s['corpus_share'])}")

    # ── per paper ─────────────────────────────────────────────────────────────
    print("\nper paper (all extractors pooled)")
    by_paper = defaultdict(list)
    for r in rows:
        by_paper[r["paper"]].append(r)
    print(f"  {'paper':18s} {'n':>3s}  {'strict':>7s} {'lenient':>7s}")
    for paper, rs in sorted(by_paper.items(), key=lambda kv: -precision(kv[1], STRICT)):
        print(f"  {paper:18s} {len(rs):3d}  {pct(precision(rs, STRICT))} "
              f"{pct(precision(rs, LENIENT))}")
        report.setdefault("by_paper", {})[paper] = {
            "n": len(rs),
            "strict": precision(rs, STRICT),
            "lenient": precision(rs, LENIENT),
        }

    print("\noverall verdict mix:", dict(Counter(r["verdict"] for r in rows)))

    # ── bootstrap CI on the weighted numbers ──────────────────────────────────
    if args.bootstrap:
        import random
        import statistics as st

        rnd = random.Random(args.seed)

        def weighted(sample_by_pred, dist):
            covered = sum(n for p, n in dist.items() if p in sample_by_pred)
            if not covered:
                return None, None
            s = l = 0.0
            for p, n in dist.items():
                rs = sample_by_pred.get(p)
                if not rs:
                    continue
                w = n / covered
                s += w * precision(rs, STRICT)
                l += w * precision(rs, LENIENT)
            return s, l

        buckets = {ex: defaultdict(list) for ex in extractors}
        for r in rows:
            buckets[r["extractor"]][r["predicate"].strip().lower()].append(r)

        draws = {ex: {"strict": [], "lenient": []} for ex in extractors}
        diffs = []
        for _ in range(args.bootstrap):
            per_ex = {}
            for ex in extractors:
                resampled = {p: [rnd.choice(rs) for _ in rs]
                             for p, rs in buckets[ex].items()}
                s, l = weighted(resampled, corpus[ex])
                per_ex[ex] = s
                draws[ex]["strict"].append(s)
                draws[ex]["lenient"].append(l)
            if len(extractors) == 2:
                diffs.append(per_ex[extractors[0]] - per_ex[extractors[1]])

        def ci(v):
            v = sorted(v)
            lo = v[int(0.025 * len(v))]
            hi = v[min(int(0.975 * len(v)), len(v) - 1)]
            return lo, hi

        print(f"\nbootstrap ({args.bootstrap} resamples, within-predicate, "
              f"corpus weights fixed)")
        for ex in extractors:
            slo, shi = ci(draws[ex]["strict"])
            llo, lhi = ci(draws[ex]["lenient"])
            print(f"  {ex:17s} weighted strict {pct(st.mean(draws[ex]['strict']))} "
                  f"[{pct(slo)},{pct(shi)}]   lenient {pct(st.mean(draws[ex]['lenient']))} "
                  f"[{pct(llo)},{pct(lhi)}]")
            report["extractors"][ex]["bootstrap"] = {
                "n": args.bootstrap,
                "weighted_strict_ci95": [slo, shi],
                "weighted_lenient_ci95": [llo, lhi],
            }
        if diffs:
            dlo, dhi = ci(diffs)
            frac = sum(d > 0 for d in diffs) / len(diffs)
            a, b = extractors
            print(f"  difference ({a} - {b}) strict: {pct(st.mean(diffs))} "
                  f"[{pct(dlo)},{pct(dhi)}]  P({a}>{b}) = {frac:.2f}")
            report["difference_strict"] = {
                "a": a, "b": b, "mean": st.mean(diffs),
                "ci95": [dlo, dhi], "p_a_greater": frac,
            }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
