#!/usr/bin/env python3
"""
scale_finalize.py
-----------------
Session 39 (2026-11-12): after scale_acquire.py, fix the scale-up corpus and its split ONCE.

  usable new paper = parsed HTML (>= --min-html-kb) + a v2 entity list with >= --min-entities
  test  = the 41 legacy fold-0 papers + the first --n-test usable new papers in candidate order
          (scale/candidates.txt is already a seeded random order with the corpus's venue mix)
  train = the 764 legacy training papers + every other usable new paper, in candidate order
  curve = nested training subsets: the legacy 764, then the first K new training papers added

Writes (refuses to overwrite an existing split — papers may already have been trained on):
  ft_data/scale/split.json       2 folds: 0 = test, 1 = train (load_split / build_ft_data format)
  papers_scale.txt               the whole corpus (test + train)
  scale/test_papers.txt, scale/train_papers.txt
  scale/curve_<N>.txt            corpus file for a curve point with N training papers
                                 (test + first N training papers), for build_ft_data --papers-file

    python scale_finalize.py --n-test 1000 --curve 1500 3000
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="scale/candidates.txt")
    ap.add_argument("--legacy-split", default="ft_data/acl/split.json")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--min-entities", type=int, default=10)
    ap.add_argument("--min-html-kb", type=float, default=8.0)
    ap.add_argument("--curve", nargs="*", type=int, default=[1500, 3000])
    ap.add_argument("--out-split", default="ft_data/scale/split.json")
    args = ap.parse_args()

    out = ROOT / args.out_split
    if out.exists():
        sys.exit(f"{out} exists — the scale split is fixed once. Move it aside deliberately to redo.")
    legacy = json.loads((ROOT / args.legacy_split).read_text(encoding="utf-8"))["papers"]
    legacy_corpus = (ROOT / "papers_acl.txt").read_text(encoding="utf-8").split()
    legacy_test = sorted(p for p in legacy_corpus if legacy[p]["fold"] == 0)
    legacy_train = sorted(p for p in legacy_corpus if legacy[p]["fold"] != 0)

    usable, why = [], {"no_html": 0, "small_html": 0, "no_v2_list": 0, "few_entities": 0}
    for p in (ROOT / args.candidates).read_text(encoding="utf-8").split():
        h = ROOT / "output" / p / "no-llm" / "output.html"
        v2 = ROOT / "output" / "acl_v2" / p / f"Entity_{p}_enriched.csv"
        if not h.exists():
            why["no_html"] += 1
        elif h.stat().st_size < args.min_html_kb * 1024:
            why["small_html"] += 1
        elif not v2.exists():
            why["no_v2_list"] += 1
        elif sum(1 for _ in v2.open(encoding="utf-8")) - 1 < args.min_entities:
            why["few_entities"] += 1
        else:
            usable.append(p)
    if len(usable) < args.n_test + max(args.curve or [0]):
        sys.exit(f"only {len(usable)} usable new papers ({why}) — not enough for the plan")
    new_test, new_train = usable[:args.n_test], usable[args.n_test:]
    test, train = legacy_test + new_test, legacy_train + new_train

    today = date.today().isoformat()
    split = {"version": 1, "folds": 2, "seed": 2026, "created": today,
             "note": "scale-up (scale_finalize.py): fold 0 = test, fold 1 = train",
             "papers": {**{p: {"fold": 0, "stratum": "legacy" if p in legacy else "new", "assigned": today} for p in test},
                        **{p: {"fold": 1, "stratum": "legacy" if p in legacy else "new", "assigned": today} for p in train}}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1), encoding="utf-8")
    (ROOT / "papers_scale.txt").write_text("\n".join(test + train) + "\n", encoding="utf-8")
    (ROOT / "scale/test_papers.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    (ROOT / "scale/train_papers.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    for n in args.curve:
        if n <= len(train):
            (ROOT / f"scale/curve_{n}.txt").write_text("\n".join(test + train[:n]) + "\n", encoding="utf-8")
    print(f"usable new papers {len(usable)}; excluded {why}")
    print(f"test {len(test)} ({len(legacy_test)} legacy + {len(new_test)} new); "
          f"train {len(train)} ({len(legacy_train)} legacy + {len(new_train)} new)")
    print(f"curve files: {[n for n in args.curve if n <= len(train)]} + full ({len(train)})")


if __name__ == "__main__":
    main()
