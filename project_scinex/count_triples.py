#!/usr/bin/env python3
"""count_triples.py — triple counts per paper / extractor / model, plus a predicate breakdown."""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
rows, per_ext = [], {}

for tf in sorted(root.glob("*/kg/*/*/triples.json")):
    paper = tf.parts[-5]
    ext, model = tf.parts[-3], tf.parts[-2]
    try:
        triples = json.load(open(tf, encoding="utf-8"))
    except Exception as e:
        rows.append((paper, ext, model, f"unreadable: {e}"))
        continue
    rows.append((paper, ext, model, len(triples)))
    b = per_ext.setdefault((ext, model), Counter())
    for t in triples:
        b[str(t.get("predicate", "?"))] += 1

if not rows:
    print(f"No triples.json under {root}/*/kg/*/*/ — nothing extracted yet.")
    sys.exit(0)

print(f"{'paper':20}{'extractor':18}{'model':26}{'triples':>8}")
print("-" * 74)
for paper, ext, model, n in rows:
    print(f"{paper:20}{ext:18}{model:26}{n:>8}")

print("\n=== predicate breakdown ===")
for (ext, model), c in sorted(per_ext.items()):
    tot = sum(c.values())
    print(f"\n{ext} / {model}  ({tot} triples, {len(c)} distinct predicates)")
    for pred, n in c.most_common():
        print(f"    {pred:24} {n:5}  {n / tot:5.1%}")
