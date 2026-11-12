#!/usr/bin/env python3
"""
scale_select.py
---------------
Session 39 (2026-11-12): pick the NEW ACL Anthology papers for the scale-up (user: "5000+
papers, test set at least 1000+"). Candidates = acl_title_index.json ids from 2010-2023, minus
the 805 papers of papers_acl.txt, minus front matter. They are drawn so that the venue-kind mix
(main conference / Findings / workshop / LREC / SemEval / ...) matches the 805-paper corpus,
in a seeded random order. More candidates than needed are written, because some PDFs fail to
download or parse; scale_finalize.py later keeps the usable ones in this order.

    python scale_select.py --count 7500 --out scale/candidates.txt
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def year(pid):
    m = re.match(r"(\d{4})\.", pid)
    if m:
        return int(m.group(1))
    m = re.match(r"[A-Z](\d\d)-", pid)
    if m:
        y = int(m.group(1))
        return 2000 + y if y < 50 else 1900 + y
    return None


def kind(pid):
    """Coarse venue kind, the same for old-style (P19-1001) and new-style (2020.acl-main.1) ids."""
    m = re.match(r"\d{4}\.([a-z]+)-(\w+)\.", pid)
    if m:
        venue, vol = m.groups()
        if venue == "findings":
            return "findings"
        if venue in ("acl", "emnlp", "naacl", "eacl", "aacl", "coling", "conll") and vol in ("main", "long", "short"):
            return "main"
        if venue in ("lrec", "semeval", "wmt"):
            return venue
        return "workshop/other"
    m = re.match(r"([A-Z])\d\d-(\d)", pid)
    if m:
        letter, vol = m.groups()
        if letter in "PDNEK" and vol in "12":
            return "main"
        return {"L": "lrec", "S": "semeval", "W": "workshop/other"}.get(letter, "workshop/other")
    return "workshop/other"


def is_front_matter(pid, title):
    if re.search(r"\.0$", pid):
        return True
    m = re.match(r"[A-Z]\d\d-(\d+)$", pid)
    if m and int(m.group(1)) % 100 == 0:
        return True
    return bool(re.match(r"(proceedings of|front matter)", title.strip().lower()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="acl_title_index.json")
    ap.add_argument("--existing", default="papers_acl.txt")
    ap.add_argument("--count", type=int, default=7500)
    ap.add_argument("--years", nargs=2, type=int, default=[2010, 2023])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="scale/candidates.txt")
    args = ap.parse_args()

    title_of = {pid: t for t, pid in json.load(open(args.index, encoding="utf-8")).items()}
    existing = set(Path(args.existing).read_text(encoding="utf-8").split())
    target_mix = Counter(kind(p) for p in existing)
    pool = defaultdict(list)
    for pid, t in title_of.items():
        y = year(pid)
        if pid in existing or y is None or not (args.years[0] <= y <= args.years[1]):
            continue
        if is_front_matter(pid, t):
            continue
        pool[kind(pid)].append(pid)

    rng = random.Random(args.seed)
    for k in pool:
        pool[k].sort()
        rng.shuffle(pool[k])
    # draw kinds in proportion to the existing corpus; stop a kind when its pool runs out
    total = sum(target_mix.values())
    weights = {k: target_mix[k] / total for k in target_mix if pool.get(k)}
    picked, used = [], Counter()
    while len(picked) < args.count and weights:
        k = rng.choices(list(weights), weights=list(weights.values()))[0]
        if used[k] >= len(pool[k]):
            weights.pop(k)
            continue
        picked.append(pool[k][used[k]])
        used[k] += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"existing corpus mix: {dict(target_mix)}")
    print(f"pool sizes: { {k: len(v) for k, v in pool.items()} }")
    print(f"picked {len(picked)}: {dict(Counter(kind(p) for p in picked))}")
    print(f"years: {sorted(Counter(year(p) for p in picked).items())}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
