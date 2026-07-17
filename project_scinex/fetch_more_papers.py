#!/usr/bin/env python3
"""
fetch_more_papers.py — scale the corpus by downloading more ACL Anthology PDFs.

Selects recent main-conference papers (ACL/EMNLP/NAACL long papers, 2018-2022)
from acl_title_index.json that we DON'T already have, and downloads their PDFs
into output/acl/pdfs/. Does NOT parse — run run_acl_batch.py afterwards.

Usage:
    python3 fetch_more_papers.py --count 50
"""
import argparse
import glob
import json
import os
import random
import re
from pathlib import Path

from acl_pipeline import download_pdf

INDEX   = "acl_title_index.json"
PDF_DIR = Path("output/acl/pdfs")

# Main-conference long papers (volume 1): old-style P/D/N + new-style *-main.
_OLD = re.compile(r"^[PDN](18|19)-1\d{3}$")
_NEW = re.compile(r"^(2018|2019|2020|2021|2022)\.(acl|emnlp|naacl)-main\.\d+$")


def _venue(pid: str) -> str:
    if pid.startswith("P") or ".acl-" in pid:
        return "ACL"
    if pid.startswith("D") or ".emnlp-" in pid:
        return "EMNLP"
    return "NAACL"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=50, help="How many new papers to fetch")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between downloads")
    ap.add_argument("--seed", type=int, default=17, help="Selection seed (reproducible)")
    args = ap.parse_args()

    idx = json.load(open(INDEX))
    ids = set(idx.values())
    inv = {v: k for k, v in idx.items()}
    have = {os.path.basename(f)[:-4] for f in glob.glob(str(PDF_DIR / "*.pdf"))}

    cands = sorted(i for i in ids if (_OLD.match(i) or _NEW.match(i)) and i not in have)

    # Even spread across the three venues.
    by_venue = {"ACL": [], "EMNLP": [], "NAACL": []}
    for i in cands:
        by_venue[_venue(i)].append(i)
    rng = random.Random(args.seed)
    pick = []
    per = max(1, args.count // 3 + 1)
    for v in ("ACL", "EMNLP", "NAACL"):
        lst = by_venue[v][:]
        rng.shuffle(lst)
        pick += lst[:per]
    rng.shuffle(pick)
    pick = pick[:args.count]

    print(f"{len(cands)} main-conf candidates not yet downloaded; selecting {len(pick)}")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for pid in pick:
        try:
            if download_pdf(pid, PDF_DIR, delay=args.delay):
                ok += 1
                print(f"  ✔ {pid:24} {inv.get(pid, '')[:55]}")
            else:
                print(f"  ✗ {pid:24} (download failed)")
        except Exception as e:
            print(f"  ✗ {pid:24} {e}")
    print(f"\nDownloaded {ok}/{len(pick)} new PDFs → {PDF_DIR}")
    print("Next: run_acl_batch.py --timeout 1200  (parse), then enrich + fixed.")


if __name__ == "__main__":
    main()
