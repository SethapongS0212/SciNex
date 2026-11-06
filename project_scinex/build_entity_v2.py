#!/usr/bin/env python3
"""
build_entity_v2.py
------------------
Cleaned fixed-entity lists for the ACL papers ("v2"), written NEXT TO the originals so every
earlier run stays reproducible:

    output/acl/<id>/Entity_<id>_enriched.csv      (v1, untouched)
    output/acl_v2/<id>/Entity_<id>_enriched.csv   (v2, this script)

Two changes, both from evidence (hands_off.md §38.3 and the 30-paragraph check in §38.6):
  1. DROP entries the audit classed GENERIC or NOT_A_CONCEPT ("training", "Models", "Example",
     "semantic", "evaluate") and per-paper CASE_COLLISIONs ("CompLex" where the paper only
     writes "complex"). REVIEW entries ("classifier", "parser", "Attention") are KEPT: they
     await the user's/professor's review of entity_cleanup/stoplist_draft.csv.
  2. ADD the paper's own names that the CS-NER gazetteer cannot know (EWISE, CluHTM, hredGAN,
     RE-NET, TinyBERT): a token with an inner capital or two or more capitals ("BERTBASE",
     "GPT-2"), used at least --min-count times in the paragraphs the extractor saw, not
     already listed (as entity, abbreviation or alias), not a venue/format acronym, a roman
     numeral or a table/figure label. At most --max-add per paper, most frequent first.

Deterministic (no model). Run from project_scinex:
    python build_entity_v2.py                      # all papers in papers_acl.txt
    python build_entity_v2.py --papers P19-1568 --show
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from entity_cleanup_audit import CASE_ALLOW, case_sensitive, category, collides, norm, paper_text

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "acl"
DST = ROOT / "output" / "acl_v2"
FIELDS = ["Entity", "Abbreviation", "Aliases", "TP", "NER_Type", "CEO_Type"]

TOKEN = re.compile(r"(?<![A-Za-z0-9\-])[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*(?![A-Za-z0-9])")
# acronyms that are venues, file formats, organisations, units or prose, not research entities
NOT_NAMES = {
    "ACL", "EMNLP", "NAACL", "EACL", "AACL", "COLING", "LREC", "TACL", "CONLL", "CoNLL", "IJCNLP",
    "AAAI", "IJCAI", "ICLR", "ICML", "NeurIPS", "NIPS", "CVPR", "ICCV", "ECCV", "SIGIR", "SIGDIAL",
    "INLG", "WMT", "SemEval", "IEEE", "ACM", "arXiv", "ArXiv", "PhD", "MSc", "BSc", "USA", "UK",
    "EU", "US", "PDF", "HTML", "XML", "JSON", "CSV", "URL", "URLs", "HTTP", "API", "APIs", "GPU",
    "GPUs", "CPU", "CPUs", "TPU", "TPUs", "RAM", "GB", "MB", "KB", "TB", "GHz", "MHz", "ms", "Hz",
    "LaTeX", "GitHub", "iPhone", "iOS", "MacBook", "YouTube", "LinkedIn", "PubMed", "OK", "TV",
    "AM", "PM", "UTC", "ID", "IDs", "No", "Fig", "Figs", "Tab", "Eq", "Eqs", "Sec", "Appx", "AI",
    "NLP", "CL", "ML", "DL", "IR", "CV", "MT", "QA", "NLU", "NLG",  # fields: too broad as subjects
    "MASK", "CLS", "SEP", "UNK", "PAD", "EOS", "BOS",                 # special tokens
}
ACRONYM_PLURAL = re.compile(r"^[A-Z]{2,}s$")                         # "KGs", "IDEs": plural nouns
ROMAN = re.compile(r"^(?=[IVXLC]+$)M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$")
LABEL = re.compile(r"^(Table|Figure|Fig|Section|Appendix|Eq|Equation)\d*$", re.I)


def is_name(tok):
    if (len(tok) < 3 or tok in NOT_NAMES or ROMAN.match(tok) or LABEL.match(tok)
            or ACRONYM_PLURAL.match(tok)):
        return False
    if tok.isupper() and len(tok) <= 3 and tok.isalpha():
        return False                       # 3-letter all-caps words: mostly prose acronyms
    caps = sum(c.isupper() for c in tok)
    return caps >= 2 or case_sensitive(tok)


def listed_forms(rows):
    forms = set()
    for r in rows:
        forms.add(norm(r["Entity"]))
        if (r.get("Abbreviation") or "").strip():
            forms.add(norm(r["Abbreviation"]))
        for a in (r.get("Aliases") or "").split("|"):
            if a.strip():
                forms.add(norm(a))
    return forms


def clean_paper(paper, min_count, max_add):
    src = SRC / paper / f"Entity_{paper}_enriched.csv"
    if not src.exists():
        return None
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    text = paper_text(ROOT, paper)
    kept, dropped = [], Counter()
    for r in rows:
        surface = r["Entity"].strip()
        k = norm(surface)
        cat = category(k)
        if cat in ("GENERIC", "NOT_A_CONCEPT"):
            dropped[cat] += 1
            continue
        if cat == "KEEP" and case_sensitive(surface) and k not in CASE_ALLOW and text \
                and collides(surface, text):
            dropped["CASE_COLLISION"] += 1
            continue
        kept.append(r)
    forms = listed_forms(rows)                 # never re-add something v1 listed (even dropped)
    counts = Counter(t for t in TOKEN.findall(text) if is_name(t))
    added = []
    for tok, n in counts.most_common():
        if n < min_count or len(added) >= max_add:
            break
        if norm(tok) in forms or category(norm(tok)) != "KEEP":
            continue
        forms.add(norm(tok))
        added.append((tok, n))
        kept.append({"Entity": tok, "Abbreviation": "", "Aliases": "", "TP": "1",
                     "NER_Type": "added_v2", "CEO_Type": ""})
    return {"rows": kept, "n_v1": len(rows), "dropped": dict(dropped), "added": added}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", nargs="*")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--max-add", type=int, default=25)
    ap.add_argument("--show", action="store_true", help="print each paper's changes")
    ap.add_argument("--summary", default="entity_v2/summary.json")
    args = ap.parse_args()
    papers = args.papers or (ROOT / "papers_acl.txt").read_text().split()
    tot = Counter()
    per_paper = {}
    for p in papers:
        res = clean_paper(p, args.min_count, args.max_add)
        if res is None:
            tot["missing"] += 1
            continue
        out = DST / p / f"Entity_{p}_enriched.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(res["rows"])
        tot["papers"] += 1
        tot["v1_entries"] += res["n_v1"]
        tot["v2_entries"] += len(res["rows"])
        tot["added"] += len(res["added"])
        for k, v in res["dropped"].items():
            tot["dropped_" + k] += v
        per_paper[p] = {"v1": res["n_v1"], "v2": len(res["rows"]), "dropped": res["dropped"],
                        "added": [t for t, _ in res["added"]]}
        if args.show:
            print(f"{p}: {res['n_v1']} -> {len(res['rows'])}  dropped {res['dropped']}  "
                  f"added {[f'{t}({n})' for t, n in res['added']]}")
    print(json.dumps(dict(tot), indent=1))
    if not args.papers:
        s = ROOT / args.summary
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"params": {"min_count": args.min_count, "max_add": args.max_add},
                                 "totals": dict(tot), "papers": per_paper}, indent=1), encoding="utf-8")
        print(f"-> {s}")


if __name__ == "__main__":
    main()
