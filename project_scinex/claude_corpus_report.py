"""
claude_corpus_report.py
-----------------------
Bundle every Claude-extracted paper into one corpus JSON and print the
Claude-vs-local-model comparison table.

    python3 claude_corpus_report.py --model-slug claude-opus-5 --baseline qwen3-8b

The output path defaults to output/<model-slug>_<extractor>_corpus.json, so running this
for a second model CANNOT overwrite the first model's corpus JSON. Pass --out to override.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_DIR = Path("output")


def load(paper, extractor, slug):
    f = OUTPUT_DIR / paper / "kg" / extractor / slug / "triples.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="relation")
    ap.add_argument("--model-slug", default="claude-opus-5")
    ap.add_argument("--baseline", default="qwen3-8b")
    # NOT a fixed default: a fixed one silently OVERWRITES another model's corpus JSON
    # as soon as a different --model-slug is passed. Derived from the slug instead, so
    # every model gets its own file and no run can clobber a previous one.
    ap.add_argument("--out", default=None,
                    help="output JSON (default: output/<model-slug>_<extractor>_corpus.json)")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"output/{args.model_slug}_{args.extractor}_corpus.json"

    papers = sorted(p.name for p in OUTPUT_DIR.iterdir()
                    if (p / "kg" / args.extractor / args.model_slug / "triples.json").exists())

    corpus, rows = [], []
    pred_c, pred_b = Counter(), Counter()
    for paper in papers:
        mine = load(paper, args.extractor, args.model_slug) or []
        base = load(paper, args.extractor, args.baseline) or []
        prompts = OUTPUT_DIR / paper / "kg" / args.extractor / args.model_slug / "prompts.jsonl"
        n_para = sum(1 for _ in open(prompts, encoding="utf-8")) if prompts.exists() else 0
        resp = OUTPUT_DIR / paper / "kg" / args.extractor / args.model_slug / "responses.jsonl"
        emitted = 0
        if resp.exists():
            for line in open(resp, encoding="utf-8"):
                if not line.strip():
                    continue
                raw = json.loads(line)["raw"]
                try:
                    emitted += len(json.loads(raw).get("triples", []))
                except Exception:
                    pass
        for t in mine:
            t = dict(t)
            t.setdefault("paper", paper)
            corpus.append(t)
            pred_c[t["predicate"].lower()] += 1
        for t in base:
            pred_b[t["predicate"].lower()] += 1
        rows.append((paper, n_para, emitted, len(mine), len(base)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "extractor": args.extractor,
        "model": args.model_slug,
        "n_papers": len(papers),
        "n_triples": len(corpus),
        "predicate_counts": dict(pred_c.most_common()),
        "triples": corpus,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'paper':18s} {'paras':>5s} {'emitted':>7s} {'stored':>6s} | {args.baseline:>9s}")
    for paper, n_para, emitted, kept, base in rows:
        print(f"{paper:18s} {n_para:5d} {emitted:7d} {kept:6d} | {base:9d}")
    tot_p = sum(r[1] for r in rows)
    tot_e = sum(r[2] for r in rows)
    tot_k = sum(r[3] for r in rows)
    tot_b = sum(r[4] for r in rows)
    print(f"{'TOTAL':18s} {tot_p:5d} {tot_e:7d} {tot_k:6d} | {tot_b:9d}")

    print(f"\npredicate mix ({args.model_slug} vs {args.baseline}, % of that model's triples)")
    print(f"  {'predicate':18s} {'claude':>7s} {'%':>6s} {'base':>6s} {'%':>6s}")
    for p in sorted(set(pred_c) | set(pred_b), key=lambda k: -(pred_c[k] + pred_b[k])):
        print(f"  {p:18s} {pred_c[p]:7d} {100*pred_c[p]/max(tot_k,1):5.1f}% "
              f"{pred_b[p]:6d} {100*pred_b[p]/max(tot_b,1):5.1f}%")
    print(f"\nWrote {args.out}  ({len(corpus)} triples over {len(papers)} papers)")


if __name__ == "__main__":
    main()
