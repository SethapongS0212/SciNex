#!/usr/bin/env python3
"""
ft_compare.py
-------------
Judge-free comparison of extraction runs on the same papers — base vs finetuned vs the
teacher. Everything here is exact counting, no LLM:

  replies        paragraphs answered; valid JSON / truncated / empty
  proposed       triples in the raw replies
  kept           triples surviving the CURRENT guards + graph-builder filter (= triples.json)
  kept rate      kept / proposed — how much of a model's output the guards throw away
  coverage       share of paragraphs yielding >= 1 kept triple
  predicates     top predicates, and how often `configures` appears (open decision)
  vs teacher     paper-level agreement with the teacher's kept triples: same subject and
                 predicate, object overlapping (token containment or Jaccard >= 0.5).
                 This measures IMITATION, not correctness — the teacher is not ground
                 truth (C4 86.2% CEO, n~40). Quality needs a judge (ontology_eval) or qa_eval.

    python3 ft_compare.py --slugs qwen3-4b-base-hf qwen3-4b-ft-cv gptoss-120b \
            --papers strabismus2026 osmotic2026 --out ft_runs/compare_f0.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ft_common import (  # noqa: E402
    OUTPUT_DIR, TEACHER_DEFAULT, exdir_for, guard_pass, is_api_run, norm, quiet_guard_logs,
    read_replies,
)


def _toks(s):
    return set(re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", norm(s)))


def _obj_match(a, b):
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta or len(ta & tb) / len(ta | tb) >= 0.5


def agreement(student, teacher):
    """(matched_student, matched_teacher) under subject+predicate equality and object overlap."""
    def key(t):
        return norm(t["subject"]), str(t["predicate"]).lower()
    ms = sum(1 for s in student if any(key(s) == key(t) and _obj_match(s["object"], t["object"])
                                       for t in teacher))
    mt = sum(1 for t in teacher if any(key(s) == key(t) and _obj_match(s["object"], t["object"])
                                       for s in student))
    return ms, mt


def run_stats(paper, ontology, slug):
    d = OUTPUT_DIR / paper / "kg" / exdir_for("relation", ontology) / slug
    if not (d / "responses.jsonl").exists():
        return None
    meta = json.loads((d / "prompt_meta.json").read_text(encoding="utf-8"))
    relations = meta["relations"]
    replies, issues = read_replies(d / "responses.jsonl")
    trunc = sum(1 for l in open(d / "responses.jsonl", encoding="utf-8")
                if l.strip() and json.loads(l).get("truncated"))
    st = Counter()
    proposed = kept_n = covered = 0
    kept = []
    api_run = is_api_run(d)
    for pid, raw in replies.items():
        status, raw_t, surv = guard_pass(raw, relations, api_run)
        st[status] += 1
        proposed += len(raw_t)
        kept_n += len(surv)
        covered += bool(surv)
        kept += surv
    return {"paragraphs": meta.get("n_paragraphs"), "replies": len(replies),
            "status": dict(st), "truncated": trunc, "reply_issues": dict(issues),
            "proposed": proposed, "kept": kept_n,
            "kept_rate": kept_n / proposed if proposed else None,
            "coverage": covered / len(replies) if replies else None,
            "predicates": Counter(t["predicate"] for t in kept).most_common(8),
            "configures": sum(1 for t in kept if t["predicate"] == "configures"),
            "_kept": kept}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slugs", nargs="+", required=True)
    ap.add_argument("--papers", nargs="+", required=True)
    ap.add_argument("--ontologies", nargs="+", choices=["ceo", "scinex"], default=["ceo", "scinex"])
    ap.add_argument("--teacher", default=TEACHER_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    quiet_guard_logs()

    report = {}
    for ont in args.ontologies:
        print(f"\n### {ont}\n")
        print("| run | papers | replies | valid JSON | truncated | proposed | kept | kept rate "
              "| coverage | vs teacher P / R | configures |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for slug in args.slugs:
            agg = Counter()
            ms = mt = n_teacher = n_papers = 0
            preds = Counter()
            for paper in args.papers:
                s = run_stats(paper, ont, slug)
                if s is None:
                    continue
                n_papers += 1
                for k in ("replies", "truncated", "proposed", "kept", "configures"):
                    agg[k] += s[k]
                agg["valid"] += s["status"].get("ok", 0)
                agg["covered"] += round(s["coverage"] * s["replies"]) if s["coverage"] is not None else 0
                preds.update(dict(s["predicates"]))
                t = run_stats(paper, ont, args.teacher) if slug != args.teacher else None
                if t is not None:
                    a, b = agreement(s["_kept"], t["_kept"])
                    ms += a
                    mt += b
                    n_teacher += t["kept"]
                s.pop("_kept")
                report.setdefault(ont, {}).setdefault(slug, {})[paper] = s

            def pct(a, b):
                return f"{100 * a / b:.1f}%" if b else "—"
            vs = (f"{pct(ms, agg['kept'])} / {pct(mt, n_teacher)}" if slug != args.teacher else "(teacher)")
            print(f"| {slug} | {n_papers} | {agg['replies']} | {pct(agg['valid'], agg['replies'])} "
                  f"| {agg['truncated']} | {agg['proposed']} | {agg['kept']} "
                  f"| {pct(agg['kept'], agg['proposed'])} | {pct(agg['covered'], agg['replies'])} "
                  f"| {vs} | {agg['configures']} |")
            report.setdefault("_summary", {}).setdefault(ont, {})[slug] = {
                **agg, "top_predicates": preds.most_common(8),
                "vs_teacher_precision": ms / agg["kept"] if agg["kept"] else None,
                "vs_teacher_recall": mt / n_teacher if n_teacher else None}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
