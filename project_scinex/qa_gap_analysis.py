#!/usr/bin/env python3
"""
qa_gap_analysis.py
------------------
Why do the knowledge graphs answer so few of the simulated-Q&A questions? (OBJECTIVE step 3.)
For one qa_eval run and a set of fixed-extractor KGs, per question:

  entity     the question or its gold answer mentions an entity of the paper's fixed list
             (a fixed-extractor KG can only hold triples whose subject is listed)
  para_kg    the KG kept >= 1 triple on the paragraph the question was written from
  ans_node   the gold answer appears (normalised substring, either way) in a KG node of the
             same paper — a looser cousin of qa_eval's answer-in-KG
  score      the graded KG answer (CORRECT 1, PARTIAL 0.5, else 0)

and reports the funnel (all questions -> entity-related -> paragraph covered -> answer in the
graph -> answered correctly), plus accuracy by question relation. Deterministic, no model.

    python qa_gap_analysis.py --run acl-f0 --kg fixed/gptoss-20b fixed/qwen3-8b-acl-ft-me30-cv \
        --out qa_eval/acl-f0/gap_analysis.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import load_entity_set_for, norm, read_jsonl  # noqa: E402
from ft_overlap import kept_by_paragraph  # noqa: E402
from qa_eval import POINTS, gkey, kg_triples  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--kg", nargs="+", required=True, help="fixed/<slug> or fixed_scinex/<slug>")
    ap.add_argument("--out")
    args = ap.parse_args()
    rd = Path("qa_eval") / args.run
    qs = read_jsonl(rd / "questions.jsonl")
    verdict = {r["gkey"]: r["verdict"] for r in read_jsonl(rd / "grades.jsonl")}

    ents = {}
    q_entity = {}
    for q in qs:
        p = q["paper"]
        if p not in ents:
            ents[p] = load_entity_set_for(p)
        q_entity[q["qid"]] = bool(ents[p] and ents[p].match_any_in_text(q["question"] + " " + q["answer"]))
    n = len(qs)
    res = {"run": args.run, "questions": n,
           "entity_related": sum(q_entity.values()),
           "by_relation_count": dict(Counter(q["relation"] for q in qs)), "kgs": {}}
    print(f"{n} questions; {res['entity_related']} ({100 * res['entity_related'] / n:.1f}%) mention "
          f"a listed entity in the question or answer")

    for spec in args.kg:
        exdir, slug = spec.split("/", 1)
        ont = "scinex" if exdir.endswith("_scinex") else "ceo"
        cond = f"kg__{spec.replace('/', '__')}"
        ans_f = rd / "answers" / f"{cond}.jsonl"
        if not ans_f.exists():
            print(f"{spec}: no answers yet — skipped")
            continue
        answers = {r["qid"]: r for r in read_jsonl(ans_f)}
        kept_para, nodes = {}, {}
        for p in {q["paper"] for q in qs}:
            kept_para[p] = kept_by_paragraph(p, ont, slug, "fixed") or {}
            nodes[p] = {norm(x) for t in (kg_triples(p, spec) or []) for x in (t[0], t[2])}
        funnel = Counter()
        by_rel = defaultdict(lambda: [0, 0.0])
        split = defaultdict(lambda: [0, 0.0])
        for q in qs:
            a = answers.get(q["qid"])
            s = 0.0
            if a and not a["abstained"]:
                s = POINTS.get(verdict.get(gkey(q["qid"], a["answer"]), "INCORRECT"), 0.0)
            ent = q_entity[q["qid"]]
            cov = bool(kept_para[q["paper"]].get(q["para_id"]) or
                       kept_para[q["paper"]].get(str(q["para_id"])))
            gold = norm(q["answer"])
            in_g = any(gold and (gold in nd or (len(nd) > 3 and nd in gold)) for nd in nodes[q["paper"]])
            funnel["all"] += 1
            funnel["entity"] += ent
            funnel["entity_and_para_covered"] += ent and cov
            funnel["para_covered"] += cov
            funnel["answer_in_graph"] += in_g
            funnel["answered_correct_pts"] += s
            by_rel[q["relation"]][0] += 1
            by_rel[q["relation"]][1] += s
            key = ("covered" if cov else "uncovered") + ("+ans_in_graph" if in_g else "")
            split[key][0] += 1
            split[key][1] += s
        acc = {k: round(100 * v[1] / v[0], 1) for k, v in split.items() if v[0]}
        res["kgs"][spec] = {"funnel": dict(funnel),
                            "accuracy_by_coverage": {k: {"n": split[k][0], "acc": acc[k]} for k in acc},
                            "accuracy_by_relation": {r: {"n": v[0], "acc": round(100 * v[1] / v[0], 1)}
                                                     for r, v in sorted(by_rel.items(), key=lambda x: -x[1][0])}}
        f = funnel
        print(f"\n{spec}: accuracy {100 * f['answered_correct_pts'] / n:.1f}")
        print(f"  question paragraph covered by the KG: {f['para_covered']} ({100 * f['para_covered'] / n:.1f}%)"
              f"; gold answer in a graph node: {f['answer_in_graph']} ({100 * f['answer_in_graph'] / n:.1f}%)")
        for k in sorted(acc):
            print(f"  {k:28s} n={split[k][0]:4d}  acc {acc[k]:5.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
