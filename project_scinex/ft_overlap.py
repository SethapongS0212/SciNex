#!/usr/bin/env python3
"""
ft_overlap.py
-------------
Where does a student extract relative to its teacher? Per paragraph (same replies and guards as
ft_compare.py), count the paragraphs where both keep >= 1 triple, only the teacher, only the
student, or neither, and the share of the student's kept triples that sit on teacher-empty
paragraphs. With --verdicts (a blind_judge `by_model/<student>/verdicts.csv`, i.e. AFTER collect),
split the student's judged C1/C3/C4 by shared vs teacher-empty paragraph.

Run from the project_scinex directory that holds output/:
    python ft_overlap.py --extractor fixed --ontology ceo --teacher gptoss-20b \
        --student qwen3-8b-acl-ft-me30-cv --papers $(cat ft_data/acl/fold0/eval_papers.txt) \
        --verdicts ontology_eval/_blind/acl-f0/ceo/by_model/qwen3-8b-acl-ft-me30-cv/verdicts.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_compare import (OUTPUT_DIR, entity_set_for, exdir_for, guard_pass,  # noqa: E402
                        is_api_run, read_replies)

SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def kept_by_paragraph(paper, ontology, slug, extractor):
    d = OUTPUT_DIR / paper / "kg" / exdir_for(extractor, ontology) / slug
    if not (d / "responses.jsonl").exists():
        return None
    meta = json.loads((d / "prompt_meta.json").read_text(encoding="utf-8"))
    relations = meta["relations"]
    replies, _ = read_replies(d / "responses.jsonl")
    api_run = is_api_run(d)
    entity_set = entity_set_for(paper, meta.get("entity_dir")) if extractor == "fixed" else None
    return {pid: guard_pass(raw, relations, api_run, entity_set)[2] for pid, raw in replies.items()}


def tkey(subject, predicate, obj, sentence):
    norm = lambda s: " ".join(str(s).lower().split())
    return (norm(subject), norm(predicate).replace(" ", ""), norm(obj), norm(sentence))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", nargs="+", required=True)
    ap.add_argument("--ontology", default="ceo")
    ap.add_argument("--extractor", default="fixed")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--verdicts", help="by_model/<student>/verdicts.csv from blind_judge collect")
    ap.add_argument("--out", help="write the result as JSON here")
    args = ap.parse_args()

    para, on_empty, student_kept = Counter(), 0, 0
    where = {}                                    # triple key -> "shared" | "teacher_empty"
    for p in args.papers:
        s = kept_by_paragraph(p, args.ontology, args.student, args.extractor)
        t = kept_by_paragraph(p, args.ontology, args.teacher, args.extractor)
        if s is None or t is None:
            sys.exit(f"missing replies for {p}: student={s is not None} teacher={t is not None}")
        for pid in set(s) | set(t):
            ss, tt = bool(s.get(pid)), bool(t.get(pid))
            para["both" if ss and tt else "only_teacher" if tt else "only_student" if ss else "neither"] += 1
            for tr in s.get(pid) or []:
                student_kept += 1
                on_empty += not tt
                where[(p,) + tkey(tr["subject"], tr["predicate"], tr["object"],
                                  tr.get("source_sentence", ""))] = "shared" if tt else "teacher_empty"
    res = {"ontology": args.ontology, "student": args.student, "teacher": args.teacher,
           "paragraphs": dict(para), "student_kept": student_kept,
           "student_kept_on_teacher_empty": on_empty,
           "share_on_teacher_empty": round(on_empty / student_kept, 3) if student_kept else None}

    if args.verdicts:
        split = {"shared": Counter(), "teacher_empty": Counter(), "unmatched": Counter()}
        for r in csv.DictReader(open(args.verdicts, encoding="utf-8")):
            k = (r["paper"],) + tkey(r["subject"], r["predicate"], r["object"], r["source_sentence"])
            c = split[where.get(k, "unmatched")]
            c["n"] += 1
            for crit in ("C1", "C4"):
                c[crit] += SCORE.get(r[crit], 0.0)
            c["C3_appropriate"] += r["C3"] == "APPROPRIATE"
        res["judged"] = {name: {"n": c["n"],
                                "C1": round(100 * c["C1"] / c["n"], 1),
                                "C3_appropriate_pct": round(100 * c["C3_appropriate"] / c["n"], 1),
                                "C4": round(100 * c["C4"] / c["n"], 1)}
                         for name, c in split.items() if c["n"]}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
