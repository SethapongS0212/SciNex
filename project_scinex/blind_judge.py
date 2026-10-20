#!/usr/bin/env python3
"""
blind_judge.py
--------------
BLIND, POOLED C1/C3/C4 census of several extractors' triples on the same papers — the
finetuning evaluation (base vs finetuned variants vs teacher, hands_off.md Session 34).

Why not `ontology_eval.py triples` per model: those bundles are per model, so the judge
knows whose triples it is grading, and they sample ~40 triples (n≈40 moved scores by up to
10 pts, §31.3). Here every kept triple of every model is judged (a census), the models are
POOLED and SHUFFLED, item ids are salted hashes, and a triple produced identically by two
models is judged ONCE and the verdict counted for both. The mapping back to models lives in
key.csv, which the judge must not open until collect.

Judge (user decision 2026-10-01): claude-opus-5-5 IN SESSION, no API. Single rater. Not
comparable with the claude-opus-5 C1-C6 numbers in results.md (different judge, census not
sample). Claude's OWN extraction can never be judged here (refused).

    python3 blind_judge.py build   --run ft-f0 --ontology ceo --papers strabismus2026 osmotic2026 \
            --slugs qwen3-4b-base-hf qwen3-4b-ft-cv qwen3-4b-ft-cons-cv gptoss-120b
    #  ... judge fills ontology_eval/_blind/ft-f0/ceo/replies/reply_NN.txt (same JSON as ontology_eval) ...
    python3 blind_judge.py status  --run ft-f0 --ontology ceo
    python3 blind_judge.py collect --run ft-f0 --ontology ceo
    python3 blind_judge.py report  --run ft-f0 --ontology ceo --compare qwen3-4b-ft-cons-cv qwen3-4b-ft-cv
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ft_common import quiet_guard_logs, refuse_excluded  # noqa: E402
from ontology_eval import (  # noqa: E402
    EVAL_DIR, POINTS_BY_CRIT, RUBRIC_TRIPLES, SCALES, _OBJ_RE, _schema_for, _tid,
)

CRITS = ["C1", "C3", "C4"]
JUDGE = "claude-opus-5-5 (in-session)"


def run_dir(run, ontology):
    if not re.fullmatch(r"[\w.-]+", run):
        sys.exit("--run must be a plain name")
    return EVAL_DIR / "_blind" / run / ontology


def _content(t):
    return "|".join(" ".join(str(t.get(k, "")).split()).lower()
                    for k in ("subject", "predicate", "object", "source_sentence"))


def cmd_build(args):
    for s in args.slugs:
        refuse_excluded(s, "extractor slug (Claude never judges Claude's extraction)")
    d = run_dir(args.run, args.ontology)
    if (d / "key.csv").exists() and not args.force:
        sys.exit(f"{d} already built — judging may be under way. --force rebuilds (replies are kept; "
                 f"verdicts for items still in the pool carry over)")
    from ft_compare import run_stats
    quiet_guard_logs()
    salt =hashlib.sha1(f"{args.run}|{args.ontology}|{args.seed}".encode()).hexdigest()
    items, key_rows, missing = {}, [], []
    for slug in args.slugs:
        for p in args.papers:
            # the SAME triple set ft_compare counts as "kept": today's guards over the stored
            # replies, conflicting duplicate replies dropped (triples.json can be stale or
            # inflated — gptoss scinex strabismus2026, §32.2)
            st = run_stats(p, args.ontology, slug)
            if st is None:
                missing.append(f"{slug}/{p}")
                continue
            seen = set()
            for t in st["_kept"]:
                c = _content(t)
                if c in seen:                     # identical triple twice in one paper: once
                    continue
                seen.add(c)
                bid = "b" + hashlib.sha1(f"{salt}|{c}".encode()).hexdigest()[:10]
                items.setdefault(bid, t)
                key_rows.append({"blind_id": bid, "slug": slug, "paper": p, "triple_id": _tid(t, slug),
                                 "subject": t["subject"], "predicate": t["predicate"],
                                 "object": t["object"],
                                 "source_sentence": " ".join(str(t.get("source_sentence", "")).split())})
    if missing:
        msg = f"no triples.json for: {', '.join(missing)}"
        if not args.allow_missing:
            sys.exit(msg + " — a missing run would score as absent, not empty. --allow-missing to proceed")
        print("  ! " + msg)

    schema = _schema_for(args.ontology)
    lower = {k.lower(): v for k, v in schema.items()}
    ids = sorted(items)
    random.Random(args.seed).shuffle(ids)
    d.mkdir(parents=True, exist_ok=True)
    (d / "replies").mkdir(exist_ok=True)
    for old in d.glob("batch_*.txt"):
        old.unlink()
    header = ("\nGrade the following {n} triples on C1, C3, C4. Items are pooled from several "
              "extractors and shuffled; which model produced a triple is deliberately hidden.\n")
    chunks = [ids[i:i + args.size] for i in range(0, len(ids), args.size)]
    for bi, chunk in enumerate(chunks, 1):
        parts = [RUBRIC_TRIPLES] if bi == 1 else []
        parts.append(header.format(n=len(chunk)))
        for bid in chunk:
            t = items[bid]
            defn = lower.get(str(t["predicate"]).lower().replace(" ", ""), "(predicate not in this ontology)")
            parts.append(f"\n----- triple_id: {bid} -----\n"
                         f"TRIPLE:     ({t['subject']})  --{t['predicate']}-->  ({t['object']})\n"
                         f"PREDICATE DEFINITION: {defn}\n"
                         f"SOURCE SENTENCE: {' '.join(str(t.get('source_sentence', '')).split())}\n")
        (d / f"batch_{bi:02d}.txt").write_text("".join(parts), encoding="utf-8")
    with open(d / "key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0]) if key_rows else ["blind_id"])
        w.writeheader()
        w.writerows(key_rows)
    meta = {"created": date.today().isoformat(), "judge": JUDGE, "ontology": args.ontology,
            "papers": args.papers, "slugs": args.slugs, "seed": args.seed, "size": args.size,
            "n_items": len(ids), "n_batches": len(chunks),
            "triples_per_slug": dict(Counter(r["slug"] for r in key_rows))}
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shared = len(key_rows) - len(ids)
    print(f"{len(key_rows)} triples from {len(args.slugs)} extractors -> {len(ids)} unique items "
          f"({shared} duplicates judged once) -> {len(chunks)} batches of <= {args.size} in {d}")
    print("  per extractor: " + ", ".join(f"{k} {v}" for k, v in meta["triples_per_slug"].items()))
    print(f"  judge writes {d / 'replies'}/reply_NN.txt; do NOT open key.csv before collect")


def _parse_replies(d):
    verdicts, bad = {}, Counter()
    for f in sorted((d / "replies").glob("*.txt")):
        txt = f.read_text(encoding="utf-8")
        try:
            objs = json.loads(re.sub(r"^```\w*|```$", "", txt.strip(), flags=re.M)).get("verdicts", [])
        except Exception:
            objs = []
            for m in _OBJ_RE.finditer(txt):
                try:
                    objs.append(json.loads(m.group(0)))
                except Exception:
                    pass
        for o in objs:
            bid = str(o.get("triple_id", "")).strip()
            row = {c: str(o.get(c, "")).strip().upper() for c in CRITS}
            if not bid or any(row[c] not in SCALES[c] for c in CRITS):
                bad["invalid"] += 1
                continue
            row["note"] = str(o.get("note", ""))
            verdicts[bid] = row                     # a later reply file overrides an earlier one
    return verdicts, bad


def _read_key(d):
    with open(d / "key.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cmd_status(args):
    d = run_dir(args.run, args.ontology)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    verdicts, bad = _parse_replies(d)
    done_batches = []
    for bi in range(1, meta["n_batches"] + 1):
        ids = set(re.findall(r"triple_id: (b[0-9a-f]{10})", (d / f"batch_{bi:02d}.txt").read_text(encoding="utf-8")))
        if ids and ids <= set(verdicts):
            done_batches.append(bi)
    todo = [b for b in range(1, meta["n_batches"] + 1) if b not in done_batches]
    print(f"{len(verdicts)}/{meta['n_items']} items judged; batches done {len(done_batches)}/{meta['n_batches']}"
          + (f"; next: batch_{todo[0]:02d}.txt" if todo else "; ALL DONE")
          + (f"; {bad['invalid']} invalid verdict rows ignored" if bad else ""))


def cmd_collect(args):
    d = run_dir(args.run, args.ontology)
    verdicts, bad = _parse_replies(d)
    rows = _read_key(d)
    out = d / "by_model"
    per = defaultdict(list)
    unjudged = set()
    for r in rows:
        v = verdicts.get(r["blind_id"])
        if v is None:
            unjudged.add(r["blind_id"])
            continue
        per[r["slug"]].append({**r, **v})
    for slug, rs in per.items():
        (out / slug).mkdir(parents=True, exist_ok=True)
        with open(out / slug / "verdicts.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["triple_id", "blind_id", "paper", "subject", "predicate",
                                               "object", "source_sentence", *CRITS, "note"])
            w.writeheader()
            w.writerows({k: x[k] for k in w.fieldnames} for x in rs)
    print(f"{sum(len(v) for v in per.values())} model-triples scored from {len(verdicts)} verdicts -> {out}"
          + (f"\n  ⚠ {len(unjudged)} items still UNJUDGED — numbers are partial" if unjudged else "")
          + (f"\n  {bad['invalid']} invalid verdict rows ignored" if bad else ""))


def _boot(vals, B, seed):
    if not vals:
        return None, None
    rng = random.Random(seed)
    s = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(B))
    return s[int(0.025 * B)], s[int(0.975 * B) - 1]


def cmd_report(args):
    d = run_dir(args.run, args.ontology)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    scores = {}
    for slug in meta["slugs"]:
        f = d / "by_model" / slug / "verdicts.csv"
        if not f.exists():
            continue
        rs = list(csv.DictReader(open(f, encoding="utf-8")))
        scores[slug] = {c: [POINTS_BY_CRIT[c][r[c]] for r in rs] for c in CRITS}
        scores[slug]["_n"] = len(rs)
        scores[slug]["_of"] = meta["triples_per_slug"].get(slug, 0)
    print(f"blind census, {args.ontology}, judge {meta['judge']}, papers {' '.join(meta['papers'])}")
    print("95% CI = bootstrap over triples (triples within a paper are not independent; read as a lower bound on uncertainty)\n")
    print("| extractor | judged / kept | C1 | C3 | C4 | C4 strictly CORRECT |")
    print("|---|---|---|---|---|---|")
    report = {"meta": meta, "models": {}, "comparisons": []}
    for slug, s in scores.items():
        cells = []
        for c in CRITS:
            m = sum(s[c]) / len(s[c]) if s[c] else None
            lo, hi = _boot(s[c], args.bootstrap, args.seed)
            cells.append("—" if m is None else f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
            report["models"].setdefault(slug, {})[c] = {"mean": m, "ci95": [lo, hi]}
        strict = sum(1 for x in s["C4"] if x == 1.0) / len(s["C4"]) if s["C4"] else None
        report["models"][slug].update(n=s["_n"], kept=s["_of"], c4_strict=strict)
        print(f"| {slug} | {s['_n']} / {s['_of']} | " + " | ".join(cells)
              + f" | {'—' if strict is None else f'{100 * strict:.1f}'} |")
    for a, b in (args.compare or []):
        if a not in scores or b not in scores:
            sys.exit(f"--compare: have {sorted(scores)}")
        rng = random.Random(args.seed)
        for c in ("C1", "C4"):
            xa, xb = scores[a][c], scores[b][c]
            diffs = sorted(sum(rng.choice(xa) for _ in xa) / len(xa) - sum(rng.choice(xb) for _ in xb) / len(xb)
                           for _ in range(args.bootstrap))
            m = sum(xa) / len(xa) - sum(xb) / len(xb)
            lo, hi = diffs[int(0.025 * args.bootstrap)], diffs[int(0.975 * args.bootstrap) - 1]
            report["comparisons"].append({"a": a, "b": b, "crit": c, "diff": m, "ci95": [lo, hi]})
            print(f"\n{a} minus {b}, {c}: {100 * m:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] (unpaired)")
    (d / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {d / 'report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--slugs", nargs="+", required=True)
    b.add_argument("--papers", nargs="+", required=True)
    b.add_argument("--size", type=int, default=25)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--allow-missing", action="store_true")
    b.add_argument("--force", action="store_true")
    for name in ("status", "collect", "report"):
        s = sub.add_parser(name)
        if name == "report":
            s.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"))
            s.add_argument("--bootstrap", type=int, default=2000)
            s.add_argument("--seed", type=int, default=13)
    for s in (b, *[sub.choices[n] for n in ("status", "collect", "report")]):
        s.add_argument("--run", required=True)
        s.add_argument("--ontology", default="ceo", choices=["ceo", "scinex"])
    args = ap.parse_args()
    {"build": cmd_build, "status": cmd_status, "collect": cmd_collect, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
