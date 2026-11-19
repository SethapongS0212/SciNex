#!/usr/bin/env python3
"""
verify_triples.py — open-model VERIFIER for extracted triples (Session 41, 2026-11-19)
---------------------------------------------------------------------------------------
Option (a) of HANDOFF "Next steps": better training targets. Table 26 showed that the student's
precision is capped by the teacher's targets (C4 ~35-40%, only 16-22% strictly correct). Here an
open model checks each teacher triple ON ITS OWN against its source sentence and predicate
definition (the blind judge's format), and returns P(YES) from the first-token logprobs. Filtering
the teacher's replies by P(YES) gives cleaner training targets; the student is then trained on them
exactly as before (build_ft_data -> ft_train_qlora, `noont`).

Claude is never the verifier (open models only; Claude stays the blind judge).

Subcommands (run with /home/dev1/envvllm/bin/python; `variant` also runs in envsam):

  probe    score the triples already judged in blind rounds (ontology_eval/_blind/<round>/...
           verdicts.csv) and report how P(YES) separates C4-correct from C4-wrong triples:
           AUC, and C4 of kept / removed at each threshold. Writes ft_runs/verify/probe_<tag>.csv.
               verify_triples.py probe --rounds acl-f0-q30b scale-test --tag q30b
  score    score every guard-surviving raw triple of stored runs; writes
           output/<paper>/kg/<exdir>/<slug>/verify_<tag>.jsonl  {para_id, i, p}. Resumable.
               verify_triples.py score --slugs qwen3-30b-a3b-v2-g2 --papers-from X --tag q30b
  variant  copy a run with every raw triple whose P(YES) < threshold removed from responses.jsonl,
           into the sibling slug <slug><suffix>, and re-ingest it (as make_guard_variant.py does).
           Every tool that re-parses stored replies then sees the filtered run.
               verify_triples.py variant --slugs qwen3-30b-a3b-v2-g2 --papers-from X --tag q30b \
                   --threshold 0.5 --suffix=-v50
"""
import argparse
import csv
import io
import json
import logging
import math
import shutil
import sys
import time
from argparse import Namespace
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ft_common import (ENTITY_DIR_DEFAULT, OUTPUT_DIR, SCINEX_OWL_DEFAULT, classify_raw,  # noqa: E402
                       entity_csv_for, exdir_for, guard_pass, is_api_run, load_entity_set_for,
                       quiet_guard_logs, read_replies, schema_from_system_prompt)

VERIFIER_DEFAULT = "RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16"
# Definitions come from the teacher's stored full-ontology prompt (the same text the blind judge
# shows); student runs use the `noont` prompt, which has none.
DEF_SOURCE = {"ceo": "output/2020.acl-main.101/kg/fixed/qwen3-30b-a3b-v2-g2/system_prompt.txt",
              "scinex": "output/2020.acl-main.101/kg/fixed_scinex/qwen3-30b-a3b-v2-g2/system_prompt.txt"}
PTS = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MAX_SRC = 1500

SYSTEM = """You check knowledge-graph triples extracted from scientific papers. Each triple is \
(subject) --predicate--> (object) and was taken from one SOURCE SENTENCE. Judge ONLY from that sentence.

Answer YES only if ALL of these hold:
1. The sentence explicitly states this relation between the subject and the object (it does not just mention both).
2. The direction is right: the subject is the one that stands in the predicate's relation to the object.
3. The subject and the object fit the predicate's definition (domain -> range).
4. Both are real, meaningful concepts from the sentence: not a citation or reference-list entry, a table or formula \
fragment, a bare number, or a generic placeholder such as "the model" or "data".

Otherwise answer NO. Reply with exactly one word: YES or NO."""


def user_prompt(s, p, o, defn, src):
    src = " ".join(str(src).split())[:MAX_SRC]
    return (f"TRIPLE: ({s})  --{p}-->  ({o})\n"
            f"PREDICATE DEFINITION: {defn or '(none given)'}\n"
            f"SOURCE SENTENCE: {src}")


def definitions(ontology):
    text = (ROOT / DEF_SOURCE[ontology]).read_text(encoding="utf-8")
    return {k.lower(): v for k, v in schema_from_system_prompt(text).items()}


# ── model ────────────────────────────────────────────────────────────────────

class Verifier:
    def __init__(self, model, max_model_len=2048, util=0.90):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=util,
                       seed=0, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def render(self, user):
        return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True)

    def p_yes(self, users):
        """P(YES) / (P(YES) + P(NO)) over the first generated token's top-20 logprobs."""
        if not users:
            return []
        reqs = [{"prompt_token_ids": self.tok(self.render(u), add_special_tokens=False)["input_ids"]}
                for u in users]
        out = []
        for o in self.llm.generate(reqs, self.sp, use_tqdm=False):
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            y = n = 0.0
            for x in lp.values():
                w = (x.decoded_token or "").strip().strip(".").upper()
                if w == "YES":
                    y += math.exp(x.logprob)
                elif w == "NO":
                    n += math.exp(x.logprob)
            out.append(y / (y + n) if y + n > 0 else 0.5)
        return out


# ── probe on judged triples ──────────────────────────────────────────────────

def auc(pos, neg):
    """P(score of a random positive > random negative), ties count half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2                      # mean 1-based rank of the tie block
        rank_sum += r * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def report(rows, label):
    m = lambda v: 100 * sum(v) / len(v) if v else float("nan")
    c4 = [r["c4"] for r in rows]
    print(f"\n=== {label}: n={len(rows)}  C4 {m(c4):.1f}  strict {m([x == 1.0 for x in c4]):.1f}")
    print(f"  AUC  correct-vs-rest {auc([r['p'] for r in rows if r['c4'] == 1.0], [r['p'] for r in rows if r['c4'] < 1.0]):.3f}"
          f"   incorrect-vs-rest {1 - auc([r['p'] for r in rows if r['c4'] == 0.0], [r['p'] for r in rows if r['c4'] > 0.0]):.3f}")
    print("  thr   removed   C4 kept   C4 removed   strict kept")
    for t in THRESHOLDS:
        kept = [r["c4"] for r in rows if r["p"] >= t]
        rem = [r["c4"] for r in rows if r["p"] < t]
        print(f"  {t:4.2f}  {100 * len(rem) / len(rows):5.1f}%   {m(kept):6.1f}    {m(rem):6.1f}      {m([x == 1.0 for x in kept]):6.1f}")


def cmd_probe(args):
    rows = []
    for rnd in args.rounds:
        for ont in ("ceo", "scinex"):
            defs = definitions(ont)
            for f in sorted((ROOT / "ontology_eval/_blind" / rnd / ont / "by_model").glob("*/verdicts.csv")):
                for r in csv.DictReader(open(f, encoding="utf-8")):
                    if r.get("C4") not in PTS:
                        continue
                    rows.append({"round": rnd, "ont": ont, "slug": f.parent.name, "triple_id": r["triple_id"],
                                 "c4": PTS[r["C4"]], "c1": PTS.get(r.get("C1"), float("nan")),
                                 "user": user_prompt(r["subject"], r["predicate"], r["object"],
                                                     defs.get(r["predicate"].lower(), ""), r["source_sentence"])})
    print(f"{len(rows)} judged triples from {args.rounds}", flush=True)
    v = Verifier(args.model, util=args.util)
    t0 = time.time()
    for r, p in zip(rows, v.p_yes([r["user"] for r in rows])):
        r["p"] = p
    print(f"scored in {time.time() - t0:.0f} s", flush=True)
    out = ROOT / "ft_runs/verify" / f"probe_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["round", "ont", "slug", "triple_id", "c4", "c1", "p"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    for rnd in args.rounds:
        report([r for r in rows if r["round"] == rnd], rnd)
        for ont in ("ceo", "scinex"):
            report([r for r in rows if r["round"] == rnd and r["ont"] == ont], f"{rnd} / {ont}")
    report(rows, "ALL")
    print(f"-> {out}")


# ── score stored runs ────────────────────────────────────────────────────────

def run_dirs(slugs, papers, ontologies):
    for slug in slugs:
        for ont in ontologies:
            exdir = exdir_for("fixed", ont)
            for paper in papers:
                d = OUTPUT_DIR / paper / "kg" / exdir / slug
                if (d / "responses.jsonl").exists():
                    yield slug, ont, paper, d


def surviving_items(d, ont, paper, defs):
    """[(para_id, i, user_prompt)] for every raw triple that survives the run's own guards alone."""
    meta = json.loads((d / "prompt_meta.json").read_text(encoding="utf-8"))
    relations, guards, api = meta["relations"], meta.get("guards", "v1"), is_api_run(d)
    ents = load_entity_set_for(paper, meta.get("entity_dir") or ENTITY_DIR_DEFAULT)
    replies, _ = read_replies(d / "responses.jsonl")
    items = []
    for pid, raw in replies.items():
        status, data = classify_raw(raw, api)
        if status != "ok":
            continue
        for i, t in enumerate(data.get("triples", [])):
            if not isinstance(t, dict):
                continue
            _, _, surv = guard_pass(json.dumps({"triples": [t]}, ensure_ascii=False), relations, api, ents, guards)
            if surv:
                s = surv[0]
                items.append((pid, i, user_prompt(s["subject"], s["predicate"], s["object"],
                                                  defs.get(str(s["predicate"]).lower(), ""), s["source_sentence"])))
    return items


def cmd_score(args):
    papers = args.papers or Path(args.papers_from).read_text(encoding="utf-8").split()
    quiet_guard_logs()
    logging.getLogger().setLevel(logging.WARNING)
    defs = {o: definitions(o) for o in args.ontologies}
    todo = [x for x in run_dirs(args.slugs, papers, args.ontologies)
            if args.force or not (x[3] / f"verify_{args.tag}.jsonl").exists()]
    print(f"{len(todo)} runs to score (tag {args.tag})", flush=True)
    if not todo:
        return
    v = Verifier(args.model, util=args.util)
    t0, n_done, n_trip = time.time(), 0, 0
    for k in range(0, len(todo), args.chunk_runs):
        chunk = todo[k:k + args.chunk_runs]
        per = [surviving_items(d, ont, paper, defs[ont]) for _, ont, paper, d in chunk]
        flat = [it for items in per for it in items]
        ps = v.p_yes([it[2] for it in flat])
        pos = 0
        for (_, _, _, d), items in zip(chunk, per):
            rows = [{"para_id": pid, "i": i, "p": round(p, 5)}
                    for (pid, i, _), p in zip(items, ps[pos:pos + len(items)])]
            pos += len(items)
            tmp = d / f"verify_{args.tag}.jsonl.tmp"
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            tmp.replace(d / f"verify_{args.tag}.jsonl")
        n_done += len(chunk)
        n_trip += len(flat)
        el = time.time() - t0
        print(f"  {n_done}/{len(todo)} runs, {n_trip} triples, {el / 60:.1f} min, "
              f"{n_trip / max(el, 1):.0f} triples/s", flush=True)
    print("score done", flush=True)


# ── filtered variant ─────────────────────────────────────────────────────────

COPY = ("prompts.jsonl", "system_prompt.txt")


def cmd_variant(args):
    papers = args.papers or Path(args.papers_from).read_text(encoding="utf-8").split()
    import claude_extract
    quiet_guard_logs()
    logging.getLogger().setLevel(logging.WARNING)
    done = missing = skipped = 0
    tot_in = tot_out = 0
    for slug, ont, paper, src in run_dirs(args.slugs, papers, args.ontologies):
        vf = src / f"verify_{args.tag}.jsonl"
        if not vf.exists():
            missing += 1
            continue
        new = slug + args.suffix
        dst = src.parent / new
        if (dst / "triples.json").exists() and not args.force:
            skipped += 1
            continue
        score = {(r["para_id"], r["i"]): r["p"] for r in map(json.loads, vf.read_text(encoding="utf-8").splitlines())}
        api = is_api_run(src)
        dst.mkdir(parents=True, exist_ok=True)
        for name in COPY:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        out_rows = []
        for line in (src / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row["raw"] if isinstance(row["raw"], str) else json.dumps(row["raw"])
            status, data = classify_raw(raw, api)
            if status == "ok" and data.get("triples"):
                ts = data["triples"]
                kept = [t for i, t in enumerate(ts) if score.get((row["para_id"], i), 1.0) >= args.threshold]
                tot_in += sum(1 for i in range(len(ts)) if (row["para_id"], i) in score)
                tot_out += sum(1 for i in range(len(ts)) if (row["para_id"], i) in score
                               and score[(row["para_id"], i)] >= args.threshold)
                row["raw"] = json.dumps({"triples": kept}, ensure_ascii=False, indent=2)
            out_rows.append(json.dumps(row, ensure_ascii=False))
        (dst / "responses.jsonl").write_text("\n".join(out_rows) + "\n", encoding="utf-8")
        meta = json.loads((src / "prompt_meta.json").read_text(encoding="utf-8"))
        meta.update(model_slug=new, verifier=args.tag, verify_threshold=args.threshold, verify_source_slug=slug)
        (dst / "prompt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        ent_dir = meta.get("entity_dir") or str(ENTITY_DIR_DEFAULT)
        ns = Namespace(paper=paper, extractor="fixed", ontology=ont,
                       ontology_file=meta.get("ontology_file") or SCINEX_OWL_DEFAULT,
                       entity_csv=entity_csv_for(paper, ent_dir), model_slug=new,
                       responses=dst / "responses.jsonl")
        with redirect_stdout(io.StringIO()):
            claude_extract.cmd_ingest(ns)
        done += 1
    print(f"variant {args.suffix} (tag {args.tag}, threshold {args.threshold}): ingested {done}, "
          f"already there {skipped}, not scored {missing}; scored triples kept {tot_out}/{tot_in} "
          f"({100 * tot_out / max(1, tot_in):.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "score", "variant"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True, help="names the verifier run (e.g. q30b)")
        if name != "variant":
            p.add_argument("--model", default=VERIFIER_DEFAULT)
            p.add_argument("--util", type=float, default=0.90)
        if name == "probe":
            p.add_argument("--rounds", nargs="+", required=True)
        else:
            p.add_argument("--slugs", nargs="+", required=True)
            p.add_argument("--papers", nargs="+")
            p.add_argument("--papers-from")
            p.add_argument("--ontologies", nargs="+", choices=["ceo", "scinex"], default=["ceo", "scinex"])
            p.add_argument("--force", action="store_true")
        if name == "score":
            p.add_argument("--chunk-runs", type=int, default=200, help="runs scored per vLLM call")
        if name == "variant":
            p.add_argument("--threshold", type=float, required=True)
            p.add_argument("--suffix", required=True)
    args = ap.parse_args()
    {"probe": cmd_probe, "score": cmd_score, "variant": cmd_variant}[args.cmd](args)


if __name__ == "__main__":
    main()
