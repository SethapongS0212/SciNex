#!/usr/bin/env python3
"""
build_ft_data.py
----------------
Build the finetuning dataset (Part 2) as chat-format JSONL, with a PAPER-LEVEL
train/eval split. One dataset is reused across base models; each model gets its
own QLoRA run (ft_train_qlora.py).

THREE PARTS
  ontology    each CEO / scinex predicate -> definition, domain -> range, direction
              judgements (as extracted vs. subject/object swapped) and relation choice.
              Targets the direction errors (splitFrom, achieves, ...) C4 keeps finding.
  vocabulary  entity -> type cards from the CS-NER gazetteer (output/acl/csner_gazetteer.csv)
              and the TRAIN papers' own entity CSVs, plus abbreviation cards.
  extraction  paragraph -> triples, from the teacher's stored replies (default
              gpt-oss-120b, Apache-2.0), run through the CURRENT guards. The target is
              exactly what `claude_extract.py ingest` would keep today, in the extractor's
              own output contract.
              --consensus-models adds triples the teacher missed but at least
              --consensus-min of the other models extracted from the same paragraph
              (exact match after the guards). The teacher stays the anchor: only paragraphs
              with a usable teacher reply are used, and an addition is skipped when the
              teacher already related the same two entities (any predicate, either
              direction). Ontology cards stay teacher-only. hands_off.md Session 34.

HARD RULE — train and eval papers are disjoint. Every paper gets a sticky fold
(ft_data/split.json); `--eval-fold k` holds fold k out. Nothing derived from an eval
paper (its paragraphs, its triples, its entity list) enters training. Adding papers
later never moves an existing paper between train and eval.

    python3 build_ft_data.py --dry-run                 # print the split + counts, write nothing
    python3 build_ft_data.py --eval-fold 0             # -> ft_data/fold0/
    python3 build_ft_data.py --eval-fold 0 --dev-fold 1   # also carve a dev fold for val loss

DEFAULTS THAT ENCODE OPEN DECISIONS (see hands_off.md Session 32)
  --configures exclude   the `configures` direction is Sam's open decision (§29.5), so no
                         example teaches either direction: paragraphs where the teacher used
                         it are skipped and it gets no ontology cards.
  --verdict-filter off   the only C1/C4 verdicts on disk were judged by claude-opus-5, and
                         Claude is out of the pipeline. Turning the filter on puts a Claude
                         judgement into data selection — Sam's call, not a default.
  paragraph-level drops  a paragraph whose target cannot be trusted is DROPPED, never edited
                         into a misleading target. Guard rejections ARE applied (the guards
                         define valid output); an empty result becomes a {"triples": []} target.

Needs the parsed HTML (output/<id>/no-llm/output.html) or stored prompts.jsonl on the
machine it runs on — both are gitignored, so run it where the pipeline ran. No model is
loaded and no API is called.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ft_common import (  # noqa: E402
    CORPUS_FILE_DEFAULT, FT_DIR, OUTPUT_DIR, SCINEX_OWL_DEFAULT, SPLIT_FILE, TEACHER_DEFAULT,
    alignment_rate, assign_folds, canonical_predicate, current_system_prompt, exdir_for,
    guard_pass, load_paragraphs, load_split, norm, parse_definition, quiet_guard_logs, read_corpus,
    read_replies, refuse_excluded, schema_from_system_prompt, sha256_file, split_sets,
    target_json, triple_id, write_jsonl,
)

ONT_NAMES = {"ceo": "CEO (Core Experiment Ontology)", "scinex": "scinex"}
CEO_TYPES = ["Model", "Method", "Dataset", "Task", "Tool", "Resource", "Language"]
GAZETTEER_DEFAULT = OUTPUT_DIR / "acl" / "csner_gazetteer.csv"
ENTITY_DIR_DEFAULT = OUTPUT_DIR / "acl"
EXCLUDED_BY_DEFAULT = {"configures"}   # open human decision, hands_off.md §29.5


# ── teacher run (per paper x ontology) ───────────────────────────────────────

def teacher_run(paper, ontology, teacher, sys_mode, cur_prompts):
    """Load one teacher run: system prompt, relations, replies. None if absent."""
    d = OUTPUT_DIR / paper / "kg" / exdir_for("relation", ontology) / teacher
    resp = d / "responses.jsonl"
    if not resp.exists():
        return None
    meta = json.loads((d / "prompt_meta.json").read_text(encoding="utf-8")) \
        if (d / "prompt_meta.json").exists() else {}
    stored = (d / "system_prompt.txt").read_text(encoding="utf-8") \
        if (d / "system_prompt.txt").exists() else None
    if sys_mode == "current":
        system, relations = cur_prompts[ontology]
    else:
        if stored is None or not meta.get("relations"):
            sys.exit(f"--system-prompt stored needs system_prompt.txt + prompt_meta.json in {d}")
        system, relations = stored, meta["relations"]
    replies, issues = read_replies(resp)
    return {"dir": d, "meta": meta, "system": system, "relations": relations,
            "stored_system": stored, "replies": replies, "issues": issues}


def load_bad_ids(teacher, ontology, level):
    """triple_ids whose C1 or C4 verdict is bad. level: 'incorrect' or 'partial'."""
    f = Path("ontology_eval") / teacher / ontology / "triples" / "verdicts.csv"
    if not f.exists():
        return set(), 0
    bad_vals = {"INCORRECT"} | ({"PARTIAL"} if level == "partial" else set())
    bad = set()
    n = 0
    with open(f, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n += 1
            if (row.get("C1") or "").upper() in bad_vals or (row.get("C4") or "").upper() in bad_vals:
                bad.add(row["triple_id"])
    return bad, n


# ── part 3: extraction pairs ─────────────────────────────────────────────────

def _key(t):
    return norm(t["subject"]), norm(t["predicate"]), norm(t["object"])


def consensus_votes(paper, ontology, args, cur_prompts, by_id, n_teacher, rep):
    """{para_id: {key: [votes, triple]}} over the consensus models' guard survivors.
    A model is skipped for this paper when its run is absent or does not line up with
    the teacher's paragraphs (count or source-sentence alignment)."""
    votes = defaultdict(dict)
    for m in args.consensus_models:
        run = teacher_run(paper, ontology, m, args.system_prompt, cur_prompts)
        if run is None:
            rep["models_skipped"][m] = "no_run"
            continue
        n = run["meta"].get("n_paragraphs")
        if n is not None and n_teacher is not None and n != n_teacher:
            rep["models_skipped"][m] = f"paragraph_count {n} vs teacher {n_teacher}"
            continue
        surv_by_pid = {}
        for pid, raw in run["replies"].items():
            status, raw_t, surv = guard_pass(raw, run["relations"])
            if status == "ok":
                surv_by_pid[pid] = (raw_t, surv)
        rate, _n = alignment_rate([(by_id[pid]["paragraph"], [t.get("source_sentence", "") for t in raw_t])
                                   for pid, (raw_t, _s) in surv_by_pid.items() if pid in by_id])
        if rate is not None and rate < args.min_alignment:
            rep["models_skipped"][m] = f"misaligned ({rate:.2f})"
            continue
        rep["models_used"].append(m)
        for pid, (_raw_t, surv) in surv_by_pid.items():
            for key, t in {_key(t): t for t in surv}.items():   # one vote per model
                votes[pid].setdefault(key, [0, t])[0] += 1
    return votes


def consensus_additions(teacher_surv, para_votes, paragraph, args, crep):
    """Triples to append to the teacher's target for one paragraph."""
    have = {_key(t) for t in teacher_surv}
    pairs = {(norm(t["subject"]), norm(t["object"])) for t in teacher_surv}
    pairs |= {(o, s) for s, o in pairs}
    text = norm(paragraph)
    cands = []
    for k, (n, t) in para_votes.items():
        if n < args.consensus_min or k in have:
            continue
        if args.configures == "exclude" and k[1] == "configures":
            crep["skipped_configures"] += 1
        elif (k[0], k[2]) in pairs:
            crep["skipped_teacher_related_pair"] += 1
        elif norm(t.get("source_sentence", ""))[:60] not in text:
            crep["skipped_sentence_not_in_paragraph"] += 1
        else:
            cands.append((n, k, t))
    # two additions on the same entity pair disagree on the relation: keep a clear
    # majority, drop a tie
    by_pair = defaultdict(list)
    for c in cands:
        by_pair[frozenset((c[1][0], c[1][2]))].append(c)
    out = []
    for group in by_pair.values():
        group.sort(key=lambda c: -c[0])
        if len(group) > 1 and group[0][0] == group[1][0]:
            crep["skipped_internal_conflict"] += len(group)
            continue
        out.append(group[0][2])
        crep["skipped_internal_conflict"] += len(group) - 1
    return out


def extraction_examples(papers, ontology, args, cur_prompts, report, bad_ids):
    """Return (examples, survivors_by_paper). survivors feed the ontology cards."""
    examples, survivors_all = [], defaultdict(list)
    for paper in papers:
        key = f"{paper}/{ontology}"
        rep = report["extraction"][key] = {"drops": Counter()}
        run = teacher_run(paper, ontology, args.teacher, args.system_prompt, cur_prompts)
        if run is None:
            rep["status"] = "no_teacher_run"
            continue
        rep["reply_issues"] = dict(run["issues"])
        if run["stored_system"] is not None:
            rep["stored_system_prompt_matches_used"] = run["stored_system"] == run["system"]

        # survivors are computed for every reply first — the ontology cards need them even
        # when the paragraphs are unavailable
        parsed = {}
        for pid, raw in run["replies"].items():
            status, raw_triples, surv = guard_pass(raw, run["relations"])
            parsed[pid] = (status, raw_triples, surv)
            if status == "ok":
                for t in surv:
                    survivors_all[paper].append(dict(t, _ontology=ontology,
                                                     _relations=run["relations"]))

        rows, source = load_paragraphs(paper, exdir_for("relation", ontology),
                                       prefer_slugs=(args.teacher,))
        rep["paragraph_source"] = source
        if rows is None:
            rep["status"] = "no_paragraphs"
            continue
        n_meta = run["meta"].get("n_paragraphs")
        if n_meta is not None and len(rows) != n_meta:
            rep["status"] = f"paragraph_count_mismatch ({len(rows)} rendered vs {n_meta} in teacher run)"
            continue
        by_id = {r["para_id"]: r for r in rows}
        rate, n_checked = alignment_rate(
            [(by_id[pid]["paragraph"], [t.get("source_sentence", "") for t in raw_t])
             for pid, (st, raw_t, _s) in parsed.items() if st == "ok" and pid in by_id])
        rep["alignment"] = {"rate": rate, "n_sentences": n_checked}
        if rate is not None and rate < args.min_alignment:
            rep["status"] = f"misaligned ({rate:.2f} < {args.min_alignment})"
            continue

        votes, crep = {}, None
        if args.consensus_models:
            crep = rep["consensus"] = {"models_used": [], "models_skipped": {}, "added": 0,
                                       "paragraphs_gap_filled": 0, "skipped_configures": 0,
                                       "skipped_teacher_related_pair": 0,
                                       "skipped_sentence_not_in_paragraph": 0,
                                       "skipped_internal_conflict": 0}
            votes = consensus_votes(paper, ontology, args, cur_prompts, by_id, n_meta, crep)

        kept = 0
        for pid, row in sorted(by_id.items()):
            if pid not in run["replies"]:
                rep["drops"]["no_reply"] += 1
                continue
            status, raw_triples, surv = parsed[pid]
            if status != "ok":
                rep["drops"][status] += 1
                continue
            if args.configures == "exclude" and any(
                    str(t.get("predicate", "")).strip().lower() == "configures" for t in raw_triples):
                rep["drops"]["configures_open_decision"] += 1
                continue
            if bad_ids and any(triple_id(t, args.teacher) in bad_ids for t in surv):
                rep["drops"]["verdict_bad"] += 1
                continue
            added = consensus_additions(surv, votes.get(pid, {}), row["paragraph"], args, crep) \
                if crep is not None else []
            if added:
                crep["added"] += len(added)
                crep["paragraphs_gap_filled"] += not surv
            target = list(surv) + added
            meta = {"part": "extraction", "paper": paper, "ontology": ontology,
                    "para_id": pid, "teacher": args.teacher,
                    "n_raw": len(raw_triples), "n_kept": len(target)}
            if crep is not None:   # teacher-only data stays byte-identical to earlier builds
                meta.update(n_teacher=len(surv), n_consensus=len(added))
            examples.append({
                "messages": [
                    {"role": "system", "content": run["system"]},
                    {"role": "user", "content": row["user_prompt"]},
                    {"role": "assistant", "content": target_json(target, run["relations"])},
                ],
                "meta": meta,
            })
            kept += 1
        rep["status"] = "ok"
        rep["examples"] = kept
    return examples, survivors_all


def cap_empty(examples, max_frac, rng):
    """Optionally cap the share of {"triples": []} targets. None keeps the teacher's mix."""
    if max_frac is None:
        return examples, 0
    empty = [e for e in examples if e["meta"]["n_kept"] == 0]
    full = [e for e in examples if e["meta"]["n_kept"] > 0]
    allow = int(max_frac / (1 - max_frac) * len(full)) if max_frac < 1 else len(empty)
    if len(empty) <= allow:
        return examples, 0
    rng.shuffle(empty)
    return full + empty[:allow], len(empty) - allow


# ── part 1: ontology cards ───────────────────────────────────────────────────

def _fmt_slot(cls):
    return "any entity (unrestricted)" if cls == "?" else cls


def ontology_examples(ontology, schema, survivors_by_paper, train_papers, args, rng):
    name = ONT_NAMES[ontology]
    excluded = EXCLUDED_BY_DEFAULT if args.configures == "exclude" else set()
    rels = [r for r in schema if r not in excluded]
    base_sys = (f"You are an expert on the {name} ontology used to build knowledge graphs "
                f"from scientific papers. Answer precisely and briefly.")
    rel_list = "\n".join(f"  - {r} ({parse_definition(schema[r])[0]} → "
                         f"{parse_definition(schema[r])[1]})" for r in rels)
    choice_sys = base_sys + "\n\nRelations in this ontology:\n" + rel_list
    out = []

    def add(system, user, answer, kind, **meta):
        out.append({"messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer}],
                    "meta": {"part": "ontology", "kind": kind, "ontology": ontology, **meta}})

    # (a) definition cards — three phrasings per relation
    for r in rels:
        dom, rng_, usage = parse_definition(schema[r])
        d, g = _fmt_slot(dom), _fmt_slot(rng_)
        usage_s = f" {usage.rstrip('.')}." if usage else ""
        add(base_sys, f"In the {name} ontology, what does the relation `{r}` mean?",
            f"`{r}` links a {d} (subject) to a {g} (object).{usage_s}", "definition", relation=r)
        add(base_sys, f"What are the domain and range of `{r}` in the {name} ontology?",
            f"Domain: {d}. Range: {g}. The subject must be a {d} and the object a {g}.",
            "definition", relation=r)
        add(base_sys, f"For `{r}` in the {name} ontology, what goes in the subject slot and "
                      f"what goes in the object slot?",
            f"Subject = {d}, object = {g}: ({d}) {r} ({g}).", "definition", relation=r)

    # pool of TRAIN-paper survivors per relation (never eval papers)
    pool = defaultdict(list)
    for paper in train_papers:
        for t in survivors_by_paper.get(paper, []):
            if t["_ontology"] != ontology:
                continue
            r = canonical_predicate(t["predicate"], t["_relations"])
            if r in rels:
                pool[r].append((paper, t))

    from kg_extraction.fixed_extractor import _splitfrom_is_inverted
    for r in rels:
        items = pool.get(r, [])
        rng.shuffle(items)
        dom, rng_, _u = parse_definition(schema[r])

        # (b) direction judgements: as extracted = yes, swapped = no. Only where a swap is
        # unambiguously wrong: domain != range, or splitFrom where the guard confirms the
        # swapped form is the inverted partition. Same-type relations (comparesAgainst,
        # comprises, cites) are skipped — swapping them is not reliably an error.
        directional = dom != "?" and rng_ != "?" and dom != rng_
        n_dir = 0
        for paper, t in items:
            if n_dir >= args.direction_per_relation:
                break
            s, o, src = t["subject"], t["object"], " ".join(t["source_sentence"].split())
            if not (directional or (r == "splitFrom" and _splitfrom_is_inverted(o, s))):
                continue
            # plain concatenation, not str.format: source sentences can contain braces
            def q(cs, co):
                return (f"Ontology: {name}\nRelation: {r} ({dom} → {rng_})\nSentence: {src}\n"
                        f"Candidate triple: ({cs}) {r} ({co})\n"
                        f"Is this triple correctly directed for `{r}`? Answer yes or no, then "
                        f"give the correctly directed triple.")
            add(base_sys, q(s, o),
                f"Yes. ({s}) {r} ({o}) — the {dom} is the subject and the {rng_} is the object.",
                "direction_pos", relation=r, paper=paper)
            add(base_sys, q(o, s),
                f"No — it is reversed. `{r}` runs {dom} → {rng_}, so the triple is "
                f"({s}) {r} ({o}).", "direction_neg", relation=r, paper=paper)
            n_dir += 1

        # (c) relation choice: sentence + subject + object -> relation
        for paper, t in items[:args.choice_per_relation]:
            src = " ".join(t["source_sentence"].split())
            add(choice_sys,
                f"Sentence: {src}\nSubject: {t['subject']}\nObject: {t['object']}\n"
                f"Which relation from the {name} ontology links this subject to this object? "
                f"Answer with the relation name only.",
                r, "relation_choice", relation=r, paper=paper)
    return out


# ── part 2: vocabulary cards ─────────────────────────────────────────────────

VOCAB_SYS = ("You are an expert in the terminology of computer-science and applied "
             "machine-learning papers. Answer briefly.")
TYPE_QS = [
    'In a scientific paper, what kind of entity is "{e}"? Choose one of: ' + ", ".join(CEO_TYPES) + ".",
    'Classify the term "{e}" as one of: ' + ", ".join(CEO_TYPES) + ". Answer with the type only.",
]


def _entity_csv(paper, entity_dir):
    for name in (f"Entity_{paper}_enriched.csv", f"Entity_{paper}.csv"):
        f = Path(entity_dir) / paper / name
        if f.exists():
            return f
    return None


def vocabulary_examples(train_papers, args, rng, report):
    out, seen = [], set()
    rep = report["vocabulary"] = {"paper_csvs": {}, "gazetteer": None}

    def add(user, answer, kind, **meta):
        out.append({"messages": [{"role": "system", "content": VOCAB_SYS},
                                 {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer}],
                    "meta": {"part": "vocabulary", "kind": kind, **meta}})

    # (a) TRAIN papers' own entity lists first — in-domain terms
    for paper in train_papers:
        f = _entity_csv(paper, args.entity_dir)
        rep["paper_csvs"][paper] = str(f) if f else None
        if f is None:
            continue
        title = ""
        meta_f = OUTPUT_DIR / paper / "kg" / "relation" / args.teacher / "prompt_meta.json"
        if meta_f.exists():
            title = json.loads(meta_f.read_text(encoding="utf-8")).get("title", "")
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                ent = (row.get("Entity") or "").strip()
                typ = (row.get("CEO_Type") or "").strip()
                abbr = (row.get("Abbreviation") or "").strip()
                if ent and typ in CEO_TYPES and ent.lower() not in seen:
                    seen.add(ent.lower())
                    add(TYPE_QS[len(out) % 2].format(e=ent), typ, "type", paper=paper)
                if ent and abbr and title:
                    add(f'In the paper "{title}", what does the abbreviation "{abbr}" stand for?',
                        ent, "abbreviation", paper=paper)

    # (b) the global CS-NER gazetteer — paper-independent, so no split constraint applies
    gaz = Path(args.gazetteer)
    if not gaz.exists():
        rep["gazetteer"] = f"missing: {gaz}"
        print(f"  ! gazetteer not found at {gaz} — vocabulary part will be thin")
        return out
    by_type = defaultdict(list)
    with open(gaz, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ent, typ = (row.get("Entity") or "").strip(), (row.get("CEO_Type") or "").strip()
            try:
                cnt = int(row.get("Count") or 0)
            except ValueError:
                cnt = 0
            if ent and typ in CEO_TYPES and cnt >= args.vocab_min_count and ent.lower() not in seen:
                by_type[typ].append(ent)
    taken = {}
    for typ in CEO_TYPES:
        ents = sorted(by_type.get(typ, []))
        rng.shuffle(ents)
        ents = ents[:args.vocab_per_type]
        taken[typ] = len(ents)
        for i, ent in enumerate(ents):
            seen.add(ent.lower())
            add(TYPE_QS[i % 2].format(e=ent), typ, "type", source="gazetteer")
    rep["gazetteer"] = {"path": str(gaz), "taken_per_type": taken}
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers-file", default=CORPUS_FILE_DEFAULT)
    ap.add_argument("--teacher", default=TEACHER_DEFAULT, help="model slug of the stored replies")
    ap.add_argument("--consensus-models", nargs="*", default=[],
                    help="other stored runs (e.g. gemma4-31b qwen3-235b ministral-14b) whose agreed "
                         "triples are added to the teacher's targets. The verdict filter does not "
                         "apply to them")
    ap.add_argument("--consensus-min", type=int, default=2,
                    help="how many consensus models must extract a triple before it is added")
    ap.add_argument("--ontologies", nargs="+", choices=["ceo", "scinex"], default=["ceo", "scinex"])
    ap.add_argument("--ontology-file", default=SCINEX_OWL_DEFAULT)
    ap.add_argument("--split", default=str(SPLIT_FILE))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13, help="split seed (fixed once the split exists)")
    ap.add_argument("--strata-file", default=None,
                    help="optional CSV paper_id,stratum — folds are balanced within each stratum")
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--dev-fold", type=int, default=None,
                    help="optional fold held out of training for validation loss (never the eval fold)")
    ap.add_argument("--out", default=None,
                    help=f"default {FT_DIR}, or {FT_DIR}/consensus with --consensus-models so the "
                         f"teacher-only data is never overwritten")
    ap.add_argument("--system-prompt", choices=["current", "stored"], default="current",
                    help="current = rendered by today's code (what ft_generate.py sends at eval "
                         "time; scinex needs rdflib); stored = the teacher run's system_prompt.txt")
    ap.add_argument("--min-alignment", type=float, default=0.9,
                    help="min share of teacher source sentences found in their paragraph")
    ap.add_argument("--configures", choices=["exclude", "keep"], default="exclude")
    ap.add_argument("--verdict-filter", choices=["off", "incorrect", "partial"], default="off",
                    help="drop paragraphs holding a triple with a bad C1/C4 verdict. The verdicts "
                         "on disk are claude-opus-5's — see the module docstring")
    ap.add_argument("--max-empty-frac", type=float, default=None,
                    help="cap the share of empty-target extraction examples (default: keep all)")
    ap.add_argument("--gazetteer", default=str(GAZETTEER_DEFAULT))
    ap.add_argument("--entity-dir", default=str(ENTITY_DIR_DEFAULT))
    ap.add_argument("--vocab-per-type", type=int, default=300)
    ap.add_argument("--vocab-min-count", type=int, default=2)
    ap.add_argument("--direction-per-relation", type=int, default=15)
    ap.add_argument("--choice-per-relation", type=int, default=15)
    ap.add_argument("--mix-seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true", help="print the split and counts; write nothing")
    args = ap.parse_args()

    refuse_excluded(args.teacher, "teacher")
    for m in args.consensus_models:
        refuse_excluded(m, "consensus model")
    if args.teacher in args.consensus_models:
        sys.exit("the teacher cannot also be a consensus model")
    if args.consensus_models and args.consensus_min > len(args.consensus_models):
        sys.exit(f"--consensus-min {args.consensus_min} > {len(args.consensus_models)} consensus models")
    if args.out is None:
        args.out = str(FT_DIR / "consensus") if args.consensus_models else str(FT_DIR)
    quiet_guard_logs()
    rng = random.Random(args.mix_seed)

    # ── split ────────────────────────────────────────────────────────────────
    corpus = read_corpus(args.papers_file)
    strata = {}
    if args.strata_file:
        with open(args.strata_file, newline="", encoding="utf-8") as fh:
            strata = {r["paper_id"]: r["stratum"] for r in csv.DictReader(fh)}
    split, new = assign_folds(corpus, args.folds, args.seed, strata, load_split(args.split))
    train, dev, heldout = split_sets(split, corpus, args.eval_fold, args.dev_fold)
    print(f"corpus {len(corpus)} papers, {args.folds} folds"
          + (f" — {len(new)} newly assigned" if new else ""))
    for f in range(args.folds):
        members = sorted(p for p in corpus if split["papers"][p]["fold"] == f)
        tag = " <- EVAL" if f == args.eval_fold else (" <- dev" if f == args.dev_fold else "")
        print(f"  fold {f}: {len(members)} {' '.join(members)}{tag}")

    # ── system prompts ───────────────────────────────────────────────────────
    cur_prompts, schemas = {}, {}
    for ont in args.ontologies:
        if args.system_prompt == "current":
            try:
                cur_prompts[ont] = current_system_prompt(ont, args.ontology_file)
            except ImportError as e:
                sys.exit(f"cannot render the current {ont} system prompt ({e}). Install rdflib, "
                         f"or pass --system-prompt stored.")
            schemas[ont] = schema_from_system_prompt(cur_prompts[ont][0])
        else:
            probe = next((OUTPUT_DIR / p / "kg" / exdir_for("relation", ont) / args.teacher / "system_prompt.txt"
                          for p in train if (OUTPUT_DIR / p / "kg" / exdir_for("relation", ont) / args.teacher / "system_prompt.txt").exists()), None)
            if probe is None:
                sys.exit(f"no stored {ont} system prompt for teacher {args.teacher}")
            schemas[ont] = schema_from_system_prompt(probe.read_text(encoding="utf-8"))

    report = {"extraction": {}, "notes": []}
    bad_ids = {}
    for ont in args.ontologies:
        if args.verdict_filter != "off":
            ids, n = load_bad_ids(args.teacher, ont, args.verdict_filter)
            bad_ids[ont] = ids
            report["notes"].append(f"verdict filter ({args.verdict_filter}) on {ont}: {len(ids)} bad "
                                   f"ids of {n} judged — verdicts by claude-opus-5")

    # ── parts ────────────────────────────────────────────────────────────────
    ext_train, ext_dev, ont_cards = [], [], []
    for ont in args.ontologies:
        ex, surv = extraction_examples(train, ont, args, cur_prompts, report, bad_ids.get(ont, set()))
        ext_train += ex
        if dev:
            dx, _ = extraction_examples(dev, ont, args, cur_prompts, report, bad_ids.get(ont, set()))
            ext_dev += dx
        ont_cards += ontology_examples(ont, schemas[ont], surv, train, args, rng)
    ext_train, capped = cap_empty(ext_train, args.max_empty_frac, rng)
    vocab = vocabulary_examples(train, args, rng, report)

    # ── HARD RULE: nothing from an eval paper ────────────────────────────────
    held = set(heldout)
    for e in ext_train + ont_cards + vocab + ext_dev:
        p = e["meta"].get("paper")
        if p in held:
            sys.exit(f"BUG: an example from eval paper {p} reached the training data — aborting")

    # ── summary ──────────────────────────────────────────────────────────────
    parts = {"extraction": ext_train, "ontology": ont_cards, "vocabulary": vocab}
    print("\nextraction, per paper x ontology:")
    for key, rep in sorted(report["extraction"].items()):
        drops = ", ".join(f"{k} {v}" for k, v in sorted(rep.get("drops", {}).items()))
        al = rep.get("alignment", {}).get("rate")
        c = rep.get("consensus")
        print(f"  {key:32s} {rep.get('status','?'):12.60s} ex={rep.get('examples', 0):3d}"
              + (f" align={al:.2f}" if al is not None else "")
              + (f" +{c['added']} consensus" if c else "")
              + (f"  drops: {drops}" if drops else "")
              + (f"  reply issues: {rep['reply_issues']}" if rep.get("reply_issues") else "")
              + (f"  consensus skipped models: {c['models_skipped']}" if c and c["models_skipped"] else ""))
    if args.consensus_models:
        tot = Counter()
        for rep in report["extraction"].values():
            tot.update({k: v for k, v in rep.get("consensus", {}).items() if isinstance(v, int)})
        n_t = sum(e["meta"].get("n_teacher", 0) for e in ext_train)
        print(f"\nconsensus (>= {args.consensus_min} of {' '.join(args.consensus_models)}): "
              f"+{tot['added']} triples on top of the teacher's {n_t} in train "
              f"({tot['paragraphs_gap_filled']} teacher-empty paragraphs now have a target); skipped: "
              f"teacher already related the pair {tot['skipped_teacher_related_pair']}, "
              f"relation conflict {tot['skipped_internal_conflict']}, "
              f"configures {tot['skipped_configures']}, "
              f"sentence not in paragraph {tot['skipped_sentence_not_in_paragraph']}")
    n_empty = sum(1 for e in ext_train if e["meta"]["n_kept"] == 0)
    print("\nparts (train):")
    for name, rows in parts.items():
        chars = sum(len(m["content"]) for e in rows for m in e["messages"])
        kinds = Counter(e["meta"].get("kind", e["meta"].get("ontology", "")) for e in rows)
        print(f"  {name:10s} {len(rows):5d} examples  ~{chars / 4 / 1e6:.2f}M tokens  {dict(kinds)}")
    print(f"  extraction targets that are empty: {n_empty}"
          + (f" (capped away {capped})" if capped else ""))
    if dev:
        print(f"  dev extraction examples: {len(ext_dev)}")
    for n in report["notes"]:
        print(f"  note: {n}")
    not_ok = [k for k, r in report["extraction"].items() if r.get("status") != "ok"]
    if not_ok:
        print(f"\n  ! {len(not_ok)} paper x ontology runs produced NO extraction examples "
              f"(see status column). Extraction pairs need output/<id>/no-llm/output.html or a "
              f"stored prompts.jsonl on this machine.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ── write ────────────────────────────────────────────────────────────────
    Path(args.split).parent.mkdir(parents=True, exist_ok=True)
    Path(args.split).write_text(json.dumps(split, indent=2), encoding="utf-8")
    out = Path(args.out) / f"fold{args.eval_fold}"
    (out / "parts").mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        write_jsonl(out / "parts" / f"{name}.jsonl", rows)
    mixed = ext_train + ont_cards + vocab
    rng.shuffle(mixed)
    write_jsonl(out / "train.jsonl", mixed)
    if dev:
        write_jsonl(out / "dev.jsonl", ext_dev)
    (out / "eval_papers.txt").write_text("\n".join(heldout) + "\n", encoding="utf-8")
    (out / "train_papers.txt").write_text("\n".join(train) + "\n", encoding="utf-8")

    manifest = {
        "created": date.today().isoformat(),
        "split_file": str(args.split), "folds": args.folds, "split_seed": args.seed,
        "eval_fold": args.eval_fold, "dev_fold": args.dev_fold,
        "eval_papers": heldout, "train_papers": train, "dev_papers": dev,
        "teacher": args.teacher, "ontologies": args.ontologies,
        "consensus_models": args.consensus_models,
        "consensus_min": args.consensus_min if args.consensus_models else None,
        "system_prompt": args.system_prompt, "configures": args.configures,
        "verdict_filter": args.verdict_filter, "max_empty_frac": args.max_empty_frac,
        "counts": {k: len(v) for k, v in parts.items()} | {"dev_extraction": len(ext_dev),
                                                            "empty_extraction_targets": n_empty},
        "files": {p.name: sha256_file(p) for p in [out / "train.jsonl"] + ([out / "dev.jsonl"] if dev else [])},
        "report": json.loads(json.dumps(report, default=lambda o: dict(o))),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}/train.jsonl ({len(mixed)} examples), manifest.json, eval_papers.txt"
          f"\nsplit -> {args.split}")
    print(f"next: python3 ft_train_qlora.py --data {out} --base-model Qwen/Qwen3-8B "
          f"--run-name qwen3-8b-f{args.eval_fold} --dry-run")


if __name__ == "__main__":
    main()
