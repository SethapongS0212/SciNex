#!/usr/bin/env python3
"""
ontology_eval.py
----------------
Six-criteria ontology-quality evaluation (C1-C6), replacing the single
CORRECT/PARTIAL/INCORRECT triple rubric.

    C1  Concept Correctness   concepts are semantically correct and evidenced in the text
    C2  Concept Completeness  the ontology covers the important concepts in the text
    C3  Concept Specificity   concepts sit at an appropriate granularity
    C4  Relation Correctness  the relation holds per the text and fits its domain -> range
    C5  Relation Completeness the important relations in the text are all extracted
    C6  Semantic Consistency  concepts / hierarchy / relations do not contradict each other

Refs: Zhang, Conia & Rago (IJCNLP-AACL 2025) for C1/C3/C4;
      Wilson et al., Semantic Web 14(6) 2023 for C2/C5/C6.

WHY THREE HARNESSES, NOT ONE
============================
The criteria do not share a unit of analysis, and forcing them into one would make
C2/C5 unanswerable:

    C1, C3, C4  judged PER TRIPLE      against its source sentence   (precision-like)
    C2, C5      judged PER PARAGRAPH   against every triple drawn from it (recall-like)
    C6          judged PER PAPER       against that paper's whole triple set

A completeness question cannot be asked of a single triple: what is missing is by
definition not in front of the judge. So C2/C5 need the paragraph plus the full set
of triples extracted from it.

USAGE
    python3 ontology_eval.py triples    --model qwen3-235b --n 60   # C1, C3, C4
    python3 ontology_eval.py paragraphs --model qwen3-235b --n 25   # C2, C5
    python3 ontology_eval.py papers     --model qwen3-235b --n 8    # C6
    #  ... a judge fills in replies/reply_<name>.txt ...
    python3 ontology_eval.py collect    --model qwen3-235b --kind triples
    python3 ontology_eval.py report     --model qwen3-235b

The judge here is claude-opus-5 running IN SESSION (no API spend). A model must never
judge its own extraction, so claude-opus-5's own extraction is excluded from `--model`.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_DIR = Path("output")
EVAL_DIR = Path("ontology_eval")

# ── scales ───────────────────────────────────────────────────────────────────
# Three levels per criterion: enough resolution to separate models, few enough
# that two judges can agree on the boundaries (and that Cohen's kappa stays defined).
SCALES = {
    "C1": ["CORRECT", "PARTIAL", "INCORRECT"],
    "C3": ["APPROPRIATE", "TOO_BROAD", "TOO_NARROW"],
    "C4": ["CORRECT", "PARTIAL", "INCORRECT"],
    "C2": ["COMPLETE", "PARTIAL", "POOR"],
    "C5": ["COMPLETE", "PARTIAL", "POOR"],
    "C6": ["CONSISTENT", "MINOR_ISSUES", "CONTRADICTORY"],
}
# Scoring is PER CRITERION, because the scales are not all ordinal.
# C1/C2/C4/C5/C6 are graded (best / middle / worst), so 1.0 / 0.5 / 0.0.
# C3 is NOT: TOO_BROAD and TOO_NARROW are two ways of being wrong, not a better
# and a worse one. Scoring one at 0.5 would silently assert that over-general
# concepts are half-acceptable while over-specific ones are worthless. Both score
# 0.0; the direction is kept in the label for diagnosis.
POINTS_BY_CRIT = {
    "C1": {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0},
    "C2": {"COMPLETE": 1.0, "PARTIAL": 0.5, "POOR": 0.0},
    "C3": {"APPROPRIATE": 1.0, "TOO_BROAD": 0.0, "TOO_NARROW": 0.0},
    "C4": {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0},
    "C5": {"COMPLETE": 1.0, "PARTIAL": 0.5, "POOR": 0.0},
    "C6": {"CONSISTENT": 1.0, "MINOR_ISSUES": 0.5, "CONTRADICTORY": 0.0},
}


def _norm(s):
    """Whitespace-collapsed, lowercased — for matching a model's source_sentence
    back to the paragraph it came from. Without this, matching fails completely:
    the model re-wraps the sentence, so a raw substring test scored 0/12 on a
    paper where the normalised test scores 12/12."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


RUBRIC_TRIPLES = """########## RUBRIC — C1, C3, C4 (per triple) ##########

You are grading knowledge-graph triples extracted from scientific papers. For each triple
you get: the triple, the ONTOLOGY DEFINITION of its predicate (allowed domain -> range),
and the SOURCE SENTENCE it was taken from.

Judge ONLY from the source sentence. A triple can be true about the paper and still wrong
here, because the sentence is the extractor's evidence.

C1 — CONCEPT CORRECTNESS: are the subject and object valid, meaningful concepts that are
     actually present in the sentence?
  CORRECT    both are real, well-formed concepts and both appear in (or are unambiguously
             denoted by) the sentence
  PARTIAL    one side is vague, a fragment, or only loosely denoted by the sentence
  INCORRECT  a side is not a concept at all (a cross-reference, a section title, a bare
             number where an entity is required), or does not come from this sentence

C3 — CONCEPT SPECIFICITY: is the granularity right for a scientific KG?
  APPROPRIATE  specific enough to be useful, general enough to recur ("Random Forest",
               "F1-score", "oxide scale")
  TOO_BROAD    a generic placeholder that would merge unrelated things ("the model",
               "data", "method", "system")
  TOO_NARROW   an over-specified fragment that will never match anything else
               ("the 3rd convolutional layer of our second variant")

C4 — RELATION CORRECTNESS: does the stated relation hold, in that direction, per the text?
  CORRECT    the sentence explicitly supports it AND the subject/object types match the
             predicate's domain -> range
  PARTIAL    supported but loosely typed, or implied rather than stated
  INCORRECT  unsupported, or subject and object are swapped relative to domain -> range

Return ONE JSON object, no markdown fences, no commentary:

{"verdicts": [
  {"triple_id": "<id>", "C1": "...", "C3": "...", "C4": "...", "note": "<short, only if not all-best>"}
]}

Include every triple_id shown.
"""

RUBRIC_PARAGRAPHS = """########## RUBRIC — C2, C5 (per paragraph) ##########

You are grading the COMPLETENESS of an extraction. For each item you get a PARAGRAPH from a
scientific paper and EVERY triple the extractor produced from it (sometimes none).

Your question is the opposite of precision: **what did the extractor MISS?**

Only count things a knowledge graph of this kind should hold: named methods, models,
datasets, metrics, materials, tasks, and the explicit relations between them. Do NOT count
narrative asides, hedges, citations, or background prose.

C2 — CONCEPT COMPLETENESS: are the important concepts in this paragraph present in the triples?
  COMPLETE  every important concept appears somewhere in the triples
  PARTIAL   some important concepts captured, others missed
  POOR      most important concepts missing, or nothing extracted from a content-rich paragraph

C5 — RELATION COMPLETENESS: are the important relations stated in this paragraph extracted?
  COMPLETE  every clearly-stated important relation is captured
  PARTIAL   some captured, others missed
  POOR      most missed, or nothing extracted where relations are clearly stated

If a paragraph genuinely holds nothing extractable (pure narrative, a heading, an author
list), score BOTH as COMPLETE and say so in the note — an empty extraction is correct there.

Return ONE JSON object, no fences:

{"verdicts": [
  {"para_key": "<key>", "C2": "...", "C5": "...",
   "missed_concepts": ["..."], "missed_relations": ["subject -> relation -> object"],
   "note": "<short>"}
]}
"""

RUBRIC_PAPERS = """########## RUBRIC — C6 (per paper) ##########

You are grading SEMANTIC CONSISTENCY across one paper's whole extracted triple set.

Look for contradictions and incoherence, not for individual wrong triples (those are C1/C4):
  * the same concept used with conflicting meanings or at conflicting levels
  * mutually exclusive relations asserted between the same pair
  * hierarchy/direction violations (X part-of Y and Y part-of X; a task "addressing" a task)
  * the same real entity split across surface variants that should be one node
    ("RT-DETR" / "RT-DETR model" / "the RT-DETR")
  * a relation whose domain/range is systematically misused across this paper

C6:
  CONSISTENT     no contradictions; surface variants are minor and would merge cleanly
  MINOR_ISSUES   some duplication or loose typing, but nothing self-contradictory
  CONTRADICTORY  real contradictions, or systematic misuse of a relation

Return ONE JSON object, no fences:

{"verdicts": [
  {"paper": "<id>", "C6": "...", "issues": ["..."], "note": "<short>"}
]}
"""


def _schema_for(ontology, ontology_file="scinex_refined_14.owl"):
    """The predicate -> 'domain -> range | usage' map the judge needs.

    Without it the bundles print '(not recorded)' and C4 collapses into "does this
    English relation sound plausible?" — exactly the failure the criterion exists to
    prevent, since half its definition is domain/range conformance.
    """
    from kg_extraction.fixed_extractor import _CEO_SCHEMA
    if ontology == "ceo":
        return dict(_CEO_SCHEMA)
    # NOTE: this used to import `_load_scinex_schema` from fixed_extractor, which
    # DOES NOT EXIST. The import raised, the bare `except` swallowed it, and every
    # scinex bundle silently got the CEO schema instead — so scinex-only relations
    # printed "(predicate not in this ontology)" and C4 degraded to a plausibility
    # check on the English relation name, which is precisely the failure this
    # function exists to prevent. The real loader is ontology_loader.load_ontology,
    # which returns (relations, schema). Failing loudly here is deliberate: a silent
    # CEO fallback invalidates every scinex C4 verdict.
    from kg_extraction.ontology_loader import load_ontology
    _relations, schema = load_ontology(ontology_file)
    if not schema:
        raise RuntimeError(f"no predicate schema parsed from {ontology_file} — "
                           f"refusing to judge scinex against the CEO schema")
    return dict(schema)


def _triples_for(paper, slug, extractor_dir):
    f = OUTPUT_DIR / paper / "kg" / extractor_dir / slug / "triples.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d.get("triples", [])


# ── corpus allowlist ──────────────────────────────────────────────────────────
# The corpus is the user's 20-paper list. `aiabstract2025` and `routepred2023`
# were SUBSTITUTES for `tripplanner2020` / `linkpred2015` and were retired once
# those PDFs were supplied, but their parsed output is still on disk — so a bare
# directory scan silently redraws them and reintroduces the wrong 22-paper frame.
#
# This DEFAULTS ON (papers_20.txt is read if present) rather than requiring a
# flag: forgetting a flag reintroduces the bug silently, which is exactly how the
# 22-paper framing survived three sessions. Pass --papers-file to override, or
# --papers-file '' to scan every parsed paper.
CORPUS_FILE_DEFAULT = "papers_20.txt"
_CORPUS = None  # None = no filter; set by _set_corpus()


def _set_corpus(path):
    """Install the allowlist. Empty/missing path disables filtering."""
    global _CORPUS
    if not path:
        _CORPUS = None
        return
    f = Path(path)
    if not f.exists():
        print(f"  ! corpus file {path} not found — scanning ALL parsed papers")
        _CORPUS = None
        return
    _CORPUS = set(f.read_text(encoding="utf-8").split())
    print(f"  corpus: {len(_CORPUS)} papers from {path}")


def _papers_with(slug, extractor_dir):
    found = sorted(p.name for p in OUTPUT_DIR.iterdir()
                   if (p / "kg" / extractor_dir / slug / "triples.json").exists())
    if _CORPUS is None:
        return found
    kept = [p for p in found if p in _CORPUS]
    skipped = [p for p in found if p not in _CORPUS]
    if skipped:
        print(f"  excluded {len(skipped)} paper(s) outside the corpus: {', '.join(skipped)}")
    return kept


def _tid(t, slug):
    raw = f"{slug}|{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _write_batches(dest, rubric, items, size, header):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "replies").mkdir(exist_ok=True)
    for old in dest.glob("*.txt"):
        old.unlink()
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    for bi, chunk in enumerate(chunks, 1):
        parts = [rubric] if bi == 1 else []
        parts.append(header.format(n=len(chunk)))
        parts.extend(chunk)
        (dest / f"{bi:02d}_batch_{bi:02d}.txt").write_text("".join(parts), encoding="utf-8")
    return len(chunks)


# ── C1 / C3 / C4 ─────────────────────────────────────────────────────────────
def cmd_triples(args):
    ex_dir = "relation" if args.ontology == "ceo" else "relation_scinex"
    papers = _papers_with(args.model, ex_dir)
    pool = []
    for p in papers:
        for t in _triples_for(p, args.model, ex_dir):
            t["_paper"] = p
            t["_id"] = _tid(t, args.model)
            pool.append(t)
    if not pool:
        sys.exit(f"no triples for {args.model} / {args.ontology}")

    # round-robin over predicates so rare relations are represented
    rng = random.Random(args.seed)
    by_pred = defaultdict(list)
    for t in pool:
        by_pred[t["predicate"].lower()].append(t)
    for v in by_pred.values():
        rng.shuffle(v)
    picked, preds = [], sorted(by_pred, key=lambda k: -len(by_pred[k]))
    while len(picked) < min(args.n, len(pool)):
        progressed = False
        for k in preds:
            if by_pred[k]:
                picked.append(by_pred[k].pop())
                progressed = True
                if len(picked) >= args.n:
                    break
        if not progressed:
            break

    schema = _schema_for(args.ontology)

    def _defn(pred):
        want = str(pred).lower().replace(" ", "")
        for k, v in schema.items():
            if k.lower() == want:
                return v
        return "(predicate not in this ontology)"

    rows = []
    for t in picked:
        rows.append(
            f"\n----- triple_id: {t['_id']} -----\n"
            f"TRIPLE:     ({t['subject']})  --{t['predicate']}-->  ({t['object']})\n"
            f"PREDICATE DEFINITION: {_defn(t['predicate'])}\n"
            f"SOURCE SENTENCE: {' '.join(str(t.get('source_sentence','')).split())}\n")

    dest = EVAL_DIR / args.model / args.ontology / "triples"
    nb = _write_batches(dest, RUBRIC_TRIPLES, rows, args.size,
                        "\nGrade the following {n} triples on C1, C3, C4.\n")
    idx = dest / "index.csv"
    with open(idx, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["triple_id", "paper", "subject", "predicate", "object", "source_sentence"])
        for t in picked:
            w.writerow([t["_id"], t["_paper"], t["subject"], t["predicate"], t["object"],
                        " ".join(str(t.get("source_sentence", "")).split())])
    print(f"{len(picked)} triples -> {nb} batch file(s) in {dest}")
    print(f"index: {idx}")


# ── C2 / C5 ──────────────────────────────────────────────────────────────────
def cmd_paragraphs(args):
    ex_dir = "relation" if args.ontology == "ceo" else "relation_scinex"
    papers = _papers_with(args.model, ex_dir)
    units = []
    for p in papers:
        pf = OUTPUT_DIR / p / "kg" / ex_dir / args.model / "prompts.jsonl"
        if not pf.exists():
            continue
        prompts = [json.loads(l) for l in open(pf, encoding="utf-8") if l.strip()]
        trips = _triples_for(p, args.model, ex_dir)
        # attach triples to their paragraph by source-sentence containment
        for pr in prompts:
            body = pr["user_prompt"].split("Extract triples using")[0].strip()
            text = body.split("Text:", 1)[-1].strip()
            ntext = _norm(text)
            matched = [t for t in trips
                       if _norm(t.get("source_sentence", ""))[:60]
                       and _norm(t.get("source_sentence", ""))[:60] in ntext]
            units.append({"paper": p, "para_id": pr["para_id"], "text": text,
                          "triples": matched})
    if not units:
        sys.exit(f"no prompts.jsonl for {args.model} — paragraph-level eval needs them")

    # STRATIFIED by whether the extractor produced anything, then reweighted at report
    # time — the same design the triple sampler uses for predicates, and for the same
    # reason. A sparse extractor yields nothing from most of the corpus (qwen3-235b:
    # 84% of paragraphs are empty), so a pure random draw spends ~84% of the judging
    # effort on empties and can miss the productive stratum entirely (seed 11 drew 0
    # of 20). Sampling both strata and weighting by their true share gives the same
    # expectation with far less judging effort.
    #
    # Empty paragraphs are NOT dropped: an extractor that silently skips content-rich
    # text is exactly what C2/C5 must catch, and the rubric lets the judge mark a
    # genuinely-empty paragraph COMPLETE.
    rng = random.Random(args.seed)
    with_t = [u for u in units if u["triples"]]
    without = [u for u in units if not u["triples"]]
    rng.shuffle(with_t)
    rng.shuffle(without)
    half = args.n // 2
    picked = with_t[:min(half, len(with_t))]
    picked += without[:args.n - len(picked)]
    for u in picked:
        u["stratum"] = "has_triples" if u["triples"] else "empty"
    # true population share of each stratum, for reweighting in `report`
    strata_share = {"has_triples": len(with_t) / len(units),
                    "empty": len(without) / len(units)}
    rng.shuffle(picked)

    rows = []
    for u in picked:
        key = f"{u['paper']}#{u['para_id']}"
        tl = "\n".join(f"    ({t['subject']}) --{t['predicate']}--> ({t['object']})"
                       for t in u["triples"]) or "    (none extracted)"
        rows.append(f"\n----- para_key: {key} -----\nPARAGRAPH:\n{u['text']}\n\n"
                    f"TRIPLES EXTRACTED ({len(u['triples'])}):\n{tl}\n")

    dest = EVAL_DIR / args.model / args.ontology / "paragraphs"
    nb = _write_batches(dest, RUBRIC_PARAGRAPHS, rows, args.size,
                        "\nGrade the following {n} paragraphs on C2 and C5.\n")
    with open(dest / "index.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["para_key", "paper", "para_id", "stratum", "n_triples"])
        for u in picked:
            wtr.writerow([f"{u['paper']}#{u['para_id']}", u["paper"], u["para_id"],
                          u["stratum"], len(u["triples"])])
    (dest / "strata.json").write_text(json.dumps({
        "share": strata_share,
        "sampled": {k: sum(1 for u in picked if u["stratum"] == k)
                    for k in ("has_triples", "empty")},
        "population": {"has_triples": len(with_t), "empty": len(without)},
    }, indent=2), encoding="utf-8")
    print(f"{len(picked)} paragraphs -> {nb} batch file(s) in {dest}")
    print(f"  sampled   : {sum(1 for u in picked if u['stratum']=='has_triples')} with triples, "
          f"{sum(1 for u in picked if u['stratum']=='empty')} empty")
    print(f"  population: {strata_share['has_triples']:.1%} of paragraphs produced a triple "
          f"({len(with_t)}/{len(units)}) — reweighted to this at report time")


# ── C6 ───────────────────────────────────────────────────────────────────────
def cmd_papers(args):
    ex_dir = "relation" if args.ontology == "ceo" else "relation_scinex"
    papers = _papers_with(args.model, ex_dir)
    rng = random.Random(args.seed)
    rng.shuffle(papers)
    picked = papers[:args.n]
    rows = []
    for p in picked:
        trips = _triples_for(p, args.model, ex_dir)
        tl = "\n".join(f"    ({t['subject']}) --{t['predicate']}--> ({t['object']})"
                       for t in trips) or "    (none)"
        rows.append(f"\n----- paper: {p} -----\nALL TRIPLES ({len(trips)}):\n{tl}\n")
    dest = EVAL_DIR / args.model / args.ontology / "papers"
    nb = _write_batches(dest, RUBRIC_PAPERS, rows, args.size,
                        "\nGrade the following {n} papers on C6.\n")
    print(f"{len(picked)} papers -> {nb} batch file(s) in {dest}")


# ── collect ──────────────────────────────────────────────────────────────────
_OBJ_RE = re.compile(r'\{[^{}]*\}')


def cmd_collect(args):
    dest = EVAL_DIR / args.model / args.ontology / args.kind
    rep = dest / "replies"
    if not rep.exists():
        sys.exit(f"no replies dir at {rep}")
    key = {"triples": "triple_id", "paragraphs": "para_key", "papers": "paper"}[args.kind]
    crits = {"triples": ["C1", "C3", "C4"], "paragraphs": ["C2", "C5"],
             "papers": ["C6"]}[args.kind]
    # Restrict to the CURRENT sample. `replies/` accumulates across redraws, so
    # without this a verdict for a triple that is no longer sampled — e.g. one a
    # new guard has since removed from the corpus — would silently survive into
    # verdicts.csv and be scored. Carrying forward verdicts for ids that ARE still
    # in the sample is intended: a verdict depends only on the triple, its source
    # sentence and the rubric, none of which change when the sample is redrawn.
    sampled = None
    idx = dest / "index.csv"
    if idx.exists():
        with open(idx, encoding="utf-8") as fh:
            sampled = {r[key] for r in csv.DictReader(fh) if r.get(key)}

    out, seen, dropped = [], set(), 0
    for f in sorted(rep.glob("*.txt")):
        txt = f.read_text(encoding="utf-8")
        objs = []
        try:
            objs = json.loads(re.sub(r'^```\w*|```$', '', txt.strip(),
                                     flags=re.M)).get("verdicts", [])
        except Exception:
            for m in _OBJ_RE.finditer(txt):          # salvage a broken tail
                try:
                    objs.append(json.loads(m.group(0)))
                except Exception:
                    pass
        for o in objs:
            k = o.get(key)
            if not k or k in seen:
                continue
            if sampled is not None and k not in sampled:
                dropped += 1
                continue
            seen.add(k)
            row = {key: k}
            for c in crits:
                row[c] = str(o.get(c, "")).strip().upper()
            for extra in ("missed_concepts", "missed_relations", "issues"):
                if extra in o:
                    row[extra] = "; ".join(o[extra]) if isinstance(o[extra], list) else str(o[extra])
            row["note"] = str(o.get("note", ""))
            out.append(row)
    if not out:
        sys.exit("no verdicts parsed")
    if dropped:
        print(f"  dropped {dropped} stale verdict(s) not in the current sample")
    if sampled is not None:
        missing = sampled - seen
        if missing:
            print(f"  ⚠ {len(missing)} sampled item(s) still UNJUDGED: "
                  + ", ".join(sorted(missing)[:8]) + ("..." if len(missing) > 8 else ""))
    cols = [key] + crits + ["missed_concepts", "missed_relations", "issues", "note"]
    csv_f = dest / "verdicts.csv"
    with open(csv_f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"{len(out)} verdicts -> {csv_f}")


# ── report ───────────────────────────────────────────────────────────────────
def cmd_report(args):
    print(f"{'model':22s} {'ont':7s} " + " ".join(f"{c:>8s}" for c in
          ["C1", "C2", "C3", "C4", "C5", "C6"]))
    print("  " + "-" * 78)
    for model in args.model:
        for ont in args.ontology:
            scores = {}
            for kind, crits in (("triples", ["C1", "C3", "C4"]),
                                ("paragraphs", ["C2", "C5"]), ("papers", ["C6"])):
                f = EVAL_DIR / model / ont / kind / "verdicts.csv"
                if not f.exists():
                    continue
                rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
                for c in crits:
                    pts = POINTS_BY_CRIT[c]
                    vals = [pts[r[c]] for r in rows if r.get(c) in pts]
                    if vals:
                        scores[c] = sum(vals) / len(vals)
            if not scores:
                continue
            cells = " ".join(f"{scores[c]:>7.1%}" if c in scores else f"{'—':>8s}"
                             for c in ["C1", "C2", "C3", "C4", "C5", "C6"])
            print(f"  {model:20s} {ont:7s} {cells}")
    print("\n  1.0 = best level, 0.5 = middle, 0.0 = worst; mean over judged items")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, dn in (("triples", cmd_triples, 60), ("paragraphs", cmd_paragraphs, 25),
                         ("papers", cmd_papers, 8)):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--ontology", default="ceo", choices=["ceo", "scinex"])
        s.add_argument("--n", type=int, default=dn)
        s.add_argument("--size", type=int, default=10 if name == "triples" else 5)
        s.add_argument("--seed", type=int, default=11)
        s.add_argument("--papers-file", default=CORPUS_FILE_DEFAULT,
                       help="file of whitespace-separated paper ids to restrict the draw to "
                            "(default papers_20.txt; pass '' to use every parsed paper)")
        s.set_defaults(func=fn)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--ontology", default="ceo", choices=["ceo", "scinex"])
    c.add_argument("--kind", required=True, choices=["triples", "paragraphs", "papers"])
    c.set_defaults(func=cmd_collect)
    r = sub.add_parser("report")
    r.add_argument("--model", nargs="+", required=True)
    r.add_argument("--ontology", nargs="+", default=["ceo"])
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    if hasattr(args, "papers_file"):
        _set_corpus(args.papers_file)
    args.func(args)


if __name__ == "__main__":
    main()
