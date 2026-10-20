#!/usr/bin/env python3
"""
qa_eval.py
----------
Part 3 — evaluate a knowledge graph by SIMULATED Q&A against the real paper.

C1-C6 ask whether each triple is right. This asks whether the graph is USEFUL: can
someone who sees only the graph answer factual questions whose answers the paper
states? It prices precision and completeness together, in one number per extractor.

    generate   questions + gold answers from the paper's own paragraphs (one call per
               paragraph). Every question is checked in code: the answer must be copied
               verbatim from the paragraph, the evidence sentence must be in it, the
               question must not contain its answer, boilerplate is rejected. The SAME
               question set is then reused for every KG, so comparisons are paired.
    answer     an answerer model sees ONE condition and nothing else:
                 kg          the paper's triples only — (subject) --relation--> (object),
                             no source sentences (they would leak the text)
                 oracle      the paragraph the question came from   (ceiling)
                 closedbook  nothing at all                          (floor: what the
                             answerer already knows about these published papers)
    grade      identical (question, predicted answer) pairs get ONE verdict, whichever
               condition produced them. Exact normalised matches are CORRECT; abstentions
               are INCORRECT. The rest, by default (--method overlap), get a DETERMINISTIC
               verdict from SQuAD-style token overlap with the reference over content words:
                 CORRECT    F1 >= 0.8, or every reference word present and precision >= 0.5
                 PARTIAL    every reference word present (too much extra), or F1 >= 0.4
                 INCORRECT  otherwise
               No LLM judges anything. Stricter than a human on verbose answers ("average
               accuracy of 83.0% on X" vs "83.0%" -> PARTIAL). --method llm restores the blind
               LLM grader. One method per run; mixing is refused.
    report     accuracy per condition with a paper-cluster bootstrap CI, per relation and
               per paper; `answer_in_kg` (deterministic, no LLM: is the gold answer a node
               of the graph?) and `grounded` (does a correct KG answer actually come from
               the triples it cites?); and paired A-vs-B comparisons.

    python3 qa_eval.py generate --run f0 --papers-from ft_data/fold0/eval_papers.txt \
            --provider openai-compat --base-url https://openrouter.ai/api/v1 \
            --model google/gemma-4-31b-it
    python3 qa_eval.py answer --run f0 --context kg \
            --kg relation/qwen3-8b-base-hf relation/qwen3-8b-ft-cv relation/gptoss-120b \
            --provider ... --model ...
    python3 qa_eval.py answer --run f0 --context oracle     --provider ... --model ...
    python3 qa_eval.py answer --run f0 --context closedbook --provider ... --model ...
    python3 qa_eval.py grade  --run f0 --provider ... --model ...
    python3 qa_eval.py report --run f0 --compare kg__relation__qwen3-8b-ft-cv kg__relation__qwen3-8b-base-hf

Outputs under qa_eval/<run>/: run.json, qgen_raw.jsonl, questions.jsonl, rejected.jsonl,
answers/<condition>.jsonl, grades.jsonl, report.json.

Model choice (recorded in run.json and the answer files; the report prints it):
  * never Claude — refused in code, as everywhere in this track;
  * the question GENERATOR should not be the finetuning teacher (gpt-oss-120b): a teacher
    writes questions about the facts it would itself extract, which favours its students;
  * use ONE answerer and ONE grader for every condition in a run, or the conditions are not
    comparable (the report warns if answer files disagree).
Calls go through run_api_extract.call (stdlib only; keys from .env; Ollama gets think=False).
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from ft_common import (  # noqa: E402
    CORPUS_FILE_DEFAULT, OUTPUT_DIR, TEACHER_DEFAULT, load_paragraphs, norm, read_corpus,
    read_jsonl, refuse_excluded, write_jsonl,
)

QA_DIR = Path("qa_eval")
PROMPT_VERSION = "qa-v1"
POINTS = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
GENERIC_REFS = ("the proposed", "the authors", "this paper", "this study", "this work",
                "this method", "this approach", "this model", "our model", "our method")

# ── prompts (placeholders filled with str.replace — the JSON examples contain braces) ──

QGEN_SYSTEM = """You write evaluation questions for a knowledge graph that was extracted from a scientific paper.

You will be given ONE paragraph of the paper. Write up to <<K>> short factual questions whose answers are stated EXPLICITLY in that paragraph.

Rules:
1. The answer is a short span COPIED VERBATIM from the paragraph (1-8 words): the name of a model, method, dataset, metric value, task, tool, material, component or setting.
2. The question names the specific thing it asks about, as written in the paragraph. Never write "the proposed model", "this method", "the authors".
3. The question must not contain its own answer, and must be answerable from the paragraph alone.
4. Ask only about scientific content: what a method uses, produces or consists of; what it was trained or evaluated on; what score it achieved; what task it addresses; what it was compared against; how data was split. Never ask about authors, affiliations, funding, licences or bibliography entries.
5. One fact per question, and no two questions about the same fact.
6. If the paragraph states no such facts, return {"questions": []}.
7. For each question give the verbatim sentence that states the answer ("evidence") and the relation from this list that best describes the fact, or "other" ("relation"):
   <<RELATIONS>>

Return ONLY JSON, no markdown:
{"questions": [{"question": "...", "answer": "...", "evidence": "...", "relation": "..."}]}"""

ANSWER_SYSTEM = {
    "kg": """You answer questions about ONE scientific paper using ONLY a knowledge graph extracted from it.
The knowledge graph is a numbered list of triples: [n] (subject) --relation--> (object).

Rules:
1. Use only the triples. Do not use outside knowledge and do not guess.
2. If no triple (or combination of triples) states the answer, return "answerable": false and "answer": "".
3. Otherwise give a short answer (a name, value or phrase taken from the triples) and list the numbers of the triples you used in "triples".
4. Answer every question id you are given.

Return ONLY JSON, no markdown:
{"answers": [{"qid": "q1", "answerable": true, "answer": "...", "triples": [3, 17]}]}""",
    "oracle": """You answer questions about ONE scientific paper using ONLY the paragraph given.

Rules:
1. Use only the paragraph. Do not use outside knowledge and do not guess.
2. If the paragraph does not state the answer, return "answerable": false and "answer": "".
3. Otherwise give a short answer (a name, value or phrase).
4. Answer every question id you are given.

Return ONLY JSON, no markdown:
{"answers": [{"qid": "q1", "answerable": true, "answer": "..."}]}""",
    "closedbook": """You answer questions about a scientific paper WITHOUT seeing the paper.

Rules:
1. Answer only if you are confident from general knowledge; otherwise return "answerable": false and "answer": "".
2. Give a short answer (a name, value or phrase).
3. Answer every question id you are given.

Return ONLY JSON, no markdown:
{"answers": [{"qid": "q1", "answerable": true, "answer": "..."}]}""",
}

GRADE_SYSTEM = """You grade short answers to questions about a scientific paper against a reference answer.

For each item decide:
  CORRECT   - the predicted answer states the same fact as the reference (paraphrase, abbreviation vs. full name, extra harmless detail, or equivalent number formatting are fine).
  PARTIAL   - it overlaps with the reference but is incomplete or imprecise (a broader or narrower term, one of several required items, a number with the wrong qualifier).
  INCORRECT - a different fact, contradicts the reference, or is empty.
Judge only against the reference answer; do not use outside knowledge about the paper.

Return ONLY JSON, no markdown:
{"grades": [{"item": "g1", "verdict": "CORRECT"}]}"""


# ── plumbing ─────────────────────────────────────────────────────────────────

def add_api_args(p, required=True):
    p.add_argument("--provider", required=required,
                   choices=["openai", "openai-compat", "gemini", "ollama", "hf"],
                   help="hf = a local Hugging Face model on this GPU (4-bit, greedy; --model is "
                        "the HF id). anthropic is deliberately not offered")
    p.add_argument("--model", required=required)
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--num-ctx", type=int, default=16384, help="ollama only")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=5.0)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None, help="max API calls this invocation (smoke test)")


def _hf_local(args):
    """A local model as the Q&A backend — same 4-bit loader and chat template as
    ft_generate.py. Loaded once per process."""
    import torch
    from ft_common import build_messages, fast_attention, render_chat
    from ft_generate import load_model
    loaded = {}

    def ask(system, user):
        if not loaded:                       # load on first use: a no-op step costs nothing
            loaded["m"] = load_model(args.model, None, False)
        model, tok = loaded["m"]
        prompt = render_chat(tok, build_messages(system, user), True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad(), fast_attention():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return ask


def api(args):
    refuse_excluded(args.model, "model")
    if args.provider == "hf":
        return _hf_local(args)
    from run_api_extract import _load_dotenv, call
    _load_dotenv()
    ns = SimpleNamespace(provider=args.provider, model=args.model, base_url=args.base_url,
                         max_tokens=args.max_tokens, num_ctx=args.num_ctx, timeout=args.timeout,
                         retries=args.retries, backoff=args.backoff)

    def ask(system, user):
        import time
        raw = call(ns, system, user)
        if args.sleep:
            time.sleep(args.sleep)
        return raw
    return ask


def parse_json(raw):
    from kg_extraction.fixed_extractor import _extract_json_object, _strip_think_and_fences
    body = _strip_think_and_fences(raw or "")
    return _extract_json_object(body) if body else None


def run_dir(name):
    if ":" in name or "/" in name or "\\" in name:
        sys.exit("--run must be a plain name")
    return QA_DIR / name


def load_run(name):
    f = run_dir(name) / "run.json"
    if not f.exists():
        sys.exit(f"no run at {f} — start with `qa_eval.py generate --run {name}`")
    return json.loads(f.read_text(encoding="utf-8"))


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl_if(path):
    return read_jsonl(path) if Path(path).exists() else []


_STOP = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was",
         "were", "with", "by", "as", "at", "from", "its", "their", "this", "that"}


def content_token_list(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", norm(s)) if t not in _STOP]


def content_tokens(s):
    return set(content_token_list(s))


def overlap(pred, ref):
    """(precision, recall, f1) — SQuAD token overlap (multiset) over content words."""
    p, r = Counter(content_token_list(pred)), Counter(content_token_list(ref))
    common = sum((p & r).values())
    if not common:
        return 0.0, 0.0, 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return prec, rec, 2 * prec * rec / (prec + rec)


def overlap_verdict(prec, rec, f1, f1_correct=0.8, f1_partial=0.4):
    if f1 >= f1_correct or (rec == 1.0 and prec >= 0.5):
        return "CORRECT"
    if rec == 1.0 or f1 >= f1_partial:
        return "PARTIAL"
    return "INCORRECT"


def _sha(*parts, n=10):
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:n]


# ── generate ─────────────────────────────────────────────────────────────────

def validate(q, para_text, relations):
    """(clean_question, None) or (None, reason). Deterministic — rerunnable from raw."""
    if not isinstance(q, dict):
        return None, "not_an_object"
    question = " ".join(str(q.get("question", "")).split())
    answer = " ".join(str(q.get("answer", "")).split()).strip(" .")
    evidence = " ".join(str(q.get("evidence", "")).split())
    rel = str(q.get("relation", "other")).strip()
    if not question or not answer or not evidence:
        return None, "missing_field"
    npara = norm(para_text)
    if norm(evidence)[:60] not in npara:
        return None, "evidence_not_in_paragraph"
    if norm(answer) not in npara:
        return None, "answer_not_verbatim"
    if len(answer.split()) > 12:
        return None, "answer_too_long"
    if norm(answer) in norm(question):
        return None, "answer_in_question"
    if any(g in norm(question) for g in GENERIC_REFS):
        return None, "generic_reference"
    from kg_extraction.fixed_extractor import _is_provenance_sentence
    if _is_provenance_sentence(evidence):
        return None, "provenance"
    return {"question": question, "answer": answer, "evidence": evidence,
            "relation": rel if rel in relations else "other"}, None


def cmd_generate(args):
    rd = run_dir(args.run)
    rd.mkdir(parents=True, exist_ok=True)
    from kg_extraction.fixed_extractor import CEO_RELATIONS
    relations = list(CEO_RELATIONS)
    if (rd / "run.json").exists():
        meta = json.loads((rd / "run.json").read_text(encoding="utf-8"))
        papers = meta["papers"]
        if meta["generator"]["model"] != args.model:
            sys.exit(f"this run's questions were generated by {meta['generator']['model']}; "
                     f"mixing generators within one question set is not allowed")
        if args.papers or args.papers_from:
            print(f"  note: run {args.run} already exists — using its {len(papers)} papers, "
                  f"ignoring --papers/--papers-from")
    else:
        if args.papers:
            papers = args.papers
        elif args.papers_from:
            papers = Path(args.papers_from).read_text(encoding="utf-8").split()
        else:
            papers = read_corpus(args.papers_file)
        meta = {"run": args.run, "created": date.today().isoformat(), "papers": papers,
                "prompt_version": PROMPT_VERSION, "relations": relations,
                "generator": {"provider": args.provider, "model": args.model},
                "per_paragraph": args.per_paragraph, "max_per_paper": args.max_per_paper,
                "min_words": args.min_words, "seed": args.seed}
        (rd / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # paragraphs, in the same form the extractors saw them
    units = []
    for p in papers:
        rows, source = load_paragraphs(p, "relation", prefer_slugs=(TEACHER_DEFAULT,))
        if rows is None:
            sys.exit(f"{p}: no paragraphs (no prompts.jsonl, no parsed HTML) on this machine")
        for r in rows:
            if len(r["paragraph"].split()) >= meta["min_words"]:
                units.append((p, r))
    raw_f = rd / "qgen_raw.jsonl"
    done = {(r["paper"], r["para_id"]) for r in read_jsonl_if(raw_f)}
    todo = [(p, r) for p, r in units if (p, r["para_id"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(units)} paragraphs over {len(papers)} papers; {len(done)} already asked; "
          f"{len(todo)} calls now")

    if todo and not args.rebuild_only:
        ask = api(args)
        system = (QGEN_SYSTEM.replace("<<K>>", str(meta["per_paragraph"]))
                  .replace("<<RELATIONS>>", ", ".join(relations)))
        for i, (p, r) in enumerate(todo, 1):
            try:
                raw = ask(system, f"Section: {r['section']}\n\nParagraph:\n{r['paragraph']}")
            except Exception as e:
                print(f"  {p}#{r['para_id']}: FAILED — {e} (will retry on the next run)")
                continue
            append_jsonl(raw_f, [{"paper": p, "para_id": r["para_id"], "raw": raw,
                                  "model": args.model}])
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)

    # validate everything from raw (cheap, deterministic), then cap per paper
    by_key = {(p, r["para_id"]): r for p, r in units}
    kept, rejected, seen = defaultdict(list), [], set()
    for rec in read_jsonl_if(raw_f):
        unit = by_key.get((rec["paper"], rec["para_id"]))
        if unit is None:
            continue
        data = parse_json(rec["raw"])
        if data is None:
            rejected.append({"paper": rec["paper"], "para_id": rec["para_id"], "reason": "unparseable_reply"})
            continue
        for q in data.get("questions", [])[:meta["per_paragraph"]]:
            clean, why = validate(q, unit["paragraph"], relations)
            if clean is not None:
                k = (rec["paper"], norm(clean["question"]))
                if k in seen:
                    clean, why = None, "duplicate"
                else:
                    seen.add(k)
            if clean is None:
                rejected.append({"paper": rec["paper"], "para_id": rec["para_id"],
                                 "reason": why, "item": q})
                continue
            clean.update({"qid": _sha(rec["paper"], norm(clean["question"])),
                          "paper": rec["paper"], "para_id": rec["para_id"],
                          "section": unit["section"], "context": unit["paragraph"]})
            kept[rec["paper"]].append(clean)

    rng = random.Random(meta["seed"])
    final = []
    for p in papers:
        by_para = defaultdict(list)
        for q in kept.get(p, []):
            by_para[q["para_id"]].append(q)
        for qs in by_para.values():
            rng.shuffle(qs)
        chosen, order = [], sorted(by_para)
        while len(chosen) < meta["max_per_paper"] and any(by_para[k] for k in order):
            for k in order:                                  # round-robin over paragraphs
                if by_para[k] and len(chosen) < meta["max_per_paper"]:
                    chosen.append(by_para[k].pop())
        final += chosen
    write_jsonl(rd / "questions.jsonl", final)
    write_jsonl(rd / "rejected.jsonl", rejected)
    per = Counter(q["paper"] for q in final)
    print(f"{len(final)} questions kept ({dict(Counter(q['relation'] for q in final))})")
    print(f"per paper: {dict(per)}")
    print(f"rejected {len(rejected)}: {dict(Counter(r['reason'] for r in rejected))}")
    thin = [p for p in papers if per.get(p, 0) < 5]
    if thin:
        print(f"  ! papers with <5 questions: {thin}")


# ── answer ───────────────────────────────────────────────────────────────────

def kg_triples(paper, spec):
    """spec 'relation/<slug>' -> deduplicated [(s, p, o)], or None if that run doesn't exist."""
    f = OUTPUT_DIR / paper / "kg" / spec / "triples.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    rows = d if isinstance(d, list) else d.get("triples", [])
    seen, out = set(), []
    for t in rows:
        key = (norm(t.get("subject", "")), norm(t.get("predicate", "")), norm(t.get("object", "")))
        if all(key) and key not in seen:
            seen.add(key)
            out.append((t["subject"], t["predicate"], t["object"]))
    return out


def select_triples(triples, questions, cap):
    """All triples when they fit; else the `cap` with the most lexical overlap with the batch."""
    if len(triples) <= cap:
        return triples, False
    qtok = set().union(*(content_tokens(q["question"]) for q in questions))
    scored = sorted(range(len(triples)),
                    key=lambda i: -len(qtok & content_tokens(" ".join(triples[i]))))
    return [triples[i] for i in sorted(scored[:cap])], True


def cmd_answer(args):
    meta = load_run(args.run)
    rd = run_dir(args.run)
    questions = read_jsonl(rd / "questions.jsonl")
    (rd / "answers").mkdir(exist_ok=True)
    if args.context == "kg":
        if not args.kg:
            sys.exit("--context kg needs --kg relation/<slug> [...]")
        conds = []
        for spec in args.kg:
            refuse_excluded(spec, "KG")
            missing = [p for p in meta["papers"] if kg_triples(p, spec) is None]
            if missing:
                sys.exit(f"{spec}: no triples.json for {missing}. A missing run is a pipeline "
                         f"failure, not an empty graph — extract/ingest those papers first.")
            conds.append((f"kg__{spec.replace('/', '__')}", spec))
    else:
        conds = [(args.context, None)]

    ask = api(args)
    calls = 0
    for key, spec in conds:
        out_f = rd / "answers" / f"{key}.jsonl"
        have = {r["qid"] for r in read_jsonl_if(out_f)}
        todo = [q for q in questions if q["qid"] not in have]
        groups = defaultdict(list)
        for q in todo:
            groups[(q["paper"], q["para_id"]) if args.context == "oracle" else (q["paper"],)].append(q)
        print(f"{key}: {len(todo)} questions to answer in {len(groups)} group(s)")
        for gkey, qs in groups.items():
            triples = kg_triples(qs[0]["paper"], spec) if spec else None
            for i in range(0, len(qs), args.batch_size):
                if args.limit and calls >= args.limit:
                    print("  --limit reached")
                    return
                batch = qs[i:i + args.batch_size]
                local = {f"q{j + 1}": q for j, q in enumerate(batch)}
                qlines = "\n".join(f"- {lid}: {q['question']}" for lid, q in local.items())
                shown, trunc = [], False
                if args.context == "kg":
                    shown, trunc = select_triples(triples, batch, args.max_triples)
                    listing = "\n".join(f"[{n}] ({s}) --{p}--> ({o})" for n, (s, p, o) in enumerate(shown, 1))
                    user = f"KNOWLEDGE GRAPH ({len(shown)} triples):\n{listing or '(empty)'}\n\nQUESTIONS:\n{qlines}"
                elif args.context == "oracle":
                    user = f"PARAGRAPH:\n{batch[0]['context']}\n\nQUESTIONS:\n{qlines}"
                else:
                    user = f"QUESTIONS:\n{qlines}"
                try:
                    raw = ask(ANSWER_SYSTEM[args.context], user)
                except Exception as e:
                    print(f"  {gkey}: FAILED — {e} (will retry on the next run)")
                    continue
                calls += 1
                data = parse_json(raw) or {}
                rows = []
                for a in data.get("answers", []) if isinstance(data.get("answers"), list) else []:
                    q = local.get(str(a.get("qid", "")).strip())
                    if q is None:
                        continue
                    ans = " ".join(str(a.get("answer", "")).split())
                    abst = (not a.get("answerable", bool(ans))) or not ans
                    cited = [n for n in a.get("triples", []) if isinstance(n, int) and 1 <= n <= len(shown)]
                    rows.append({"qid": q["qid"], "paper": q["paper"], "answer": "" if abst else ans,
                                 "abstained": abst,
                                 "cited_triples": [list(shown[n - 1]) for n in cited],
                                 "n_kg_triples": len(triples) if triples is not None else None,
                                 "kg_truncated": trunc, "answerer": args.model})
                    local.pop(str(a.get("qid", "")).strip(), None)
                append_jsonl(out_f, rows)
                if local:
                    print(f"  {gkey}: {len(local)} question(s) not answered in the reply — rerun to retry")
    print(f"{calls} calls")


# ── grade ────────────────────────────────────────────────────────────────────

def gkey(qid, pred):
    return _sha(qid, norm(pred), n=12)


def cmd_grade(args):
    rd = run_dir(args.run)
    qs = {q["qid"]: q for q in read_jsonl(rd / "questions.jsonl")}
    grades_f = rd / "grades.jsonl"
    prior = read_jsonl_if(grades_f)
    other = {g.get("method") for g in prior} - {"exact", args.method}
    if other:
        sys.exit(f"{grades_f} already holds '{', '.join(sorted(other))}' grades — one grading method "
                 f"per run (start a new --run, or delete grades.jsonl to regrade)")
    if args.method == "llm" and not (args.provider and args.model):
        sys.exit("--method llm needs --provider and --model")
    graded = {g["gkey"] for g in prior}
    need = {}
    for f in sorted((rd / "answers").glob("*.jsonl")):
        for a in read_jsonl(f):
            if a["abstained"] or a["qid"] not in qs:
                continue
            k = gkey(a["qid"], a["answer"])
            if k not in graded and k not in need:
                need[k] = (a["qid"], a["answer"])
    exact = [{"gkey": k, "qid": qid, "verdict": "CORRECT", "grader": "exact-match", "method": "exact",
              "f1": 1.0} for k, (qid, pred) in need.items() if norm(pred) == norm(qs[qid]["answer"])]
    append_jsonl(grades_f, exact)
    for g in exact:
        need.pop(g["gkey"])
    if args.method == "overlap":
        rows = []
        for k, (qid, pred) in need.items():
            prec, rec, f1 = overlap(pred, qs[qid]["answer"])
            rows.append({"gkey": k, "qid": qid, "grader": "token-overlap", "method": "overlap",
                         "verdict": overlap_verdict(prec, rec, f1, args.f1_correct, args.f1_partial),
                         "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)})
        append_jsonl(grades_f, rows)
        print(f"{len(exact)} exact matches + {len(rows)} overlap-graded "
              f"({dict(Counter(r['verdict'] for r in rows))}); no LLM -> {grades_f}")
        return
    items = list(need.items())
    random.Random(args.seed).shuffle(items)       # conditions interleaved -> grader is blind
    print(f"{len(exact)} exact matches graded; {len(items)} answers for the LLM grader")
    if not items:
        return
    ask = api(args)
    calls = 0
    for i in range(0, len(items), args.batch_size):
        if args.limit and calls >= args.limit:
            print("  --limit reached")
            break
        batch = items[i:i + args.batch_size]
        local = {f"g{j + 1}": kv for j, kv in enumerate(batch)}
        user = "\n\n".join(f"item: {lid}\nQuestion: {qs[qid]['question']}\n"
                           f"Reference: {qs[qid]['answer']}\nPredicted: {pred}"
                           for lid, (_k, (qid, pred)) in local.items())
        try:
            raw = ask(GRADE_SYSTEM, user)
        except Exception as e:
            print(f"  batch {i // args.batch_size + 1}: FAILED — {e}")
            continue
        calls += 1
        data = parse_json(raw) or {}
        rows = []
        for g in data.get("grades", []) if isinstance(data.get("grades"), list) else []:
            kv = local.get(str(g.get("item", "")).strip())
            v = str(g.get("verdict", "")).upper().strip()
            if kv and v in POINTS:
                rows.append({"gkey": kv[0], "qid": kv[1][0], "verdict": v,
                             "grader": args.model, "method": "llm"})
        append_jsonl(grades_f, rows)
    print(f"{calls} grading calls -> {grades_f}")


# ── report ───────────────────────────────────────────────────────────────────

def cluster_bootstrap(points_by_paper, B, seed):
    """Mean over questions, resampling PAPERS (the unit the split and the extraction vary by)."""
    papers = [p for p, v in points_by_paper.items() if v]
    if not papers:
        return None, None, None
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        xs = [x for p in (rng.choice(papers) for _ in papers) for x in points_by_paper[p]]
        stats.append(sum(xs) / len(xs))
    stats.sort()
    allx = [x for p in papers for x in points_by_paper[p]]
    return sum(allx) / len(allx), stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def answer_in_kg(gold, triples):
    g = content_tokens(gold)
    return bool(g) and any(g <= content_tokens(el) for t in triples for el in (t[0], t[2]))


def grounded(pred, cited):
    p = content_tokens(pred)
    need = max(1, math.ceil(len(p) / 2))
    return bool(p) and any(len(p & content_tokens(el)) >= need for t in cited for el in (t[0], t[2]))


def cmd_report(args):
    meta = load_run(args.run)
    rd = run_dir(args.run)
    qs = {q["qid"]: q for q in read_jsonl(rd / "questions.jsonl")}
    glines = read_jsonl_if(rd / "grades.jsonl")
    grades = {g["gkey"]: g["verdict"] for g in glines}
    f1s = {g["gkey"]: g.get("f1") for g in glines}      # None for LLM grades
    graders = Counter(g["grader"] for g in glines)
    conds, answerers = {}, {}
    for f in sorted((rd / "answers").glob("*.jsonl")):
        rows = {a["qid"]: a for a in read_jsonl(f)}          # last write wins on a retry
        conds[f.stem] = rows
        answerers[f.stem] = sorted({a["answerer"] for a in rows.values()})
    if len({tuple(v) for v in answerers.values()}) > 1:
        print(f"  ! conditions were answered by DIFFERENT models — not comparable: {answerers}")

    report = {"run": args.run, "generator": meta["generator"], "graders": dict(graders),
              "answerers": answerers, "n_questions": len(qs), "conditions": {}}
    per_q = {}
    for key, rows in conds.items():
        pts, ungraded, f1v = {}, 0, []
        for qid, a in rows.items():
            if qid not in qs:
                continue
            if a["abstained"]:
                pts[qid] = 0.0
                f1v.append(0.0)
                continue
            v = grades.get(gkey(qid, a["answer"]))
            if v is None:
                ungraded += 1
                continue
            pts[qid] = POINTS[v]
            f1v.append(f1s.get(gkey(qid, a["answer"])))
        per_q[key] = pts
        by_paper = defaultdict(list)
        for qid, x in pts.items():
            by_paper[qs[qid]["paper"]].append(x)
        mean, lo, hi = cluster_bootstrap(by_paper, args.bootstrap, args.seed)
        answered = [q for q in pts if not rows[q]["abstained"]]
        c = {"n_scored": len(pts), "n_unanswered": sum(1 for q in qs if q not in rows),
             "n_ungraded": ungraded,
             "answered_rate": len(answered) / len(pts) if pts else None,
             "accuracy": mean, "ci95": [lo, hi],
             "strict": (sum(1 for q in pts if pts[q] == 1.0) / len(pts)) if pts else None,
             "precision_answered": (sum(pts[q] for q in answered) / len(answered)) if answered else None,
             "mean_f1": (sum(f1v) / len(f1v)) if f1v and None not in f1v else None,
             "by_relation": {}, "by_paper": {p: sum(v) / len(v) for p, v in by_paper.items()}}
        rel = defaultdict(list)
        for qid, x in pts.items():
            rel[qs[qid]["relation"]].append(x)
        c["by_relation"] = {r: {"n": len(v), "accuracy": sum(v) / len(v)} for r, v in sorted(rel.items())}
        if key.startswith("kg__"):
            spec = key[4:].replace("__", "/", 1)
            kgs = {p: kg_triples(p, spec) or [] for p in meta["papers"]}
            c["answer_in_kg"] = sum(answer_in_kg(q["answer"], kgs[q["paper"]]) for q in qs.values()) / len(qs)
            correct = [q for q in pts if pts[q] == 1.0]
            c["grounded_share_of_correct"] = (sum(grounded(rows[q]["answer"], rows[q]["cited_triples"])
                                                  for q in correct) / len(correct)) if correct else None
            c["kg_truncated_batches"] = sum(1 for a in rows.values() if a.get("kg_truncated"))
        report["conditions"][key] = c

    comparisons = []
    for a_key, b_key in (args.compare or []):
        if a_key not in per_q or b_key not in per_q:
            sys.exit(f"unknown condition in --compare; have: {sorted(per_q)}")
        both = [q for q in per_q[a_key] if q in per_q[b_key]]
        diffs = defaultdict(list)
        for q in both:
            diffs[qs[q]["paper"]].append(per_q[a_key][q] - per_q[b_key][q])
        mean, lo, hi = cluster_bootstrap(diffs, args.bootstrap, args.seed)
        wins = sum(1 for q in both if per_q[a_key][q] > per_q[b_key][q])
        losses = sum(1 for q in both if per_q[a_key][q] < per_q[b_key][q])
        comparisons.append({"a": a_key, "b": b_key, "n_paired": len(both), "diff": mean,
                            "ci95": [lo, hi], "wins": wins, "losses": losses,
                            "ties": len(both) - wins - losses,
                            "by_paper": {p: sum(v) / len(v) for p, v in diffs.items()}})
    report["comparisons"] = comparisons
    (rd / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def f(x):
        return "—" if x is None else f"{100 * x:.1f}"
    print(f"run {args.run}: {len(qs)} questions over {len(meta['papers'])} papers; "
          f"generator {meta['generator']['model']}; graders {dict(graders)}")
    print("\n| condition | scored | answered | accuracy [95% CI] | strict | token F1 | precision (answered) "
          "| answer-in-KG | grounded |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, c in report["conditions"].items():
        print(f"| {key} | {c['n_scored']} | {f(c['answered_rate'])} | {f(c['accuracy'])} "
              f"[{f(c['ci95'][0])}, {f(c['ci95'][1])}] | {f(c['strict'])} | {f(c['mean_f1'])} "
              f"| {f(c['precision_answered'])} "
              f"| {f(c.get('answer_in_kg'))} | {f(c.get('grounded_share_of_correct'))} |")
        if c["n_unanswered"] or c["n_ungraded"]:
            print(f"|   ! {c['n_unanswered']} unanswered, {c['n_ungraded']} ungraded — rerun answer/grade | | | | | | | | |")
    for cmp_ in comparisons:
        print(f"\n{cmp_['a']} minus {cmp_['b']}: {f(cmp_['diff'])} pts "
              f"[{f(cmp_['ci95'][0])}, {f(cmp_['ci95'][1])}] over {cmp_['n_paired']} paired questions "
              f"(W/L/T {cmp_['wins']}/{cmp_['losses']}/{cmp_['ties']}); CI resamples papers")
    print(f"\n-> {rd / 'report.json'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--papers", nargs="+")
    g.add_argument("--papers-from", default=None, help="e.g. ft_data/fold0/eval_papers.txt")
    g.add_argument("--papers-file", default=CORPUS_FILE_DEFAULT)
    g.add_argument("--per-paragraph", type=int, default=3)
    g.add_argument("--max-per-paper", type=int, default=30)
    g.add_argument("--min-words", type=int, default=40)
    g.add_argument("--seed", type=int, default=13)
    g.add_argument("--rebuild-only", action="store_true",
                   help="re-validate stored raw replies without any API call")
    add_api_args(g)

    a = sub.add_parser("answer")
    a.add_argument("--run", required=True)
    a.add_argument("--context", choices=["kg", "oracle", "closedbook"], required=True)
    a.add_argument("--kg", nargs="+", help="relation/<slug> or relation_scinex/<slug>")
    a.add_argument("--batch-size", type=int, default=10)
    a.add_argument("--max-triples", type=int, default=300,
                   help="whole paper graph if it fits; else lexical top-N for the batch")
    add_api_args(a)

    r = sub.add_parser("grade")
    r.add_argument("--run", required=True)
    r.add_argument("--method", choices=["overlap", "llm"], default="overlap",
                   help="overlap = deterministic token overlap, no model (default); llm = blind LLM grader")
    r.add_argument("--f1-correct", type=float, default=0.8)
    r.add_argument("--f1-partial", type=float, default=0.4)
    r.add_argument("--batch-size", type=int, default=20)
    r.add_argument("--seed", type=int, default=13)
    add_api_args(r, required=False)          # only --method llm uses them

    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)

    args = ap.parse_args()
    {"generate": cmd_generate, "answer": cmd_answer, "grade": cmd_grade,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
