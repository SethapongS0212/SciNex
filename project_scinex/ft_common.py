"""
ft_common.py
------------
Shared plumbing for the finetuning track (build_ft_data.py, ft_train_qlora.py,
ft_generate.py) and the Q&A evaluation (qa_eval.py).

  * the paper-level train/eval SPLIT — fold-based and STICKY, so it works at any
    corpus size and adding papers never moves an existing paper between train and eval
  * loading a paper's paragraphs exactly as the extractors saw them
  * reading a model's stored replies (responses.jsonl) defensively
  * running a reply through the CURRENT post-parse guards + graph-builder filter
  * ONE chat-template helper used by both training and inference, so the prompt a
    finetuned model is trained on is byte-identical to the one it is evaluated with

Nothing in this file loads a model or calls an API.
"""

import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

OUTPUT_DIR = Path("output")
FT_DIR = Path("ft_data")
SPLIT_FILE = FT_DIR / "split.json"
CORPUS_FILE_DEFAULT = "papers_20.txt"
TEACHER_DEFAULT = "gptoss-120b"
SCINEX_OWL_DEFAULT = "scinex_refined_14.owl"

# run_api_extract.py writes this exact string when a call FAILS, so it is not an
# abstention and must never become a "{"triples": []}" training target. A genuine
# empty reply from a model is formatted by that model (gpt-oss: '{\n  "triples": []\n}').
FAILED_CALL_SENTINEL = '{"triples": []}'

# Claude is out of the pipeline — not extractor, judge, teacher, or data source.
EXCLUDED_MODEL_MARKERS = ("claude", "anthropic")


def refuse_excluded(name, what="model"):
    """Hard stop for any Claude model id / slug anywhere in the finetuning or QA path."""
    if any(m in str(name).lower() for m in EXCLUDED_MODEL_MARKERS):
        sys.exit(f"{what} '{name}' is a Claude model. Claude is never an extractor, teacher, "
                 f"training-data source or API-called model here; its only role is the in-session "
                 f"blind judge (blind_judge.py, user decision 2026-10-01). Refusing.")


def quiet_guard_logs():
    """_parse_fixed_output logs every rejected triple at WARNING — fine for one
    paper, unreadable over a corpus. The rejection tallies are counted here instead."""
    for name in ("kg_extraction", "kg_extraction.fixed_extractor",
                 "kg_extraction.relation_extractor", "kg_extraction.kg_builder"):
        logging.getLogger(name).setLevel(logging.ERROR)


def exdir_for(extractor, ontology):
    return extractor if ontology == "ceo" else f"{extractor}_scinex"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s):
    """Whitespace-collapsed, lowercased. Same normalisation ontology_eval uses to
    match a model's source_sentence back to its paragraph."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def triple_id(t, slug):
    """Identical to ontology_eval._tid, so verdicts join on the same key."""
    raw = f"{slug}|{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def read_corpus(papers_file):
    """The corpus allowlist (default papers_20.txt), same convention as ontology_eval."""
    if not papers_file:
        return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name != "acl")
    f = Path(papers_file)
    if not f.exists():
        sys.exit(f"corpus file {papers_file} not found")
    return f.read_text(encoding="utf-8").split()


# ── paper-level split ────────────────────────────────────────────────────────
# Every paper gets a FOLD in [0, K). "Evaluate on fold k" = train on every other
# fold. One mechanism covers both a single hold-out (use fold 0) and full
# cross-validation (run folds 0..K-1, so every paper is held out exactly once).
#
# The split is STICKY: once a paper has a fold it keeps it, and a new paper is
# placed in whichever fold is currently smallest (within its stratum, if strata
# are given), ties broken by a seeded hash of the id. So growing the corpus never
# moves a paper that a model already trained on into an eval fold, or vice versa.
# A rank- or modulo-based split would silently reshuffle on every corpus change.

def _hash_key(seed, paper):
    return hashlib.sha1(f"{seed}:{paper}".encode("utf-8")).hexdigest()


def load_split(path=SPLIT_FILE):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def assign_folds(papers, folds, seed, strata=None, existing=None):
    """Return a split dict. Existing assignments are kept; new papers are balanced in."""
    strata = strata or {}
    split = existing or {"version": 1, "folds": folds, "seed": seed,
                         "created": date.today().isoformat(), "papers": {}}
    if split["folds"] != folds or split["seed"] != seed:
        sys.exit(f"existing split has folds={split['folds']} seed={split['seed']}; you asked "
                 f"for folds={folds} seed={seed}. Changing either would reassign papers that "
                 f"may already have been trained on. Keep the existing values, or move the "
                 f"split file aside deliberately to start a new one.")
    assigned = split["papers"]
    new = sorted((p for p in papers if p not in assigned), key=lambda p: _hash_key(seed, p))
    for p in new:
        stratum = strata.get(p, "all")

        def load(f):
            same = sum(1 for v in assigned.values() if v["fold"] == f and v["stratum"] == stratum)
            total = sum(1 for v in assigned.values() if v["fold"] == f)
            return (same, total, f)

        fold = min(range(folds), key=load)
        assigned[p] = {"fold": fold, "stratum": stratum, "assigned": date.today().isoformat()}
    return split, new


def split_sets(split, corpus, eval_fold, dev_fold=None):
    """(train, dev, eval) paper lists, restricted to the current corpus."""
    if not 0 <= eval_fold < split["folds"]:
        sys.exit(f"--eval-fold must be in [0, {split['folds']})")
    if dev_fold is not None and (dev_fold == eval_fold or not 0 <= dev_fold < split["folds"]):
        sys.exit("--dev-fold must be a different, valid fold")
    fold = {p: v["fold"] for p, v in split["papers"].items()}
    missing = [p for p in corpus if p not in fold]
    if missing:
        sys.exit(f"papers not in the split yet: {missing} — run build_ft_data.py to assign them")
    ev = sorted(p for p in corpus if fold[p] == eval_fold)
    dev = sorted(p for p in corpus if dev_fold is not None and fold[p] == dev_fold)
    tr = sorted(p for p in corpus if p not in ev and p not in dev)
    if not ev:
        sys.exit(f"eval fold {eval_fold} is empty — the corpus is smaller than the fold count")
    assert not set(tr) & set(ev), "train/eval overlap — split logic is broken"
    return tr, dev, ev


# ── ontology / prompts ───────────────────────────────────────────────────────

def current_system_prompt(ontology, ontology_file=SCINEX_OWL_DEFAULT):
    """(system_prompt, relations) exactly as claude_extract.py / kg_main would build them
    for relation-only extraction today. scinex needs rdflib (the OWL loader)."""
    from kg_extraction.relation_extractor import RelationOnlyExtractor
    relations = schema = None
    if ontology == "scinex":
        from kg_extraction.ontology_loader import load_ontology
        relations, schema = load_ontology(ontology_file)
    ex = RelationOnlyExtractor(relations=relations, schema=schema, model_name="n/a")
    return ex._system_prompt, list(ex.relations)


_REL_LINE = re.compile(r"^  - (\w+)  \((.*)\)\s*$")
_REL_BARE = re.compile(r"^  - (\w+)\s*$")


def schema_from_system_prompt(text):
    """relation -> 'Domain → Range  |  usage', read back from a rendered system prompt.

    _make_fixed_system_prompt writes each relation as `  - name  (hint)`, so this
    recovers the exact definitions a model was shown, for either ontology, without
    needing rdflib. Only the RELATIONS block is read.
    """
    block = text.split("RELATIONS:", 1)[-1].split("OBJECTS:", 1)[0]
    out = {}
    for line in block.splitlines():
        m = _REL_LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
            continue
        m = _REL_BARE.match(line)
        if m:
            out[m.group(1)] = ""
    return out


def parse_definition(hint):
    """'Model/Experiment → Dataset  |  usage...' -> ('Model/Experiment', 'Dataset', 'usage...')."""
    head, _, usage = hint.partition("|")
    dom, arrow, rng = head.partition("→")
    if not arrow:
        return "?", "?", hint.strip()
    return dom.strip() or "?", rng.strip() or "?", usage.strip()


# ── paragraphs ───────────────────────────────────────────────────────────────

def render_paragraphs(paper):
    """Paragraph prompts for relation-only extraction, rendered from the parsed HTML with
    the same code path as `claude_extract.py prompts` (which is what every API run used).
    Returns None if the paper has no parsed HTML."""
    from kg_extraction.fixed_extractor import _is_garbled_section
    from kg_extraction.html_parser import parse_html, split_into_sentences
    from kg_extraction.relation_extractor import RelationOnlyExtractor, _make_relation_user_prompt

    html = OUTPUT_DIR / paper / "no-llm" / "output.html"
    if not html.exists():
        return None
    _title, sections = parse_html(str(html))
    splitter = RelationOnlyExtractor(model_name="n/a")   # no weights are loaded
    rows = []
    for sec in sections:
        section = sec["section"]
        if _is_garbled_section(section):
            continue
        sentences = split_into_sentences(sec["text"])
        if not sentences:
            continue
        for para in splitter._sentences_to_paragraphs(sentences):
            rows.append({"para_id": len(rows), "section": section, "paragraph": para,
                         "user_prompt": _make_relation_user_prompt(para, section)})
    return rows


def load_paragraphs(paper, exdir, prefer_slugs=()):
    """(rows, source) for a paper. Prefers a stored prompts.jsonl — the exact prompts a
    run sent — then renders from the parsed HTML. rows=None when neither exists.

    A stored prompts.jsonl from a DIFFERENT run, or a fresh render, may not share the
    para_ids a given responses.jsonl was written against (parser changes, spaCy vs
    regex sentence splitting). Callers pairing replies with paragraphs must verify
    alignment — see `alignment_rate`.
    """
    kg = OUTPUT_DIR / paper / "kg" / exdir
    cands = [kg / s / "prompts.jsonl" for s in prefer_slugs]
    if kg.exists():
        cands += sorted(kg.glob("*/prompts.jsonl"))
    for f in cands:
        if f.exists():
            rows = read_jsonl(f)
            if rows and all("user_prompt" in r and "paragraph" in r for r in rows):
                return rows, str(f)
    rows = render_paragraphs(paper)
    if rows is not None:
        return rows, f"rendered:{OUTPUT_DIR / paper / 'no-llm' / 'output.html'}"
    return None, "missing"


def alignment_rate(pairs):
    """pairs: [(paragraph_text, [source_sentence, ...])]. Share of source sentences found
    in the paragraph they are attributed to (first 60 normalised chars, as ontology_eval
    does). A low rate means para_ids do not line up and every target would be attached
    to the wrong input."""
    hit = tot = 0
    for text, sents in pairs:
        nt = norm(text)
        for s in sents:
            key = norm(s)[:60]
            if not key:
                continue
            tot += 1
            hit += key in nt
    return (hit / tot if tot else None), tot


# ── replies ──────────────────────────────────────────────────────────────────

def read_replies(path):
    """{para_id: raw} plus a Counter of problems.

    Defensive against the §30.8 incident (two processes appending to one file):
    unreadable lines are counted and skipped, and a para_id answered twice with
    DIFFERENT text is dropped — there is no way to know which reply is right.
    """
    seen = defaultdict(list)
    issues = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                issues["corrupt_line"] += 1
                continue
            if not isinstance(r, dict) or "para_id" not in r or "raw" not in r:
                issues["corrupt_line"] += 1
                continue
            raw = r["raw"] if isinstance(r["raw"], str) else json.dumps(r["raw"])
            seen[r["para_id"]].append(raw)
    out = {}
    for pid, raws in seen.items():
        if len(raws) > 1:
            if len(set(raws)) > 1:
                issues["duplicate_conflict"] += 1
                continue
            issues["duplicate_identical"] += 1
        out[pid] = raws[0]
    return out, issues


def is_api_run(run_dir):
    """True if a run dir was written by run_api_extract.py (which writes FAILED_CALL_SENTINEL on a
    failed call). ft_generate.py records `base_model` in prompt_meta.json and raises on failure,
    so there the same string is a genuine answer — measured: 21 of the base Qwen3-4B's 148
    replies were exactly '{"triples": []}', and the finetuned model is TRAINED to emit it."""
    meta = Path(run_dir) / "prompt_meta.json"
    if not meta.exists():
        return True
    return "base_model" not in json.loads(meta.read_text(encoding="utf-8"))


def classify_raw(raw, api_run=True):
    """('ok', data) or (reason, None). Only 'ok' replies can become training targets.
    `api_run` — see is_api_run; only there is the compact empty reply a failure marker."""
    from kg_extraction.fixed_extractor import _extract_json_object, _strip_think_and_fences
    if api_run and raw == FAILED_CALL_SENTINEL:
        return "failed_call", None
    if not raw.strip():
        return "empty_content", None
    body = _strip_think_and_fences(raw)
    if not body:
        return "empty_content", None       # reasoning ate the budget (hands_off §26)
    data = _extract_json_object(body)
    if data is None or not isinstance(data.get("triples", []), list):
        return "unparseable", None
    return "ok", data


def guard_pass(raw, relations, api_run=True):
    """(status, raw_triples, survivors) — survivors are what `claude_extract.py ingest`
    would write to triples.json today: the current `_parse_fixed_output` guards, then the
    graph builder's entity filter. Predicates come back LOWERCASED (as in triples.json);
    `canonical_predicate` restores the ontology spelling for training targets."""
    from kg_extraction.fixed_extractor import _parse_fixed_output
    from kg_extraction.kg_builder import _is_valid_entity
    status, data = classify_raw(raw, api_run)
    if status != "ok":
        return status, [], []
    raw_triples = [t for t in data.get("triples", []) if isinstance(t, dict)]
    kept = _parse_fixed_output(raw, None, relations)      # relation mode, as ingest does
    survivors = [t for t in kept
                 if _is_valid_entity(t["subject"], t["predicate"], "subject")
                 and _is_valid_entity(t["object"], t["predicate"], "object")]
    return "ok", raw_triples, survivors


def canonical_predicate(pred, relations):
    return {r.lower(): r for r in relations}.get(str(pred).lower(), pred)


def target_json(survivors, relations):
    """The assistant turn: the extractor's own output contract, canonical predicate names."""
    return json.dumps({"triples": [{
        "subject": t["subject"],
        "predicate": canonical_predicate(t["predicate"], relations),
        "object": t["object"],
        "object_type": t.get("object_type", "Other"),
        "source_sentence": t["source_sentence"],
    } for t in survivors]}, ensure_ascii=False)


# ── attention kernel ─────────────────────────────────────────────────────────

def fast_attention(training=False):
    """Context manager: pick the torch SDPA kernel deliberately instead of by fallback.

    Measured on the RTX 5070 Laptop (sm_120, Windows, torch 2.12 nightly): the FLASH and
    MEM-EFFICIENT kernels are unavailable for grouped-query attention, so Qwen3's attention
    silently fell back to MATH — which materialises the full score matrix and peaked at
    7.27 GB for ONE layer at 3.8k tokens (forward+backward), vs 0.34 GB with cuDNN. On an
    8 GB card that pushed training into Windows' system-RAM fallback (100% util at ~40 W,
    no step in 10 min).

    training=True  flash -> cuDNN -> mem-efficient -> math. Fixed sequence lengths, and the
                   backward pass is where MATH's memory explodes.
    training=False flash -> mem-efficient -> math, NO cuDNN. cuDNN builds an execution plan
                   per shape, and during generation the key length grows every token, so it
                   re-plans at every step: decoding measured ~3x slower (5.7 vs 18 tok/s).
                   Inference has no backward pass, so MATH's prefill peak fits (5.3 GB at
                   batch 4 on 8 GB).
    Flash is listed first everywhere; it is skipped automatically where unavailable (this
    laptop) and used where it exists (the VM).
    """
    import contextlib
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return contextlib.nullcontext()
    order = ([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
              SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH] if training else
             [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    try:
        return sdpa_kernel(order, set_priority=True)
    except TypeError:                        # torch without set_priority
        return sdpa_kernel(order)


# ── chat template (shared by training AND inference) ─────────────────────────

def build_messages(system, user, assistant=None):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def _merge_system(messages):
    """For chat templates that reject a system role: fold it into the first user turn."""
    if messages and messages[0]["role"] == "system":
        sys_msg, rest = messages[0], list(messages[1:])
        if rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": sys_msg["content"] + "\n\n" + rest[0]["content"]}
        return rest
    return messages


def render_chat(tokenizer, messages, add_generation_prompt):
    """Render with thinking OFF (Qwen3: an empty <think></think> block, exactly what
    `enable_thinking=False` produces at inference). Templates that don't know the flag
    ignore it."""
    kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    try:
        return tokenizer.apply_chat_template(messages, **kw)
    except Exception:
        if messages and messages[0]["role"] == "system":
            return tokenizer.apply_chat_template(_merge_system(messages), **kw)
        raise


def split_prompt_completion(tokenizer, messages):
    """(prompt_text, completion_text, mode). prompt_text is exactly what inference feeds
    the model; completion_text is what it should generate, end-of-turn token included."""
    prompt = render_chat(tokenizer, messages[:-1], add_generation_prompt=True)
    full = render_chat(tokenizer, messages, add_generation_prompt=False)
    if full.startswith(prompt) and len(full) > len(prompt):
        return prompt, full[len(prompt):], "template"
    return prompt, messages[-1]["content"] + (tokenizer.eos_token or ""), "fallback"
