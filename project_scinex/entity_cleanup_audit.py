#!/usr/bin/env python3
"""
entity_cleanup_audit.py
-----------------------
DRAFT audit of the per-paper fixed-entity lists (output/acl/<id>/Entity_<id>_enriched.csv),
the C3 (specificity) ceiling found by the blind judging (hands_off §37.10): generic CS-NER
entries such as "Models" or "Experiments" become triple subjects. Read-only: it changes no
entity list and no triple, it only says what a cleanup WOULD remove and what that would touch.

Three removal categories:
  CASE_COLLISION  per paper, automatic. The gazetteer is matched case-insensitively, so named
                  artifacts collide with common words ("CompLex" <- "complex", "SETS" <- "sets",
                  "ZeRO" <- "zero"). An entry whose surface form has a capital letter inside a
                  word is kept for a paper only if that exact casing occurs in the paper's text.
  NOT_A_CONCEPT   adjectives / verbs / participles ("semantic", "supervised", "evaluate").
  GENERIC         generic nouns that merge unrelated things ("Models", "Experiments", "Text").
plus REVIEW (task-level words such as "classifier" or "Attention": a human decides) and KEEP.
The word lists cover entries in >= --min-df-pct of papers; the long tail is KEEP by default.

Impact is reported on (a) the entity lists, (b) the teacher's kept triples over all papers,
(c) the blind-judged triples (C1/C3/C4 by subject category, read from the batch and reply
files only, so no model identity is involved).

    python entity_cleanup_audit.py --root /home/dev1/SciNex/project_scinex \
        --pools ontology_eval/_blind/acl-f0/ceo ontology_eval/_blind/acl-f0/scinex
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NOT_A_CONCEPT = {
    "evaluate", "semantic", "supervised", "unsupervised", "semi-supervised", "linguistic",
    "syntactic", "statistical", "generating", "generative", "bidirectional", "probabilistic",
    "morphological", "distributional", "stochastic", "discriminative", "predictive",
    "conversational", "subjective", "redundant", "classified", "covers", "cluster", "parse",
    "adapting", "annotating", "translating", "hybrid", "rule-based", "phrase-based",
}
# NOT listed on purpose: complex, character, mutual, zero, red, flat, bit, etc, map, tag(s). Each
# is a real named artifact in a few papers (CompLex, CharacTer, MuTual, ZeRO, MAP, ...); the
# per-paper exact-case rule removes the collisions and keeps the genuine uses.

GENERIC = {
    # single words
    "training", "example", "examples", "language", "languages", "information", "models", "words",
    "learning", "text", "texts", "experiments", "evaluation", "input", "sentence", "sentences",
    "context", "features", "analysis", "terms", "knowledge", "sets", "datasets", "sequence",
    "probability", "probabilities", "search", "algorithm", "similarity", "relations", "entity",
    "entities", "meaning", "corpora", "understanding", "categories", "phrases", "news", "errors",
    "numbers", "measures", "patterns", "annotations", "rules", "rule", "documents", "domains",
    "speech", "dependency", "dependencies", "topic", "topics", "questions", "web", "description",
    "bias", "biases", "definition", "definitions", "evidence", "nouns", "verbs", "constraints",
    "plan", "tools", "names", "events", "construction", "adaptation", "graphs", "translations",
    "reasoning", "image", "images", "arguments", "concepts", "supervision", "transformation",
    "generalization", "queries", "mining", "reviews", "actions", "attributes", "opinions",
    "guidelines", "objects", "adjectives", "clusters", "ambiguity", "robustness", "engineering",
    "alignments", "punctuation", "polarity", "heuristics", "senses", "frame", "frames", "tweets",
    "keywords", "strings", "coherence", "synonyms", "conversation", "conversations", "lemma",
    "equations", "summaries", "dictionaries", "dictionary", "paragraphs", "expectation",
    "pronouns", "claims", "gpus", "genres", "genre", "video", "negation", "nlp", "matching",
    # multi-word
    "natural language", "language processing", "test sets", "nlp tasks", "evaluation metrics",
    "ground truth", "sentence pairs", "human evaluation", "contextual information",
    "vocabulary size", "computer science", "different languages", "baseline models",
    "upper bound", "model training", "learning models", "two languages", "all languages",
    "high accuracy", "different domains", "source text", "sentence length", "context information",
    "semantic information", "neural models", "classification task", "classification tasks",
    "translation quality", "error analysis", "news articles", "unlabeled data", "word order",
    "parallel data", "automatic evaluation", "precision and recall", "embedding space",
    "vector representations", "word representations", "word vectors", "word vector",
    "vector space", "semantic relations", "noun phrases", "computational linguistics",
    "natural language processing", "machine learning", "deep learning",
}

REVIEW = {
    "classification", "translation", "summarization", "parsing", "tagging", "segmentation",
    "tokenization", "normalization", "coreference", "entailment", "paraphrase", "disambiguation",
    "clustering", "regression", "sentiment", "attention", "alignment", "decoding", "ranking",
    "retrieval", "inference", "generation", "annotation", "embeddings", "smoothing",
    "convolution", "aggregation", "induction", "lexicon", "lexicons", "grammar", "syntax",
    "morphology", "semantics", "discourse", "dialogue", "dialog", "optimization", "filtering",
    "detection", "recognition", "sampling", "visualization", "pre-training", "fine-tuning",
    "preprocessing", "parser", "parsers", "classifier", "classifiers", "tagger", "generator",
    "ensemble", "kernel", "treebank", "n-gram", "n-grams", "bigram", "bigrams", "precision",
    "pos", "social media", "language understanding", "language generation", "word embeddings",
    "word embedding", "neural network", "neural networks", "language model", "language models",
    "attention mechanism", "supervised learning", "entity recognition", "named entity",
}

# well-known artifacts that papers often write in plain casing ("Adam", "glove", "relu"): never
# treated as collisions
CASE_ALLOW = {"adam", "glove", "relu", "ner", "elmo", "fasttext", "word2vec", "roberta", "softmax",
              "lstm", "bilstm", "crf", "svm", "bert", "gpt", "rnn", "gru", "bleu", "rouge"}

INNER_CAP = re.compile(r"[A-Za-z][A-Z]")       # a capital letter after the first of a word


def case_sensitive(surface):
    return any(INNER_CAP.search(w) for w in re.split(r"[\s\-/]+", surface))


def norm(s):
    return " ".join(s.strip().lower().split())


def load_lists(root, papers):
    lists = {}
    for p in papers:
        f = root / "output" / "acl" / p / f"Entity_{p}_enriched.csv"
        if f.exists():
            lists[p] = list(csv.DictReader(open(f, encoding="utf-8")))
    return lists


def paper_text(root, p):
    """Paragraph text the extractor saw (teacher prompts), joined."""
    out = []
    for ont in ("fixed", "fixed_scinex"):
        f = root / "output" / p / "kg" / ont / "gptoss-20b" / "prompts.jsonl"
        if f.exists():
            for line in open(f, encoding="utf-8"):
                out.append(json.loads(line).get("paragraph", ""))
            break
    return "\n".join(out)


def collides(surface, text):
    """True if the text contains the entry ONLY in plain casing (lowercase, or capitalised first
    letters): "CompLex" where the paper only writes "complex". Any occurrence with a capital
    inside a word ("fastText", "bidirectional LSTM") is a genuine use. No occurrence at all:
    undecidable here, kept."""
    hits = [m.group(0) for m in re.finditer(
        r"(?<![A-Za-z0-9])" + re.escape(surface) + r"(?![A-Za-z0-9])", text, re.IGNORECASE)]
    return bool(hits) and not any(case_sensitive(h) for h in hits)


def category(key):
    if key in NOT_A_CONCEPT:
        return "NOT_A_CONCEPT"
    if key in GENERIC:
        return "GENERIC"
    if key in REVIEW:
        return "REVIEW"
    return "KEEP"


def parse_pool(pool):
    """-> {triple_id: (subject, source sentence)} from batch files, {triple_id: verdict} from replies."""
    subj, verd = {}, {}
    rx = re.compile(r"----- triple_id: (\S+) -----\nTRIPLE:\s+\((.*?)\)\s+--(\S+?)-->\s+\((.*)\)\n"
                    r"(?:.*\n)*?SOURCE SENTENCE: (.*)")
    for b in sorted(pool.glob("batch_*.txt")):
        for m in rx.finditer(b.read_text(encoding="utf-8")):
            subj[m.group(1)] = (m.group(2), m.group(5))
    for r in sorted((pool / "replies").glob("reply_*.txt")):
        for v in json.loads(r.read_text(encoding="utf-8"))["verdicts"]:
            verd[v["triple_id"]] = v
    return subj, verd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="project_scinex checkout that holds output/ (data)")
    ap.add_argument("--papers-file", default="papers_acl.txt")
    ap.add_argument("--teacher", default="gptoss-20b")
    ap.add_argument("--pools", nargs="*", default=[], help="judged blind pools (relative to --root)")
    ap.add_argument("--min-df-pct", type=float, default=5.0)
    ap.add_argument("--out", default="entity_cleanup")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    papers = [l.strip() for l in open(root / args.papers_file) if l.strip()]
    lists = load_lists(root, papers)
    n = len(lists)

    # ── (a) the entity lists ──────────────────────────────────────────────────────────────
    df, forms, ner = Counter(), defaultdict(Counter), defaultdict(Counter)
    for p, rows in lists.items():
        seen = set()
        for r in rows:
            k = norm(r["Entity"])
            forms[k][r["Entity"].strip()] += 1
            ner[k][f'{r["NER_Type"]}/{r["CEO_Type"]}'] += 1
            if k not in seen:
                seen.add(k)
                df[k] += 1
    # per-paper removal decision
    removed = {}                       # (paper, key) -> category
    collisions = defaultdict(lambda: [0, 0])   # surface key -> [papers listed, papers collided]
    for p, rows in lists.items():
        text = None
        for r in rows:
            k, surface = norm(r["Entity"]), r["Entity"].strip()
            cat = category(k) if 100 * df[k] / n >= args.min_df_pct else "KEEP"
            if cat == "KEEP" and case_sensitive(surface) and k not in CASE_ALLOW:
                text = paper_text(root, p) if text is None else text
                collisions[k][0] += 1
                if text and collides(surface, text):
                    collisions[k][1] += 1
                    cat = "CASE_COLLISION"
            removed[(p, k)] = cat
    cats = Counter(removed.values())
    total_entries = sum(cats.values())

    with open(out / "stoplist_draft.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity", "surface_form", "papers", "papers_pct", "types", "category",
                    "case_collision_papers"])
        for k, c in df.most_common():
            pct = 100 * c / n
            cat = category(k) if pct >= args.min_df_pct else "KEEP"
            col = collisions.get(k, [0, 0])[1]
            if pct < args.min_df_pct and not col:
                continue
            w.writerow([k, forms[k].most_common(1)[0][0], c, f"{pct:.1f}",
                        "; ".join(t for t, _ in ner[k].most_common(2)), cat, col])

    # ── (b) the teacher's kept triples ────────────────────────────────────────────────────
    teach = {}
    for ont in ("fixed", "fixed_scinex"):
        by = Counter()
        for p in lists:
            f = root / "output" / p / "kg" / ont / args.teacher / "triples.json"
            if f.exists():
                for t in json.load(open(f, encoding="utf-8")):
                    by[removed.get((p, norm(t["subject"])), "NOT_IN_LIST")] += 1
        teach[ont] = dict(by)

    # ── (c) judged triples (blind pools; batch + reply files only) ────────────────────────
    judged = defaultdict(lambda: {"n": 0, "C1": 0.0, "C4": 0.0, "C3_broad": 0, "C3_ok": 0})
    score = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
    for pool in args.pools:
        subj, verd = parse_pool(root / pool)
        for tid, v in verd.items():
            surface, sentence = subj.get(tid, ("", ""))
            k = norm(surface)
            cat = category(k) if 100 * df.get(k, 0) / n >= args.min_df_pct else "KEEP*"
            if (cat in ("KEEP", "KEEP*") and case_sensitive(surface) and k not in CASE_ALLOW
                    and collides(surface, sentence)):
                cat = "CASE_COLLISION"
            j = judged[cat]
            j["n"] += 1
            j["C1"] += score.get(v.get("C1"), 0)
            j["C4"] += score.get(v.get("C4"), 0)
            j["C3_broad"] += v.get("C3") == "TOO_BROAD"
            j["C3_ok"] += v.get("C3") == "APPROPRIATE"
    judged_out = {c: {"n": j["n"], "C1": round(100 * j["C1"] / j["n"], 1),
                      "C4": round(100 * j["C4"] / j["n"], 1),
                      "C3_appropriate_pct": round(100 * j["C3_ok"] / j["n"], 1),
                      "C3_too_broad_pct": round(100 * j["C3_broad"] / j["n"], 1)}
                  for c, j in judged.items() if j["n"]}

    summary = {"papers_with_list": n, "min_df_pct": args.min_df_pct,
               "entity_list_entries": total_entries, "entries_by_category": dict(cats),
               "case_sensitive_entries_checked": sum(v[0] for v in collisions.values()),
               "case_collisions": sum(v[1] for v in collisions.values()),
               "top_case_collisions": sorted(((k, v[1], v[0]) for k, v in collisions.items() if v[1]),
                                             key=lambda x: -x[1])[:25],
               "teacher_triples_by_subject_category": teach,
               "judged_by_subject_category": judged_out,
               "note": "KEEP* in judged = below the df threshold (long tail), not reviewed"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    short = dict(summary, top_case_collisions=[f"{k} {c}/{m}" for k, c, m in summary["top_case_collisions"]])
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
