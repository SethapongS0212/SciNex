#!/usr/bin/env python3
"""
text_baseline.py — content-similarity baselines for the citation-prediction task.

Instead of KG embeddings, represent each paper by its TEXT (title, or title+abstract),
embed with TF-IDF, and rank candidate papers by cosine similarity. Evaluated with the
EXACT same machinery as the KGE pipeline (same candidate pool, ground-truth citation
edges, val/test split, Hits@k / MRR / temporal-filtered metrics) via
kg_transe_pipeline.evaluate(predict_fn=...), so the numbers are directly comparable to
the RotatE/ComplEx results.

Two baselines (the user request):
  --text-source title            : title  -> TF-IDF -> cosine
  --text-source title_abstract   : title + abstract -> TF-IDF -> cosine
  --text-source both (default)   : run both and print a comparison

CPU-only (no GPU, no new deps beyond scikit-learn) — safe to run alongside a GPU job.

Example:
  python3 text_baseline.py --extractor fixed --model Qwen3-14B --text-source both \
      --eval-split test --split-seed 42 --val-frac 0.5 \
      --save-results output/text_baseline_results.json
"""
import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from kg_transe_pipeline import (
    load_all_data, build_unified_graph, evaluate, PAPER_PREFIX,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("text_baseline")


def build_text_maps(output_dir: str) -> tuple[dict, dict]:
    """Aggregate title + abstract for every paper node, keyed the way the graph keys
    them: corpus papers by their FOLDER id (= the ACL id used as the graph node), and
    external cited papers by their raw S2 hash (from the network's `nodes`).

    NB: the network root's `paper_id` field is the S2/arXiv hash, NOT the folder id —
    so corpus papers MUST be keyed by the folder, else every query gets empty text.
    """
    title_map, abstract_map = {}, {}
    for f in glob.glob(os.path.join(output_dir, "*", "citation_network.json")):
        folder = os.path.basename(os.path.dirname(f))   # = corpus/ACL graph key
        try:
            d = json.load(open(f))
        except Exception:
            continue
        title_map.setdefault(folder, (d.get("title") or "").strip())
        abstract_map.setdefault(folder, (d.get("abstract") or "").strip())
        for s2, nd in d.get("nodes", {}).items():
            title_map.setdefault(s2, (nd.get("title") or "").strip())
            abstract_map.setdefault(s2, (nd.get("abstract") or "").strip())
    # Fallback: fill any empty corpus title from the parsed output.json
    for of in glob.glob(os.path.join(output_dir, "*", "no-llm", "output.json")):
        folder = os.path.basename(os.path.dirname(os.path.dirname(of)))
        if not title_map.get(folder):
            try:
                title_map[folder] = (json.load(open(of)).get("title") or "").strip()
            except Exception:
                pass
    return title_map, abstract_map


def make_text_predict_fn(graph, title_map, abstract_map, use_abstract: bool):
    """Return predict_fn(paper_id) -> [{paper_id, score}, ...] sorted desc, ranking all
    candidate paper nodes by TF-IDF cosine similarity to the query paper's text."""
    # Candidate keys = the same paper nodes the KGE eval ranks over.
    keys = [name.replace(PAPER_PREFIX, "") for name in graph["paper_node_ids"]]

    def text_of(k: str) -> str:
        t = title_map.get(k, "")
        if use_abstract:
            a = abstract_map.get(k, "")
            return (t + " " + a).strip() if a else t
        return t

    texts = [text_of(k) for k in keys]
    n_nonempty = sum(1 for t in texts if t)
    logger.info(f"  paper nodes: {len(keys)}, with text: {n_nonempty} "
                f"({'title+abstract' if use_abstract else 'title'})")

    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                          min_df=1, max_df=0.9)
    M = vec.fit_transform(texts)        # (n_papers, vocab) sparse
    M = normalize(M, norm="l2", axis=1)  # row-normalise → cosine = dot product
    key_to_row = {k: i for i, k in enumerate(keys)}

    def predict_fn(paper_id: str):
        qi = key_to_row.get(paper_id)
        if qi is None:
            return []
        sims = (M @ M[qi].T).toarray().ravel()  # cosine of query vs every paper node
        order = np.argsort(-sims)
        out = []
        for i in order:
            if i == qi:
                continue
            out.append({"paper_id": keys[i], "score": round(float(sims[i]), 4)})
        return out

    return predict_fn


def run_one(data, graph, title_map, abstract_map, use_abstract, args) -> dict:
    predict_fn = make_text_predict_fn(graph, title_map, abstract_map, use_abstract)
    return evaluate(None, graph, data, device="cpu",
                    eval_split=args.eval_split, val_frac=args.val_frac,
                    split_seed=args.split_seed, predict_fn=predict_fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--extractor", default="fixed",
                    help="Which extractor's graph to build the candidate pool from "
                         "(matches the KGE eval pool; default: fixed)")
    ap.add_argument("--model", default="Qwen3-14B")
    ap.add_argument("--text-source", default="both",
                    choices=["title", "title_abstract", "both"])
    ap.add_argument("--eval-split", default="all", choices=["all", "val", "test"])
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--save-results", default=None)
    args = ap.parse_args()

    logger.info("Loading data + building candidate graph ...")
    data = load_all_data(args.output_dir, args.extractor, args.model)
    graph = build_unified_graph(data)
    title_map, abstract_map = build_text_maps(args.output_dir)

    sources = (["title", "title_abstract"] if args.text_source == "both"
               else [args.text_source])
    results = {}
    for src in sources:
        logger.info(f"=== TEXT BASELINE: {src} (eval_split={args.eval_split}) ===")
        results[src] = run_one(data, graph, title_map, abstract_map,
                               src == "title_abstract", args)

    # Compact comparison table
    def g(r, k):
        return r.get(k, r.get("metrics", {}).get(k, float("nan")))
    print(f"\n── TEXT-SIMILARITY BASELINES (TF-IDF cosine, eval_split={args.eval_split}) ──")
    print(f"  {'source':<16}{'Hits@1':>9}{'Hits@5':>9}{'Hits@10':>9}{'MRR':>9}"
          f"{'filtMRR':>9}{'filtH@10':>10}")
    for src in sources:
        r = results[src]
        print(f"  {src:<16}{g(r,'hits@1'):>9.3f}{g(r,'hits@5'):>9.3f}{g(r,'hits@10'):>9.3f}"
              f"{g(r,'mrr'):>9.3f}{g(r,'filt_mrr'):>9.3f}{g(r,'filt_hits@10'):>10.3f}")

    if args.save_results:
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"eval_split": args.eval_split, "val_frac": args.val_frac,
                   "split_seed": args.split_seed, "extractor": args.extractor,
                   "results": results}, open(args.save_results, "w"), indent=2)
        print(f"\n  Saved → {args.save_results}")


if __name__ == "__main__":
    main()
