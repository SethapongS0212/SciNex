#!/usr/bin/env python3
"""
kg_masked_node_pipeline.py
--------------------------
Masked-BIG-NODE link prediction over the persistent global KG
(output/global_kg/<extractor>/<model>/graph.graphml) — the successor eval to
kg_transe_pipeline.py's hold-out-edges scheme (which is left untouched).

The task
    Papers are BIG NODES connected to each other by real `cites` edges.
    A test paper is MASKED: every paper↔paper citation edge touching it is
    removed from training. The KGE model must then rank the masked paper's
    true citation neighbours above all other papers — i.e. re-find the missing
    big-node links.

Two graph versions (--graph), same task, same split, same KGE machinery:
    citation   Big nodes + `cites` edges ONLY. A masked paper keeps no edges
               at all, so nothing about it is learnable — the structural
               floor. (How well can citation topology alone recover a
               new paper's links? It can't — that's the point.)
    augmented  Big nodes + `cites` edges + each paper's extracted entities as
               SUPPORT nodes (`mentions` edges + entity–entity triples). The
               masked paper keeps its mentions/entity edges, so its embedding
               is learned from CONTENT alone. Augmented nodes are never
               masked, never predicted, never counted in metrics — they only
               exist to position the big nodes.

The augmented-vs-citation delta on identical splits isolates exactly how much
the extracted triples help big-node link prediction.

Reuses kg_transe_pipeline.py's KGE models + training loop (TransE/ComplEx/
RotatE, margin or self-adversarial loss) — only graph construction, masking,
and evaluation differ.

Usage:
    python3 kg_masked_node_pipeline.py --extractor fixed --model gemma-3-12b-it \
        --graph augmented --kge rotate --seeds 1 2 3 --epochs 1000
    # citation-only ablation on the SAME split:
    python3 kg_masked_node_pipeline.py --extractor fixed --model gemma-3-12b-it \
        --graph citation --kge rotate --seeds 1 2 3 --epochs 1000
"""

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import torch
import torch.nn.functional as F

from kg_extraction.global_graph import (
    load_global_graph, PAPER_PREFIX, CITES_REL, MENTIONS_REL,
)
from kg_transe_pipeline import train_kge, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("masked_node")


# ── Graph → triples ───────────────────────────────────────────────────────────

def build_triples(graph: nx.MultiDiGraph, mode: str) -> dict:
    """
    Flatten the global graph into KGE triples for one of the two versions.

    citation : (paper, cites, paper) only
    augmented: + (paper, mentions, entity) + (entity, <rel>, entity)

    Paper stubs (cited papers outside the corpus) count as big nodes — they are
    valid ranking candidates/targets, exactly like cited papers in the old eval.
    """
    paper_nodes  = set()
    entity_nodes = set()
    for n, d in graph.nodes(data=True):
        if d.get("type") in ("paper", "paper_stub"):
            paper_nodes.add(n)
        else:
            entity_nodes.add(n)

    cites, mentions, entity_entity = [], [], []
    for u, v, data in graph.edges(data=True):
        rel = data.get("relation", "")
        if rel == CITES_REL:
            if u in paper_nodes and v in paper_nodes:
                cites.append((u, CITES_REL, v))
        elif rel == MENTIONS_REL:
            mentions.append((u, MENTIONS_REL, v))
        else:
            if u in entity_nodes and v in entity_nodes and rel:
                entity_entity.append((u, rel, v))

    if mode == "citation":
        raw = cites
    elif mode == "augmented":
        raw = cites + mentions + entity_entity
    else:
        raise ValueError(f"Unknown --graph mode '{mode}'")

    logger.info(f"Graph [{mode}]: {len(paper_nodes)} big nodes "
                f"({sum(1 for n in paper_nodes if graph.nodes[n].get('type') == 'paper')} corpus, "
                f"rest stubs), {len(entity_nodes)} entity nodes | "
                f"cites={len(cites)}, mentions={len(mentions)}, "
                f"entity-entity={len(entity_entity)} → using {len(raw)} raw triples")

    return {"raw": raw, "paper_nodes": paper_nodes, "cites": cites}


def make_split(graph: nx.MultiDiGraph, cites: list, test_frac: float, split_seed: int):
    """
    Split CORPUS papers (not stubs) that have ≥1 citation edge into train/test
    big nodes. Membership depends only on split_seed — identical across graph
    versions, KGE models, and training seeds, so every comparison shares the
    same masked papers.
    """
    import random as _random
    deg = {}
    for h, _, t in cites:
        deg[h] = deg.get(h, 0) + 1
        deg[t] = deg.get(t, 0) + 1
    eligible = sorted(
        n for n, d in graph.nodes(data=True)
        if d.get("type") == "paper" and deg.get(n, 0) > 0
    )
    rng = _random.Random(split_seed)
    rng.shuffle(eligible)
    n_test = max(1, int(len(eligible) * test_frac))
    test  = set(eligible[:n_test])
    train = set(eligible[n_test:])
    logger.info(f"Split (seed={split_seed}): {len(eligible)} eligible corpus papers "
                f"→ {len(train)} train / {len(test)} test (masked)")
    return train, test


def mask_and_index(triples: dict, test_papers: set) -> dict:
    """
    Remove every cites edge incident to a masked (test) paper from training,
    recording those removed edges as per-paper ground truth. Mentions/entity
    triples are NEVER removed — in the augmented version they are what the
    masked paper's embedding is learned from.
    """
    ground_truth = {p: set() for p in test_papers}
    train_raw = []
    for h, r, t in triples["raw"]:
        if r == CITES_REL and (h in test_papers or t in test_papers):
            if h in test_papers:
                ground_truth[h].add(t)
            if t in test_papers:
                ground_truth[t].add(h)
            continue
        train_raw.append((h, r, t))

    # Index entities/relations over the FULL vocabulary (all nodes incl. masked
    # papers) so masked papers have embeddings — untrained in the citation
    # version, content-trained in the augmented version.
    all_nodes = set(triples["paper_nodes"])
    for h, r, t in triples["raw"]:
        all_nodes.add(h)
        all_nodes.add(t)
    entity2id = {n: i for i, n in enumerate(sorted(all_nodes))}
    rels      = sorted({r for _, r, _ in triples["raw"]})
    rel2id    = {r: i for i, r in enumerate(rels)}

    train_ids = [(entity2id[h], rel2id[r], entity2id[t]) for h, r, t in train_raw]

    n_masked_edges = sum(len(v) for v in ground_truth.values())
    logger.info(f"Masked {len(test_papers)} big nodes → removed cites edges "
                f"({n_masked_edges} directed GT links kept aside); "
                f"training on {len(train_ids)} triples, "
                f"{len(entity2id)} entities, {len(rel2id)} relations")

    return {
        "entity2id": entity2id,
        "id2entity": {i: n for n, i in entity2id.items()},
        "rel2id": rel2id,
        "triples": train_ids,
        "ground_truth": ground_truth,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_masked(model, graph_idx: dict, paper_nodes: set, test_papers: set,
                    device: str = "cuda") -> dict:
    """
    For each masked big node: rank ALL other big nodes by cosine similarity of
    entity embeddings (same prediction path as kg_transe_pipeline.py), score
    against its held-out citation neighbours. Augmented/entity nodes are never
    candidates and never targets.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    e2id = graph_idx["entity2id"]

    candidates = sorted(paper_nodes)
    cand_ids   = torch.tensor([e2id[c] for c in candidates], dtype=torch.long, device=device)
    cand_pos   = {c: i for i, c in enumerate(candidates)}

    model.eval()
    with torch.no_grad():
        all_embs  = F.normalize(model.entity_emb.weight, p=2, dim=1).to(device)
        cand_embs = all_embs[cand_ids]                       # (n_papers, dim)

        hits1 = hits5 = hits10 = 0.0
        mrr_sum = 0.0
        per_paper = {}
        n_eval = 0

        for p in sorted(test_papers):
            gt = graph_idx["ground_truth"].get(p, set())
            gt = {g for g in gt if g in cand_pos}
            if not gt:
                continue
            q_emb  = all_embs[e2id[p]].unsqueeze(0)
            scores = (cand_embs @ q_emb.T).squeeze(1)        # (n_papers,)
            scores[cand_pos[p]] = -1e9                       # never rank self

            order = torch.argsort(scores, descending=True).cpu().tolist()
            rank_of = {candidates[idx]: rank + 1 for rank, idx in enumerate(order)}
            best = min(rank_of[g] for g in gt)

            hits1  += best <= 1
            hits5  += best <= 5
            hits10 += best <= 10
            mrr_sum += 1.0 / best
            n_eval += 1
            per_paper[p.replace(PAPER_PREFIX, "")] = {
                "n_gt": len(gt),
                "best_rank": best,
                "gt_ranks": sorted(rank_of[g] for g in gt),
            }

    if n_eval == 0:
        raise ValueError("No masked papers had ground-truth links inside the graph")

    return {
        "n_eval": n_eval,
        "n_candidates": len(candidates),
        "hits@1": hits1 / n_eval,
        "hits@5": hits5 / n_eval,
        "hits@10": hits10 / n_eval,
        "mrr": mrr_sum / n_eval,
        "per_paper": per_paper,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--global-graph-dir", default="output/global_kg")
    ap.add_argument("--extractor", default="fixed",
                    help="fixed or fixed_scinex (subdir of the global graph dir)")
    ap.add_argument("--model", default="gemma-3-12b-it",
                    help="model slug subdir (e.g. gemma-3-12b-it, Qwen3-14B)")
    ap.add_argument("--graph", choices=["citation", "augmented"], required=True,
                    help="citation = big nodes only; augmented = + entity support nodes")
    ap.add_argument("--kge", choices=["transe", "complex", "rotate"], default="rotate")
    ap.add_argument("--loss", choices=["margin", "adv"], default="margin")
    ap.add_argument("--gamma", type=float, default=9.0)
    ap.add_argument("--adv-temp", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--neg-ratio", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1],
                    help="training seeds; results reported as mean ± std")
    ap.add_argument("--test-frac", type=float, default=0.5,
                    help="fraction of eligible corpus papers masked as test big nodes")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="fixes WHICH papers are masked — keep identical across "
                         "the citation/augmented comparison")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-summary", default=None,
                    help="output JSON path (default: output/masked_node_"
                         "<graph>_<kge>_<loss>_<model>.json)")
    args = ap.parse_args()

    graph, meta = load_global_graph(args.global_graph_dir, args.extractor, args.model)
    if graph.number_of_nodes() == 0:
        raise SystemExit(f"Global graph is empty — run kg_main.py --extractor "
                         f"{args.extractor} first ({args.global_graph_dir}/"
                         f"{args.extractor}/{args.model})")
    logger.info(f"Loaded global graph: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges, "
                f"{meta['stats'].get('papers_merged', '?')} papers merged")

    triples = build_triples(graph, args.graph)
    _, test_papers = make_split(graph, triples["cites"], args.test_frac, args.split_seed)
    graph_idx = mask_and_index(triples, test_papers)

    runs = []
    for seed in args.seeds:
        logger.info(f"── seed {seed} ──")
        set_seed(seed)
        model = train_kge(
            graph_idx, kge=args.kge, dim=args.dim, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size, device=args.device,
            seed=seed, neg_ratio=args.neg_ratio, loss_type=args.loss,
            gamma=args.gamma, adv_temp=args.adv_temp,
        )
        res = evaluate_masked(model, graph_idx, triples["paper_nodes"],
                              test_papers, device=args.device)
        logger.info(f"seed {seed}: MRR={res['mrr']:.3f}  H@1={res['hits@1']:.3f}  "
                    f"H@5={res['hits@5']:.3f}  H@10={res['hits@10']:.3f}  "
                    f"({res['n_eval']} masked papers, {res['n_candidates']} candidates)")
        runs.append(res)

    def agg(key):
        vals = [r[key] for r in runs]
        return {"mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    summary = {
        "task": "masked_big_node_link_prediction",
        "graph_version": args.graph,
        "extractor": args.extractor,
        "model": args.model,
        "kge": args.kge,
        "loss": args.loss,
        "gamma": args.gamma if args.loss == "adv" else None,
        "dim": args.dim, "epochs": args.epochs, "neg_ratio": args.neg_ratio,
        "test_frac": args.test_frac, "split_seed": args.split_seed,
        "seeds": args.seeds,
        "papers_merged_in_graph": meta["stats"].get("papers_merged"),
        "n_eval": runs[0]["n_eval"],
        "n_candidates": runs[0]["n_candidates"],
        "metrics": {k: agg(k) for k in ("mrr", "hits@1", "hits@5", "hits@10")},
        "per_seed": [
            {k: r[k] for k in ("mrr", "hits@1", "hits@5", "hits@10")} for r in runs
        ],
        "per_paper_last_seed": runs[-1]["per_paper"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out = args.save_summary or (
        f"output/masked_node_{args.graph}_{args.kge}_{args.loss}_{args.model}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    m = summary["metrics"]
    logger.info(f"\n{'═'*66}\n"
                f"MASKED-BIG-NODE RESULT  [{args.graph} | {args.kge} | {args.loss}]\n"
                f"  MRR    : {m['mrr']['mean']:.3f} ± {m['mrr']['std']:.3f}\n"
                f"  Hits@1 : {m['hits@1']['mean']:.3f} ± {m['hits@1']['std']:.3f}\n"
                f"  Hits@5 : {m['hits@5']['mean']:.3f} ± {m['hits@5']['std']:.3f}\n"
                f"  Hits@10: {m['hits@10']['mean']:.3f} ± {m['hits@10']['std']:.3f}\n"
                f"  ({summary['n_eval']} masked papers, {summary['n_candidates']} candidates, "
                f"{len(args.seeds)} seed(s))\n"
                f"  → {out}\n{'═'*66}")


if __name__ == "__main__":
    main()
