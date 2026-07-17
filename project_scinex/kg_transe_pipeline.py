"""
kg_transe_pipeline.py
---------------------
KG-embedding-based related paper prediction and KG quality evaluation.

Supports three KG embedding models (select with --kge):
  - transe   TransE  (Bordes et al., 2013)  — translation: h + r ≈ t
  - complex  ComplEx (Trouillon et al., 2016) — complex bilinear, handles asymmetry
  - rotate   RotatE  (Sun et al., 2019)     — relation = rotation in complex space
  - all      train+evaluate all three and print a comparison table

The evaluation logic:
  1. Load extractor KG triples → entity-entity edges + paper-entity edges
  2. Load citation networks → paper-paper edges
  3. Train the chosen KGE model(s) on the unified graph
  4. For each paper, predict top-k related papers (cosine sim on entity embeddings)
  5. Evaluate predictions against actual citation links (Hits@k, MRR)

Better triples from the extractor → richer entity representations →
better paper embeddings → more accurate related paper predictions.

Usage:
    # Train and evaluate with TransE (default)
    python3 kg_transe_pipeline.py --output-dir output --extractor fixed --model Qwen3-14B

    # Use a specific embedding model
    python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge rotate

    # Compare all three (TransE vs ComplEx vs RotatE)
    python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge all --epochs 500

    # Predict related papers for a specific paper
    python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B \
        --predict BERT --top-k 10

    # Train only, save embeddings (--kge all suffixes the path per model)
    python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B \
        --epochs 500 --save-model output/kge_model.pt
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Reuse the canonical keyword extractor so candidate nodes that S2 returned with
# no abstract (empty `concepts`) can fall back to concepts mined from their title.
from citation.network import extract_concepts

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PAPER_PREFIX  = "paper:"   # distinguishes paper nodes from entity nodes
CITES_REL     = "cites"
MENTIONS_REL  = "mentions"


# ── Data loading ──────────────────────────────────────────────────────────────

def _norm_entity(s: str) -> str:
    """Normalize an entity/concept surface form so equivalent strings link.
    NOTE: deliberately just lowercase+strip. A Session-17 attempt to also
    singularize the head word (models→model) REGRESSED the best models
    (RotatE MRR 0.423→0.386, ComplEx 0.386→0.351, both with higher variance)
    via over-merging distinct entities/concepts — reverted. Do not re-add
    singularization without an ablation showing it helps."""
    return s.strip().lower()


def load_all_data(output_dir: str, extractor: str, model_name: str) -> dict:
    """
    Scan output_dir for all papers that have:
      - KG triples from the specified extractor
      - citation_network.json

    Returns:
        {
          "kg_triples":       list of (paper_id, subject, predicate, object),
          "citation_edges":   list of (source_s2_id, target_s2_id, paper_id),
          "paper_meta":       {paper_id: {s2_id, title, nodes, edges}},
          "papers_with_kg":   set of paper_ids that have triples,
          "papers_with_cit":  set of paper_ids that have valid citation nets,
        }
    """
    out = Path(output_dir)
    kg_triples     = []   # (paper_id, subj, pred, obj)
    citation_edges = []   # (src_s2, tgt_s2, paper_acl_id)
    paper_meta     = {}

    papers_with_kg  = set()
    papers_with_cit = set()

    for paper_dir in sorted(out.iterdir()):
        if not paper_dir.is_dir() or paper_dir.name == "acl":
            continue
        paper_id = paper_dir.name

        # ── Load KG triples ───────────────────────────────────────────────────
        # rebel extractor: output/{paper}/kg/rebel/triples.json (no model subdir)
        # others:          output/{paper}/kg/{extractor}/{model}/triples.json
        if extractor == "rebel":
            triples_path = paper_dir / "kg" / "rebel" / "triples.json"
        else:
            triples_path = paper_dir / "kg" / extractor / model_name / "triples.json"
        if triples_path.exists():
            try:
                triples = json.loads(triples_path.read_text())
                for t in triples:
                    subj = t.get("subject", "").strip()
                    pred = t.get("predicate", "").strip()
                    obj  = t.get("object", "").strip()
                    if subj and pred and obj:
                        kg_triples.append((paper_id, subj, pred, obj))
                if triples:
                    papers_with_kg.add(paper_id)
            except Exception as e:
                logger.warning(f"  Could not load triples for {paper_id}: {e}")

        # ── Load citation network ─────────────────────────────────────────────
        cnet_path = paper_dir / "citation_network.json"
        if cnet_path.exists():
            try:
                cnet = json.loads(cnet_path.read_text())
                s2_id   = cnet.get("paper_id", "")
                title   = cnet.get("title", "")
                nodes   = cnet.get("nodes", {})
                edges   = cnet.get("edges", [])

                # Validate: title must not be just the ACL ID repeated
                # (sign that S2 lookup failed)
                is_valid = (
                    title
                    and title.lower() != paper_id.lower()
                    and paper_id.lower() not in title.lower().replace(" ", "")
                    and len(title) > 8
                )

                # Full-text concepts for the seed/query paper — richer than the
                # ~15 abstract root_concepts (and more accurate when S2 returned a
                # wrong abstract). Gives query papers a dense content vector.
                fulltext_concepts = []
                oj = paper_dir / "no-llm" / "output.json"
                if oj.exists():
                    try:
                        odata = json.loads(oj.read_text())
                        body = " ".join(
                            f"{sec.get('heading','')} {sec.get('content','')}"
                            for sec in odata.get("sections", [])
                        )
                        fulltext_concepts = extract_concepts(body, top_n=50)
                    except Exception:
                        pass

                paper_meta[paper_id] = {
                    "s2_id":    s2_id,
                    "title":    title,
                    "valid_cit": is_valid,
                    "nodes":    nodes,
                    "edges":    edges,
                    "root_concepts": cnet.get("root_concepts", []),
                    "fulltext_concepts": fulltext_concepts,
                    "year":     cnet.get("year"),
                }

                if is_valid and edges:
                    for edge in edges:
                        src = edge.get("source", "")
                        tgt = edge.get("target", "")
                        if src and tgt:
                            citation_edges.append((src, tgt, paper_id))
                    papers_with_cit.add(paper_id)

            except Exception as e:
                logger.warning(f"  Could not load citation network for {paper_id}: {e}")

    logger.info(f"Loaded {len(kg_triples)} KG triples from {len(papers_with_kg)} papers")
    logger.info(f"Loaded {len(citation_edges)} citation edges from {len(papers_with_cit)} papers")
    logger.info(f"Papers with KG:       {sorted(papers_with_kg)}")
    logger.info(f"Papers with citations: {sorted(papers_with_cit)}")

    return {
        "kg_triples":      kg_triples,
        "citation_edges":  citation_edges,
        "paper_meta":      paper_meta,
        "papers_with_kg":  papers_with_kg,
        "papers_with_cit": papers_with_cit,
    }


# ── Graph construction ────────────────────────────────────────────────────────

def build_unified_graph(data: dict) -> dict:
    """
    Build the training graph for TransE from:
      1. Entity-entity triples from KG extraction (CEO predicates)
      2. Paper-entity "mentions" triples  (paper → entity, one per entity in triples)

    Citation edges are deliberately excluded from training — they serve as
    the held-out ground truth for evaluation only.  The evaluation question is:
    "Do paper embeddings learned purely from KG content predict actual citation
    links?" A yes means the triples capture real scientific relationships.

    Returns vocab dicts and indexed triple arrays.
    """
    entity2id: dict[str, int] = {}
    rel2id:    dict[str, int] = {}
    triples_raw: list[tuple[str, str, str]] = []

    def eid(name: str) -> int:
        if name not in entity2id:
            entity2id[name] = len(entity2id)
        return entity2id[name]

    def rid(name: str) -> int:
        if name not in rel2id:
            rel2id[name] = len(rel2id)
        return rel2id[name]

    paper_node_ids: set[str] = set()
    paper_entities: dict[str, set[str]] = defaultdict(set)

    # 1. Entity-entity triples (KG extraction output)
    for (paper_id, subj, pred, obj) in data["kg_triples"]:
        subj_n = _norm_entity(subj)
        obj_n  = _norm_entity(obj)
        triples_raw.append((subj_n, pred, obj_n))
        paper_entities[paper_id].add(subj_n)
        paper_entities[paper_id].add(obj_n)

    # 2. Paper-entity mentions (links paper nodes to their extracted entities)
    for paper_id, entities in paper_entities.items():
        pnode = PAPER_PREFIX + paper_id
        paper_node_ids.add(pnode)
        for ent in sorted(entities):   # sorted: set order is non-deterministic across runs
            triples_raw.append((pnode, MENTIONS_REL, ent))

    # Also add paper nodes for papers that have citation data but no KG triples
    # so they can still be ranked in evaluation (their embedding initialised randomly)
    s2_to_acl: dict[str, str] = {}
    for pid, meta in data["paper_meta"].items():
        s2 = meta.get("s2_id", "")
        if s2:
            s2_to_acl[s2] = pid

    for (src_s2, tgt_s2, acl_id) in data["citation_edges"]:
        src_node = PAPER_PREFIX + s2_to_acl.get(src_s2, src_s2)
        tgt_node = PAPER_PREFIX + s2_to_acl.get(tgt_s2, tgt_s2)
        paper_node_ids.add(src_node)
        paper_node_ids.add(tgt_node)
        # NOTE: citation edges are NOT added to triples_raw —
        # they are held out for evaluation only

    # 3. Concept content for every paper in citation networks.
    #    Each citation-network node stores `concepts` extracted from its abstract.
    #    These give cited/citing papers a content representation so they can be
    #    compared against query papers. Without this, cited papers would be
    #    isolated nodes with random embeddings and never predictable.
    #    Concepts share the entity vocabulary via CONCEPT_PREFIX-free normalisation
    #    so overlapping terms ("language model", "attention") link papers together.
    for pid, meta in data["paper_meta"].items():
        nodes = meta.get("nodes", {})
        for s2_node_id, node_data in nodes.items():
            paper_node = PAPER_PREFIX + s2_to_acl.get(s2_node_id, s2_node_id)
            paper_node_ids.add(paper_node)
            # Fall back to title-mined concepts when S2 gave no abstract → empty
            # concepts (~27% of nodes). Without this they are isolated random-
            # embedding nodes that can be neither retrieved nor meaningfully ranked.
            node_concepts = node_data.get("concepts", []) or extract_concepts(node_data.get("title", ""))
            for concept in node_concepts:
                concept_n = _norm_entity(concept)
                if concept_n and len(concept_n) > 2:
                    triples_raw.append((paper_node, MENTIONS_REL, concept_n))
        # Also give the seed (query) paper itself concept content from
        # root_concepts (top level of citation_network.json). Without this the
        # corpus paper is represented ONLY by its CEO entities, while candidate
        # papers carry abstract concepts — a vocabulary mismatch that makes the
        # cosine ranking near-random. This puts both in the same concept space.
        seed_node = PAPER_PREFIX + pid
        paper_node_ids.add(seed_node)
        # Union of abstract root_concepts + full-text concepts (sorted for
        # deterministic vocab ordering across runs).
        seed_concepts = set(meta.get("root_concepts", [])) | set(meta.get("fulltext_concepts", []))
        for concept in sorted(seed_concepts):
            concept_n = _norm_entity(concept)
            if concept_n and len(concept_n) > 2:
                triples_raw.append((seed_node, MENTIONS_REL, concept_n))

    # Build vocab
    for (h, r, t) in triples_raw:
        eid(h); eid(t); rid(r)

    # Register all paper nodes in vocab even if they have no training triples
    for pnode in sorted(paper_node_ids):   # sorted: deterministic vocab ids across runs
        eid(pnode)

    # Indexed triples (training set only — no citation edges)
    # sorted (not list(set(...))): set→list order is non-deterministic across runs,
    # which would shuffle batching and break seed reproducibility.
    triples_idx = sorted(set(
        (entity2id[h], rel2id[r], entity2id[t])
        for h, r, t in triples_raw
        if h in entity2id and t in entity2id
    ))

    logger.info(f"Training graph: {len(entity2id)} nodes, "
                f"{len(rel2id)} relations, {len(triples_idx)} triples")
    logger.info(f"Paper nodes in vocab: {len(paper_node_ids)} "
                f"(citation edges held out for evaluation)")
    logger.info(f"Relations: {list(rel2id.keys())}")

    return {
        "entity2id":      entity2id,
        "id2entity":      {v: k for k, v in entity2id.items()},
        "rel2id":         rel2id,
        "id2rel":         {v: k for k, v in rel2id.items()},
        "triples":        triples_idx,
        "paper_node_ids": paper_node_ids,
        "paper_entities": dict(paper_entities),
        "s2_to_acl":      s2_to_acl,
    }


# ── TransE model ──────────────────────────────────────────────────────────────

class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    (Bordes et al., 2013)

    Score(h, r, t) = -||h + r - t||  (L2 norm, negated so higher = better)
    Training: margin-based ranking loss.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        dim: int = 64,
        margin: float = 1.0,
        norm: int = 2,
    ):
        super().__init__()
        self.dim      = dim
        self.margin   = margin
        self.norm     = norm

        self.entity_emb   = nn.Embedding(n_entities,  dim)
        self.relation_emb = nn.Embedding(n_relations, dim)

        # Uniform initialisation in [-6/sqrt(dim), 6/sqrt(dim)]
        bound = 6.0 / (dim ** 0.5)
        nn.init.uniform_(self.entity_emb.weight,   -bound, bound)
        nn.init.uniform_(self.relation_emb.weight, -bound, bound)

        # Normalise entity embeddings to unit sphere
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(
                self.entity_emb.weight.data, p=2, dim=1
            )

    def forward(self, heads, rels, tails):
        """Return TransE scores (negative distance — higher is better)."""
        h = F.normalize(self.entity_emb(heads), p=2, dim=1)
        r = self.relation_emb(rels)
        t = F.normalize(self.entity_emb(tails), p=2, dim=1)
        return -torch.norm(h + r - t, p=self.norm, dim=1)

    def loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self.forward(pos_h, pos_r, pos_t)
        neg_score = self.forward(neg_h, neg_r, neg_t)
        return F.relu(self.margin - pos_score + neg_score).mean()


# ── ComplEx model ───────────────────────────────────────────────────────────────

class ComplEx(nn.Module):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction (Trouillon et al., 2016).

    Entities/relations live in complex space C^dim (stored as real‖imag of width
    2*dim). Score(h, r, t) = Re(<h, r, conj(t)>), higher = better. Handles
    asymmetric relations (which TransE/DistMult struggle with) via the conjugate.

    `entity_emb` holds the full [real‖imag] entity vector so the downstream
    cosine-similarity prediction code works unchanged.
    """

    def __init__(self, n_entities, n_relations, dim=64, margin=1.0, norm=2):
        super().__init__()
        self.dim    = dim                 # complex dim; stored width is 2*dim
        self.margin = margin
        self.entity_emb   = nn.Embedding(n_entities,  2 * dim)
        self.relation_emb = nn.Embedding(n_relations, 2 * dim)
        bound = 6.0 / (dim ** 0.5)
        nn.init.uniform_(self.entity_emb.weight,   -bound, bound)
        nn.init.uniform_(self.relation_emb.weight, -bound, bound)

    def forward(self, heads, rels, tails):
        h_re, h_im = self.entity_emb(heads).chunk(2, dim=1)
        r_re, r_im = self.relation_emb(rels).chunk(2, dim=1)
        t_re, t_im = self.entity_emb(tails).chunk(2, dim=1)
        # Re(<h, r, conj(t)>)
        score = (
            (h_re * r_re * t_re).sum(1)
            + (h_re * r_im * t_im).sum(1)
            + (h_im * r_re * t_im).sum(1)
            - (h_im * r_im * t_re).sum(1)
        )
        return score  # higher = better

    def loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self.forward(pos_h, pos_r, pos_t)
        neg_score = self.forward(neg_h, neg_r, neg_t)
        return F.relu(self.margin - pos_score + neg_score).mean()


# ── RotatE model ────────────────────────────────────────────────────────────────

class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation (Sun et al., 2019).

    Each relation is an element-wise rotation in complex space (modulus 1), so
    t ≈ h ∘ r. Score(h, r, t) = -||h ∘ r - t|| (complex modulus, L1 over dims),
    higher = better. Models symmetry, antisymmetry, inversion and composition.

    Entities are complex (real‖imag, width 2*dim); relations are phases (width
    dim) mapped to unit-modulus rotations via (cos θ, sin θ). `entity_emb` holds
    the [real‖imag] vector so the cosine-similarity prediction code works as-is.
    """

    def __init__(self, n_entities, n_relations, dim=64, margin=1.0, norm=2):
        super().__init__()
        self.dim    = dim
        self.margin = margin
        self.entity_emb   = nn.Embedding(n_entities,  2 * dim)   # real‖imag
        self.relation_emb = nn.Embedding(n_relations, dim)       # phases θ
        bound = 6.0 / (dim ** 0.5)
        nn.init.uniform_(self.entity_emb.weight, -bound, bound)
        # phases in (-pi, pi]
        nn.init.uniform_(self.relation_emb.weight, -3.14159265, 3.14159265)

    def forward(self, heads, rels, tails):
        h_re, h_im = self.entity_emb(heads).chunk(2, dim=1)
        t_re, t_im = self.entity_emb(tails).chunk(2, dim=1)
        phase = self.relation_emb(rels)
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        # complex rotation h ∘ r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        # complex modulus of (h∘r - t), summed over dims (L1 of moduli)
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        dist = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-9).sum(1)
        return -dist  # higher = better

    def loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self.forward(pos_h, pos_r, pos_t)
        neg_score = self.forward(neg_h, neg_r, neg_t)
        return F.relu(self.margin - pos_score + neg_score).mean()


# ── Model registry ──────────────────────────────────────────────────────────────

KGE_MODELS = {
    "transe":  TransE,
    "complex": ComplEx,
    "rotate":  RotatE,
}


def build_kge_model(kge: str, n_entities: int, n_relations: int,
                    dim: int = 64, margin: float = 1.0):
    """Instantiate a KG-embedding model by name (transe/complex/rotate)."""
    kge = kge.lower()
    if kge not in KGE_MODELS:
        raise ValueError(f"Unknown KGE model '{kge}'. Choose from {list(KGE_MODELS)}")
    return KGE_MODELS[kge](n_entities, n_relations, dim=dim, margin=margin)


# ── Training ──────────────────────────────────────────────────────────────────

def _corrupt(triple, n_entities: int, rng: random.Random) -> tuple:
    """Replace head or tail randomly for negative sampling."""
    h, r, t = triple
    if rng.random() < 0.5:
        h = rng.randint(0, n_entities - 1)
    else:
        t = rng.randint(0, n_entities - 1)
    return (h, r, t)


def set_seed(seed: int) -> None:
    """Seed all RNGs that affect training so a run is reproducible.

    Covers the two unseeded torch sources (embedding init via nn.init.uniform_
    and the per-epoch torch.randperm shuffle) plus python's random. Without
    this, weight init alone makes every run land on different Hits@k/MRR.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_kge(
    graph: dict,
    kge: str = "transe",
    dim: int = 64,
    margin: float = 1.0,
    lr: float = 0.01,
    epochs: int = 500,
    batch_size: int = 256,
    device: str = "cuda",
    seed: int | None = None,
    neg_ratio: int = 1,
    loss_type: str = "margin",
    gamma: float = 9.0,
    adv_temp: float = 1.0,
) -> nn.Module:
    """Train a KG-embedding model (transe/complex/rotate) on the unified graph.

    If ``seed`` is given the run is fully reproducible; if None the historical
    behaviour is kept (torch unseeded, negative sampling fixed at 42).
    """
    if seed is not None:
        set_seed(seed)
    n_ent = len(graph["entity2id"])
    n_rel = len(graph["rel2id"])
    triples = graph["triples"]

    if not triples:
        raise ValueError("No triples to train on — check extraction output")

    logger.info(f"Training {kge.upper()}: {n_ent} entities, {n_rel} relations, "
                f"{len(triples)} triples, dim={dim}, epochs={epochs}, "
                f"loss={loss_type}" + (f", γ={gamma}, α={adv_temp}" if loss_type == "adv" else ""))

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = build_kge_model(kge, n_ent, n_rel, dim=dim, margin=margin).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Keep triples on-device; sample negatives with a seeded torch generator so
    # the run stays reproducible WITHOUT the slow per-element Python _corrupt loop
    # (the old loop was CPU-bound and ~20-30× slower, esp. with neg_ratio>1).
    triple_tensor = torch.tensor(triples, dtype=torch.long, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed if seed is not None else 42)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        # Shuffle triples each epoch
        perm = torch.randperm(len(triple_tensor), device=device, generator=gen)
        shuffled = triple_tensor[perm]

        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]

            if loss_type == "adv":
                # Self-adversarial negative sampling (Sun et al. 2019). Negatives
                # are kept GROUPED per positive ([B, K]) so hard negatives can be
                # softmax-weighted. forward() returns score (higher = better).
                B = batch.size(0)
                pos_h, pos_r, pos_t = batch[:, 0], batch[:, 1], batch[:, 2]
                K = max(neg_ratio, 1)
                ph = pos_h.unsqueeze(1).expand(B, K).reshape(-1)
                pr = pos_r.unsqueeze(1).expand(B, K).reshape(-1)
                pt = pos_t.unsqueeze(1).expand(B, K).reshape(-1)
                n = B * K
                rand_ent     = torch.randint(0, n_ent, (n,), device=device, generator=gen)
                corrupt_head = torch.rand(n, device=device, generator=gen) < 0.5
                neg_h = torch.where(corrupt_head, rand_ent, ph)
                neg_t = torch.where(corrupt_head, pt, rand_ent)
                neg_r = pr

                optimizer.zero_grad()
                pos_score = model.forward(pos_h, pos_r, pos_t)             # [B]
                neg_score = model.forward(neg_h, neg_r, neg_t).view(B, K)  # [B, K]
                pos_loss = -F.logsigmoid(gamma + pos_score)               # [B]
                if adv_temp and adv_temp > 0:
                    weights  = torch.softmax(neg_score * adv_temp, dim=1).detach()
                    neg_loss = -(weights * F.logsigmoid(-(gamma + neg_score))).sum(1)
                else:
                    neg_loss = -F.logsigmoid(-(gamma + neg_score)).mean(1)
                loss = (pos_loss + neg_loss).mean() / 2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
                continue

            # ── margin-ranking loss (historical default) ────────────────────────
            # Tile each positive neg_ratio times so it is contrasted against
            # neg_ratio distinct corruptions (more negatives → sharper ranking).
            if neg_ratio > 1:
                batch = batch.repeat(neg_ratio, 1)
            pos_h, pos_r, pos_t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Vectorized negative sampling: corrupt head or tail of each positive
            # with a random entity (relations kept). Pure torch → runs on GPU.
            n = batch.size(0)
            rand_ent     = torch.randint(0, n_ent, (n,), device=device, generator=gen)
            corrupt_head = torch.rand(n, device=device, generator=gen) < 0.5
            neg_h = torch.where(corrupt_head, rand_ent, pos_h)
            neg_t = torch.where(corrupt_head, pos_t, rand_ent)
            neg_r = pos_r

            optimizer.zero_grad()
            loss = model.loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        if epoch % 50 == 0 or epoch == 1:
            avg = total_loss / max(n_batches, 1)
            logger.info(f"  Epoch {epoch:4d}/{epochs} — loss: {avg:.4f}")

    model.eval()
    return model


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_related_papers(
    query_paper: str,
    model: TransE,
    graph: dict,
    top_k: int = 10,
    device: str = "cuda",
) -> list[dict]:
    """
    Given a paper ID, predict the top-k most related papers using embedding similarity.

    Strategy: find other paper nodes whose entity embedding is closest to the
    query paper's entity embedding (cosine similarity). Closer papers in
    embedding space share more similar entity/concept structures.

    Returns list of {paper_id, score, title} sorted by score desc.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    e2id   = graph["entity2id"]
    id2e   = graph["id2entity"]

    query_node = PAPER_PREFIX + query_paper
    if query_node not in e2id:
        logger.warning(f"Paper '{query_paper}' not in graph (node: {query_node})")
        return []

    query_id = e2id[query_node]

    # All paper node IDs in the graph
    paper_ids_in_graph = [
        (name, e2id[name])
        for name in graph["paper_node_ids"]
        if name in e2id and name != query_node
    ]

    if not paper_ids_in_graph:
        logger.warning("No other paper nodes in graph")
        return []

    model.eval()
    with torch.no_grad():
        all_embs  = F.normalize(model.entity_emb.weight, p=2, dim=1).to(device)
        query_emb = all_embs[query_id].unsqueeze(0)  # (1, dim)

        # Get embeddings for all paper nodes
        paper_indices = torch.tensor(
            [eid for _, eid in paper_ids_in_graph], dtype=torch.long, device=device
        )
        paper_embs    = all_embs[paper_indices]  # (n_papers, dim)

        # Cosine similarity
        scores = (paper_embs @ query_emb.T).squeeze(1)  # (n_papers,)

        topk_scores, topk_indices = scores.topk(
            min(top_k, len(paper_ids_in_graph))
        )

    results = []
    for i, score in zip(topk_indices.cpu().tolist(), topk_scores.cpu().tolist()):
        node_name = paper_ids_in_graph[i][0]
        paper_key = node_name.replace(PAPER_PREFIX, "")
        results.append({
            "paper_id": paper_key,
            "score":    round(score, 4),
        })

    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: TransE,
    graph: dict,
    data: dict,
    device: str = "cuda",
    eval_split: str = "all",
    val_frac: float = 0.5,
    split_seed: int = 42,
    predict_fn=None,
) -> dict:
    """
    Evaluate link prediction quality for papers with known citation links.

    For each paper P with a valid citation network:
      1. Predict top-k related papers
      2. Check how many actual citing/cited papers appear in top-k
      3. Compute Hits@1, Hits@5, Hits@10, MRR

    ``eval_split`` controls which query papers are scored, for honest
    hyperparameter selection (avoid tuning on the test set):
      - "all"  : every eligible paper (historical behaviour, default).
      - "val"  : a fixed ``val_frac`` fraction of the papers — tune γ/dim/epochs here.
      - "test" : the disjoint remainder — report final numbers here, untouched by tuning.
    The val/test membership is fixed by ``split_seed`` ONLY (independent of the
    training ``--seed``), so the same papers are val across every config and every
    training seed — that's what makes model selection on "val" consistent.

    Returns evaluation metrics dict.
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    e2id       = graph["entity2id"]
    s2_to_acl  = {}
    for pid, meta in data["paper_meta"].items():
        s2 = meta.get("s2_id", "")
        if s2:
            s2_to_acl[s2] = pid

    # Readable title for any paper node — whether it ended up keyed by our ACL id
    # (a corpus paper) or by a raw S2 hash (a cited paper outside the corpus).
    s2_to_title = {}
    for pid, meta in data["paper_meta"].items():
        if meta.get("s2_id") and meta.get("title"):
            s2_to_title.setdefault(meta["s2_id"], meta["title"])
        for s2_nid, nd in meta.get("nodes", {}).items():
            if nd.get("title"):
                s2_to_title.setdefault(s2_nid, nd["title"])

    def label(pid: str) -> str:
        """Human-readable title for a predicted/ground-truth paper id."""
        if pid in data["paper_meta"]:
            return data["paper_meta"][pid].get("title", "") or pid
        return s2_to_title.get(pid, "")

    # Publication year for any paper node (keyed by the id form used in predictions:
    # ACL id for corpus papers, raw S2 hash otherwise) — for the temporal filter.
    node_year = {}
    for pid, meta in data["paper_meta"].items():
        if meta.get("year"):
            node_year[pid] = meta["year"]
        for s2_nid, nd in meta.get("nodes", {}).items():
            key = s2_to_acl.get(s2_nid, s2_nid)
            if nd.get("year") and key not in node_year:
                node_year[key] = nd["year"]

    hits_at = {1: [], 5: [], 10: []}
    mrr_scores = []
    # Temporal-filtered ("papers it cited" task): rank only candidates published
    # ≤ the query's year, GT = neighbors ≤ query year. Removes provably-impossible
    # future candidates. Reported separately (filt_*) — a different, smaller task.
    filt_hits_at = {1: [], 5: [], 10: []}
    filt_mrr_scores = []
    paper_results = {}

    papers_to_eval = [
        pid for pid in data["papers_with_cit"]
        if (PAPER_PREFIX + pid) in e2id
    ]

    if not papers_to_eval:
        logger.warning("No papers with both citation data and graph presence for evaluation")
        return {}

    # ── Validation/test split of the QUERY papers (for honest tuning) ──────────
    # Deterministic, fixed by split_seed alone so val/test membership is identical
    # across every hyperparameter config and every training seed.
    if eval_split not in ("all", "val", "test"):
        raise ValueError(f"eval_split must be all/val/test, got {eval_split!r}")
    if eval_split != "all":
        ordered = sorted(papers_to_eval)
        random.Random(split_seed).shuffle(ordered)
        n_val = int(round(len(ordered) * val_frac))
        val_set = set(ordered[:n_val])
        papers_to_eval = [p for p in papers_to_eval
                          if (p in val_set) == (eval_split == "val")]
        logger.info(f"Eval split '{eval_split}' (val_frac={val_frac}, split_seed={split_seed}): "
                    f"{len(papers_to_eval)} of {len(ordered)} papers")

    logger.info(f"Evaluating {len(papers_to_eval)} papers ...")

    for paper_id in papers_to_eval:
        meta   = data["paper_meta"].get(paper_id, {})
        edges  = meta.get("edges", [])
        nodes  = meta.get("nodes", {})

        # Ground truth: all S2 IDs directly connected by a citation edge
        gt_s2_ids = set()
        for edge in edges:
            gt_s2_ids.add(edge.get("source", ""))
            gt_s2_ids.add(edge.get("target", ""))
        gt_s2_ids.discard("")
        # Convert to ACL IDs where possible
        gt_nodes = set()
        for s2 in gt_s2_ids:
            acl = s2_to_acl.get(s2, s2)
            gt_nodes.add(PAPER_PREFIX + acl)

        # Get all paper predictions (no top-k limit for rank calculation).
        # A predict_fn (e.g. a text-similarity baseline) can replace the KGE
        # ranker; it must return the same [{paper_id, score}, ...] sorted desc,
        # so all metrics/splits/ground-truth below are computed identically.
        if predict_fn is not None:
            all_preds = predict_fn(paper_id)
        else:
            all_preds = predict_related_papers(
                paper_id, model, graph, top_k=len(graph["paper_node_ids"]),
                device=device
            )

        # Compute rank of each ground truth paper
        pred_ids = [p["paper_id"] for p in all_preds]
        ranks = []
        for gt_node in gt_nodes:
            gt_pid = gt_node.replace(PAPER_PREFIX, "")
            if gt_pid in pred_ids:
                rank = pred_ids.index(gt_pid) + 1
                ranks.append(rank)

        if not ranks:
            continue

        # Metrics per paper
        best_rank = min(ranks)
        h1  = int(best_rank <= 1)
        h5  = int(best_rank <= 5)
        h10 = int(best_rank <= 10)
        mrr = 1.0 / best_rank

        hits_at[1].append(h1)
        hits_at[5].append(h5)
        hits_at[10].append(h10)
        mrr_scores.append(mrr)

        # Top-10 predictions, each flagged as a real citation or not, with title
        # so a "wrong" prediction can be eyeballed for topical relatedness.
        top10 = []
        for i, p in enumerate(all_preds[:10]):
            pid = p["paper_id"]
            top10.append({
                "rank":            i + 1,
                "paper_id":        pid,
                "title":           label(pid),
                "score":           p["score"],
                "is_true_citation": (PAPER_PREFIX + pid) in gt_nodes,
            })

        # Where every true citation neighbor actually ranked (None = not rankable,
        # i.e. not present as a node in the graph). Sorted best-rank first.
        gt_ranks = []
        for gt_node in gt_nodes:
            gt_pid = gt_node.replace(PAPER_PREFIX, "")
            r = pred_ids.index(gt_pid) + 1 if gt_pid in pred_ids else None
            gt_ranks.append({"paper_id": gt_pid, "title": label(gt_pid), "rank": r})
        gt_ranks.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0))

        # ── Directional reference-prediction metric ("papers it cites") ──────
        # GT = the papers the query ACTUALLY cites, taken from the true citation
        # edge DIRECTION (edge source == this paper → target is a reference), NOT
        # from a year proxy. This is the consistent directional task: a paper can
        # only "predict" the references it points to; newer papers that cite it
        # (citers) belong to THOSE papers' reference sets, not this one's. The
        # whole candidate pool is ranked (no year filtering) — references are
        # naturally older, so future papers are just distractors.
        # (filt_* keys kept for backward compat; meaning is now edge-direction,
        #  superseding the old year-filtered definition.)
        seed_s2 = meta.get("s2_id", "")
        ref_nodes = set()
        for edge in edges:
            if edge.get("source") == seed_s2 and edge.get("target"):
                ref_acl = s2_to_acl.get(edge["target"], edge["target"])
                ref_nodes.add(PAPER_PREFIX + ref_acl)
        filt_best = None
        filt_ranks = [
            pred_ids.index(n.replace(PAPER_PREFIX, "")) + 1
            for n in ref_nodes
            if n.replace(PAPER_PREFIX, "") in pred_ids
        ]
        if filt_ranks:
            filt_best = min(filt_ranks)
            filt_hits_at[1].append(int(filt_best <= 1))
            filt_hits_at[5].append(int(filt_best <= 5))
            filt_hits_at[10].append(int(filt_best <= 10))
            filt_mrr_scores.append(1.0 / filt_best)

        paper_results[paper_id] = {
            "title":           label(paper_id),
            "best_rank":       best_rank,
            "hits@1":          h1,
            "hits@5":          h5,
            "hits@10":         h10,
            "mrr":             round(mrr, 4),
            "gt_papers_found": len(ranks),
            "gt_papers_total": len(gt_nodes),
            "total_candidates": len(all_preds),
            "filt_best_rank":  filt_best,   # temporal-filtered best rank (None if n/a)
            "top10_predictions": top10,
            "gt_ranks":         gt_ranks,
        }

    if not mrr_scores:
        logger.warning("No evaluation data — check citation network validity")
        return {}

    n = len(mrr_scores)
    summary = {
        "n_papers_evaluated": n,
        "hits@1":  round(sum(hits_at[1])  / n, 4),
        "hits@5":  round(sum(hits_at[5])  / n, 4),
        "hits@10": round(sum(hits_at[10]) / n, 4),
        "mrr":     round(sum(mrr_scores)  / n, 4),
        "per_paper": paper_results,
    }

    # Temporal-filtered ("papers it cited") metric — separate, smaller task.
    fn = len(filt_mrr_scores)
    if fn:
        summary.update({
            "filt_n_papers_evaluated": fn,
            "filt_hits@1":  round(sum(filt_hits_at[1])  / fn, 4),
            "filt_hits@5":  round(sum(filt_hits_at[5])  / fn, 4),
            "filt_hits@10": round(sum(filt_hits_at[10]) / fn, 4),
            "filt_mrr":     round(sum(filt_mrr_scores)  / fn, 4),
        })

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="TransE-based related paper prediction for KG evaluation"
    )
    ap.add_argument("--output-dir",  default="output",
                    help="Pipeline output directory (default: output)")
    ap.add_argument("--extractor",   default="fixed",
                    choices=["fixed", "fixed_scinex", "llm", "rebel"],
                    help="Which extractor's triples to use (default: fixed). "
                         "'fixed_scinex' = fixed extraction under the scinex ontology.")
    ap.add_argument("--model",       default="Qwen3-32B",
                    help="LLM model name subdirectory (default: Qwen3-32B)")
    ap.add_argument("--kge",         default="transe",
                    choices=["transe", "complex", "rotate", "all"],
                    help="KG embedding model to train/evaluate (default: transe; "
                         "'all' runs transe+complex+rotate and prints a comparison)")
    ap.add_argument("--dim",         type=int,   default=128,
                    help="Embedding dimension (default: 128)")
    ap.add_argument("--margin",      type=float, default=1.0,
                    help="Margin for ranking loss (default: 1.0)")
    ap.add_argument("--lr",          type=float, default=0.01,
                    help="Learning rate (default: 0.01)")
    ap.add_argument("--epochs",      type=int,   default=1000,
                    help="Training epochs (default: 1000)")
    ap.add_argument("--neg-ratio",   type=int,   default=10,
                    help="Negative samples per positive (default: 10)")
    ap.add_argument("--loss",        default="margin", choices=["margin", "adv"],
                    help="Training loss: 'margin' (ranking, the historical default) "
                         "or 'adv' (self-adversarial logsigmoid, Sun et al. 2019). "
                         "'margin' keeps the reproducible 0.432 baseline.")
    ap.add_argument("--gamma",       type=float, default=9.0,
                    help="Margin/offset γ for the --loss adv logsigmoid (default: 9.0)")
    ap.add_argument("--adv-temp",    type=float, default=1.0,
                    help="Self-adversarial temperature α for --loss adv "
                         "(0 = uniform negative weighting; default: 1.0)")
    ap.add_argument("--eval-split",  default="all", choices=["all", "val", "test"],
                    help="Which query papers to score: 'all' (default), 'val' "
                         "(tune hyperparameters here), or 'test' (report here). "
                         "val/test are disjoint, fixed by --split-seed.")
    ap.add_argument("--val-frac",    type=float, default=0.5,
                    help="Fraction of query papers in the validation split (default: 0.5)")
    ap.add_argument("--split-seed",  type=int,   default=42,
                    help="Seed for the val/test query-paper partition — independent "
                         "of --seed so membership is stable across configs (default: 42)")
    ap.add_argument("--batch-size",  type=int,   default=256,
                    help="Batch size (default: 256)")
    ap.add_argument("--device",      default="cuda",
                    help="Device: cuda or cpu (default: cuda)")
    ap.add_argument("--predict",     default=None,
                    help="Paper ID to predict related papers for")
    ap.add_argument("--top-k",       type=int,   default=10,
                    help="Number of related papers to predict (default: 10)")
    ap.add_argument("--save-model",  default=None,
                    help="Path to save trained model + graph vocab")
    ap.add_argument("--load-model",  default=None,
                    help="Path to load pretrained model (skips training)")
    ap.add_argument("--eval-only",   action="store_true",
                    help="Only evaluate, skip training (requires --load-model)")
    ap.add_argument("--save-results", default=None,
                    help="Save evaluation results to JSON file")
    ap.add_argument("--seed",        type=int,   default=None,
                    help="Random seed for reproducible training (seeds torch "
                         "init + shuffle + negative sampling). Omit for the "
                         "historical non-deterministic behaviour. For a "
                         "defensible result, run several seeds and report "
                         "mean ± std (see kge_multiseed.py).")
    args = ap.parse_args()

    if args.seed is not None:
        print(f"  Seed: {args.seed} (reproducible run)")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n── Loading data from '{args.output_dir}' ──")
    data = load_all_data(args.output_dir, args.extractor, args.model)

    if not data["kg_triples"] and not data["citation_edges"]:
        print("❌ No data found. Check --output-dir, --extractor, --model.")
        return

    # ── Build graph ───────────────────────────────────────────────────────────
    print("\n── Building unified graph ──")
    graph = build_unified_graph(data)

    if not graph["triples"]:
        print("❌ Empty graph. No triples to train on.")
        return

    # ── Run one or more KGE models ──────────────────────────────────────────────
    kge_list = list(KGE_MODELS) if args.kge == "all" else [args.kge]
    all_metrics = {}
    for kge in kge_list:
        results = run_kge(kge, graph, data, args)
        if results:
            all_metrics[kge] = results

    # ── Comparison table (when running >1 model) ────────────────────────────────
    if len(kge_list) > 1 and all_metrics:
        print(f"\n── KGE comparison (extractor: {args.extractor}/{args.model}) ──")
        print(f"  {'Model':<10}{'Hits@1':>9}{'Hits@5':>9}{'Hits@10':>9}{'MRR':>9}")
        print(f"  {'-'*45}")
        for kge in kge_list:
            r = all_metrics.get(kge)
            if r:
                print(f"  {kge.upper():<10}{r['hits@1']:>9.4f}{r['hits@5']:>9.4f}"
                      f"{r['hits@10']:>9.4f}{r['mrr']:>9.4f}")


def run_kge(kge: str, graph: dict, data: dict, args) -> dict | None:
    """Train (or load) one KGE model, optionally predict, then evaluate + save."""
    # ── Train or load model ───────────────────────────────────────────────────
    if args.load_model and Path(args.load_model).exists():
        print(f"\n── Loading model from {args.load_model} ──")
        checkpoint = torch.load(args.load_model, map_location="cpu")
        ckpt_kge = checkpoint.get("kge", kge)
        model = build_kge_model(
            ckpt_kge,
            len(graph["entity2id"]),
            len(graph["rel2id"]),
            dim=checkpoint.get("dim", args.dim),
        )
        model.load_state_dict(checkpoint["model_state"])
        kge = ckpt_kge
    else:
        print(f"\n── Training {kge.upper()} ({args.epochs} epochs) ──")
        model = train_kge(
            graph,
            kge=kge,
            dim=args.dim,
            margin=args.margin,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            neg_ratio=args.neg_ratio,
            loss_type=args.loss,
            gamma=args.gamma,
            adv_temp=args.adv_temp,
        )

    # ── Save model ────────────────────────────────────────────────────────────
    if args.save_model:
        # When running multiple models, suffix the path with the kge name
        save_path = Path(args.save_model)
        if len(KGE_MODELS) > 1 and args.kge == "all":
            save_path = save_path.with_name(f"{save_path.stem}_{kge}{save_path.suffix}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "kge":         kge,
            "entity2id":   graph["entity2id"],
            "rel2id":      graph["rel2id"],
            "dim":         args.dim,
        }, save_path)
        print(f"Model saved → {save_path}")

    # ── Predict related papers ────────────────────────────────────────────────
    if args.predict:
        print(f"\n── [{kge.upper()}] Top-{args.top_k} related papers for '{args.predict}' ──")
        predictions = predict_related_papers(
            args.predict, model, graph, top_k=args.top_k, device=args.device
        )
        if predictions:
            for i, p in enumerate(predictions, 1):
                pid = p["paper_id"]
                score = p["score"]
                # Try to get title from citation network metadata
                meta = data["paper_meta"]
                title = ""
                if pid in meta:
                    title = meta[pid].get("title", "")
                else:
                    # Look through all citation networks for this S2 ID
                    for pm in meta.values():
                        if pid in pm.get("nodes", {}):
                            title = pm["nodes"][pid].get("title", "")
                            break
                print(f"  {i:2d}. [{score:.4f}] {pid[:30]:30s} {title[:50]}")
        else:
            print("  No predictions (paper not in graph)")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n── [{kge.upper()}] Evaluation (extractor: {args.extractor}/{args.model}) ──")
    results = evaluate(model, graph, data, device=args.device,
                       eval_split=args.eval_split, val_frac=args.val_frac,
                       split_seed=args.split_seed)

    if results:
        results["kge"] = kge
        print(f"  Papers evaluated: {results['n_papers_evaluated']}")
        print(f"  Hits@1:           {results['hits@1']:.4f}")
        print(f"  Hits@5:           {results['hits@5']:.4f}")
        print(f"  Hits@10:          {results['hits@10']:.4f}")
        print(f"  MRR:              {results['mrr']:.4f}")
        print()
        print("  Per-paper results:")
        for pid, pr in results.get("per_paper", {}).items():
            n_hit = sum(1 for p in pr["top10_predictions"] if p["is_true_citation"])
            print(f"    {pid:15s} rank={pr['best_rank']:3d}  "
                  f"MRR={pr['mrr']:.3f}  "
                  f"true-cites in top10: {n_hit}/{pr['gt_papers_total']}")

        if args.save_results:
            # When running multiple models, suffix the path with the kge name
            res_path = Path(args.save_results)
            if args.kge == "all":
                res_path = res_path.with_name(f"{res_path.stem}_{kge}{res_path.suffix}")
            res_path.parent.mkdir(parents=True, exist_ok=True)
            res_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False)
            )
            print(f"\n  Results saved → {res_path}")
    else:
        print("  No evaluation results (insufficient citation data)")
        print("  Papers with valid citation networks:",
              sorted(data["papers_with_cit"]))

    return results


if __name__ == "__main__":
    main()