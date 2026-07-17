[README.md](https://github.com/user-attachments/files/30121371/README.md)
# SciNex

**End-to-end scientific paper knowledge graph (KG) extraction and evaluation pipeline**

SciNex parses scientific papers into structured HTML, builds citation networks, extracts knowledge graph triples using both open-vocabulary LLM extraction and schema-constrained extraction (Core Experiment Ontology / SciNex ontology), and evaluates the resulting graphs with KG embedding models (TransE, ComplEx, RotatE) against held-out citation edges.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Setup](#setup)
- [Full Workflow (Quick Start)](#full-workflow-quick-start)
- [Stage 1 — Acquire Papers](#stage-1--acquire-papers)
- [Stage 2 — Parse PDFs + Build Citation Networks](#stage-2--parse-pdfs--build-citation-networks)
- [Stage 3 — Enrich Entity CSVs](#stage-3--enrich-entity-csvs)
- [Stage 4 — Extract Knowledge Graphs](#stage-4--extract-knowledge-graphs)
- [Stage 5 — Evaluate Extraction Quality](#stage-5--evaluate-extraction-quality)
- [Stage 6 — Citation Fusion](#stage-6--citation-fusion)
- [Stage 7 — Train & Evaluate KG Embeddings](#stage-7--train--evaluate-kg-embeddings)
- [Output Directory Reference](#output-directory-reference)
- [Common Gotchas](#common-gotchas)

---

## Pipeline Overview

```
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ 1. ACQUIRE PAPERS    │────▶│ 2. PARSE + CITATIONS  │────▶│ 3. ENRICH ENTITIES     │
│ acl_pipeline.py      │     │ main.py               │     │ enrich_entity_csv.py   │
│ citation_expand_     │     │ run_acl_batch.py      │     │                        │
│   pipeline.py         │     │ (batch wrapper)        │     │                        │
└─────────────────────┘     └──────────────────────┘     └───────────────────────┘
                                                                        │
                                                                        ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ 6. CITATION FUSION   │◀────│ 5. EVALUATE QUALITY   │◀────│ 4. EXTRACT KG TRIPLES  │
│ kg_citation_fusion.py│     │ kg_evaluate.py        │     │ kg_main.py             │
└─────────────────────┘     └──────────────────────┘     └───────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│ 7. TRAIN & EVALUATE KG EMBEDDINGS    │
│ kg_transe_pipeline.py / kge_multiseed.py │
│ (TransE / ComplEx / RotatE → MRR, Hits@k) │
└─────────────────────────────────────┘
```

Two extraction strategies are supported at Stage 4:
- **Open (schema-free):** LLM extracts triples with no vocabulary constraints (`--extractor llm`)
- **Schema-guided:** subjects constrained to the CS-NER entity gazetteer, predicates constrained to either the Core Experiment Ontology (`--extractor fixed`) or a refined SciNex ontology variant (`--extractor fixed --ontology scinex`)

---

## Setup

```bash
git clone <this-repo>
cd scinex

pip install -r requirements_kg.txt --break-system-packages
# ITER (SciERC baseline) is installed separately:
pip install git+https://github.com/fleonce/iter --break-system-packages
```

**PowerShell users:** every command below works unchanged — this is a pure-Python CLI pipeline, no bash-only syntax involved except where noted (batch loops).

**GPU note:** default LLM model is `Qwen/Qwen3-32B` / `Qwen3-14B`, sized for a 20GB+ VRAM GPU (e.g. JAIST HAKUSAN A100). For local 8GB-VRAM machines, pass a smaller model explicitly:
```bash
--model Qwen/Qwen2.5-7B-Instruct
# or, for very constrained setups:
--model microsoft/phi-3-mini-4k-instruct
```

---

## Full Workflow (Quick Start)

The shortest path from zero to an evaluated KG, in order:

```bash
# One-time: build the local ACL title→ID index
python acl_pipeline.py --build-index

# 1. Acquire papers (from CS-NER IOB dataset)
python acl_pipeline.py --all --out-dir output/acl --limit 500

# 2. Parse PDFs → HTML + citation networks (skips papers already done)
python run_acl_batch.py --pdf-dir output/acl/pdfs --out-dir output

# 3. Enrich entity CSVs (CS-NER gazetteer, non-circular)
python enrich_entity_csv.py --all --source csner --out-dir output/acl

# 4. Extract KG triples — both strategies
python kg_main.py --all --extractor llm --model Qwen/Qwen2.5-7B-Instruct --skip-existing
python kg_main.py --all --extractor fixed --skip-existing
python kg_main.py --all --extractor fixed --ontology scinex --skip-existing

# 5. Evaluate extraction quality (LLM-as-judge)
python kg_evaluate.py --all --extractor all

# 6. Fuse each paper's KG with its citation network
python kg_citation_fusion.py --paper <paper_id> --extractor fixed/Qwen2.5-7B-Instruct

# 7. Train + evaluate KG embeddings
python kge_multiseed.py --extractor fixed --model Qwen2.5-7B-Instruct --kge all --seeds 1 2 3 4 5 --epochs 500 --device cuda
```

---

## Stage 1 — Acquire Papers

Two ways to grow your paper set.

### 1a. From the CS-NER IOB dataset (primary source)

`acl_pipeline.py` fetches CS-NER's IOB-annotated ACL papers, resolves each title to an ACL Anthology ID (local index → DBLP → Semantic Scholar), downloads the PDF, and writes an entity CSV per paper (used later by the `fixed` extractor).

```bash
# Build the local title index once — takes a few minutes, do this first
python acl_pipeline.py --build-index

# Full pipeline, capped at 500 papers
python acl_pipeline.py --all --out-dir output/acl --limit 500

# Or run stages individually:
python acl_pipeline.py --parse-iob     --out-dir output/acl
python acl_pipeline.py --resolve-ids   --out-dir output/acl
python acl_pipeline.py --download-pdfs --out-dir output/acl
```

| Flag | Purpose |
|---|---|
| `--limit N` | Cap total papers pulled from the IOB dataset |
| `--out-dir` | Where `papers.json` + `pdfs/` + entity CSVs go (default `output/acl`) |
| `--delay` | Seconds between HTTP requests (default 1.0) |

### 1b. From the citation graph (expansion beyond CS-NER)

`citation_expand_pipeline.py` walks the citation networks of papers you've already parsed (Stage 2) and pulls in new papers that are (a) linked by citation and (b) confirmed to exist on ACL Anthology. It processes one seed paper's citation network at a time; once a seed is exhausted it automatically moves to the next.

```bash
python citation_expand_pipeline.py --limit 800
```

| Flag | Purpose |
|---|---|
| `--limit N` | **Required.** Stop once total PDFs on disk reaches N |
| `--seeds ID1,ID2` | Start from specific papers instead of everything already downloaded |
| `--no-recurse` | Only expand one hop from the original seeds (don't snowball into newly-pulled papers) |
| `--pdf-dir` / `--out-dir` / `--papers-dir` | Override default paths |

> Requires citation networks to already exist for the seed papers — run Stage 2 first.

---

## Stage 2 — Parse PDFs + Build Citation Networks

### Single paper

```bash
python main.py output/acl/pdfs/P19-1028.pdf --no-llm --citation-limit 10
```

| Mode | LLM | Citations | Speed |
|---|---|---|---|
| (default) | ✅ | ✅ | Slowest, needs GPU |
| `--no-llm` | ❌ | ✅ | **Recommended** — fast, still gets citation network |
| `--fast` | ❌ | ❌ | Fastest, no citation data |
| `--no-citations` | ✅ | ❌ | Rare — LLM parse without citation lookup |

Can also run directly against SciClaimEval (auto-downloads from HuggingFace):
```bash
python main.py --paper 1810.04805 1706.03762   # by arXiv ID
python main.py --domain NLP --limit 5
python main.py --all
```

### Batch — all PDFs, skipping ones already parsed

```bash
python run_acl_batch.py --pdf-dir output/acl/pdfs --out-dir output
```

Automatically skips any paper that already has both `output/<id>/no-llm/output.html` and `output/<id>/citation_network.json`. Safe to re-run; persists progress after every paper so an interruption doesn't lose work.

```bash
python run_acl_batch.py --rerun-failed          # retry only previously-failed papers
python run_acl_batch.py --paper C16-1036        # single paper by ID
python run_acl_batch.py --timeout 1800          # raise per-paper timeout (citation rate-limit backoff can be slow)
```

**PowerShell one-liner equivalent** (if you need to call `main.py` directly per-PDF instead):
```powershell
Get-ChildItem output/acl/pdfs/*.pdf | ForEach-Object {
    $outfile = "output/$($_.BaseName)/no-llm/output.html"
    if (!(Test-Path $outfile)) {
        python main.py $_.FullName --no-llm --citation-limit 10
    }
}
```

---

## Stage 3 — Enrich Entity CSVs

The CS-NER IOB annotations only cover paper *titles*. This step expands each paper's entity CSV with entities found in the *body text*, using an external (non-circular) source so the `fixed` extractor isn't evaluated against its own extractions.

```bash
python enrich_entity_csv.py --all --source csner --out-dir output/acl
```

| `--source` | Description |
|---|---|
| `csner` (default) | Intersects the global CS-NER gazetteer with each paper's body text — external, non-circular |
| `iter` | Runs the ITER/SciERC model over the parsed HTML |
| `llm` | Harvests entities from your own LLM triples — **circular, retired**, avoid for evaluation |

```bash
python enrich_entity_csv.py --paper D17-1028 --source csner   # single paper
python enrich_entity_csv.py --all --source csner --rebuild-gazetteer  # force gazetteer re-download
```

Output: `Entity_<id>_enriched.csv` alongside the original `Entity_<id>.csv` — the `fixed` extractor auto-prefers the enriched version.

---

## Stage 4 — Extract Knowledge Graphs

Core extraction step. `--paper <id>` for one paper, `--all --limit N` for a batch.

```bash
# Single paper, all extractors
python kg_main.py --paper 2205.11361 --extractor all

# Batch, resumable
python kg_main.py --all --extractor llm   --model Qwen/Qwen2.5-7B-Instruct --skip-existing --max-new-tokens 2048
python kg_main.py --all --extractor fixed --skip-existing
python kg_main.py --all --extractor fixed --ontology scinex --ontology-file scinex_refined_14.owl --skip-existing
```

| Extractor | Flag | Notes |
|---|---|---|
| REBEL baseline | `--extractor rebel` | General-domain relation extraction baseline |
| ITER / SciERC baseline | `--extractor iter` | Science-specific baseline |
| Open LLM | `--extractor llm` | Schema-free, open-vocabulary |
| Schema-guided (CEO) | `--extractor fixed` | Entities from gazetteer, predicates from Core Experiment Ontology |
| Schema-guided (SciNex) | `--extractor fixed --ontology scinex` | Same entities, refined predicate set — writes to `kg/fixed_scinex/` |
| All | `--extractor all` | Runs every extractor above |

Key flags:

| Flag | Purpose |
|---|---|
| `--skip-existing` | Resume — skip papers that already have `triples.json` for this extractor+model |
| `--model` | HF model ID for the LLM extractor (default `Qwen/Qwen3-32B`; use a 7B/mini model for ≤8GB VRAM) |
| `--limit N` | With `--all`, process only first N papers |
| `--no-gpu` | Force CPU |
| `--no-viz` | Skip the interactive HTML visualization (faster) |
| `--entity-csv` | Override auto-resolved entity CSV path for the `fixed` extractor |

Output per paper: `output/<id>/kg/<extractor>/[<model>/]{triples.json, kg.graphml, kg_stats.json, kg_viz.html}`

---

## Stage 5 — Evaluate Extraction Quality

LLM-as-judge scoring of extracted triples.

```bash
python kg_evaluate.py --all --extractor all
python kg_evaluate.py --paper 2205.11361 --extractor fixed llm --entity-csv output/acl/2205.11361/Entity_2205.11361_enriched.csv
```

Passing `--entity-csv` tells the judge about known abbreviations so it doesn't penalize a triple for using "BERT" instead of the full model name.

---

## Stage 6 — Citation Fusion

Merges a paper's extracted KG with its citation network into one unified graph (paper–paper edges + entity–entity edges).

```bash
python kg_citation_fusion.py --paper 1810.04805 --extractor fixed/Qwen2.5-7B-Instruct
python kg_citation_fusion.py --paper 1810.04805 --extractor llm/Qwen2.5-7B-Instruct
python kg_citation_fusion.py --paper 1810.04805 --extractor rebel
```

---

## Stage 7 — Train & Evaluate KG Embeddings

Trains a KG embedding model on the unified graph (entity–entity + paper–entity + paper–paper citation edges) and evaluates predicted related-papers against real citation links.

```bash
# Single run
python kg_transe_pipeline.py --output-dir output --extractor fixed --model Qwen2.5-7B-Instruct --kge rotate --epochs 500 --device cuda

# Compare all three embedding models in one run
python kg_transe_pipeline.py --extractor fixed --model Qwen2.5-7B-Instruct --kge all --epochs 500
```

**Recommended: multi-seed for a defensible result.** A single KGE run varies run-to-run (random init + negative sampling); this reports mean ± std over several fixed, reproducible seeds.

```bash
python kge_multiseed.py --extractor fixed --model Qwen2.5-7B-Instruct \
    --kge all --seeds 1 2 3 4 5 --epochs 500 --device cuda
```

| KGE model | Captures |
|---|---|
| `transe` | Translation: h + r ≈ t |
| `complex` | Complex bilinear — handles asymmetric relations |
| `rotate` | Relation as rotation in complex space — best empirical result in our runs (self-adversarial negative sampling, MRR ~0.60, Hits@10 ~0.85) |

---

## Output Directory Reference

```
output/
├── acl/
│   ├── papers.json                          ← paper index (title, ACL ID, entities)
│   ├── pdfs/<paper_id>.pdf
│   ├── csner_gazetteer.csv                  ← cached CS-NER gazetteer
│   └── <paper_id>/
│       ├── Entity_<paper_id>.csv            ← title-only entities (from IOB)
│       └── Entity_<paper_id>_enriched.csv   ← + body entities (Stage 3)
│
└── <paper_id>/
    ├── citation_network.json                ← shared across all parse modes
    ├── no-llm/output.html                   ← parsed structure (Stage 2)
    ├── default/output.html
    └── kg/
        ├── rebel/{triples.json, kg.graphml, kg_stats.json, kg_viz.html}
        ├── iter/{...}
        ├── llm/<model>/{...}
        ├── fixed/{...}                      ← CEO ontology
        └── fixed_scinex/{...}               ← SciNex ontology
```

---

## Common Gotchas

- **Forgetting `--entity-csv`** on manual/evaluation commands silently disables alias resolution for the `fixed` extractor. Prefer letting it auto-resolve (omit the flag) unless you need to override.
- **`--skip-existing` is model-aware** — switching `--model` on the `llm` extractor will *not* skip papers processed under a different model; each model gets its own subdirectory.
- **BIOES, not BIO** — CS-NER IOB files use `B/I/E/S/O` tags; `acl_pipeline.py`'s IOB parser accounts for this.
- **Citation networks are required before `citation_expand_pipeline.py` can expand from a seed** — that seed must have gone through Stage 2 (`main.py` without `--no-citations`/`--fast`) first.
- **`enrich_entity_csv.py --source llm`** is circular (evaluating LLM triples against LLM-derived entities) — only use `csner` or `iter` for anything you intend to report as evaluation.
- **8GB VRAM machines** should always pass `--model Qwen/Qwen2.5-7B-Instruct` or smaller to `kg_main.py --extractor llm`; the default `Qwen3-32B`/`Qwen3-14B` will OOM.
- **PowerShell vs bash** — every script here is pure argparse Python, so commands are identical across shells; only ad-hoc batch loops (e.g. the PowerShell one-liner in Stage 2) need shell-specific syntax.

---

## Acknowledgments

Developed by Sam under the supervision of Professor Nathawut at JAIST, for the SciClaimEval shared task. CS-NER gazetteer and ACL entity annotations from the [CS-NER](https://github.com/jd-coderepos/contributions-ner-cs) project. Core Experiment Ontology (CEO) maintained at [wpatipon/core-experiment-ontology](https://github.com/wpatipon/core-experiment-ontology).
