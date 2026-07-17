# Scientific-Paper Knowledge Graph Pipeline

Turns scientific-paper PDFs into a queryable knowledge graph, then tests whether
that graph actually captures something real: can it predict a paper's true
citation neighbours from nothing but its extracted content?

Three stages, each usable standalone:

1. **PDF → HTML Parser** — clean structured text + tables from raw PDFs
2. **KG Extraction** — LLM-extracted `(subject, predicate, object)` triples,
   constrained to an ontology, folded into one persistent cross-paper graph
3. **KG Embedding Evaluation** — train TransE/ComplEx/RotatE on the graph,
   test whether it ranks a paper's real citations above random papers

---

## ⭐ Headline result

**RotatE + self-adversarial negative sampling predicts held-out citation links
with MRR ≈ 0.60 and Hits@10 ≈ 0.85** (5 seeds, hyperparameters selected on a
disjoint validation split, reported on held-out test papers). Random baseline
≈ 0.2%. Full methodology, caveats, and reproduction commands in
[`results.md`](results.md).

---

## Quickstart

```bash
pip install -r requirement.txt

# 1. Parse a PDF → structured HTML
python3 main.py paper.pdf --no-llm

# 2. Extract a knowledge graph from it (constrained-entity extraction, CEO ontology)
python3 kg_main.py --paper paper --extractor fixed --model Qwen/Qwen3-14B

# 3. Evaluate KG quality via citation-link prediction
python3 kg_transe_pipeline.py --output-dir output --extractor fixed --model Qwen3-14B --kge all
```

`--all` in place of `--paper <name>` runs any stage over the whole corpus;
`--skip-existing` resumes an interrupted batch without re-running the LLM.

---

## 1. PDF → HTML Parser (`main.py`)

Parses academic PDFs (arXiv, ACL Anthology, PeerJ, …) into clean HTML, with
optional LLM text cleanup and citation-network enrichment via Semantic Scholar.

```bash
python3 main.py paper.pdf --fast    # no GPU, no internet, ~10-30 sec
python3 main.py paper.pdf           # + LLM cleanup + citation fetch, 5-20 min
```

| Bottleneck | Time cost | Skip with |
|---|---|---|
| Qwen2.5-14B LLM text cleanup | 5–15 min | `--no-llm` / `--fast` |
| Semantic Scholar citation fetch | 1–5 min | `--no-citations` / `--fast` |
| Table extraction | 10–30 sec | always runs |

Output: `output/<paper>/no-llm/output.html` (or `output.html` in full mode) +
`citation_network.json`. Citation fetches are cached — re-running the same
paper reads instantly.

---

## 2. KG Extraction (`kg_main.py`)

Extracts `(subject, predicate, object)` triples from the parsed HTML. Five
extractors, selected with `--extractor`:

| Extractor | Model | Subjects | Relations |
|---|---|---|---|
| `rebel` | `Babelscape/rebel-large` | open | open |
| `iter` | `fleonce/iter-scierc-deberta-large` | open | SciERC-typed |
| `llm` | Qwen3-14B | open | open vocabulary |
| **`fixed`** | Qwen3-14B / Gemma3-12B | entity CSV | CEO ontology — **main extractor** |
| `pair` | Qwen3-14B | entity CSV | entity CSV, relation free |

```bash
python3 kg_main.py --paper 2020.acl-main.130 --extractor fixed --model Qwen/Qwen3-14B
python3 kg_main.py --all --extractor fixed --model google/gemma-3-12b-it --skip-existing
```

`--entity-csv` is optional — auto-resolved per paper from CS-NER-derived
entity lists (`enrich_entity_csv.py`) when omitted.

**Models validated for `fixed` extraction:** Qwen3-14B (original, ~7–8GB
4-bit) and **Gemma3-12B** (`google/gemma-3-12b-it`, gated — requires
`HF_TOKEN` in `.env`; multimodal checkpoint, loaded text-only). Both produce
correctly-grounded triples and merge cleanly into the global KG — see
[`Claude.md`](Claude.md) for the swap-in notes and per-model comparison.

### Persistent global KG

Every `fixed`/`fixed_scinex` extraction also folds into **one incrementally-merged
cross-paper graph** (not just each paper's isolated `triples.json`): each paper
is a big node (`paper:<id>`), its entities hang off it as augmented nodes via
`mentions` edges, and real `cites` edges connect papers to each other directly.

```
output/global_kg/<extractor>/<model>/graph.graphml   # the merged graph
output/global_kg/<extractor>/<model>/meta.json        # merge bookkeeping
```

Handles node identity across time — a citation to a paper not yet in the
corpus gets a `paper_stub` node that resolves into the real node once that
paper is actually extracted, preserving any edges already attached.

---

## 3. KG Embedding Evaluation (`kg_transe_pipeline.py`)

Trains a KG embedding model on the extracted graph, then tests whether a
paper's learned embedding ranks its **real citation neighbours** above other
papers — citation edges are held out of training and used only as ground
truth.

```bash
python3 kg_transe_pipeline.py --output-dir output --extractor fixed --model Qwen3-14B --kge all
# multi-seed (report mean ± std):
python3 kge_multiseed.py --extractor fixed --model Qwen3-14B --kge all --seeds 1 2 3 4 5 --epochs 1000
```

Three models (`--kge {transe,complex,rotate,all}`), two objectives (margin
vs. self-adversarial), two ontologies (CEO / scinex — statistically tied on
this metric), reported as Hits@1/5/10 + MRR. Full protocol and honest caveats
in [`results.md`](results.md).

---

## Corpus

800+ papers currently parsed and entity-enriched (scaling toward 2000+),
fetched by walking ACL Anthology citation networks
(`citation_expand_pipeline.py`) from an 83k-title local index
(`acl_index.py`). Entity subject lists are built non-circularly from the
**CS-NER** human-annotated gazetteer intersected with each paper's text — not
back-filled from the pipeline's own extraction.

---

## Project structure

```
main.py                     — PDF→HTML parser CLI
config.py                   — OUTPUT_DIR, CITATION_LIMIT
compare.py                  — HTML output similarity scorer
rebuild_html.py              — offline table-fix tool (standalone)

parser/
    pdf_loader.py            — PDF loading (PyMuPDF)
    layout.py                — text block extraction (pdfplumber/PyMuPDF)
    table_extractor.py       — multi-strategy table detection
    structure_builder.py     — raw blocks → structured sections
    html_generator.py        — structured data → final HTML
    llm_refiner.py           — optional Qwen2.5-14B text cleanup

citation/
    network.py                — Semantic Scholar citation graph builder
    auto_fetcher.py           — auto-download related papers
    semantic.py                — S2 API helpers
    concept_builder.py        — concept keywords from abstracts

kg_main.py                  — KG extraction CLI (rebel/iter/llm/fixed/pair/all)
kg_extraction/
    fixed_extractor.py        — fixed-subject extractor (CEO ontology; Qwen3-14B / Gemma3-12B)
    llm_extractor.py           — free LLM extractor (open vocabulary)
    extractor.py                — REBEL baseline
    iter_extractor.py           — ITER/SciERC baseline
    entity_loader.py            — entity CSV loading, alias lookup
    kg_builder.py                — NetworkX KG + entity normalisation
    global_graph.py              — persistent cross-paper global KG (incremental merge, stub resolution)
    html_parser.py               — sentence extraction from output.html
    visualizer.py                 — interactive pyvis visualization

kg_transe_pipeline.py       — KG embedding evaluation (TransE/ComplEx/RotatE)
kge_multiseed.py            — multi-seed KGE runner
kg_evaluate.py               — LLM-as-Judge faithfulness eval (retired, kept for reference)
kg_compare.py / kg_fixed_compare.py / kg_citation_fusion.py — comparison/fusion utilities

acl_index.py                — title→ACL-ID index (anthology.json.gz / XML fallback)
acl_pipeline.py              — ACL paper download pipeline (CS-NER → entity CSVs + PDFs)
enrich_entity_csv.py         — per-paper entity lists from the CS-NER gazetteer
run_acl_batch.py              — batch PDF parsing with success/failure tracking
citation_expand_pipeline.py   — snowball paper downloads via citation networks
expand_and_validate.py        — global-KG validation smoke test
paper_registry.py              — SciClaimEval paper ID registry
```

---

## Dependencies

```bash
pip install -r requirement.txt
```

Requires a CUDA-capable GPU (≥16GB VRAM) for LLM extraction/refinement.
`requirement.txt` pins PyTorch for specific CUDA toolkits — check the comment
at the top of the file if installing on new hardware (works out of the box on
CUDA 12.x/13.x, including RTX 50-series).

Gated models (e.g. `google/gemma-3-12b-it`) need a Hugging Face token —
accept the license on the model page, then add `HF_TOKEN=hf_...` to `.env`
(same file as `SEMANTIC_SCHOLAR_API_KEY`).

---

## Further reading

- [`Claude.md`](Claude.md) — architecture, model notes, GPU quirks, current status
- [`results.md`](results.md) — consolidated paper-ready results and caveats
- [`hands_off.md`](hands_off.md) — dated session-by-session change log
