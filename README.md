# Scientific-Paper Knowledge Graph Pipeline

Turns scientific-paper PDFs into a queryable knowledge graph, then tests whether
that graph actually captures something real: can it predict a paper's true
citation neighbours from nothing but its extracted content?

```
PDF → HTML → entity lists → KG triples → persistent global KG → KGE link prediction
```

---

## ⭐ Headline result

**RotatE + self-adversarial negative sampling predicts held-out citation links
with MRR ≈ 0.60 and Hits@10 ≈ 0.85** (5 seeds, hyperparameters selected on a
disjoint validation split, reported on held-out test papers). Random baseline
≈ 0.2%. Full methodology, caveats, and reproduction commands in
[`results.md`](results.md).

---

## Run the pipeline end-to-end

Follow the steps **in order** — each one consumes the previous step's output.

### Step 0 — Setup (once)

```bash
pip install -r requirement.txt
```

Create a `.env` file in the project root:

```bash
SEMANTIC_SCHOLAR_API_KEY=your_s2_key    # citation networks (free: api.semanticscholar.org)
HF_TOKEN=hf_...                         # only needed for gated models (e.g. Gemma3)
```

You need a CUDA GPU with ≥16GB VRAM for the extraction steps (4 and beyond).
Steps 1–3 run on CPU.

### Step 1 — Get PDFs

**Option A — you already have a PDF:** skip to Step 2.

**Option B — build an ACL Anthology corpus** (what the reported results use):

```bash
# 1a. Build the local title→ACL-ID index (one-time, ~83k entries)
python3 -c "from acl_index import ACLIndex; ACLIndex('acl_title_index.json').build(save=True)"

# 1b. Fetch seed papers: CS-NER annotations → entity CSVs + PDF downloads
python3 acl_pipeline.py --all

# 1c. (Optional) grow the corpus by walking citation networks of papers you already have
python3 citation_expand_pipeline.py --limit 500        # stop at 500 PDFs on disk
```

PDFs land in `output/acl/pdfs/`.

### Step 2 — Parse PDFs → structured HTML

```bash
# Single paper
python3 main.py paper.pdf --no-llm

# Whole downloaded corpus (tracks success/failure in batch_run_log.json)
python3 run_acl_batch.py
```

This also fetches each paper's **citation network** from Semantic Scholar
(cached; needs the S2 key from Step 0). Output per paper:

```
output/<paper_id>/no-llm/output.html      ← structured text, the KG input
output/<paper_id>/citation_network.json   ← cites/cited-by graph, the eval ground truth
```

Flags: `--fast` (skip LLM+citations), `--no-citations`, `--citation-limit N`.

### Step 3 — Build entity lists (subjects for extraction)

The `fixed` extractor only accepts subjects from a per-paper entity list,
built **non-circularly** from the CS-NER human-annotated gazetteer
intersected with each paper's text:

```bash
python3 enrich_entity_csv.py --all --source csner
# or one paper:
python3 enrich_entity_csv.py --paper 2020.acl-main.130 --source csner
```

Output: `output/acl/<paper_id>/Entity_<paper_id>_enriched.csv`
(~70–150 entities per paper). First run also builds the global gazetteer
cache (`output/acl/csner_gazetteer.csv`, ~51k entities).

### Step 4 — Extract the knowledge graph (GPU)

```bash
# One paper
python3 kg_main.py --paper 2020.acl-main.130 --extractor fixed --model Qwen/Qwen3-14B

# Whole corpus, resumable (skips papers already extracted)
python3 kg_main.py --all --extractor fixed --model google/gemma-3-12b-it --skip-existing
```

The entity CSV from Step 3 is auto-resolved per paper — no flag needed.
Validated models: **Qwen3-14B** and **Gemma3-12B** (`google/gemma-3-12b-it`,
gated → needs `HF_TOKEN`). Both are loaded 4-bit NF4; budget ~5–10 min/paper.

Output per paper: `output/<paper_id>/kg/fixed/<model>/triples.json` + `kg.graphml`
+ an interactive `kg_viz.html`.

**The persistent global KG is built automatically during this step** — every
extracted paper is folded into one cross-paper graph at
`output/global_kg/fixed/<model>/graph.graphml`: papers are big nodes, their
entities hang off them via `mentions` edges, and real `cites` edges connect
papers directly. Citations to not-yet-extracted papers become stub nodes that
resolve automatically once those papers are extracted.

> Note: the global graph is saved every 25 papers and on clean exit — don't
> `kill -9` a batch mid-run (per-paper `triples.json` files are always safe;
> the merge can be replayed from them if needed).

Other extractors for comparison baselines: `--extractor rebel` (open
relations), `iter` (SciERC-typed), `llm` (free LLM vocabulary), `pair`
(both ends constrained). `--ontology scinex` swaps the CEO predicate set for
the scinex ontology.

### Step 5 — KG embedding evaluation / link prediction (GPU)

Trains a KG embedding on the extracted triples + paper–entity links, with
**citation edges held out**, then ranks candidate papers per query paper and
checks whether the true citation neighbours come out on top:

```bash
# Single run, all three KGE models (TransE / ComplEx / RotatE)
python3 kg_transe_pipeline.py --output-dir output --extractor fixed --model Qwen3-14B --kge all

# Multi-seed with honest val/test split (the headline configuration)
python3 kge_multiseed.py --extractor fixed --model Qwen3-14B --kge rotate \
  --seeds 1 2 3 4 5 --epochs 1000 --loss adv --gamma 28 \
  --eval-split test --split-seed 42 --val-frac 0.5
```

Metrics: Hits@1/5/10 + MRR, both bidirectional and temporal-filtered
("papers it could actually have cited"). Per-paper diagnostics
(`top10_predictions`, `gt_ranks`) are written alongside the summary JSONs in
`output/`. Full protocol and caveats: [`results.md`](results.md).

---

## Corpus

800+ papers currently parsed and entity-enriched (scaling toward 2000+).
Entity subject lists come from external human annotation (CS-NER), never from
the pipeline's own extraction — no circularity.

---

## Project structure

```
main.py                     — Step 2: PDF→HTML parser CLI
config.py                   — OUTPUT_DIR, CITATION_LIMIT
compare.py                  — HTML output similarity scorer
rebuild_html.py             — offline table-fix tool (standalone)

parser/
    pdf_loader.py           — PDF loading (PyMuPDF)
    layout.py               — text block extraction (pdfplumber/PyMuPDF)
    table_extractor.py      — multi-strategy table detection
    structure_builder.py    — raw blocks → structured sections
    html_generator.py       — structured data → final HTML
    llm_refiner.py          — optional Qwen2.5-14B text cleanup

citation/
    network.py              — Semantic Scholar citation graph builder
    auto_fetcher.py         — auto-download related papers
    semantic.py             — S2 API helpers
    concept_builder.py      — concept keywords from abstracts

acl_index.py                — Step 1: title→ACL-ID index (83k entries)
acl_pipeline.py             — Step 1: CS-NER → entity CSVs + PDF downloads
citation_expand_pipeline.py — Step 1: snowball corpus via citation networks
run_acl_batch.py            — Step 2: batch PDF parsing with success tracking
enrich_entity_csv.py        — Step 3: per-paper entity lists from CS-NER gazetteer

kg_main.py                  — Step 4: KG extraction CLI (rebel/iter/llm/fixed/pair)
kg_extraction/
    fixed_extractor.py      — fixed-subject extractor (CEO ontology; Qwen3-14B / Gemma3-12B)
    llm_extractor.py        — free LLM extractor (open vocabulary)
    extractor.py            — REBEL baseline
    iter_extractor.py       — ITER/SciERC baseline
    entity_loader.py        — entity CSV loading, alias lookup
    kg_builder.py           — NetworkX KG + entity normalisation
    global_graph.py         — persistent cross-paper global KG (incremental merge)
    html_parser.py          — sentence extraction from output.html
    visualizer.py           — interactive pyvis visualization

kg_transe_pipeline.py       — Step 5: KGE evaluation (TransE/ComplEx/RotatE)
kge_multiseed.py            — Step 5: multi-seed KGE runner

expand_and_validate.py      — end-to-end smoke test (fetch → parse → enrich → extract → merge check)
kg_evaluate.py              — LLM-as-Judge faithfulness eval (retired, kept for reference)
kg_compare.py / kg_fixed_compare.py / kg_citation_fusion.py — comparison/fusion utilities
paper_registry.py           — SciClaimEval paper ID registry
```

---

## Dependencies

```bash
pip install -r requirement.txt
```

Works out of the box on CUDA 12.x/13.x, including RTX 50-series — see the
comment at the top of `requirement.txt` if installing on other hardware.

---

## Further reading

- [`Claude.md`](Claude.md) — architecture, model notes, GPU quirks, current status
- [`results.md`](results.md) — consolidated paper-ready results and caveats
- [`hands_off.md`](hands_off.md) — dated session-by-session change log
