# CLAUDE.md — Project Context for Claude Code

## Project: Scientific Paper Knowledge Graph Pipeline
**Running on:** Private VM (Ubuntu, CUDA GPU)
**Working directory:** `/home/ubuntu/project_clean_9/`
**User:** Sam (graduate student/researcher)

---

## ⚠ DOCUMENTATION DISCIPLINE — READ FIRST (standing user instruction)
**After ANY meaningful change — new result, code change, decision, fix, new run — UPDATE THE MARKDOWN DOCS in the SAME session, before finishing.** The user relies on these files to hand off to the next session and to feed the paper writer; out-of-date docs break that. This is a standing instruction, no need to re-ask.
- **`hands_off.md`** — append a dated session note (what changed, why, exact numbers, file paths, next steps). The running log.
- **`Claude.md`** — update the STATUS block + any architecture/convention that changed. The persistent overview.
- **`results.md`** — keep the consolidated paper-ready tables/findings in sync whenever a result changes. Hand THIS to the paper/abstract writer.
- Use exact numbers pulled from the `output/*.json` files (don't transcribe from memory). Convert relative dates to absolute. Note which result files on disk back each claim.

---

## Architecture

### Part 1 — PDF Parser (main.py)
Converts scientific paper PDFs into structured HTML.
- Input: PDF file (from arXiv, ACL Anthology, PeerJ)
- Process: pdfplumber/PyMuPDF → text blocks → structure_builder → html_generator
- Optional: LLM refinement (Qwen2.5-14B), citation network via Semantic Scholar
- Output: `output/{paper_id}/no-llm/output.html` + `citation_network.json`
- Run: `python main.py paper.pdf --no-llm`

### Part 2 — KG Extraction Pipeline (kg_main.py)
Extracts (subject, predicate, object) triples from Part 1 HTML output.
- Four extractors:
  - **REBEL** (`kg_extraction/extractor.py`) — `Babelscape/rebel-large` baseline, open relations
  - **ITER** (`kg_extraction/iter_extractor.py`) — `fleonce/iter-scierc-deberta-large`, SciERC typed relations (Used-for, Feature-of, Hyponym-of, Part-of, Compare, Conjunction)
  - **Free LLM** (`kg_extraction/llm_extractor.py`) — Qwen3-14B, open subjects/objects, free relation vocabulary
  - **Fixed LLM** (`kg_extraction/fixed_extractor.py`) — Qwen3-14B, subjects constrained to entity CSV, CEO ontology predicates — **best quality, main focus**
  - **Pair LLM** (`kg_extraction/pair_extractor.py`) — Qwen3-14B, subject AND object both constrained to the entity CSV, relation **free-form** (LLM chooses a short verb phrase). Closed-entity / open-relation. `--extractor pair`
- Output: `output/{paper}/kg/{extractor}/{model}/triples.json`
- Run: `python3 kg_main.py --paper BERT --extractor fixed --entity-csv Entity-BERTv2.csv --model Qwen/Qwen3-14B`
- **`--entity-csv` is optional**: if omitted, the entity CSV is auto-resolved per paper from its id (enriched ACL CSV → title-only ACL CSV → project-root manual CSV). So `python3 kg_main.py --paper C16-1036 --extractor fixed --model Qwen/Qwen3-14B` just works, and `--all` resolves a different CSV per paper (papers with no CSV skip fixed extraction).

### Part 3 — KG Embedding Evaluation (kg_transe_pipeline.py) — PRIMARY triple evaluation
This is now the **main way to evaluate triples** (the `kg_evaluate.py` LLM-as-Judge faithfulness eval is retired — files kept, not used going forward).
Evaluates KG quality by training a KG embedding model on extracted triples + paper-entity links, then testing if paper embeddings predict actual citation links.
- Three KGE models via `--kge {transe,complex,rotate,all}`:
  - **TransE** (Bordes 2013) — translation `h + r ≈ t`
  - **ComplEx** (Trouillon 2016) — complex bilinear, handles asymmetric relations
  - **RotatE** (Sun 2019) — relation as rotation in complex space (symmetry/inversion/composition)
  - `all` — trains+evaluates all three and prints a Hits@k/MRR comparison table
- All three share one training loop (margin-ranking loss, negative sampling) and one model-agnostic predict/evaluate path (cosine sim on `entity_emb`); add new models by registering an nn.Module with `entity_emb`/`relation_emb`/`forward`/`loss` in `KGE_MODELS`
- Citation edges held out from training (used only as evaluation ground truth)
- Training data: entity→predicate→entity + paper→mentions→entity + paper→mentions→concept (concepts include BOTH cited-paper abstract concepts AND the seed paper's own `root_concepts` — the latter added Session 17 to fix the query/candidate vocabulary mismatch)
- Metrics: Hits@1, Hits@5, Hits@10, MRR. Per-paper diagnostic output: `top10_predictions`, `gt_ranks`.
- Defaults (Session 17): `--dim 128`, `--epochs 1000`, `--neg-ratio 10` (negatives per positive)
- Run: `python3 kg_transe_pipeline.py --output-dir output --extractor fixed --model Qwen3-14B --kge all`
- Multi-seed (report mean±std): `python3 kge_multiseed.py --extractor fixed --model Qwen3-14B --kge all --seeds 1 2 3 4 5 --epochs 1000 --device cuda`

---

## CEO Ontology (Core Experiment Ontology)
From colleague's repo: `https://github.com/wpatipon/core-experiment-ontology`

Predicates used in fixed extractor: `cites`, `publishedIn`, `writtenBy`, `reports`, `affiliatedWith`, `employs`, `locatedIn`, `addresses`, `motivates`, `achieves`, `encompasses`, `comprises`, `uses`, `produces`, `trainedOn`, `evaluatedOn`, `splitFrom`, `designedFor`, `comparesAgainst`, `configures`, `evaluates`, `supports`

Each predicate has strict domain/range constraints enforced in both the LLM prompt and post-parse code-level validation in `fixed_extractor.py`.

**Triple shape (fixed extractor):** subject = from the entity CSV, predicate = CEO ontology relation, **object = free text** (not constrained to the list). The system prompt is **paper-agnostic** (placeholders like `<Model>`/`<Dataset>`, not BERT-specific examples) so it works across diverse ACL papers, not just BERT.

---

## Key Files

```
main.py                     — PDF→HTML parser CLI
config.py                   — OUTPUT_DIR, CITATION_LIMIT
rebuild_html.py             — offline table-fix tool (standalone)
compare.py                  — HTML output similarity scorer (cosine/edit/structure)
parser/
    structure_builder.py    — converts raw PDF blocks to structured sections
    html_generator.py       — builds final HTML from structured data
    text_cleaner.py         — ftfy + regex text cleaning
    layout.py               — PDF block extraction (pdfplumber/PyMuPDF)
    llm_refiner.py          — optional LLM text refinement (Qwen2.5-14B)
citation/
    network.py              — Semantic Scholar citation graph builder
    semantic.py             — S2 API helpers
kg_main.py                  — KG extraction CLI (rebel/iter/llm/fixed/all)
kg_extraction/
    fixed_extractor.py      — Fixed-subject extractor (CEO ontology, constrained entities)
    llm_extractor.py        — Free LLM extractor (open vocabulary)
    extractor.py            — REBEL baseline
    iter_extractor.py       — ITER/SciERC extractor (fleonce/iter-scierc-deberta-large)
    entity_loader.py        — loads entity CSV, builds alias lookup
    kg_builder.py           — NetworkX KG + entity normalisation → triples.json + kg.graphml
    html_parser.py          — extracts sentences from output.html for extraction
    visualizer.py           — interactive pyvis HTML visualization
kg_evaluate.py              — LLM-as-Judge evaluation (source sentence faithfulness)
kg_fixed_compare.py         — compare fixed-subject vs free triples side by side
kg_compare.py               — compare results across all extractors
kg_citation_fusion.py       — merge KG with citation network
kg_transe_pipeline.py       — KG embedding evaluation (TransE/ComplEx/RotatE via --kge)
acl_index.py                — builds local title→ACL paper ID index from anthology.json.gz
acl_pipeline.py             — ACL paper download pipeline (CS-NER IOB → entity CSVs + PDFs)
enrich_entity_csv.py        — enrich title-only ACL CSVs into a real subject pool for fixed extraction (harvest body entities from open-extraction triples, or ITER/SciERC)
run_acl_batch.py            — batch runner for parsing multiple ACL PDFs (tracks success/failure)
paper_registry.py           — SciClaimEval paper ID registry
```

---

## Output Structure
```
output/{paper}/
    no-llm/output.html          — parsed HTML
    citation_network.json       — S2 citation graph (paper nodes + edges)
    kg/
        rebel/triples.json           — REBEL extraction
        iter/triples.json            — ITER/SciERC extraction
        llm/{model}/triples.json     — free LLM extraction
        fixed/{model}/triples.json   — fixed LLM extraction (CEO ontology; subject∈list, relation∈ontology, object free)
        fixed_scinex/{model}/triples.json — fixed LLM extraction using the scinex ontology (--ontology scinex); coexists with CEO fixed/
        fixed/{model}/evaluation.json    — LLM-as-Judge results
        pair/{model}/triples.json    — pair LLM extraction (subject∈list, relation free, object∈list)
        fused/fused_kg.graphml       — merged KG + citation graph
```

---

## Models Used
- **Qwen3-14B** (`Qwen/Qwen3-14B`) — fixed and free extraction (~7–8GB at 4-bit NF4); chosen for the 20GB GPU VRAM limit. Replaced Qwen3-32B (OOMed) and Qwen3-30B-A3B (MoE loads all 30B params into VRAM → occupied ~19.3/19.8GB, leaving no room for KV cache/generation → froze)
- **Qwen2.5-14B** — optional PDF text refinement
- **Qwen2.5-7B-Instruct** — LLM-as-Judge evaluation default; can upgrade to **Qwen2.5-14B-Instruct** (`--model`, ~9GB at 4-bit, fits 20GB) for a stronger, different-family judge. Judge loads 4-bit with `device_map={"":0}` (no CPU offload).
- **REBEL** — `Babelscape/rebel-large` (general baseline)
- **ITER** — `fleonce/iter-scierc-deberta-large` (SciERC science-specific baseline)
- All LLMs quantized with BitsAndBytes 4-bit NF4 + bfloat16 compute

### GPU notes (20GB VRAM)
- This GPU is a **vGPU (H100-20C)**: do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it uses CUDA virtual-memory APIs vGPUs don't support and fails on any allocation (`CUDA driver error: operation not supported`)
- **`nvidia-smi` GPU-utilization is broken on this vGPU — it reads 0% even when the GPU is fully busy.** Do NOT use util% to judge if a run is alive/on-GPU. Use instead: log advancing, VRAM ~10–11GB, SM clock ~1755MHz under load (`--query-gpu=clocks.sm`). "GPU 0% + one CPU core 100%" is normal during decode, NOT proof of CPU execution. (See hands_off.md Session 4.)
- **Use stable torch, never a dev-nightly.** A torch `2.12.0.dev*+cu128` nightly silently crippled bitsandbytes 4-bit decode to ~2 tok/s. Stable `torch 2.6.0+cu124` (via `--index-url https://download.pytorch.org/whl/cu124`) restores ~16 tok/s; cu124 wheels run fine on the 12.8 driver.
- At ~16 tok/s, kg_main's default `--max-new-tokens 4096` (×2 in postprocess) takes several silent minutes per generation — use `--max-new-tokens 512` for visibly-progressing runs. Run ONE extraction process at a time (the GPU holds only one model copy; two → OOM).
- Both extractors load the model with `device_map={"": 0}` (force GPU 0, no silent CPU offload — which manifests as a "freeze"); 14B 4-bit uses ~10GB / 20GB
- Both force greedy decoding (`model.generation_config.do_sample=False`, `top_k/top_p/temperature=None`) — Qwen3 model defaults set `do_sample=True` which must be overridden for deterministic structured output
- 32B and 30B-A3B do NOT fit; 14B is the largest practical Qwen3 dense model here

---

## ACL Paper Pipeline (acl_pipeline.py + acl_index.py)
Downloads and processes ACL Anthology papers for KG extraction:
1. `acl_index.py --build` — builds title→ID index from `anthology.json.gz` (81k entries, one-time)
2. `acl_pipeline.py --all` — fetches CS-NER IOB annotations → entity CSVs + PDF download
3. CS-NER dataset (`jd-coderepos/contributions-ner-cs/acl`) annotates paper titles with 7 entity types (solution, method, dataset, research_problem, tool, resource, language) in BIOES format
4. Entity CSVs (2-3 entities per paper from title) serve as subject list for fixed extractor
5. ID lookup: local index (anthology.json) → DBLP fallback → Semantic Scholar fallback
6. PDFs downloaded from `aclanthology.org/{paper_id}.pdf`
7. `run_acl_batch.py` — runs `main.py --no-llm` on all downloaded PDFs, tracks success/failure in `batch_run_log.json`

---

## Current State & Known Issues

### Fixed Extractor Results (BERT paper)
- 80% precise precision, 85% lenient precision (30 triples, 24 CORRECT)
- 12+ iterations of prompt engineering
- Code-level validation guards: subject presence, object presence, IS-A guard, direction guards, designedFor keyword guard, encompasses direction guard, addresses type guard

### Known Bugs (fixed)
- `structure_builder.py`: `KeyError: 'text'` on table/figure/image blocks → fixed with `.get("text", "")` and `if "text" not in block: continue`
- `citation/network.py`: `TypeError: 'NoneType' object is not iterable` when S2 returns None → fixed with `or []`
- `main.py`: citation fetch used PDF stem as title search → fixed to use `external_id` for direct S2 lookup
- W-series ACL IDs: wrong 5-digit format (W13-31005) → fixed to 4-digit (W13-3105)
- Some citation networks pulled wrong papers from S2 (BERT got Sentence-BERT instead)

### Broken PDF parses — FIXED 2026-06-14 (both were parser bugs, not bad PDFs)
A full-corpus scan once found 2 papers with failed Part-1 parses (both produced 0 fixed triples downstream). Both PDFs had healthy text layers; the failures were two bugs in `structure_builder.py`, now fixed:
- **J13-4001** (Hobbs, "Influences and Inferences", ACL Lifetime Achievement essay) — `is_heading` only accepted `N Title` / `N.N Title` numbering and rejected dotted `N. Title` headers (`1. False and True Starts`), so 0 headings were detected and the whole body was dropped as "before first heading". Fixed: added a dotted-numeric branch with a guard (short, non-sentence remainder) that distinguishes headers from numbered body list items. Now 5 sections / ~8.8k words.
- **D17-1028** ("Exploiting Morphological Regularities…") — the borderless table detector over-extracted 13 "tables" from a 7-page paper; their cell tokens fed `is_table_data_paragraph(threshold=4)`, which then deleted real body prose (~60% of body suppressed). Fixed: added a match-density gate so long, sparsely-matching paragraphs are no longer suppressed (genuine table-row dumps are short/dense). Now 10 sections / ~3.2k words.
Regression-checked on 8 known-good papers: 0 spurious dotted-headings, section/char counts unchanged. The other 31 papers parsed fine (800–5000 words). Low-but-nonzero papers (D17-1245=3, E17-3026=3, C16-1036=5) are genuinely SHORT (~800-1000 words), not broken.

### Entity list source for the fixed extractor — CS-NER gazetteer (IMPLEMENTED 2026-06-16)
**The fixed extractor's subject pool must NOT be back-filled from our own open (`llm`) extraction** (circular). It is built from the **CS-NER dataset** (`github.com/jd-coderepos/contributions-ner-cs`), a human-annotated scientific-entity source — non-circular.

**Mechanism (B — per-paper intersection), via `enrich_entity_csv.py --source csner` (now the DEFAULT):**
1. **Build one global gazetteer once** — aggregate ALL CS-NER files (`acl/{train,dev,test}.data` = 7 types title-level + `full dataset/ncg/pwc/scierc/ftd` `*-abs.data` = method/research_problem). Unique entities mapped to CEO types via `CSNER_TO_CEO` (solution→Model, method→Method, dataset→Dataset, research_problem→Task, tool→Tool, resource→Resource, language→Language). Cached at `output/acl/csner_gazetteer.csv` (~51k entities after filtering; `--rebuild-gazetteer` to refresh).
2. **Per paper**, keep only gazetteer entities that actually appear in `output/<id>/no-llm/output.html` (n-gram match, up to 8 words) → write `Entity_<id>_enriched.csv`. The paper-intersection is what makes a 51k global list paper-specific (~70-150 entities/paper).
3. Feed that per-paper list to the fixed extractor (existing prompt mechanism, unchanged). CS-NER `acl/` title entities remain the seeds.

**Gazetteer quality gate** (`_gazetteer_keep`): multi-word entries kept (full); single-token entries must be distinctive — not an English function word (`_FUNCTION_WORDS`) and seen ≥2× — else common words like "and"/"use" match every paper. (Some generic single-word technical terms like "training"/"rule" remain — they are real CS-NER annotations.)

**Why not CS-NER abstract data per paper, and why not ITER:** the only ACL-id-keyed CS-NER slice (`acl/`) is title-only (`acl/train-abs.data` → 404); the abstract-level data belongs to other corpora (NCG/PWC/SciERC), not our ACL papers. So we use CS-NER as a **global gazetteer intersected with each paper's text**, not a per-paper lookup. ITER/SciERC (`--source iter`) is still available as a model-based non-circular fallback but is **not** the chosen path.
- Columns expected by `entity_loader` / fixed extractor: Entity, Abbreviation, Aliases, TP, NER_Type, CEO_Type.

### STATUS (2026-06-23) — scinex fixed extraction COMPLETE & verified clean; KGE compare is the next step
- **155/155 parsed papers extracted with BOTH ontologies.** CEO `kg/fixed/` = 6,574 triples; scinex `kg/fixed_scinex/` = 6,776 triples (ratio 1.031). The earlier --max-new-tokens 512 truncation of 80 scinex papers (hands_off 2l) is **resolved** — those 80 were re-extracted at the 4096 cap (verified 2026-06-23, hands_off 2m).
- **Zero-triple papers: L16-1593, W14-5502 — both also 0 in CEO, benign (paper content).**
- **⭐⭐ DEFENSIBLE HEADLINE — RotatE + self-adversarial loss (Sun 2019), γ selected on a held-out VALIDATION split, reported on disjoint TEST: MRR ~0.60, Hits@10 ~0.85** (CEO bidir 0.598/filt 0.594; scinex bidir 0.599/filt 0.606; 5 seeds, γ=28). +0.19-0.21 over margin on the same test split — clean model selection, no test-set tuning. This is the number for the paper. Full table hands_off.md §2r; summaries `output/kge_multiseed_summary_test_*`. Eval split via `--eval-split {all,val,test} --split-seed 42`. (All-papers version, no split: §2q, MRR ~0.57.) Self-adv helps RotatE only; ComplEx/TransE stay on margin (default).
- **Ontology comparison (CEO vs scinex) ≈ TIE** under both margin and adv — scinex matches CEO, doesn't degrade it (hands_off.md §2p/§2q). Margin summaries: CEO `output/kge_multiseed_summary.json` (backup `…_ceo.json`) + scinex `…_scinex.json`.
- **⚠ TEXT BASELINES (`text_baseline.py`, TF-IDF title / title+abstract, CPU) look STRONGER than KGE** — title+abstract MRR ~0.70 / H@10 ~0.90 (preliminary, in-flux 315-corpus) vs KGE ~0.60/0.85. A simple lexical baseline beating the KG embedding means the paper can't claim "KGE best" naively — reframe needed. Definitive comparison pending: run text + KGE on the SAME final corpus (`--split-seed 42`). hands_off.md §2s.
- **Corpus scaling IN PROGRESS:** 315 parsed (target ~300 hit); fixed CEO extraction DONE (315/315); scinex extraction running (~156/315). KGE re-run + text baseline still to do on the final corpus.
- **CITATION-EDGE DIRECTION FIX (hands_off §2t):** `filt_*` metric in `evaluate()` now = TRUE reference prediction (GT = papers the query actually cites, by edge direction `source==query`), not the old year≤ proxy. Supersedes all prior `filt_*` numbers (bidir `mrr`/`hits@*` unchanged). Re-run KGE for fresh directional numbers.
- (older status below from Session 17)

### ⚠ STATUS (2026-06-18, Session 17) — 102 papers; KGE precision FIXED (0.05→0.39); multi-seed pending
- **Corpus = 102 parsed papers** (was 54). CS-NER ACL source exhausted; further growth = fetch ARBITRARY ACL Anthology papers (the gazetteer∩text entity list works for any paper). Fetch chain validated on 10 recent main-conf papers (9/10). `--source csner` enrich + fixed extraction are the per-new-paper steps.
- **S2 key wired** in `.env` (auto-loaded by `network.py` via `python-dotenv`; loads from PROJECT ROOT regardless of CWD). Clears the 403. Transient 429-with-retry is normal.
- **KGE precision FIXED — the big result.** Root cause of low scores: query papers were represented only by CEO entities while candidates carried abstract concepts (vocabulary mismatch; the `root_concepts` seed-concept code was unimplemented → 90% of true neighbors ranked 100+). Four changes in `kg_transe_pipeline.py`: (1) **seed papers now get `root_concepts` as mentions edges** (the structural unlock); (2) **`--neg-ratio` default 10** (multi-negative sampling); (3) **`--dim` 64→128**; (4) **`--epochs` 500→1000**. Single-run result on 79-paper eval:

  | Model   | Hits@1 | Hits@5 | Hits@10 | MRR   |
  |---------|--------|--------|---------|-------|
  | TransE  | 0.165  | 0.304  | 0.430   | 0.254 |
  | ComplEx | 0.253  | 0.506  | 0.582   | 0.362 |
  | **RotatE** | **0.266** | 0.494 | **0.608** | **0.387** |

  → ~4–7× jump vs pre-fix; Hits@10 ≈ 0.61, Hits@1 ≈ 0.27. **Genuinely useful now, not just above-random.**
- **MULTI-SEED, 155-paper corpus (latest, `output/kge_multiseed_summary.json`, 5 seeds): ComplEx best** — bidir MRR **0.432 ± 0.018** / temporal-filtered MRR **0.456 ± 0.027**, Hits@10 ~0.74-0.76; RotatE 0.403/0.423; TransE 0.315/0.334. ComplEx & RotatE close throughout (~0.03); **best model is corpus-dependent** (RotatE won at 79 papers w/ MRR 0.505; ComplEx wins at 155 and scales better). Absolute MRR dropped 0.505→0.43 going 79→155 papers — EXPECTED (bigger candidate pool = harder); report lift-over-random. Precision arc that got here: pre-fix ~0.05 → seed-concepts+tuning 0.364 → #1 empty-concept backfill 0.423 → #2 full-text seed concepts 0.505 (79 papers); #4 vocab-normalization reverted (net negative). Eval now reports TWO metrics: bidirectional (`mrr`/`hits@*`) + temporal-filtered "papers it cited" (`filt_*`, candidates/GT restricted to year ≤ query year).
- **KGE eval emits DIAGNOSTIC per-paper output**: `top10_predictions` (title + `is_true_citation`), `gt_ranks` (rank of every true neighbor, `null`=not a node). Use to separate near-misses / coverage gaps / genuine misses.
- **Next:** multi-seed (`kge_multiseed.py … --epochs 1000`, inherits dim/neg defaults) to settle RotatE vs ComplEx; then cheap precision headroom — backfill the 27% empty-concept nodes + tighten the candidate pool. (Stale single-run `kge_fixed_results_*.json` reflect the LATEST fixed code; the 37-paper multiseed table in Session 16 is OLD/pre-fix.)

---

## Conventions
- **Always run commands with `python3`, not `python`** (use `python3 kg_main.py ...`, `pip` as `python3 -m pip` if needed). On this VM `python` does not work as expected for running the pipeline.
- Files are always delivered individually, never as zip archives
- Code runs locally on the private VM; output files stay in `output/`
- Entity CSVs named `Entity_{paper_id}.csv` or `Entity_-_{name}v2.csv`
- Paper IDs: old-style ACL (C16-1036, P18-1001), new-style (2020.acl-main.130), or arXiv (1810.04805)
