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
  - **Relation-only** (`kg_extraction/relation_extractor.py`) — NEW 2026-08-13. Relation ∈ ontology, **subject free** (any specific named entity, but it must appear literally in the source sentence). Subclasses `FixedTripleExtractor` and reuses its prompt and every post-parse guard — only the subject constraint differs, so `fixed` vs `relation` isolates exactly what the curated entity list buys. `--extractor relation` → `kg/relation/<model>/` (`relation_scinex/` with `--ontology scinex`). **This is the fallback for papers the CS-NER gazetteer can't cover** (non-CS domains); use `entity_coverage.py` to decide per paper.

  Constraint ladder: `llm` (all free) → `relation` (relation fixed) → `fixed` (subject + relation fixed) → `pair` (subject + object fixed).
- Output: `output/{paper}/kg/{extractor}/{model}/triples.json`
- Run: `python3 kg_main.py --paper BERT --extractor fixed --entity-csv Entity-BERTv2.csv --model Qwen/Qwen3-14B`
- **`--entity-csv` is optional**: if omitted, the entity CSV is auto-resolved per paper from its id (enriched ACL CSV → title-only ACL CSV → project-root manual CSV). So `python3 kg_main.py --paper C16-1036 --extractor fixed --model Qwen/Qwen3-14B` just works, and `--all` resolves a different CSV per paper (papers with no CSV skip fixed extraction).

### Part 2b — Persistent global KG (kg_extraction/global_graph.py) — NEW 2026-07-17
Fixed/fixed_scinex extraction now ALSO folds every paper into one persistent, incrementally-merged cross-paper graph, in addition to (not replacing) each paper's own isolated `triples.json`/`kg.graphml`. Framing: each paper is a **big node** (`paper:<id>`); its fixed-extraction entities are **augmented nodes** hanging off it via `mentions` edges; and — unlike `kg_transe_pipeline.py`'s eval-time graph, which holds citation edges out as ground truth only — citation edges here are **real `paper→cites→paper` edges**, so big nodes connect to each other directly.
- Output: `output/global_kg/<extractor>/<model_slug>/graph.graphml` + `meta.json` (extractor = `fixed` or `fixed_scinex`, mirroring the existing `kg/<extractor>/<model>/` split).
- Wired into `kg_main.py` automatically (merges after every paper's extraction, including on the `--skip-existing` path, so the first run over an already-extracted corpus backfills the whole thing). New flags: `--global-graph-dir` (default `output/global_kg`), `--no-global-merge`, `--global-save-every` (default 25 — periodic checkpoint during a big `--all` batch, since re-writing the full graphml after every single paper would be wasteful).
- Handles node identity across time: a citation may reference a neighbor paper (by raw Semantic Scholar id) that isn't in the corpus yet → gets a `paper_stub` node; once that paper is actually extracted, its stub is `nx.relabel_nodes`-merged into the real `paper:<id>` node, preserving any edges already attached to the stub. Verified with a standalone synthetic test (idempotent re-merge, stub creation + resolution, citation edges surviving relabel, graphml round-trip, `connectivity_report`) — all passed.
- **`expand_and_validate.py`** (new, project root) — validation smoke test: given `--seed <paper_id>` (a real ACL Anthology paper already in the corpus, e.g. `2020.acl-main.185` — NOT the manually-added one-off "BERT" folder, which has no ACL id), pulls ~8 new papers from that seed's `citation_network.json` (reusing `citation_expand_pipeline.py`), parses them, enriches entities, runs fixed extraction (GPU), and prints a connectivity report confirming the new papers actually attached to the graph (via citation and/or shared-entity edges) rather than landing as isolated islands.
- **Scope/what's NOT done yet**: this is foundation only. `kg_transe_pipeline.py`'s training/eval is untouched — it still builds its own ephemeral graph the old way, citation-edges-held-out. Two follow-ups are explicitly deferred: (1) a masked-node train/test scheme (hide 1-2 big nodes, predict whether they link back to the query node) using this persistent graph instead of the current hold-out-edges eval; (2) a citation-only ablation (paper-paper edges, no augmented/entity nodes) run through the same masked-node task, to isolate how much the augmented nodes actually help. **Not yet run on the VM** — `expand_and_validate.py` needs to be executed there (GPU) to confirm the merge logic holds up on real extraction, not just the synthetic unit test.

### Part 2c — Per-triple quality evaluation (kg_evaluate.py + gold_eval.py) — REVIVED 2026-08-13
**LLM-as-Judge is un-retired** and is the evaluation track for the current task (fixed extraction accuracy on ~20 handpicked papers, CEO vs scinex). Loop:
1. `kg_evaluate.py` — judges every triple against its own `source_sentence` (CORRECT / PARTIAL / INCORRECT / UNVERIFIABLE) → `kg/<extractor>/<model>/evaluation.json` + `eval_summary.json`. Additions this session: stable `triple_id` on every triple (content hash incl. extractor family, so labels survive re-runs), **the predicate's ontology definition is now shown to the judge** (CEO from `_CEO_SCHEMA`, scinex parsed from the OWL via `--ontology-file`) so a verdict tests domain/range conformance, not just plausibility of the English relation name, `by_predicate` precision breakdown, per-paper entity-CSV alias auto-resolution (was one global `--entity-csv`), exact extractor-family matching (`--extractor fixed` no longer also selects `fixed_scinex`), `--resume`, `--max-per-paper`, `--summary-out` corpus report.
2. `gold_eval.py export` — stratified sample (round-robin over extractor×predicate buckets, so rare predicates are represented) → CSV with a blank `verdict` column and a `predicate_definition` column carrying the same ontology definition the judge sees. Judge verdict deliberately withheld to avoid anchoring the labeller. The same CSV is what a *different* model fills in when used as a second judge. **Verdict rubric (identical for human and judge — printed by `export`):** CORRECT = sentence explicitly states it AND predicate fits its domain→range; PARTIAL = implied, or loose predicate fit; INCORRECT = unsupported, or subject/object swapped relative to domain→range.
3. `gold_eval.py agree` — joins labels back by `triple_id`: raw agreement, Cohen's κ (3-class + CORRECT/not binary collapse), confusion matrix, per-extractor and per-predicate tables, full disagreement dump. `--b other.csv` compares two labellers instead of labeller-vs-judge.
4. `gold_eval.py errors` — judge's PARTIAL/INCORRECT grouped by predicate with examples = the worklist for the next round of prompt/guard fixes in `fixed_extractor.py`.
- Human verdict is the ground truth; the judge is validated against it before its corpus-wide numbers are trusted.

### Part 3 — KG Embedding Evaluation (kg_transe_pipeline.py) — citation-prediction evaluation
This is the **KG-level** evaluation (Part 2c is the per-triple one).
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
fetch_corpus.sh             — download the open-access papers listed in papers/manifest.csv → papers/<paper_id>.pdf
run_relation_extraction.sh  — VM runner: relation (+ --with-fixed) × both ontologies × the corpus, one GPU process at a time, then a triple-count table
run_local.py                — LOCAL runner: drives kg_main through Ollama (kg_extraction/ollama_backend.py)
run_local_judge.py          — LOCAL LLM-as-Judge runner (kg_evaluate through Ollama; different model family)
count_triples.py            — triple counts per paper/extractor + predicate breakdown
ontology_eval.py            — ⭐ C1-C6 ontology-quality evaluation (Session 27). THREE harnesses because the criteria have different units: `triples` (C1/C3/C4, per triple), `paragraphs` (C2/C5, per paragraph + all its triples), `papers` (C6, per paper). Then `collect` + `report`. Judge = claude-opus-5 IN SESSION, never via OpenRouter.
run_api_judge.py            — drives judge_paste.py bundles through an API model (rubric as system prompt on every call, so batches can't drift); replies land where `judge_paste.py collect` looks
check_provider.py           — preflight for an API extraction run: finds the key (env/.env), LISTS the models that key can actually see (ids drift — looked up, not guessed), picks the strongest, prints the exact smoke-test + full-corpus + report commands. `python3 check_provider.py groq|gemini|openai`
papers/manifest.csv         — the 20-paper handpicked corpus: paper_id, DOI, access status, PDF URL
ft_common.py                — ⭐ Part 2/3 shared: sticky paper-level fold split, paragraph loading, defensive reply reader, guard pass, the chat-template helper shared by training AND inference, Claude refusal
build_ft_data.py            — ⭐ Part 2: finetuning data (ontology cards + vocabulary cards + gptoss extraction pairs) as chat JSONL → ft_data/fold<k>/ ; --dry-run writes nothing
ft_train_qlora.py           — ⭐ Part 2: QLoRA on one base model → ft_runs/<run>/adapter + run_manifest.json ; --dry-run = tokenizer stats only
ft_generate.py              — ⭐ Part 2 eval: base or base+adapter through the same relation-extraction path → responses.jsonl → claude_extract.py ingest; refuses the adapter's training papers
ft_compare.py               — ⭐ Part 2 eval, judge-free: base vs finetuned vs teacher — valid JSON, truncation, proposed vs kept, coverage, teacher agreement (imitation, not quality)
run_ft_pipeline.py          — ⭐ unattended one-fold run: base_gen → train → ft_gen → compare → qa, resumable via .done markers, keeps a Windows laptop awake
qa_eval.py                  — ⭐ Part 3: simulated Q&A — generate (verified questions from paper text) / answer (kg | oracle | closedbook) / grade (deterministic token overlap by default; --method llm) / report (cluster-bootstrap CI, token F1, answer_in_kg, grounded, paired compare)
blind_judge.py              — ⭐ Part 2 eval: BLIND pooled C1/C3/C4 census of several extractors on the same papers (dedup, shuffle, salted ids, key.csv hidden) — judged in session by claude-opus-5-5
build_ft_data.py --consensus-models — adds triples ≥2 other models agree on to the teacher's targets → ft_data/consensus/
ft_runs/wait_then_run.py    — wait for a pipeline PID (keeping it + children out of Windows EcoQoS), then run the next command
entity_coverage.py          — per paper, count the entities the fixed extractor would get and decide fixed vs relation (no GPU)
parse_check.py              — parser regression harness: title/sections/body-vs-reference word counts per PDF (no GPU); always run it with BERT.pdf included
kg_extraction/
    fixed_extractor.py      — Fixed-subject extractor (CEO ontology, constrained entities)
    relation_extractor.py   — Relation-only extractor (ontology relations, free subjects) — fallback when no entity list covers the paper
    llm_extractor.py        — Free LLM extractor (open vocabulary)
    extractor.py            — REBEL baseline
    iter_extractor.py       — ITER/SciERC extractor (fleonce/iter-scierc-deberta-large)
    entity_loader.py        — loads entity CSV, builds alias lookup
    kg_builder.py           — NetworkX KG + entity normalisation → triples.json + kg.graphml
    global_graph.py         — persistent cross-paper global KG (paper big-nodes + entity augmented nodes + real cites edges); incremental merge with stub resolution
    html_parser.py          — extracts sentences from output.html for extraction
    visualizer.py           — interactive pyvis HTML visualization
kg_evaluate.py              — LLM-as-Judge evaluation (source sentence faithfulness); stable triple ids, per-predicate breakdown, --resume
gold_eval.py                — human/cross-model labelling: export sample CSV → agree (accuracy, Cohen's κ, confusion, disagreements) → errors (prompt-fix worklist)
gold_report.py              — filled gold CSVs → sampled + corpus-weighted precision, per-predicate/per-paper tables, bootstrap CI (reproduces every results.md per-triple table)
claude_extract.py           — Claude (or any API model) as the extraction backend: `prompts` renders the real system+user prompts per paragraph, `ingest` feeds replies back through the same guards + KG builder
claude_corpus_report.py     — bundles every Claude-extracted paper into output/<model-slug>_relation_corpus.json + prints the Claude-vs-local-model volume/predicate-mix table
judge_paste.py              — package a gold sample so ANOTHER model can judge our triples through a chat UI: `batches` writes the rubric + triples, `collect` writes a filled label CSV for gold_report.py / gold_eval.py agree. Claude must never judge Claude's own extraction
chat_paste_extract.py       — same experiment through a FREE CHAT UI: `batches` writes copy-pasteable batch_NN.txt (system prompt once per conversation), `collect` reads reply_NN.txt back into responses.jsonl. Whole corpus = 49 messages at --size 12. ⚠ batching ≠ one-paragraph-per-call, and a free UI can switch model version mid-session — sanity check only, not a paper number
run_api_extract.py          — drives the prompts→responses step with ANY model (anthropic/openai/openai-compat/gemini/ollama); stdlib only, keys from env, --resume/--limit/--sleep/--ingest/--all. 486 calls per corpus run (~1.6M input tokens uncached, ~162k with the system prompt cached)
kg_fixed_compare.py         — compare fixed-subject vs free triples side by side
kg_compare.py               — compare results across all extractors
kg_citation_fusion.py       — merge KG with citation network
kg_transe_pipeline.py       — KG embedding evaluation (TransE/ComplEx/RotatE via --kge)
acl_index.py                — builds local title→ACL paper ID index from anthology.json.gz
acl_pipeline.py             — ACL paper download pipeline (CS-NER IOB → entity CSVs + PDFs)
enrich_entity_csv.py        — enrich title-only ACL CSVs into a real subject pool for fixed extraction (harvest body entities from open-extraction triples, or ITER/SciERC)
run_acl_batch.py            — batch runner for parsing multiple ACL PDFs (tracks success/failure)
citation_expand_pipeline.py — snowball paper downloads by walking citation networks
expand_and_validate.py      — global-KG validation smoke test: pull N papers from one seed's citation network → parse → enrich → fixed-extract → connectivity report
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
        relation/{model}/triples.json — relation-only extraction (CEO; relation∈ontology, subject free)
        relation_scinex/{model}/triples.json — same with the scinex ontology
        fixed_scinex/{model}/triples.json — fixed LLM extraction using the scinex ontology (--ontology scinex); coexists with CEO fixed/
        fixed/{model}/evaluation.json    — LLM-as-Judge results
        pair/{model}/triples.json    — pair LLM extraction (subject∈list, relation free, object∈list)
        fused/fused_kg.graphml       — merged KG + citation graph
output/global_kg/{extractor}/{model}/
    graph.graphml               — persistent cross-paper global KG (paper big-nodes, entity augmented nodes, real cites edges)
    meta.json                   — merge bookkeeping (merged_papers, s2_to_paper stub-resolution map, stats)
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
- **Use stable torch, never a dev-nightly.** A torch `2.12.0.dev*+cu128` nightly silently crippled bitsandbytes 4-bit decode to ~2 tok/s. Stable `torch 2.6.0+cu124` (via `--index-url https://download.pytorch.org/whl/cu124`) restores ~16 tok/s; cu124 wheels run fine on the 12.8 driver. **⚠ This applies to the H100 vGPU only. Blackwell sm_120 GPUs (the RTX 5090 VM, the RTX 5070 laptop) need a CUDA 12.8+ build; cu124 has no sm_120 kernels. See `hands_off.md` §34.12.**
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

### Non-ACL venue parsing — FIXED 2026-08-25 (the 20-paper corpus is Nature/MDPI/IEEE/Elsevier, not ACL)
The parser was tuned on ACL papers, which **number their headings**. The handpicked corpus is not ACL, and three failure modes followed. All verified by running the parser locally (see the local-Python note below) and regression-checked against `BERT.pdf`.
- **Unnumbered headings → whole body dropped.** `is_heading` accepted only numbered/roman/lettered/ALL-CAPS headings, so Nature-style `Introduction` / `Results` / `Methods` matched nothing; the only detected heading was `References`, and `build_structure`'s `if not current_section: continue` discarded everything before it. The 3 Scientific Reports papers kept **only their bibliography** (0 body words). Fixed with `_SECTION_VOCAB` — a **closed** set of ~55 section names (a general "short title-case line" rule would fire on body prose). strabismus2026 0→6,456 body words, osmotic2026 0→3,077, oxidecrack2025 0→4,978.
- **The abstract was dropped from EVERY paper** (it precedes the first heading). Pre-heading blocks are now buffered and filtered by `looks_like_abstract_prose()` into a leading `Abstract` section — which also acts as a safety net if heading detection ever fails wholesale again.
- **Titles were journal mastheads** (`sensors`, `applied sciences`) because `extract_title` scored blocks by **length alone**. `layout.py` now annotates blocks with font size (`get_text("dict")` spans) and `extract_title` takes the largest-font page-1 blocks, joins wrapped title lines, strips `Article`/`Review` labels and rejects DOI/ISSN/`Citation:`/masthead furniture.
- **Crash on roadwaylight2018:** table blocks with `page=None` reached the reading-order `sorted()` → `TypeError: '<' not supported between NoneType and int`, killing the parse and leaving an empty output dir. `_reading_order_key` is now None-safe.
- **All of the above are now RESOLVED** (see `hands_off.md §20`): the 2 MDPI titles (whitespace-collapse before the masthead test; the author-list guard no longer fires on "YOLOv8 and Tracking A"), the Nature heading-glued-to-paragraph layout (`layout.py:_split_glued_headings` splits on font evidence and marks `is_heading_hint`, so `source_meta["section"]` is real again), and roadwaylight2018's running-header junk headings (`_is_running_header`). Remaining issues are cosmetic only: `Task #2:` truncated in BERT, one nested run-in heading left inline in oxidecrack2025, and traveltime2022's `References`-before-`5. Conclusions` reading-order artifact.

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

### STATUS (2026-10-26, Session 36 close) — ⭐⭐⭐ FOLD-0 FINETUNING EXPERIMENT COMPLETE AND JUDGED. A finetuned 1.7B student matches the 120B teacher per triple, but under-extracts badly. READ `hands_off.md` §36.
- **⭐⭐⭐ THE RESULT (CEO, blind census of all 174 paragraphs × 4 extractors, §36.12 / `results.md`
  Table 14).** Judge: claude-opus-5-5 in session, BLIND, 355 model-triples → 315 unique items.
  | extractor | C1 | C3 | C4 | C4 strict | kept triples | coverage |
  |---|---|---|---|---|---|---|
  | qwen3-1.7b-base-hf | 59.8 | 52.3 | **29.4** | 22.4 | 123 | 20.7% |
  | qwen3-1.7b-ft-cv (teacher-only) | 90.7 | 81.5 | **83.3** | 74.1 | 29 | 4.6% |
  | qwen3-1.7b-ft-cons-cv (consensus) | 90.2 | 87.8 | **85.4** | 78.0 | 47 | 7.5% |
  | gptoss-120b (teacher) | 96.4 | 87.2 | **86.4** | 74.4 | 180 | 42.0% |
  Outright relation errors (C4 INCORRECT) fall from **63.6% of the base model's triples to ~7%**
  (teacher 1.7%). **Teaching the ontology, not scaling the model, is what fixes relation typing.**
- **⚖ Teacher-only vs consensus = the experiment of record: quality is a TIE, volume is not.**
  C4 **+2.0 pts [−12.7, +16.8]**, C1 **−0.5 [−12.7, +12.4]** (n=27 vs 41, unresolved). Consensus wins
  on everything countable: **47 vs 29 kept triples (+62%)**, coverage 7.5% vs 4.6%, guard rate 73.4%
  vs 64.4%, C3 87.8 vs 81.5. **Adopt consensus.** Its lower teacher-agreement (55.3% vs 72.4%) is by
  construction, not a regression.
- **⛔ The open problem: precision without recall.** Both students answer `{"triples": []}` for
  90–95% of paragraphs. 29/47 triples against the teacher's 180. **Cause: 402 of 619 training
  targets (65%) are empty.** Built and NOT yet trained: `ft_data/maxempty30/` (310 examples, 30%
  empty, 744 steps) via `--max-empty-frac 0.3`.
- **Also done:** format is solved (valid JSON 87.9% → 100%, truncations 18 → 0, and the finetuned
  model is **12.5× faster to run**: 0.19 vs 2.40 min/paragraph).
- **⛔⛔ THE EXPENSIVE BUG, FIXED (§36.11): `no_power_throttling(proc.pid)` was opting out a
  TRAMPOLINE.** A Python-Install-Manager venv's `python.exe` re-execs the real interpreter as a
  child, so the trainer stayed on efficiency cores: **48.4 s/step → ~9 s/step (8×)**, GPU 0% util /
  9.8 W → 71% / 56 W; consensus training 96.7 min vs the teacher-only run's throttled 315.9 min.
  Fixed by **`no_power_throttling_tree()`** (opts out the whole descendant tree; used by every
  pipeline step) and **`ft_runs/ecoqos_watchdog.py`** for already-running pipelines.
  **Diagnostic: `nvidia-smi` at 0% util / <10 W with no throttle reason = GPU starved by the CPU
  side, not throttled.**
- **Machine:** RTX 4060 Laptop, 8 GB, sm_89, stable torch 2.11.0+cu128 (no nightly, no
  `--allow-dev-torch`). Student `Qwen/Qwen3-1.7B` at `C:\Users\cei008\hf_local\Qwen3-1.7B`.
- **Next, in order:** scinex blind census (CEO only so far) → train `ft_data/maxempty30` as the
  third variant → 4-fold CV with the winner (~5.5 h/fold now the throttling is fixed) → Part 3 Q&A
  with an independent non-Claude model (the user has OpenRouter credit earmarked for this, §36.6).

### STATUS (2026-10-24, Session 36) — ⭐⭐ FIRST FINETUNING RESULT: format+precision learned, coverage collapsed. ⛔ Everything had been running 8× slow. READ `hands_off.md` §36.
- **⭐⭐ Teacher-only Qwen3-1.7B, fold 0, all 5 held-out papers, census of 174 paragraphs/ontology**
  (§36.10). CEO: valid JSON **87.9% → 100%**, truncations **18 → 0**, guard kept rate
  **33.6% → 64.4%** (teacher 81.8%), agreement with the teacher **13.8% → 72.4%** precision.
  scinex moves identically (35.7% → 63.6%). **⛔ But coverage fell 20.7% → 4.6%** (teacher 42.0%):
  the model returns `{"triples": []}` for **162 of 174 paragraphs (93.1%)**, recall vs teacher
  12.8%. It learned the teacher's precision AND its silence — the §34.11 failure, worse than on the
  4B. **Cause: 402 of 619 training targets (65%) are empty.**
- **Prepared, NOT run (open decision 5):** `build_ft_data.py --eval-fold 0 --max-empty-frac 0.3
  --out ft_data/maxempty30` is built — **310 extraction examples, 93 empty (30%)**, 744 steps.
  Halves the examples, fixes the abstain prior.
- **▶ RUNNING:** the consensus variant (`train` → `ft_gen` → `compare`), started 13:02.
- **⛔⛔ THE EXPENSIVE BUG (§36.11): `no_power_throttling(proc.pid)` was opting out a TRAMPOLINE.**
  On a venv built by the Windows **Python Install Manager**, `.venv\Scripts\python.exe` re-execs the
  real interpreter (`pythoncore-3.12-64\python.exe`) as a **child**, so the opt-out landed on a
  launcher that does no work while the trainer stayed on the efficiency cores. Measured:
  **48.4 s/step → ~9 s/step (≈8×)**, GPU 0% util / 9.8 W / 930 MHz → 71% / 26–56 W / 2475 MHz;
  consensus training ETA 11 h → ~2 h. The teacher-only run's 315.9 min training and 14.1 h
  `base_gen` were both throttled. **Results are unaffected — greedy decoding is deterministic — so
  nothing needs re-running**, it only cost wall-clock.
  Fixed two ways: **`run_ft_pipeline.no_power_throttling_tree()`** opts out the PID *and every
  descendant* (sweeps 3×, psutil with a WMI fallback) and `sh()` now uses it for every step;
  **`ft_runs/ecoqos_watchdog.py`** covers an already-running pipeline by sweeping matching command
  lines every 60 s. **Diagnostic to remember: `nvidia-smi` showing 0% util at <10 W with no
  throttle reason means the GPU is STARVED by the CPU side, not throttled.**
- **Measured generation cost** (§36.9): base model **2.40 min/paragraph** (348 paragraphs ≈ 14 h,
  throttled); the finetuned model **0.19 min/paragraph** — 12.5× faster, because it stops rambling
  (base truncated 18 replies at 3,072 tokens; finetuned and teacher truncate 0).

### STATUS (2026-10-23, Session 36) — setup FINISHED and verified on the 4060; smokes pass; both Qwen3-1.7B variants RUNNING unattended. READ `hands_off.md` §36.
- **▶ RUNNING NOW:** the detached chain `ft_runs/chain_session35.py` (relaunched 15:38) is doing
  teacher-only then consensus on fold 0: `base_gen → train → ft_gen → compare` twice. Watch
  `ft_runs/chain_session35.log`, then `ft_runs/_pipeline_qwen3-1.7b-f0/pipeline.log`. Budget
  ~10–12 h. **Check the log before touching anything.**
- **Stack verified:** `torch 2.11.0+cu128 | cuda True | cap (8, 9) | bnb 0.49.2 | peft 0.21.1 |
  transformers 5.5.4 | numpy 2.4.4`, 8.0 GiB VRAM. Qwen3-1.7B snapshot 3.80 GiB, byte-exact, at
  `C:\Users\cei008\hf_local\Qwen3-1.7B`.
- **Smokes pass (§36.5):** generation 2 paragraphs in 0.7 min at **peak VRAM 2.3 GB** with valid
  JSON; training 2 steps at **~14 s/step, peak 2.49 GB**, LoRA 17.4M trainable (1.00%). ⚠ 14 s/step
  was measured under EcoQoS, so the pipeline should beat it; 822 steps ≈ 3 h per variant.
- **⭐ sm_89 does NOT hit the sm_120 SDPA→MATH trap** (§33.3): training the 1.7B peaked at 2.49 GB
  total, where sm_120 took 7.27 GB for ONE layer. This machine does not depend on the cuDNN work-around.
- **⛔ POST-MORTEM of the first chain run (§36.1):** it reported `STEP B FAILED` on 10-23 for a step
  that had SUCCEEDED on 10-20. The laptop slept at 14:39 (a keep-awake request stops idle sleep, not
  a lid close) and `step_b` checked its absolute 8 h deadline BEFORE re-probing readiness. Fixed:
  readiness first, and every wait is now measured from the last observed PROGRESS. Any unattended
  runner on a laptop needs both.
- **⭐ The link is now ~6.8 MB/s, ~20× faster than 10-20** (§36.3). The bandwidth reason for choosing
  the 1.7B over `Qwen3-4B-Instruct-2507` (an 8.05 GB, ~20-minute download now) is gone. The run in
  flight stays 1.7B; whether to add the 4B for comparability with §33/§34 is the user's call.
- **OpenRouter credit is back on the table (§36.6, user-raised).** To be discussed AFTER this run.
  The natural target is the Q&A writer/answerer (open decision 1), which is what currently makes
  Part 3 a pipeline test rather than a result; second, the ~$0.01 qwen3-235b sentinel re-call.
  ⛔ Still never Claude through an API.

### STATUS (2026-10-20, Session 35) — NEW MACHINE: RTX 4060 Laptop 8 GB (sm_89). Data recovered locally, install green, student = Qwen3-1.7B. READ `hands_off.md` §35.
- **Work moved to a new Windows laptop: RTX 4060 Laptop GPU, 8 GB, sm_89 (Ada), driver 592.82.**
  The user called it a 4070; `nvidia-smi` says 4060 Laptop. **The RTX 5090 VM run (§34.12) did not
  happen — training runs here.** Python **3.12.10** (installed with `py install 3.12`; the machine
  had only 3.14.7, which has no torch/bitsandbytes wheels), venv at `project_scinex/.venv`.
- **⭐ torch `2.11.0+cu128` — a STABLE wheel. `--allow-dev-torch` is NOT needed on sm_89.** Only the
  sm_120 machines (RTX 5070/5090) needed a nightly; `2.6.0+cu124` remains H100-vGPU-only.
- **⭐ The private bundle was unnecessary: an older local copy of the project
  (`C:\Users\cei008\Downloads\project_scinex`) held every gitignored input.** Restored into the
  clone: 22 PDFs, 22 parsed `no-llm/`, **293 `prompts.jsonl`**, `output/acl/` (gazetteer + per-paper
  entity CSVs). `git status` stayed clean. Check an older local copy before chasing the bundle.
- **⭐ The Session 33 "only 3 usable training papers" limit is GONE.** With the teacher's stored
  prompts present, **all 15 fold-0 training papers pair on both ontologies** (prompts == replies).
  Fold 0's only mismatch is `strabismus2026` scinex (51 vs 61 replies) = the known §30.8 double
  write, dropped by `read_replies`. Teacher paragraphs are byte-identical to today's render for
  `strabismus2026` (51/51) and `osmotic2026` (23/23), so old and new runs share identical inputs.
  ⚠ The `ft_data` example counts in §33.4/§34.4 were built from 3 papers and will change.
- **Install check green** (`SETUP_NEW_MACHINE.md` §7): CEO `qwen3-4b-base-hf` 493 → 172 (34.9%) and
  `gptoss-120b` 107 → 91 (85.0%); scinex 464 → 164 (35.3%) and 70 → 57 (81.4%) — all as expected.
- **⛔ The binding constraint on this machine is download bandwidth (~145–470 KB/s), not VRAM.**
  torch wheel 2.75 GB; Qwen3-4B-Instruct-2507 8.05 GB; Qwen3-1.7B 3.88 GB.
  **Student decided by the user: `Qwen/Qwen3-1.7B`** (weights kept OUTSIDE the repo at
  `C:\Users\cei008\hf_local\Qwen3-1.7B`, passed as `--base-model <path>`; slugs
  `qwen3-1.7b-base-hf` / `-ft-cv` / `-ft-cons-cv`). ⚠ **Its numbers are NOT comparable with the
  Qwen3-4B figures of §33/§34** (38% → 86% kept, 8 vs 21 paragraphs) — always name the model.

### STATUS (2026-10-01, Session 34; updated 2026-10-20) — multi-model data BUILT; local runs KILLED; training moves to an RTX 5090 VM. START WITH `HANDOFF_NEXT_SESSION.md`, then `hands_off.md` §34.
- **▶ NEXT = the RTX 5090 VM run, which the user runs** (changed from the DGX Spark on 2026-10-20;
  exact commands: `hands_off.md` §34.12): Qwen3-8B, fold 0, teacher-only (`qwen3-8b-ft-cv`) vs
  consensus (`qwen3-8b-ft-cons-cv`), then a blind judge census here. Nothing is running locally; no
  results from the VM yet. **⛔ The 5090 is sm_120: it needs a CUDA 12.8+ torch. The "2.6.0+cu124"
  GPU note below is for the old H100 vGPU only.**
- **Repo = https://github.com/SethapongS0212/SciNex, folder `project_scinex/` (PUBLIC).** New machine
  → `SETUP_NEW_MACHINE.md`. Files kept out of GitHub (PDFs, full text, adapters) travel in the private
  `SciNex_private_bundle.zip`. Work in a clone; pull before, push after. `hands_off.md` §34.14.
- **`build_ft_data.py --consensus-models gemma4-31b qwen3-235b ministral-14b`** adds triples that
  ≥2 of those models agree on to gptoss's targets (teacher stays the anchor; additions never
  override a teacher-related entity pair). Local fold 0: +66 triples on the teacher's 192 (+34%) →
  `ft_data/consensus/fold0/`. The teacher-only rebuild is byte-identical.
- **⏹ Local runs KILLED (user); training moves to the RTX 5090 VM** (first planned for a DGX Spark). Teacher-only Qwen3-4B adapter
  finished; partial extraction (44 paragraphs) shows no collapse, kept rate 38% → 86% (teacher
  88%), but under-extraction (8 vs 21 paragraphs). The consensus variant was never trained.
  `hands_off.md` §34.11.
- **⛔ Windows EcoQoS throttled the hidden training run 4× (40 → 7 s/step once lifted).**
  `run_ft_pipeline` now opts out itself + children; `ft_runs/wait_then_run.py` for old runners.
- Census: other models agree with gptoss only 3–21% (exact), so consensus filtering, not stacking.
- **⭐ JUDGE DECIDED (user, 2026-10-01): claude-opus-5-5 IN SESSION, BLIND, CENSUS: `blind_judge.py`.**
  This reverses the Session 32 "no Claude judge" rule for judging ONLY; Claude is still never an
  extractor, teacher, training-data source or API-called model. Not comparable with the
  claude-opus-5 Tables 10–12. Runs once both finetuned KGs exist. `hands_off.md` §34.8.
- **Q&A grading is deterministic by default** (token overlap + token-F1 column; `--method llm`
  optional). `hands_off.md` §34.9.
- **⚠ qwen3-235b: 375/567 CEO replies are the failed-call sentinel string**, so its 16.4% coverage
  and C6 vacuous passes may be failures, not abstention. Unresolved; caveat added under results.md Table 10.

### STATUS (2026-10-01, Session 33) — FIRST LOCAL TEST RUN on the 8 GB laptop. READ `hands_off.md` §33.
- **Local student = `Qwen/Qwen3-4B-Instruct-2507`** — Qwen3-8B cannot be TRAINED in 8 GB (~6 GB of
  weights+embeddings before activations). Laptop = RTX 5070 Laptop, 8 GB, sm_120, torch cu128 nightly.
- **⛔ On this laptop torch SDPA falls back to the MATH kernel (7.27 GB for one layer at 3.8k tokens)**
  — `ft_common.fast_attention(training=True)` forces cuDNN for TRAINING (0.34 GB). **Never use cuDNN
  for generation** — it re-plans every decoded token (5.7 vs 18 tok/s); the inference default skips it.
- Test run: train on gamlprop2025 / parkingyolo2023 / traveltime2022 (168 extraction pairs + 2,312
  cards), evaluate on strabismus2026 / osmotic2026. Pipeline test only — Q&A uses the base 4B.
- **Outcome:** base extraction done (CEO 493 proposed → 172 kept, 34.9%, vs teacher 85.0%). Training #1
  **collapsed** (always `{"triples": []}`) from per-example loss averaging → fixed to token-level. The
  fixed retrain stopped at step 72/620. **No usable finetuned model yet.** Resume per
  `HANDOFF_NEXT_SESSION.md`; slide summary in `SUMMARY_finetune_qa_2026-10-01.md`.

### STATUS (2026-10-01, Session 32) — FINETUNING (Part 2) + Q&A EVAL (Part 3) CODE BUILT; NOTHING RUN. READ `hands_off.md` §32.

- **The project is now three parts:** (1) paper → KG (built), (2) **finetune an open-weight LLM on
  the domain ontology + vocabulary** — current work, (3) **evaluate KGs by simulated Q&A against the
  real papers**. Run order: `TASK.md` top section.
- **⛔ Claude is OUT of the pipeline** — not extractor, judge, teacher, or training data. Enforced in
  code: every new script refuses a Claude model id / slug. **This supersedes the "judge = claude-opus-5
  in-session" convention below for all new evaluation** — the finetuned-vs-base C1/C3/C4 comparison
  needs a non-Claude judge (recommended gemma-4-31b: not the teacher, not the student's family).
- **Teacher = gptoss-120b's stored relation-mode replies** (Apache-2.0). Student = `Qwen/Qwen3-8B`
  pending Sam's confirmation; then ministral-14b; gemma-4-31b only if those gain.
- **⛔ HARD RULE: train and eval papers are disjoint**, via sticky folds in `ft_data/split.json` (K=4;
  fold 0 = csysguard2024, linkpred2015, osmotic2026, strabismus2026, textaug2023). Checked at build,
  train and generate time. Cross-validate by reusing ONE `--model-slug` across folds.
- **Extraction pairs need `no-llm/output.html` or `prompts.jsonl`** — both gitignored, absent from a
  GitHub checkout. Build the data on the VM.
- **Guard parity verified:** re-running today's guards on gptoss's stored replies reproduces the
  on-disk `triples.json` exactly (CEO 547 = 547). **Side finding:** gptoss-120b scinex **323 is
  inflated** by duplicated reply lines in `strabismus2026` (§30.8 incident); true count 312–323.
- **Open for Sam:** base model; judge + Q&A models and the spend; `--verdict-filter` (the only C1/C4
  verdicts are Claude's, so it defaults OFF); `configures` (defaults to excluded from training).

### STATUS (2026-09-08, Session 31) — ⛔ n≈40 IS THE BINDING CONSTRAINT. READ `hands_off.md` §31.

- **⛔⛔ AT n≈40, A FRESH DRAW MOVES SCORES BY UP TO 10 POINTS AND CAN FLIP A RANKING.** Applying the
  junk filter (0.7% of triples) and redrawing the samples moved C1/C3/C4 by up to **+10.3 pts** and
  **reversed the C1 top pair** (gemma4-31b 93.8 now leads gptoss-120b 88.8). This is sampling
  variance, not the filter. **Any gap under ~10 points between two models is NOT distinguishable.**
  Never quote a C1 ordering; C4's ranking survives because its spread is ~45 pts.
- **FINAL NUMBERS (20 papers, post-filter, post-redraw).** CEO — gptoss-120b C1 88.8 / C3 80.0 /
  **C4 86.2**; gemma4-31b **93.8** / 80.0 / 73.8; qwen3-235b 75.0 / 60.0 / 47.5; ministral-14b
  70.0 / 52.5 / 41.2. C2/C5/C6 unchanged (not redrawn). scinex (triples only) — gptoss **C4 90.0**,
  gemma4 59.0, ministral 46.2, qwen3-235b 46.2.
- **⭐ THE JUNK FILTER TOUCHED ONLY ONE MODEL.** 40 of 6,054 triples removed (0.7%), **all of them
  ministral-14b's**. The other three decline boilerplate unprompted — boilerplate mining is a
  property of the weakest extractor, not of extraction.
- **⚠ MOST scinex DELTAS ARE INSIDE THE NOISE BAND.** Only gemma4-31b's **−14.8** on C4 clearly is
  not. State the ontology result as **"no consistent effect detectable at this sample size"**.
- **⭐ `configures` HAS THREE CORRECT USES, ALL ministral-14b's** (`hands_off.md` §31.6) — so the
  direction is not unlearnable, just inconsistent. User has said to IGNORE the ontology question;
  evidence preserved, nothing changed.
- **8B SETTLED:** no Gemma 8B was ever in the line-up (it is **Gemma-4-31B**). `qwen3-8b-v2` DID run
  everything — 20/20 papers on BOTH ontologies with stored replies — and could be judged **for free**
  as a fifth capability point. Outside the fixed line-up, so not done.
- **⛔ FOUR GUARD GAPS STILL OPEN** (`hands_off.md` §31.5): a third author-bio form, a third
  bibliography form, CRediT lines split by OCR (`re view`), and `splitFrom` when the SUBJECT is the
  partition word. **Batch them into ONE guard round** — patching mid-measurement is what forced this
  session's 146-judgement redraw.
- **`ontology_eval.py collect` is now INDEX-AWARE** — it used to let verdicts for no-longer-sampled
  triples survive into `verdicts.csv`. It now drops them, says how many, and warns about unjudged
  sampled items. `show_unjudged.py` prints only what still needs grading after a redraw.

### STATUS (2026-09-07, Session 30) — ⛔ CORPUS IS 20 PAPERS, NOT 22. Qwen3-14B retired.

> ### ⛔⛔ THE CORPUS IS THE USER'S 20-PAPER LIST
> `aiabstract2025` and `routepred2023` were **SUBSTITUTES** (see their `note` column in
> `papers/manifest.csv`) added only while `tripplanner2020` and `linkpred2015` were unobtainable.
> The user supplied those two PDFs in Session 27, so **the substitutes are RETIRED**. Keeping both
> is what produced the wrong "22 papers" framing. The corpus **includes** `tripplanner2020` and
> `linkpred2015` and **excludes** `aiabstract2025` and `routepred2023`.
> Canonical id list: **`papers_20.txt`**. Regenerate it by excluding manifest rows whose note starts
> with `SUBSTITUTE for` — **do NOT filter on the substring "SUBSTITUTE"**, because the two recovered
> papers say "SUBSTITUTED" in their own notes and a naive filter silently drops them too (it did,
> and produced a list of 18). **Never write "22 papers" again.**

- **⛔ QWEN3-14B IS RETIRED (user instruction).** Do not plan or reference a VM re-run with it; that
  item is struck from `TASK.md`. Work continues with the Session 27 line-up only: **qwen3-235b,
  gptoss-120b, gemma4-31b, ministral-14b**.
- **⛔ Judging is IN-SESSION by claude-opus-5. Never an OpenRouter judge.** OpenRouter credit is for
  EXTRACTION only. C1–C6 is **single-rater by design** (Session 29) — a stated limitation.
- **⚠ HOW FAR THE 22-PAPER FRAMING LEAKED INTO C1–C6** (join `verdicts.csv` × `index.csv` — the
  paper id is in **index.csv**; `verdicts.csv` has no `paper` column for the triples harness, and
  filtering on it silently reports a false "0 of 40"):
  triples **1–3 of 40** per model (gemma4 worst at 3), paragraphs 0–2 of 16, **C6 2 of 6 in every
  model**. **Restricting the scores to the 20 papers moves them by ≤4.5 points and changes no
  ordering** — C1 78.8→79.5 / 88.8→89.7 / 85.0→85.1 / 63.7→65.4, C4 45.0→44.9 / 82.5→**84.6** /
  62.5→63.5 / 38.8→39.7. **So C1–C5 stand as measured.**
- **✅ C6 REDRAWN AND RE-JUDGED on the 20 papers (2026-09-07):** qwen3-235b **50.0%**, gptoss-120b
  **25.0%**, gemma4-31b **33.3%** (was 16.7%), ministral-14b **8.3%** (was 0.0%).
  **⛔ THE MONOTONICITY CLAIM IS WITHDRAWN** — by volume 367→50.0%, 547→25.0%, **615→33.3%**,
  1,548→8.3%: gemma4 has more triples than gptoss and a HIGHER C6. **Spearman ρ = −0.80, not −1.00.**
  Say "C6 falls as volume rises (ρ = −0.80), though not strictly monotonically". With four models one
  swap moves ρ by 0.20, so claim the direction, never the ordering.
- **⚠ The vacuous-pass artifact is now visible in the data:** qwen3-235b's redrawn C6 sample holds
  **two zero-triple papers**, both scoring the maximum 1.0 — 2 of its 6 points earned by extracting
  nothing, from the model that fires on only 16.4% of paragraphs. **Never quote C6 without C2/C5.**
- **`ontology_eval` now defaults to a corpus allowlist** (`papers_20.txt`); the retired substitutes
  cannot be redrawn by forgetting a flag. `--papers-file ''` restores the old scan-everything behaviour.
- **▶ scinex extraction RUNNING** — `run_scinex_20.sh` → `scinex_20_extraction.log`, three models
  (`gptoss-120b`, `gemma4-31b`, `ministral-14b`) over `papers_20.txt`. `relation_scinex/qwen3-235b`
  already existed (600 triples). OpenRouter at launch: $1.2725 of $3.00 used.
- **⚠ CONFOUND TO CLOSE BEFORE ANY CEO-vs-scinex CLAIM:** the CEO runs predate the Session 29 guards
  and the scinex runs have them active. **Re-ingest CEO from the stored `responses.jsonl`** (all 22
  papers × 4 models have them) — no new model calls, puts both ontologies on one code version.
- **⭐⭐ SCINEX JUDGED (triples harness, 3 of 4 models) — §26.16's CAPABILITY STORY DOES NOT
  REPLICATE.** CEO→scinex deltas by CEO capability: ministral-14b (C4 39.7%) **+5.9/+6.1/+5.3**;
  qwen3-235b (44.9%) **−7.0/−4.0/+1.4**; gemma4-31b (63.5%) +3.3/+6.5/**−4.5**. No monotone relation
  with capability in either direction — the weakest model gains most. **Say "the ontology effect is
  model-specific and does not order by extraction quality."** ⚠ Not a refutation of §26.16: n≈40,
  and that result was WITHIN one family (qwen3:8b vs 235b) on a different rubric, while these are
  three families — family and ontology effects are confounded. Volume ratios also fail to track
  capability (qwen3-235b 1.53x, ministral 1.01x, gemma4 **0.85x**). `hands_off.md` §30.12–30.13.
- **⛔ `_schema_for` SILENTLY GAVE SCINEX THE CEO SCHEMA** — it imported a non-existent
  `_load_scinex_schema` and a bare `except` swallowed it, so scinex bundles printed "(predicate not
  in this ontology)" and C4 would have degraded to a plausibility check. Fixed to use
  `ontology_loader.load_ontology` **and to raise rather than fall back**. Caught only by reading a
  bundle before judging. Also: `triple_id` hashes only (slug, s, p, o), so one triple from two
  sentences dedupes on collect — gemma4 scinex came back n=39, not 40.
- **⛔ GUARD GAP: `_is_provenance_sentence` misses SENTENCE-level boilerplate inside legitimate
  sections** — a **fictional UX persona** (`Persona: Dan is a senior salesman…` → `(Dan)
  affiliatedWith (a company in Europe)`), CRediT lines, declarations, data-availability statements,
  affiliation blocks, **OCR table debris** as a subject, and author bios lacking "degree"/"currently".
  All five correctly-typed `employs` triples in the study come from author biographies — the one
  relation models get right is the one whose evidence should have been excluded. `hands_off.md` §30.14.
- **⛔⛔ TaskStop DOES NOT KILL THE PYTHON CHILD.** Stopping the sequential scinex runner left its
  `run_api_extract.py` alive; the orphan walked to the next model and **two processes wrote the same
  `responses.jsonl`**, mangling 13 lines and double-spending. `ps` under Git Bash shows **nothing**
  even while they run — check with PowerShell `Get-CimInstance Win32_Process` grouped by
  `--model-slug` (a real runner shows **2** entries, so >2 means duplicates). **Never background a
  script that loops over models — one background task per model.** Signature of the corruption: a
  `.jsonl` line starting mid-sentence with no `{"para_id":` prefix. Full write-up: `hands_off.md` §30.8.
- **Local Gemma:** the user noted a Gemma could run locally. **Not actioned** — `ollama list` shows
  only `llama3.1:8b` and `qwen3:8b`, and `gemma-4-31b` will not fit the 8 GB laptop GPU. Needs one
  clarifying answer before pulling several GB.

### STATUS (2026-09-07, Session 29) — guards from C1-C6 built; C1-C6 declared single-rater

- **⛔ C1-C6 IS SINGLE-RATER BY USER DECISION. Do not re-propose a second judge or κ.**
  claude-opus-5 in-session is the judge of record for all 248 judgements. Report it as a **stated
  limitation**, never as an unfinished item.
- **✅ TWO NEW GUARDS in `kg_extraction/fixed_extractor.py`** (`hands_off.md` §29.2):
  **`_is_provenance_sentence()`** rejects a triple whose SOURCE SENTENCE is publisher/biographical/
  bibliographic matter (IEEE licence watermarks, author bios, acknowledgements, bibliography entries,
  ACM CCS blocks, ISSN/masthead lines). Keyed on the SENTENCE, not the section name, because the
  parser glues watermarks and running headers into legitimate body paragraphs.
  **`_splitfrom_is_inverted()`** rejects a `splitFrom` whose object names a partition and whose
  subject does not — both sides are Datasets, so a type check cannot catch the inversion.
- **Measured over the whole corpus:** removes **2.0%** of qwen3-235b, **0.9%** of gptoss-120b,
  **1.5%** of gemma4-31b, **1.7%** of ministral-14b. Every removal eyeballed; no false positives.
  ⚠ Far lower than the C1-C6 sample implied — that sample is stratified over predicates and
  over-represents rare ones. Same sampled-vs-weighted gap as the old track. **State which you quote.**
- **▶ OPEN, needs a HUMAN DECISION: the `configures` direction.** Defined
  `ExperimentalSpecification → Experiment`; **every model writes `Model → setting-value`**, wrong in
  100% of observed uses. Deliberately NOT guarded — unlike `splitFrom` this is not a simple
  inversion (the object is a VALUE, not an Experiment), so swapping arguments yields nothing valid.
  Either **(a)** change the ontology to `Model/Experiment → ExperimentalSpecification`, or **(b)**
  reject the inverted form and lose real hyperparameter facts. Evidence favours (a). See §29.5.
- **⛔ TOOLING TRAP, SECOND OCCURRENCE:** writing regex source through a Python heredoc in the Bash
  tool **collapses backslashes** — a word-boundary escape became a literal 0x08 backspace byte and
  killed 15 anchors silently (patterns still compiled). **Build backslashes with `chr(92)` and verify
  with `repr(open(f,'rb').read())`, never with grep** — a terminal renders 0x08 by erasing the
  preceding character. Also: a global `_re.IGNORECASE` makes `[a-z-]` match uppercase, which disables
  lowercase-only lookarounds; scope it with `(?i:...)` instead.
- **⚠ Every number on disk predates these two guards**, as with every previous guard round.
- **▶ NEXT:** scinex side of the four models (**needs OpenRouter spend ~$0.9 of $1.73 — the user
  asked to discuss before committing any**); regenerate `RESULTS_REPORT.md` + add a C1-C6 section to
  `build_results_report.py`; the `configures` decision.

### STATUS (2026-09-05, Session 28) — ⭐⭐⭐ C1-C6 JUDGING COMPLETE. READ `hands_off.md` §28 FIRST

> ### ⛔⛔ HARD CONSTRAINT STILL IN FORCE: NEVER SPEND OPENROUTER CREDIT ON CLAUDE
> Judging is done **BY THE ASSISTANT IN-SESSION** (read batch file, write reply file). `anthropic/*`
> models are listed on OpenRouter — never call them. OpenRouter credit is for EXTRACTION only.
> **$1.27 spent of $3.00.**

- **✅ ALL 40 BATCHES JUDGED — 248 judgements** (40 triples + 16 paragraphs + 6 papers × 4 models),
  the full sample sizes the user requested. Verdicts in
  `ontology_eval/<slug>/ceo/{triples,paragraphs,papers}/verdicts.csv`. Rebuild the table with
  `python3 ontology_eval.py report --model qwen3-235b gptoss-120b gemma4-31b ministral-14b`.
  Tables + findings: **`results.md` Table 10 and Findings 1–6**; narrative in `hands_off.md` §28.

| extractor | triples | para coverage | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|
| qwen3-235b | 395 | 16.4% | 78.8% | 31.2% | 60.0% | 45.0% | 28.1% | **50.0%** |
| **gptoss-120b** | 583 | 42.9% | **88.8%** | **34.4%** | **75.0%** | **82.5%** | 31.2% | 25.0% |
| gemma4-31b | 684 | 51.7% | 85.0% | 28.1% | 72.5% | 62.5% | 25.0% | 16.7% |
| ministral-14b | **1,709** | **69.1%** | 63.7% | 31.2% | 55.0% | 38.8% | **34.4%** | **0.0%** |

- **⭐ MODELS DIFFER ON TYPING, NOT CONCEPT IDENTIFICATION.** C1 spans 1.4× (63.7–88.8); **C4 spans
  2.1× (38.8–82.5)**. The concepts extracted are mostly real and present in the sentence — what
  separates extractors is domain/range conformance. The old single rubric hid this entirely.
- **⭐⭐⭐ VOLUME DOES NOT BUY COMPLETENESS — the headline finding.** ministral-14b makes **4.3×**
  qwen3-235b's triples and fires on **4.2×** more paragraphs, yet **C2 is identical (31.2% vs 31.2%)**
  and C5 differs by only +6.3 pts. All four models sit in a **25–34% band** on both completeness
  criteria. The extra volume restates captured facts, enumerates pairwise combinations and mines
  boilerplate; it does not reach the missed paragraphs. Every extractor leaves ~2/3 of each paper
  unrecorded — and they leave *different* thirds.
- **⭐⭐ C6 FALLS MONOTONICALLY WITH VOLUME, perfect rank correlation:** 395 triples → 50.0%,
  583 → 25.0%, 684 → 16.7%, 1,709 → **0.0% (all 6 papers CONTRADICTORY)**. Three mechanisms:
  superlatives with the scope qualifier stripped, metric nodes bound to two values because the class
  label was dropped, and surface-variant splitting letting incompatible claims attach to one node.
  Clearest case: gemma4 asserts `(VGG16) achieves (F1 w-avg (0.23))` **and** `(VGG16) achieves (over
  90% accuracy)`. Clearest structural violation: ministral's `(route prediction model) comprises
  (route prediction model)`.
- **⚠ C6 SCORING ARTIFACT — never report C6 alone.** An empty triple set cannot contradict itself, so
  it scores CONSISTENT (1.0) **vacuously**; qwen3-235b's `textaug2023` (0 triples) is 1 of the 6
  papers behind its 50.0%. **C6 rewards extracting nothing** and is only interpretable beside C2/C5.
- **⭐ SYSTEMATIC PREDICATE FAILURES (guard worklist).** `configures` — **direction wrong in 100% of
  observed uses**, every model writes Model→setting instead of Specification→Experiment; `employs`
  (Organisation→Person) — matched on the English verb, and **the only 2 correct uses in 160 triples
  came from author biographies**; `splitFrom` — inverted by 3 of 4 models (gptoss is the exception);
  `evaluatedOn` — hardware, other models and evaluation protocols in the Dataset slot.
- **⭐ BOILERPLATE IS MINED AS CONTENT AND C1–C4 CANNOT SEE IT.** Triples scoring CORRECT on all three
  per-triple criteria came from author bios, IEEE download watermarks, bibliographies,
  acknowledgements and the ACM CCS block. The §26 boilerplate guard covers CRediT/funding/declarations
  but not these. **A triple can be perfect on C1/C3/C4 and still be worthless** — the criteria never
  ask whether the source sentence belonged to the paper.
- **✅ DECIDED 2026-09-07 (user): C1–C6 IS SINGLE-RATER BY CHOICE.** claude-opus-5 in-session is
  the judge of record for all 248 judgements; **no second labeller and no κ.** Report it as a
  *stated limitation*, never as an unfinished item, and do not re-propose a second judge. The
  old-rubric κ (results.md Table 9) measures a different rubric and does not transfer.
- **▶ NEXT, in priority order:** (1) extend the guards to the boilerplate classes above + a `configures`/`splitFrom` direction fix;
  (3) run the **scinex** side of these four extractors (CEO only so far); (4) regenerate
  `RESULTS_REPORT.md` and add a C1–C6 section to `build_results_report.py` — now unblocked.
- **⚠ n=6 papers for C6.** Quote the monotone trend across four models (24 papers), not per-model
  point estimates. n=40 triples / 16 paragraphs per model for the other criteria. Single rater
  throughout — by design (see above).
- Judging conventions used (C2/C5 thresholds, C1-vs-C3 split, C4 PARTIAL-vs-INCORRECT,
  C6 CONTRADICTORY bar) are recorded in `hands_off.md` §28.7 — **reuse them if the sample is
  extended**, or the rounds will not be comparable.

### STATUS (2026-09-05, Session 27) — ⭐ READ `hands_off.md` §27 FIRST

> ### ⛔⛔ HARD CONSTRAINT: NEVER SPEND OPENROUTER CREDIT ON CLAUDE
> User, verbatim: *"claude opus 5 that we will run as a judge will be through my claude. NOT FROM
> OPEN ROUTER. DO NOT SPEND ANY MONEY OF OPENROUTER ON CLAUDE."* `anthropic/*` models are listed on
> OpenRouter — never call them. **Judging is done BY THE ASSISTANT IN-SESSION**: read the batch file,
> write the reply file. OpenRouter credit is for EXTRACTION only. **$1.27 spent of $3.00.**

- **⭐⭐⭐ THE EVALUATION CRITERIA CHANGED.** The CORRECT/PARTIAL/INCORRECT rubric is **superseded**
  by a six-criterion framework the user supplied (C1 Concept Correctness · C2 Concept Completeness ·
  C3 Concept Specificity · C4 Relation Correctness · C5 Relation Completeness · C6 Semantic
  Consistency; refs Zhang/Conia/Rago IJCNLP-AACL 2025 and Wilson et al. Semantic Web 14(6) 2023).
  New harness: **`ontology_eval.py`** with `triples` / `paragraphs` / `papers` / `collect` / `report`.
- **⭐ THREE harnesses, because the criteria have different units of analysis.** C1/C3/C4 per TRIPLE
  (vs its source sentence), C2/C5 per PARAGRAPH (vs all triples drawn from it), C6 per PAPER (vs its
  whole triple set). A completeness question cannot be asked of one triple — what is missing is not
  in front of the judge. **Everything measured before Session 27 was precision only; C2/C5 are a new
  axis.**
- **⭐ MODEL LINE-UP FIXED BY THE USER:** extract with **Qwen3-235B · GPT-OSS-120B · Gemma-4-31B ·
  Ministral-14B** (OpenRouter), judge all with **claude-opus-5 in-session**. Correct ids:
  `qwen/qwen3-235b-a22b-2507`, `openai/gpt-oss-120b`, **`google/gemma-4-31b-it` (31B, not 32B)**,
  `mistralai/ministral-14b-2512`.
- **⭐ CORPUS IS NOW 22 PAPERS.** The user supplied the two previously-unobtainable PDFs; both
  verified, parsed clean, and extracted. So it is **the 20 papers from the original list + the 2
  substitutes** (`aiabstract2025`, `routepred2023`). Never write plain "20 papers".
- **⚠ `relation/qwen3-8b` (v1) IS FROZEN AT 20 PAPERS ON PURPOSE** — it is the "before" half of the
  §26.14 engineering result (+8.3 pts). Extending it needs the *current* code, which is what that
  baseline predates. Report it as a paired 20-paper comparison against v2.
- **⭐ A RESULT THAT EXISTS BEFORE ANY JUDGING — paragraph coverage varies 4x and runs OPPOSITE to
  precision:** qwen3-235b produces a triple from only **16.4%** of the 567 paragraphs, gptoss-120b
  42.9%, gemma4-31b 51.7%, ministral-14b **69.1%**. **The precise models are precise partly by
  declining to extract at all** — which is exactly what C2/C5 exist to price.
- **⭐ FIRST PARTIAL C1-C6 NUMBERS (qwen3-235b, n=10): C1 75.0% · C3 60.0% · C4 45.0%.** C4 is far
  weaker than C1 — the concepts are real, the **typing** fails. Recurring modes: direction reversed
  (`RT-DETR uses object detection` from "object detection USING RT-DETR"), domain/range violation
  (`dataset comprises video footage`; `RT-DETR employs ...` when `employs` is Organisation->Person),
  role confusion (`performance evaluates BERT's F1 score`). The old single rubric hid all of this.
- **▶ JUDGING IS 3 OF 40 BATCHES DONE.** Bundles for all 4 models x 3 harnesses are **already
  generated** under `ontology_eval/<slug>/ceo/`. Resume at
  `ontology_eval/qwen3-235b/ceo/triples/04_batch_04.txt`. Full sample sizes were requested:
  40 triples + 16 paragraphs + 6 papers per model.
- **⚠ `RESULTS_REPORT.md` IS STALE** — it describes the 20-paper corpus and the old rubric and knows
  nothing of the 4 new extractors or C1-C6. Regenerate after the C1-C6 numbers exist and add a
  C1-C6 section to `build_results_report.py`.

### STATUS (2026-08-28, Session 26) — read `hands_off.md` §26 first
- **⭐ `RESULTS_REPORT.md` is the self-contained write-up** (project overview → pipeline → both
  ontologies → both evaluation tracks → results → methodology → negative results → limitations →
  reproduction). **Generated** by `build_results_report.py` from `output/` + `gold/` — re-run it
  after any new result rather than editing the markdown. Hand THIS to the paper/slide writer.
- **⛔ Two routes are dead, do not re-attempt:** free chat-UI extraction (ChatGPT/Gemini returned
  nothing usable; bundles in `chat_upload/` are abandoned) and free-tier APIs *for extraction*
  (Groq 8k tok/min → ~8 h; Gemini 20 req/day/model). Those same tiers are **fine for judging**.
- **⭐⭐⭐ HEADLINE — both extractors under ONE judge (`gemini-3.6-flash`), identical paragraphs /
  prompt / guards, only the model differs: claude-opus-5 CEO **89.1% weighted strict** [82.3, 94.5],
  97.9% lenient (n=100, 86 CORRECT / 13 PARTIAL / 1 INCORRECT) vs qwen3:8b CEO **25.6%** (n=58) and
  scinex **11.7%** (n=52).** ~3.5x. Every predicate at 0% strict for qwen scores >=66% for Claude or
  **Claude never emits it** (`affiliatedWith`, `employs`, `configures`, `publishedIn`, `cites` are
  absent from its output). With Table 7 this is the paper's story: the stronger model trades volume
  for precision, and the volume it declines is the volume that was wrong. `results.md` Table 8.
- **⭐⭐ THE HEADLINE SURVIVES A SECOND INDEPENDENT JUDGE.** `openai/gpt-oss-120b` (OpenAI family,
  via Groq) re-judged the same 100 Claude triples from scratch: **82.8% weighted strict / 92.6%
  lenient**, vs gemini-3.6-flash's 89.0% / 97.9% — **sampled precision 85.0% vs 86.0%, near-identical**.
  Report the headline as a **range, 82.8-89.0%**, not a point estimate. Agreement between the two
  judges on Claude's triples: 86.0% raw, **κ 0.457 (3-class) / 0.637 binary (substantial)** — much
  higher than the 0.342/0.480 on qwen's triples. **Judges agree more about good extraction than bad;
  low κ is itself a symptom of poor extraction.** ⚠ **Groq was wrongly written off earlier** — its
  8k tok/min kills *extraction* (~6.2k/call) but not *judging* (~1.4k/call); use `--sleep 40` since
  Groq reserves `input + max_tokens` against the TPM budget.
- **⭐⭐⭐ ONTOLOGY x CAPABILITY — THE RANKING REVERSES. RQ2 has a real answer now.**
  All four cells n=100, same guards/code/judge (`gpt-oss-120b`):
  **8B: CEO 38.1% vs scinex 12.1% (−26.0 pts, CEO wins) · 235B: CEO 48.8% vs scinex 54.7%
  (+5.9 pts, scinex wins) → interaction +31.9 pts.** Volume agrees (scinex/CEO ratio 0.94x at 8B,
  **1.48x** at 235B) so it is not precision bought with recall — at 235B scinex wins on BOTH axes.
  **⚠ The old "CEO vs scinex is a tie" is WITHDRAWN** — every measurement behind it was on qwen3:8b,
  where the schema is not the binding constraint, which is also why it flipped between gold rounds.
  Mechanism: scinex is richer (27 relations vs 22, tighter domain/range) — more surface area to get
  wrong for a weak model, an asset for a capable one. **Claim the SIGN and the interaction, not the
  8B magnitude** (−11.5 pre-guards vs −26.0 post-guards).
- **⭐⭐ PIPELINE ENGINEERING IS A SECOND, INDEPENDENT LEVER — +8.3 pts on a FIXED model.**
  `qwen3-8b-v2` = same local Ollama build, same paragraphs, only the code differs (3 guards + 2 bug
  fixes): strict **29.8% → 38.1%**, lenient 64.0% → 66.1%, **and volume UP 1,185 → 1,357 (+14.5%)** —
  not a precision/recall trade. Per-predicate movement matches each fix: `evaluates` 13→**48**
  (+269%, numeric-object bug), `comparesAgainst` +53%, `comprises` +50%; `affiliatedWith` 29→**13**
  (−55%) and `employs` −38% (boilerplate guard), `evaluatedOn` −27% (cross-ref guard).
  **Two additive levers: model choice 29.8→82.8, engineering 29.8→38.1.**
- **⛔⛔ GENERALISED FAILURE CLASS — has bitten 3×: a reasoning model's token budget consumed by
  REASONING yields EMPTY content that is indistinguishable from "found nothing".** Ollama returns
  reasoning in a separate `thinking` field; `run_api_extract`'s adapter read only `content`, so
  qwen3:8b returned **93/515 empty replies** and a 7-hour run had to be discarded (it nearly went
  into the report as "the guards removed 58% of triples"). `ollama_backend.py:91` had always set
  `"think": False`; the API adapter now does too, with a `thinking`-field fallback. Also seen on
  `gpt-oss-120b` at `max_tokens=64`. **When a model "finds nothing", check for empty content +
  populated reasoning BEFORE believing it.**
- **⭐⭐⭐ THE CAPABILITY CURVE — four extractors, one judge (`openai/gpt-oss-120b`), one prompt,
  one guard set.** gemma-3-12b **29.2%** · qwen3:8b **29.8%** · qwen3-235b **48.8%** ·
  claude-opus-5 **82.8%** weighted strict. Precision and volume are inversely ordered without
  exception (3.47 → 0.56 triples/paragraph; junk-predicate share 20.1% → 0.7%).
  **✅ THE CONFOUND IS CLOSED:** qwen3:8b vs qwen3-235b holds family/vendor/tokenizer fixed and
  varies only scale → 29.8% → 48.8%. **⛔ But parameter count does NOT transfer across families:**
  gemma-3-12b is 50% larger than qwen3:8b and no better (29.2%), with the worst junk share of all.
  Write "more capable extractor", never "bigger model". ⚠ The 2 new runs saw 515 paragraphs vs 525
  (the boilerplate guard landed in between — that is its live validation, and it mildly helps them).
- **OpenRouter is now wired** (`--provider openai-compat --base-url https://openrouter.ai/api/v1`,
  `OPENROUTER_API_KEY` in `.env`). ~419 models, **$0.34 of $3 for two full corpus runs**.
  ⚠ Two bugs fixed: key resolution was preference-ordered so it sent the **Groq key to OpenRouter**
  (now endpoint-aware via `_HOST_KEYS`); and `claude_corpus_report.py --out` defaults to Claude's
  corpus JSON regardless of `--model-slug`, so it **clobbers** it — always pass `--out`.
- **⭐⭐ COMPLETE SINGLE-JUDGE COMPARISON (quote this one).** `gpt-oss-120b` judged the FULL sample of
  both extractors: **claude-opus-5 CEO 82.8%** (n=100) vs **qwen3:8b CEO 29.8%** (n=114) /
  **scinex 18.3%** (n=126) weighted strict. No partial-sample caveat — supersedes the quota-capped
  gemini rows (n=58/52). Both judges independently put Claude at ~3x qwen.
- **⛔ A κ CLAIM IS WITHDRAWN.** "Judges agree more about good extraction than bad" compared two
  different judge PAIRS (claude-vs-gemini on qwen, gemini-vs-gpt-oss on Claude) — a confound. With
  the pair held constant: raw agreement IS higher on Claude's triples (86.0% vs 75.5%) but **κ is
  LOWER** (0.457 vs 0.552) — the **kappa paradox**, because Claude's triples are 81/100 joint-CORRECT
  so skewed marginals inflate chance agreement. **κ is not comparable across samples with different
  class balance.** Also: **Claude was the outlier labeller** (24.5% vs gemini 13.6% / gpt-oss 14.5%);
  the two non-Claude judges agree far more with each other (0.552) than with Claude (0.342), so
  prefer the gemini/gpt-oss figures.
- **⭐ FIRST COHEN'S κ IN THE PROJECT.** Claude and Gemini independently labelled the same 110 qwen
  triples: raw agreement 59.1%, **κ 0.342 (3-class)**; CORRECT-vs-not 83.6%, **κ 0.480 (binary)**.
  Disagreement is almost entirely the PARTIAL boundary (Claude PARTIAL → Gemini INCORRECT 25x; joint
  INCORRECT 44; Gemini never promoted a Claude-INCORRECT to CORRECT) — **the empirical case for
  reporting strict AND lenient.** Gemini is the *harsher* judge, which makes the 89.1% more robust,
  not less. `results.md` Table 9.
- **`run_api_judge.py` (new)** drives the existing `judge_paste.py` bundles over HTTP — 100 triples
  judged in 2.3 min instead of a day of pasting. The rubric is the **system prompt on every call**, so
  batches cannot drift the way a long chat conversation does.
- **⛔ FREE-TIER LIMITS, measured (do not re-derive): Gemini = 20 requests/day/MODEL** (judging is
  fine at 10-25 triples/call; extraction at 486 calls is impossible). **Groq = 8,000 tokens/minute**
  → our ~6.2k-token calls run ~1/min → ~8 h for the corpus, **slower than the local GPU**. Gemini's
  quota is per model, so a fresh pinned id grants another 20 — but never judge one sample with two
  different models. Also: **Groq 403s Python's default user-agent** (Cloudflare `error code: 1010`),
  fixed with a `USER_AGENT` header; and **`gemini-2.5-flash` is listed to new keys but returns
  NOT_FOUND when called** — the model list is not a list of *usable* models, so always smoke-test.
- **CORPUS IS 20, BY SUBSTITUTION.** `tripplanner2020` / `linkpred2015` are unobtainable (paywalled
  Springer; **verified NOT in `bulk-download/`**). Per the user, two unused same-group papers were
  substituted: **`aiabstract2025`** (ISCON 2025, 9 sections / 4,160 words / **171** entities) and
  **`routepred2023`** (ICCAE 2023, 7 sections / 2,865 words / 41 entities). Both parsed clean,
  titles verified, BERT regression unchanged. `papers/manifest.csv` records the substitution.
  **⚠ Report it as "18 of the listed papers + 2 substitutes", never plain "20 papers".**
- **✅ CORPUS COMPLETE AT 20/20 FOR BOTH MODELS.** `relation/qwen3-8b` **1,185** triples ·
  `relation_scinex/qwen3-8b` **1,117** · `relation/claude-opus-5` **296**, all over 20 papers /
  525 paragraphs. The **25% volume ratio held on 39 paragraphs neither model had seen** (0.56 vs
  2.26 triples/paragraph), so the volume gap is a property of the models, not of the paper selection.
  All 17 of Claude's new triples passed the guards with zero rejections. qwen truncation on the 2 new
  papers was 14.1% (vs 5.2% on the original 18) — denser paragraphs hit the 3,072-token cap; settings
  were deliberately left identical to the other 18.
- **⛔ The free-chat-UI route is ABANDONED.** ChatGPT and Gemini produced no usable triples;
  `chat_upload/{gpt,gemini}/replies/` are empty. Do not restart it — §25.4d already limited it to a
  sanity check (batching changes the experimental condition; the model version is not pinnable).
  **The API route replaces it** and is strictly better: pinned model id, one paragraph per call
  (genuinely comparable to the qwen3:8b / claude-opus-5 runs), no pasting.
- **✅ TWO PIPELINE BUGS FIXED AND MEASURED.** `kg_builder._is_valid_entity()` now takes
  `(text, predicate, role)` and admits a measurement value (`88.5`, `0.923`, `97%`) as the **object of
  a value predicate** (`evaluates`/`achieves`/`reports`) only, plus all-caps acronyms under the
  `len < 3` floor. Re-ingesting Claude's stored replies (no new model calls) took it
  **268 → 279 triples**, `evaluates` **1 → 6**. **⚠ `evaluates`' 0%-strict score was an artefact of
  this bug** — the correct instances were deleted and only malformed ones were judged; it must be
  **re-judged**. **qwen3:8b's 1,101 cannot be re-ingested** (its `kg_main` run stored no raw replies),
  so the two columns are no longer strictly like-for-like — state that when reporting.
- **▶ IN FLIGHT: llama3.1:8b (Meta) over the full corpus** — a third extractor family, no key needed.
  `run_api_extract.py --all --provider ollama --model llama3.1:8b --model-slug llama31-8b
  --extractor relation --ontology ceo --max-tokens 3072 --ingest` → `llama31_extraction.log`.
  ~3 calls/min, **~2.7 h**; resumable. Then `claude_corpus_report.py --model-slug llama31-8b`.
- **Fast lane for more models — `check_provider.py` (new).** `python3 check_provider.py groq|gemini`
  validates the key, **looks up** the model ids the key can see (they drift), and prints the exact
  smoke-test + corpus + report commands. `run_api_extract.py` now loads `.env` and resolves
  `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` / … so several providers coexist.
  **Groq hosts `openai/gpt-oss-120b`** — a pinned OpenAI-family model, the defensible replacement for
  the failed ChatGPT run — and does the corpus in well under an hour on the free tier.
- **Still waiting on Sam:** the 2 missing PDFs → `papers/tripplanner2020.pdf` +
  `papers/linkpred2015.pdf` (corpus is 18/20). The `judge_upload/` bundles are also unreturned; with
  3-4 extractor families now on disk, **one judge pass covering all of them is worth more than
  another extraction run**. Everything else below is still current.

### STATUS (2026-08-27) — 18/20 papers parsed + enriched + planned; extraction is the only thing left
- **Corpus = 18 of the user's 20 papers on disk and fully prepared.** 7 originally OA, 8 recovered
  from a user-supplied `bulk-download/` folder, 3 downloaded by hand. **Missing 2:** `tripplanner2020`
  and `linkpred2015` (both Springer). `bulk-download/` also holds 12 papers by the same group that are
  **NOT** in this corpus — verified against the user's own citation list; ignore them.
- **All 18 parsed (`--no-llm`) and title-verified against `papers/manifest.csv` — 18/18 correct.**
  Body words 1,985–9,147; body-capture 77–102% of raw PDF text. Enriched entity CSVs for all 18
  (35–169 entities). `entity_coverage.py` → **18/18 `fixed`** (`output/corpus_plan.json`).
- **⚠ The corpus now splits into two halves and MUST be reported that way.** The ~11 CS papers get
  genuinely paper-specific entities (`Text Classification`, `Video Segmentation`, `License Plate
  Recognition`, `Spatial Pyramid Pooling`) because CS-NER is annotated over CS/NLP work; the ~7
  non-CS papers (chemistry/materials/clinical) still get only generic ML vocabulary and **zero**
  domain terms. `fixed` vs `relation` should diverge sharply between the halves — that contrast is a
  better result than one corpus-wide precision number. See `hands_off.md` §22.4.
- **⛔ The laptop cannot run extraction.** Windows 11 **Smart App Control** is ON and blocks torch's
  unsigned DLLs (`OSError [WinError 4551]`); the laptop GPU is 8GB besides. Disabling Smart App
  Control is a **one-way** change — do not. **`scipy` is pinned at 1.17.1**: 1.18.1's DLLs get blocked
  too, which took the whole parser down mid-session (`hands_off.md` §22.3). Do not upgrade scipy, and
  do not install torch here.
- **LOCAL EXTRACTION WORKS via Ollama** (`hands_off.md` §23) — not torch, which is still blocked, but
  a signed Ollama binary: `winget install Ollama.Ollama --source winget`, `ollama pull qwen3:8b`, then
  `python run_local.py --all --extractor relation --both-ontologies`. `kg_extraction/ollama_backend.py`
  shims only the two attributes the extractors use to reach a model, so prompt/ontology/guards/output
  are unchanged. `run_local_judge.py` does the same for the LLM-as-Judge.
- **⚠ Three Ollama settings are load-bearing — do not re-derive them:** `num_ctx=8192` (Ollama
  defaults to 4096 regardless of model, and the system prompt alone is ~3.4k tokens; 16384 overflows
  the 8GB GPU and spills to CPU), `--max-new-tokens 3072` (at 1024, 34% of calls truncated and a
  truncated JSON yields ZERO triples), and **the model slug must not contain a colon** — `qwen3:8b` is
  an illegal Windows path, which silently discarded every triple at write time.
- **✅ RELATION EXTRACTION COMPLETE — 36/36 runs, 0 failures, 2,132 triples** (~5.5h, qwen3:8b via
  Ollama) → `output/<paper>/kg/relation{,_scinex}/qwen3-8b/`. CEO 1,101 / scinex 1,031 (ratio 0.94).
  **PRELIMINARY** — `qwen3:8b` is not Qwen3-14B; the paper's numbers still come from the VM.
- **⭐⭐ KEY RESULT — JUDGED, n=240 (`hands_off.md` §25, tables in `results.md`): CEO vs scinex is a
  TIE.** Claude labelled two disjoint stratified samples (80 → `gold/sample_claude.csv`, then 160 →
  `gold/sample_round2_claude.csv`). Corpus-weighted strict precision **CEO 34.7% vs scinex 39.6%**;
  bootstrap difference **−4.9 pts, 95% CI [−22.5, +13.9]**, `P(CEO>scinex)=0.30`.
  **⚠ Session 24's "CEO beats scinex (46.2 vs 34.4)" is WITHDRAWN** — that estimate rested on ~8
  labels covering half the CEO corpus mass (`addresses` + `uses`) and **reversed sign** on the
  disjoint round-2 draw (CEO 28.9 vs scinex 41.7). Reproduce every table with
  `python3 gold_report.py --labels gold/sample_claude.csv gold/sample_round2_claude.csv --bootstrap 2000`.
- **What survives the correction:** CEO keeps a **lenient** edge (66.8% vs 59.9% weighted) — its
  errors are on-topic but loosely typed, scinex's are more often flatly wrong. And **always state the
  weighting**: the stratified sample deliberately over-represents rare predicates (sampled strict is
  20.2% / 24.6%). The KGE/citation-prediction ontology tie is a separate axis and is unaffected.
- **Guard worklist — predicates at 0% strict in BOTH rounds and BOTH ontologies** (systematic, not
  noise): `employs` (defined `Organisation→Person`, used for method-uses-method), `evaluates`,
  `configures`, `evaluatedOn`, `trainedOn` (nearly all PARTIAL — generic objects), `cites`,
  `affiliatedWith` (mines CRediT author-contribution statements), `supports`.
  Good in both: `achieves` (CEO 6/6, scinex 4/6).
- **⛔ REAL BUG — schema placeholder leak, now confirmed twice:** `(Kalman filter, mentions,
  AcademicPaper)` in round 1 and `(Merck, affiliatedWith, Organisation)` in round 2 — the model copies
  an ontology CLASS NAME out of the prompt as an object value. Fix must reject **any** object matching
  an ontology class name, not just patch `mentions`. New sibling error class: cross-reference objects
  (`→ "Table III"`, `→ "Section II DATA PREPARATION"`, `→ "ASSE 2025"`), 6/240.
- **⛔ GUARD BUG FIXED 2026-08-27 — `_subject_in_sentence()` dropped every parenthetical subject.**
  It wrapped the subject in `\b…\b`; `\b` needs a word/non-word transition, so a subject ending in `)`
  (i.e. the standard `Full Name (ABBR)` form) never matched its own source sentence. Evidence: 2 of
  2,132 corpus triples have a subject ending in `)`. Now `_contains_term()` in `fixed_extractor.py`.
  **Recall-only fix; every number on disk predates it.**
- **⚠ n=240 sizes the bands, not the cells** (4–6 samples per predicate; ±13 pts on the weighted
  figures). No per-predicate percentage is quotable yet. **κ is still unmeasured** — no second
  labeller has scored the same ids; `gold/sample_round2.csv` is exported with blank verdicts for one.
- **⭐ Claude can also run as the EXTRACTOR, and the full corpus is done** — `claude_extract.py
  prompts|ingest` splits the extractor at the model boundary and reuses the real system prompt, the
  real `_parse_fixed_output()` guards and the real `KnowledgeGraphBuilder`, so only the model differs.
  **All 18 papers / 486 paragraphs, CEO relation mode: 268 triples vs qwen3:8b's 1,101** on identical
  input (`output/<model-slug>_relation_corpus.json`; per paper in `output/<id>/kg/relation/claude-opus-5/`).
  **The missing 76% is concentrated in the broken predicates:** the 11 relations scoring 0% strict hold
  **22.7% of qwen's corpus but only 6.0% of Claude's** (`affiliatedWith` 29→0, `employs` 26→0,
  `trainedOn` 54→1, `configures`/`publishedIn`/`cites` →0), while the two ≥50%-strict relations rise
  from 16.9% → 29.5%. CRediT/Funding/running-header paragraphs return `{"triples": []}` unprompted.
  **Not judged, and Claude must not judge them** — extractor and judge must be different models; use
  `run_local_judge.py`. Rebuild the comparison with `python3 claude_corpus_report.py`.
- **Any model can now be dropped into the same slot** — `run_api_extract.py` (2026-08-28) supplies
  the missing prompts→responses driver for anthropic / openai / openai-compat (OpenRouter, Together,
  DeepSeek, vLLM, LM Studio) / gemini / ollama, stdlib only. Results land under their own
  `--model-slug` directory and compare with `claude_corpus_report.py --model-slug <slug>`.
  Verified end to end against qwen3:8b over HTTP. Fixed while doing it: `cmd_ingest` crashed on any
  reply that was not clean JSON (fences, `<think>`, truncation) — Claude's never were, qwen3:8b's
  first one was.
- **⛔ SECOND PIPELINE BUG (found 2026-08-27, NOT fixed): `evaluates` can never be correct.**
  `kg_builder._is_valid_entity()` requires `[a-zA-Z]{2,}` in every node, so a bare numeric object is
  deleted *after* the guards pass — but the prompt's own GOOD example for `evaluates` is
  `(<Metric>, evaluates, 88.5)`. `(precision, evaluates, 0.923)`, `(RMSE, evaluates, 14.76)` etc. were
  produced and silently dropped; only malformed wordy ones survive to be judged, which is why
  `evaluates` scores 0%. The same filter's `len < 3` rule deletes acronym objects (`SF`, `SD`, `SP`).
  Fix = allow numerics for `evaluates`/`achieves` + exempt all-caps acronyms. Every number on disk
  predates this fix.
- **Defects in `relation` output, MEASURED at corpus scale** (`hands_off.md` §23.5) — smaller than the
  first single-paper glance suggested: subject fragmentation **6.2%** (`RT-DETR` / `Random Forest` /
  `Random Forest model`; `relation` has no canonicalisation step, but the variants collapse under
  "lowercase + strip a trailing type noun"), figure/table-reference objects **1.2%**, generic objects
  **0.4%** — 98.4% of objects are clean. Extraction quality tracks parse quality (videoseg2025 8.3
  triples/paragraph vs llamacorrupt2025 1.09).

### STATUS (2026-08-13) — NEW TASK: fixed-extraction accuracy, judged by LLM + validated against human labels
> **Full task brief: `TASK.md`** — goal, corpus, the fixed-vs-relation decision, the stage-by-stage commands, the verdict rubric, what to report, and a live checklist. Read it before starting work on this task.
- **Scope narrowed by the user: fixed extraction ONLY.** New corpus = **~20 handpicked research papers** (not the 315-paper ACL corpus — those results stand as they are). **Both ontologies extracted and compared** (`fixed` = CEO, `fixed_scinex` = scinex). Goal: push per-triple accuracy as high as possible, measured properly.
- **Method:** extract → LLM-as-Judge every triple → human-label a stratified sample → measure judge-vs-human agreement (Cohen's κ) → fix the prompt/guards where both agree the extractor is wrong → re-extract → re-measure. Judge model stays **Qwen2.5-7B-Instruct** for now; the user may add a second judge (a stronger local model, or Claude) later — `gold_eval.py agree --b` compares any two labellers, so swapping judges needs no new code.
- **Built this session (code only, NOT yet run):** `gold_eval.py` (new: export / agree / errors) + `kg_evaluate.py` upgrades (see Part 2c). **No Python on the Windows laptop** (Store stub only) and no local `output/` — everything runs on the VM; nothing here has been executed or syntax-checked. First VM step: `python3 -c "import gold_eval, kg_evaluate"`.
- **The corpus (given 2026-08-13): 20 papers by R. Chawuthai's group** — applied ML across ophthalmology, chemistry, materials, traffic/ITS, cloud systems, CV and NLP. Full list with DOIs + access status in `papers/manifest.csv`. **7 downloaded** by `fetch_corpus.sh` (Sci Rep ×3, Sensors, Applied Sciences, Sensors & Materials, Chem Eng Transactions); **13 need manual download** — 3 are open access but bot-blocked (IEEE Xplore ×2, ScienceDirect ×1), 10 are paywalled (IEEE conf ×5, Springer ×2, ACM ×1, Elsevier ×1). Paper id = PDF stem (e.g. `papers/strabismus2026.pdf` → `output/strabismus2026/`).
- **Domain-shift problem + the fallback the user chose:** CS-NER is annotated over CS/NLP papers, so the chemistry/materials/clinical papers will intersect the gazetteer thinly — `fixed` would then produce almost nothing for coverage reasons, not quality reasons. User's instruction: *try every entity source, and where there is none, fix only the relation.* → new `relation` extractor (above) + `entity_coverage.py` to make the per-paper call before any GPU time is spent.
- **Per-paper chain:** `main.py papers/<id>.pdf --no-llm` → `enrich_entity_csv.py --paper <id> --source csner` (works for any parsed paper, ACL id not required) → `entity_coverage.py --plan output/corpus_plan.json` → `kg_main.py --paper <id> --extractor {fixed|relation}` ×2 ontologies (`--ontology scinex` for the second).
- Untouched and still valid: all KGE/citation-prediction results (`results.md`), the global-graph work below, the 315-paper corpus.

### STATUS (2026-07-17) — persistent global KG implemented; VM smoke test is the next step
- **New direction (user):** move from per-paper isolated KGs toward ONE persistent global graph — paper big-nodes connected to each other by real citation edges, each carrying its fixed-extraction entities as augmented nodes. Eventually: masked-node link-prediction training (cut 1-2 big nodes, predict the link) + a citation-only ablation without augmented nodes. **This session built the foundation only** (see "Part 2b" in Architecture): `kg_extraction/global_graph.py` (incremental merge, stub resolution, connectivity report), `kg_main.py` wiring (auto-merge incl. `--skip-existing` backfill; flags `--global-graph-dir`/`--no-global-merge`/`--global-save-every`), and `expand_and_validate.py` (smoke test).
- **Verified locally (Windows): synthetic unit test** of merge/idempotency/stub-resolution/round-trip passed; CLIs import clean. **Live smoke test IN FLIGHT locally at session close:** `python expand_and_validate.py --seed D13-1109 --count 8 --model Qwen/Qwen2.5-7B-Instruct` (local GPU ≈7B ceiling; Qwen2.5-7B-Instruct is cached and works — the extractor's Qwen3 think-block handling no-ops on it; output isolated under `.../Qwen2.5-7B-Instruct/`). **Outcome unknown — check the connectivity report / `output/global_kg/fixed/Qwen2.5-7B-Instruct/` first thing next session.**
- **⚠ Seed choice lesson (hands_off §Session-18.3):** seed MUST be an ACL-Anthology-pulled paper (user instruction; NOT the manual "BERT" folder), and not every ACL seed works — `2020.acl-main.185` gave 0/8 resolvable neighbors (its citers are post-index arXiv papers, its refs mostly arXiv/ICLR). Pre-scan seeds offline against `acl_title_index.json`; best current seeds: **D13-1109** (10 new resolvable neighbors), W14-5318 (10), E14-4024 (9). Old-style-id seeds (P/D/N/W 11-19) resolve instantly from the local index.
- **After the smoke test passes:** full-corpus backfill via `kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B --skip-existing` (VM) populates `output/global_kg/fixed/Qwen3-14B/` from the existing papers' triples without re-running the LLM.
- Deferred (parts 2-3 of the user's plan): masked-big-node train/eval on the global graph; citation-only (no augmented nodes) comparison graph. `kg_transe_pipeline.py` untouched — all existing KGE results remain valid.

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
- **The Windows laptop DOES have Python** (3.14.7, with pymupdf/pdfplumber/ftfy/bs4/networkx/pandas — no torch/transformers). Part-1 **parser work can be developed and tested locally**; only GPU/LLM stages need the VM. Run from the project root with `PYTHONPATH=.` and **`PYTHONIOENCODING=utf-8`** — the console is cp1252 and `citation/network.py:69` prints a `✔`, which otherwise raises `UnicodeEncodeError` at import time.
- Files are always delivered individually, never as zip archives
- Code runs locally on the private VM; output files stay in `output/`
- Entity CSVs named `Entity_{paper_id}.csv` or `Entity_-_{name}v2.csv`
- Paper IDs: old-style ACL (C16-1036, P18-1001), new-style (2020.acl-main.130), or arXiv (1810.04805)
