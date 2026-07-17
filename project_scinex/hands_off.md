# Hands-Off Notes — Model Swap Session

**Date:** 2026-06-09
**Context:** Switching the KG extraction LLM off Qwen3-32B because the VM's GPU (20GB VRAM) can't run it.

> Read `Claude.md` first for the full project overview. This file only covers the model-swap work from this session.

---

## What we changed

### 1. Model swapped: Qwen3-32B → Qwen3-14B
We tried several models before landing on Qwen3-14B:

| Model | Result |
|---|---|
| Qwen3-32B (original) | OOM — too big for 20GB |
| google/gemma-3-27b-it | Rejected — HuggingFace repo is **gated** (needs license acceptance) |
| Qwen3-30B-A3B | Loaded but **froze during generation** — MoE loads all 30B params into VRAM (~19.3/19.8GB used), leaving no room for KV cache/generation |
| **Qwen/Qwen3-14B** | **CURRENT** — ~7–8GB at 4-bit, ~12GB headroom. Largest practical Qwen3 dense model for this GPU |

Changed in:
- `kg_extraction/llm_extractor.py` — `MODEL_NAME = "Qwen/Qwen3-14B"`
- `kg_extraction/fixed_extractor.py` — `DEFAULT_MODEL = "Qwen/Qwen3-14B"`
- `Claude.md` — updated all model references + run commands + added a "GPU notes" section

### 2. Fragmentation fix
Hit `torch.OutOfMemoryError` that was actually fragmentation (only 8.5GB allocated but reserved-unallocated was 10GB). Added to the top of **both** extractors:
```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

### 3. Greedy decoding fix
Qwen3's model `generation_config` defaults to `do_sample=True`, which overrode our pipeline's `do_sample=False` at runtime (and threw `top_k not valid` warnings). This is bad for deterministic structured extraction and was suspected in the 30B freeze. Added after `model.eval()` in **both** extractors:
```python
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None
```

### 4. CPU-offload "freeze" fix (2026-06-10)
A `kg_main.py --extractor llm` run appeared frozen "after calling the model". It was **not** a hang: `device_map="auto"` silently offloaded layers to CPU on the 20GB vGPU (H100-20C), so a 14B generation was running on CPU (276% CPU, 37GB RAM, only ~449MiB VRAM) — effectively never finishing, and it blocked the foreground terminal.

Fix — force the whole model onto GPU 0 so it either loads fully on-GPU or fails loudly (never silently offloads):
```python
device_map={"": 0},   # was device_map="auto"
```
Changed in **both** `kg_extraction/llm_extractor.py` (line ~291) and `kg_extraction/fixed_extractor.py` (line ~393).

Verified with Qwen3-14B: loads in ~34s, all layers on GPU 0, **10GB / 20GB VRAM** (~10GB headroom for KV cache), generation completes in ~9s. The `['temperature'] not valid` warning is harmless (greedy-decode config being ignored), unrelated to the freeze.

> If a future run looks frozen again: `nvidia-smi`. Low VRAM (<2GB) + high CPU/RAM on the python PID = CPU offload, not a hang. Kill the PID to free the terminal.

### 5. expandable_segments removed — fatal on vGPU (2026-06-10)
After fix #4, `kg_main.py` hit `RuntimeError: CUDA driver error: operation not supported` during checkpoint loading. Cause: the session-3 "fragmentation fix" `os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"` uses CUDA **virtual-memory** APIs that vGPUs (H100-20C here) don't support — it fails on *any* CUDA allocation, even `torch.zeros(1000, device='cuda')`. (Standalone test scripts passed only because they didn't import the extractors / set the flag.)

Fix: **removed** the `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...)` line (and now-unused `import os`) from both `kg_extraction/llm_extractor.py` and `kg_extraction/fixed_extractor.py`. Verified: importing the extractor then allocating on CUDA now works. Do NOT re-add expandable_segments on this hardware. Also remove the matching note in `Claude.md` (line ~112).

---

## Current problem / state
- **Freeze is fixed (see #4 above). Model now verified to load + generate on GPU.** Still pending: a full end-to-end pipeline run producing/evaluating triples.
- Qwen3-14B is **not in the HF cache yet** — first run will download (~28GB) before loading. Don't mistake the download/load wait for a freeze (see Claude.md: loading 14B into VRAM takes a few min).
- The previous Qwen3-30B-A3B process needs to be **killed** if still running.

## Current work
- Run the pipeline once with Qwen3-14B and confirm:
  1. Model loads without OOM
  2. Generation completes (no freeze) and produces triples
  3. Output quality is acceptable vs. the old Qwen3-32B BERT results (was 80% precise / 85% lenient precision, 30 triples)

Test command:
```bash
python kg_main.py --paper BERT --extractor fixed --entity-csv Entity_-_BERTv2.csv --model Qwen/Qwen3-14B
```

---

## Session 19 — New VM (2× RTX 5090), env rebuild, Gemma3-12B swap-in + full-corpus run (2026-07-17)

**Context: project moved to a new machine** — 2× RTX 5090 (32GB VRAM each), Ubuntu 24.04, working dir now `/home/dev1/project_scinex`, venv at `~/envsam`. The old 20GB-vGPU constraints (H100-20C notes in Claude.md "GPU notes") no longer apply on this box.

**1. `envsam` venv rebuilt from scratch.** The venv had been renamed from `env1` → `envsam`, which broke pip (scripts' shebangs still pointed at `/home/dev1/env1/bin/python3`). Deleted and recreated; installed everything from `requirement.txt` + `requirements_kg.txt` extras (bs4, pyvis). **Torch note:** the April-2026 nightly pin (`2.12.0.dev*+cu128`) is superseded — stable **torch 2.13.0+cu130 from plain PyPI** now supports RTX 5090 (sm_120) out of the box, verified with GPU matmul on both cards. No special index URL needed anymore. (The old "never use a dev-nightly" lesson from Session 4 still stands — but the fix is now just "use current stable".)

**2. Gemma3-12B (`google/gemma-3-12b-it`) wired into the fixed extractor** (user request: run with 12B).
- `.env` gained `HF_TOKEN=hf_...` (gated repo; token reused from the KGRAG project). `fixed_extractor.py` loads it via the same dotenv convention as `citation/network.py` and passes `token=` to both `from_pretrained` calls.
- Qwen3-specific handling (`/no_think` prefix, `enable_thinking=False`) is now **conditional on `"qwen3" in model_name`** — for other families the raw prompt is used, with a fallback that folds the system prompt into the user turn if the chat template rejects a system role.
- Added a multimodal fallback: if `AutoModelForCausalLM` raises (gemma-3-*-it registers as `Gemma3ForConditionalGeneration`), load via `AutoModelForImageTextToText` + manual tokenize/generate/decode instead of the text-generation pipeline. **In practice NOT hit** — current transformers (5.5.4) loads gemma-3-12b-it fine through `AutoModelForCausalLM` (`_is_multimodal=False`) — but the path is there for other checkpoints.
- Single-sentence smoke test passed: model loads in ~9s (4-bit NF4, ~8GB), correctly extracts `BERT trainedOn BooksCorpus` / `trainedOn English Wikipedia`.

**3. `acl_title_index.json` was corrupt (truncated at exactly 2,883,584 bytes — same size as `output.zip`, likely an interrupted copy). Rebuilt**: `anthology.json.gz` now 404s upstream, XML fallback used → **83,385 entries** (was ~81k). Corrupt original kept at `acl_title_index.json.corrupt.bak`. NB: `acl_index.py` has NO CLI entry point (docs' `--build` flag is stale) — rebuild via `ACLIndex('acl_title_index.json').build(save=True)`.

**4. Gemma3-12B validated end-to-end via `expand_and_validate.py --seed D13-1109 --count 8`** (the Session-18 smoke test, now on real GPU extraction). 7/8 papers fully extracted before we intentionally stopped it (2020.acl-main.101/103/118/130/136/148/185; .194 was mid-flight). **~8.7 min/paper**, avg **71 triples/paper** (vs ~42 for the Qwen3-14B corpus). Triples spot-checked correct; ontology guard correctly rejected invalid relations (`gatheredBy`, `used in`, `attends`).
- **Merge gotcha discovered:** killing kg_main.py with SIGTERM skips BOTH the periodic save (`--global-save-every 25`) and the `finally` save — in-memory merges are lost. Replayed the merge offline from the saved `triples.json` files (no GPU needed) → `output/global_kg/fixed/gemma-3-12b-it/graph.graphml` (618 nodes, 1,170 edges, 7 papers). **Connectivity report: all 7 attached (20 cites edges + 6 shared-entity neighbors each), zero isolated. Global-KG merge logic confirmed working on real extraction** — the Session-18 "VM smoke test" TODO is done.

**5. Full-corpus Gemma3 run IN FLIGHT at session close** in a detached screen session:
- `screen -r gemma3kg` — running `kg_main.py --all --extractor fixed --ontology ceo --model google/gemma-3-12b-it --skip-existing --max-new-tokens 4096`
- Log: `~/project_scinex/gemma3_fullrun.log`
- Corpus is **806 parsed papers** (802 entity-enriched) — much bigger than the 155/315 in older notes. At ~8.7 min/paper ≈ **~4.8 days** to complete.
- **User goal: ≥2000 papers.** Decision: finish the existing 806 first, THEN fetch ~1200 more via `citation_expand_pipeline.py` (index has 83k candidates) and extract those. Total ≈ 12 days GPU time.

**6. README.md fully rewritten** for GitHub — now covers all three pipeline stages, both extraction models, the persistent global KG, headline results, and the corpus; links to Claude.md/results.md/hands_off.md.

---

## Session 18 — Persistent global KG (paper big-nodes + augmented entity nodes + real cites edges) (2026-07-17)

**New direction from the user.** Reframe the KG: each paper = a **big node**; its fixed-extraction entities = **augmented nodes** that exist to help link prediction between big nodes. Instead of each extraction returning one isolated KG, every extracted paper is **added into one growing global graph** until the whole corpus lives in a single space. Explicit design choice (user): **citation edges become REAL paper→paper edges in this graph** — a deliberate break from `kg_transe_pipeline.py`'s methodology where citations are held out as eval-only ground truth (that pipeline is untouched; all prior KGE numbers remain valid). Future work (parts 2-3, NOT in this session): masked-node training (cut 1-2 big nodes, predict the missing link) and a citation-only ablation graph (no augmented nodes) evaluated the same way.

**1. NEW `kg_extraction/global_graph.py` — persistent incremental cross-paper KG.**
- Store: `output/global_kg/<extractor>/<model_slug>/graph.graphml` + `meta.json` (extractor ∈ {fixed, fixed_scinex}, mirroring the per-paper `kg/<extractor>/<model>/` convention). MultiDiGraph.
- Node model: `paper:<id>` big-nodes (`type="paper"`, title/year attrs) — entity nodes (`type="entity"`, produced by the SAME `KnowledgeGraphBuilder` used per paper, so normalisation/canonicalisation is identical) — `paper_stub` nodes for citation neighbors not yet in the corpus.
- Edges: entity→predicate→entity (from triples, deduped by relation|paper|section key), `paper→mentions→entity`, and `paper→cites→paper` (from `citation_network.json` edges; source cites target per `citation/network.py`).
- **Stub resolution (the key identity problem):** citation edges reference neighbors by raw S2 id. A neighbor gets a `paper:<s2_id>` stub with its title; `meta["s2_to_paper"]` accumulates the s2→acl mapping as papers merge; when a previously-stubbed paper is later extracted for real, its stub is `nx.relabel_nodes`-merged into `paper:<acl_id>` and all previously-attached edges survive. Merge order doesn't matter.
- Idempotency: `meta["merged_papers"][id].n_triples` — re-merging an unchanged paper is a no-op; a changed triple count re-merges automatically.
- `connectivity_report(graph, ids)` — per paper: # cites edges, # other papers reachable via shared entities, isolated yes/no.

**2. `kg_main.py` — auto-merge wired in (fixed/fixed_scinex only, `GLOBAL_GRAPH_EXTRACTORS`).**
- After each successful fixed extraction the paper merges into the in-memory global graph; **also on the `--skip-existing` path** (reads the existing `triples.json`) — so ONE `--all --skip-existing` pass over the current 155-paper corpus backfills the entire global graph WITHOUT re-running the LLM.
- Save strategy: in-memory merges, checkpoint every `--global-save-every` N papers (default 25) + final save in a `finally:` (partial batches persist — this VM has a history of killed runs). New flags: `--global-graph-dir` (default `output/global_kg`), `--no-global-merge`, `--global-save-every`. Per-paper triples.json/kg.graphml outputs unchanged.

**3. NEW `expand_and_validate.py` — the smoke test the user asked for.** Uses ONE seed paper's `citation_network.json` to download new papers from ACL Anthology (reusing `citation_expand_pipeline.py`'s resolve/download + `run_acl_batch.run_paper` parse + `enrich_entity_csv --source csner` + `kg_main --skip-existing` extraction), then prints a connectivity report on exactly the new ids: present in graph? # cites edges? # shared-entity papers? isolated (=bug signal)? Flags to resume mid-chain: `--skip-download/--skip-parse/--skip-extract`.
- **⚠ Seed must be an ACL-Anthology-pulled paper (user instruction), NOT the manually-added "BERT" folder** — BERT has no ACL id and its S2 network historically pulled wrong papers (Sentence-BERT, see Known Bugs).
- **⚠ Seed choice matters — `2020.acl-main.185` FAILED with 0/8 resolvable (tried live 2026-07-17):** its 20 neighbors = 10 recent citing papers (2024-25, not on ACL Anthology) + 10 references that are mostly arXiv/ICLR (Seq2Sick, HotFlip-with-mismatched-title, Show-and-Fool…). Each unresolvable title costs ~12s of DBLP/S2 fallback → 4 min for nothing. **Fix: pre-scan seeds OFFLINE against `acl_title_index.json`** (scratchpad script `find_seed.py`, no network): best seeds on the current local corpus = **`D13-1109` (10 new locally-index-resolvable ACL neighbors: J17-2001, D15-1163, Q13-1035, N13-1056, P12-1017, P11-1002…)** and `W14-5318` (10, mostly W-workshop); also E14-4024 (9), W19-8615 / P19-1117 / P19-1454 (7 each). Old-style-id seeds (P/D/N/W 11-19) resolve from the local index instantly; recent *-main seeds mostly don't (their citers are post-index arXiv papers).

**4. LOCAL smoke-test constraints (user is running this on their Windows machine, NOT the VM):** local GPU tops out around a 7B model → use the already-cached judge model as the extractor: `--model Qwen/Qwen2.5-7B-Instruct` (works fine — the extractor's Qwen3-specific bits, `/no_think` prefix + `<think>`-stripping, are no-ops on Qwen2.5). Outputs land under `kg/fixed/Qwen2.5-7B-Instruct/` + `output/global_kg/fixed/Qwen2.5-7B-Instruct/`, cleanly ISOLATED from the VM's Qwen3-14B data — triple quality is below corpus standard but irrelevant for validating merge logic. Also learned: `expand_and_validate.py` is the FULL chain (step 4 loads the LLM ~10GB for 14B), so on a small GPU either pass a small `--model` or use `--skip-extract`. **The command in flight when the session closed:**
```bash
python expand_and_validate.py --seed D13-1109 --count 8 --model Qwen/Qwen2.5-7B-Instruct
```
(User was about to run this locally; outcome unknown — check `output/global_kg/fixed/Qwen2.5-7B-Instruct/` and the printed connectivity report next session.)

**Verified this session (Windows laptop, no GPU — code-level only):** `py_compile` clean on all touched files; synthetic unit test of `global_graph.py` PASSED (merge A→stub for B created → idempotent re-merge skipped → B extracted → stub resolved, cites edge survived relabel → graphml round-trip counts identical → connectivity report correct → changed triples re-merge). `kg_main.py --help` + `expand_and_validate.py --help` import/parse clean.

**NOT yet done / next session:**
1. **Smoke test:** locally in flight as `python expand_and_validate.py --seed D13-1109 --count 8 --model Qwen/Qwen2.5-7B-Instruct` (see #4 — outcome unknown, check first thing). On the VM the same test runs with the default `--model Qwen/Qwen3-14B`. Want: 0 isolated new papers in the report.
2. **Full-corpus backfill (no LLM cost):** `python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B --skip-existing` → populates `output/global_kg/fixed/Qwen3-14B/` from the existing triples (repeat with `--ontology scinex` for the scinex graph if wanted).
3. Then parts 2-3: masked-big-node link-prediction training on the global graph + citation-only ablation (no augmented nodes) — the actual experiment this foundation is for.

**Files:** new `kg_extraction/global_graph.py`, new `expand_and_validate.py`, edited `kg_main.py` + `kg_extraction/__init__.py`. No changes to `kg_transe_pipeline.py`/`kg_evaluate.py`/any output format; `results.md` unaffected (no result numbers changed).

---

## Session 17 — Corpus → 102 papers, fetch-arbitrary-ACL validated, KGE precision FIXED (0.05→0.39) (2026-06-17/18)

**1. Corpus expanded 54 → 102 parsed papers.**
- Ran `run_acl_batch.py --timeout 1200` with the S2 key live → the 40 unparsed CS-NER papers parsed (9× citation-bearing). `network.py` `load_dotenv()` now loads `.env` from the PROJECT ROOT (absolute path off `__file__`), fixing a CWD-dependent 403.
- **CS-NER ACL source is now exhausted** (93/97 downloaded; the few left are malformed ids). To grow further we fetch ARBITRARY ACL Anthology papers — the fixed extractor's entity list (CS-NER gazetteer ∩ paper text) works for ANY parsed paper, not just CS-NER-annotated ones.
- **Validated the fetch-arbitrary-ACL chain:** picked 10 recent main-conf papers (ACL/EMNLP/NAACL 2019–2022) from `acl_title_index.json` (81k entries, title→id), downloaded via `acl_pipeline.download_pdf`, parsed → **9/10 succeeded** (P19-1211, 2020.acl-main.321, P19-1648, 2020.acl-main.513, 2022.emnlp-main.716, 2021.emnlp-main.298, 2021.emnlp-main.669, 2022.naacl-main.238, 2021.naacl-main.50). Then enrich (`--source csner`) + fixed extraction ran on the new papers. Scaling to ~50/100 is the same flow with a bigger selection.
- **Stragglers:** D19-1395, S16-1185 fail on `timeout after 1200s` (S2 citation rate-limit, not a bug). Retry with `--rerun-failed --timeout 1800` or parse `--no-citations`.

**2. KGE precision problem diagnosed and FIXED — the big win.** First run on the 79-paper corpus was poor (ComplEx MRR 0.101, RotatE 0.053; the diagnostic `gt_ranks` showed **90% of true citation neighbors ranked 100+**, but only 6% null → NOT a coverage problem). Root cause: **representation/vocabulary mismatch** — query (corpus) papers were represented ONLY by their CEO entities while candidate papers carried abstract `concepts`; cosine ranking was near-random. Specifically the `root_concepts` block in `build_unified_graph` was an **unimplemented empty comment**, so seed papers had zero concept content.

Four changes in `kg_transe_pipeline.py`:
- **Seed concepts implemented** — seed paper now gets `paper→mentions→concept` edges from its `root_concepts` (loaded into `paper_meta`). Puts query + candidate papers in the SAME concept vocabulary. *(This was the structural unlock.)*
- **Multi-negative sampling** — new `--neg-ratio` (default **10**); tiles each positive against 10 corruptions (all 3 model losses are element-wise margin, so tiling works unchanged).
- **`--dim` 64 → 128**, **`--epochs` 500 → 1000** (new defaults).

**Result (same 79-paper eval, single run each):**

| Model   | MRR before | MRR after | Hits@1 | Hits@10 |
|---------|-----------|-----------|--------|---------|
| TransE  | 0.052 | 0.254 | 0.165 | 0.430 |
| ComplEx | 0.101 | 0.362 | 0.253 | 0.582 |
| **RotatE** | 0.053 | **0.387** | **0.266** | **0.608** |

→ ~4–7× jump. Hits@10 ≈ 0.61 (real citation in top-10 for ~60% of papers), Hits@1 ≈ 0.27. Rank-100+ dropped 90% → 65%. **This is now a genuinely useful result, not just above-random.**

**2b. MULTI-SEED CONFIRMS it (5 seeds, 79 papers, dim128/1000ep/neg10, `output/kge_multiseed_summary.json`):**

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            |
|---------|---------------|---------------|---------------|----------------|
| TransE  | 0.170 ± 0.019 | 0.316 ± 0.009 | 0.433 ± 0.016 | 0.259 ± 0.010  |
| ComplEx | 0.213 ± 0.052 | 0.499 ± 0.051 | 0.592 ± 0.053 | 0.341 ± 0.044  |
| **RotatE** | **0.246 ± 0.033** | 0.489 ± 0.039 | **0.608 ± 0.022** | **0.364 ± 0.031** |

→ **RotatE is best and settled** — leads MRR/Hits@1/Hits@10, wins 4/5 seeds on MRR, lowest variance (ComplEx only edges Hits@5). **Verdict FLIPPED from the old ComplEx (37-paper, pre-fix MRR 0.157) to RotatE (MRR 0.364 ± 0.031).** RotatE & ComplEx overlap within ~1 std but RotatE wins on more metrics + more seeds + stability. Reportable: "RotatE MRR 0.364 ± 0.031, Hits@10 0.608 ± 0.022 (5 seeds, 79 papers)". The single-run 0.387 sits inside the seed range (max 0.416) → holds up.

**3. K19-1053 parser bug fixed** (see 4b in Session 16): `html_generator.py` `_subsub` seed `{}`→`0` + coerce. Re-parsed clean.

**2c. PRECISION PUSH #1 — empty-concept backfill (IMPLEMENTED 2026-06-19).** ~27% (386) of citation-network candidate nodes had empty `concepts` (S2 returned no abstract) → isolated random-embedding nodes. Fix in `build_unified_graph`: when a node's `concepts` is empty, fall back to `extract_concepts(title)` (imported from `citation.network`). All 386 recover content from their titles (0 still empty); no json files rewritten — done at graph-build time. **Baseline to beat = the Session-17 multiseed RotatE MRR 0.364 ± 0.031.**

**RESULT — #1 is a clear win (5 seeds, 1000ep, post-backfill `output/kge_multiseed_summary.json`):**

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            | (MRR was) |
|---------|---------------|---------------|---------------|----------------|-----------|
| TransE  | 0.187 ± 0.021 | 0.349 ± 0.007 | 0.461 ± 0.014 | 0.287 ± 0.010  | 0.259 |
| ComplEx | 0.263 ± 0.016 | 0.504 ± 0.039 | 0.623 ± 0.033 | 0.386 ± 0.015  | 0.341 |
| **RotatE** | **0.296 ± 0.019** | **0.554 ± 0.049** | **0.678 ± 0.028** | **0.423 ± 0.014** | 0.364 |

→ All three models up; **RotatE still best and now DECISIVE** (MRR 0.423 vs ComplEx 0.386 — gap ≈ 2.5× their stds, no longer overlapping). RotatE MRR +16%, Hits@1 0.246→0.296, Hits@10 0.608→0.678 (real citation in top-10 for ~68% of papers). **Variance also halved** (RotatE ±0.031→±0.014; ComplEx ±0.044→±0.015) — backfilling the random-embedding nodes removed ranking noise. **New reportable headline: RotatE MRR 0.423 ± 0.014, Hits@10 0.678 ± 0.028, Hits@1 0.296 (5 seeds, 79 papers).**

**2d. PERF — vectorized negative sampling (IMPLEMENTED 2026-06-19, REQUIRED fix).** With `--neg-ratio 10` the old per-element Python `_corrupt` loop (with `.item()` calls) made a 5-seed/3-model/1000ep multiseed take **~16 HOURS** (CPU-bound, GPU idle). Rewrote `train_kge`'s negative sampling as pure on-GPU torch ops (`torch.randint`/`torch.rand` + `torch.where`, triples kept on-device) with a seeded `torch.Generator` for reproducibility. **~13× faster: 100 epochs 6.5min → 30s; full multiseed ~16h → ~75min.** `_corrupt` is now dead code (left in place). NOTE: the torch-generator stream differs from the old `random.Random` stream, so seed N won't reproduce pre-vectorization seed-N numbers exactly — still fully reproducible going forward. Loss converges by ~100-200 epochs now, so 1000ep is generous (kept for comparability with the 0.364 baseline).

**2e. PRECISION PUSH #4 — vocabulary normalization (IMPLEMENTED 2026-06-19).** `_norm_entity` was just `strip().lower()`; upgraded to also collapse whitespace, strip surrounding punctuation, and **singularize the head (last) word** via a dependency-free `_singularize` + `_NON_PLURAL` exception set (bias/series/species/physics/analysis…). Applied uniformly to BOTH KG entities and paper concepts, so `Language Models`↔`language model`, `embeddings`↔`embedding`, `datasets`↔`dataset` now link. Entity-side merge is modest (~65 forms, 2%); the intended win is cross-linking entity↔concept singular/plural. **Baseline to beat = post-#1 RotatE MRR 0.423 ± 0.014.**

**RESULT — #4 REGRESSED the best models → REVERTED.** Post-#4: TransE MRR 0.303±0.008 (+0.016), ComplEx 0.351±0.051 (−0.035, variance ↑), **RotatE 0.386±0.039 (−0.037, variance ↑)**. Singularization over-merged distinct entities/concepts, adding spurious links + ranking noise (only TransE, the weakest, nudged up). `_norm_entity` reverted to plain `strip().lower()` (the validated post-#1 state; a comment in-code warns not to re-add singularization without an ablation). Re-ran multiseed to restore the canonical 0.423 summary. **Lesson: aggressive surface-form merging hurts — the entities/concepts carry signal in their exact forms.**

**2f. PRECISION PUSH #2 — richer seed content from full text (IMPLEMENTED 2026-06-19).** Seed/query papers were represented by only ~15 abstract `root_concepts` (often inaccurate — e.g. BERT's root_concepts were actually SBERT's, wrong S2 abstract). Now `load_all_data` also reads each paper's `no-llm/output.json`, concatenates section text, and `extract_concepts(body, top_n=50)` → stored as `fulltext_concepts`; `build_unified_graph` adds the UNION of root+fulltext concepts as seed `mentions` edges. Graph grew 26.3k→30.6k triples. **CONFIRMED by multiseed (5 seeds, 1000ep) — #2 is a BIG win, even bigger than #1:**

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            | (MRR post-#1) |
|---------|---------------|---------------|---------------|----------------|---------------|
| TransE  | 0.309 ± 0.017 | 0.562 ± 0.017 | 0.676 ± 0.033 | 0.426 ± 0.013  | 0.287 |
| ComplEx | 0.284 ± 0.058 | 0.587 ± 0.038 | 0.701 ± 0.060 | 0.422 ± 0.049  | 0.386 |
| **RotatE** | **0.359 ± 0.033** | **0.684 ± 0.043** | **0.780 ± 0.026** | **0.505 ± 0.025** | 0.423 |

→ **RotatE MRR 0.423→0.505 (+19%), Hits@10 0.678→0.780 (real citation in top-10 for 78% of papers), Hits@1 0.296→0.359.** Tight variance (all 5 seeds ∈ [0.47,0.54]). RotatE decisively best (MRR 0.505 vs ComplEx 0.422). NOTE: #4 (vocab normalization) stays REVERTED; this is separate/additive.

**2g. TEMPORAL-FILTERED second metric added (IMPLEMENTED 2026-06-19, user request).** A paper can't cite the future, so added a second eval: rank only candidates with year ≤ query year, GT = neighbors ≤ query year ("papers it cited" / reference-prediction). Reported ALONGSIDE the bidirectional metric (not replacing it) — `evaluate()` adds flat `filt_hits@{1,5,10}`/`filt_mrr` keys + `filt_best_rank` per paper; `kge_multiseed.py` METRICS extended so both aggregate. Years come from `citation_network.json` (`year` field, stored in paper_meta + node_year map). NOTE: the bidirectional GT is ~50% newer "citing" papers, so the filtered task is SMALLER (n≈56 vs 79 — 23 papers have no in-corpus older refs) and NOT 1:1 comparable. Preliminary single run (rotate 200ep): bidirectional MRR 0.468 / filtered MRR 0.517.

**2h. NEW TOOL `fetch_more_papers.py` (scale the corpus).** Selects recent main-conf ACL/EMNLP/NAACL long papers (old-style P/D/N 18-19 + new-style 2018-2022 *-main) from `acl_title_index.json` (6,864 candidates not yet downloaded) excluding what we have, downloads PDFs to `output/acl/pdfs/`. `--count N` (default 50), reproducible `--seed`. Full scale chain (run in screen): `python3 fetch_more_papers.py --count 50 && python3 run_acl_batch.py --timeout 1200 && python3 enrich_entity_csv.py --all --source csner && python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B --skip-existing`. ⚠ Last step is GPU (Qwen 10GB) — do NOT run concurrently with a KGE multiseed/other GPU job on the 20GB vGPU.

**2i. SCALE-UP to 155 parsed (102→155, +53 main-conf papers) + enrich discovery BUG FIXED.** `fetch_more_papers.py --count 50` + parse ran fine (155 parsed). BUT `enrich_entity_csv.discover_paper_ids` only iterated `output/acl/<id>/` folders, and fetched papers are PDF-only (no per-paper acl folder) → enrich silently skipped them → no entity CSV → fixed extraction skipped them (fixed-triples stuck at 94 despite 155 parsed). **Fixed:** `discover_paper_ids` now also includes ANY parsed paper (`output/<id>/no-llm/output.html`) for csner/iter sources (write path already mkdir'd output/acl/<id>/). Verified 93→155 discovered. **TODO: still need to run `enrich --all --source csner` + `kg_main --all --extractor fixed --skip-existing` to actually create triples for the ~62 new papers, then re-run KGE on the bigger corpus.**

**2j. SCALED CORPUS KGE — 155 papers, 5 seeds (2026-06-20, `output/kge_multiseed_summary.json`).** Fixed extraction finished (155 papers w/ triples). Both metrics:

| Model   | Bidir MRR     | Bidir H@10 | Filt MRR      | Filt H@10 |
|---------|---------------|------------|---------------|-----------|
| TransE  | 0.315 ± 0.008 | 0.567      | 0.334 ± 0.006 | 0.583 |
| **ComplEx** | **0.432 ± 0.018** | **0.742** | **0.456 ± 0.027** | 0.760 |
| RotatE  | 0.403 ± 0.021 | 0.708      | 0.423 ± 0.025 | 0.760 |

→ **Best model FLIPPED back to ComplEx at 155 papers** (was RotatE at 79). ComplEx & RotatE are close throughout (~0.03 apart); ComplEx scales better (0.386→0.432 from 79→155 while RotatE dropped 0.505→0.403). **Absolute scores dropped vs the 79-paper run — EXPECTED (2× corpus = ~2× candidate pool = harder ranking); report lift-over-random.** Temporal filter gives a small consistent lift (+~0.02 MRR, filt H@10 0.76) — confirms the "papers it cited" task is slightly cleaner. **Headline: ComplEx MRR 0.432 (bidir) / 0.456 (temporal-filtered), Hits@10 ~0.74-0.76, 155 papers, 5 seeds.**

**2k. SECOND ONTOLOGY (scinex) added for fixed extraction — both kept, for comparison (2026-06-21).** New refined ontology `scinex_refined_14.owl` (Turtle, "Core Experiment Ontology / scinex" v0.7.0, 47 classes, 27 object properties). Added as a SELECTABLE ontology without touching the hardcoded CEO path:
- NEW `kg_extraction/ontology_loader.py` — `load_ontology(owl) -> (relations, schema)` via rdflib (installed). Builds schema hints `Domain → Range | <first sentence of rdfs:comment>`, expands `owl:unionOf` domains.
- `fixed_extractor.py`: `_make_fixed_system_prompt(relations, schema=None)` + `FixedTripleExtractor(..., schema=None)` — default None = hardcoded `_CEO_SCHEMA` (CEO unchanged); scinex passes its own.
- `kg_main.py`: `--ontology {ceo,scinex}` (default ceo) + `--ontology-file` (default scinex_refined_14.owl). scinex routes to a SEPARATE output dir **`kg/fixed_scinex/<model>/`** (CEO stays `kg/fixed/`), so both coexist. `ENTITY_EXTRACTORS` + model-slug path updated to include `fixed_scinex`.
- KGE: `--extractor fixed_scinex` added to `kg_transe_pipeline.py` + `kge_multiseed.py` (path is generic: `kg/<extractor>/<model>/triples.json`).
- scinex relations = CEO + 5 new (achievesResult, extractedFrom, mentions, producedBy, writes); domain/range refined. CEO-specific validation guards in the parser still fire on shared relation names (fine for comparison); new relations get generic validation. Verified: loads 27 rels, builds scinex prompt, CEO default untouched.
- **TODO (user): run scinex fixed extraction → `kg_main.py --all --extractor fixed --ontology scinex --model Qwen/Qwen3-14B --skip-existing` (GPU, ~hours, same entity CSVs); then KGE `--extractor fixed_scinex` (save to a DIFFERENT summary file) and compare vs CEO `--extractor fixed`.**

**2l. ✅ RESOLVED in 2m — scinex extraction was TRUNCATED by --max-new-tokens 512, 80 papers re-extracted (2026-06-22; verified fixed 2026-06-23).** scinex fixed extraction finished (155 papers, 152 w/ triples, 4922 total). BUT the run used a MIX of token caps: papers #1-75 at 4096 (default), #76-155 at 512 (a mid-run speedup that backfired). Truncation check vs CEO at the #75 boundary: **4096 batch ratio scinex/CEO = 1.03 (clean, scinex≈CEO); 512 batch ratio = 0.40 (60% of triples LOST).** So the comparison is invalid until the 80 truncated papers are redone. The 3 zero-triple papers (L16-1593, W14-5502 zero in CEO too; N19-5002 CEO=4) are a side-effect / benign. **LESSON: do NOT use --max-new-tokens 512 for fixed extraction — triple-rich paragraphs need more; the Claude.md "use 512 for snappy runs" note is about SPEED not completeness. Use the 4096 default (or ≥2048) for real extraction.** **FIX (TODO): delete the latest-80-by-mtime fixed_scinex triples, re-run `kg_main.py --all --extractor fixed --ontology scinex --model Qwen/Qwen3-14B --skip-existing` (no --max-new-tokens → 4096), then KGE compare. Encouraging: at equal 4096 cap scinex yields ≈ CEO triple count (1.03×).**

**2m. ✅ scinex truncation FIXED + VERIFIED — re-extraction is complete and clean (2026-06-23).** Verified the 2l TODO was carried out. State on disk now:
- **155/155 parsed papers have `kg/fixed_scinex/Qwen3-14B/triples.json`** — full coverage, none missing.
- **scinex total = 6,776 triples vs CEO `fixed` = 6,574 → ratio 1.031** (was 4922 / ratio 0.40 on the bad 512 batch). This is the clean scinex≈CEO ratio 2l predicted at the 4096 cap → **the CEO-vs-scinex comparison is now VALID.**
- **mtime distribution confirms the re-run:** 75 files dated 06-21 (original clean #1-75 @4096) + 80 files dated 06-22/06-23 (= exactly the 80 truncated #76-155, deleted and re-extracted @4096). 75+80=155.
- **Zero-triple papers = 2, both benign:** L16-1593 and W14-5502 — **both also 0 in CEO `fixed`**, so it's the paper content, not a scinex/truncation artifact. (N19-5002, flagged in 2l at CEO=4, is now scinex=3 — no longer a concern.)
- **4 papers run scinex < CEO (ratio<0.6): D19-1408 (37 vs 79), D19-1528 (23 vs 47), E17-1082 (14 vs 30), N18-1013 (49 vs 82). Checked — these are NOT truncated:** all valid JSON (cleanly closed), and triples span from "1 Introduction" through late sections (5.3.3 / Conclusion / 6.1 / 3.4). The first 3 are from the 06-21 clean batch; N18-1013 was re-extracted 06-22. The lower count is a **genuine ontology difference** (scinex's refined domain/range + validation produce fewer triples per section than CEO), not a token cutoff. No action needed.
- **CONCLUSION: scinex extraction is done across the whole corpus and ready for KGE comparison (`--extractor fixed_scinex`, save to a separate summary file) vs CEO `--extractor fixed`.**

**2n. scinex KGE multiseed — SET UP, user runs it (2026-06-23).** Everything is staged so the scinex eval can run without disturbing the CEO baseline:
- **CEO baseline PRESERVED.** Canonical CEO multiseed = `output/kge_multiseed_summary.json` (155 papers, seeds 1-5, 1000ep) + per-seed JSONs in `output/kge_seedruns/`. A labeled safety copy was made at **`output/kge_multiseed_summary_ceo.json`** (identical). DO NOT overwrite these.
- **scinex run writes to SEPARATE paths** (`--save-summary output/kge_multiseed_summary_scinex.json`, `--run-dir output/kge_seedruns_scinex`). The default save path is the CEO file, so these two flags are mandatory or the CEO result is clobbered.
- **Exact command (matches CEO params for a fair comparison — seeds 1-5, 1000ep, kge all, dim128 default):**
  ```bash
  python3 kge_multiseed.py --extractor fixed_scinex --model Qwen3-14B --kge all \
    --seeds 1 2 3 4 5 --epochs 1000 --device cuda \
    --run-dir output/kge_seedruns_scinex \
    --save-summary output/kge_multiseed_summary_scinex.json
  ```
  GPU job (~10GB, ~75 min). One GPU job at a time. A 1000ep/dim128 smoke confirmed it loads cleanly (6776 triples / 153 papers, 2205 held-out citation edges, 25 relations incl. scinex's achievesresult/mentions/extractedfrom/producedby, training on cuda) before being stopped for the user to run.
- **AFTER it finishes:** compare `kge_multiseed_summary_scinex.json` (scinex) vs `kge_multiseed_summary.json` (CEO). CEO baseline = ComplEx bidir MRR 0.432 / filt MRR 0.456 (full CEO table §2o). **DONE — results in §2p below.**

**2p. ✅ scinex KGE multiseed COMPLETE + scinex-vs-CEO comparison (2026-06-23, `output/kge_multiseed_summary_scinex.json`; 155 papers, seeds 1-5, 1000ep, dim128, neg10 — identical params to CEO).**

scinex per-model results (mean ± std):

| Model | Bidir MRR | Bidir H@1 | Bidir H@5 | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|---|---|
| TransE  | 0.325 ± 0.005 | 0.202 ± 0.013 | 0.441 ± 0.012 | 0.565 ± 0.016 | 0.330 ± 0.006 | 0.587 ± 0.014 |
| **ComplEx** | **0.441 ± 0.028** | **0.297 ± 0.043** | **0.618 ± 0.016** | **0.744 ± 0.015** | **0.472 ± 0.041** | **0.774 ± 0.022** |
| RotatE  | 0.391 ± 0.033 | 0.253 ± 0.039 | 0.552 ± 0.041 | 0.703 ± 0.027 | 0.407 ± 0.019 | 0.727 ± 0.051 |

**scinex − CEO deltas (bidir MRR):** TransE +0.010, ComplEx +0.009, RotatE −0.011. **ComplEx Δ across all metrics: +0.009 MRR, +0.012 H@1, +0.020 H@5, +0.002 H@10, +0.016 filt-MRR, +0.015 filt-H@10.**

**2p-bis. NO-FILTER vs YEAR-FILTERED comparison (both MULTISEED, 5 seeds, 155 papers, 1000ep) — the temporal filter helps everywhere.** Each seed's run computes BOTH metrics on the same trained model; table below is the 5-seed mean (full per-seed values in the summary JSONs):

| Ontology | Model | MRR no-filter | MRR filtered | ΔMRR | H@10 no-filter | H@10 filtered | ΔH@10 |
|---|---|---|---|---|---|---|---|
| CEO | TransE | 0.315 | 0.334 | +0.019 | 0.567 | 0.583 | +0.017 |
| CEO | **ComplEx** | 0.432 | **0.456** | +0.024 | 0.742 | **0.760** | +0.017 |
| CEO | RotatE | 0.403 | 0.423 | +0.020 | 0.708 | 0.760 | +0.052 |
| scinex | TransE | 0.325 | 0.330 | +0.005 | 0.565 | 0.587 | +0.022 |
| scinex | **ComplEx** | 0.441 | **0.472** | +0.031 | 0.744 | **0.774** | +0.030 |
| scinex | RotatE | 0.391 | 0.407 | +0.016 | 0.703 | 0.727 | +0.024 |

→ **Year filter improves every model × both ontologies: +0.02-0.03 MRR, up to +0.05 H@10.** Best cell overall = **scinex ComplEx filtered: MRR 0.472, H@10 0.774.** ⚠ **NOT a strictly 1:1 lift** — the filtered task runs on a SMALLER/different test set (drops papers with no in-corpus older refs, n≈56 vs ~79), so the gain is partly cleaner-task + partly easier-subset. Report BOTH; lead with filtered as the "reference-prediction (temporal)" task (more correct setup) but disclose the reduced test set. Both metrics are 5-seed multiseed (NOT single-run).

**VERDICT: the two ontologies are statistically INDISTINGUISHABLE on citation prediction.** Every delta is smaller than the seed-to-seed std (ComplEx +0.009 MRR vs ±0.028 std → error bars fully overlap). ComplEx is best under BOTH ontologies. scinex shows a small CONSISTENT edge on ComplEx (best on all 4 ComplEx metrics, biggest = filt-MRR +0.016) but it's within noise; RotatE is the one model slightly worse under scinex. **Defensible paper claim: the refined scinex ontology (47 classes / 27 object properties, +5 relations vs CEO) MATCHES CEO's downstream citation-prediction quality and does not degrade it — with a small non-significant improvement on the best model. NOT "scinex wins".** At near-equal triple count (scinex 6776 vs CEO 6574, 1.03×) the ontology choice is roughly neutral for this metric.

**2q. ⭐ BIG WIN — self-adversarial loss (Sun et al. 2019) on RotatE: MRR 0.40→0.57 (2026-06-23).** Implemented self-adversarial negative sampling + logsigmoid loss as an OPT-IN training objective; the historical margin-ranking loss stays the default so all prior baselines reproduce.
- **Code:** `kg_transe_pipeline.py` — new flags `--loss {margin,adv}` (default margin), `--gamma` (γ offset, default 9.0), `--adv-temp` (α, default 1.0). The adv branch in `train_kge` keeps negatives grouped per-positive `[B,K]`, softmax-weights hard negatives by α (detached), uses `−logσ(γ+s_pos) − Σ wᵢ logσ(−(γ+s_negᵢ))` where `s=forward()` (higher=better). Margin path is byte-identical to before. `kge_multiseed.py` also got `--loss/--gamma/--adv-temp` and passes them through (REQUIRED — without this the multiseed silently runs margin).
- **Tuning (single seed=1, 600ep, CEO/fixed):** RotatE needs a LARGE γ (distance-model scale): **γ=24 → bidir MRR 0.548**; γ=9 broke it (0.05). ComplEx adv was WORSE than its own margin (best 0.357 @ γ12 vs 0.432 margin) and *declines with training* (cosine-sim eval ↔ adv objective mismatch — its 20-epoch 0.485 was an underfit blip). TransE adv broke (0.012). **So self-adversarial is RotatE-only here** — it's RotatE's native objective; margin remains best for ComplEx/TransE.
- **CONFIRMED multiseed (5 seeds, 1000ep, γ=24, α=1.0, dim128; `output/kge_multiseed_summary_adv_{ceo,scinex}.json`, per-seed in `output/kge_seedruns_adv_{ceo,scinex}/`):**

| Config | Bidir MRR | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|
| CEO RotatE **margin** (old) | 0.403 ± 0.021 | 0.708 | 0.423 ± 0.025 | 0.760 |
| **CEO RotatE adv** | **0.566 ± 0.026** | **0.847** | **0.555 ± 0.029** | **0.855** |
| scinex RotatE **margin** (old) | 0.391 ± 0.033 | 0.703 | 0.407 ± 0.019 | 0.727 |
| **scinex RotatE adv** | **0.568 ± 0.022** | **0.838** | **0.575 ± 0.018** | **0.844** |

→ **RotatE adv vs margin: CEO +0.164 bidir / +0.133 filt; scinex +0.177 / +0.168.** Tight std (±0.02-0.03 over 5 seeds → not seed-luck). **NEW HEADLINE BEST = RotatE + self-adversarial: MRR ~0.57, Hits@10 ~0.84-0.85** (vs the old ComplEx-margin best 0.432/0.456, H@10 0.74). Legitimate method improvement (the model's native training objective), NOT eval tuning. **Ontology comparison still ≈ TIE at the higher level** (scinex filt MRR 0.575 vs CEO 0.555 — within std; scinex edges ahead but not significant). Run cmd (both ontologies, ~50-60 min total): `kge_multiseed.py --kge rotate --loss adv --gamma 24 --adv-temp 1.0 --seeds 1 2 3 4 5 --epochs 1000` with `--extractor {fixed,fixed_scinex}` + separate `--run-dir`/`--save-summary`. **TODO maybe: γ tuning was coarse (18/24/30 tried, 24 best single-seed); a finer sweep or ComplEx-with-adv-and-score-based-eval could squeeze more, but 0.57 is already a strong, defensible headline.**

**2t. CITATION-EDGE DIRECTION FIX — `filt_*` is now TRUE reference prediction, not a year proxy (2026-06-28).** Problem (raised by user/professor): the old temporal metric used `year ≤ query year` as a PROXY for citation direction, which is inconsistent — the same edge "4 cites 1" was counted as a link for query=1 (where it's a newer *citer*, dropped by the year filter) AND query=4 (where it's an older *reference*). Year is also a bad proxy (same-year cites, preprint-vs-pub mismatches).
- **Fix in `evaluate()` (`kg_transe_pipeline.py`):** `filt_*` now uses the ACTUAL edge direction. Citation-network edges are `{source, target, type:'cites'}` = source cites target; `relation` field confirms `cited_by` = reference (older), `citing` = citer (newer). GT for query P = papers P actually cites = edges where `source == P.s2_id` → targets. Ranked against the FULL candidate pool (no year filtering — references are naturally older, future papers are just distractors). This is the consistent directional "what does this paper cite" task.
- **Effect (text baseline, title+abstract, test split):** filt MRR 0.740→**0.524**, filt H@10 0.955→**0.850** — lower but HONEST (old was inflated by the shrunk ≤year candidate pool + direction-agnostic GT). Bidirectional `mrr`/`hits@*` UNCHANGED (still undirected "relatedness": any citation neighbor).
- **⚠ Supersedes all prior `filt_*` numbers** (§2o–2s year-based filt are stale; bidir there still valid). Re-run KGE on the final corpus to get fresh directional numbers. `kge_multiseed.py` needs no change (same keys). `node_year` map in evaluate() is now unused by filt (left in place, harmless).
- **Two metrics now, both kept:** `mrr`/`hits@*` = bidirectional relatedness (undirected); `filt_*` = directional reference prediction (edge-direction, the defensible "papers it cites" task).

**2s. ⚠ TEXT-SIMILARITY BASELINES added — and they look STRONGER than KGE (2026-06-28).** User request: baselines that embed paper TEXT and predict citations by cosine similarity, to test whether KG embeddings actually beat naive text matching. Built `text_baseline.py` (CPU-only, TF-IDF via scikit-learn — sentence-transformers NOT installed and avoided to not disturb the pinned torch). Two baselines: **title** and **title+abstract**. Title/abstract pulled from `citation_network.json` (query = root title/abstract keyed by FOLDER id — NOT root `paper_id`, which is an S2 hash; candidates = `nodes[s2]` title/abstract; output.json title fallback). Reuses `kg_transe_pipeline.evaluate(predict_fn=...)` (new `predict_fn` hook) so candidate pool / ground-truth / val-test split / metrics are IDENTICAL to the KGE eval → directly comparable.
- **PRELIMINARY result (test split, current IN-FLUX 315-paper corpus, `output/text_baseline_test.json`):** title TF-IDF MRR 0.606 / H@10 0.829; **title+abstract TF-IDF MRR 0.703 / H@10 0.904 / filt MRR 0.740 / filt H@10 0.955.**
- **⚠ This BEATS the KGE headline** (RotatE+adv ~0.60 MRR / 0.85 H@10) — and on a LARGER, harder corpus, so the gap is likely real. **Implication for the paper: a simple lexical baseline over abstracts outperforms the KG embedding → cannot claim "KGE is best" naively. Reframe needed** (e.g. "KG structure alone, without abstract text, approaches a strong content baseline"; or combine KG+text; or position KGE as complementary). NOT fatal, but strategic.
- **NOT yet apples-to-apples:** text ran on 315-corpus (146 test); KGE §2r was 155-corpus (66 test). **DEFINITIVE comparison = run BOTH on the FINAL corpus after extraction finishes + KGE re-run** (`--split-seed 42` guarantees identical test papers). Bug fixed during build: corpus papers were initially keyed by S2-hash `paper_id` → empty query text → near-random; now keyed by folder id (100% text coverage).
- Run: `python3 text_baseline.py --extractor fixed --model Qwen3-14B --text-source both --eval-split test --save-results output/text_baseline_<corpus>.json`

**2r. ⭐⭐ DEFENSIBLE HEADLINE — validation/test split + RotatE adv on TEST (2026-06-24).** Closed the "hyperparameters tuned on the test set" rigor gap. Implemented a query-paper val/test split (`--eval-split {all,val,test}`, `--val-frac 0.5`, `--split-seed 42` — split fixed by split_seed ONLY, independent of training --seed, so val/test membership is identical across configs/seeds; default `all` = old behaviour). `evaluate()` in `kg_transe_pipeline.py` partitions `papers_to_eval`; `kge_multiseed.py` passes the flags through. 132 eval papers → 66 val + 66 test (disjoint).
- **γ selected on VALIDATION only** (RotatE adv, seed1, 600ep, `output/val_tune/`): γ16→0.522, γ20→0.580, γ24→0.567, **γ28→0.628 (peak)**, γ32→0.619. Curve turns over → **val-selected γ = 28** (the earlier γ=24 was picked on the full set; 28 is the clean choice).
- **FINAL multiseed on the disjoint TEST split (66 papers, RotatE, 5 seeds, 1000ep, γ=28; `output/kge_multiseed_summary_test_{ceo,scinex}_{margin,adv}.json`):**

| Config | Bidir MRR | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|
| CEO RotatE margin | 0.406 ± 0.053 | 0.727 | 0.418 ± 0.036 | 0.759 |
| **CEO RotatE adv** | **0.598 ± 0.041** | **0.861** | **0.594 ± 0.027** | 0.844 |
| scinex RotatE margin | 0.391 ± 0.043 | 0.712 | 0.417 ± 0.027 | 0.737 |
| **scinex RotatE adv** | **0.599 ± 0.023** | 0.845 | **0.606 ± 0.054** | 0.844 |

→ **adv vs margin on the SAME test split: CEO +0.19 bidir / +0.18 filt; scinex +0.21 / +0.19.** The self-adversarial win SURVIVES clean model selection (actually a touch higher, ~0.60 vs the §2q full-set 0.57, since γ=28>24 + this test half runs slightly high). **DEFENSIBLE PAPER HEADLINE: RotatE + self-adversarial, MRR ~0.60, Hits@10 ~0.85, 5 seeds, with γ selected on a disjoint validation set (no test-set tuning).** Ontology comparison STILL a tie (CEO filt 0.594 vs scinex 0.606, within std). NOTE: test-split numbers are on 66 papers (half the corpus) — that's the cost of an honest held-out split; the §2q all-papers numbers (155, MRR 0.57) remain valid as the "evaluated on all papers" figure, but §2r is the one to report as the primary result because its hyperparameters weren't chosen on the eval data.

**2o. CEO BASELINE — full multiseed numbers for the writeup (155 papers, 5 seeds, 1000ep, dim128, neg10; `output/kge_multiseed_summary.json`).** Exact mean ± std, both metrics (bidirectional = predict any citation neighbor; temporal-filtered `filt_*` = predict only older papers it could have cited):

| Model | Bidir MRR | Bidir H@1 | Bidir H@5 | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|---|---|
| TransE  | 0.315 ± 0.008 | 0.197 ± 0.011 | 0.429 ± 0.007 | 0.567 ± 0.021 | 0.334 ± 0.006 | 0.583 ± 0.018 |
| **ComplEx** | **0.432 ± 0.018** | **0.285 ± 0.029** | **0.598 ± 0.028** | **0.742 ± 0.029** | **0.456 ± 0.027** | **0.760 ± 0.028** |
| RotatE  | 0.403 ± 0.021 | 0.259 ± 0.033 | 0.573 ± 0.017 | 0.708 ± 0.011 | 0.423 ± 0.025 | 0.760 ± 0.025 |

→ **ComplEx is best for CEO at 155 papers.** Random baseline ≈ 0.2% Hits@1 over ~500 candidate papers → report lift-over-random, not the absolute number (1.0 is not the target for unsupervised citation prediction). **scinex equivalent table = §2p (done; scinex ≈ CEO, indistinguishable).**

**Open items next session:**
00. **`results.md` (project root) = consolidated PAPER-READY results** (final tables, methodology, findings, caveats, reproduction, scaling decision). Generated 2026-06-24. Hand THIS to the paper/abstract writer; keep it in sync if numbers change.
0. **DONE: scinex extracted (§2m), KGE run (§2p) — scinex ≈ CEO on citation prediction, statistically indistinguishable.** The CEO-vs-scinex ontology comparison is complete.
0b. **⭐⭐ DEFENSIBLE HEADLINE (§2r): RotatE + self-adversarial, γ selected on a held-out VALIDATION set, reported on disjoint TEST = MRR ~0.60 / Hits@10 ~0.85 (5 seeds, both ontologies).** This is the number to put in the paper (clean model selection, no test-set tuning). Self-adv helps RotatE only. **The method/experiments are DONE — next is write-up.**
1. **CURRENT BEST = RotatE self-adversarial (γ=28), TEST split: CEO bidir 0.598 / filt 0.594, scinex bidir 0.599 / filt 0.606, Hits@10 ~0.85 (§2r, `output/kge_multiseed_summary_test_*`).** All-papers version (§2q, γ=24, no val/test split): CEO/scinex MRR ~0.57. Margin reference: ComplEx 0.432 (§2o). Ontology = tie throughout.
2. **Precision arc: ~0.05 → 0.36 (seed concepts+tuning) → 0.423 (#1 empty-concept backfill) → 0.505 (#2 full-text seed content).** Wins: #1, #2. Negative/reverted: #4 vocab normalization. Untried: self-adversarial negatives (RotatE technique, principled but loss-rewrite); tighter eval pool (risks gaming — skip).
3. **Recommended next: scale papers OR write up.** RotatE MRR 0.51 / Hits@10 0.78 is strong & defensible. If scaling: fetch more ACL via `acl_title_index.json` (validated chain) → parse → enrich → fixed → re-run KGE; expect absolute scores to dip with a bigger pool, so report lift-over-random.

---

## Session 16 — Multi-seed KGE settles ComplEx as best + diagnostic predictions + S2 key wired (2026-06-16)

**1. Multi-seed KGE result is the new source of truth (`output/kge_multiseed_summary.json`, seeds 1–5, 500 epochs, fixed extractor, 37 papers).** This resolves the Session-15 single-run ambiguity (one run said ComplEx 0.184, another RotatE 0.178 — just variance over ~37 held-out edges). Mean ± std:

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            |
|---------|---------------|---------------|---------------|----------------|
| TransE  | 0.054 ± 0.000 | 0.065 ± 0.015 | 0.103 ± 0.012 | 0.082 ± 0.002  |
| **ComplEx** | **0.070 ± 0.024** | **0.232 ± 0.065** | **0.330 ± 0.073** | **0.157 ± 0.014** |
| RotatE  | 0.070 ± 0.031 | 0.162 ± 0.033 | 0.287 ± 0.070 | 0.137 ± 0.019  |

→ **ComplEx is best, now with confidence** — wins MRR/Hits@5/Hits@10, beats RotatE on 4 of 5 seeds, tied at Hits@1. TransE consistently weakest. Results are ~16–40× the random baseline (~0.2% Hits@1 over ~500 candidate papers): a solid *proof-of-signal*, not a SOTA retrieval result, and that's the honest framing for any writeup (report the lift-over-random, not the absolute number — 1.0 is not the target for unsupervised citation prediction).

**2. `kg_transe_pipeline.py` `evaluate()` now saves DIAGNOSTIC predictions (per paper).** Old output stored 5 raw S2 hashes, undiagnosable. New per-paper fields: `title`; `top10_predictions` (rank, paper_id, **title**, score, **`is_true_citation`** flag); `gt_ranks` (every real citation neighbor + **where it actually ranked**, or `null` if not a graph node); `total_candidates`. Added a `label()` title lookup (maps ACL ids via paper_meta, raw S2 hashes via aggregated citation-network node titles). **Only affects future runs** — the on-disk `kge_fixed_results_*.json` were written by old code. Purpose: diagnose WHY scores are low — true neighbors at rank 11–30 = near-misses (window too small), `null` = coverage gap (paper not a node), rank 100+ = genuine miss.

**3. Semantic Scholar API key is now wired (clears the 403/429 that blocked parsing).**
- Key stored in **`/home/ubuntu/project_clean_9/.env`** as `SEMANTIC_SCHOLAR_API_KEY=...`. `network.py` auto-loads it via `load_dotenv()` and sends it as the `x-api-key` header.
- **Installed `python-dotenv`** (`python3 -m pip install python-dotenv`) — it was MISSING, so `load_dotenv()` was silently no-opping and `.env` was never read. Without this package the `.env` does nothing. Verified: `import citation.network` prints `Semantic Scholar API key loaded ✔`; live Graph API call returns HTTP 200.
- S2 endpoints the parse uses (all Graph API v1, covered by a free key): `/paper/ACL:{id}`, `/paper/ARXIV:{id}`, `/paper/search`, `/paper/{id}`, `/paper/{id}/citations`, `/paper/{id}/references`. ~5–6 requests/paper, <600 total for the corpus.
- The prior 403 was unauthenticated S2 refusing requests (not the PDFs — those are local in `output/acl/pdfs/`). `citation/apikey.env` is a dead placeholder; only root `.env` is auto-loaded.
- **Hardened `network.py` `load_dotenv()` to load `.env` from the PROJECT ROOT (parent of `citation/`), not the CWD.** A 403 reappeared on a batch run because `load_dotenv()` searched the current dir; now it uses an absolute path off `__file__`, so launching the parse from anywhere still picks up the key. Verified the key loads even when imported from `/tmp`. **Transient 429 backoffs still occur even WITH the key** (per-key pagination limits) — the retry loop absorbs them; D12-1051 parsed fine through them (10 nodes/10 edges). Don't mistake a 429-with-retry for the old 403 failure.

**4b. Parser bug fixed — `html_generator.py` subsubsection counter (K19-1053).** `IDGenerator._subsub` is a flat int counter, but `next_section` seeded it as `{}` (dict) while `next_subsection`/`next_subsubsection` treat it as int → a subsubsection directly under a section did `dict += 1` → `TypeError: unsupported operand type(s) for +=: 'dict' and 'int'`, killing the whole parse. Fixed: seed `self._subsub[sid] = 0` (not `{}`) + `next_subsubsection` now coerces any non-int to 0 before incrementing. K19-1053 re-parsed clean (20 sections). Same class as the Session-13 structural-edge-case fixes. Since `run_acl_batch` spawns a fresh `main.py` per paper, the running batch picks this up automatically for all later papers.

**Open items / the actual next step (parse the 40 remaining papers, then re-run):**
1. **Parse:** `python3 run_acl_batch.py --timeout 1200` — key is wired, auto-skips the 54 done papers, processes the 40 unparsed (7 prior S2 failures + 33 untouched). May still see occasional 429 backoffs (per-key pagination) — the 1200s timeout absorbs them.
2. **Enrich:** `python3 enrich_entity_csv.py --all --source csner` (cached gazetteer, only new papers cost anything).
3. **Fixed extract:** `python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B --skip-existing` (skips the 54 with triples; one LLM extractor at a time on the 20GB vGPU).
4. **Re-run KGE** (now writes the diagnostic per-paper output): `python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge all --epochs 500 --save-results output/kge_fixed_results.json`. Then inspect `gt_ranks` across papers to see if low scores are near-misses vs coverage gaps. For a stable verdict, also re-run the multi-seed sweep.

---

## Session 15 — Corpus expansion + CS-NER entity lists live + new KGE result (2026-06-16)

**Corpus growth:** parsed corpus **34 → 54 papers** (of 93 ACL PDFs) via `run_acl_batch.py`. 55 papers now have fixed triples.
- **`run_acl_batch.py` hardened:** catches `subprocess.TimeoutExpired` per paper (one slow paper no longer aborts the batch), `--timeout` flag (default 900s), and writes `batch_run_log.json` after EVERY paper (resumable).
- **7 papers failed — ALL rate-limit timeouts, NOT parse failures:** `D12-1051, K19-1053, L18-1051, N18-1166, N19-1179, P13-2003, P19-1416`. Cause: Semantic Scholar citation fetch (unauthenticated ~100 req/5min → 429 backoffs exceed the timeout). Clear them with an S2 API key (`export SEMANTIC_SCHOLAR_API_KEY=...`) then `run_acl_batch.py --rerun-failed`, or parse `--no-citations`.

**CS-NER entity lists are now LIVE (the Session-14 migration is done):**
- Ran `enrich_entity_csv.py --all --source csner` (06-16 00:49) → all `Entity_<id>_enriched.csv` rebuilt from the CS-NER gazetteer (BERT→154, conll→102, D17-1028→69 entities — csner-scale, not the old ~16 llm-scale).
- Fixed extraction was re-run AFTER that → **current fixed triples reflect the CS-NER methodology** (verified by mtimes: enrich 00:49 < fixed triples 01:47–09:32 < KGE 11:42).

**New KGE result (06-16 11:42) — 37 papers (up from 22), KG 5,660 entities / 13,779 triples, reflects CS-NER entity lists:**

| Model   | Hits@1 | Hits@5 | Hits@10 | MRR   |
|---------|--------|--------|---------|-------|
| TransE  | 0.054  | 0.054  | 0.108   | 0.081 |
| **ComplEx** | **0.108** | **0.243** | **0.378** | **0.184** |
| RotatE  | 0.027  | 0.189  | 0.378   | 0.112 |

→ **ComplEx now best (MRR 0.184, Hits@1 0.108)** — a shift from the 22-paper baseline where RotatE led (MRR ~0.13). Bigger corpus + CS-NER lists improved MRR ~0.13 → 0.184 and Hits@10 0.318 → 0.378.
> ⚠ This run was printed but NOT saved (no `--save-results`) — recovered from the screen buffer. Re-run to persist: `python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge all --save-results output/kge_fixed_results.json`.

**Open items for next session:**
1. Re-run KGE **with `--save-results`** to persist the table (the 0.184 ComplEx result is only in this doc + a screen buffer).
2. Finish parsing the remaining ~39 PDFs (the 7 failed + ~32 untouched) — needs an **S2 API key** to beat rate-limiting.
3. After more papers are parsed: rebuild their csner entity lists, re-run fixed (+pair), re-run KGE.
4. (Optional) `pair` extractor could also be re-run on the csner lists if a pair eval is ever wanted (currently pair is NOT KGE-evaluated).

---

## Session 14 — CORRECTION: entity list must come from CS-NER per paper, not open extraction (2026-06-15)

**This corrects a wrong assumption baked into Sessions 6-8.** The fixed extractor's subject pool was being built by `enrich_entity_csv.py --source llm`, which harvests entities from our OWN open (`llm`) extraction's `triples.json`. **That is the wrong source** — using one LLM extraction to seed the "constrained" fixed extractor is circular. Confirmed in code: [enrich_entity_csv.py:180](enrich_entity_csv.py#L180) reads `output/<id>/kg/llm/<model>/triples.json`; in practice the open extraction provided the *bulk* of each enriched CSV (e.g. D17-1028 = ~2 title seeds + ~14 open-extraction entities; J13-4001 = 1 seed + 4).

**Intended design (per user):** for a given paper, look up its entity list **from the CS-NER dataset by title/id**, and use THAT as the fixed extractor's subject pool. CS-NER repo: `github.com/jd-coderepos/contributions-ner-cs`.

**Key finding — the repo has richer data than we were using:**
- `acl/` (`train/dev/test.data`) = **title-level only** (~2-3 entities/paper). This is the only ACL-id-keyed slice and all `acl_pipeline.py` currently pulls. Verified: train.data REC 1 = `"PUT at SemEval-2016 Task 4: The ABC of Twitter Sentiment Analysis"` — literally the title of S16-1018, with 3 title entities. So the old "CS-NER = titles only, 2-3 entities" note was correct *but only about the `acl/` folder*.
- `ftd/`, `ncg/`, `pwc/`, `scierc/`, `full dataset/` = **`*-abs.data` = abstract-level annotations** (full abstracts; `full dataset/train-abs.data` has 5,957 records, multi-sentence, many entities each). **This is the richer entity source we want.**

**Coverage investigation → CS-NER abstract data RULED OUT for our papers (2026-06-16):**
- Verified the data flow in code: PDFs come from `aclanthology.org/{id}.pdf` ([acl_pipeline.py:45](acl_pipeline.py#L45)); entities come from `parse_iob()` of the CS-NER **`acl/` title files** matched title→ACL-id → `Entity_<id>.csv` (title-level only).
- **`acl/train-abs.data` → HTTP 404.** There is NO abstract-level annotation for ACL papers in CS-NER. The `acl/` folder is title-only, period.
- The rich `*-abs.data` files live only under `ftd/ncg/pwc/scierc/full dataset/` = **different corpora (NCG/PapersWithCode/SciERC/FTD), not our ACL Anthology papers**, with no id/title to match on. (Common tokens like "BERT"=935 lines just reflect BERT being *cited* widely, not our BERT paper being a record.)
- **Conclusion:** CS-NER cannot provide a richer per-paper entity list for our corpus — title-level (2-3 entities) is all it has for ACL.

**DECISION + IMPLEMENTED (user, 2026-06-16): entity list ← CS-NER global gazetteer, intersected per paper.** (Briefly considered ITER over full text, but the user chose to use the CS-NER annotations directly.) Built as `enrich_entity_csv.py --source csner` (now the DEFAULT source). Non-circular, no GPU.

**How it works (mechanism B):**
- `build_gazetteer()` downloads ALL CS-NER files, aggregates → one global gazetteer (`output/acl/csner_gazetteer.csv`, ~51k entities, 7 CEO types) cached once (`--rebuild-gazetteer` to refresh). Types mapped via `CSNER_TO_CEO`.
- `extract_body_entities_csner()` keeps only gazetteer entities that appear in the paper's `output.html` (n-gram match ≤8 words) → per-paper `Entity_<id>_enriched.csv`.
- Quality gate `_gazetteer_keep`: multi-word entries kept (full); single tokens must be non-function-word (`_FUNCTION_WORDS`) and seen ≥2× — this killed the "and/for/use" noise that otherwise matched every paper. User picked "full" gazetteer (not the ≥2× core) since paper-intersection self-filters most noise.
- `_clean` now also strips surrounding brackets/quotes (fixed `(ZeRO` → `ZeRO`).

**Verified:** BERT→154, P17-1128→114, S16-1018→111, 2020.conll-1.24→102, D17-1028→69 entities, real & properly typed (e.g. BERT paper finds BERT/Pre-training/fine-tuning/GPT/LSTM). `--source llm` (open-extraction harvest) is retired; `--source iter` (ITER/SciERC model) kept as a fallback, not used.

**TODO (next):**
1. `python3 enrich_entity_csv.py --all --source csner` — rebuild ALL entity CSVs from the gazetteer.
2. Re-run **fixed** (and **pair**) extraction off them.
3. Re-run the KGE eval (fixed-only) — triples will have changed.

> ⚠ Existing `Entity_<id>_enriched.csv` from the old `--source llm` are a STOPGAP until overwritten by `--source csner`. `Claude.md` "Entity list source for the fixed extractor" updated to match.

---

## Session 13 — Fixed the 2 "broken" PDF parses + disk cleanup (2026-06-14)

**Disk:** deleted the old cached models `Qwen3-32B` (62G) and `Qwen3-30B-A3B` (57G) from `~/.cache/huggingface/hub/` — freed ~118GB (73G→191G free). Remaining: Qwen3-14B (extractor) + Qwen2.5-7B/14B-Instruct (judges, retired but kept).

**Re-parse of J13-4001 + D17-1028:** both were flagged broken (0 / 83 words) and assumed to have dead text layers. **Both PDFs actually have healthy text layers** (9k / 10.5k chars in first 3 pages) — the failures were two bugs in `parser/structure_builder.py`, now fixed:

1. **`is_heading` missed dotted headers (J13-4001).** Its numeric regex `^\d+(\.\d+)*\s+[A-Z]` accepts `1 Introduction`/`2.1 Model` but not `1. Introduction` (number-dot-space). J13 is an essay with dotted headers (`1. False and True Starts`, `3. Interpretation`) → 0 headings detected → all 95 body blocks dropped via the `if not current_section: continue` guard. **Fix:** added a dotted-numeric branch that also accepts `N. Title`, guarded against numbered *body list items* (`1. All morphemes are created equal.`) by requiring a short remainder (≤8 words) that doesn't end in sentence punctuation.

2. **Table over-detection suppressed body prose (D17-1028).** The borderless detector found 13 "tables" in a 7-page paper; their cell tokens built a 171-entry `table_cell_set`, and `is_table_data_paragraph(threshold=4)` then deleted real body paragraphs containing ≥4 incidental cell words (body 20.6k→8.4k chars). **Fix:** added a match-density gate — long paragraphs (>200 chars) are only suppressed when matches are dense (≤60 chars/match); genuine table-row dumps are short/dense and still caught.

**Verified:** J13-4001 0→5 sections / ~8.8k words; D17-1028 83→3.2k words / 10 sections. Regression-checked on 8 known-good papers (2020.acl-main.130, conll, emnlp, etc.): **0 spurious dotted-headings, section/char counts unchanged** — the new branches only fire on papers that need them. Both regenerated to the canonical `output/<id>/no-llm/output.html` (kg_main's default mode).

**Post-parse extraction results (ran open→enrich→fixed on both):**
- **D17-1028** — open extraction rich; fixed = **7 triples**. Fine: a short single-method paper. (Note: if re-enriching, regenerate the open `llm/triples.json` from the NEW parse first — the enriched CSV is harvested from it, and a stale one built on the old broken parse starves the subject pool.)
- **J13-4001 — EXCLUDE from the fixed/KGE pipeline (content mismatch, NOT a parse bug).** The parse is now correct (open extraction yields **31 real triples**), but the content is Jerry Hobbs' ACL *Lifetime Achievement essay* — a retrospective on computational semantics/abduction/knowledge representation (`Davidson proposed_by Donald Davidson`, `weighted abduction algorithm proposed_by Mark Stickel`, …). It has almost no experimental entities (datasets/models/metrics/tasks), so the enrichment quality filter salvaged only **5 vaguely-typed entities** and the fixed extractor's domain/range validation passed **0**. The CEO ontology simply has nothing to grab onto in a philosophical essay. It also has no valid citations (the S2 network pulled wrong "Bacillus" papers), so it contributes nothing to the KGE eval regardless. **Do not re-investigate** — it's the wrong *kind* of paper, not a pipeline fault.

---

## Session 12 — KG embedding eval is now the PRIMARY triple evaluation (2026-06-13)

**Decision:** the KG-embedding citation-prediction eval (`kg_transe_pipeline.py`, `--kge transe/complex/rotate`) is now the **main way to evaluate triples**. `kg_evaluate.py` (LLM-as-Judge faithfulness) is **retired** — keep its files, but don't rely on it going forward.
- Why: embedding eval measures whether the KG is *useful* (predicts held-out citations across the whole corpus); the judge only measured per-sentence faithfulness.
- These are two different things — don't confuse them. Embedding eval = corpus-level, one results file in `output/` (NOT per-paper). Judge = per-paper, under `output/<id>/kg/fixed/<model>/eval_summary.json`.

**First baseline run** (`--extractor fixed --model Qwen3-14B --kge all --epochs 500`, 22 papers w/ citations, 410 held-out edges, KG 3804 entities / 7931 triples):

| Model   | Hits@1 | Hits@5 | Hits@10 | MRR   |
|---------|--------|--------|---------|-------|
| TransE  | 0.091  | 0.136  | 0.227   | 0.140 |
| ComplEx | 0.000  | 0.182  | 0.364   | 0.085 |
| RotatE  | 0.091  | 0.182  | 0.318   | 0.155 |

→ **RotatE best overall (MRR 0.155); ComplEx best Hits@10 (0.364).** Well above random (~0.2% Hits@1 over 588 paper nodes). Results saved to `output/kge_fixed_results_{transe,complex,rotate}.json`. Run command:
`python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge all --epochs 500 --save-results output/kge_fixed_results.json`

> ⚠ **Evaluate ONLY the `fixed` extractor with KGE.** Open (`llm`) and `pair` extractions are NOT evaluated — do not run the embedding eval on them or compare extractors on this metric. (An open-vs-fixed comparison was run once and discarded; the open result files were deleted.)

---

## Session 11 — Bigger judge model + evaluator device_map fix (2026-06-13)

- **kg_evaluate.py (LLM-as-Judge) can use a bigger judge.** It loads the judge in 4-bit NF4, so `--model Qwen/Qwen2.5-14B-Instruct` (~9GB) fits the 20GB vGPU. Recommended over the default `Qwen2.5-7B-Instruct` — stronger, and a *different family* from the Qwen3-14B extractor (less self-judging bias). 32B still won't fit.
- **Fixed a latent CPU-offload bug:** `kg_evaluate.py` loaded the judge with `device_map="auto"` — the exact trap from Session 4 (silent CPU offload → "freeze"). Changed to `device_map={"": 0}` to match both extractors. Mattered more now since a 14B judge is likelier to trigger offload than the 7B.
- Pass `--entity-csv output/acl/<id>/Entity_<id>_enriched.csv` for `fixed` so the judge resolves abbreviations via aliases (more accurate verdicts). Run command:
  `python3 kg_evaluate.py --paper <id> --extractor fixed --model Qwen/Qwen2.5-14B-Instruct --entity-csv output/acl/<id>/Entity_<id>_enriched.csv`

---

## Session 10 — New "pair" extractor: closed entities, open relation (2026-06-12)

**New extraction mode requested:** fix BOTH subject and object to the entity list, let the LLM choose a FREE-FORM relation. Spectrum now:
- `llm`   : subject free,          relation free,        object free
- `fixed` : subject ∈ list,        relation ∈ ontology,  object free
- `pair`  : subject ∈ list,        relation FREE,        object ∈ list   ← NEW

**Built `kg_extraction/pair_extractor.py` → `EntityPairExtractor`:**
- Subclasses `FixedTripleExtractor` (reuses 4-bit model load, paragraph splitting, present-entity detection, `extract_from_sentences` loop). Overrides only the system/user prompt and the parser.
- Parser `_parse_pair_output`: keeps a triple only if BOTH subject and object resolve to a listed entity (`entity_set.match`), they differ, BOTH are named in the source sentence (`_subject_in_sentence`), and the free relation is ≤5 words. Output uses key `predicate` (the chosen free phrase) + `object_type="Entity"`.
- Skips paragraphs with <2 present entities (a pair needs two).
- Added `EXTRACTION_MODE` class attr on `FixedTripleExtractor` (default "fixed"); the pair subclass sets "pair" so triples are tagged correctly.

**Wiring:** exported in `kg_extraction/__init__.py`; `kg_main.py` got `--extractor pair` (and `pair` in `all`), shares the entity-CSV resolution/auto-resolution with `fixed` (generalized `ENTITY_EXTRACTORS = ("fixed","pair")`), and writes to `output/<paper>/kg/pair/<model>/triples.json`.

**Verified (no model load):** imports, CLI choice, and the parser unit test (kept a valid listed-entity pair, rejected a pair whose object wasn't listed). End-to-end LLM run not yet done.

**✅ END-TO-END VALIDATED 2026-06-15** (`kg_main.py --all --extractor pair --model Qwen/Qwen3-14B`):
- **34 papers, 434 pair triples** (avg ~12.8/paper; lower than fixed's 734, as expected since both ends are constrained). All `object_type="Entity"`.
- **3 zero-triple papers** — all explainable, not failures: J13-4001 (award essay, no entity pairs), C16-1036 + E17-3026 (genuinely short ~800–1000-word papers).
- Quality is good: both ends are real listed entities, relations are accurate and pair-appropriate — `BlueBERT outperforms BERT`, `MS-BERT pre-trained on 70,000 MS consult notes`, `classification system consists of {Random Forests, Gradient Boosting Trees, SVMs}`, `TextRunner uses Naïve Bayes classifier`.
- **Note:** the free relations are un-normalized → 210 "distinct" relations inflated by surface variants (`uses`/`use`, `is evaluated on`/`evaluated on`/`evaluates on`, `is based on`/`are based on`). Inherent to open-relation extraction. If pair triples ever feed something keyed on the relation string, add light normalization (lowercase, strip leading `is/are`, lemmatize). Pair is NOT KGE-evaluated (KGE = fixed only), so cosmetic for now.

> ⚠ VRAM: like `fixed`/`llm`, `pair` loads its own Qwen3-14B (~10GB). Do NOT run `--extractor all` on the 20GB vGPU — it would load llm+fixed+pair simultaneously and OOM. Run one LLM extractor at a time.

**Run it:** `python3 kg_main.py --paper C16-1036 --extractor pair --model Qwen/Qwen3-14B` (entity CSV auto-resolves).

---

## Session 9 — Generalize fixed-extractor prompt for multiple papers (2026-06-12)

**Decisions locked in this session:**
- **Triple shape:** subject = from the entity list, predicate = from the CEO ontology, **object = free text** (NOT constrained to the list). An entity→entity experiment was tried and **reverted** per the user — objects stay free.
- **No longer BERT-only.** The `fixed_extractor.py` system prompt was saturated with BERT/NLP examples (BERT, GLUE, SQuAD, MLM, NSP, ELMo, WordPiece, BooksCorpus, "Radford et al."), which biased it toward BERT-style papers.

**Change:** rewrote every illustrative example in `_make_fixed_system_prompt` + the `_CEO_SCHEMA` relation hints to **paper-agnostic placeholders** (`<Model>`, `<Method>`, `<Dataset>`, `<Task>`, `<Metric>`, "a named problem", etc.). The ontology domain/range RULES are unchanged — only the examples were genericized. Verified the built system prompt contains zero paper-specific tokens. Generic metric names (F1/Accuracy/BLEU/RMSE/WER) kept since they're domain-neutral.

> Left as-is (intentionally): BERT mentions in **code comments** and the garbled-section regex (`E[CLS]`, `[unused\d+]`) — those are defensive PDF-artifact detection, not part of the LLM prompt, and harmless for other papers.

**Not yet validated end-to-end on multiple papers** — prompt builds clean, syntax OK. Next: run `kg_main --all --extractor fixed` on a few diverse ACL papers and eyeball that the ontology predicates fire sensibly across domains (speech, NER, MT, etc.), not just BERT.

**✅ VALIDATED 2026-06-14** (via the existing corpus — Session 12's fixed run on 06-12/06-13 already used this generalized prompt, so no re-run needed; just inspected the on-disk triples):
- Corpus-wide: **734 fixed triples across 35 papers, 17 of the 22 CEO predicates firing**, well distributed — `uses` 21.7%, `addresses` 21.4%, `achieves` 19.1%, then `comprises`/`evaluatedon`/`comparesagainst`/`splitfrom`/`trainedon`/`encompasses`/`produces`/… The Session-8 `addresses → Concept` collapse is down to **12.5%** (was 75% on C16-1036 pre-filter).
- Spot-checked a diverse, non-BERT sample (clinical text, Twitter sentiment, word-sense induction, Chinese hypernymy, emotion-distribution meta-learning, type-driven composition, Chinese open relation extraction). The prompt is genuinely domain-adaptive: e.g. clinical paper → `BlueBERT trainedOn PubMed abstracts`, `MS-BERT trainedOn 70,000 MS consult notes`; relation-extraction paper → `TextRunner uses Naïve Bayes classifier`. Subjects are always real in-paper entities; no BERT/GLUE/SQuAD bias leaked.
- Known minor weakness (acceptable, not a regression): **short papers lean on one predicate** (E14-4003 = 8×`uses`/9; P17-1128 heavy on `addresses` with a repeated generic object). Expected when a paper genuinely only describes a method using components — not the Session-8 garbage collapse.

→ **Conclusion: the genericized prompt works across domains. This item is closed.**

---

## Session 8 — Quality filter on entity enrichment (2026-06-12)

**Why:** with the min-count-1 enriched pool, fixed extraction on C16-1036 gave 12 triples but ~quality was poor — **9/12 were `addresses → Concept`** with free-text objects, and several were unfaithful. Root cause: the free-LLM harvest had dumped vague descriptive phrases into the entity list ("human expressiveness", "concept of a continuous system", "ideal expressive TTS system", "new model", "present paper"), all typed `Concept`/`Other`. The extractor obeyed its constraint (all 12 subjects WERE in the list) — the list itself was the problem. Goal per user: not BERT precision, but *proper* use of the entity list + CEO ontology.

**Fix in `enrich_entity_csv.py` (quality filter, default ON):**
- Body entities kept only if their CEO type ∈ `HARD_CEO_TYPES` (Model/Method/Dataset/Tool/Metric/EvaluationMetric/Task/Resource/Language). Drops `Concept`/`Other`/`Generic`.
- `_is_noise` now also drops generic-leading phrases (`_GENERIC_LEADING`: the/a/this/new/present/proposed/concept/… as first word) and over-long phrases via `--max-words` (default 6).
- **Title seeds are exempt from both filters** — always kept.
- Escape hatches: `--keep-all-types`, `--max-words 0`.

**Result:** C16-1036 went 42 noisy → **21 proper ontology-typed entities**; all 8 junk subjects removed. Re-run fixed extraction to get cleaner ontology relations (uses/produces/trainedOn/…) instead of `addresses → Concept`.

---

## Session 7 — Auto-resolve entity CSV for fixed extraction (2026-06-12)

**Goal:** stop having to pass `--entity-csv` for fixed extraction — the CSV is deterministically tied to the paper, so resolve it from the paper id automatically.

**Changes in `kg_main.py`:**
- Added `resolve_entity_csv(paper_name)` — search order: `output/acl/<id>/Entity_<id>_enriched.csv` → `output/acl/<id>/Entity_<id>.csv` → project-root `Entity_<id>.csv` / `Entity-<id>v2.csv` / `Entity - <id>v2.csv`. Returns first hit.
- `--entity-csv` is now **optional**. If omitted (`auto_entity=True`), the fixed extractor is built with an empty placeholder `EntitySet([], {}, {})`, and `process_paper` resolves + loads the CSV **per paper**, swapping `extractors["fixed"].entity_set` before extracting. Works because `FixedTripleExtractor` reads `self.entity_set` at call time and its system prompt is relations-only (entities go in the per-call user prompt).
- If a paper has no resolvable CSV in auto mode, fixed extraction is skipped **for that paper only** (other extractors still run); explicit `--entity-csv` keeps the original single-set-for-all behaviour.

**Verified:** resolver returns enriched CSV for C16-1036/E17-3026, title-only for D17-1028 (no enriched), and the manual `Entity-BERTv2.csv` for BERT. Full extraction not run here (loads the LLM) but wiring is syntax-clean and the per-paper swap is in place.

**Usage now:**
```bash
python3 kg_main.py --paper C16-1036 --extractor fixed --model Qwen/Qwen3-14B   # no --entity-csv needed
python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B              # different CSV per paper
```

---

## Session 6 — Entity-CSV enrichment for fixed extraction (2026-06-12)

**Problem found:** the ACL/CS-NER entity CSVs only have ~2-3 entities each because **CS-NER annotates paper TITLES only** (verified: source files are one ~8.8-token title per record; 98 CSVs avg 2.15 entities). The fixed extractor is hard-gated on its subject list (`fixed_extractor.py` prompt: "Use ONLY these as subjects"), so a 2-entity CSV → ~0 triples. The user's main workflow is now ACL papers, so these CSVs must be enriched into a real subject pool.

**Built `enrich_entity_csv.py`:** keeps the ACL title entities as guaranteed-correct **seeds**, harvests real in-paper entities, dedupes (seeds win), and writes the standard 6-col CSV (`Entity,Abbreviation,Aliases,TP,NER_Type,CEO_Type`, all TP=1).
- Two sources via `--source`:
  - **`llm` (default)** — harvests subjects+objects from the open-extraction `triples.json` we already computed (`output/<id>/kg/llm/<model>/triples.json`). No install, no GPU; entities come pre-typed (Method/Task/Model/...). **This is why we did NOT install ITER** — ITER (`pip install git+https://github.com/fleonce/iter`) is not installed and could repin torch and undo the Session-4 fix.
  - **`iter`** — ITER/SciERC over `output/<id>/no-llm/output.html` (kept as an option; added `ITERExtractor.extract_entities()` to return all entities, not just those in a relation).
- `--in-place` overwrites `Entity_<id>.csv` (default writes `Entity_<id>_enriched.csv`); `--min-count N` trims one-off noise; `--max-entities N` caps.

**Result:** `--all` enriched 31/33 papers, **2-3 → 22-241 entities each**. Verified the output loads via `entity_loader.load_entity_csv(..., tp_only=True)`. (2 papers skipped: their llm triples.json was empty.)

**Use it:**
```bash
python3 enrich_entity_csv.py --all          # min-count 1 (default) — keep it here, see below
python3 kg_main.py --paper <id> --extractor fixed \
    --entity-csv output/acl/<id>/Entity_<id>_enriched.csv --model Qwen/Qwen3-14B
```

> ⚠ **Keep `--min-count` at 1.** A single paper rarely repeats an entity, so `--min-count 2+`
> deletes most of the pool. Session-7 hit this: C16-1036 was enriched with `--min-count 2`,
> dropped 41→9 entities, and fixed extraction produced only **1 triple**. Re-enriched at
> min-count 1 → 42 entities (35 matching text). Frequency filtering only makes sense across a
> corpus, not within one paper.

> Note on TP: the ACL CSVs are all `TP=1` by construction, so the `tp_only` filter in `entity_loader` is a no-op on them — the list length, not TP, was the real constraint.

---

## Session 5 — Added ComplEx + RotatE to KG embedding eval (2026-06-11)

**Goal:** evaluate KG embeddings with TransE, ComplEx, and RotatE (TransE already existed).

**Changes in `kg_transe_pipeline.py`:**
- Added `ComplEx` (complex bilinear, `Re(<h,r,conj(t)>)`) and `RotatE` (relation = unit-modulus rotation, score `-||h∘r - t||`) nn.Module classes alongside the existing `TransE`.
- For both, `entity_emb` stores the full `[real‖imag]` vector (width `2*dim`) so the existing `predict_related_papers`/`evaluate` (cosine sim on `entity_emb.weight`) work **unchanged** — they were already model-agnostic.
- Added `KGE_MODELS` registry + `build_kge_model(kge, ...)`. Renamed `train_transe` → `train_kge(graph, kge=..., ...)` (one training loop: margin-ranking loss + head/tail corruption negative sampling, used by all three).
- New CLI flag `--kge {transe,complex,rotate,all}` (default `transe`). `all` trains+evaluates all three and prints a Hits@1/5/10 + MRR comparison table.
- `--save-model` / `--save-results` get a `_{kge}` suffix when `--kge all`; checkpoints now store the `kge` name so `--load-model` rebuilds the right class.
- `--dim` is the **complex** dim for ComplEx/RotatE (entity storage is `2*dim`).

**To add a 4th model later:** register an nn.Module with `entity_emb` (full entity vector), `relation_emb`, `forward(h,r,t)→score` (higher=better), `loss(...)` in `KGE_MODELS`. Nothing else needs changing.

**Verified:** smoke test `--kge all --epochs 5 --device cpu --extractor llm --model Qwen3-14B` — all three train, evaluate (24 papers), and the comparison table prints. Run real evals with `--epochs 500` (and `--device cuda`).

> Note: `--model` = the LLM subdir (e.g. Qwen3-14B); `--kge` = the embedding model. Two different "models".

---

## Session 4 — "freeze" was a torch-nightly regression, NOT CPU offload (2026-06-10)

**Symptom:** `kg_main --extractor llm` appeared to *freeze right after the model loaded* — long silent gap, `nvidia-smi` showing GPU util 0% and one CPU core pegged at 100%. Looked exactly like the session-3 CPU-offload freeze (#4), but it was a different cause.

**What it was NOT (ruled out with evidence):**
- NOT CPU offload / wrong device — a clean load showed all 443 params on `cuda:0`, generation grew VRAM 9.6→10.8GB, and a profiler showed the work running as the CUDA kernel `kgemm_4bit_inference_naive` (`bitsandbytes::gemv_4bit`). It runs on GPU.
- NOT `llm_extractor.py` / `kg_main.py` — proven by reproducing the slowness in a from-scratch load that bypasses our code.
- NOT bitsandbytes itself — a 4-bit `Linear4bit` microbench was GPU-fast.
- NOT `python` vs `python3` as the perf cause — both resolve to the same `/usr/bin/python3.10` + same `~/.local` site-packages. (NOTE: always invoke commands as `python3` anyway — `python` does not work as expected for running the pipeline on this VM.)
- NOT a clock throttle — SM clock idles at 345MHz but boosts to ~1755MHz under load.

**Root cause:** the installed torch was a **dev nightly** (`2.12.0.dev20260408+cu128`) that had a regression crippling bitsandbytes 4-bit *decode* → ~**2 tok/s** (an H100 should do 30–60). Throughput work (big matmuls) was fine; only latency-bound token-by-token decode was destroyed.

**Fix:** replaced the whole nightly torch stack with **stable** builds:
```bash
pip uninstall -y torch torchvision torchaudio triton
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# → torch 2.6.0+cu124, torchvision 0.21.0, torchaudio 2.6.0, triton 3.2.0, cuDNN 9.1
```
cu124 wheels run fine on the 12.8 driver (CUDA is backward-compatible). After the swap: **~16 tok/s** (8× faster). Do NOT reinstall a torch nightly on this VM.

**CRITICAL gotcha — `nvidia-smi` GPU-util is broken on this H100-20C vGPU.** It reads **0% even when the GPU is fully busy** (verified: a sustained 4096³ matmul loop showed 0% util while the SM clock sat at 1.6GHz). So "GPU 0% + one CPU core at 100%" is **NOT** evidence of CPU execution here, and one core at 100% during decode is normal (kernel-launch dispatch). To tell if a run is alive, check instead:
- log advancing (`Graph: N nodes, M edges` per section)
- VRAM ~10–11GB (`nvidia-smi --query-gpu=memory.used`)
- SM clock ~1755MHz under load (`nvidia-smi --query-gpu=clocks.sm`)
- growing CPU time on the PID (`/proc/<pid>/stat` fields 14+15)

**Also:** at ~16 tok/s, kg_main's default `--max-new-tokens 4096` (and ×2=8192 in the postprocess pass, `llm_extractor.py:488`) means a single generation runs several silent minutes — looks frozen but isn't. Use `--max-new-tokens 512` for snappy, visibly-progressing runs. The first `generate()` after `loaded.` is silent for ~60–75s with no per-token logging — that gap is the thing most easily mistaken for a freeze.

---

## Future plan
- If Qwen3-14B quality is noticeably worse than 32B, options:
  - Try the non-gated `mistralai/Mistral-Small-3.1-22B-Instruct-2503` (~11–12GB, fits 20GB)
  - Request access to gated `google/gemma-3-27b-it` (closest to 32B quality that fits)
- Re-run the BERT precision evaluation (`kg_evaluate.py`) on the new model and compare against the documented 32B baseline.
- Cached models in `~/.cache/huggingface/hub/`: still have the old `Qwen3-32B` and `Qwen3-30B-A3B` (57GB) — can delete to free disk once 14B is confirmed working.
