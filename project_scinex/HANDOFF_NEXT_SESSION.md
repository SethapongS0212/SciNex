# Handoff — next session

**Written at the end of Session 36 (2026-10-26).**
**⭐⭐⭐ The fold-0 finetuning experiment is COMPLETE AND JUDGED.** Both data variants were trained,
generated and blind-judged on CEO. Headline: **a finetuned Qwen3-1.7B matches the 120B teacher on
per-triple quality (C4 85.4 vs 86.4) but extracts ~6× less.** Numbers: `results.md` Tables 13–14,
detail `hands_off.md` §36.12.
**Machine: Windows laptop, RTX 4060 Laptop GPU, 8 GB, sm_89**, stable torch `2.11.0+cu128` — no
nightly, so **never pass `--allow-dev-torch` here**. Student `Qwen/Qwen3-1.7B`, weights outside the
repo at `C:\Users\cei008\hf_local\Qwen3-1.7B`, passed as `--base-model <that path>`.
Read this file, then `hands_off.md` §36 (this machine and the result), §35 (setup), §34 (design).

## The project in one paragraph
SciNex, *Knowledge Graph Construction for Open Science*, has three parts: (1) paper → knowledge
graph (built: parser, extractors, guards, CEO + scinex ontologies); (2) **finetune an open-weight LLM
on the ontology + vocabulary** ← **CURRENT FOCUS, fold 0 now done**; (3) evaluate KGs by simulated
Q&A against the real papers (tooling built, needs an independent model). In Part 2 a small student is
trained with QLoRA to extract triples like the teacher (gptoss-120b's stored replies). Success = on
papers it never saw, the student's triples pass the guards and are judged correct (C1/C3/C4) more
often than the untuned model's, **without extracting too little** — and that last clause is exactly
where it currently fails.

## What is DONE (do not redo)
- **Setup on this machine, verified.** Python 3.12.10, venv, stable torch on sm_89, all data restored
  from an older local copy (§35.2). Install check reproduces the committed numbers (§35.4).
- **Both fold-0 datasets built** from 13 of 15 training papers: **619 extraction examples** (292 CEO /
  327 scinex) on 495 teacher triples, consensus **+129**; 3,285 examples → 822 steps (§35.6).
- **Both variants trained, generated and compared** (§36.10, §36.12): `qwen3-1.7b-ft-cv`
  (teacher-only) and `qwen3-1.7b-ft-cons-cv` (consensus), plus the base run `qwen3-1.7b-base-hf`,
  all on the 5 held-out fold-0 papers × both ontologies (348 paragraphs each).
- **Blind C1/C3/C4 census for CEO** (`ontology_eval/_blind/ft-f0/ceo/`): 355 model-triples → 315
  unique items, 13 batches, judged in session, `collect` + `report` done.
- **The EcoQoS trampoline bug found and fixed** (§36.11) — an 8× speed difference. Any new unattended
  run gets the fix automatically via `no_power_throttling_tree()`.

## The result in one table (CEO, census of 174 paragraphs per extractor)
| extractor | C1 | C3 | C4 | C4 strict | kept triples | coverage | guard kept rate |
|---|---|---|---|---|---|---|---|
| qwen3-1.7b-base-hf | 59.8 | 52.3 | **29.4** | 22.4 | 123 | 20.7% | 33.6% |
| qwen3-1.7b-ft-cv | 90.7 | 81.5 | **83.3** | 74.1 | 29 | 4.6% | 64.4% |
| **qwen3-1.7b-ft-cons-cv** | 90.2 | 87.8 | **85.4** | 78.0 | **47** | **7.5%** | **73.4%** |
| gptoss-120b (teacher) | 96.4 | 87.2 | **86.4** | 74.4 | 180 | 42.0% | 81.8% |
- **Quality: solved.** C4 INCORRECT falls from 63.6% of the base model's triples to ~7% (teacher 1.7%).
- **Teacher-only vs consensus: quality tie** (C4 +2.0 pts [−12.7, +16.8]), **consensus wins on volume**
  (+62% triples). **Adopt consensus.**
- **⛔ Recall is the open problem:** both students answer `{"triples": []}` for 90–95% of paragraphs.

## Next steps, in order
1. **scinex blind census** — the same thing already done for CEO, which is CEO-only so far:
   ```
   python blind_judge.py build --run ft-f0 --ontology scinex \
          --papers csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023 \
          --slugs qwen3-1.7b-base-hf qwen3-1.7b-ft-cv qwen3-1.7b-ft-cons-cv gptoss-120b
   python blind_judge.py status  --run ft-f0 --ontology scinex     # which batch is next
   #   judge ontology_eval/_blind/ft-f0/scinex/batch_NN.txt -> replies/reply_NN.txt (JSON verdicts)
   python blind_judge.py collect --run ft-f0 --ontology scinex
   python blind_judge.py report  --run ft-f0 --ontology scinex --compare qwen3-1.7b-ft-cons-cv qwen3-1.7b-ft-cv
   ```
   **Never open `key.csv` before `collect`.** Expect ~200 items (scinex volumes are lower than CEO).
2. **Train the third variant, which targets the actual failure** — data already built at
   `ft_data/maxempty30/` (310 extraction examples, 93 empty = 30%, 744 steps):
   ```
   python run_ft_pipeline.py --base-model C:\Users\cei008\hf_local\Qwen3-1.7B --prefix qwen3-1.7b \
       --fold 0 --papers csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023 \
       --gen-args=--batch-size=4 --tag me30 --data ft_data/maxempty30/fold0 \
       --also-slugs qwen3-1.7b-ft-cv qwen3-1.7b-ft-cons-cv --steps train ft_gen compare
   ```
   Slug `qwen3-1.7b-ft-me30-cv`. **The question it answers: does capping empty targets at 30% raise
   coverage without losing the C4 gain?** Budget ~1.5 h training + ~1 h generation at the unthrottled
   rates. Then add it to a fresh blind census.
3. **4-fold cross-validation with the winning variant** (`--eval-fold 1..3`, the SAME slugs), so every
   paper is scored by an adapter that never saw it. **~5.5 h per fold** now the throttling is fixed
   (it was ~20 h/fold before, §36.11) — `base_gen` is the bulk of it.
4. **Part 3 Q&A with an independent model** — the user has OpenRouter credit earmarked for this
   (§36.6). The pipeline default uses the student itself, which makes every Q&A number a pipeline
   test rather than a result. Grading is already deterministic (§34.9), so an API model is needed only
   to write and answer questions. ⛔ Never Claude through an API, and never OpenRouter credit on Claude.
5. **Guard fixes the census exposed** (§36.12) — concrete and cheap, all visible in the base model's
   output: reference-list bleed (`Hoonlor writtenBy Buyya`, `Snyder affiliatedWith <book title>`) is
   getting past the boilerplate guard; self-loops (`Belvèze et al. affiliatedWith Belvèze et al.`);
   acronym tautologies (`SMOTE uses Synthetic Minority Over-sampling Technique`); hyperparameters as
   datasets (`max_features trainedOn sqrt`); hardware as datasets (`cSysGuard trainedOn <Intel CPU>`);
   and a **prompt-template leak** (`RT1 evaluates '88.5 F1 on <Dataset>'`). A self-loop check and a
   reference-section skip would remove a visible slice of the base model's errors.

## Decisions already made — do not re-open
- **Student = `Qwen/Qwen3-1.7B` on this laptop** (user, 2026-10-20), chosen when the link was slow.
  ⚠ **Its numbers are NOT comparable with the Qwen3-4B figures of §33/§34** — name the model with
  every number. Qwen3-8B still cannot be TRAINED in 8 GB (§33.2). The link is now ~6.8 MB/s, so a 4B
  run for comparability is cheap if wanted (open decision 6).
- **Consensus is the better training-data variant** (same quality, +62% volume, §36.12).
- **Claude's ONLY role is the in-session blind triple judge** (user, 2026-10-01). Never an extractor,
  teacher, training-data source, Q&A model or API-called model; scripts refuse Claude ids there.
- **Judge = Claude Opus 5.5, BLIND, census** (§34.8). ⛔ **Not comparable with the claude-opus-5
  C1–C6 tables (Tables 10–12)** — different judge, never mix the two sets.
- **Train and eval papers are disjoint by paper**, via the sticky `ft_data/split.json` (4 folds × 5;
  fold 0 = csysguard2024, linkpred2015, osmotic2026, strabismus2026, textaug2023). Never change
  `--folds`/`--seed`, never regenerate the file.
- **Q&A grading is deterministic** by default (token overlap + token-F1; §34.9).
- Corpus = the user's 20 papers; size ON HOLD. Extraction line-up: qwen3-235b, gptoss-120b,
  gemma4-31b, ministral-14b. No Qwen3-14B. Judging is single-rater.
- `configures` excluded from training until the user decides its direction (§29.5).
- **Generation batch size stays 4.** Raising it to 6 made generation **2.85× SLOWER** (§36.8).

## Open decisions for the user
1. **Q&A writer/answerer model + spend** — see step 4. The user raised OpenRouter credit for exactly
   this and asked to discuss it once the local run finished (§36.6); that point has now arrived.
2. qwen3-235b sentinel check (§34.3, how-to §34.10): 375/567 of its CEO replies are the failed-call
   string, which may invalidate its Table 10 coverage/C6. The OpenRouter activity log settles it free.
3. `configures` direction. 4. `--verdict-filter` (off).
5. ✔ **Resolved:** `--max-empty-frac 0.3` is built and is step 2 above.
6. **Qwen3-4B (or 8B on a bigger GPU) for comparability with Sessions 33–34** — now a ~20-minute
   download at the current link speed.

## Traps already hit — do not repeat
- **⛔ Verify that a PID is the process doing the work.** A Python-Install-Manager venv's
  `python.exe` is a trampoline that re-execs the real interpreter as a child; opting only the
  launcher out of EcoQoS left training **8× slow** (§36.11). `nvidia-smi` showing **0% util at <10 W
  with no throttle reason** means the GPU is starved by the CPU side, not throttled.
- **Do not raise the generation batch size on an 8 GB Windows card.** Batch 6 was 2.85× slower than
  batch 4 on identical paragraphs (§36.8): residency hit 7.1 of 8 GB and Windows pages VRAM to system
  RAM silently instead of raising OOM. Predict VRAM from `nvidia-smi` residency, not from a KV-cache
  calculation, and A/B any change on the same paragraphs.
- **Every wait in an unattended runner must be progress-based, and readiness must be checked BEFORE
  any deadline.** A laptop sleep (lid close — a keep-awake request does not prevent it) made the
  Session 35 chain fail a step that had already succeeded (§36.1).
- **Single-`*` gitignore patterns stop matching a deeper directory.** `ft_data/*/train.jsonl` did not
  cover `ft_data/consensus/fold0/train.jsonl`, leaving 11 MB of paper text unignored in this PUBLIC
  repo (never committed; fixed to `ft_data/**/…`, §35.8). Run `git check-ignore -v <path>` whenever a
  script writes to a new, deeper directory, and always read `git status` before `git add -A`.
- **Throughput from the first few batches of a run is worthless.** The first 12 paragraphs suggested
  0.67 min each; the true average over 311 was **2.40** (§36.9). Difference the cumulative timer over
  whole jobs instead.
- **Python 3.14 is not usable** — no torch/bitsandbytes cp314 wheels. `py install 3.12`.
- **Check an older local copy of the project before chasing the private bundle** (§35.2).
- **`prompts.jsonl` changes which papers are usable** — with it, 13 of 15 fold-0 training papers pair;
  without it, rendering lost two more. Any run must state its paragraph source (`prompt_meta.json`).
- **Stopping a background task does not kill its Python child** (or its `curl`). Kill by command line
  (`Get-CimInstance Win32_Process`).
- **`{"triples": []}` is the API runner's failed-call marker** but a genuine answer from local models;
  `ft_common.is_api_run()` decides which.
- **Per-example loss averaging made the model always abstain** (§33.5b; collapsed adapter kept in
  `ft_runs/qwen3-4b-f0-collapsed-exampleweighted/`, do not use). Token-level loss is the fix.
- **sm_120 only:** SDPA falls back to MATH and `fast_attention(training=True)` forces cuDNN for
  training, never generation. On sm_89 + stable torch this does not bite — training the 1.7B peaked
  at 2.49 GB (§36.5).
- Closing a console kills runs (`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`). Parse with the committed
  `citation_network.json` or paragraphs will not match the teacher's. `python` on Windows (3.12 venv)
  with `PYTHONIOENCODING=utf-8`; `python3` on Linux.

## Files added/changed in Session 36
`run_ft_pipeline.py` (**`no_power_throttling_tree()`** + its use in every step) ·
`ft_runs/ecoqos_watchdog.py` (new) · `ft_runs/chain_session35.py` (new, machine-local unattended
runner; hardcoded local paths) · `ft_data/maxempty30/` (third variant, built not trained) ·
`ft_runs/qwen3-1.7b-f0/` + `ft_runs/qwen3-1.7b-f0-cons/` (run manifests; adapters gitignored) ·
`output/<5 papers>/kg/relation{,_scinex}/qwen3-1.7b-{base-hf,ft-cv,ft-cons-cv}/` (replies + triples) ·
`ontology_eval/_blind/ft-f0/ceo/` (the whole census) · `hands_off.md` §36 · `Claude.md` STATUS ·
`results.md` Tables 13–14 · this file · `SETUP_NEW_MACHINE.md` · `TASK.md` ·
`SUMMARY_finetune_qa_2026-10-01.md` · root `README.md`.
