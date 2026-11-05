# Handoff — next session

**Written at the end of Session 37 (2026-11-05).** Read this, then `hands_off.md` §37 (37.2 for
the corpus decision, 37.7–37.8 for the hardware incident and the results).

## ⛔⛔ The finetuning runs use the ACL ANTHOLOGY papers with the FIXED extractor
User decision, 2026-10-26: *"we were supposed to be using the acl anthology paper, not the paywalled
professor noted papers anymore … since we have fixed entity for that"*. Part 2 now runs on **805 ACL
papers** (`papers_acl.txt`) with **`--extractor fixed`**: subjects come only from each paper's
`output/acl/<id>/Entity_<id>_enriched.csv`, and outputs go to `kg/fixed{,_scinex}/<slug>/`.
**Do not start new finetuning runs on the 20-paper corpus.** Its results (Tables 1–14) stand as they
are.

## Machine
Linux box `kubegpu-g04`, **2× RTX 5090 (32 GB, sm_120)**, 125 GB RAM. venv **`/home/dev1/envsam`**
(Python 3.12.3, stable torch `2.13.0+cu130`, so **no `--allow-dev-torch`**; plus `kernels==0.12.3`
for gpt-oss and `peft==0.21.1`). Repo at `/home/dev1/SciNex`; `/home/dev1/project_scinex` is a
symlink to its `project_scinex/`. Driver scripts: `/home/dev1/scinex_runs/` (copies in
`ft_runs/acl_scripts/`). The July ACL copy (PDFs, parsed HTML, entity CSVs) is in
`/home/dev1/project_scinex_old_2026-07`. Use `python` from the venv; long runs go in `screen`.

## The result (`results.md` Tables 15–17)
**Teacher** `gpt-oss-20b` (local, MXFP4, reasoning low) over all 805 papers × both ontologies:
37,970 paragraphs in 30.6 h. Valid JSON 99.5% (CEO) / 99.6% (scinex); kept triples 8,488 / 5,479
(79.6% / 78.4% pass the guards); coverage 22.8% / 16.4%. About 80% of its answers are empty, and
`uses` is 65% of its CEO triples.

**Students** (Qwen3-8B QLoRA, 1 epoch) on the **41 held-out papers** (980 paragraphs per ontology),
judge-free counts:

| CEO | valid JSON | proposed → kept | kept rate | coverage |
|---|---|---|---|---|
| `qwen3-8b-acl-base-hf` (untuned) | 95.3% | 6,139 → 2,363 | 38.5% | 74.5% |
| `qwen3-8b-acl-ft-cv` (teacher-only data, 80.5% empty) | 100% | 15 → 15 | 100% | **0.8% (collapsed)** |
| **`qwen3-8b-acl-ft-me30-cv` (empty targets ≤ 30%)** | **99.9%** | **820 → 670** | **81.7%** | **43.8%** |
| `gptoss-20b` (teacher) | 99.2% | 619 → 501 | 80.9% | 25.6% |

**⭐ Blind C1/C3/C4 judging (2026-11-05)**, 50 sampled kept triples per model and ontology, judged
by Claude in session (`ontology_eval/_blind/acl-f0/{ceo,scinex}/`):

| | CEO C1 | CEO C3 | **CEO C4** | scinex C1 | scinex C3 | **scinex C4** |
|---|---|---|---|---|---|---|
| base (untuned) | 60.0 | 30.0 | **23.0** | 66.0 | 28.0 | **19.0** |
| **me30 student** | 90.0 | 28.0 | **51.0** | 90.0 | 32.0 | **61.0** |
| gptoss-20b teacher | 80.0 | 30.0 | **49.0** | 81.0 | 36.0 | **62.0** |

- **The student is as correct as its teacher** (C4 +2.0 [−14, +17] CEO, −1.0 [−17, +14] scinex) and
  far better than the untuned model (+28 [+13, +42] and +42 [+30, +55]).
- **Its extra triples are not junk.** 55% of its CEO triples sit on paragraphs where the teacher
  found nothing. Judged by paragraph type: shared paragraphs C4 62.5 (n=16), teacher-empty
  paragraphs C4 45.6 (n=34), against the teacher's own 49.0.
- **The ceiling is the teacher and the entity lists:** only ~50–60% of the teacher's relations are
  correct, and specificity (C3) is ~30% for every model, because the generic CS-NER entries
  ("Models", "Experiments") become subjects.

## State
- **⏳ IN FLIGHT: milder caps, `screen -r scinexcaps`** (`/home/dev1/scinex_runs/run_acl_caps.sh`,
  copy in `ft_runs/acl_scripts/`), started 2026-11-05 12:13 server time (the server clock runs ~7 h
  behind the user's). GPU0 trains `ft_data/acl_maxempty50/fold0` → `qwen3-8b-acl-ft-me50-cv`
  (2,870 steps); GPU1 trains `ft_data/acl_maxempty60/fold0` → `qwen3-8b-acl-ft-me60-cv` (3,290
  steps). Both use `--save-steps 200` and then ft_gen. **Expected done ~19:30–20:00 server time.**
  Then `ft_compare` writes `/home/dev1/scinex_runs/acl_compare_caps.log` and
  `ft_runs/qwen3-8b-acl-f0/compare_caps.json`. Progress: `/home/dev1/scinex_runs/acl_driver.log`.
  Re-running the script resumes.
- **GPU power is capped at 400 W** (user, `sudo nvidia-smi -pl 400`), because both cards hit 89–90 °C
  at full power. Without persistence mode the cap can reset when the driver unloads, so re-apply it
  before long jobs.
- Adapters (gitignored): `ft_runs/qwen3-8b-acl-f0/adapter/` (teacher-only, **recovered from
  `checkpoint-5493`**; see its `run_manifest.json`) and `ft_runs/qwen3-8b-acl-f0-me30/adapter/`.
- Presentation page: private claude.ai artifact "SciNex ACL Finetuning Results"
  (https://claude.ai/artifact/YEfrmYnhmqMQxNAAio3tEt), updated with the judged numbers.
- **GitHub pushes work from this server** with the deploy key `~/.ssh/scinex_deploy_ed25519`
  (write access to SciNex only): `GIT_SSH_COMMAND="ssh -i ~/.ssh/scinex_deploy_ed25519 -o
  IdentitiesOnly=yes" git push git@github.com:SethapongS0212/SciNex.git main`.

## Next steps, in order (set by the user on 2026-11-05 for the next session)
The presentation is at **07:00 user time = ~00:00 server time** (the server clock runs ~7 h behind).

**Before the presentation (≈ 2 h of work, start when the caps run is done, ~19:30–20:00 server):**
1. **Check the caps run** (5 min). `tail /home/dev1/scinex_runs/acl_driver.log` should end with
   "caps run done". Read `/home/dev1/scinex_runs/acl_compare_caps.log`: pick the cap whose coverage
   is closest to the teacher's (CEO 25.6%, scinex 17.3%) without collapsing. If a lane FAILED, re-run
   `screen -dmS scinexcaps /home/dev1/scinex_runs/run_acl_caps.sh`; it resumes from the step
   checkpoints. If it cannot finish in time, the presentation uses the current page as it stands.
2. **Blind-judge me50 and me60, CEO first** (~45 min per ontology): build `acl-f0-caps` with fresh
   base/teacher decoys (commands below), judge every batch, then `collect` + `report`. Do scinex
   too if there is time.
   ```
   python blind_judge.py build --run acl-f0-caps --ontology ceo --extractor fixed --max-per-slug 50 --seed 29 \
       --papers $(cat ft_data/acl/fold0/eval_papers.txt) \
       --slugs qwen3-8b-acl-ft-me50-cv qwen3-8b-acl-ft-me60-cv qwen3-8b-acl-base-hf gptoss-20b
   python blind_judge.py collect --run acl-f0-caps --ontology ceo
   python blind_judge.py report  --run acl-f0-caps --ontology ceo \
       --compare qwen3-8b-acl-ft-me50-cv gptoss-20b --compare qwen3-8b-acl-ft-me60-cv gptoss-20b
   ```
   **Never open `key.csv` before `collect`.**
3. **Update and publish** (~20 min): `results.md` Table 18 (counts from `compare_caps.json` +
   judged C1/C3/C4), the presentation page (artifact slug `fcf3ea23-a2e2-46a8-836b-401a7d74e2e1`,
   file `/home/dev1/scinex_runs/presentation/acl_results.html`; add the me50/me60 rows, and change
   the headline only if a cap clearly matches the teacher's coverage and correctness),
   `hands_off.md` §37.11, then commit + push with the deploy key (see State).

**After the presentation:**
4. **Guard fixes for errors the judging exposed in every model** (CPU only): drop the leaked
   template placeholder `88.5` (an object not in the sentence), self-loops (subject == object),
   and objects absent from the source sentence. Then re-run `ft_compare.py --extractor fixed`.
   Not before the presentation, so the numbers shown stay fixed.
5. **Entity-list cleanup** (the C3 ceiling): remove generic CS-NER entries ("Models",
   "Experiments", "Tagger") from `output/acl/*/Entity_*_enriched.csv`, re-run the teacher (~30 h),
   and retrain the winning cap.
6. **A stronger teacher** (the C4 ceiling): it needs OpenRouter credit/key or more GPU memory.
   That is the user's decision.
7. **More folds** (2–3 of the 20 in `ft_data/acl/split.json`) with the winning cap.
8. **Part 3 Q&A** with an independent, non-Claude model.

## Decisions already made — do not re-open
- **Corpus = 805 ACL papers, extractor = fixed** (user, 2026-10-26). **Teacher = gpt-oss-20b run
  locally; corpus = all parsed ACL papers with an entity CSV** (user, 2026-10-27). **Student =
  `Qwen/Qwen3-8B`.**
- **ACL split = `ft_data/acl/split.json`** (20 folds, seed 13; fold 0 = 41 eval papers). It is as
  sticky as the 20-paper `ft_data/split.json`, which stays untouched. Never regenerate either file.
- **Claude's only role is the in-session blind triple judge**: never an extractor, teacher,
  training-data source or API-called model. **Reaffirmed by the user on 2026-11-05**: Claude as an
  extractor (judged by other models) was discussed and dropped. Using Claude outputs as training
  data would also need checking against Anthropic's terms.
- Generation batch size 8 on the 5090 (measured); 1 epoch on the ACL data.

## Traps hit in Session 37 — do not repeat
- **"No new process can start CUDA" after a GPU fault = the shared `nvidia_uvm` module is in an
  error state** (`/dev/nvidia-uvm` → EIO). Fix it without a reboot: `sudo rmmod nvidia_uvm && sudo
  modprobe nvidia_uvm`. Never touch a card `nvidia-smi` cannot see.
- **GPU heat:** under load GPU0 ran at 89–90 °C, fan 100%, thermally throttled. GPU1 died under the
  same load. Check the cooling before loading both cards for days.
- **A 1-epoch run with `save_strategy="epoch"` has no checkpoint.** Use `ft_train_qlora.py
  --save-steps 200`; a re-run resumes from the latest checkpoint (smoke-tested).
- **A local teacher's `{"triples": []}` is a genuine answer, not a failed API call.**
  `build_ft_data` now asks `is_api_run()`. Before the fix, every empty target would have been dropped.
- **The July ACL HTML keeps PDF hyphenation**, so the strict alignment check rejected correctly
  paired papers. Fixed builds use `alignment_rate(..., loose=True)`.
- **A `--limit` smoke run writes a truncated `prompts.jsonl` that later runs pick up.** Delete smoke
  slug folders.
- gpt-oss needs `kernels==0.12.3` (0.17 breaks transformers 5.5.4) and `--attn-implementation
  flex_attention` (eager attention runs out of memory at batch 8).
- `pkill -f` / `pgrep -f` with a pattern contained in your own command line matches your own shell.
  `nvidia-smi -L` takes no `-i`; use `-i N --query-gpu=name --format=csv,noheader`.

---

## History — the Session 36 handoff (20-paper corpus, Windows laptop)

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
