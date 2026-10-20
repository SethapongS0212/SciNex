# Handoff — next session

**Written at the end of Session 34 (work done 2026-10-01; docs re-checked 2026-10-20).**
**⚠ 2026-10-20: the training machine is now an RTX 5090 VM (32 GB, sm_120), NOT the DGX Spark.**
**⚠ 2026-10-20: the project now lives in GitHub, https://github.com/SethapongS0212/SciNex
(folder `project_scinex/`, PUBLIC).** New machine → **`SETUP_NEW_MACHINE.md`** first.
Files kept out of GitHub (PDFs, full-text files, adapters) are in the private
`SciNex_private_bundle.zip` (`hands_off.md` §34.14). The July snapshot held a credential file
(removed). Remind the user to confirm that key was revoked.
Read this first, then `hands_off.md` §34 (detail), then `TASK.md` top section (run order).
Older slide summary: `SUMMARY_finetune_qa_2026-10-01.md` (its top section covers Session 34).

## The project in one paragraph
SciNex, *Knowledge Graph Construction for Open Science*, has three parts: (1) paper → knowledge
graph (built: parser, extractors, guards, CEO + scinex ontologies); (2) **finetune an open-weight LLM
on the ontology + vocabulary** ← **CURRENT FOCUS**; (3) evaluate KGs by simulated Q&A against the
real papers (tooling built: `qa_eval.py`). In Part 2 a small student model is trained with QLoRA to
extract triples like the teacher. Success = on papers it never saw, the student's triples pass the
guards and are judged correct (C1/C3/C4) more often than the untuned model's, without extracting too little.

## Where things stand RIGHT NOW
- **Nothing is running on the laptop.** The user killed all local runs. Training will run on an
  **RTX 5090 VM** (changed from the DGX Spark on 2026-10-20). No results from it yet; if the user
  reports some, they supersede this file.
- **Training data, two variants, both built** (fold 0; on the laptop only 3 training papers had
  parsed text):
  - teacher-only: `ft_data/fold0/` (gptoss-120b targets);
  - **consensus**: `ft_data/consensus/fold0/` = gptoss targets + triples that ≥2 of gemma4-31b /
    qwen3-235b / ministral-14b agree on. Laptop build: **+66 triples on the teacher's 192 (+34%)**,
    empty targets 87 → 80. Built with `build_ft_data.py --consensus-models …` (`hands_off.md` §34.4).
- **Laptop test result (early signal, not a result):** teacher-only Qwen3-4B-Instruct-2507 adapter
  trained (`ft_runs/qwen3-4b-f0/adapter`). Extraction was killed after 44 of 51 paragraphs of
  strabismus2026 (CEO). **No collapse** (the loss fix works). Kept rate: base 38% → finetuned **86%**
  vs teacher 88%. **But it under-extracts:** triples in 8 paragraphs vs the teacher's 21 (§34.11).
  The consensus variant was never trained.
- **Judge decided:** **Claude Opus 5.5 (`claude-opus-5-5`) in session, BLIND, census**, via
  `blind_judge.py` (§34.8). Not comparable with the claude-opus-5 C1–C6 tables in `results.md`.
- **Q&A grading is deterministic** by default (token overlap + token-F1; `qa_eval.py grade
  --method overlap`, §34.9).

## Next steps, in order
1. **When the user brings 5090 VM results**, check them: collapse (finetuned `responses.jsonl` not all
   `{"triples": []}`), under-extraction (paragraphs with kept triples vs the teacher), and
   `ft_runs/qwen3-8b-f0-cons/compare.json`. Agreement-with-teacher favours the teacher-only model by
   construction, so don't read it as quality.
2. **Blind judge census here** (needs `output/<5 fold-0 papers>/kg/relation{,_scinex}/qwen3-8b-*/`
   copied back):
   ```
   python blind_judge.py build --run ft-f0 --ontology ceo \
          --papers csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023 \
          --slugs qwen3-8b-base-hf qwen3-8b-ft-cv qwen3-8b-ft-cons-cv gptoss-120b
   python blind_judge.py status --run ft-f0 --ontology ceo     # which batch is next
   #   judge ontology_eval/_blind/ft-f0/ceo/batch_NN.txt -> replies/reply_NN.txt (JSON verdicts)
   python blind_judge.py collect --run ft-f0 --ontology ceo
   python blind_judge.py report  --run ft-f0 --ontology ceo --compare qwen3-8b-ft-cons-cv qwen3-8b-ft-cv
   ```
   **Never open `key.csv` before `collect`**: it reveals which model made each triple. CEO first,
   then scinex. One reply file per batch, same JSON as `ontology_eval` triples.
3. Report the comparison (kept rate, coverage, C1/C3/C4, Q&A) and update `hands_off.md`,
   `Claude.md` and `results.md` with exact numbers.
4. Then 4-fold cross-validation with the winning variant (`--eval-fold 1..3`, the SAME slugs).

**If the user asks how to run on the 5090 VM:** `hands_off.md` §34.12 has the full plan and commands:
- copy code + `ft_data/` + `output/`, NOT `ft_runs/`;
- **torch must be a CUDA 12.8+ build** (sm_120). The old "2.6.0+cu124" rule fails on a 5090;
- bitsandbytes ≥0.49 (worked on the laptop's sm_120);
- `--allow-dev-torch` only if that torch is a nightly;
- `tmux`/`nohup` on Linux;
- rebuild data if all 20 papers are parsed there.

## Decisions already made — do not re-open
- **Claude's ONLY role is the in-session blind triple judge** (user, 2026-10-01). Claude is never an
  extractor, teacher, training-data source, Q&A model or API-called model; scripts refuse Claude ids
  there. Never spend OpenRouter credit on Claude.
- **Consensus training data: adopted as the second variant** (user: "if more model means better
  finetuning then do it"). Whether it IS better is the 5090 experiment.
- **Train and eval papers are disjoint by paper**, via the sticky `ft_data/split.json` (4 folds ×
  5; fold 0 = csysguard2024, linkpred2015, osmotic2026, strabismus2026, textaug2023). Never change
  `--folds`/`--seed`, never regenerate the file.
- Corpus = the user's 20 papers (`papers_20.txt`); size ON HOLD.
- Extraction line-up: qwen3-235b, gptoss-120b, gemma4-31b, ministral-14b. No Qwen3-14B. Judging is
  single-rater.
- `configures` excluded from training until the user decides its direction (§29.5).

## Open decisions for the user
1. Q&A question writer/answerer for reportable numbers. The pipeline default uses the base student,
   which makes it a pipeline test only. Needs an independent non-Claude model, possibly API spend.
2. qwen3-235b sentinel check (§34.3, how-to §34.10): 375/567 of its CEO replies are the
   failed-call string, which may invalidate its Table 10 coverage/C6. Caveat is in `results.md`.
3. `configures` direction. 4. `--verdict-filter` (off). 5. If the 5090 run under-extracts:
   `--max-empty-frac 0.3` as a third variant.

## Traps already hit — do not repeat
- **Windows EcoQoS throttles hidden/background processes ~4×** (40 vs 7–9 s/step).
  `run_ft_pipeline.no_power_throttling()` opts out the runner + children; `ft_runs/wait_then_run.py`
  for an already-running runner. Linux: not an issue.
- **Stopping a background task does not kill its Python child.** Kill by command line
  (`Get-CimInstance Win32_Process`). Session 33 wrongly believed a run was stopped (§34.5).
- **`{"triples": []}` is the API runner's failed-call marker** but a genuine answer from local
  models; `ft_common.is_api_run()` decides which.
- **Per-example loss averaging made the model always abstain** (run #1, collapsed adapter kept in
  `ft_runs/qwen3-4b-f0-collapsed-exampleweighted/`, do not use). Token-level loss is the fix.
- Laptop only (RTX 5070 8 GB, sm_120, torch cu128 nightly): SDPA falls back to MATH (7.27 GB/layer);
  `fast_attention(training=True)` forces cuDNN for training only, since cuDNN is 3× slower for
  generation. `--allow-dev-torch` there. Windows pages VRAM to RAM instead of OOM. Closing a console
  kills runs (`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`).
- Parse with the committed `citation_network.json` or paragraphs won't match the teacher's.
- `requirements_ft.txt`'s and `Claude.md`'s "torch 2.6.0+cu124" rule is for the old H100 vGPU only.
  On the RTX 5090 (sm_120) a cu124 wheel has no kernels for the GPU; use cu128+.
- `python` on the laptop (3.12) with `PYTHONIOENCODING=utf-8`; `python3` on Linux machines.

## Files added/changed in Session 34
`build_ft_data.py` (`--consensus-models`, `--consensus-min`; default out `ft_data/consensus/`) ·
`run_ft_pipeline.py` (`--data`, `--tag`, `--also-slugs`, shared base_gen, EcoQoS opt-out) ·
`qa_eval.py` (deterministic grading, token-F1 column) · **`blind_judge.py`** (new) ·
`ft_runs/wait_then_run.py` (new) · `ft_runs/launch_qwen3-4b-f0-cons.cmd` (laptop launcher, unused) ·
`ft_common.py` (refusal message).
