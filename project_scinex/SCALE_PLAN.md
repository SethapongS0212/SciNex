# Scale-up plan (Session 39, started 2026-11-12)

User, 2026-11-12: *"lets go to 5000+ papers, and the test set should reach at least 1000+. we have
so many weeks before the meeting so i want to expand it to be as big as possible. and do what you
have to do with the results"*.

## Question
Does the ontology-free finetuned extractor keep its quality (simulated Q&A = main evaluation;
blind C1/C3/C4 = correctness diagnostic) when the corpus grows from 805 to 5,000+ papers, and does
more training data help? Measured on a test set of 1,000+ held-out papers.

## Design (decisions so far)
- **Corpus:** the 805 legacy ACL papers + new ACL Anthology papers, 2010-2023, drawn by
  `scale_select.py` (seed 2026) so the venue mix (main / workshop / LREC / SemEval / Findings)
  matches the 805-paper corpus. 7,500 candidates (`scale/candidates.txt`) to absorb download/parse
  failures; the first usable ones in that order are kept.
- **Acquisition** (`scale_acquire.py`, driver `run_scale_acquire.sh`, CPU + network): PDF download
  (1 s between requests) → `main.py --no-llm --no-citations` (8 workers; ~4 s/paper; output moved
  `no-llm_no-citations/` → `no-llm/`) → CS-NER gazetteer entity list (same defaults as the 805; no
  title seeds for new papers) → `build_entity_v2.py` v2 lists. The 805 papers' files are never
  touched (the batch enricher's `--all` would overwrite them, so new papers go through an
  in-process loop).
- **Test set:** the 41 legacy fold-0 papers (with their 1,222 questions, for continuity with
  Tables 20-23) + ≥ 1,000 new papers drawn at random from the usable pool. **Train:** the 764 legacy
  training papers + all remaining new papers (target ≥ 5,000 training papers in total).
- **Teacher:** Qwen3-30B-A3B-Instruct-2507 W4A16 via vLLM (`ft_generate.py --backend vllm`), the
  same `recall` prompt and v2 entity lists as `qwen3-30b-a3b-v2`, **guards v2** (`--guards v2`).
- **Students:** Qwen3-8B QLoRA, ontology-free prompt, max 30% empty targets, 1 epoch: a **learning
  curve** on nested training sets — 764 (curve point 0, running) → ~1,500 → ~3,000 → all (~5,000+).
- **Student generation on 1,000+ papers:** vLLM + LoRA (`--backend vllm --adapter`; bf16 base —
  vLLM 0.32 has no bitsandbytes). The HF path (~10 min/paper) would take weeks. Equivalence with
  the HF path is being checked on the 41 legacy test papers before use.
- **Q&A on the new test set:** Mistral-Small-24B served by vLLM (OpenAI-compatible server) for
  question generation and answering; questions per paper capped so the run stays tractable. The
  legacy 41-paper run is re-answered with the same backend for comparability.
- **Correctness:** blind C1/C3/C4 rounds (in-session judge) on samples from the largest student,
  the curve-point-0 student and the teacher.

## Status and pipeline (all unattended; every driver resumes when re-run)
| screen | driver (`/home/dev1/scinex_runs/`, copies in `ft_runs/acl_scripts/`) | does |
|---|---|---|
| `scaleacq` | `run_scale_acquire.sh` | ✔ done 2026-11-12 21:59: 7,500 candidates → PDF 7,413 → parsed 7,409 → v2 list 7,396 |
| `scalep0` | `run_scale_p0.sh` | ✔ done 2026-11-12 14:26: curve point 0 (764 legacy papers, guards-v2 Qwen3-30B data) |
| `scaleteacher` | `run_scale_teacher.sh` | ✔ split: usable new 7,272 (excluded no_html 91, small_html 109, few_entities 28) → **test 1,041 (41 legacy + 1,000 new), train 7,036 (764 legacy + 6,272 new)**; teacher on all 7,272 new papers 2026-11-12 22:00 → 11-13 18:09 |
| `scalecurve` | `run_scale_curve.sh` | waits for the teacher; GPU0: gen n764 → Mistral server (questions for run `scale`, ≤ 10/paper; controls; teacher answers) → train full → gen full; GPU1: train 1500 → gen → train 3000 → gen; then Q&A for every student (`scale`, `acl-f0-vllm`), report, `ft_compare` |
| `scaleinterim` | `run_scale_qa_interim.sh` | (Session 40) waits for gen n3000, then Mistral on GPU1 (port 8012): Q&A for n764/1500/3000 on `scale` + `acl-f0-vllm` → `qa_eval/<run>/report_interim.json`, `ft_runs/scale/compare_interim.json`. The final pass then only adds nfull |
GPU heat settings applied 2026-11-12 09:29; root `gpurestore` (`gpu_restore_after.sh 'run_scale_[[:alnum:]_]+[.]sh'`)
restores them 10 min after the last scale driver exits. Logs: `scale_driver.log`, `scale_acquire*.log`,
`scale_teacher_gpu{0,1}.log`, `scale_train_*.log`, `scale_gen_*.log`, `scale_qa.log`, `scale_report_*.log`.
**Results land in:** `qa_eval/scale/report_scale.json`, `qa_eval/acl-f0-vllm/report_scale.json`,
`ft_runs/scale/compare_curve.json`. Then: write results.md tables, a blind C1/C3/C4 round on the
largest student vs n764 vs teacher, and update the docs.

Rough timeline: acquisition done ~2026-11-12 16:30; teacher ~1-1.5 days; curve ~2 days
(train full ≈ 33 h is the longest step). Expected done ≈ 2026-11-16.

## Backend checks (2026-11-12) — both pass, both adopted for the whole scale-up
- **Student generation via vLLM + LoRA (bf16 base)** vs the HF NF4 path, Qwen3-30B-data student on
  the 41 legacy test papers: 56 min vs 6.5 h. Kept 2,446 / 2,436 vs 2,429 / 2,527 (CEO / scinex),
  coverage −5 points, truncations 49 / 65 vs 34 / 35 (the student writes 45-51 triples on dense
  paragraphs and hits the 3,072-token cap; kept unchanged for comparability). **Q&A (same HF
  answerer): +0.4 [−1.4, 2.2] / −1.3 [−2.8, 0.2] — equivalent.** Slug `qwen3-8b-acl-q30b-noont-me30-vllm`.
- **Cross-paper batching** (`ft_generate.py --vllm-group 16`, `--max-model-len 6144`, util 0.92): same
  student, 41 papers: 10.9 min vs 56 min, kept 2,464 / 2,451 vs 2,446 / 2,436 — equivalent, 5× faster
  (≈ 4.6 h per student on the 1,041 test papers). Used by the teacher and every student.
- **Q&A answerer via vLLM** (`RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w4a16`,
  `serve_mistral.sh`, `qa_eval.py --provider openai-compat --concurrency 32`): 1,222 questions × 2
  KGs in 99 s vs ~25 min. Same score on 96% of questions; abstains more (answered 29 / 33% vs 36 /
  40%), accuracy −1.7 / −1.8 points vs the HF NF4 answerer. A small systematic shift → **every
  scale-up Q&A uses the vLLM answerer**, and the legacy run is re-answered with it as
  `qa_eval/acl-f0-vllm` (same questions) to bridge old and new tables.

## Progress (2026-11-14 07:40 server, Session 40)
| step | status |
|---|---|
| questions, run `scale` | ✔ 10,402 (≤ 10/paper) from 25,388 paragraphs; controls + teacher answered (11-14 00:58) |
| teacher on test set (Q&A) | ✔ 11.1 / 12.0 (oracle 86.3, closedbook 5.6) — `results.md` Table 25 (interim) |
| n764 gen | ✔ 11-13 22:07 (GPU0, 3.9 h) |
| n1500 train → gen | ✔ train 8.5 h (11-14 02:41), gen 3.3 h (06:02) |
| n3000 train → gen | ⏳ train started 06:03, 18,645 steps × 3.2 s → ~22:30; gen → ~11-15 02:00 (GPU1) |
| nfull train → gen | ⏳ train started 11-14 01:00, 41,872 steps × 3.2 s → **~11-15 15:30**; gen → ~19:30 (GPU0) |
| interim Q&A n764/1500/3000 | ⏳ `scaleinterim`, ~11-15 02:00 → ~03:30 |
| final Q&A + report + ft_compare | ⏳ ~11-15 19:30 → ~21:00 server (≈ 11-16 04:00 the user's local time) |

nfull training data: 315,958 extraction examples (94,787 empty targets = the 30% cap), 1,691
ontology + 17,325 vocabulary cards; n1500 67,304; n3000 134,434.

## Built (2026-11-12)
1. ✔ `scale_finalize.py` (runs inside `run_scale_teacher.sh`). ✔ teacher driver `run_scale_teacher.sh`
   (`screen scaleteacher`, waits for acquisition + p0). ✔ `qa_eval.py --concurrency`.
2. Q&A question generation for the scale test set (run `scale`, vLLM Mistral, capped per paper).
3. Curve drivers (build data per size, train, vLLM-generate on the test set, Q&A, report).
