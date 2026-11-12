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

## Status
- 2026-11-12 09:27 `screen scaleacq` (acquisition, ~4 h). 09:29 `screen scalep0` (curve point 0
  training on GPU0, `ft_data/acl_q30b_g2_noont_me30`: 34,320 extraction examples). GPU heat settings
  applied; root `gpurestore` waits for every `run_scale_*.sh` driver.
- vLLM + LoRA works (3 papers: 3.7 min); the 41-paper equivalence check is running on GPU1.

## Still to build
1. `scale_finalize.py`: usable-paper filter, test/train split file (`ft_data/scale/split.json`),
   corpus list `papers_scale.txt`.
2. Teacher driver for all new papers (both GPUs, ~1-2 days).
3. `qa_eval.py` concurrency for `--provider openai-compat` + vLLM-served Mistral; new run `scale`.
4. Curve drivers (build data per size, train, vLLM-generate on the test set, Q&A, report).
