# SciNex: Knowledge Graph Construction for Open Science

Turns scientific-paper PDFs into knowledge graphs of `(subject, predicate, object)` triples typed by
a research ontology (CEO or scinex). Then it measures whether those graphs are any good: per triple
(judged against the source sentence), per graph (citation prediction), and by simulated Q&A
against the real papers.

**All code and documentation are in [`project_scinex/`](project_scinex/). Run every command from that folder.**

---

## Start here

| If you want to… | read |
|---|---|
| **set the project up on a new machine** | [`project_scinex/SETUP_NEW_MACHINE.md`](project_scinex/SETUP_NEW_MACHINE.md) |
| **know where the work stands and what is next** | [`project_scinex/HANDOFF_NEXT_SESSION.md`](project_scinex/HANDOFF_NEXT_SESSION.md) |
| read the full write-up (pipeline, ontologies, evaluation, results, limitations) | [`project_scinex/RESULTS_REPORT.md`](project_scinex/RESULTS_REPORT.md) |
| see paper-ready result tables | [`project_scinex/results.md`](project_scinex/results.md) |
| see the finetuning + Q&A progress summary | [`project_scinex/SUMMARY_finetune_qa_2026-10-01.md`](project_scinex/SUMMARY_finetune_qa_2026-10-01.md) |
| understand the architecture and conventions | [`project_scinex/Claude.md`](project_scinex/Claude.md) |
| read the session-by-session log (every trap found) | [`project_scinex/hands_off.md`](project_scinex/hands_off.md) |
| see the 20-paper corpus | [`project_scinex/PAPERS.md`](project_scinex/PAPERS.md), `project_scinex/papers/manifest.csv` |
| use the PDF → HTML parser (Part 1, `main.py`) | [`project_scinex/README.md`](project_scinex/README.md) |

---

## The project in three parts

1. **Paper → knowledge graph** (built). PDF parser → paragraphs → an LLM extracts triples whose
   relation must come from the ontology → rule-based guards reject malformed triples.
2. **Finetune an open-weight LLM on the ontology and vocabulary** (current work). A small student
   (Qwen3, QLoRA) learns from a teacher model's stored extractions (GPT-OSS-120B), optionally
   augmented with triples that other models agree on. It is evaluated on papers it never trained on.
3. **Evaluate graphs by simulated Q&A** (tooling built, `qa_eval.py`). Questions verified against the
   paper text are answered from the graph alone and graded deterministically.

## Status (October 2026)

- **Extraction comparison, 20 papers, CEO ontology** (C1 concept correctness / C3 specificity /
  C4 relation correctness, ~40 judged triples per model): GPT-OSS-120B 88.8 / 80.0 / **86.2**,
  Gemma-4-31B 93.8 / 80.0 / 73.8, Qwen3-235B 75.0 / 60.0 / 47.5, Ministral-14B 70.0 / 52.5 / 41.2.
  The models differ most on relation typing (C4). At this sample size, gaps under ~10 points
  are not distinguishable. Details and caveats: `results.md`.
- **Finetuning:** pipeline built; the first laptop test (Qwen3-4B) raised the share of triples
  passing the guards from 38% to 86% on one held-out paper (the teacher: 88%). It still extracts
  less than the teacher. The full experiment (Qwen3-8B, teacher-only vs multi-model training data,
  blind judging) is next.
- **Earlier result (ACL corpus, citation prediction):** RotatE knowledge-graph embeddings predict
  held-out citation links with MRR ≈ 0.60 and Hits@10 ≈ 0.85 (5 seeds, held-out test papers).

## What is deliberately NOT in this repo

This repo is public. API keys, the paper PDFs (several are paywalled), and any file containing full
paper text (parsed HTML, prompt files, finetuning datasets, paragraph-level judging bundles) are
excluded by `project_scinex/.gitignore`. So are trained adapter weights (over GitHub's file-size
limit). `papers/manifest.csv` lists every paper's DOI so it can be obtained legitimately;
`SETUP_NEW_MACHINE.md` explains how to regenerate everything else.
