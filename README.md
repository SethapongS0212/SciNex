# SciNex: Knowledge Graph Construction for Open Science

Turns scientific-paper PDFs into knowledge graphs of `(subject, predicate, object)` triples typed by
a research ontology (CEO or scinex). Then it measures whether those graphs are any good: per triple
(judged against the source sentence), per graph (citation prediction), and by simulated Q&A
against the real papers.

**All code and documentation are in [`project_scinex/`](project_scinex/). Run every command from that folder.**

---

## Start here

> **⛔ Read [`project_scinex/OBJECTIVE.md`](project_scinex/OBJECTIVE.md) first.** The professor's
> objective: (1) research paper → knowledge graph; (2) finetune an LLM to understand the domain
> ontology and vocabulary, so it extracts with **only the entity list, no ontology in the prompt**;
> (3) evaluate the graph with **simulated Q&A against the real papers** (the main evaluation).

| If you want to… | read |
|---|---|
| **know the objective** | [`project_scinex/OBJECTIVE.md`](project_scinex/OBJECTIVE.md) |
| **know where the work stands and what is next** | [`project_scinex/HANDOFF_NEXT_SESSION.md`](project_scinex/HANDOFF_NEXT_SESSION.md) |
| set the project up on a new machine | [`project_scinex/SETUP_NEW_MACHINE.md`](project_scinex/SETUP_NEW_MACHINE.md) |
| see the result tables | [`project_scinex/results.md`](project_scinex/results.md) |
| understand the architecture and conventions | [`project_scinex/Claude.md`](project_scinex/Claude.md) |
| read the session-by-session log (every trap found) | [`project_scinex/hands_off.md`](project_scinex/hands_off.md) |
| use the PDF → HTML parser (step 1, `main.py`) | [`project_scinex/README.md`](project_scinex/README.md) |

---

## The project in three steps

1. **Paper → knowledge graph.** PDF parser → paragraphs → an extractor writes triples whose
   subjects come from the paper's fixed entity list → rule-based guards reject malformed triples.
2. **Finetune an open LLM to understand the ontology and vocabulary.** A local teacher
   (gpt-oss-20b) labels 805 ACL Anthology papers using the ontology; students (Qwen3, Llama-3.1,
   Gemma-3, Phi-4) learn from those answers plus ontology and vocabulary cards, and then extract
   **without the ontology in their prompt**. Tested on 41 papers they never saw.
3. **Evaluate graphs by simulated Q&A** (`qa_eval.py`): questions verified against the paper text
   are answered from the graph alone and graded deterministically. Per-triple blind judging is a
   diagnostic.

## Status (November 2026)

- Students that still read the full ontology prompt already match their teacher per triple
  (relation correctness 58 / 71 vs 45 / 55 on CEO / scinex, `results.md` Table 18). They are the
  baseline.
- **In progress:** ontology-free students of four model families, then the simulated Q&A
  evaluation, then a second teacher. See `project_scinex/HANDOFF_NEXT_SESSION.md`.

## What is deliberately NOT in this repo

This repo is public. API keys, the paper PDFs (several are paywalled), and any file containing full
paper text (parsed HTML, prompt files, finetuning datasets, paragraph-level judging bundles) are
excluded by `project_scinex/.gitignore`. So are trained adapter weights (over GitHub's file-size
limit). `papers/manifest.csv` lists every paper's DOI so it can be obtained legitimately;
`SETUP_NEW_MACHINE.md` explains how to regenerate everything else.
