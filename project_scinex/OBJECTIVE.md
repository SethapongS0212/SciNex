# ⛔ OBJECTIVE — read this first, every session

Set by the professor; confirmed with the user on 2026-11-06. **Every task serves these three
steps. If a task does not, stop and ask the user.**

> **Knowledge Graph Construction for Open Science**
> 1. **Research paper → Knowledge Graph.**
> 2. **Finetune an LLM to understand the domain ontology and the vocabulary corpus.**
> 3. **Evaluate the Knowledge Graph with simulated Q&A against the real papers.**

## What this means in practice

- **The finetuned model extracts triples WITHOUT the ontology in its prompt.** At inference it
  gets only the paper's **entity list** and the **text**. The ontology (CEO or scinex relations,
  their direction and typing) and the vocabulary must be **learned into the weights** by
  finetuning. A model that needs the ontology spelled out in every prompt has not met step 2.
  - Code: `--prompt-variant noont` (`ft_common.noont_system_prompt`), for BOTH the training data
    (`build_ft_data.py`) and inference (`ft_generate.py`). The prompt names the ontology only.
  - Training data: the teacher's ontology-constrained answers + ontology cards (relation
    definitions, direction, choice) + vocabulary cards (term types).
- **Teacher → student distillation is only the means of making training data**, not the
  contribution. Several teachers may be compared (does the teacher change what the student
  learns?), but the deliverable is the ontology-free extractor and its KG.
- **The MAIN evaluation is simulated Q&A** (`qa_eval.py`): questions written from the real papers,
  answered from each KG, graded against the paper. Per-triple blind judging (C1/C3/C4,
  `blind_judge.py`) and judge-free counts (`ft_compare.py`) are **diagnostics**, not the headline.
- **Corpus:** the 805 ACL Anthology papers with fixed entity lists (`papers_acl.txt`,
  `output/acl/<id>/Entity_<id>_enriched.csv`), fold 0 of `ft_data/acl/split.json` held out
  (41 papers). Not the old 20-paper corpus.
- **Extractor:** fixed (subjects only from the entity list). Ontologies: CEO and scinex.

## Do not mix up

| Not the objective (earlier phases, finished) | The objective |
|---|---|
| Students that read the full ontology prompt (~2,700 tokens): Tables 16–18 in `results.md` | Ontology-free students (`noont`) |
| Triple-level C1/C3/C4 as the headline | Simulated Q&A as the headline |
| The 20-paper corpus (Sessions ≤ 36) | 805 ACL papers |
| Claude as the extractor | Open models only (teacher local, students finetuned) |

Where the work stands: `HANDOFF_NEXT_SESSION.md`. Session log: `hands_off.md`. Tables: `results.md`.
