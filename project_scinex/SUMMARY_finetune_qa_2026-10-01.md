# Finetuning & Q&A Evaluation — Progress Summary

## ⭐⭐⭐ UPDATE — Session 36, 26 Oct 2026 (supersedes everything below)

**Status in one line: the finetuning experiment worked. A finetuned 1.7-billion-parameter model
extracts triples as accurately as the 120-billion-parameter teacher — but it stays silent on most
paragraphs, and that is now the problem to solve.**

Run on a laptop (RTX 4060, 8 GB). Student: **Qwen3-1.7B**. Teacher: **GPT-OSS-120B**'s stored
extractions. Trained on 13 papers, evaluated on **5 papers it never saw**, every paragraph judged
(not a sample).

### 1. Accuracy: the student caught up with the teacher

Each triple was graded blind — all four models' triples pooled, shuffled, names hidden until after
grading — on concept correctness (C1), concept specificity (C3) and **relation correctness (C4)**.

| model | C1 | C3 | **C4 relation correctness** | triples that were plainly wrong |
|---|---|---|---|---|
| Qwen3-1.7B, untuned | 59.8% | 52.3% | **29.4%** | **63.6%** |
| Qwen3-1.7B, finetuned (one teacher) | 90.7% | 81.5% | **83.3%** | 7.4% |
| **Qwen3-1.7B, finetuned (multi-model data)** | 90.2% | 87.8% | **85.4%** | 7.3% |
| GPT-OSS-120B (the teacher) | 96.4% | 87.2% | **86.4%** | 1.7% |

Relation typing — which relation holds, and in which direction — is what the ontology training fixes:
**29.4% → 85.4%**, level with a model 70× its size. The output format problem disappeared too:
valid JSON went **87.9% → 100%** and over-long replies **18 → 0**, which also makes the finetuned
model **12.5× faster to run**.

### 2. Multi-model training data: the same quality, but more of it

Teacher-only vs multi-model (a triple the teacher missed is added when at least 2 of 3 other models
extracted exactly the same one):
- **Quality: a tie.** C4 differs by +2.0 points with a confidence interval of [−12.7, +16.8] — at
  27 vs 41 judged triples, this experiment cannot separate them.
- **Volume: multi-model wins.** **47 kept triples vs 29 (+62%)**, paragraph coverage 7.5% vs 4.6%,
  and 73.4% vs 64.4% of its triples survive the rule checks.
- **Recommendation: use the multi-model data.**

### 3. ⛔ The open problem: it is too quiet

Both finetuned models answer "no triples here" for **90–95% of paragraphs**, producing 29 and 47
triples where the teacher produces 180 — paragraph coverage **4.6–7.5% against the teacher's 42%**.
Accurate triples are of little use if there are too few of them.

**The cause is in the training data: 402 of 619 training answers (65%) are empty**, because the
teacher itself declines on most paragraphs. The student learned the teacher's silence along with its
precision. **The fix is already built and not yet trained:** a dataset capped at 30% empty answers
(310 examples). That is the next run.

### 4. What is left

1. Repeat the blind grading for the **second ontology** (scinex); the table above is CEO only.
2. **Train the capped-empty variant** — the direct attack on the coverage problem.
3. **4-fold cross-validation** so every paper is scored by a model that never saw it (~5.5 h/fold).
4. **Part 3 (Q&A evaluation)** needs an independent model to write and answer the questions; the
   grading is already automatic. This is what the OpenRouter credit would be for.

### 5. Two engineering findings worth keeping

- **Windows was running everything 8× slow.** The power-saving opt-out was being applied to a
  launcher process, not to the real interpreter underneath it (a quirk of venvs made by the Python
  Install Manager). Training went from 48 s/step to ~9 s/step once fixed — and the earlier 14-hour
  extraction run had been throttled the same way. Diagnostic: the GPU reporting 0% use at under
  10 W with no throttle reason means it is being starved by the CPU side, not throttled.
- **A bigger generation batch made things worse, not better** — batch 6 was 2.85× slower than batch 4
  on identical paragraphs, because VRAM residency hit 7.1 of 8 GB and Windows silently pages VRAM to
  system memory instead of reporting an error.

---

*The Session 34 summary follows, kept for the record. Its numbers are Qwen3-4B on a single paper and
are NOT comparable with the 1.7B figures above.*

## ⭐ UPDATE — Session 34 (supersedes the status line below)

**Status in one line:** the finetuning fix works (no collapse), and the training data now uses
several models instead of one. The real experiment, teacher-only vs multi-model on a larger GPU
(an RTX 5090 VM, 32 GB), is next. **No reportable finetuning result yet.**

**1. Training data from several models, not one.**
Before, every training answer came from one teacher (GPT-OSS-120B), chosen because it is the most
accurate on relation direction/typing (C4 86.2% CEO / 90.0% scinex). Replies from the other models
were also on disk, but they agree with the teacher on only 3–21% of triples, so simply adding them
would give the student conflicting answers. Instead, a triple the teacher missed is added when
**at least 2 of the 3 other models (Gemma-4-31B, Qwen3-235B, Ministral-14B) extracted exactly the
same triple** from the same paragraph. The teacher stays in charge: an addition is skipped if the
teacher already linked the same two entities.
- Test-laptop data (3 training papers): **+66 triples on top of the teacher's 192 (+34%)**; 7
  paragraphs where the teacher found nothing now have a target.
- On the full 15 training papers the pool is larger (before skip rules: 143 new CEO, 124 new
  scinex triples).

**2. First finetuned model: the collapse is fixed.**
Qwen3-4B trained on the teacher-only data, checked on 44 paragraphs of one held-out paper (CEO):

| Model | Paragraphs with triples | Triples kept by the rule checks |
|---|---|---|
| Untuned Qwen3-4B | 32 | 38% |
| **Finetuned Qwen3-4B** | 8 | **86%** |
| Teacher (GPT-OSS-120B) | 21 | 88% |

It now matches the teacher's precision, but **extracts too little** (8 vs 21 paragraphs). That is
what the multi-model data is meant to fix. *Early signal from one paper, not a result.*

**3. Evaluation design settled.**
- **Triple correctness (C1/C3/C4): Claude Opus 5.5 as the judge, blind.** All models' triples are
  pooled, shuffled and stripped of model names; identical triples get one verdict. Every triple on
  the held-out papers is judged (not a 40-triple sample, which proved too noisy). Claude contributes
  no training data, so it is independent of teacher, voters and student. Single judge; a different
  judge from the earlier C1–C6 tables, so the two sets are not mixed.
- **Q&A grading no longer needs an LLM:** answers are scored by word overlap with the reference
  (standard token-F1). An LLM still writes the questions (each checked in code) and answers them.

**4. Engineering finding.** Windows throttled the hidden background training process onto
power-saving CPU cores: 40 s per step instead of 7–9 s. Fixed in the pipeline.

**5. Next.** On an RTX 5090 VM (32 GB): Qwen3-8B, teacher-only vs multi-model, 5 held-out papers → blind
judging → Q&A → 4-fold cross-validation with the better variant.

**Side finding (affects older results, not finetuning):** 375 of Qwen3-235B's 567 CEO replies are
exactly the text our script writes when an API call fails. If they were failures, its reported 16.4%
paragraph coverage and C6 score are wrong. Checkable for free in the OpenRouter activity log.

---

*Original Sessions 32–33 summary follows, unchanged.*

**Project:** SciNex — Knowledge Graph Construction for Open Science
**Scope of this summary:** Part 2 (finetune an LLM on the domain ontology + vocabulary) and
Part 3 (evaluate knowledge graphs by simulated Q&A against the real papers).
**Status in one line:** both pipelines are fully built and the local end-to-end test ran through
base-model extraction and one training run; the first finetuned model **collapsed** (it outputs no
triples), the cause was found and fixed, and the corrected retraining was stopped at step 72 of 620
before producing a model. **There is no usable finetuned-model result yet.**

---

## 1. Context

The project has three parts:

1. **Research paper → knowledge graph** (built earlier): PDF parser → paragraphs → an LLM extracts
   (subject, predicate, object) triples constrained to an ontology → rule-based "guards" reject
   malformed triples. Two ontologies are used: **CEO** (Core Experiment Ontology, 22 relations) and
   **scinex** (27 relations).
2. **Finetune an LLM to understand the ontology and vocabulary** ← this work.
3. **Evaluate the KG by simulated Q&A against the real papers** ← tooling built in this work.

Decisions that shaped the design:
- **Teacher = GPT-OSS-120B** (Apache-2.0). Its stored extraction replies on the 20-paper corpus are
  the finetuning targets. It was the best extractor on relation correctness in the earlier
  evaluation (C4 = 86.2% CEO, 90.0% scinex, n≈40 triples).
- **Claude is excluded** from the pipeline — not extractor, judge, teacher, or training data.
  Every new script refuses Claude model ids.
- **Training and evaluation papers must be disjoint** (split by paper, never by paragraph).
- The corpus stays at the user's 20 papers (no new papers were added).

---

## 2. What was built

| Component | File | What it does |
|---|---|---|
| Training-data builder | `build_ft_data.py` | Builds 3 data parts as chat-format JSONL with a paper-level train/eval split |
| QLoRA trainer | `ft_train_qlora.py` | 4-bit QLoRA finetuning; loss only on the answer; `--dry-run` checks lengths without a GPU |
| Extraction with the finetuned model | `ft_generate.py` | Runs base or finetuned model through the *same* extraction path as every other model |
| Judge-free comparison | `ft_compare.py` | Base vs finetuned vs teacher: valid JSON, triples proposed vs kept, coverage, agreement with teacher |
| Q&A evaluation (Part 3) | `qa_eval.py` | Generates verified questions from paper text → answers them from the KG only → blind grading → report with confidence intervals |
| Unattended runner | `run_ft_pipeline.py` | One command: base extraction → train → finetuned extraction → compare → Q&A; resumable |
| Shared plumbing | `ft_common.py` | Split, paragraph loading, guards, chat template shared by training and inference |

### 2.1 The three parts of the training data

| Part | Content | Why |
|---|---|---|
| **Ontology** | Each relation's definition and domain → range; "is this triple correctly directed?" (as extracted = yes, subject/object swapped = no); "which relation links this subject to this object?" | Targets the direction/typing errors found in evaluation (e.g. `splitFrom`, `achieves`) |
| **Vocabulary** | Entity → type cards (Model, Method, Dataset, Task, Tool, Resource, Language) from the human-annotated CS-NER gazetteer and training papers' entity lists | Domain terminology |
| **Extraction** | Paragraph → triples, from the teacher's stored replies, keeping only triples that pass the current guards | The actual task |

### 2.2 Safeguards built into the data (each one caught a real problem)
- **Paper-level split that never reshuffles.** Each paper gets a fixed fold (4 folds × 5 papers);
  adding papers later can never move a training paper into the test set. The rule is enforced three
  times: when building data, when training, and when generating.
- **Pairing check.** Each teacher reply must line up with the paragraph it was written for (paragraph
  count must match, ≥90% of the teacher's source sentences found in that paragraph). This rejected
  2 papers whose re-parsed paragraphs had shifted (see §5).
- **Bad replies are dropped, never "repaired"**: empty replies, failed API calls, truncated JSON,
  and duplicated replies from a past file-corruption incident.
- **Open decision respected:** the `configures` relation's direction is still undecided, so no
  training example teaches either direction.

### 2.3 Q&A evaluation design (Part 3)
- Questions are written from each real paragraph and **checked in code**: the answer must be copied
  verbatim from the paragraph, the question must not contain its answer, no boilerplate (authors,
  funding, references).
- The **same question set** is used for every KG, so comparisons are paired.
- Three conditions: **KG only** (triples, without source sentences), **oracle** (the paragraph —
  ceiling), **closed-book** (nothing — floor, measures what the answering model already knows).
- Grading is blind; identical answers get one grade whichever model produced them.
- Reports accuracy with a paper-level bootstrap 95% CI, plus two LLM-free checks: *is the answer a
  node of the graph at all?* and *did a correct answer actually come from the cited triples?*
- **Status: built, not yet run.**

---

## 3. Test run on a local 8 GB laptop

**Hardware:** NVIDIA RTX 5070 Laptop GPU (8 GB VRAM), 15.6 GB RAM, Windows 11.

**Model choice.** Qwen3-8B can *run* on 8 GB but cannot be *trained* there: its 4-bit weights plus
embeddings take ~6 GB before any training memory. We used **Qwen3-4B-Instruct-2507** (Apache-2.0,
same family), which loads in 2.62 GB.

**Test data.** Only the 7 open-access corpus PDFs were available on this machine.
- Evaluation (held-out) papers: **strabismus2026, osmotic2026**
- Training papers used: **gamlprop2025, parkingyolo2023, traveltime2022**

| Training data (fold 0) | Examples |
|---|---|
| Extraction pairs | 168 (83 CEO / 85 scinex; 87 with empty targets) |
| Ontology cards | 700 (141 definition, 167 correct-direction, 167 reversed-direction, 225 relation-choice) |
| Vocabulary cards | 1,612 |
| **Total** | **2,480** |

Extraction examples are 3,206 tokens at the median, 3,821 at most (the system prompt alone is ~3.1k).

### Timeline (1 Oct 2026, local time)
| Time | Event |
|---|---|
| 01:50 – 04:57 | Base-model extraction on the 2 held-out papers, both ontologies (186.7 min) |
| 04:57 – 06:21 | Training run #1: 620 steps, 2 epochs, 83.9 min, peak VRAM 4.60 GB |
| 06:21 – 06:30 | Finetuned extraction started → **collapse detected** → stopped |
| 06:35 – 06:46 | Training run #2 with the fix → aborted at step 72/620 (console window closed) |
| after | Stopped at the user's request |

---

## 4. Results

### 4.1 Baseline: untuned Qwen3-4B vs the teacher (2 held-out papers)

**CEO ontology**
| Model | Valid JSON | Triples proposed | Kept by guards | Kept rate | Paragraph coverage | Agreement with teacher (precision / recall) |
|---|---|---|---|---|---|---|
| Qwen3-4B (base) | 100.0% | 493 | 172 | **34.9%** | 66.2% | 8.7% / 16.5% |
| GPT-OSS-120B (teacher) | 97.3% | 107 | 91 | **85.0%** | 44.6% | — |

**scinex ontology**
| Model | Valid JSON | Triples proposed | Kept by guards | Kept rate | Paragraph coverage | Agreement with teacher (precision / recall) |
|---|---|---|---|---|---|---|
| Qwen3-4B (base) | 98.6% | 464 | 164 | **35.3%** | 63.5% | 1.8% / 5.3% |
| GPT-OSS-120B (teacher) | 100.0% | 70 | 57 | **81.4%** | 23.6% | — |

**Reading:** the untuned small model extracts ~4–7× more triples than the teacher, but **about
two-thirds are rejected by the guards** (vs ~15–19% for the teacher). It invents relations not in the
ontology (e.g. `definedAs`, `classifiedInto`), writes ~1,000 tokens per paragraph, and once ran to the
token limit mid-JSON. Agreement with the teacher is low. This is the gap finetuning is meant to close.
"Agreement with teacher" measures imitation, not correctness; no independent judge was run.

### 4.2 Finetuned model — result: training run #1 collapsed

- Training loss fell normally (3.31 → 0.14; epoch-1 mean 0.42, epoch-2 mean 0.14).
- But the finetuned model answered **`{"triples": []}` (no triples) for 40 of 40 held-out paragraphs**
  checked, including the abstract where the teacher found 7 triples.
- Probed directly, it also returned no triples for **5 of 5 *training* paragraphs** whose target had
  triples. So it was not overfitting: the model had learned to always abstain.
- **Cause:** a loss-weighting bug. The loss was averaged per example, so a 7-token "no triples"
  target counted as much as a 300-token extraction. The abstain decision carried ~40× more weight
  than the "start a triple" decision, and with ~half the targets empty (87 of 168) the model learned
  to always abstain.
- **Fix (implemented, verified to run):** standard token-level loss normalisation — every target
  token weighs the same.
- **Retraining with the fix** reached step 72 of 620 (losses ~0.1–0.5, 4.30 GB peak) before the run
  was interrupted, and was then stopped at the user's request. **No corrected finetuned model exists
  yet.** Retraining takes ~85 min on this laptop, finetuned extraction ~1 h.

### 4.3 Engineering findings (reusable)

| Finding | Measurement | Fix |
|---|---|---|
| On this Windows GPU/PyTorch build, fast attention kernels are unavailable for Qwen3's grouped-query attention, so PyTorch silently falls back to a slow kernel | **7.27 GB** for one attention layer at 3.8k tokens vs **0.34 GB** with cuDNN; training stalled (100% GPU at ~40 W, no step in 10 min) | Force cuDNN attention for training → one step on the longest example: 6–8 s, 5.10 GB |
| cuDNN attention is wrong for text generation (re-plans every token) | 5.7 vs 18 tokens/s | Use it for training only |
| peft's standard QLoRA setup upcasts embeddings to fp32 | +0.8 GB on Qwen3-4B | Lean bf16 setup by default |
| Computing vocabulary logits over the whole ~3.8k-token prompt wastes memory | 3–5 GB | Compute logits only over the answer span |
| Per-example loss averaging + many empty targets | Model always abstains | Token-level loss normalisation |

Throughput on the laptop: generation ~18 tokens/s (batch of 4, 5.3–6.4 GB), training ~8–10 s per
optimizer step (8 examples).

---

## 5. Data-quality findings along the way

- **Existing corpus numbers are reproducible.** Re-running today's guards on GPT-OSS-120B's stored
  replies reproduces the saved CEO triple count exactly (547 = 547).
- **One existing number is slightly inflated.** GPT-OSS-120B's scinex count (323) includes triples
  double-counted from a past file-corruption incident on one paper (strabismus2026: 7 paragraphs
  answered twice). The true count is **312–323**; the caveat was added to the results tables.
- **The pairing check worked.** Two papers were rejected because re-parsing no longer matches the
  paragraphs the teacher saw (oxidecrack2025: one paragraph boundary shifted, alignment 0.88/0.71;
  roadwaylight2018: 30 vs 31 paragraphs). Without the check, their targets would have been
  attached to the wrong paragraphs silently.
- **A measurement trap.** The API runner writes `{"triples": []}` when a call *fails*; a local
  model writes the same string as a genuine "nothing here" answer (21 of the base model's 148
  replies). The comparison now treats it as a failure only for API runs.

---

## 6. What is not done / caveats

- **No working finetuned model yet.** The corrected retraining must be run to completion (~85 min).
- **This was a pipeline test, not an experiment:** 3 training papers, 2 test papers, one fold.
  A real result needs the full 20-paper corpus (the other 13 PDFs are on the lab machine), all 4
  folds, and an independent judge.
- **Q&A evaluation has not been run.**
- **No independent judge was used.** C1–C6 correctness of the finetuned model still needs a
  non-Claude judge (suggested: Gemma-4-31B — neither the teacher nor the student's model family).

## 7. Next steps

1. Re-run training with the token-level loss fix (~85 min) and confirm the model no longer abstains
   on every paragraph.
2. Finetuned extraction on the 2 held-out papers (~1 h) → base vs finetuned vs teacher comparison.
3. Run the Q&A evaluation (question generation → KG answering → grading).
4. Scale up on the 20 GB lab GPU: all 20 papers, 4-fold cross-validation, Qwen3-8B, then
   Ministral-14B.
5. Decisions needed from the lead researcher: base model, judge model and API budget, the
   `configures` direction, whether to use the (Claude-made) C1/C4 verdicts as a data filter.

---

## 8. Suggested slide outline

1. **Title** — Finetuning an LLM for ontology-constrained scientific KG extraction: progress report
2. **Project in three parts** — where this work sits (Parts 2 and 3)
3. **Design principles** — teacher–student, Claude excluded, paper-level split
4. **Training data** — three parts, with example counts (§2.1, §3)
5. **Safeguards** — split, pairing check, drop-don't-repair (§2.2)
6. **Q&A evaluation design** — conditions and checks (§2.3)
7. **Local test setup** — 8 GB GPU, why Qwen3-4B, timeline (§3)
8. **Baseline vs teacher** — the two tables (§4.1); key message: 35% vs 85% kept
9. **Finetuning result: collapse** — symptom, diagnosis, fix (§4.2); an honest negative result
10. **Engineering lessons** — memory/speed table (§4.3)
11. **Data-quality findings** (§5)
12. **Next steps and open decisions** (§6–7)
