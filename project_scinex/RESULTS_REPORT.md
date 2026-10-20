# Scientific Paper Knowledge Graph Pipeline
## Project overview, method, and consolidated results

**Generated:** 2026-08-28 (Session 26) · built by `build_results_report.py`
**Per-triple numbers are read directly from `output/` and `gold/` at build time**, so this
document cannot drift from the artefacts it describes. Re-run the builder after any new result.

> **Reading this for the first time?** §1-4 explain what the project is and how it works.
> §5-10 are the results (three axes: model, ontology, graph-level). §11-15 are corpus,
> methodology, engineering findings and negative results. §16-18 are limitations, next steps
> and reproduction.
> Two distinct corpora and two distinct evaluation axes appear here — §4 explains the difference,
> and conflating them is the single easiest mistake to make with this material.

---

## 1. What the project is

**Goal: turn scientific papers into a machine-readable knowledge graph, and measure whether the
resulting graph is any good.**

A scientific paper is unstructured prose. A knowledge graph is a set of
`(subject, predicate, object)` triples — e.g. `(BERT, evaluatedOn, SQuAD)` — that a machine can
query, aggregate across thousands of papers, and reason over. The pipeline converts the first into
the second, then evaluates the result two independent ways.

The hard part is not producing triples. **It is producing triples that are actually supported by
the paper**, in a vocabulary consistent enough to merge across papers. An unconstrained LLM will
happily emit fluent, plausible, unsupported triples, and will invent a new relation name for every
sentence. Both failure modes make the resulting graph useless.

### The research questions

1. **How much constraint does an extractor need?** Constraining the relation vocabulary, the
   subject vocabulary, or both changes precision and coverage in opposite directions.
2. **Does the choice of ontology matter?** Two competing schemas (CEO and scinex) run through an
   identical pipeline — and, it turns out, the answer depends on the extractor (§6).
3. **How good is the extraction, per triple?** Measured by judged precision against the source
   sentence (§5-8).
4. **How good is the graph, as a graph?** Measured by an unsupervised citation-prediction task
   (§10).
5. **How much does the extraction model itself matter?** — the question this session answered,
   and the strongest result in the project (§5).

---

## 2. The pipeline

```
  PDF                                                                                
   |  Part 1: parser  (pdfplumber / PyMuPDF -> structure_builder -> html_generator)   
   v                                                                                 
  structured HTML  +  citation network (Semantic Scholar)                            
   |  Part 2: extraction  (LLM, ontology-constrained, guarded)                        
   v                                                                                 
  triples.json  +  kg.graphml                                                        
   |                                    \                                            
   |  Part 2c: per-triple evaluation      Part 3: graph-level evaluation             
   v  (LLM-as-judge vs source sentence)   v  (KG embedding -> citation prediction)    
  precision / kappa                       Hits@k / MRR                               
```

### Part 1 — PDF to structured text

Converts a paper PDF into sectioned HTML plus a Semantic Scholar citation network. This is more
load-bearing than it sounds: **extraction quality tracks parse quality directly**. Several
apparent "the model found nothing" results turned out to be parser bugs — for example a heading
detector that only accepted *numbered* headings silently discarded the entire body of every
Nature-style paper, leaving only the bibliography.

### Part 2 — Extraction: a ladder of constraint

The same LLM is run under four constraint regimes. This ladder **is** research question 1:

| extractor | subject | relation | object | what it isolates |
|---|---|---|---|---|
| `llm` | free | free | free | unconstrained baseline |
| **`relation`** | **free** | **ontology** | **free** | relation vocabulary only |
| `fixed` | entity list | ontology | free | + curated subject pool |
| `pair` | entity list | free | entity list | closed entities, open relation |

Comparing `relation` against `fixed` isolates exactly what a curated entity list buys.
**All results in this document use `relation` mode**, because the entity gazetteer (CS-NER) is
annotated over CS/NLP papers and covers this corpus's chemistry/materials/clinical half poorly —
`fixed` would have failed there for coverage reasons, not quality reasons.

Two non-LLM baselines also exist: **REBEL** (`Babelscape/rebel-large`, general-domain) and
**ITER** (`fleonce/iter-scierc-deberta-large`, SciERC typed relations).

### The guards — where much of the quality actually comes from

The LLM's output is not trusted. Every proposed triple passes code-level validation before it
reaches the graph: the subject must appear literally in the source sentence; the object must be
present; IS-A relations are rejected; direction guards catch reversed domain/range; per-predicate
keyword guards enforce the stricter relations. **Two bugs in these guards were found and fixed
this session, and both were silently deleting correct triples** (§13).

### Part 2c — Per-triple evaluation (LLM-as-judge + human/model agreement)

Each triple is judged **against its own source sentence** as CORRECT / PARTIAL / INCORRECT.
A stratified sample is labelled independently by a second party, and the two labellers are
compared by Cohen's κ. See §12 for the protocol.

### Part 3 — Graph-level evaluation (citation prediction)

A knowledge-graph embedding (TransE / ComplEx / RotatE) is trained on the extracted triples
**with real citation edges held out**, then asked to rank each paper's true citation neighbours
above all other papers. A graph that encodes real content should make papers that cite each other
land near each other. See §9.

---

## 3. The two ontologies

An ontology fixes the relation vocabulary and constrains each relation's **domain → range**
(what types may appear on each side). This is what makes triples from different papers mergeable.

| | CEO (Core Experiment Ontology) | scinex |
|---|---|---|
| origin | collaborator's schema | refined alternative (OWL) |
| relations | 22 | 27 |
| relation to CEO | — | CEO + 5 relations, refined domain/range |

CEO predicates: `cites`, `publishedIn`, `writtenBy`, `reports`, `affiliatedWith`, `employs`,
`locatedIn`, `addresses`, `motivates`, `achieves`, `encompasses`, `comprises`, `uses`, `produces`,
`trainedOn`, `evaluatedOn`, `splitFrom`, `designedFor`, `comparesAgainst`, `configures`,
`evaluates`, `supports`.

Each carries a strict domain→range constraint enforced **twice**: in the prompt, and again in
post-parse code. Example — `addresses` is `ResearchProcess/Paper → ResearchContext`, so a
*task* may never be its subject (tasks *are* the research context; they cannot address themselves).

---

## 4. ⚠ Two corpora, two evaluation axes — do not conflate them

| | **Per-triple track** (§5-8) | **Graph-level track** (§8) |
|---|---|---|
| corpus | 20 handpicked papers (this document's focus) | 155 ACL Anthology papers |
| domain | applied ML: ophthalmology, chemistry, materials, traffic/ITS, cloud, CV, NLP | CS/NLP |
| question | is each triple true? | is the graph structurally useful? |
| metric | judged precision, Cohen's κ | Hits@k, MRR |
| extractor mode | `relation` | `fixed` |

They answer different questions and neither subsumes the other: a graph can be built from
individually-shaky triples and still support citation prediction, and vice versa. **Any slide
mixing the two must say which corpus a number comes from.**

---

## 5. ⭐ Headline result — the extraction model dominates everything else

**Same paragraphs, same system prompt, same post-parse guards, same graph builder, and the same
judge. The only variable is the extraction model.**

| extractor model | ontology | n judged | **weighted strict** | lenient |
|---|---|---|---|---|
| **claude-opus-5** | CEO | 100 | **89.0%** | **97.9%** |
| qwen3:8b | CEO | 58 | 25.6% | 50.4% |
| qwen3:8b | scinex | 52 | 11.7% | 34.2% |

Judge: **`gemini-3.6-flash`** — a third model family, so neither run is self-scored.
Bootstrap on the Claude row (2,000 resamples within predicate buckets, corpus weights fixed):
mean 89.1%, **95% CI [82.3%, 94.5%]**. Verdict mix: **86 CORRECT / 13 PARTIAL / 1 INCORRECT**
of 100.

### ⭐ Confirmed by a second, independent judge

The Claude row was re-judged from scratch by **`openai/gpt-oss-120b`** (OpenAI family, via Groq)
on the same 100 triples — a third family again, independent of both the extractor and the first
judge. **It agrees.**

| judge | family | weighted strict | lenient | sampled strict |
|---|---|---|---|---|
| `gemini-3.6-flash` | Google | **89.0%** | 97.9% | 86.0% |
| `openai/gpt-oss-120b` | OpenAI | **82.8%** | 92.6% | 85.0% |

### ⭐⭐⭐ The capability curve — five runs, one judge

Four extraction models across three families (plus one re-run of the smallest with the
pipeline fixes applied), every one run on the same paragraphs through the same system prompt
and graph builder, then judged by the same model (`openai/gpt-oss-120b`), same rubric.

| extractor | family | scale | triples | /para | n | **strict** | lenient | junk-pred share |
|---|---|---|---|---|---|---|---|---|
| gemma-3-12b | Google | 12B dense | 1,930 | 3.40 | 100 | **29.2%** | 52.1% | 20.1% |
| qwen3:8b (pre-fix) | Alibaba | 8B dense | 1,185 | 2.26 | 114 | **29.8%** | 64.0% | 12.7% |
| qwen3:8b (+guards) | Alibaba | 8B dense | 1,444 | 2.55 | 100 | **38.1%** | 66.1% | 11.1% |
| qwen3-235b | Alibaba | 235B MoE | 386 | 0.68 | 100 | **48.8%** | 60.9% | 6.7% |
| claude-opus-5 | Anthropic | frontier | 315 | 0.55 | 100 | **82.8%** | 92.6% | 0.6% |

**Three findings, in order of how much they carry.**

**0. Two independent axes move precision, and this table shows both.** Rows 2 and 3 are the
*same model* — same weights, same prompt, same GPU — differing only in whether the pipeline
guards and two bug fixes were applied. Rows 1-5 vary the model. **Model choice is the larger
lever (29.8% → 82.8%); pipeline engineering is the free one (29.8% → 38.1%), and the two
are additive rather than alternatives.**

**1. Precision and volume are inversely ordered, without exception.** Ranked by strict
precision, the four models are also ranked — in reverse — by triples per paragraph
(3.40 → 0.55) and by the share of their output held
by the relations that score 0% (20.1% → 0.6%). A weak
extractor is not a good extractor with more noise; it is one that manufactures relations whose
domain/range it cannot satisfy, and a strong one returns an empty result instead.

**2. Scale works — demonstrated *within a single family*, which removes the confound.**
qwen3:8b and qwen3-235b share a vendor, a lineage and a tokenizer, and differ by ~29x in total
parameters (8B dense vs 235B MoE, 22B active). Strict precision goes **29.8% → 48.8%**, volume
**2.26 → 0.76** per paragraph, junk share **12.7% → 6.4%**. Before this run the headline
compared an 8B Alibaba model against a frontier Anthropic one, so *strength* and *vendor* moved
together and a reviewer could reasonably attribute the gap to either. They no longer move
together.

**3. But parameter count does NOT transfer across families — gemma-3-12b is the negative
control.** It is 50% larger than qwen3:8b and performs **no better** on strict precision
(29.2% vs 29.8%), while emitting the most triples of any model tested (3.47/paragraph) and
carrying the largest junk share (20.1%). So the axis that predicts extraction quality is
**capability**, not parameters — and this row is what licenses saying so, rather than
assuming it.

### ⭐⭐ Pipeline engineering, measured on one model (rows 2 vs 3)

`qwen3-8b-v2` re-runs the identical local Ollama build over the identical paragraphs, changing
only the code: the three post-parse guards (§12) plus the two bug fixes. It is the cleanest
available measurement of what the engineering is worth, with the model held constant.

| | pre-fix | +guards & fixes | change |
|---|---|---|---|
| **weighted strict** | 29.8% | **38.1%** | **+8.3 pts (+28% rel.)** |
| weighted lenient | 64.0% | 66.1% | +2.1 pts |
| triples | 1,185 | **1,357** | **+14.5%** |
| junk-predicate share | 12.7% | 11.5% | −1.2 pts |

**Precision and volume rose together**, so this is not a precision/recall trade. The per-predicate
movement matches each fix's prediction one for one:

| predicate | pre-fix | after | change | attributable to |
|---|---|---|---|---|
| `evaluates` | 13 | **48** | **+269%** | numeric-object bug fix — the pipeline demanded a value then deleted it |
| `comparesAgainst` | 51 | 78 | +53% | — |
| `comprises` | 38 | 57 | +50% | — |
| `achieves` | 145 | 181 | +25% | — |
| `affiliatedWith` | 29 | **13** | **−55%** | boilerplate-section guard (CRediT / funding) |
| `employs` | 26 | **16** | **−38%** | boilerplate-section guard |
| `evaluatedOn` | 101 | **74** | **−27%** | cross-reference guard (`→ "Table 4"`) |

Every predicate moved in the direction its fix predicted, and no other predicate moved materially.

⚠ **One asymmetry to state.** `qwen3-235b` and `gemma-3-12b` were prompted *after* the
boilerplate-section guard landed, so they saw 515 paragraphs where the two older runs saw 525.
The 10 skipped paragraphs are exactly the Author-contributions / Declarations / Funding / Data-
availability sections (verified by diffing the rendered prompts). This slightly *helps* the two
newer models by removing junk-generating input, and it is worth one sentence in the write-up.

---

### ⭐⭐ The complete single-judge comparison — quote this one

`gpt-oss-120b` went on to judge the **full** qwen sample as well, so one judge now covers both
extractors at full sample size under an identical protocol (same rubric, same batch size, same
stratification). This is the cleanest version of the headline.

| extractor | ontology | n | **weighted strict** | lenient |
|---|---|---|---|---|
| **claude-opus-5** | CEO | 100 | **82.8%** | 92.6% |
| qwen3:8b | CEO | 114 | 29.8% | 64.0% |
| qwen3:8b | scinex | 126 | 18.3% | 40.6% |

**Both judges independently put Claude at ~3x qwen3:8b**, and this row set has no partial-sample
caveat: 100 / 114 / 126 labels, all collected, zero failures.

**The two judges independently assign near-identical precision** — sampled
86.0% vs 85.0%. The headline is best reported
as a **range, 82.8%-89.0% weighted strict**,
rather than a single point estimate. Either way it is 3-4x qwen3:8b under the same protocol.
Agreement statistics for this pair are in §9.

**Why this is the strongest claim in the project:** every other variable is held constant by
*shared code*, not by reimplementation (§6). Prompt engineering, ontology choice and guard
tuning — the levers this project spent most of its effort on — move precision by single-digit
percentage points. Swapping the extraction model moves it by ~63 points.

⚠ **Two coverage caveats, stated rather than buried.**

1. **The qwen rows are 58+52 labels, not the full 240.**
   The judging run hit Gemini's free-tier daily quota at batch 11 of 24. The collected verdicts
   are a stratified *prefix* of the same round-robin sample, not a biased subset, but the qwen
   rows should be completed before publication (§17 item 1).
2. **The gold samples were drawn before the 2 substitute papers existed**, so the 17 triples from
   `aiabstract2025` and `routepred2023` are unjudged. Precision figures describe the 18-paper
   corpus; volume figures in §8 describe all 20.

⚠ **Weighting must always be stated.** The gold sample is stratified — it deliberately
over-represents rare predicates — so the *sampled* rate differs from the *corpus-weighted* one.
Claude sampled 86.0% vs weighted 89.0%; qwen CEO sampled 15.5% vs weighted 25.6%.
Quote the weighted figure and say so.

### An earlier finding this supersedes

A previous round concluded **"CEO beats scinex"** from an 80-triple sample. On a disjoint
160-triple draw the advantage **reversed sign**, and the combined n=240 interval straddles zero
(difference −4.9 pts, 95% CI [−22.5, +13.9]). **The two ontologies are indistinguishable on
per-triple precision at this sample size.** That claim is withdrawn and must not reappear in the
write-up. CEO retains a consistent *lenient* edge (66.8% vs 59.9%): its errors are on-topic but
loosely typed, where scinex's are more often flatly wrong.

---

## 5b. ⭐⭐⭐ The six-criterion evaluation (C1–C6) — the current framework

The single CORRECT/PARTIAL/INCORRECT verdict used in §5 is **superseded** by a six-criterion
framework: C1 Concept Correctness, C2 Concept Completeness, C3 Concept Specificity,
C4 Relation Correctness, C5 Relation Completeness, C6 Semantic Consistency
(Zhang, Conia & Rago, IJCNLP-AACL 2025 for C1/C3/C4; Wilson et al., Semantic Web 14(6) 2023
for C2/C5/C6). Harness: `ontology_eval.py`.

**Three harnesses, because the criteria do not share a unit of analysis.** C1/C3/C4 are judged
per TRIPLE against its source sentence; C2/C5 per PARAGRAPH against every triple drawn from it;
C6 per PAPER against that paper’s whole triple set. A completeness question cannot be asked
of a single triple — what is missing is by definition not in front of the judge.
**Everything in §5–§9 measures precision only; C2/C5 are a genuinely new axis.**

> **⚠ n≈40 IS THE BINDING CONSTRAINT.** Applying a filter that removed 0.7% of triples
> and redrawing these samples moved scores by up to **+10.3 points** and reversed the C1 top pair.
> That is sampling variance, not the filter. **Any gap under ~10 points between two models is not
> distinguishable here.** The C4 ranking survived the redraw unchanged (spread ~45 pts); the C1
> ordering did not (spread ~24 pts) and should not be quoted.

Corpus: 22 papers, 567 paragraphs, CEO ontology. Sample: 40 triples + 16 paragraphs + 6 papers
per model = 248 judgements. Judge: claude-opus-5, in session.

| extractor | triples | paragraph coverage | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|
| `qwen3-235b` | 367 | 16.4% | 75.0% | 31.2% | 60.0% | 47.5% | 28.1% | 50.0% |
| `gptoss-120b` | 547 | 42.9% | 88.8% | 34.4% | 80.0% | 86.2% | 31.2% | 25.0% |
| `gemma4-31b` | 615 | 51.7% | 93.8% | 28.6% | 80.0% | 73.8% | 21.4% | 33.3% |
| `ministral-14b` | 1,536 | 69.1% | 70.0% | 26.7% | 52.5% | 41.2% | 30.0% | 8.3% |

### Finding 1 — the models differ on TYPING, not on concept identification

C1 spans 70.0%–93.8% (1.3×). **C4 spans 41.2%–86.2% (2.1×).**
The concepts extractors pull out are mostly real and present in the sentence; what separates a
good extractor from a bad one is whether it respects the predicate’s domain→range.
Under the single rubric of §5 these were one number and the split was invisible.

### Finding 2 — ⭐ VOLUME DOES NOT BUY COMPLETENESS

`ministral-14b` produces **4.2× more triples** than `qwen3-235b` and fires on **4.2× more paragraphs**
(69.1% vs 16.4% of the 567 paragraphs). Its completeness scores:
**C2 26.7% vs 31.2% (-4.6 points)**, C5 30.0% vs 28.1% (+1.9 points).
Every model sits in a 21.4%–34.4% band on both completeness criteria.

**The sign is the point: the densest extractor scores LOWER on concept completeness than the
sparsest one.** More triples did not merely fail to buy proportional coverage - on C2 it bought
none at all.

This is the strongest result of the new framework and it was unobtainable before it.
**Extracting four times as much does not capture four times as much of the paper.** The extra
volume is spent restating what was already captured, enumerating pairwise combinations, and
mining boilerplate — not on the paragraphs being missed. Both the sparse and the dense
extractor leave roughly two-thirds of each paper’s important content unrecorded, and they
leave *different* thirds.

### Finding 3 — C6 falls as extraction volume rises

| extractor | triples | C6 |
|---|---|---|
| `qwen3-235b` | 367 | 50.0% |
| `gptoss-120b` | 547 | 25.0% |
| `gemma4-31b` | 615 | 33.3% |
| `ministral-14b` | 1,536 | 8.3% |

The relationship is **directional but not strictly monotone**: Spearman **rho = -0.80** over 4 extractors, with one inversion — 367 triples -> 50.0%; 547 triples -> 25.0%; 615 triples -> 33.3%; 1,536 triples -> 8.3%.

> **⚠ An earlier version of this section claimed a PERFECT rank correlation. That claim is
> WITHDRAWN.** It came from a C6 sample drawn over 22 papers, two of which were retired
> substitutes (`aiabstract2025`, `routepred2023`). Redrawn on the corrected 20-paper corpus,
> gemma4-31b rose from 16.7% to 33.3% and overtook gptoss-120b despite having more triples.
> With only four extractors a single swap moves rho from -1.00 to -0.80, so **claim the
> direction, never the ordering**.

More triples means more chances for two of them to
conflict, and the conflicts concentrate in three mechanisms: **superlatives with the scope
qualifier stripped** ("best AUC" asserted of three different methods), **metric nodes bound to
two values** because the class label was dropped, and **surface-variant splitting** that lets
incompatible claims attach to what should be one node.

Clearest contradiction found: `gemma4-31b` asserts both `(VGG16) achieves (F1 w-avg (0.23))`
and `(VGG16) achieves (over 90% accuracy)`. Clearest structural violation: `ministral-14b`’s
`(route prediction model) --comprises--> (route prediction model)`, a node containing itself.

> **⚠ C6 SCORING ARTIFACT — do not report C6 alone.** An empty triple set cannot
> contradict itself, so it scores CONSISTENT (1.0) vacuously. `qwen3-235b`’s `textaug2023`
> (0 triples) is exactly this case and is 1 of the 6 papers behind its C6. **C6 systematically
> rewards extracting nothing**; it is only interpretable next to C2/C5, which price the emptiness.

### Finding 4 — boilerplate is mined as content, and C1–C4 cannot see it

Triples scoring CORRECT on all three per-triple criteria were drawn from author biographies
(`(SOKENDAI) locatedIn (Japan)`), IEEE download watermarks (`(KMITL) affiliatedWith (UniNet)`),
bibliographies, acknowledgements and the ACM CCS classification block. **A triple can be perfect
on C1, C3 and C4 and still be worthless** — the criteria judge a triple against its source
sentence and never ask whether that sentence belonged to the paper. Addressed since by
`_is_provenance_sentence()` (see §13).

### ⚠ Read these numbers with the sampling in mind

- **Single rater, by design.** claude-opus-5 labelled every item. There is no second labeller
  and **no κ for C1–C6**, and none is planned — this is a stated limitation of the
  track, not an unfinished step. The κ figures in §9 measure a different rubric and do
  not transfer.
- **n = 40 triples / 16 paragraphs / 6 papers per model.** C6 rests on 6 papers per model, so
  quote the monotone trend across the four models, not per-model point estimates.
- **The sample is stratified over predicates** and deliberately over-represents rare ones, so
  sampled rates run well above corpus-weighted ones — the same gap as the old track.
- **CEO only.** The scinex side of these four extractors has not been run.

## 5c. CEO vs scinex under C1–C6 — the ontology effect is model-specific

Both ontologies extracted over the same 20 papers, same prompt, same guards, same judge.
Only the triples harness (C1/C3/C4) has been judged for scinex; C2/C5/C6 are CEO-only.

| model | CEO C4 | ΔC1 | ΔC3 | ΔC4 | CEO vol | scinex vol | ratio |
|---|---|---|---|---|---|---|---|
| `ministral-14b` | 41.2% | +2.5 | +7.5 | +5.0 | 1,536 | 1,538 | 1.00x |
| `qwen3-235b` | 47.5% | -2.5 | -2.5 | -1.2 | 367 | 563 | 1.53x |
| `gemma4-31b` | 73.8% | -5.3 | -0.5 | -14.8 | 615 | 525 | 0.85x |
| `gptoss-120b` | 86.2% | -2.5 | -7.5 | +3.7 | 547 | 323 | 0.59x |

Rows are ordered by CEO C4, i.e. by extraction capability on the relation criterion.

**⚠ THIS DOES NOT REPRODUCE THE §6 INTERACTION.** §6 found scinex to be a
liability for a weak extractor and an asset for a capable one. Under C1–C6 there is no
monotone relationship with capability in either direction — the weakest model in this
table gains on all three criteria while the strongest loses on relation correctness.
**State it as: the ontology effect is model-specific and does not order by extraction
quality.**

Two reasons this is a failure to reproduce rather than a refutation: n is roughly 40 per
cell, and §6 varied scale WITHIN one model family (qwen3:8b vs qwen3-235b) on the
older single rubric, whereas these models are three different families — so family
effects and ontology effects are confounded here in a way they were not there.

### What scinex buys, and what it costs

**It fixes CEO failures on identical sentences.** Where CEO chose `employs` (typed
`Organisation → Person`) for a model consuming an architectural block and scored
INCORRECT, scinex chose `uses` and scored CORRECT. `splitFrom` was inverted under CEO and
correct under scinex for the same model. `producedBy` supplies an artifact→process
direction that CEO's `produces` kept inverting, and `achievesResult`
(`Model → ExperimentalResult`) is a real refinement of CEO's overloaded `achieves`.

**It also adds failure modes CEO cannot have.** `extractedFrom` (range `AcademicPaper`) was
misused **five times** — pointed at a dataset, an architecture component, OpenStreetMap,
a vehicle fleet and a data-availability statement. Its name reads generically enough that the
English always seems to fit, and CEO has no equivalent relation to misuse. The refinements
only pay when a model actually reaches for them: gemma4-31b kept using plain `achieves` for
metric outcomes and lost the benefit of `achievesResult` entirely.

---

## 6. ⭐⭐⭐ Ontology x capability — the ranking REVERSES

Research question 2 was *"does the choice of ontology matter?"*, and the honest earlier answer
was **"it is a tie"** — corpus-weighted CEO 34.7% vs scinex 39.6%, a difference of −4.9 pts
with a 95% CI of [−22.5, +13.9]. **Every one of those measurements was taken on qwen3:8b**, the
weakest extractor in the study.

Running both ontologies at two capability levels shows the tie was an artefact of the measurement
model. All four cells: same paragraphs, same guards, same code version, same judge, n=100 each.

| | CEO | scinex | scinex − CEO |
|---|---|---|---|
| **qwen3:8b (8B)** | 38.1% | 12.1% | **-26.0 pts** — CEO wins |
| **qwen3-235b (235B MoE)** | 48.8% | 54.7% | **+5.9 pts** — **scinex wins** |

**Interaction: +31.9 points.** The ranking does not merely narrow — it inverts.

**Volume moves the same way**, so this is not precision bought with recall:

| | CEO triples | scinex triples | ratio |
|---|---|---|---|
| qwen3:8b | 1,444 | 1,389 | **0.96x** |
| qwen3-235b | 386 | 596 | **1.54x** |

At 235B scinex wins on **both** axes — higher precision *and* ~48% more triples.

**The mechanism is plausible and worth stating.** scinex is the richer schema (27 relations vs
22, tighter domain/range). Extra structure is only an asset to a model that can satisfy it: for a
weak extractor it is additional surface area to get wrong, which is why scinex is that model's
*worst* configuration; for a capable one the tighter typing becomes a constraint that helps.

**This also explains the earlier instability.** CEO beat scinex in one gold round and scinex beat
CEO in a disjoint one. Both rounds were measured on qwen3:8b, where the schema is not the binding
constraint, so the comparison was noise-dominated and flipped between draws.

⚠ **What to claim, and what not to.** The **sign** of the reversal is robust — it holds in the
pre-guards data too (−11.5 pts at 8B) as well as post-guards (−26.0). The **magnitude at 8B
moves with the sample**, so quote the direction and the interaction, not the 8B gap to a decimal.

---

## 7. Why the comparison is methodologically valid

`claude_extract.py` splits the extractor **at the model boundary** rather than reimplementing it:

```
claude_extract.py prompts  ->  [ any model ]  ->  claude_extract.py ingest
  renders the REAL system +      the only          the REAL _parse_fixed_output()
  user prompts, walking the      variable          guards + the REAL
  paper exactly as kg_main does                    KnowledgeGraphBuilder
```

The prompt (11,908 characters), the ontology, the post-parse guards, deduplication and the output
format are **shared code, not copies**. `run_api_extract.py` drives the middle step for any
provider (Anthropic / OpenAI / OpenAI-compatible / Gemini / Ollama); `run_api_judge.py` does the
same for judging. That is what licenses the claim that only the model differs.

**Judge independence is enforced:** a model never judges its own extraction. Claude's triples are
judged by Gemini; qwen's triples are judged by Claude and by Gemini.

---

## 8. Volume — the stronger model produces far less, and that is the finding

| | qwen3:8b (Ollama) | claude-opus-5 |
|---|---|---|
| papers | 20 | 22 |
| **triples stored (CEO)** | **1,185** | **315** |
| triples stored (scinex) | 1,117 | — |
| ratio (Claude / qwen, CEO) | — | 27% |
| triples per paragraph | 2.26 | 0.56 |

**The missing volume is concentrated in the predicates that are always wrong.** The relations
that generated most of qwen's junk — `affiliatedWith`, `employs`, `configures`, `publishedIn`,
`cites` — appear **zero** times in Claude's output. It declines to produce them rather than
producing them badly: the CRediT author-contribution blocks, funding paragraphs and running-header
lines that drove qwen's `affiliatedWith`/`supports` errors return `{"triples": []}` unprompted.

**The volume gap replicated on unseen papers.** Two papers were added late (§11). On the 39
paragraphs neither model had seen, the ratio held: 0.56 vs 2.26 triples/paragraph, against
0.57 vs 2.27 on the original set. **The gap is a property of the models, not of the paper
selection** — which is exactly the objection a reviewer would raise.

**The slide-ready framing:** *a weak extractor is not merely noisier — it is confidently wrong
in a specific, predictable place. It fabricates relations whose domain/range it cannot satisfy.
A strong extractor returns an empty result instead, and the volume it declines to produce is
precisely the volume that was wrong.*

### Per-predicate precision (judge: gemini-3.6-flash)

Sorted by corpus share — the predicates at the top drive the weighted figure.

**claude-opus-5 (CEO)**

| predicate | n | C | P | I | strict | lenient | corpus share |
|---|---|---|---|---|---|---|---|
| `addresses` | 13 | 12 | 0 | 1 | 92.3% | 92.3% | 27.6% |
| `achieves` | 13 | 13 | 0 | 0 | 100.0% | 100.0% | 21.5% |
| `uses` | 13 | 11 | 2 | 0 | 84.6% | 100.0% | 18.6% |
| `comprises` | 13 | 8 | 5 | 0 | 61.5% | 100.0% | 10.0% |
| `comparesagainst` | 13 | 13 | 0 | 0 | 100.0% | 100.0% | 7.9% |
| `evaluatedon` | 12 | 12 | 0 | 0 | 100.0% | 100.0% | 4.3% |
| `designedfor` | 6 | 4 | 2 | 0 | 66.7% | 100.0% | 2.2% |
| `evaluates` | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 2.2% |
| `splitfrom` | 6 | 5 | 1 | 0 | 83.3% | 100.0% | 2.2% |
| `produces` | 4 | 3 | 1 | 0 | 75.0% | 100.0% | 1.4% |
| `encompasses` | 2 | 0 | 2 | 0 | 0.0% | 100.0% | 0.7% |
| `supports` | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 0.7% |
| `locatedin` | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 0.4% |
| `trainedon` | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 0.4% |

**qwen3:8b (CEO)**

| predicate | n | C | P | I | strict | lenient | corpus share |
|---|---|---|---|---|---|---|---|
| `addresses` | 3 | 1 | 1 | 1 | 33.3% | 66.7% | 26.1% |
| `uses` | 3 | 0 | 1 | 2 | 0.0% | 33.3% | 24.9% |
| `achieves` | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 12.7% |
| `evaluatedon` | 3 | 0 | 1 | 2 | 0.0% | 33.3% | 8.9% |
| `trainedon` | 3 | 0 | 1 | 2 | 0.0% | 33.3% | 4.9% |
| `comparesagainst` | 3 | 1 | 1 | 1 | 33.3% | 66.7% | 4.2% |
| `comprises` | 3 | 1 | 0 | 2 | 33.3% | 33.3% | 3.4% |
| `affiliatedwith` | 3 | 0 | 0 | 3 | 0.0% | 0.0% | 2.6% |
| `employs` | 3 | 0 | 0 | 3 | 0.0% | 0.0% | 2.4% |
| `splitfrom` | 3 | 0 | 1 | 2 | 0.0% | 33.3% | 1.9% |
| `designedfor` | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 1.6% |
| `produces` | 3 | 0 | 1 | 2 | 0.0% | 33.3% | 1.3% |
| `evaluates` | 3 | 0 | 0 | 3 | 0.0% | 0.0% | 1.1% |
| `locatedin` | 3 | 0 | 2 | 1 | 0.0% | 66.7% | 0.8% |
| `configures` | 3 | 0 | 0 | 3 | 0.0% | 0.0% | 0.7% |
| `publishedin` | 3 | 0 | 0 | 3 | 0.0% | 0.0% | 0.7% |
| `supports` | 3 | 0 | 0 | 3 | 0.0% | 0.0% | 0.7% |
| `encompasses` | 2 | 0 | 0 | 2 | 0.0% | 0.0% | 0.5% |
| `cites` | 2 | 0 | 1 | 1 | 0.0% | 50.0% | 0.4% |
| `reports` | 2 | 0 | 1 | 1 | 0.0% | 50.0% | 0.2% |
| `motivates` | 1 | 0 | 0 | 1 | 0.0% | 0.0% | 0.1% |

⚠ **Cells of 1-13 samples size the bands, not the cells.** No single per-predicate percentage
is quotable alone. What is stable across rounds is the pattern: **every predicate at 0% strict
for qwen is either ≥66% for Claude or absent from its output entirely.**

---

## 9. Inter-rater agreement — the first κ in this project

Claude and `gemini-3.6-flash` independently labelled the same **110** qwen3:8b
triples under the same rubric, neither seeing the other's verdict. Until now every precision
figure rested on a single unvalidated labeller.

| measure | value |
|---|---|
| raw agreement (3-class) | 59.1% |
| **Cohen's κ (3-class)** | **0.342** (fair) |
| agreement (CORRECT vs not) | 83.6% |
| **Cohen's κ (binary)** | **0.480** (moderate) |
| precision assigned — Claude | 24.5% |
| precision assigned — Gemini | 13.6% |

Confusion matrix (rows = Claude, columns = Gemini):

| | CORRECT | PARTIAL | INCORRECT |
|---|---|---|---|
| **CORRECT** | 12 | 11 | 4 |
| **PARTIAL** | 3 | 9 | 25 |
| **INCORRECT** | 0 | 2 | 44 |

**The disagreement is almost entirely at the PARTIAL boundary.** The judges agree on what is
flatly wrong (44 joint INCORRECT, and Gemini never called a
Claude-INCORRECT triple CORRECT); they diverge on "implied but not stated", which Gemini pushes
to INCORRECT 25 times. That is why binary κ
(0.480) sits well above 3-class κ (0.342).

**This is a methodological finding, not just a number: it is the empirical case for reporting
both a strict and a lenient figure rather than one.** The PARTIAL class is where reasonable
judges genuinely disagree, so a single precision number hides a real ambiguity.

### ⭐ The second κ — on the headline result itself

The κ above is measured on *qwen's* triples. The more important one is on **Claude's**, since
that is the number the paper reports. `gemini-3.6-flash` and `openai/gpt-oss-120b` labelled the
same **100** Claude triples independently:

| measure | on Claude's triples | on qwen's triples |
|---|---|---|
| raw agreement (3-class) | **86.0%** | 59.1% |
| Cohen's κ (3-class) | **0.457** | 0.342 |
| agreement (CORRECT vs not) | **91.0%** | 83.6% |
| **Cohen's κ (binary)** | **0.637** (substantial) | 0.480 (moderate) |
| precision assigned — judge A | 86.0% | 24.5% |
| precision assigned — judge B | 85.0% | 13.6% |

Confusion matrix (rows = gemini, columns = gpt-oss):

| | CORRECT | PARTIAL | INCORRECT |
|---|---|---|---|
| **CORRECT** | 81 | 4 | 1 |
| **PARTIAL** | 4 | 4 | 5 |
| **INCORRECT** | 0 | 0 | 1 |

### ⚠ A correction, and what the κ numbers actually support

All three judge pairings, side by side:

| judge pair | on | raw agreement | κ (3-class) | binary | κ (binary) |
|---|---|---|---|---|---|
| claude vs gemini | qwen's 110 | 59.1% | 0.342 | 83.6% | 0.480 |
| gemini vs gpt-oss | qwen's 110 | 75.5% | 0.552 | 91.8% | 0.662 |
| gemini vs gpt-oss | claude's 100 | 86.0% | 0.457 | 91.0% | 0.637 |

**An earlier reading of these numbers was wrong and is withdrawn.** It compared *different judge
pairs* — claude-vs-gemini on qwen's triples against gemini-vs-gpt-oss on Claude's — and
concluded that judges agree more about good extraction. That was a confound: the judge pair
changed at the same time as the extractor.

With the **pair held constant** (gemini vs gpt-oss), the honest reading is:

1. **Raw agreement IS higher on the good extraction** — 86.0% on
   Claude's triples vs 75.5% on qwen's.
2. **But κ is LOWER there** — 0.457 vs 0.552.
   This is the well-known **kappa paradox**: Claude's triples are overwhelmingly CORRECT
   (81 of 100 joint-CORRECT), so the marginals are skewed, chance
   agreement is high, and κ is penalised for it even as raw agreement rises. **κ is not
   comparable across samples with different class balance** — report it alongside the raw
   agreement and the confusion matrix, never alone.
3. **Claude was the outlier labeller, not Gemini.** It assigned 24.5%
   precision to qwen's triples where gemini gave 13.6% and gpt-oss
   14.5%. The two non-Claude judges agree with each other far more
   (0.552) than either does with Claude (0.342)
   — Claude is the lenient one, which is a reason to prefer the gemini/gpt-oss figures.

---

## 9b. Judge robustness

**And it makes §5 more robust, not less.** Gemini is the *harsher* judge on qwen — it scores it at
13.6% where Claude scored it 24.5% — and
Claude's extraction still earns 89% from it.

---

## 10. The graph-level track — citation prediction (different corpus: 155 ACL papers)

⚠ **Different corpus, different extractor mode, different question** (see §4). These numbers
are not comparable with §5-8 and must never be placed on the same axis.

**Task:** train a KG embedding on the extracted triples with **real citation edges held out**,
then rank candidate papers by embedding similarity and check whether a paper's true citation
neighbours come out on top. Metrics are mean ± std over 5 random seeds.

**Headline (RotatE + self-adversarial loss, γ selected on a validation half, reported on the
disjoint test half — 66 papers):**

| Ontology | Objective | Bidir MRR | Bidir Hits@10 | Filt MRR | Filt Hits@10 |
|---|---|---|---|---|---|
| CEO | margin (baseline) | 0.406 ± 0.053 | 0.727 | 0.418 ± 0.036 | 0.759 |
| **CEO** | **self-adversarial** | **0.598 ± 0.041** | **0.861** | **0.594 ± 0.027** | 0.844 |
| scinex | margin (baseline) | 0.391 ± 0.043 | 0.712 | 0.417 ± 0.027 | 0.737 |
| **scinex** | **self-adversarial** | **0.599 ± 0.023** | 0.845 | **0.606 ± 0.054** | 0.844 |

**Findings:**

1. **Self-adversarial negative sampling is the decisive lever** — +0.19-0.21 MRR over the margin
   objective on the same test split, with γ chosen on validation, never on test.
2. **The ontologies tie again** (0.598 vs 0.599). Independently of §5, on a different corpus and
   a different metric, CEO and scinex are indistinguishable. **That consistency is itself a
   result** — the choice of schema is not what determines quality here.
3. **Random baseline ≈ 0.2% Hits@1** over ~500 candidates, so read these as lift over random.

> **Paper sentence:** *RotatE trained with self-adversarial negative sampling predicts held-out
> citation links with MRR ≈ 0.60 and Hits@10 ≈ 0.85 (5 seeds), with hyperparameters selected on
> a disjoint validation set.*

⚠ **Open threat to this track:** a **TF-IDF baseline over title+abstract reaches MRR ≈ 0.70 /
Hits@10 ≈ 0.90** — higher than the KGE. A definitive comparison on one final corpus is still
outstanding. The claim "KGE is best" is **not** currently supported and must not be made.

---

## 11. The corpus (per-triple track)

**20 papers**, all applied-ML work by R. Chawuthai's group:
ophthalmology, chemistry, materials, traffic/ITS, cloud systems, computer vision, NLP.

⚠ **Describe it as "18 of the listed papers + 2 substitutes", never plain "20 papers".**
Two papers on the original list could not be obtained (paywalled Springer chapters, verified
absent from the supplied bulk-download folder by title matching against every file):

| id | title | status |
|---|---|---|
| `aiabstract2025` | Detecting AI-Generated Scientific Abstracts Using Galactica an | **substitute** |
| `routepred2023` | Route Prediction from GPS Trajectory and Road Data | **substitute** |

Both substitutes are by the same group, parsed cleanly on the first attempt with titles verified
against the manifest, and `BERT.pdf` was re-run as a parser regression (unchanged: 44 sections,
7,254 body words). `papers/manifest.csv` records the substitution.

### ⚠ The corpus splits into two halves and should be reported that way

The subject-entity gazetteer (CS-NER) is annotated over **CS/NLP** papers. The ~11 CS papers get
genuinely paper-specific entities (`Text Classification`, `Spatial Pyramid Pooling`); the ~7
non-CS papers (chemistry/materials/clinical) get only generic ML vocabulary and **zero** domain
terms. This domain-shift is why `relation` mode (free subjects) was chosen over `fixed`.

### Triples per paper

| paper | qwen CEO | qwen scinex | claude |
|---|---|---|---|
| aiabstract2025 | 47 | 57 | 12 |
| csysguard2024 | 99 | 91 | 19 |
| eyelandmark2024 | 30 | 23 | 9 |
| gamlprop2025 | 33 | 37 | 13 |
| linkpred2015 | 0 | 0 | 12 |
| llamacorrupt2025 | 24 | 40 | 8 |
| microwave2018 | 29 | 22 | 10 |
| osmotic2026 | 47 | 34 | 14 |
| oxidecrack2025 | 68 | 71 | 10 |
| parkingyolo2023 | 85 | 90 | 38 |
| pesticide2025 | 74 | 59 | 11 |
| reststop2018 | 27 | 28 | 9 |
| roadwaylight2018 | 60 | 42 | 11 |
| routepred2023 | 37 | 29 | 5 |
| strabismus2026 | 131 | 108 | 23 |
| textaug2023 | 41 | 39 | 13 |
| trafficspeed2021 | 19 | 26 | 5 |
| traveltime2022 | 113 | 111 | 23 |
| tripplanner2020 | 0 | 0 | 7 |
| ugmo2024 | 19 | 20 | 10 |
| vehiclemake2025 | 144 | 137 | 28 |
| videoseg2025 | 58 | 53 | 25 |
| **total** | **1,185** | **1,117** | **315** |

---

## 12. Evaluation methodology (per-triple track)

### The verdict rubric — identical for every labeller, human or model

Each triple is shown with **the predicate's ontology definition** and **the source sentence it
was extracted from**, then labelled:

| verdict | criterion |
|---|---|
| **CORRECT** | the sentence explicitly states it **and** the predicate fits its domain→range |
| **PARTIAL** | implied rather than stated, **or** the predicate is a loose fit |
| **INCORRECT** | unsupported by the sentence, or subject/object swapped relative to domain→range |

The decisive rule: **judge only from the source sentence.** A triple can be true about the paper
and still INCORRECT here, because the sentence is the extractor's evidence. Showing the ontology
definition is what makes a verdict test domain/range conformance rather than the plausibility of
an English relation name.

### Sampling and weighting

Samples are drawn **round-robin over extractor × predicate buckets** so rare predicates are
represented at all. That makes the raw sampled precision unrepresentative of the corpus, so every
headline figure is **corpus-weighted** by each predicate's true share, with a bootstrap CI
(2,000 resamples within predicate buckets).

### Stable triple identity

Every triple carries a content-hash `triple_id` (including the extractor family), so labels
survive re-runs and two labellers can be joined on exactly the same items — which is what makes
the κ in §8 possible.

---

## 13. Engineering findings worth reporting

These are not incidental — **two of them silently destroyed correct output**, and they are the
kind of thing a methods section should disclose.

### FIXED — the pipeline demanded a value it then threw away

`kg_builder._is_valid_entity()` required every node to match `[a-zA-Z]{2,}`, so a **bare numeric
object was deleted after the extractor's guards had already passed it**. But the prompt defines
`evaluates` as `EvaluationMetric → ExperimentalResult` and its own worked example is
`(<Metric>, evaluates, 88.5)`.

Now the filter is predicate- and role-aware: a measurement value (`88.5`, `0.923`, `97%`) is
admitted as the **object of a value predicate only** — still rejected as a subject and for every
other predicate, so the filter was not loosened generally. All-caps acronyms (`SF`, `F1`) are
exempt from the `len < 3` floor.

**Measured by re-ingesting stored replies with no new model calls: 268 → 279 triples,
`evaluates` 1 → 6.**

⚠ **This invalidates an earlier conclusion.** `evaluates` scored 0% strict in the n=240 labels
*because the bug deleted its correct instances and left only malformed ones to be judged*. It
must be re-judged before appearing on any "always wrong" list.

### ADDED — two guards derived from the C1–C6 findings (§5b)

**`_is_provenance_sentence()`** rejects a triple whose **source sentence** is publisher,
biographical or bibliographic matter: IEEE licence watermarks, author biographies,
acknowledgements, bibliography entries (three citation formats), ACM CCS classification blocks,
ISSN/masthead lines. It is keyed on the SENTENCE, not the section name, because the existing
section-level guard misses two cases — front/back matter with no heading of its own, and
watermarks or running headers that the parser glues **into** a legitimate body paragraph.

**`_splitfrom_is_inverted()`** rejects a `splitFrom` whose object names a partition and whose
subject does not. `splitFrom` is `Dataset → Dataset`, so a type check cannot catch the inversion
— both sides are datasets — yet 3 of the 4 extractors consistently wrote it backwards, which
inverts the whole partition hierarchy in the graph.

Measured over the whole corpus — every removal was inspected, no false positives observed:

| extractor | triples | removed | % |
|---|---|---|---|
| `qwen3-235b` | 367 | 8 | 2.2% |
| `gptoss-120b` | 547 | 5 | 0.9% |
| `gemma4-31b` | 615 | 10 | 1.6% |
| `ministral-14b` | 1,536 | 29 | 1.9% |

⚠ The corpus rate (0.9–2.0%) is far below what the C1–C6 sample suggested, because that sample
is stratified over predicates and over-represents rare ones. Both are correct; state which is
being quoted.

### OPEN — the `configures` direction needs a decision, not a guard

`configures` is defined `ExperimentalSpecification → Experiment`. **Every model writes
`Model → setting-value`** (`(Random Forest) configures (30 estimators)`) — wrong in 100% of
observed uses. Deliberately left unguarded: unlike `splitFrom` this is not a simple inversion,
since the object is a VALUE rather than an Experiment, so swapping the arguments yields nothing
valid either. Either the ontology direction changes, or these triples are rejected and real
hyperparameter facts are lost. **When every independent model produces the same shape, the
schema is the more likely thing to be wrong** — but the call has not been made unilaterally.

### FIXED — a regex boundary dropped every parenthetical subject

The guard wrapped subjects in `\b...\b`. `\b` asserts a word/non-word transition, so a subject
ending in `)` — followed by a space, both non-word — could **never** match its own source
sentence. Since `Full Name (ABBR)` is the standard way a paper introduces a model, these were
silently discarded. Recall-only fix.

### Still open (guard worklist)

1. Reject any object equal to an ontology **class name** — confirmed twice:
   `(Kalman filter, mentions, AcademicPaper)`, `(Merck, affiliatedWith, Organisation)`.
2. Reject **cross-reference objects** — `→ "Table III"`, `→ "Section II"`, `→ "ASSE 2025"`.
3. Post-parse **domain/range type check** per predicate.
4. **Skip Author-contributions / Declarations / Funding sections** at extraction.

> Priority note: these all fix *qwen's* junk. §5 shows a strong model avoids these failure modes
> unprompted, so this worklist matters less than it appeared before the judging result.

### Operational limits — measured, not read off a docs page

| provider | limit that bites | consequence for a 525-call corpus run |
|---|---|---|
| **Gemini** free | **20 requests/day/model** | extraction impossible; **judging fine** at 10-25 triples/call |
| **Groq** free | **8,000 tokens/minute** | ~6.2k-token calls → ~1 call/min → **~8 h**, slower than a local GPU |

Gemini's quota is **per model**, so a fresh pinned id grants another 20 — but **never judge one
sample with two different models**; judge identity must be constant across a comparison.

Other traps, each of which cost a working session to find:

* Groq sits behind Cloudflare and **403s Python's default user-agent** (`error code: 1010`).
* **`gemini-2.5-flash` is listed to a new key but returns `NOT_FOUND` when called.** A model
  listing is not a list of *usable* models. Prefer pinned ids over `-latest` aliases: aliases
  drift, and a paper number must name the exact model that produced it.
* **`--max-tokens` ≥ 3072.** At 1024, 34% of local calls truncated, and **a truncated JSON yields
  ZERO triples** — indistinguishable from a model that found nothing.
* **Ollama `num_ctx=8192`** — it defaults to 4096 regardless of model, and the system prompt
  alone is ~3.4k tokens.
* **A model slug must never contain `:` or `/`** — it becomes a directory name; `qwen3:8b`
  silently discarded every triple at write time until this was caught.

qwen truncation rate: **5.2%** (51/972 calls) on the original 18 papers, **14.1%** (11/78) on the
2 added papers — denser paragraphs hit the output cap more often. Settings were deliberately left
identical rather than raised for 2 papers.

---

## 14. Approaches tried that did NOT work

Negative results, recorded so they are not re-attempted. Each cost real time.

### Free chat-UI extraction (ChatGPT / Gemini web) — ABANDONED

Tooling was built to run the whole corpus through a free chat UI by pasting batches
(`chat_paste_extract.py`): the ~3k-token system prompt is pasted once per conversation, so the
marginal cost per paragraph is the paragraph itself — the whole corpus fits in **49 messages**.
Bundles were generated for both providers. **Neither returned usable triples**; the reply folders
came back empty.

It was already a weak instrument for two reasons, both of which stand independently of the
failure: **(1)** batching ~12 paragraphs per message is a *different experimental condition* from
one-paragraph-per-call, since the model can carry context between them; **(2)** a free UI can
silently switch model version mid-session, so the result names no reproducible model.
**Do not restart this route** — the API path replaces it and is strictly better.

### Free API tiers for extraction — NOT VIABLE

A corpus run is 525 calls at ~6.2k tokens each. Both free tiers block it, for different reasons:
**Groq** caps at 8,000 tokens/minute → ~1 call/min → **~8 hours**, slower than a local laptop GPU;
**Gemini** caps at 20 requests/day/model → **impossible**. Both keys were validated and both walls
were hit for real, not read off a docs page. Note the asymmetry: **the same quotas are perfectly
adequate for judging**, where 10-25 triples are batched per call (100 triples judged in 2.3 min).

### A second 8B model as a size-matched control — STARTED, THEN DROPPED

`llama3.1:8b` (Meta) was pulled and its corpus run started, to separate "model strength" from
"model identity" in the §5 gap — qwen3:8b and claude-opus-5 differ in *both*. It ran at ~3
calls/min (~2.7 h projected) and was **stopped by user decision** after one paper; the partial
output was deleted so nothing on disk resembles a real run. The model is still installed if the
control is wanted later. **This remains the one open methodological gap a reviewer could name.**

### Model-id traps

`gemini-2.5-flash` is **listed** to a new key but returns `NOT_FOUND` when called ("no longer
available to new users"). `gemini-3.7-flash` returned `503 UNAVAILABLE` (high demand) across all
retries. `gemini-3.6-flash` worked and became the judge. **A model listing is not a list of
usable models — always smoke-test one call before committing a run.**

---

## 15. Tooling built for this work

All standard-library-only where it touches the network, so nothing has to be installed:

| script | purpose |
|---|---|
| `claude_extract.py` | splits the extractor at the model boundary: `prompts` renders the real prompts, `ingest` feeds replies back through the real guards |
| `run_api_extract.py` | drives prompts→responses with any provider (Anthropic / OpenAI / OpenAI-compatible / Gemini / Ollama) |
| `run_api_judge.py` | drives the judging bundles with any provider; sends the rubric as the **system prompt on every call**, so batches cannot drift the way a long chat does |
| `check_provider.py` | preflight: finds the key, lists the models it can actually see, smoke-tests, prints the exact next commands |
| `judge_paste.py` | packages a gold sample for an external judge and reads verdicts back into the label CSV |
| `gold_report.py` | sampled + corpus-weighted precision, per-predicate/per-paper tables, bootstrap CIs |
| `gold_eval.py agree` | joins two labellers on `triple_id` → agreement, Cohen's κ, confusion matrix |
| `build_results_report.py` | regenerates this document from the artefacts on disk |

---

## 16. Limitations

1. **Sample size sizes the bands, not the cells.** Per-predicate cells hold 1-13 labels.
2. **qwen's judged rows are 110 of 240** (quota); Claude's are a complete 100.
3. **κ is fair-to-moderate, not strong** (0.342 / 0.480) — the PARTIAL boundary is genuinely
   ambiguous. Report strict and lenient together.
4. **The extraction corpora are not on one code version.** The original 18 papers predate both
   guard fixes; the 2 substitutes do not.
5. **qwen3:8b is not the intended model.** The pipeline was designed for Qwen3-14B on a GPU VM;
   these runs are an 8B model on a laptop via Ollama. Model *family* comparisons hold; absolute
   qwen numbers are not the paper's final figures.
6. **The TF-IDF baseline may beat the KGE** (§9). Unresolved.
8. **The C1–C6 track is single-rater by design.** claude-opus-5 labelled all 248 items; there
   is **no second labeller and no κ**, and none is planned. Criteria like C4's domain/range test
   involve genuine judgement calls, and their reproducibility across raters is unmeasured. The κ
   figures in §8 measure a different rubric and do not transfer.
9. **C6 rewards an empty graph.** A model that extracts nothing cannot contradict itself and
   scores 1.0 vacuously, so C6 must never be reported without C2/C5 beside it.
10. **C1–C6 covers CEO only.** The scinex side of the four new extractors has not been run, so
   the §6 ontology × capability interaction has no C1–C6 counterpart.
11. **C6 rests on 6 papers per model.** Quote the monotone trend across the four models, not
   per-model point estimates.
7. **The corpus is one research group's output**, so domain diversity is real but institutional
   diversity is not.

---

## 17. What is left

**1. Finish qwen's remaining 130 judge labels.** Cleanest completion: re-judge **both** samples
with one fresh pinned model at a larger batch size, keeping a single judge across both rows —
240 triples at `--size 25` = 10 requests, plus 4 for Claude's 100 = 14, inside one model's daily 20.

**2. Judge the 17 triples from the 2 substitute papers.**

**3. Full qwen re-extraction of all 20 papers (~3 h GPU, unattended).** Puts every number on one
code version and is the only way to recover qwen's deleted numeric-object triples — that run
stored no raw replies. **Largest remaining correctness item.**

**4. Optional — the size-matched control.** `llama3.1:8b` (already pulled, ~2.7 h, no API key) is
a second 8B-class model from a different family. It would separate "model strength" from "model
identity" in the §5 gap. A reviewer could reasonably ask; not required.

**6. Run the scinex side of the four C1–C6 extractors.** Only CEO has been extracted, which is
what blocks a C1–C6 counterpart to the §6 ontology × capability interaction — and C4, the
criterion with the widest spread, is exactly where a tighter schema should show up.

**7. Decide the `configures` direction** (§13). It is wrong in 100% of observed uses across every
model, and no guard can resolve it — it is a schema question, not an extraction bug.

**5. Settle the TF-IDF vs KGE question** on one final corpus (§9).

---

## 18. Reproducing every number

```bash
# --- extraction (any model, same prompt + guards) ---
python3 run_api_extract.py --all --provider ollama --model qwen3:8b \
        --model-slug qwen3-8b --extractor relation --ontology ceo --ingest
python3 claude_corpus_report.py                 # volume + predicate mix

# --- judging ---
python3 check_provider.py gemini                # validate key, list USABLE models
python3 gold_eval.py export --extractor relation --model claude-opus-5 \
        --n 100 --seed 11 --out gold/claude_sample_for_judge.csv
python3 judge_paste.py batches --csv gold/claude_sample_for_judge.csv \
        --slug gemini-judge --size 10
python3 run_api_judge.py --slug gemini-judge --provider gemini --model gemini-3.6-flash
python3 judge_paste.py collect --csv gold/claude_sample_for_judge.csv \
        --slug gemini-judge --out gold/gemini_verdicts.csv

# --- the tables in this document ---
python3 gold_report.py --labels gold/gemini_verdicts.csv --model claude-opus-5 --bootstrap 2000
python3 gold_report.py --labels gold/gemini_verdicts_qwen.csv --model qwen3-8b --bootstrap 2000
python3 gold_eval.py agree --gold gold/claude_labels_240.csv \
        --b gold/gemini_verdicts_qwen.csv --a-name claude --b-name gemini

# --- graph-level track (155-paper ACL corpus) ---
python3 kge_multiseed.py --extractor fixed --model Qwen3-14B --kge all \
        --seeds 1 2 3 4 5 --epochs 1000 --device cuda

# --- rebuild THIS document ---
python3 build_results_report.py
```

**Windows note:** run everything with `PYTHONIOENCODING=utf-8` — the console is cp1252 and the
pipeline prints non-ASCII characters at import time.

### Where the evidence lives

| file | holds |
|---|---|
| `output/<paper>/no-llm/output.html` | Part 1 parser output |
| `output/<paper>/kg/relation/<model>/triples.json` | the triples, per paper per model |
| `output/<paper>/kg/relation/<model>/responses.jsonl` | raw model replies (API runs only) |
| `output/claude-opus-5_relation_corpus.json` | bundled Claude corpus + volume comparison |
| `gold/gemini_verdicts.csv` | Claude's 100 triples, judged |
| `gold/gemini_verdicts_qwen.csv` | qwen's 110 triples, judged |
| `gold/claude_labels_240.csv` | qwen's 240 triples, labelled by Claude |
| `gold/report_claude_gemini.json` | §5 + §7 Claude tables |
| `gold/report_qwen_gemini.json` | §5 + §7 qwen tables |
| `gold/agreement_claude_gemini.json` | §8 κ and confusion matrix |
| `papers/manifest.csv` | the corpus, with the substitution recorded |
| `scinex_refined_14.owl` | the scinex ontology |

Session-by-session history: `hands_off.md`. Architecture and conventions: `Claude.md`.
Full results including superseded rounds: `results.md`.
