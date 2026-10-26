# TASK — Fixed-extraction accuracy on the 20-paper corpus

> ## ⭐ CURRENT TASK (Session 36, 2026-10-26): Part 2 finetuning — FOLD 0 DONE AND JUDGED
> **Start with `HANDOFF_NEXT_SESSION.md`.** Detail: `hands_off.md` §36; numbers: `results.md`
> Tables 13–14. The RTX 5090 VM plan was dropped; everything below ran on a **Windows laptop with an
> RTX 4060 Laptop GPU, 8 GB, sm_89**, student **`Qwen/Qwen3-1.7B`** (not the 4B/8B of earlier plans).
> * **DONE:** both data variants built from 13 of 15 training papers (619 extraction examples;
>   consensus +129), both trained, both generated on the 5 held-out fold-0 papers × both ontologies,
>   compared judge-free, and **blind-judged on CEO** (census: 355 model-triples → 315 unique items).
> * **⭐⭐⭐ RESULT (CEO):** C4 relation correctness **29.4% untuned → 83.3% (teacher-only) / 85.4%
>   (consensus)** against the **120B teacher's 86.4%**; outright relation errors 63.6% → ~7%.
>   **A finetuned 1.7B student matches the teacher per triple.**
> * **⚖ Teacher-only vs consensus:** quality is a tie (C4 +2.0 pts [−12.7, +16.8]); consensus gives
>   **+62% more triples**. **Adopt consensus.**
> * **⛔ THE OPEN PROBLEM:** both students answer `{"triples": []}` for 90–95% of paragraphs —
>   coverage 4.6–7.5% against the teacher's 42.0%. Cause: **402 of 619 training targets (65%) are
>   empty**. The fix is built but untrained: `ft_data/maxempty30/` (`--max-empty-frac 0.3`).
> * **NEXT:** scinex census → train the maxempty30 variant → 4-fold CV with the winner → Part 3 Q&A
>   with an independent non-Claude model (OpenRouter credit earmarked). Commands in the handoff.
>
> ### Session 32 plan (kept; see the notes above for what changed)
> Full narrative + open decisions: `hands_off.md` §32. Order of operations:
>
> | step | command | model / API? |
> |---|---|---|
> | 1 build data (fold 0 held out) | `python3 build_ft_data.py --eval-fold 0` — check every train paper×ontology says `ok`, alignment ≥ 0.9 | no |
> | 2 tokenizer check | `python3 ft_train_qlora.py --data ft_data/fold0 --base-model Qwen/Qwen3-8B --run-name qwen3-8b-f0 --dry-run` | tokenizer only |
> | 3 train — **only after Sam confirms the base model** | same command without `--dry-run` | GPU |
> | 4 generate held-out KGs | `python3 ft_generate.py --adapter ft_runs/qwen3-8b-f0 --model-slug qwen3-8b-ft-cv --ingest` and `python3 ft_generate.py --base-model Qwen/Qwen3-8B --model-slug qwen3-8b-base-hf --papers-from ft_data/fold0/eval_papers.txt --ingest` | GPU |
> | 5 C1/C3/C4, census of the held-out papers | `python3 ontology_eval.py triples --model <slug> --n 100000 --size 25 --papers-file ft_data/fold0/eval_papers.txt` → `python3 run_api_judge.py --dest ontology_eval/<slug>/ceo --slug triples --rubric triples --provider … --model <non-Claude judge>` → `ontology_eval.py collect` / `report` — same judge for base and FT | judge API |
> | 6 Q&A eval | `qa_eval.py generate` → `answer --context kg/oracle/closedbook` → `grade` → `report --compare kg__relation__qwen3-8b-ft-cv kg__relation__qwen3-8b-base-hf` | API |
> | 7 cross-validation | repeat 1–4 with `--eval-fold 1..3`, **same `--model-slug`**, then run 5–6 over all 20 papers | GPU |
>
> **Hard rule, enforced in code three times** (build, train, generate): training and eval papers are
> disjoint, by paper id, via the sticky fold file `ft_data/split.json`.

**Started:** 2026-08-13 · **Owner:** Sam · **Status (2026-10-26):** Part 1 and Part 2c complete
(relation extraction over the 20-paper corpus, C1–C6 judged — see `results.md`). **Part 2
finetuning: fold 0 complete and judged** (Tables 13–14) — a finetuned Qwen3-1.7B matches the 120B
teacher per triple, but extracts ~6× less; recall is the open problem. **Part 3 Q&A: tooling built,
needs an independent question writer/answerer.** Earlier note, still true: CEO vs scinex is a TIE —
the "CEO wins" reading is withdrawn (`hands_off.md` §25).

This file explains the *current* task end to end: what we are trying to measure, why the
pipeline is set up this way, and what to run in what order. `Claude.md` is the persistent
architecture overview, `hands_off.md` is the dated running log, `results.md` holds the
paper-ready numbers. **This file is the task brief that ties those three together.**

---

## 1. The goal

Push the **per-triple accuracy of fixed extraction** as high as it will go, and measure it
in a way that survives review.

"Measured properly" means two things:

1. **An LLM judge labels every triple** against the sentence it came from.
2. **A human labels a stratified sample of the same triples**, and we report how well the
   judge agrees with the human (Cohen's κ). The human is ground truth. The judge's
   corpus-wide numbers are only quotable *after* that agreement is established.

Only then do we act on the judge's error list: fix the prompt and the post-parse guards,
re-extract, re-measure.

**Done looks like:** a defensible precision figure per ontology on ~20 papers, with a κ
that justifies trusting the judge, plus a before/after showing what the prompt fixes bought.

### What is deliberately out of scope

- **Only fixed extraction.** REBEL, ITER, free-LLM and pair extraction are not part of this task.
- **All existing KGE / citation-prediction results are frozen.** RotatE MRR ~0.60, the
  ontology tie, the text-baseline problem — untouched and still valid. Nothing here
  re-opens them.
- **The 315-paper ACL corpus is not the corpus for this task.** Its results stand as they are.

---

## 2. The corpus

20 papers from R. Chawuthai's group, given 2026-08-13. Applied ML across ophthalmology,
chemistry, materials, traffic/ITS, cloud systems, computer vision and NLP.

Authoritative list with DOIs, access status and PDF URLs: **`papers/manifest.csv`**.

**Paper id = PDF stem.** `papers/strabismus2026.pdf` → `output/strabismus2026/`. The id
propagates through every downstream path, so do not rename PDFs after parsing.

### Access status

| Status | Count | Meaning |
|---|---|---|
| `oa` | 7 | Downloaded by `fetch_corpus.sh` — Sci Rep ×3, Sensors, Applied Sciences, Sensors & Materials, Chem Eng Transactions |
| `oa_blocked` | 3 | Open access, but the publisher blocks scripted downloads (IEEE Xplore ×2, ScienceDirect ×1) — fetch by hand |
| `paywalled` | 10 | Institutional access needed (IEEE conf ×5, Springer ×2, ACM ×1, Elsevier ×1) |

```bash
bash fetch_corpus.sh          # idempotent; prints the manual-download list for the rest
```

Manual downloads go to `papers/<paper_id>.pdf` using **the id in the manifest** — the
filename is the id, so a wrong name silently forks the output tree.

> **⚠ Verify what you downloaded.** A DOI suffix is not the publisher's PDF number:
> guessing `SM1842.pdf` from DOI `10.18494/SAM.2018.1842` returned a valid PDF of a
> *different paper* (the real one is `SM1674.pdf`). PDF text streams are compressed, so
> `grep`/`strings` cannot confirm a title. **The check is `main.py`'s parsed `Title:` log
> line — compare all of them against `manifest.csv` on the first parse run.**

---

## 3. Why there are two extraction modes

The fixed extractor's subject pool is the **CS-NER gazetteer ∩ the paper's own text**.
CS-NER is annotated over CS/NLP papers. Half this corpus is not: an osmotic-coefficient
paper, an oxide-scale-cracking paper, a pesticide spectrophotometer, a strabismus screen.
Their ML vocabulary ("random forest", "CNN", "F1") hits the gazetteer; their *domain*
vocabulary ("alkyl ammonium salts", "oxide scale", "horizontal strabismus") does not.

Left alone, those papers would produce almost no triples — and the number would look like
an **extraction-quality** failure when it is really an **entity-coverage** failure. Two
different problems that must not be reported as one.

So, per the user's instruction — *try every entity source, and where there is none, fix
only the relation*:

| Extractor | Subject | Relation | Object | Use |
|---|---|---|---|---|
| `llm` | free | free | free | not in this task |
| **`relation`** | **free** (must appear literally in the sentence) | **∈ ontology** | free | **fallback** when no entity list covers the paper |
| **`fixed`** | **∈ entity CSV** | **∈ ontology** | free | **main path** |
| `pair` | ∈ entity CSV | free | ∈ entity CSV | not in this task |

`RelationOnlyExtractor` subclasses `FixedTripleExtractor` and reuses its prompt and *every*
post-parse guard. Only the subject constraint differs — which is what makes a `fixed` vs
`relation` comparison meaningful: it isolates exactly what the curated entity list buys.

**Both ontologies are run for both modes**, into parallel output dirs so nothing overwrites
anything: `kg/fixed/`, `kg/fixed_scinex/`, `kg/relation/`, `kg/relation_scinex/`.

---

## 4. The pipeline, in order

All on the VM. **`python3`, never `python`.** One GPU process at a time.

### Stage 1 — parse
```bash
for f in papers/*.pdf; do python3 main.py "$f" --no-llm; done
```
→ `output/<id>/no-llm/output.html`. **Check every `Title:` against `manifest.csv` here.**

> **⚠ 2026-08-25 — the parser was broken for this corpus and is now MOSTLY FIXED
> (`hands_off.md §9` = the audit, `§10` = the fix and what's left).**
> All 7 papers now produce real body text: the 3 Scientific Reports papers went from
> **0 body words** (only their bibliography survived) to 3,077–6,456, and
> `roadwaylight2018` went from a crash to 4,116. The abstract — previously dropped from
> **every** paper — is now captured. 6 of 8 titles are correct.
> **Still open before extraction:** 2 MDPI titles are still wrong (exact one-line fixes
> named in `§10`); the 3 Nature papers have their body inside a single `Abstract` section
> because Nature glues headings onto the paragraph in a larger font — which matters
> because `source_meta["section"]` is fed to the extractor prompt.
> **`output/*/fast/*` on disk is STALE (old parser) — re-parse everything with `--no-llm`
> before extracting.** Verify a re-parse by body-word count (total minus the `References`
> section), not by file existence: a paper whose body was never parsed reports near-zero
> triples, which looks identical to an entity-coverage failure and to an
> extraction-quality failure.
>
> Local dev note: this laptop has **Python 3.14.7 with pymupdf/pdfplumber installed**, so
> Part-1 parsing can be tested here — `PYTHONPATH=. PYTHONIOENCODING=utf-8 python …`
> (the second is required, or a `✔` in `citation/network.py` crashes on cp1252).

### Stage 2 — build each paper's entity list
```bash
python3 enrich_entity_csv.py --all --source csner
```
CS-NER gazetteer (~51k entities, cached at `output/acl/csner_gazetteer.csv`) intersected
with each paper's HTML → `Entity_<id>_enriched.csv`. Works for any parsed paper; an ACL id
is not required.

The entity source **must stay non-circular** — never back-fill the subject pool from our own
open (`llm`) extraction, or the evaluation measures the extractor against itself.

### Stage 3 — decide fixed vs relation, per paper
```bash
python3 entity_coverage.py --plan output/corpus_plan.json
```
Reads files only, no GPU. Counts the TP entities each paper would actually get and splits
the corpus at `--min-entities` (default 15). It prints the exact run commands for both
groups.

Before accepting a `relation` verdict, rule out the two boring causes it reminds you about:
enrichment never ran for that paper, or the PDF never parsed.

### Stage 4 — extract, both ontologies
```bash
python3 kg_main.py --paper <id> --extractor fixed  --model Qwen/Qwen3-14B
python3 kg_main.py --paper <id> --extractor fixed  --ontology scinex --model Qwen/Qwen3-14B
# or, for the fallback group:
python3 kg_main.py --paper <id> --extractor relation --model Qwen/Qwen3-14B
python3 kg_main.py --paper <id> --extractor relation --ontology scinex --model Qwen/Qwen3-14B
```
→ `output/<id>/kg/<extractor>/<model>/triples.json`.
`--entity-csv` is not needed — the CSV auto-resolves per paper from the id.

### Stage 5 — judge every triple
```bash
python3 kg_evaluate.py --all --extractor fixed --model Qwen/Qwen2.5-7B-Instruct \
        --summary-out output/eval_summary_fixed.json
```
Each triple is judged against its own `source_sentence` as CORRECT / PARTIAL / INCORRECT /
UNVERIFIABLE, **with the predicate's ontology definition shown to the judge** — so a verdict
tests domain/range conformance, not just whether the English relation name sounds plausible.
→ `evaluation.json` + `eval_summary.json`, with a `by_predicate` precision breakdown.

Supports `--resume` and `--max-per-paper`. Every triple carries a stable `triple_id`
(content hash including the extractor family), so labels survive re-runs.

### Stage 6 — human labels a sample
```bash
python3 gold_eval.py export --n 150 --out gold_sample.csv
```
Stratified round-robin over extractor × predicate buckets, so rare predicates are actually
represented. The CSV has a blank `verdict` column and a `predicate_definition` column
carrying the same definition the judge saw. **The judge's verdict is deliberately withheld
to avoid anchoring the labeller.**

### Stage 7 — agreement, then the error worklist
```bash
python3 gold_eval.py agree  --labels gold_sample.csv     # judge vs human
python3 gold_eval.py agree  --labels a.csv --b b.csv     # or any two labellers
python3 gold_eval.py errors --extractor fixed fixed_scinex
```
`agree` joins by `triple_id` and reports raw agreement, Cohen's κ (3-class and
CORRECT/not binary collapse), the confusion matrix, per-extractor and per-predicate tables,
and a full disagreement dump. `errors` groups the judge's PARTIAL/INCORRECT by predicate
with examples — **that grouping is the worklist for the next round of fixes.**

### Stage 8 — fix, re-extract, re-measure
Change the prompt or the guards in `fixed_extractor.py` **only where judge and human agree
the extractor is wrong**, then repeat from stage 4. Record each round's numbers.

---

## 5. The verdict rubric

Identical for human and judge — `gold_eval.py export` prints it:

- **CORRECT** — the sentence explicitly states it **and** the predicate fits its domain→range.
- **PARTIAL** — implied rather than stated, or a loose predicate fit.
- **INCORRECT** — unsupported by the sentence, or subject/object swapped relative to domain→range.
- **UNVERIFIABLE** — the sentence does not carry enough to decide.

Note that a triple can be *true about the paper* and still INCORRECT here: the claim must be
supported by **the stored source sentence**, because that sentence is the extractor's evidence.

---

## 6. What to report

- **Precision per ontology** (CEO vs scinex), precise and lenient (lenient counts PARTIAL as
  a hit), with n.
- **`by_predicate` precision** — which relations the extractor actually gets right. This is
  where prompt fixes are targeted.
- **Judge-vs-human κ**, 3-class and binary. Rough reading: <0.40 weak, 0.40–0.60 moderate,
  0.60–0.80 substantial, >0.80 strong. **A weak κ means the judge's corpus numbers are not
  quotable yet** — fix the judge (or its prompt), not just the extractor.
- **fixed vs relation**, on papers where both could run — the value of the curated entity list.
- **Coverage separately from quality** — entities per paper and how many papers fell back to
  `relation`. Never fold a coverage failure into a precision number.

Reference point from the old corpus: the BERT paper reached **80% precise / 85% lenient**
precision (30 triples, 24 CORRECT) after 12+ prompt iterations. That is the bar this corpus
is being measured against, not a target to assume.

---

## 7. Files this task owns

```
TASK.md                        — this brief
papers/manifest.csv            — the 20 papers: id, DOI, access, PDF URL
fetch_corpus.sh                — downloads the open-access subset, verifies %PDF
entity_coverage.py             — per-paper fixed-vs-relation decision (no GPU)
kg_extraction/relation_extractor.py — relation-only fallback extractor
kg_extraction/fixed_extractor.py    — the extractor under test (prompt + guards)
kg_evaluate.py                 — LLM-as-Judge over every triple
gold_eval.py                   — export sample → agree (κ) → errors worklist
output/corpus_plan.json        — the fixed/relation split actually used
```

---

## 8. Practical constraints

- **GPU:** 20GB vGPU (H100-20C). Qwen3-14B at 4-bit ≈ 10GB. **One extraction process at a
  time.** Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — unsupported on
  this vGPU. Never install a torch dev-nightly; stable `2.6.0+cu124` keeps decode at
  ~16 tok/s instead of ~2.
- **`nvidia-smi` utilization reads 0% even when busy on this vGPU.** Judge liveness by the
  log advancing, VRAM ~10–11GB, and SM clock ~1755MHz.
- **Nothing Python in this task has been executed yet** — the Windows laptop has no
  interpreter. First VM command:
  `python3 -c "import gold_eval, kg_evaluate, entity_coverage, kg_extraction"`.
- **Documentation discipline:** after any meaningful change, update `hands_off.md`
  (dated note), `Claude.md` (status block), `results.md` (paper-ready tables) in the same
  session, with exact numbers pulled from the `output/*.json` files.

---

## 8b. ⭐ SESSION 27 — THE TASK CHANGED. READ THIS BEFORE §9.

**The evaluation criteria are now C1-C6**, supplied by the user, replacing CORRECT/PARTIAL/INCORRECT:

| | criterion | unit it must be judged at |
|---|---|---|
| C1 | Concept Correctness | per triple |
| C2 | Concept Completeness | **per paragraph** (needs the paragraph + all its triples) |
| C3 | Concept Specificity | per triple |
| C4 | Relation Correctness | per triple |
| C5 | Relation Completeness | **per paragraph** |
| C6 | Semantic Consistency | **per paper** |

Refs: Zhang, Conia & Rago (IJCNLP-AACL 2025) for C1/C3/C4; Wilson et al. (Semantic Web 14(6), 2023)
for C2/C5/C6. Harness: **`ontology_eval.py`**.

**Models (fixed by the user):** Qwen3-235B, GPT-OSS-120B, Gemma-4-31B, Ministral-14B — extracted via
OpenRouter, **judged by claude-opus-5 in-session**.

> ⛔⛔ **NEVER spend OpenRouter credit on Claude.** User was explicit. Judging is in-session only.

- [x] Corpus grown to **22 papers** — user supplied the 2 missing PDFs; both parsed and extracted
- [x] All four models extracted, 22/22 papers, 0 failures (567 paragraphs each)
- [x] `ontology_eval.py` built and validated end to end (bundle -> judge -> collect -> report)
- [x] Paragraph-coverage finding: 16.4% / 42.9% / 51.7% / 69.1% of paragraphs produce any triple —
      **runs opposite to precision**, and is the reason C2/C5 matter
- [x] ⭐⭐⭐ **C1-C6 JUDGING COMPLETE (2026-09-05) — 40/40 batches, 248 judgements** (40 triples +
      16 paragraphs + 6 papers x 4 models), judged by claude-opus-5 in-session. Table + findings in
      `results.md` Table 10; narrative in `hands_off.md` §28. Headline scores (C1/C2/C3/C4/C5/C6):
      **gptoss-120b 88.8/34.4/75.0/82.5/31.2/25.0** (best), gemma4-31b 85.0/28.1/72.5/62.5/25.0/16.7,
      qwen3-235b 78.8/31.2/60.0/45.0/28.1/50.0, ministral-14b 63.7/31.2/55.0/38.8/34.4/0.0.
      **VOLUME DOES NOT BUY COMPLETENESS**: 4.3x the triples and 4.2x the paragraph coverage leaves
      C2 identical (31.2% vs 31.2%). **C6 falls monotonically with volume** (50.0 -> 25.0 -> 16.7 ->
      0.0). ⚠ C6 rewards an EMPTY graph vacuously - never report it alone
- [x] **DECIDED 2026-09-07 (user): C1-C6 stays SINGLE-RATER. No second labeller, no kappa.**
      claude-opus-5 in-session is the judge of record for all 248 judgements. This is a deliberate
      choice, not an omission - **state it as a stated limitation in the write-up**, do not describe
      C1-C6 agreement as "not yet measured" or leave it looking unfinished. The old-rubric kappa
      (results.md Table 9) measures a different rubric and does NOT transfer. Do not re-propose a
      second judge.
- [x] **Guard worklist from C1-C6 — DONE 2026-09-07 for 2 of 3.** `_is_provenance_sentence()`
      (author bios, IEEE watermarks, bibliographies, acknowledgements, ACM CCS, ISSN/masthead) and
      `_splitfrom_is_inverted()` are in `kg_extraction/fixed_extractor.py`. Removes 0.9-2.0% of each
      model's corpus; every removal eyeballed, no false positives (`hands_off.md` §29.2-29.3)
- [ ] **`configures` direction — NEEDS A HUMAN DECISION, not a guard** (`hands_off.md` §29.5).
      Defined ExperimentalSpecification->Experiment; every model writes Model->setting-value, wrong
      in 100% of observed uses. Not a simple inversion, so no guard can fix it: either change the
      ontology direction (evidence favours this) or reject and lose real hyperparameter facts
- [x] **CORPUS CORRECTED 2026-09-07 — it is the user's 20-PAPER LIST, not 22.**
      `aiabstract2025` / `routepred2023` were SUBSTITUTES for `tripplanner2020` /
      `linkpred2015`; the user supplied those PDFs in Session 27, so the substitutes are
      RETIRED. Canonical ids: `papers_20.txt`. Never write "22 papers".
- [x] **C6 REDRAWN AND RE-JUDGED on the 20-paper corpus (2026-09-07)** — 24 papers re-judged.
      qwen3-235b 50.0 / gptoss-120b 25.0 / gemma4-31b 33.3 (was 16.7) / ministral-14b 8.3 (was 0.0).
      ⛔ The monotonicity claim is WITHDRAWN: Spearman rho = -0.80, not a strict ordering.
      ⚠ The earlier "C1/C3/C4 are clean, 0/40" figure was WRONG — it read verdicts.csv, which has
      no `paper` column for the triples harness. True contamination was 1-3 of 40 per model; the
      corrected scores move by <=4.5 pts and change no ordering (`hands_off.md` §30.2)
- [x] **CEO RE-INGESTED through the Session 29 guards (2026-09-07)** — no model calls.
      395→386, 583→578, 684→674, 1709→1680 over 22 papers. All eight runs (4 models x 2 ontologies)
      are now on ONE code version, which is the precondition for any CEO-vs-scinex claim
- [x] **`RESULTS_REPORT.md` REGENERATED** — `build_results_report.py` gained §5b (C1-C6) and
      §5c (CEO vs scinex); both read the verdict CSVs at build time, restrict to the 20-paper corpus,
      and compute the Spearman rho rather than hardcoding a claim. Rebuild is idempotent
- [~] **scinex EXTRACTION COMPLETE for all 4 models** (20/20 papers each: qwen3-235b 563,
      gptoss-120b 323, gemma4-31b 525, ministral-14b 1,566). **Judged on the TRIPLES harness only** —
      C1/C3/C4, 160 judgements. ⛔ §26.16's ontology x capability interaction DOES NOT REPRODUCE
      (`hands_off.md` §30.12). **STILL OPEN: scinex C2/C5 (paragraphs) and C6 (papers)** — ~24 more
      batches, and the only way to learn whether the richer schema helps or hurts completeness and
      internal coherence

## 8c. ⭐ WHAT IS ACTUALLY LEFT (reconciled 2026-09-08)

Extraction is **finished** — 8 runs, 4 models x 2 ontologies, 20/20 papers each, all on one code
version. Judging is **complete for CEO on all six criteria** and **for scinex on three of six**.
Everything below is what remains.

**Blocked on a decision from Sam:**
1. **`configures` direction** — defined `ExperimentalSpecification -> Experiment`; every model writes
   `Model -> setting-value`, wrong in 100% of observed uses. No guard can fix it (it is not a simple
   inversion). Either change the ontology direction — the evidence favours this — or reject the
   triples and lose real hyperparameter facts. `hands_off.md` §29.5.
2. **Which Gemma for the local Ollama run.** Only `llama3.1:8b` and `qwen3:8b` are pulled;
   `gemma-4-31b` will not fit the 8 GB laptop GPU. Unanswered since Session 30.
3. **Whether to judge scinex C2/C5/C6** (~24 batches) or bank the triples-only result as a stated
   sample limitation.

**DONE 2026-09-08 (Session 31):**
- [x] Junk filter extended and applied; all 8 runs re-ingested from stored replies, **zero API calls**.
      Removed 40 of 6,054 triples (0.7%), **all ministral-14b's**.
- [x] All 5 affected samples REDRAWN and re-judged (146 new judgements) so the scores describe the
      filtered corpus.
- [x] ⛔ **KEY RESULT: n≈40 is the binding constraint.** The redraw moved scores up to +10.3 pts and
      flipped the C1 top pair. **Gaps under ~10 pts are not distinguishable.** `hands_off.md` §31.3.
- [x] `collect` made index-aware; `show_unjudged.py` added.
- [x] 8B question settled: no Gemma 8B was ever in the line-up; `qwen3-8b-v2` ran everything (20/20,
      both ontologies, replies stored) and could be judged free as a fifth capability point.

**Ready to do, no decision needed:**
4. **Raise n from ~40 to ~120 per cell** — now the single highest-value change. Costs judging time
   only, no money, and every narrow comparison is currently undecidable without it.
4b. **Batch the four remaining guard gaps into ONE round** (`hands_off.md` §31.5), then re-ingest and
   redraw once. Patching mid-measurement is what forced this session's 146-judgement redraw.
   (`hands_off.md` §30.14): a FICTIONAL UX PERSONA, CRediT lines, declarations, data-availability
   statements, affiliation blocks, OCR table debris as a subject, and author bios lacking
   "degree"/"currently". **Deliberately not done mid-measurement** — it would split the code version
   again and stale the samples just judged. Do it first in the next round, then re-ingest everything.
5. **Add the scinex tables to `results.md`.** `RESULTS_REPORT.md` has them (§5b, §5c); `results.md`
   currently carries only the correction banner.

**Open but outside the current task:**
6. `fixed` vs `relation` comparison across the CS / non-CS corpus halves — never run.
7. The old-rubric gold track (130 unlabelled qwen ids, Stage-8 re-measure). Superseded by C1-C6;
   retire it explicitly rather than leaving it looking unfinished.

---

## 9. Current state

> **⭐ The consolidated, self-contained write-up is `RESULTS_REPORT.md`** (project overview, pipeline,
> both ontologies, both evaluation tracks, results, methodology, negative results, limitations).
> It is **generated** by `build_results_report.py`, which reads every per-triple number straight from
> `output/` and `gold/` — re-run it after any new result rather than editing the markdown by hand.

- [x] Corpus identified, DOIs resolved → `papers/manifest.csv`
- [x] 7/20 PDFs downloaded and verified as real PDFs
- [x] **20 PDFs on disk (18 listed + 2 SUBSTITUTES)** — 7 OA + 8 from `bulk-download/` + 3 hand-downloaded.
      `tripplanner2020` and `linkpred2015` are **unobtainable** (paywalled Springer chapters, and verified
      NOT present in `bulk-download/` by title-matching every file). Per the user, two unused same-group
      papers were substituted: **`aiabstract2025`** (ISCON 2025, 9 sections / 4,160 words / 171 entities)
      and **`routepred2023`** (ICCAE 2023, 7 sections / 2,865 words / 41 entities). Both parsed clean,
      titles manifest-verified, BERT regression unchanged. `papers/manifest.csv` records the substitution.
      ⚠ **Report as "18 of the listed papers + 2 substitutes", never plain "20 papers".**
- [x] `relation` fallback extractor built and wired (`--extractor relation`)
- [x] `entity_coverage.py` built
- [x] Import check — `gold_eval`, `kg_evaluate`, `entity_coverage`, `enrich_entity_csv`, `kg_main`, `kg_extraction` all import; both CLIs render `--help` (run locally; no torch here, so extractor runtime paths are still unexercised)
- [x] **Parser fixed for this corpus** — crash, unnumbered headings, dropped abstracts, font-based titles (`hands_off.md §10`); verified against `BERT.pdf` as the ACL regression check
- [x] **Parser leftovers DONE** — 2 MDPI titles (8/8 now correct, manifest-verified); Nature heading-splitting via font evidence (`layout.py:_split_glued_headings`); roadwaylight2018 running-header junk headings (`hands_off.md` §20)
- [x] **Re-parsed all 7 with `--no-llm`** → `output/<id>/no-llm/`; body words 1,985–8,015, citation networks for 6/7 (gamlprop2025 not in S2). `output/*/fast/*` is obsolete
- [x] Stage 1–4 DONE for all 18 (parse → enrich → coverage → **extract**). Coverage: 18/18 `fixed` (`output/corpus_plan.json`). Extraction: `relation` × both ontologies, **36/36 runs, 0 failures, 2,132 triples** via Ollama/`qwen3:8b` (`hands_off.md` §23.5). ⚠ Report the CS half and the non-CS half SEPARATELY — only the CS papers get meaningful entity lists (`hands_off.md` §22.4). ⚠ The laptop cannot run torch (Smart App Control, §22.3) — local runs go through `run_local.py` (Ollama); the VM path is `bash run_relation_extraction.sh --with-fixed`.
- [~] Stage 5–7: **judged n=240 by Claude**, two disjoint stratified samples →
      `gold/sample_claude.csv` (80) + `gold/sample_round2_claude.csv` (160), reasons on every row.
      Corpus-weighted strict **CEO 34.7% / scinex 39.6%**, bootstrap difference −4.9 pts,
      95% CI [−22.5, +13.9] → **a tie** (`results.md` Tables 2–4, `gold/report_combined.json`).
      Reproduce with `python3 gold_report.py --labels gold/sample_claude.csv
      gold/sample_round2_claude.csv --bootstrap 2000`.
- [x] **κ MEASURED (2026-08-28) — the first in this project.** `gemini-3.6-flash` independently
      re-labelled 110 of the same 240 ids (verdicts blanked; `judge_paste.py batches` never emits the
      verdict column). Raw agreement 59.1%, **Cohen's κ 0.342 (3-class)**; CORRECT-vs-not 83.6%,
      **κ 0.480 (binary)**. Disagreement is almost entirely the PARTIAL boundary (Claude PARTIAL →
      Gemini INCORRECT 25×; joint INCORRECT 44; Gemini never promoted a Claude-INCORRECT to CORRECT)
      — the empirical case for reporting strict AND lenient. `gold/agreement_claude_gemini.json`.
- [~] **130 of the 240 qwen ids still need Gemini labels** — the run hit Gemini's free-tier quota
      (20 requests/day/model) at batch 11 of 24. Cleanest completion: re-judge BOTH samples with one
      fresh pinned model at `--size 25` (240 = 10 requests + Claude's 100 = 4, total 14, inside one
      model's daily 20), so a single judge covers both rows.
- [x] **Stage 8 guards (a)(b)(d) IMPLEMENTED and measured (2026-08-28)** — ontology class-name
      objects rejected (23 CEO class names derived from the schema itself); cross-reference objects
      rejected (pointer words even unnumbered, ordinary nouns only when numbered; boundary
      `(?![\w-])` not `\b`, which wrongly rejected *figure-ground segmentation*); boilerplate
      sections skipped (CRediT / funding / declarations / ethics / availability). Would remove
      **3.4%** of qwen CEO, **2.9%** of qwen scinex, **0.7%** of Claude — the same asymmetry as
      everywhere else. **Every number on disk predates them.** Item (c), a general post-parse
      domain/range type check, remains open.
- [ ] Stage 8 remainder: re-extract, re-measure. **Worklist is concrete** (`hands_off.md` §24.3, §25.2):
      (1) post-parse domain/range type check per predicate; (2) direction guards for `splitFrom`,
      `supports`, `evaluates`, `configures`; (3) reject objects matching an ontology class name
      (placeholder-leak bug — confirmed on `mentions` AND `affiliatedWith`); (3b) reject
      section/table/figure/running-header cross-references as objects; (4) skip
      Author-contributions/Declarations/Funding sections at extraction.
- [x] Enlarge gold set to 150–200 → **240 labels** (80 + 160 disjoint)
- [x] `_subject_in_sentence()` boundary bug fixed (`_contains_term()`) — parenthetical subjects like
      `Segment Anything Model (SAM2)` were silently dropped corpus-wide. Recall-only fix; **nothing on
      disk reflects it yet**
- [x] **Claude-as-extractor run COMPLETE for all 18 papers** (486 paragraphs, CEO relation mode):
      268 triples vs qwen3:8b's 1,101 → `output/claude-opus-5_relation_corpus.json`. The 11 predicates at
      0% strict hold 22.7% of qwen's corpus but only 6.0% of Claude's (`results.md` Table 7)
- [x] ⭐⭐⭐ **Claude's triples JUDGED (2026-08-28) — by `gemini-3.6-flash`, a different family.**
      **Corpus-weighted strict 89.0%** (bootstrap mean 89.1%, 95% CI [82.3, 94.5]), **lenient 97.9%**,
      n=100, verdict mix 86 CORRECT / 13 PARTIAL / 1 INCORRECT. `gold/report_claude_gemini.json`.
- [x] ⭐⭐⭐ **RQ2 ANSWERED — ONTOLOGY x CAPABILITY INTERACTION (2026-09-03).** 2x2, n=100/cell,
      one judge: **8B CEO 38.1% > scinex 12.1%; 235B CEO 48.8% < scinex 54.7%** → interaction
      **+31.9 pts**, the ranking INVERTS. Volume agrees (0.94x vs 1.48x). **The old "it's a tie" is
      WITHDRAWN** — it was measured entirely on qwen3:8b, which also explains why it flipped between
      gold rounds. Claim the sign + interaction, not the 8B magnitude. Cost $0.16.
- [x] **Cross-reference guard extended to SUBJECTS** — `(Section 3, reports, ...)` was slipping
      through an object-only check; 0.84% of gemma's output, 0 of Claude's.
- [x] ⭐⭐ **PIPELINE ENGINEERING MEASURED, +8.3 pts strict on a FIXED model.** `qwen3-8b-v2`
      (same Ollama build, same paragraphs, only the guards + bug fixes differ): **29.8% → 38.1%**
      strict, and volume UP +14.5% — not a precision/recall trade. Two additive levers: model
      choice (29.8→82.8) and engineering (29.8→38.1).
      ⛔ Cost a discarded 7-hour run first: `run_api_extract`'s Ollama adapter didn't send
      `"think": False`, so 93/515 replies were EMPTY (reasoning ate the budget). Fixed.
- [x] ⭐⭐⭐ **CAPABILITY CURVE — 4 extractors, 1 judge (2026-08-28).** gemma-3-12b 29.2% ·
      qwen3:8b 29.8% · qwen3-235b 48.8% · claude-opus-5 82.8% weighted strict, judged by
      `openai/gpt-oss-120b`. Precision and volume inversely ordered without exception.
      **The size/vendor confound is CLOSED** by qwen3:8b vs qwen3-235b (same family, scale only:
      29.8% → 48.8%). **But parameter count does not transfer across families** — gemma-3-12b is
      50% larger than qwen3:8b and no better, with the worst junk share. Say "more capable
      extractor", not "bigger model". Run via OpenRouter, $0.34 of $3.
- [x] ⭐⭐ **SECOND INDEPENDENT JUDGE on Claude's triples — the headline is cross-checked.**
      `openai/gpt-oss-120b` (OpenAI family, via Groq) re-judged the same 100: **82.8% weighted strict /
      92.6% lenient** vs gemini's 89.0% / 97.9%; **sampled 85.0% vs 86.0%**. Report the headline as a
      **range 82.8-89.0%**. Second κ: 86.0% raw, **0.457 (3-class) / 0.637 binary** — far above the
      0.342/0.480 measured on qwen's triples. **Judges agree more about good extraction than bad.**
      `gold/gptoss_verdicts.csv`, `gold/agreement_claude_twojudges.json`.
      ⚠ Groq was wrongly dismissed earlier: 8k tok/min kills extraction, not judging.
- [x] ⭐⭐⭐ **SINGLE-JUDGE EXTRACTOR COMPARISON — the project's strongest result.** The same judge
      scored qwen3:8b: **CEO 25.6% / scinex 11.7%** strict vs **claude-opus-5 89.0%**. Same paragraphs,
      same prompt, same guards, same builder, same judge — only the model differs. Every predicate at
      0% strict for qwen is ≥66% for Claude **or absent from its output** (`affiliatedWith`, `employs`,
      `configures`, `publishedIn`, `cites` = 0 occurrences). `results.md` Tables 8-9,
      `RESULTS_REPORT.md` §5.
- [x] **API judging path built** — `run_api_judge.py` drives the existing `judge_paste.py` bundles
      over HTTP (rubric as the SYSTEM PROMPT on every call, so batches cannot drift as a long chat
      does). 100 triples judged in 2.3 min instead of a day of pasting. `check_provider.py` validates
      a key and lists the models it can actually call before a run is committed.
- [x] **Corpus complete at 20 papers for BOTH models** — `relation/qwen3-8b` **1,185** triples,
      `relation_scinex/qwen3-8b` **1,117**, `relation/claude-opus-5` **296**, over 20 papers.
      The 25% volume ratio **held on the 39 paragraphs neither model had seen** (0.56 vs 2.26
      triples/paragraph) → the volume gap is a property of the models, not of the paper selection.
- [✗] ~~GPT / Gemini extraction via free chat UI~~ — **ABANDONED, do not restart.** Bundles were
      built for both (`chat_upload/gpt/`, `chat_upload/gemini/`, 49 files each) and **neither returned
      usable triples**; the reply folders came back empty. It was already weak: batching ~12 paragraphs
      per message is a *different experimental condition* from one-paragraph-per-call, and a free UI
      can silently switch model version mid-session. The API path (`run_api_extract.py`) replaces it.
- [✗] ~~Free API tiers for extraction~~ — **NOT VIABLE, measured.** A corpus run is 525 calls at
      ~6.2k tokens. **Groq = 8,000 tokens/minute** → ~1 call/min → ~8 h, slower than the laptop GPU.
      **Gemini = 20 requests/day/model** → impossible. The same quotas are perfectly adequate for
      **judging** (10-25 triples batched per call).
- [x] **`kg_builder._is_valid_entity()` FIXED** — now predicate- and role-aware: a measurement value
      (`88.5`, `0.923`, `97%`) is admitted as the OBJECT of a value predicate (`evaluates`/`achieves`/
      `reports`) only; still rejected as a subject and for every other predicate, so the filter was
      not loosened generally. All-caps acronyms exempt from the `len < 3` floor. **Measured by
      re-ingesting stored replies with no new model calls: 268 → 279 triples, `evaluates` 1 → 6.**
      ⚠ **This invalidates the `evaluates` 0% figure** — the bug deleted its correct instances and
      left only malformed ones to be judged. It must be RE-JUDGED before appearing on any
      "always wrong" list. ⚠ qwen3:8b's triples cannot be recovered this way (its `kg_main` run
      stored no raw replies) — only re-extraction recovers its equivalent losses.
- [✗] ~~Full qwen re-extraction of all 20 papers~~ — **RETIRED.** This is a `qwen3:8b` item from
      the old-rubric track. The model line-up is now fixed at qwen3-235b / gptoss-120b / gemma4-31b /
      ministral-14b, and qwen3:8b survives only as the frozen "before" half of the §26.14 engineering
      result. Superseded text follows: the 18 original papers
      predate both guard fixes, the 2 substitutes do not. Puts every number on one code version and
      is the only way to recover qwen's deleted numeric-object triples. **Largest remaining
      correctness item.**
- [✗] ~~Judge the 17 triples from the 2 substitute papers~~ — **MOOT.** `aiabstract2025` and
      `routepred2023` are retired substitutes and are excluded from the corpus everywhere
- [✗] ~~Optional size-matched control with `llama3.1:8b`~~ — **RETIRED** with the rest of the
      8B-class track. Superseded text follows: `llama3.1:8b` (pulled, ~2.7 h, no key) is a second 8B-class
      model from a different family; it would separate "model strength" from "model identity" in the
      89% vs 26% gap. Started this session and **stopped at the user's request** after 1 paper;
      partial output deleted. The one open methodological gap a reviewer could name.
- [ ] Run `fixed` for the fixed-vs-relation comparison across the CS/non-CS split
- [✗] ~~Re-run everything on the VM with Qwen3-14B~~ — **RETIRED 2026-09-07 by the user:**
      *"we are not gonna use the 14b anymore, we will continue with only the newest list of
      model i provided"*. The line-up is qwen3-235b / gptoss-120b / gemma4-31b /
      ministral-14b. Do not re-propose a 14B run.
