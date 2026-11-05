# Entity-list cleanup — DRAFT audit (2026-11-05, Session 38)

**Status: analysis only.** No entity list or triple was changed. `entity_cleanup_audit.py` reads the
802 per-paper lists (`output/acl/<id>/Entity_<id>_enriched.csv`), the teacher's kept triples on all
papers, and the 299 blind-judged triples of `ontology_eval/_blind/acl-f0/{ceo,scinex}` (batch and
reply files only, so no model identity is used). It writes `stoplist_draft.csv` (every entry in
≥ 5% of papers, plus every case collision, with its category) and `summary.json`.

```
python entity_cleanup_audit.py --root /home/dev1/SciNex/project_scinex \
    --pools ontology_eval/_blind/acl-f0/ceo ontology_eval/_blind/acl-f0/scinex --out entity_cleanup
```

## Why
Blind judging put specificity (C3) at ~30% for every model, teacher included (`results.md`
Table 17). The cause is upstream of all models: the lists are a CS-NER gazetteer matched
case-insensitively against the paper text, so common English words become "entities". The most
frequent entries are `training` (86% of papers), `Example`, `Language`, `Information` (typed
*Model*), `Models`, `Words`, `Learning`, `Text` and `Experiments` (typed *Model*). A list averages
~100 entries per paper.

## Categories
| category | rule | entries (of 79,875) |
|---|---|---|
| GENERIC | hand list: nouns that merge unrelated things (`Models`, `Experiments`, `Text`, `natural language`, `Test Sets`) | 29,870 (37.4%) |
| NOT_A_CONCEPT | hand list: adjectives, verbs, participles (`semantic`, `supervised`, `evaluate`, `generating`) | 4,173 (5.2%) |
| CASE_COLLISION | automatic, per paper: an entry with a capital inside a word (`CompLex`, `ZeRO`, `MuTual`) whose every occurrence in that paper is plain casing (`complex`, `zero`) | 1,779 (2.2%) |
| REVIEW | hand list: task-level words a human should decide (`classifier`, `parser`, `Attention`, `embeddings`, `Language Model`) | 9,746 (12.2%) |
| KEEP | everything else, including the whole long tail below 5% of papers | 34,307 (43.0%) |

The hand lists cover only entries in ≥ 5% of papers (371 entries). Top collisions: `complex` in
288 of 292 papers, `tags` 184/187, `tag` 161/167, `map` 151/185, `character` 138/139, `zero`
128/129, `mutual` 83/84, `nice` 34/35, `tesla` 34/35. Artifacts that papers often write in
plain casing (`Adam`, `glove`, `relu`, `BERT`, ...) are allow-listed, because the first version of
the rule wrongly flagged "Adam" in 186 papers.

## What it would touch
**Teacher (gpt-oss-20b) kept triples, all papers, by subject category:**

| | KEEP | REVIEW | GENERIC | NOT_A_CONCEPT | CASE_COLLISION |
|---|---|---|---|---|---|
| CEO (8,488) | 54.7% | 16.4% | 25.6% | 2.9% | 0.4% |
| scinex (5,479) | 55.8% | 14.7% | 26.5% | 2.7% | 0.3% |

**Blind-judged triples (n = 299, base + me30 + teacher pooled), by subject category:**

| subject category | n | C3 appropriate | C3 too broad | C4 |
|---|---|---|---|---|
| GENERIC | 104 | 9.6% | 84.6% | 42.3 |
| NOT_A_CONCEPT | 12 | 0.0% | 100.0% | 33.3 |
| REVIEW | 59 | 10.2% | 83.1% | 48.3 |
| KEEP (≥ 5% of papers) | 51 | 66.7% | 27.5% | 42.2 |
| KEEP (long tail) | 72 | 58.3% | 37.5% | 46.5 |
| CASE_COLLISION | 1 | 0.0% | 100.0% | 0.0 |

**Projection on the judged sample:** C3 appropriate is 30.8% now. Dropping GENERIC +
NOT_A_CONCEPT + CASE_COLLISION would leave 182 of 299 triples at **45.1%**. Dropping REVIEW as
well would leave 123 at **61.8%**. C4 stays at 44–46 in every case.

## Reading
- **The entity lists explain the C3 ceiling.** Triples with a generic subject are judged too broad
  85% of the time; triples with a kept subject, 28–38% of the time. The REVIEW words behave like the
  generic ones (83% too broad), which argues for dropping most of them too.
- **They do not explain the C4 ceiling.** Relation correctness is flat across categories (42–48),
  so it is a property of the teacher, not of the subjects. That is next-step 6 (a stronger
  teacher), not this one.
- **The cost is volume.** About 29% of the teacher's triples have a GENERIC / NOT_A_CONCEPT /
  collision subject, and 45% once REVIEW is included. A re-run teacher would not simply lose them:
  on those paragraphs it could pick a specific listed subject instead. Only the re-run can say.
- ⚠ The projection assumes the remaining triples keep their scores, uses a single rater, and mixes
  three models. The hand lists are my judgment and need review, `REVIEW` especially.

## Next (after the presentation, handoff step 5)
1. The user (or the professor) reviews `stoplist_draft.csv`, mainly the REVIEW rows.
2. Write cleaned lists next to the originals (never overwrite `Entity_*_enriched.csv`), and give
   the fixed extractor a way to read them.
3. Re-run the teacher on the cleaned lists (~30 h at the measured rate; less with the clock locks
   if heat allows), rebuild the capped data, retrain the winning cap (`--share-system-prompt`
   should halve that), and judge blind as in Table 17.
