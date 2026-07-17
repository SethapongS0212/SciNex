# Results — Scientific-Paper Knowledge-Graph Pipeline

**Consolidated, paper-ready results.** Self-contained: every number, the methodology
needed to read it, and the honest caveats. Source of truth = the JSON summaries under
`output/` (paths given per table). Last updated 2026-06-24.

---

## 1. What the system does (one paragraph)

We parse scientific-paper PDFs (ACL Anthology, arXiv, etc.) into structured text, extract
a knowledge graph of (subject, predicate, object) triples with an LLM (Qwen3-14B)
constrained by an ontology, and evaluate the resulting KG by an **unsupervised
citation-prediction** task: train a knowledge-graph embedding (KGE) on KG + paper–concept
links *with citation edges held out*, then test whether a paper's learned embedding ranks
its true citation neighbours above other papers. We compare **two ontologies** (CEO and
scinex), **three KGE models** (TransE, ComplEx, RotatE), and **two training objectives**
(margin-ranking vs. self-adversarial).

---

## 2. Corpus & extraction

| | Value |
|---|---|
| Parsed papers | **155** |
| Papers with a usable citation network (eval pool) | **132** |
| KGE training graph | ~14.3k nodes, 22 (CEO) / 25 (scinex) relations, ~54k triples |
| CEO fixed-extractor triples | **6,574** |
| scinex fixed-extractor triples | **6,776** (ratio 1.03× CEO) |
| Zero-triple papers | 2 (L16-1593, W14-5502 — empty under *both* ontologies; benign) |

Entity (subject) lists are built non-circularly from the **CS-NER** human-annotated
scientific-entity gazetteer intersected with each paper's text (≈70–150 entities/paper).
Both ontologies use the same entity lists; they differ only in the predicate vocabulary
(scinex = CEO + 5 relations, refined domain/range; 47 classes / 27 object properties).

---

## 3. Evaluation protocol

- **Task:** for each query paper, rank all candidate papers by embedding similarity;
  ground truth = its real citation neighbours (held out of training).
- **Metrics:** Hits@1/5/10 and MRR (mean reciprocal rank), reported as **mean ± std over
  5 random seeds**.
- **Two scorings, always reported together:**
  - **bidirectional** — predict *any* citation neighbour.
  - **temporal-filtered (`filt`)** — rank only candidates published ≤ the query's year
    (a paper can't cite the future); the more correct "reference-prediction" setup, on a
    smaller test set.
- **Honest model selection (added 2026-06-24):** the 132 eval papers are split into a
  fixed **validation half (66)** and **test half (66)** (`--split-seed 42`, independent of
  training seed). All tunable hyperparameters (γ in particular) are chosen on validation;
  final numbers are reported on the **untouched test half**.
- **Random baseline ≈ 0.2% Hits@1** over ~500 candidate papers → results should be read as
  **lift over random**, not against 1.0 (1.0 is not the target for unsupervised citation
  prediction).

---

## 4. ⭐ Headline result (defensible — clean val/test split)

**RotatE + self-adversarial loss, γ = 28 selected on validation, reported on the disjoint
test half (66 papers, 5 seeds).** File: `output/kge_multiseed_summary_test_*`.

| Ontology | Objective | Bidir MRR | Bidir Hits@10 | Filt MRR | Filt Hits@10 |
|---|---|---|---|---|---|
| CEO | margin (baseline) | 0.406 ± 0.053 | 0.727 | 0.418 ± 0.036 | 0.759 |
| **CEO** | **self-adversarial** | **0.598 ± 0.041** | **0.861** | **0.594 ± 0.027** | 0.844 |
| scinex | margin (baseline) | 0.391 ± 0.043 | 0.712 | 0.417 ± 0.027 | 0.737 |
| **scinex** | **self-adversarial** | **0.599 ± 0.023** | 0.845 | **0.606 ± 0.054** | 0.844 |

**Self-adversarial vs. margin on the same test split: +0.19–0.21 MRR.** The win survives
clean model selection (γ chosen on validation, not test).

> **Headline sentence for the paper:** *RotatE trained with self-adversarial negative
> sampling predicts held-out citation links with MRR ≈ 0.60 and Hits@10 ≈ 0.85 (5 seeds),
> with hyperparameters selected on a disjoint validation set.*

**Validation γ-sweep** (RotatE adv, val half) that selected γ=28: γ16 0.522 · γ20 0.580 ·
γ24 0.567 · **γ28 0.628** · γ32 0.619 (curve peaks at 28).

---

## 5. Full-corpus results (all 155 papers, no val/test split)

Use these as the "evaluated on the entire corpus" figures. The §4 test-split numbers are
the *primary* result (clean selection); these are the *complete-coverage* companion.

### 5a. Margin baseline — all three models (5 seeds, `kge_multiseed_summary*.json`)

| Model | CEO bidir MRR | CEO filt MRR | CEO H@10 | scinex bidir MRR | scinex filt MRR | scinex H@10 |
|---|---|---|---|---|---|---|
| TransE | 0.315 ± 0.008 | 0.334 ± 0.006 | 0.567 | 0.325 ± 0.005 | 0.330 ± 0.006 | 0.565 |
| ComplEx | 0.432 ± 0.018 | 0.456 ± 0.027 | 0.742 | 0.441 ± 0.028 | 0.472 ± 0.041 | 0.744 |
| RotatE | 0.403 ± 0.021 | 0.423 ± 0.025 | 0.708 | 0.391 ± 0.033 | 0.407 ± 0.019 | 0.703 |

→ Under the margin objective, **ComplEx is best**; RotatE and ComplEx are close.

### 5b. RotatE + self-adversarial — all 155 papers (γ=24, 5 seeds, `..._adv_*.json`)

| Ontology | Bidir MRR | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|
| CEO | 0.566 ± 0.026 | 0.847 | 0.555 ± 0.029 | 0.855 |
| scinex | 0.568 ± 0.022 | 0.838 | 0.575 ± 0.018 | 0.844 |

→ Same ~0.57 level; consistent with the cleaner §4 test-split result (~0.60).

---

## 6. Key findings (claims the data supports)

1. **Self-adversarial negative sampling is the decisive lever for RotatE** — +0.19–0.21
   MRR over margin (test split), lifting the best result from ~0.43 to ~0.60. This is the
   model's native training objective (Sun et al. 2019); plain margin loss under-trains it.
2. **The effect is model-specific.** Self-adversarial helps RotatE; it *hurt* ComplEx and
   TransE in tuning (their best remains margin). So the reported pipeline pairs RotatE with
   self-adversarial and leaves the others on margin.
3. **The two ontologies are statistically equivalent** on citation prediction (CEO filt
   0.594 vs. scinex filt 0.606 — within one std), at near-equal triple counts. The refined
   scinex ontology **matches CEO's downstream quality without degrading it**; it does not
   *beat* it on this metric.
4. **The temporal-filtered ("reference-prediction") task scores ~equal-or-higher** than
   bidirectional and is the more correct citation setup.
5. **All results are well above the ~0.2% random baseline** — a strong proof-of-signal that
   KG content learned purely from paper text predicts real citation links.

---

## 7. Honest caveats (state these in the paper)

- **Test-split numbers are on 66 papers** (half the eval pool) — the cost of an honest
  held-out validation split. The all-papers numbers (§5) cover 132 but had no separate
  tuning set.
- **Ontology result is a tie, not a win.** If the paper's contribution is the ontology
  comparison, citation prediction alone is too blunt to distinguish them; a per-triple
  precision or ontology-coverage analysis would be needed to say more.
- **Report lift-over-random**, not absolute MRR vs. 1.0.
- **Self-adversarial hurts ComplEx/TransE** here, partly a mismatch with the cosine-similarity
  prediction path — don't over-generalise "self-adversarial is better."

---

## 8. Reproduction

```bash
# Headline: RotatE adv, val-selected γ=28, TEST split (per ontology)
python3 kge_multiseed.py --extractor fixed --model Qwen3-14B --kge rotate \
  --seeds 1 2 3 4 5 --epochs 1000 --loss adv --gamma 28 --adv-temp 1.0 \
  --eval-split test --split-seed 42 --val-frac 0.5 \
  --save-summary output/kge_multiseed_summary_test_ceo_adv.json \
  --run-dir output/kge_seedruns_test_ceo_adv
# (swap --extractor fixed_scinex + matching output paths for the scinex row;
#  swap --loss margin for the baseline rows)

# Re-select γ on validation:
#   ... --kge rotate --loss adv --gamma <G> --eval-split val   (pick best val MRR)
```

Result JSONs:
- Test split (headline): `output/kge_multiseed_summary_test_{ceo,scinex}_{margin,adv}.json`
- All papers, margin: `output/kge_multiseed_summary.json` (CEO), `..._scinex.json`
- All papers, RotatE adv: `output/kge_multiseed_summary_adv_{ceo,scinex}.json`

---

## 9. Open question — scale to more papers?

Currently 155 papers. **Recommendation: not required for the current claims.**
- More papers *lower* absolute MRR (bigger candidate pool = harder ranking) — they add
  scale/robustness, not a new finding. Going 79→155 already dropped absolute scores while
  conclusions held.
- Worth doing **only** if a reviewer would call 155 too small for the generalisation claim.
  The fetch→parse→enrich→extract→KGE chain is validated and can scale, but expect to
  re-frame around lift-over-random as the pool grows.
- The method/experiments are otherwise **complete and ready to write up**.
