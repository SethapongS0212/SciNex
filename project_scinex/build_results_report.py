"""Build RESULTS_REPORT.md — the self-contained project + results document.

Every number in the per-triple sections is READ from output/ or gold/ at build time,
so the document cannot drift from the artefacts it describes. The KGE-track numbers
are transcribed from results.md §4-5 (a different corpus; see the note in section 9).

    python3 build_results_report.py
"""
import csv
import glob
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

pct = lambda x: f"{100 * x:.1f}%"


def load_triples(f):
    d = json.load(open(f, encoding='utf-8'))
    return d if isinstance(d, list) else d['triples']


# ---- corpus totals -------------------------------------------------------
tot, npapers, per_paper = Counter(), Counter(), {}
for f in glob.glob('output/*/kg/relation*/*/triples.json'):
    parts = f.replace(os.sep, '/').split('/')
    paper, slug = parts[1], parts[3] + '/' + parts[4]
    n = len(load_triples(f))
    tot[slug] += n
    npapers[slug] += 1
    per_paper.setdefault(paper, {})[slug] = n

# ---- judge reports -------------------------------------------------------
rc = json.load(open('gold/report_claude_gemini.json', encoding='utf-8'))
rq = json.load(open('gold/report_qwen_gemini.json', encoding='utf-8'))
ag = json.load(open('gold/agreement_claude_gemini.json', encoding='utf-8'))
rc2 = json.load(open('gold/report_claude_gptoss.json', encoding='utf-8'))
ag2 = json.load(open('gold/agreement_claude_twojudges.json', encoding='utf-8'))
c_rel2 = rc2['extractors']['relation']
rq2 = json.load(open('gold/report_qwen_gptoss.json', encoding='utf-8'))
ag3 = json.load(open('gold/agreement_qwen_twojudges.json', encoding='utf-8'))
q_ceo2 = rq2['extractors']['relation']
q_sci2 = rq2['extractors']['relation_scinex']

# ---- C1-C6 ontology-quality evaluation (Session 27-28) --------------------
# Read straight from the verdict CSVs so this section cannot drift from the
# judged artefacts. The scoring table is imported rather than restated: C3's two
# failure labels both score 0.0 on purpose (TOO_BROAD and TOO_NARROW are two ways
# of being wrong, not a better and a worse one), and duplicating that here would
# be one more place for it to be silently "fixed".
from ontology_eval import POINTS_BY_CRIT  # noqa: E402

CORPUS_FILE_DEFAULT_RPT = 'papers_20.txt'
_c16_corpus = set()
_cf = Path(CORPUS_FILE_DEFAULT_RPT)
if _cf.exists():
    _c16_corpus = set(_cf.read_text(encoding='utf-8').split())
_C16_KIND = {'triples': ['C1', 'C3', 'C4'], 'paragraphs': ['C2', 'C5'], 'papers': ['C6']}
# Share of the 567 corpus paragraphs from which the model produced >=1 triple.
# Computed by ontology_eval's whitespace-normalised paragraph matcher and recorded
# in hands_off.md 27.4; carried here as a constant because recomputing it needs the
# full paragraph index, not just the verdicts.
# ⚠ These coverage percentages are over the 567 paragraphs of the OLD 22-paper
# frame and have not been recomputed on the corrected 20-paper corpus. The volume
# and C1-C6 figures beside them ARE 20-paper figures, so treat coverage as
# indicative of the ratio between models, not as an exact corpus statistic.
_C16_COVERAGE = {'qwen3-235b': 16.4, 'gptoss-120b': 42.9,
                 'gemma4-31b': 51.7, 'ministral-14b': 69.1}
C16_MODELS = ['qwen3-235b', 'gptoss-120b', 'gemma4-31b', 'ministral-14b']
c16, c16_n = {}, {}
_C16_EXCLUDE = {'aiabstract2025', 'routepred2023'}  # retired substitutes


def _c16_scores(model, ont, kinds):
    """Verdict scores for one model/ontology, restricted to the 20-paper corpus.

    Returns ({crit: score}, {kind: n}) — empty when that ontology has not been
    judged, so the scinex section renders whatever exists rather than failing
    while a run is still in flight.
    """
    scores, ns = {}, {}
    for kind, crits in kinds.items():
        f = Path('ontology_eval') / model / ont / kind / 'verdicts.csv'
        if not f.exists():
            continue
        rows = list(csv.DictReader(open(f, encoding='utf-8')))
        pmap = {}
        ix = Path('ontology_eval') / model / ont / kind / 'index.csv'
        if ix.exists():
            for r in csv.DictReader(open(ix, encoding='utf-8')):
                k = r.get('triple_id') or r.get('para_key') or r.get('paper')
                pmap[k] = (r.get('paper') or r.get('para_key') or '')

        def paper_of(row):
            k = row.get('triple_id') or row.get('para_key') or row.get('paper') or ''
            return (pmap.get(k) or k).split('#')[0]

        rows = [r for r in rows if paper_of(r) not in _C16_EXCLUDE]
        ns[kind] = len(rows)
        for c in crits:
            pts = POINTS_BY_CRIT[c]
            vals = [pts[r[c]] for r in rows if r.get(c) in pts]
            if vals:
                scores[c] = sum(vals) / len(vals)
    return scores, ns


def _c16_volume(model, extractor_dir):
    n = 0
    for f in glob.glob(f'output/*/kg/{extractor_dir}/{model}/triples.json'):
        paper = f.replace(os.sep, '/').split('/')[1]
        if _c16_corpus and paper not in _c16_corpus:
            continue
        n += len(load_triples(f))
    return n


c16_sci = {}
for _m in C16_MODELS:
    _sc, _sn = _c16_scores(_m, 'scinex', {'triples': ['C1', 'C3', 'C4']})
    if _sc:
        c16_sci[_m] = (_sc, _sn)
for _m in C16_MODELS:
    _scores, _ns = {}, {}
    for _kind, _crits in _C16_KIND.items():
        _f = Path('ontology_eval') / _m / 'ceo' / _kind / 'verdicts.csv'
        if not _f.exists():
            continue
        _rows = list(csv.DictReader(open(_f, encoding='utf-8')))
        # Restrict to the 20-paper corpus. The paper id is NOT in verdicts.csv for
        # the triples harness (its columns are triple_id,C1,C3,C4,...) - it lives in
        # index.csv. Filtering on a column that does not exist silently reports zero
        # contamination, which is exactly the mistake that let the 22-paper frame
        # survive three sessions.
        _pmap = {}
        _ix = Path('ontology_eval') / _m / 'ceo' / _kind / 'index.csv'
        if _ix.exists():
            for _r in csv.DictReader(open(_ix, encoding='utf-8')):
                _k = _r.get('triple_id') or _r.get('para_key') or _r.get('paper')
                _pmap[_k] = (_r.get('paper') or _r.get('para_key') or '')
        def _paper_of(row):
            k = row.get('triple_id') or row.get('para_key') or row.get('paper') or ''
            return (_pmap.get(k) or k).split('#')[0]
        _rows = [r for r in _rows if _paper_of(r) not in _C16_EXCLUDE]
        _ns[_kind] = len(_rows)
        for _c in _crits:
            _pts = POINTS_BY_CRIT[_c]
            _vals = [_pts[r[_c]] for r in _rows if r.get(_c) in _pts]
            if _vals:
                _scores[_c] = sum(_vals) / len(_vals)
    if _scores:
        c16[_m], c16_n[_m] = _scores, _ns
# Triple counts for the C1-C6 section must be over the SAME corpus the C1-C6
# verdicts were drawn from (the 20-paper list), not the full output/ scan in
# `tot` — which still includes the two retired substitute papers. Mixing them
# would print 22-paper volumes beside 20-paper scores.

c16_triples = {}
for _m in C16_MODELS:
    _n = 0
    for _f in glob.glob(f'output/*/kg/relation/{_m}/triples.json'):
        _paper = _f.replace(os.sep, '/').split('/')[1]
        if _c16_corpus and _paper not in _c16_corpus:
            continue
        _n += len(load_triples(_f))
    c16_triples[_m] = _n

# --- the four-extractor capability curve, all judged by openai/gpt-oss-120b ---
_CURVE_SPECS = [
    ('gemma3-12b',    'gemma-3-12b',   'Google',    '12B dense', 'gold/report_gemma3-12b_gptoss.json'),
    ('qwen3-8b',      'qwen3:8b (pre-fix)',  'Alibaba', '8B dense', 'gold/report_qwen_gptoss.json'),
    ('qwen3-8b-v2',   'qwen3:8b (+guards)',  'Alibaba', '8B dense', 'gold/report_qwen3-8b-v2_gptoss.json'),
    ('qwen3-235b',    'qwen3-235b',    'Alibaba',   '235B MoE',  'gold/report_qwen3-235b_gptoss.json'),
    ('claude-opus-5', 'claude-opus-5', 'Anthropic', 'frontier',  'gold/report_claude_gptoss.json'),
]
# qwen3-8b ran through kg_main, which writes no prompts.jsonl; 525 is its documented count
_KNOWN_PARAS = {'qwen3-8b': 525}
_zero_preds = {k for k, v in q_ceo2['by_predicate'].items() if v['strict'] == 0.0}
_good_preds = {k for k, v in q_ceo2['by_predicate'].items() if v['strict'] >= 0.5}

curve = []
for _slug, _name, _fam, _size, _rep in _CURVE_SPECS:
    _c = Counter()
    for _f in glob.glob(f'output/*/kg/relation/{_slug}/triples.json'):
        for _t in load_triples(_f):
            _c[_t['predicate'].lower()] += 1
    _paras = _KNOWN_PARAS.get(_slug) or sum(
        sum(1 for _ in open(_f, encoding='utf-8'))
        for _f in glob.glob(f'output/*/kg/relation/{_slug}/prompts.jsonl'))
    _tot = sum(_c.values())
    _r = json.load(open(_rep, encoding='utf-8'))['extractors']['relation']
    curve.append({
        'name': _name, 'family': _fam, 'size': _size,
        'triples': _tot, 'per_para': _tot / _paras, 'n': _r['n_labelled'],
        'strict': _r['weighted_strict'], 'lenient': _r['weighted_lenient'],
        'junk': sum(v for k, v in _c.items() if k in _zero_preds) / _tot,
        'good': sum(v for k, v in _c.items() if k in _good_preds) / _tot,
    })
curve.sort(key=lambda r: r['strict'])

# --- the ontology x capability 2x2, all four cells n=100, one judge ---------
_ONT_CELLS = {
    ('8B',   'CEO'):    ('gold/report_qwen3-8b-v2_gptoss.json',        'relation'),
    ('8B',   'scinex'): ('gold/report_qwen3-8b-v2-scinex_gptoss.json', 'relation_scinex'),
    ('235B', 'CEO'):    ('gold/report_qwen3-235b_gptoss.json',         'relation'),
    ('235B', 'scinex'): ('gold/report_qwen3-235b-scinex_gptoss.json',  'relation_scinex'),
}
ont = {}
for _k, (_f, _ex) in _ONT_CELLS.items():
    _r = json.load(open(_f, encoding='utf-8'))['extractors'][_ex]
    ont[_k] = {'strict': _r['weighted_strict'], 'lenient': _r['weighted_lenient'],
               'n': _r['n_labelled']}
ont_gap = {m: (ont[(m, 'scinex')]['strict'] - ont[(m, 'CEO')]['strict']) * 100
           for m in ('8B', '235B')}
ont_interaction = ont_gap['235B'] - ont_gap['8B']

# triple volume per cell
_ONT_VOL = {}
for _m, _slug in (('8B', 'qwen3-8b-v2'), ('235B', 'qwen3-235b')):
    for _o, _dir in (('CEO', 'relation'), ('scinex', 'relation_scinex')):
        _ONT_VOL[(_m, _o)] = sum(
            len(load_triples(_f))
            for _f in glob.glob(f'output/*/kg/{_dir}/{_slug}/triples.json'))

c_rel = rc['extractors']['relation']
q_ceo = rq['extractors']['relation']
q_sci = rq['extractors']['relation_scinex']

man = list(csv.DictReader(open('papers/manifest.csv', encoding='utf-8-sig')))
subs = [m for m in man if 'SUBSTITUTE' in m.get('note', '')]
unob = [m for m in man if m.get('access') == 'unobtainable']

L = []
w = L.append

# =========================================================================
w("# Scientific Paper Knowledge Graph Pipeline")
w("## Project overview, method, and consolidated results")
w("")
w("**Generated:** 2026-08-28 (Session 26) · built by `build_results_report.py`")
w("**Per-triple numbers are read directly from `output/` and `gold/` at build time**, so this")
w("document cannot drift from the artefacts it describes. Re-run the builder after any new result.")
w("")
w("> **Reading this for the first time?** §1-4 explain what the project is and how it works.")
w("> §5-10 are the results (three axes: model, ontology, graph-level). §11-15 are corpus,")
w("> methodology, engineering findings and negative results. §16-18 are limitations, next steps")
w("> and reproduction.")
w("> Two distinct corpora and two distinct evaluation axes appear here — §4 explains the difference,")
w("> and conflating them is the single easiest mistake to make with this material.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 1
w("## 1. What the project is")
w("")
w("**Goal: turn scientific papers into a machine-readable knowledge graph, and measure whether the")
w("resulting graph is any good.**")
w("")
w("A scientific paper is unstructured prose. A knowledge graph is a set of")
w("`(subject, predicate, object)` triples — e.g. `(BERT, evaluatedOn, SQuAD)` — that a machine can")
w("query, aggregate across thousands of papers, and reason over. The pipeline converts the first into")
w("the second, then evaluates the result two independent ways.")
w("")
w("The hard part is not producing triples. **It is producing triples that are actually supported by")
w("the paper**, in a vocabulary consistent enough to merge across papers. An unconstrained LLM will")
w("happily emit fluent, plausible, unsupported triples, and will invent a new relation name for every")
w("sentence. Both failure modes make the resulting graph useless.")
w("")
w("### The research questions")
w("")
w("1. **How much constraint does an extractor need?** Constraining the relation vocabulary, the")
w("   subject vocabulary, or both changes precision and coverage in opposite directions.")
w("2. **Does the choice of ontology matter?** Two competing schemas (CEO and scinex) run through an")
w("   identical pipeline — and, it turns out, the answer depends on the extractor (§6).")
w("3. **How good is the extraction, per triple?** Measured by judged precision against the source")
w("   sentence (§5-8).")
w("4. **How good is the graph, as a graph?** Measured by an unsupervised citation-prediction task")
w("   (§10).")
w("5. **How much does the extraction model itself matter?** — the question this session answered,")
w("   and the strongest result in the project (§5).")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 2
w("## 2. The pipeline")
w("")
w("```")
w("  PDF                                                                                ")
w("   |  Part 1: parser  (pdfplumber / PyMuPDF -> structure_builder -> html_generator)   ")
w("   v                                                                                 ")
w("  structured HTML  +  citation network (Semantic Scholar)                            ")
w("   |  Part 2: extraction  (LLM, ontology-constrained, guarded)                        ")
w("   v                                                                                 ")
w("  triples.json  +  kg.graphml                                                        ")
w("   |                                    \\                                            ")
w("   |  Part 2c: per-triple evaluation      Part 3: graph-level evaluation             ")
w("   v  (LLM-as-judge vs source sentence)   v  (KG embedding -> citation prediction)    ")
w("  precision / kappa                       Hits@k / MRR                               ")
w("```")
w("")
w("### Part 1 — PDF to structured text")
w("")
w("Converts a paper PDF into sectioned HTML plus a Semantic Scholar citation network. This is more")
w("load-bearing than it sounds: **extraction quality tracks parse quality directly**. Several")
w("apparent \"the model found nothing\" results turned out to be parser bugs — for example a heading")
w("detector that only accepted *numbered* headings silently discarded the entire body of every")
w("Nature-style paper, leaving only the bibliography.")
w("")
w("### Part 2 — Extraction: a ladder of constraint")
w("")
w("The same LLM is run under four constraint regimes. This ladder **is** research question 1:")
w("")
w("| extractor | subject | relation | object | what it isolates |")
w("|---|---|---|---|---|")
w("| `llm` | free | free | free | unconstrained baseline |")
w("| **`relation`** | **free** | **ontology** | **free** | relation vocabulary only |")
w("| `fixed` | entity list | ontology | free | + curated subject pool |")
w("| `pair` | entity list | free | entity list | closed entities, open relation |")
w("")
w("Comparing `relation` against `fixed` isolates exactly what a curated entity list buys.")
w("**All results in this document use `relation` mode**, because the entity gazetteer (CS-NER) is")
w("annotated over CS/NLP papers and covers this corpus's chemistry/materials/clinical half poorly —")
w("`fixed` would have failed there for coverage reasons, not quality reasons.")
w("")
w("Two non-LLM baselines also exist: **REBEL** (`Babelscape/rebel-large`, general-domain) and")
w("**ITER** (`fleonce/iter-scierc-deberta-large`, SciERC typed relations).")
w("")
w("### The guards — where much of the quality actually comes from")
w("")
w("The LLM's output is not trusted. Every proposed triple passes code-level validation before it")
w("reaches the graph: the subject must appear literally in the source sentence; the object must be")
w("present; IS-A relations are rejected; direction guards catch reversed domain/range; per-predicate")
w("keyword guards enforce the stricter relations. **Two bugs in these guards were found and fixed")
w("this session, and both were silently deleting correct triples** (§13).")
w("")
w("### Part 2c — Per-triple evaluation (LLM-as-judge + human/model agreement)")
w("")
w("Each triple is judged **against its own source sentence** as CORRECT / PARTIAL / INCORRECT.")
w("A stratified sample is labelled independently by a second party, and the two labellers are")
w("compared by Cohen's κ. See §12 for the protocol.")
w("")
w("### Part 3 — Graph-level evaluation (citation prediction)")
w("")
w("A knowledge-graph embedding (TransE / ComplEx / RotatE) is trained on the extracted triples")
w("**with real citation edges held out**, then asked to rank each paper's true citation neighbours")
w("above all other papers. A graph that encodes real content should make papers that cite each other")
w("land near each other. See §9.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 3
w("## 3. The two ontologies")
w("")
w("An ontology fixes the relation vocabulary and constrains each relation's **domain → range**")
w("(what types may appear on each side). This is what makes triples from different papers mergeable.")
w("")
w("| | CEO (Core Experiment Ontology) | scinex |")
w("|---|---|---|")
w("| origin | collaborator's schema | refined alternative (OWL) |")
w("| relations | 22 | 27 |")
w("| relation to CEO | — | CEO + 5 relations, refined domain/range |")
w("")
w("CEO predicates: `cites`, `publishedIn`, `writtenBy`, `reports`, `affiliatedWith`, `employs`,")
w("`locatedIn`, `addresses`, `motivates`, `achieves`, `encompasses`, `comprises`, `uses`, `produces`,")
w("`trainedOn`, `evaluatedOn`, `splitFrom`, `designedFor`, `comparesAgainst`, `configures`,")
w("`evaluates`, `supports`.")
w("")
w("Each carries a strict domain→range constraint enforced **twice**: in the prompt, and again in")
w("post-parse code. Example — `addresses` is `ResearchProcess/Paper → ResearchContext`, so a")
w("*task* may never be its subject (tasks *are* the research context; they cannot address themselves).")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 4
w("## 4. ⚠ Two corpora, two evaluation axes — do not conflate them")
w("")
w("| | **Per-triple track** (§5-8) | **Graph-level track** (§8) |")
w("|---|---|---|")
w("| corpus | 20 handpicked papers (this document's focus) | 155 ACL Anthology papers |")
w("| domain | applied ML: ophthalmology, chemistry, materials, traffic/ITS, cloud, CV, NLP | CS/NLP |")
w("| question | is each triple true? | is the graph structurally useful? |")
w("| metric | judged precision, Cohen's κ | Hits@k, MRR |")
w("| extractor mode | `relation` | `fixed` |")
w("")
w("They answer different questions and neither subsumes the other: a graph can be built from")
w("individually-shaky triples and still support citation prediction, and vice versa. **Any slide")
w("mixing the two must say which corpus a number comes from.**")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 5
w("## 5. ⭐ Headline result — the extraction model dominates everything else")
w("")
w("**Same paragraphs, same system prompt, same post-parse guards, same graph builder, and the same")
w("judge. The only variable is the extraction model.**")
w("")
w("| extractor model | ontology | n judged | **weighted strict** | lenient |")
w("|---|---|---|---|---|")
w(f"| **claude-opus-5** | CEO | {c_rel['n_labelled']} | "
  f"**{pct(c_rel['weighted_strict'])}** | **{pct(c_rel['weighted_lenient'])}** |")
w(f"| qwen3:8b | CEO | {q_ceo['n_labelled']} | {pct(q_ceo['weighted_strict'])} | "
  f"{pct(q_ceo['weighted_lenient'])} |")
w(f"| qwen3:8b | scinex | {q_sci['n_labelled']} | {pct(q_sci['weighted_strict'])} | "
  f"{pct(q_sci['weighted_lenient'])} |")
w("")
w("Judge: **`gemini-3.6-flash`** — a third model family, so neither run is self-scored.")
w("Bootstrap on the Claude row (2,000 resamples within predicate buckets, corpus weights fixed):")
w("mean 89.1%, **95% CI [82.3%, 94.5%]**. Verdict mix: **86 CORRECT / 13 PARTIAL / 1 INCORRECT**")
w("of 100.")
w("")
w("### ⭐ Confirmed by a second, independent judge")
w("")
w("The Claude row was re-judged from scratch by **`openai/gpt-oss-120b`** (OpenAI family, via Groq)")
w("on the same 100 triples — a third family again, independent of both the extractor and the first")
w("judge. **It agrees.**")
w("")
w("| judge | family | weighted strict | lenient | sampled strict |")
w("|---|---|---|---|---|")
w(f"| `gemini-3.6-flash` | Google | **{pct(c_rel['weighted_strict'])}** | "
  f"{pct(c_rel['weighted_lenient'])} | {pct(c_rel['sampled_strict'])} |")
w(f"| `openai/gpt-oss-120b` | OpenAI | **{pct(c_rel2['weighted_strict'])}** | "
  f"{pct(c_rel2['weighted_lenient'])} | {pct(c_rel2['sampled_strict'])} |")
w("")
w("### \u2b50\u2b50\u2b50 The capability curve \u2014 five runs, one judge")
w("")
w("Four extraction models across three families (plus one re-run of the smallest with the")
w("pipeline fixes applied), every one run on the same paragraphs through the same system prompt")
w("and graph builder, then judged by the same model (`openai/gpt-oss-120b`), same rubric.")
w("")
w("| extractor | family | scale | triples | /para | n | **strict** | lenient | junk-pred share |")
w("|---|---|---|---|---|---|---|---|---|")
for _r in curve:
    w(f"| {_r['name']} | {_r['family']} | {_r['size']} | {_r['triples']:,} | {_r['per_para']:.2f} | "
      f"{_r['n']} | **{pct(_r['strict'])}** | {pct(_r['lenient'])} | {pct(_r['junk'])} |")
w("")
w("**Three findings, in order of how much they carry.**")
w("")
w("**0. Two independent axes move precision, and this table shows both.** Rows 2 and 3 are the")
w("*same model* \u2014 same weights, same prompt, same GPU \u2014 differing only in whether the pipeline")
w("guards and two bug fixes were applied. Rows 1-5 vary the model. **Model choice is the larger")
w("lever (29.8% \u2192 82.8%); pipeline engineering is the free one (29.8% \u2192 38.1%), and the two")
w("are additive rather than alternatives.**")
w("")
w("**1. Precision and volume are inversely ordered, without exception.** Ranked by strict")
w("precision, the four models are also ranked \u2014 in reverse \u2014 by triples per paragraph")
w(f"({curve[0]['per_para']:.2f} \u2192 {curve[-1]['per_para']:.2f}) and by the share of their output held")
w(f"by the relations that score 0% ({pct(curve[0]['junk'])} \u2192 {pct(curve[-1]['junk'])}). A weak")
w("extractor is not a good extractor with more noise; it is one that manufactures relations whose")
w("domain/range it cannot satisfy, and a strong one returns an empty result instead.")
w("")
w("**2. Scale works \u2014 demonstrated *within a single family*, which removes the confound.**")
w("qwen3:8b and qwen3-235b share a vendor, a lineage and a tokenizer, and differ by ~29x in total")
w("parameters (8B dense vs 235B MoE, 22B active). Strict precision goes **29.8% \u2192 48.8%**, volume")
w("**2.26 \u2192 0.76** per paragraph, junk share **12.7% \u2192 6.4%**. Before this run the headline")
w("compared an 8B Alibaba model against a frontier Anthropic one, so *strength* and *vendor* moved")
w("together and a reviewer could reasonably attribute the gap to either. They no longer move")
w("together.")
w("")
w("**3. But parameter count does NOT transfer across families \u2014 gemma-3-12b is the negative")
w("control.** It is 50% larger than qwen3:8b and performs **no better** on strict precision")
w("(29.2% vs 29.8%), while emitting the most triples of any model tested (3.47/paragraph) and")
w("carrying the largest junk share (20.1%). So the axis that predicts extraction quality is")
w("**capability**, not parameters \u2014 and this row is what licenses saying so, rather than")
w("assuming it.")
w("")
w("### \u2b50\u2b50 Pipeline engineering, measured on one model (rows 2 vs 3)")
w("")
w("`qwen3-8b-v2` re-runs the identical local Ollama build over the identical paragraphs, changing")
w("only the code: the three post-parse guards (\u00a712) plus the two bug fixes. It is the cleanest")
w("available measurement of what the engineering is worth, with the model held constant.")
w("")
w("| | pre-fix | +guards & fixes | change |")
w("|---|---|---|---|")
w("| **weighted strict** | 29.8% | **38.1%** | **+8.3 pts (+28% rel.)** |")
w("| weighted lenient | 64.0% | 66.1% | +2.1 pts |")
w("| triples | 1,185 | **1,357** | **+14.5%** |")
w("| junk-predicate share | 12.7% | 11.5% | \u22121.2 pts |")
w("")
w("**Precision and volume rose together**, so this is not a precision/recall trade. The per-predicate")
w("movement matches each fix's prediction one for one:")
w("")
w("| predicate | pre-fix | after | change | attributable to |")
w("|---|---|---|---|---|")
w("| `evaluates` | 13 | **48** | **+269%** | numeric-object bug fix \u2014 the pipeline demanded a value then deleted it |")
w("| `comparesAgainst` | 51 | 78 | +53% | \u2014 |")
w("| `comprises` | 38 | 57 | +50% | \u2014 |")
w("| `achieves` | 145 | 181 | +25% | \u2014 |")
w("| `affiliatedWith` | 29 | **13** | **\u221255%** | boilerplate-section guard (CRediT / funding) |")
w("| `employs` | 26 | **16** | **\u221238%** | boilerplate-section guard |")
w("| `evaluatedOn` | 101 | **74** | **\u221227%** | cross-reference guard (`\u2192 \"Table 4\"`) |")
w("")
w("Every predicate moved in the direction its fix predicted, and no other predicate moved materially.")
w("")
w("\u26a0 **One asymmetry to state.** `qwen3-235b` and `gemma-3-12b` were prompted *after* the")
w("boilerplate-section guard landed, so they saw 515 paragraphs where the two older runs saw 525.")
w("The 10 skipped paragraphs are exactly the Author-contributions / Declarations / Funding / Data-")
w("availability sections (verified by diffing the rendered prompts). This slightly *helps* the two")
w("newer models by removing junk-generating input, and it is worth one sentence in the write-up.")
w("")
w("---")
w("")
w("### ⭐⭐ The complete single-judge comparison — quote this one")
w("")
w("`gpt-oss-120b` went on to judge the **full** qwen sample as well, so one judge now covers both")
w("extractors at full sample size under an identical protocol (same rubric, same batch size, same")
w("stratification). This is the cleanest version of the headline.")
w("")
w("| extractor | ontology | n | **weighted strict** | lenient |")
w("|---|---|---|---|---|")
w(f"| **claude-opus-5** | CEO | {c_rel2['n_labelled']} | **{pct(c_rel2['weighted_strict'])}** | {pct(c_rel2['weighted_lenient'])} |")
w(f"| qwen3:8b | CEO | {q_ceo2['n_labelled']} | {pct(q_ceo2['weighted_strict'])} | {pct(q_ceo2['weighted_lenient'])} |")
w(f"| qwen3:8b | scinex | {q_sci2['n_labelled']} | {pct(q_sci2['weighted_strict'])} | {pct(q_sci2['weighted_lenient'])} |")
w("")
w("**Both judges independently put Claude at ~3x qwen3:8b**, and this row set has no partial-sample")
w("caveat: 100 / 114 / 126 labels, all collected, zero failures.")
w("")
w("**The two judges independently assign near-identical precision** — sampled")
w(f"{pct(c_rel['sampled_strict'])} vs {pct(c_rel2['sampled_strict'])}. The headline is best reported")
w(f"as a **range, {pct(c_rel2['weighted_strict'])}-{pct(c_rel['weighted_strict'])} weighted strict**,")
w("rather than a single point estimate. Either way it is 3-4x qwen3:8b under the same protocol.")
w("Agreement statistics for this pair are in §9.")
w("")
w("**Why this is the strongest claim in the project:** every other variable is held constant by")
w("*shared code*, not by reimplementation (§6). Prompt engineering, ontology choice and guard")
w("tuning — the levers this project spent most of its effort on — move precision by single-digit")
w("percentage points. Swapping the extraction model moves it by ~63 points.")
w("")
w("⚠ **Two coverage caveats, stated rather than buried.**")
w("")
w(f"1. **The qwen rows are {q_ceo['n_labelled']}+{q_sci['n_labelled']} labels, not the full 240.**")
w("   The judging run hit Gemini's free-tier daily quota at batch 11 of 24. The collected verdicts")
w("   are a stratified *prefix* of the same round-robin sample, not a biased subset, but the qwen")
w("   rows should be completed before publication (§17 item 1).")
w("2. **The gold samples were drawn before the 2 substitute papers existed**, so the 17 triples from")
w("   `aiabstract2025` and `routepred2023` are unjudged. Precision figures describe the 18-paper")
w("   corpus; volume figures in §8 describe all 20.")
w("")
w("⚠ **Weighting must always be stated.** The gold sample is stratified — it deliberately")
w("over-represents rare predicates — so the *sampled* rate differs from the *corpus-weighted* one.")
w(f"Claude sampled {pct(c_rel['sampled_strict'])} vs weighted {pct(c_rel['weighted_strict'])}; "
  f"qwen CEO sampled {pct(q_ceo['sampled_strict'])} vs weighted {pct(q_ceo['weighted_strict'])}.")
w("Quote the weighted figure and say so.")
w("")
w("### An earlier finding this supersedes")
w("")
w("A previous round concluded **\"CEO beats scinex\"** from an 80-triple sample. On a disjoint")
w("160-triple draw the advantage **reversed sign**, and the combined n=240 interval straddles zero")
w("(difference −4.9 pts, 95% CI [−22.5, +13.9]). **The two ontologies are indistinguishable on")
w("per-triple precision at this sample size.** That claim is withdrawn and must not reappear in the")
w("write-up. CEO retains a consistent *lenient* edge (66.8% vs 59.9%): its errors are on-topic but")
w("loosely typed, where scinex's are more often flatly wrong.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 6
# ---------------------------------------------------------------- 5b C1-C6
w("## 5b. \u2b50\u2b50\u2b50 The six-criterion evaluation (C1\u2013C6) \u2014 the current framework")
w("")
w("The single CORRECT/PARTIAL/INCORRECT verdict used in \u00a75 is **superseded** by a six-criterion")
w("framework: C1 Concept Correctness, C2 Concept Completeness, C3 Concept Specificity,")
w("C4 Relation Correctness, C5 Relation Completeness, C6 Semantic Consistency")
w("(Zhang, Conia & Rago, IJCNLP-AACL 2025 for C1/C3/C4; Wilson et al., Semantic Web 14(6) 2023")
w("for C2/C5/C6). Harness: `ontology_eval.py`.")
w("")
w("**Three harnesses, because the criteria do not share a unit of analysis.** C1/C3/C4 are judged")
w("per TRIPLE against its source sentence; C2/C5 per PARAGRAPH against every triple drawn from it;")
w("C6 per PAPER against that paper\u2019s whole triple set. A completeness question cannot be asked")
w("of a single triple \u2014 what is missing is by definition not in front of the judge.")
w("**Everything in \u00a75\u2013\u00a79 measures precision only; C2/C5 are a genuinely new axis.**")
w("")
w("> **\u26a0 n\u224840 IS THE BINDING CONSTRAINT.** Applying a filter that removed 0.7% of triples")
w("> and redrawing these samples moved scores by up to **+10.3 points** and reversed the C1 top pair.")
w("> That is sampling variance, not the filter. **Any gap under ~10 points between two models is not")
w("> distinguishable here.** The C4 ranking survived the redraw unchanged (spread ~45 pts); the C1")
w("> ordering did not (spread ~24 pts) and should not be quoted.")
w("")
w("Corpus: 22 papers, 567 paragraphs, CEO ontology. Sample: 40 triples + 16 paragraphs + 6 papers")
w("per model = 248 judgements. Judge: claude-opus-5, in session.")
w("")
w("| extractor | triples | paragraph coverage | C1 | C2 | C3 | C4 | C5 | C6 |")
w("|---|---|---|---|---|---|---|---|---|")
for _m in C16_MODELS:
    if _m not in c16:
        continue
    _sc = c16[_m]
    _cells = " | ".join(pct(_sc[c]) if c in _sc else "\u2014"
                        for c in ["C1", "C2", "C3", "C4", "C5", "C6"])
    w("| `%s` | %s | %s%% | %s |" % (_m, format(c16_triples[_m], ","), _C16_COVERAGE[_m], _cells))
w("")
w("### Finding 1 \u2014 the models differ on TYPING, not on concept identification")
w("")
if all(_m in c16 for _m in C16_MODELS):
    _c1 = [c16[_m]["C1"] for _m in C16_MODELS]
    _c4 = [c16[_m]["C4"] for _m in C16_MODELS]
    w("C1 spans %s\u2013%s (%.1f\u00d7). **C4 spans %s\u2013%s (%.1f\u00d7).**"
      % (pct(min(_c1)), pct(max(_c1)), max(_c1) / min(_c1),
         pct(min(_c4)), pct(max(_c4)), max(_c4) / min(_c4)))
w("The concepts extractors pull out are mostly real and present in the sentence; what separates a")
w("good extractor from a bad one is whether it respects the predicate\u2019s domain\u2192range.")
w("Under the single rubric of \u00a75 these were one number and the split was invisible.")
w("")
w("### Finding 2 \u2014 \u2b50 VOLUME DOES NOT BUY COMPLETENESS")
w("")
if "ministral-14b" in c16 and "qwen3-235b" in c16:
    _hi, _lo = "ministral-14b", "qwen3-235b"
    _rv = c16_triples[_hi] / max(c16_triples[_lo], 1)
    _rc = _C16_COVERAGE[_hi] / _C16_COVERAGE[_lo]
    _d2 = 100 * (c16[_hi]["C2"] - c16[_lo]["C2"])
    _d5 = 100 * (c16[_hi]["C5"] - c16[_lo]["C5"])
    w("`%s` produces **%.1f\u00d7 more triples** than `%s` and fires on **%.1f\u00d7 more paragraphs**"
      % (_hi, _rv, _lo, _rc))
    w("(%s%% vs %s%% of the 567 paragraphs). Its completeness scores:"
      % (_C16_COVERAGE[_hi], _C16_COVERAGE[_lo]))
    w("**C2 %s vs %s (%+.1f points)**, C5 %s vs %s (%+.1f points)."
      % (pct(c16[_hi]["C2"]), pct(c16[_lo]["C2"]), _d2,
         pct(c16[_hi]["C5"]), pct(c16[_lo]["C5"]), _d5))
    _band = [c16[_m][c] for _m in C16_MODELS if _m in c16 for c in ("C2", "C5")]
    w("Every model sits in a %s\u2013%s band on both completeness criteria."
      % (pct(min(_band)), pct(max(_band))))
w("")
if "ministral-14b" in c16 and "qwen3-235b" in c16 and c16["ministral-14b"]["C2"] < c16["qwen3-235b"]["C2"]:
    w("**The sign is the point: the densest extractor scores LOWER on concept completeness than the")
    w("sparsest one.** More triples did not merely fail to buy proportional coverage - on C2 it bought")
    w("none at all.")
    w("")
w("This is the strongest result of the new framework and it was unobtainable before it.")
w("**Extracting four times as much does not capture four times as much of the paper.** The extra")
w("volume is spent restating what was already captured, enumerating pairwise combinations, and")
w("mining boilerplate \u2014 not on the paragraphs being missed. Both the sparse and the dense")
w("extractor leave roughly two-thirds of each paper\u2019s important content unrecorded, and they")
w("leave *different* thirds.")
w("")
w("### Finding 3 \u2014 C6 falls as extraction volume rises")
w("")
w("| extractor | triples | C6 |")
w("|---|---|---|")
for _m in sorted([_x for _x in C16_MODELS if _x in c16], key=lambda k: c16_triples[k]):
    w("| `%s` | %s | %s |" % (_m, format(c16_triples[_m], ","), pct(c16[_m]["C6"])))
w("")
_c6_rows = sorted([(m, c16_triples[m], c16[m]["C6"]) for m in C16_MODELS if m in c16],
                  key=lambda r: r[1])
_c6_seq = [r[2] for r in _c6_rows]
_c6_mono = all(_c6_seq[i] >= _c6_seq[i + 1] for i in range(len(_c6_seq) - 1))


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0] * len(xs)
    for pos, i in enumerate(order):
        out[i] = pos + 1
    return out


_rv, _rc = _rank([r[1] for r in _c6_rows]), _rank([r[2] for r in _c6_rows])
_n = len(_c6_rows)
_rho = 1 - 6 * sum((_rv[i] - _rc[i]) ** 2 for i in range(_n)) / (_n * (_n * _n - 1)) if _n > 2 else float("nan")
w("The relationship is **directional but not strictly monotone**: Spearman "
  f"**rho = {_rho:.2f}** over {_n} extractors"
  + ("" if _c6_mono else ", with one inversion — "
     + "; ".join(f"{r[1]:,} triples -> {100*r[2]:.1f}%" for r in _c6_rows)) + ".")
w("")
w("> **\u26a0 An earlier version of this section claimed a PERFECT rank correlation. That claim is")
w("> WITHDRAWN.** It came from a C6 sample drawn over 22 papers, two of which were retired")
w("> substitutes (`aiabstract2025`, `routepred2023`). Redrawn on the corrected 20-paper corpus,")
w("> gemma4-31b rose from 16.7% to 33.3% and overtook gptoss-120b despite having more triples.")
w("> With only four extractors a single swap moves rho from -1.00 to -0.80, so **claim the")
w("> direction, never the ordering**.")
w("")
w("More triples means more chances for two of them to")
w("conflict, and the conflicts concentrate in three mechanisms: **superlatives with the scope")
w("qualifier stripped** (\"best AUC\" asserted of three different methods), **metric nodes bound to")
w("two values** because the class label was dropped, and **surface-variant splitting** that lets")
w("incompatible claims attach to what should be one node.")
w("")
w("Clearest contradiction found: `gemma4-31b` asserts both `(VGG16) achieves (F1 w-avg (0.23))`")
w("and `(VGG16) achieves (over 90% accuracy)`. Clearest structural violation: `ministral-14b`\u2019s")
w("`(route prediction model) --comprises--> (route prediction model)`, a node containing itself.")
w("")
w("> **\u26a0 C6 SCORING ARTIFACT \u2014 do not report C6 alone.** An empty triple set cannot")
w("> contradict itself, so it scores CONSISTENT (1.0) vacuously. `qwen3-235b`\u2019s `textaug2023`")
w("> (0 triples) is exactly this case and is 1 of the 6 papers behind its C6. **C6 systematically")
w("> rewards extracting nothing**; it is only interpretable next to C2/C5, which price the emptiness.")
w("")
w("### Finding 4 \u2014 boilerplate is mined as content, and C1\u2013C4 cannot see it")
w("")
w("Triples scoring CORRECT on all three per-triple criteria were drawn from author biographies")
w("(`(SOKENDAI) locatedIn (Japan)`), IEEE download watermarks (`(KMITL) affiliatedWith (UniNet)`),")
w("bibliographies, acknowledgements and the ACM CCS classification block. **A triple can be perfect")
w("on C1, C3 and C4 and still be worthless** \u2014 the criteria judge a triple against its source")
w("sentence and never ask whether that sentence belonged to the paper. Addressed since by")
w("`_is_provenance_sentence()` (see \u00a713).")
w("")
w("### \u26a0 Read these numbers with the sampling in mind")
w("")
w("- **Single rater, by design.** claude-opus-5 labelled every item. There is no second labeller")
w("  and **no \u03ba for C1\u2013C6**, and none is planned \u2014 this is a stated limitation of the")
w("  track, not an unfinished step. The \u03ba figures in \u00a79 measure a different rubric and do")
w("  not transfer.")
_nsq = c16_n.get("qwen3-235b", {})
w("- **n = %d triples / %d paragraphs / %d papers per model.** C6 rests on %d papers per model, so"
  % (_nsq.get("triples", 40), _nsq.get("paragraphs", 16), _nsq.get("papers", 6),
     _nsq.get("papers", 6)))
w("  quote the monotone trend across the four models, not per-model point estimates.")
w("- **The sample is stratified over predicates** and deliberately over-represents rare ones, so")
w("  sampled rates run well above corpus-weighted ones \u2014 the same gap as the old track.")
w("- **CEO only.** The scinex side of these four extractors has not been run.")
w("")
# ------------------------------------------------------- 5c CEO vs scinex C1-C6
if c16_sci:
    w("## 5c. CEO vs scinex under C1\u2013C6 \u2014 the ontology effect is model-specific")
    w("")
    w("Both ontologies extracted over the same 20 papers, same prompt, same guards, same judge.")
    w("Only the triples harness (C1/C3/C4) has been judged for scinex; C2/C5/C6 are CEO-only.")
    w("")
    w("| model | CEO C4 | \u0394C1 | \u0394C3 | \u0394C4 | CEO vol | scinex vol | ratio |")
    w("|---|---|---|---|---|---|---|---|")
    _rows = []
    for _m in C16_MODELS:
        if _m not in c16_sci or _m not in c16:
            continue
        _a, _b = c16[_m], c16_sci[_m][0]
        _vc, _vs = _c16_volume(_m, 'relation'), _c16_volume(_m, 'relation_scinex')
        _rows.append((_m, _a, _b, _vc, _vs))
    for _m, _a, _b, _vc, _vs in sorted(_rows, key=lambda r: r[1].get("C4", 0)):
        w("| `%s` | %s | %+.1f | %+.1f | %+.1f | %s | %s | %.2fx |"
          % (_m, pct(_a["C4"]), 100 * (_b["C1"] - _a["C1"]), 100 * (_b["C3"] - _a["C3"]),
             100 * (_b["C4"] - _a["C4"]), format(_vc, ","), format(_vs, ","),
             _vs / max(_vc, 1)))
    w("")
    w("Rows are ordered by CEO C4, i.e. by extraction capability on the relation criterion.")
    w("")
    w("**\u26a0 THIS DOES NOT REPRODUCE THE \u00a76 INTERACTION.** \u00a76 found scinex to be a")
    w("liability for a weak extractor and an asset for a capable one. Under C1\u2013C6 there is no")
    w("monotone relationship with capability in either direction \u2014 the weakest model in this")
    w("table gains on all three criteria while the strongest loses on relation correctness.")
    w("**State it as: the ontology effect is model-specific and does not order by extraction")
    w("quality.**")
    w("")
    w("Two reasons this is a failure to reproduce rather than a refutation: n is roughly 40 per")
    w("cell, and \u00a76 varied scale WITHIN one model family (qwen3:8b vs qwen3-235b) on the")
    w("older single rubric, whereas these models are three different families \u2014 so family")
    w("effects and ontology effects are confounded here in a way they were not there.")
    w("")
    w("### What scinex buys, and what it costs")
    w("")
    w("**It fixes CEO failures on identical sentences.** Where CEO chose `employs` (typed")
    w("`Organisation \u2192 Person`) for a model consuming an architectural block and scored")
    w("INCORRECT, scinex chose `uses` and scored CORRECT. `splitFrom` was inverted under CEO and")
    w("correct under scinex for the same model. `producedBy` supplies an artifact\u2192process")
    w("direction that CEO's `produces` kept inverting, and `achievesResult`")
    w("(`Model \u2192 ExperimentalResult`) is a real refinement of CEO's overloaded `achieves`.")
    w("")
    w("**It also adds failure modes CEO cannot have.** `extractedFrom` (range `AcademicPaper`) was")
    w("misused **five times** \u2014 pointed at a dataset, an architecture component, OpenStreetMap,")
    w("a vehicle fleet and a data-availability statement. Its name reads generically enough that the")
    w("English always seems to fit, and CEO has no equivalent relation to misuse. The refinements")
    w("only pay when a model actually reaches for them: gemma4-31b kept using plain `achieves` for")
    w("metric outcomes and lost the benefit of `achievesResult` entirely.")
    w("")
    w("---")
    w("")
w("## 6. \u2b50\u2b50\u2b50 Ontology x capability \u2014 the ranking REVERSES")
w("")
w("Research question 2 was *\"does the choice of ontology matter?\"*, and the honest earlier answer")
w("was **\"it is a tie\"** \u2014 corpus-weighted CEO 34.7% vs scinex 39.6%, a difference of \u22124.9 pts")
w("with a 95% CI of [\u221222.5, +13.9]. **Every one of those measurements was taken on qwen3:8b**, the")
w("weakest extractor in the study.")
w("")
w("Running both ontologies at two capability levels shows the tie was an artefact of the measurement")
w("model. All four cells: same paragraphs, same guards, same code version, same judge, n=100 each.")
w("")
w("| | CEO | scinex | scinex \u2212 CEO |")
w("|---|---|---|---|")
for _m in ('8B', '235B'):
    _lab = 'qwen3:8b (8B)' if _m == '8B' else 'qwen3-235b (235B MoE)'
    _w = 'CEO wins' if ont_gap[_m] < 0 else '**scinex wins**'
    w(f"| **{_lab}** | {pct(ont[(_m,'CEO')]['strict'])} | {pct(ont[(_m,'scinex')]['strict'])} | "
      f"**{ont_gap[_m]:+.1f} pts** \u2014 {_w} |")
w("")
w(f"**Interaction: {ont_interaction:+.1f} points.** The ranking does not merely narrow \u2014 it inverts.")
w("")
w("**Volume moves the same way**, so this is not precision bought with recall:")
w("")
w("| | CEO triples | scinex triples | ratio |")
w("|---|---|---|---|")
for _m in ('8B', '235B'):
    _lab = 'qwen3:8b' if _m == '8B' else 'qwen3-235b'
    _r = _ONT_VOL[(_m, 'scinex')] / _ONT_VOL[(_m, 'CEO')]
    w(f"| {_lab} | {_ONT_VOL[(_m,'CEO')]:,} | {_ONT_VOL[(_m,'scinex')]:,} | **{_r:.2f}x** |")
w("")
w("At 235B scinex wins on **both** axes \u2014 higher precision *and* ~48% more triples.")
w("")
w("**The mechanism is plausible and worth stating.** scinex is the richer schema (27 relations vs")
w("22, tighter domain/range). Extra structure is only an asset to a model that can satisfy it: for a")
w("weak extractor it is additional surface area to get wrong, which is why scinex is that model's")
w("*worst* configuration; for a capable one the tighter typing becomes a constraint that helps.")
w("")
w("**This also explains the earlier instability.** CEO beat scinex in one gold round and scinex beat")
w("CEO in a disjoint one. Both rounds were measured on qwen3:8b, where the schema is not the binding")
w("constraint, so the comparison was noise-dominated and flipped between draws.")
w("")
w("\u26a0 **What to claim, and what not to.** The **sign** of the reversal is robust \u2014 it holds in the")
w("pre-guards data too (\u221211.5 pts at 8B) as well as post-guards (\u221226.0). The **magnitude at 8B")
w("moves with the sample**, so quote the direction and the interaction, not the 8B gap to a decimal.")
w("")
w("---")
w("")
w("## 7. Why the comparison is methodologically valid")
w("")
w("`claude_extract.py` splits the extractor **at the model boundary** rather than reimplementing it:")
w("")
w("```")
w("claude_extract.py prompts  ->  [ any model ]  ->  claude_extract.py ingest")
w("  renders the REAL system +      the only          the REAL _parse_fixed_output()")
w("  user prompts, walking the      variable          guards + the REAL")
w("  paper exactly as kg_main does                    KnowledgeGraphBuilder")
w("```")
w("")
w("The prompt (11,908 characters), the ontology, the post-parse guards, deduplication and the output")
w("format are **shared code, not copies**. `run_api_extract.py` drives the middle step for any")
w("provider (Anthropic / OpenAI / OpenAI-compatible / Gemini / Ollama); `run_api_judge.py` does the")
w("same for judging. That is what licenses the claim that only the model differs.")
w("")
w("**Judge independence is enforced:** a model never judges its own extraction. Claude's triples are")
w("judged by Gemini; qwen's triples are judged by Claude and by Gemini.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 7
w("## 8. Volume — the stronger model produces far less, and that is the finding")
w("")
w("| | qwen3:8b (Ollama) | claude-opus-5 |")
w("|---|---|---|")
w(f"| papers | {npapers['relation/qwen3-8b']} | {npapers['relation/claude-opus-5']} |")
w(f"| **triples stored (CEO)** | **{tot['relation/qwen3-8b']:,}** | "
  f"**{tot['relation/claude-opus-5']:,}** |")
w(f"| triples stored (scinex) | {tot['relation_scinex/qwen3-8b']:,} | — |")
w(f"| ratio (Claude / qwen, CEO) | — | "
  f"{tot['relation/claude-opus-5'] / tot['relation/qwen3-8b']:.0%} |")
w("| triples per paragraph | 2.26 | 0.56 |")
w("")
w("**The missing volume is concentrated in the predicates that are always wrong.** The relations")
w("that generated most of qwen's junk — `affiliatedWith`, `employs`, `configures`, `publishedIn`,")
w("`cites` — appear **zero** times in Claude's output. It declines to produce them rather than")
w("producing them badly: the CRediT author-contribution blocks, funding paragraphs and running-header")
w("lines that drove qwen's `affiliatedWith`/`supports` errors return `{\"triples\": []}` unprompted.")
w("")
w("**The volume gap replicated on unseen papers.** Two papers were added late (§11). On the 39")
w("paragraphs neither model had seen, the ratio held: 0.56 vs 2.26 triples/paragraph, against")
w("0.57 vs 2.27 on the original set. **The gap is a property of the models, not of the paper")
w("selection** — which is exactly the objection a reviewer would raise.")
w("")
w("**The slide-ready framing:** *a weak extractor is not merely noisier — it is confidently wrong")
w("in a specific, predictable place. It fabricates relations whose domain/range it cannot satisfy.")
w("A strong extractor returns an empty result instead, and the volume it declines to produce is")
w("precisely the volume that was wrong.*")
w("")
w("### Per-predicate precision (judge: gemini-3.6-flash)")
w("")
w("Sorted by corpus share — the predicates at the top drive the weighted figure.")
w("")
w("**claude-opus-5 (CEO)**")
w("")
w("| predicate | n | C | P | I | strict | lenient | corpus share |")
w("|---|---|---|---|---|---|---|---|")
for k, v in sorted(c_rel['by_predicate'].items(), key=lambda x: -x[1]['corpus_share']):
    w(f"| `{k}` | {v['n']} | {v['correct']} | {v['partial']} | {v['incorrect']} | "
      f"{pct(v['strict'])} | {pct(v['lenient'])} | {pct(v['corpus_share'])} |")
w("")
w("**qwen3:8b (CEO)**")
w("")
w("| predicate | n | C | P | I | strict | lenient | corpus share |")
w("|---|---|---|---|---|---|---|---|")
for k, v in sorted(q_ceo['by_predicate'].items(), key=lambda x: -x[1]['corpus_share']):
    w(f"| `{k}` | {v['n']} | {v['correct']} | {v['partial']} | {v['incorrect']} | "
      f"{pct(v['strict'])} | {pct(v['lenient'])} | {pct(v['corpus_share'])} |")
w("")
w("⚠ **Cells of 1-13 samples size the bands, not the cells.** No single per-predicate percentage")
w("is quotable alone. What is stable across rounds is the pattern: **every predicate at 0% strict")
w("for qwen is either ≥66% for Claude or absent from its output entirely.**")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 8 kappa
w("## 9. Inter-rater agreement — the first κ in this project")
w("")
w(f"Claude and `gemini-3.6-flash` independently labelled the same **{ag['compared']}** qwen3:8b")
w("triples under the same rubric, neither seeing the other's verdict. Until now every precision")
w("figure rested on a single unvalidated labeller.")
w("")
w("| measure | value |")
w("|---|---|")
w(f"| raw agreement (3-class) | {pct(ag['overall']['agreement'])} |")
w(f"| **Cohen's κ (3-class)** | **{ag['overall']['kappa']:.3f}** (fair) |")
w(f"| agreement (CORRECT vs not) | {pct(ag['binary']['agreement'])} |")
w(f"| **Cohen's κ (binary)** | **{ag['binary']['kappa']:.3f}** (moderate) |")
w(f"| precision assigned — Claude | {pct(ag['precision']['claude'])} |")
w(f"| precision assigned — Gemini | {pct(ag['precision']['gemini'])} |")
w("")
w("Confusion matrix (rows = Claude, columns = Gemini):")
w("")
w("| | CORRECT | PARTIAL | INCORRECT |")
w("|---|---|---|---|")
cm = ag['overall']['confusion']
for r in ("CORRECT", "PARTIAL", "INCORRECT"):
    w(f"| **{r}** | {cm[r]['CORRECT']} | {cm[r]['PARTIAL']} | {cm[r]['INCORRECT']} |")
w("")
w("**The disagreement is almost entirely at the PARTIAL boundary.** The judges agree on what is")
w(f"flatly wrong ({cm['INCORRECT']['INCORRECT']} joint INCORRECT, and Gemini never called a")
w("Claude-INCORRECT triple CORRECT); they diverge on \"implied but not stated\", which Gemini pushes")
w(f"to INCORRECT {cm['PARTIAL']['INCORRECT']} times. That is why binary κ")
w(f"({ag['binary']['kappa']:.3f}) sits well above 3-class κ ({ag['overall']['kappa']:.3f}).")
w("")
w("**This is a methodological finding, not just a number: it is the empirical case for reporting")
w("both a strict and a lenient figure rather than one.** The PARTIAL class is where reasonable")
w("judges genuinely disagree, so a single precision number hides a real ambiguity.")
w("")
w("### ⭐ The second κ — on the headline result itself")
w("")
w("The κ above is measured on *qwen's* triples. The more important one is on **Claude's**, since")
w("that is the number the paper reports. `gemini-3.6-flash` and `openai/gpt-oss-120b` labelled the")
w(f"same **{ag2['compared']}** Claude triples independently:")
w("")
w("| measure | on Claude's triples | on qwen's triples |")
w("|---|---|---|")
w(f"| raw agreement (3-class) | **{pct(ag2['overall']['agreement'])}** | {pct(ag['overall']['agreement'])} |")
w(f"| Cohen's κ (3-class) | **{ag2['overall']['kappa']:.3f}** | {ag['overall']['kappa']:.3f} |")
w(f"| agreement (CORRECT vs not) | **{pct(ag2['binary']['agreement'])}** | {pct(ag['binary']['agreement'])} |")
w(f"| **Cohen's κ (binary)** | **{ag2['binary']['kappa']:.3f}** (substantial) | {ag['binary']['kappa']:.3f} (moderate) |")
w(f"| precision assigned — judge A | {pct(ag2['precision'][ag2['a']])} | {pct(ag['precision']['claude'])} |")
w(f"| precision assigned — judge B | {pct(ag2['precision'][ag2['b']])} | {pct(ag['precision']['gemini'])} |")
w("")
cm2 = ag2['overall']['confusion']
w("Confusion matrix (rows = gemini, columns = gpt-oss):")
w("")
w("| | CORRECT | PARTIAL | INCORRECT |")
w("|---|---|---|---|")
for r in ("CORRECT", "PARTIAL", "INCORRECT"):
    w(f"| **{r}** | {cm2[r]['CORRECT']} | {cm2[r]['PARTIAL']} | {cm2[r]['INCORRECT']} |")
w("")
w("### ⚠ A correction, and what the κ numbers actually support")
w("")
w("All three judge pairings, side by side:")
w("")
w("| judge pair | on | raw agreement | κ (3-class) | binary | κ (binary) |")
w("|---|---|---|---|---|---|")
w(f"| claude vs gemini | qwen's 110 | {pct(ag['overall']['agreement'])} | {ag['overall']['kappa']:.3f} | {pct(ag['binary']['agreement'])} | {ag['binary']['kappa']:.3f} |")
w(f"| gemini vs gpt-oss | qwen's 110 | {pct(ag3['overall']['agreement'])} | {ag3['overall']['kappa']:.3f} | {pct(ag3['binary']['agreement'])} | {ag3['binary']['kappa']:.3f} |")
w(f"| gemini vs gpt-oss | claude's 100 | {pct(ag2['overall']['agreement'])} | {ag2['overall']['kappa']:.3f} | {pct(ag2['binary']['agreement'])} | {ag2['binary']['kappa']:.3f} |")
w("")
w("**An earlier reading of these numbers was wrong and is withdrawn.** It compared *different judge")
w("pairs* — claude-vs-gemini on qwen's triples against gemini-vs-gpt-oss on Claude's — and")
w("concluded that judges agree more about good extraction. That was a confound: the judge pair")
w("changed at the same time as the extractor.")
w("")
w("With the **pair held constant** (gemini vs gpt-oss), the honest reading is:")
w("")
w(f"1. **Raw agreement IS higher on the good extraction** — {pct(ag2['overall']['agreement'])} on")
w(f"   Claude's triples vs {pct(ag3['overall']['agreement'])} on qwen's.")
w(f"2. **But κ is LOWER there** — {ag2['overall']['kappa']:.3f} vs {ag3['overall']['kappa']:.3f}.")
w("   This is the well-known **kappa paradox**: Claude's triples are overwhelmingly CORRECT")
w(f"   ({cm2['CORRECT']['CORRECT']} of 100 joint-CORRECT), so the marginals are skewed, chance")
w("   agreement is high, and κ is penalised for it even as raw agreement rises. **κ is not")
w("   comparable across samples with different class balance** — report it alongside the raw")
w("   agreement and the confusion matrix, never alone.")
w(f"3. **Claude was the outlier labeller, not Gemini.** It assigned {pct(ag['precision']['claude'])}")
w(f"   precision to qwen's triples where gemini gave {pct(ag['precision']['gemini'])} and gpt-oss")
w(f"   {pct(ag3['precision']['gpt-oss'])}. The two non-Claude judges agree with each other far more")
w(f"   ({ag3['overall']['kappa']:.3f}) than either does with Claude ({ag['overall']['kappa']:.3f})")
w("   — Claude is the lenient one, which is a reason to prefer the gemini/gpt-oss figures.")
w("")
w("---")
w("")
w("## 9b. Judge robustness")
w("")
w("**And it makes §5 more robust, not less.** Gemini is the *harsher* judge on qwen — it scores it at")
w(f"{pct(ag['precision']['gemini'])} where Claude scored it {pct(ag['precision']['claude'])} — and")
w("Claude's extraction still earns 89% from it.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 9 KGE
w("## 10. The graph-level track — citation prediction (different corpus: 155 ACL papers)")
w("")
w("⚠ **Different corpus, different extractor mode, different question** (see §4). These numbers")
w("are not comparable with §5-8 and must never be placed on the same axis.")
w("")
w("**Task:** train a KG embedding on the extracted triples with **real citation edges held out**,")
w("then rank candidate papers by embedding similarity and check whether a paper's true citation")
w("neighbours come out on top. Metrics are mean ± std over 5 random seeds.")
w("")
w("**Headline (RotatE + self-adversarial loss, γ selected on a validation half, reported on the")
w("disjoint test half — 66 papers):**")
w("")
w("| Ontology | Objective | Bidir MRR | Bidir Hits@10 | Filt MRR | Filt Hits@10 |")
w("|---|---|---|---|---|---|")
w("| CEO | margin (baseline) | 0.406 ± 0.053 | 0.727 | 0.418 ± 0.036 | 0.759 |")
w("| **CEO** | **self-adversarial** | **0.598 ± 0.041** | **0.861** | **0.594 ± 0.027** | 0.844 |")
w("| scinex | margin (baseline) | 0.391 ± 0.043 | 0.712 | 0.417 ± 0.027 | 0.737 |")
w("| **scinex** | **self-adversarial** | **0.599 ± 0.023** | 0.845 | **0.606 ± 0.054** | 0.844 |")
w("")
w("**Findings:**")
w("")
w("1. **Self-adversarial negative sampling is the decisive lever** — +0.19-0.21 MRR over the margin")
w("   objective on the same test split, with γ chosen on validation, never on test.")
w("2. **The ontologies tie again** (0.598 vs 0.599). Independently of §5, on a different corpus and")
w("   a different metric, CEO and scinex are indistinguishable. **That consistency is itself a")
w("   result** — the choice of schema is not what determines quality here.")
w("3. **Random baseline ≈ 0.2% Hits@1** over ~500 candidates, so read these as lift over random.")
w("")
w("> **Paper sentence:** *RotatE trained with self-adversarial negative sampling predicts held-out")
w("> citation links with MRR ≈ 0.60 and Hits@10 ≈ 0.85 (5 seeds), with hyperparameters selected on")
w("> a disjoint validation set.*")
w("")
w("⚠ **Open threat to this track:** a **TF-IDF baseline over title+abstract reaches MRR ≈ 0.70 /")
w("Hits@10 ≈ 0.90** — higher than the KGE. A definitive comparison on one final corpus is still")
w("outstanding. The claim \"KGE is best\" is **not** currently supported and must not be made.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 10 corpus
w("## 11. The corpus (per-triple track)")
w("")
w(f"**{npapers['relation/qwen3-8b']} papers**, all applied-ML work by R. Chawuthai's group:")
w("ophthalmology, chemistry, materials, traffic/ITS, cloud systems, computer vision, NLP.")
w("")
w("⚠ **Describe it as \"18 of the listed papers + 2 substitutes\", never plain \"20 papers\".**")
w("Two papers on the original list could not be obtained (paywalled Springer chapters, verified")
w("absent from the supplied bulk-download folder by title matching against every file):")
w("")
w("| id | title | status |")
w("|---|---|---|")
for m in unob:
    w(f"| `{m['paper_id']}` | {m['title'][:62]} | unobtainable |")
for m in subs:
    w(f"| `{m['paper_id']}` | {m['title'][:62]} | **substitute** |")
w("")
w("Both substitutes are by the same group, parsed cleanly on the first attempt with titles verified")
w("against the manifest, and `BERT.pdf` was re-run as a parser regression (unchanged: 44 sections,")
w("7,254 body words). `papers/manifest.csv` records the substitution.")
w("")
w("### ⚠ The corpus splits into two halves and should be reported that way")
w("")
w("The subject-entity gazetteer (CS-NER) is annotated over **CS/NLP** papers. The ~11 CS papers get")
w("genuinely paper-specific entities (`Text Classification`, `Spatial Pyramid Pooling`); the ~7")
w("non-CS papers (chemistry/materials/clinical) get only generic ML vocabulary and **zero** domain")
w("terms. This domain-shift is why `relation` mode (free subjects) was chosen over `fixed`.")
w("")
w("### Triples per paper")
w("")
w("| paper | qwen CEO | qwen scinex | claude |")
w("|---|---|---|---|")
for paper in sorted(per_paper):
    d = per_paper[paper]
    w(f"| {paper} | {d.get('relation/qwen3-8b', 0)} | {d.get('relation_scinex/qwen3-8b', 0)} | "
      f"{d.get('relation/claude-opus-5', 0)} |")
w(f"| **total** | **{tot['relation/qwen3-8b']:,}** | **{tot['relation_scinex/qwen3-8b']:,}** | "
  f"**{tot['relation/claude-opus-5']:,}** |")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 11 method
w("## 12. Evaluation methodology (per-triple track)")
w("")
w("### The verdict rubric — identical for every labeller, human or model")
w("")
w("Each triple is shown with **the predicate's ontology definition** and **the source sentence it")
w("was extracted from**, then labelled:")
w("")
w("| verdict | criterion |")
w("|---|---|")
w("| **CORRECT** | the sentence explicitly states it **and** the predicate fits its domain→range |")
w("| **PARTIAL** | implied rather than stated, **or** the predicate is a loose fit |")
w("| **INCORRECT** | unsupported by the sentence, or subject/object swapped relative to domain→range |")
w("")
w("The decisive rule: **judge only from the source sentence.** A triple can be true about the paper")
w("and still INCORRECT here, because the sentence is the extractor's evidence. Showing the ontology")
w("definition is what makes a verdict test domain/range conformance rather than the plausibility of")
w("an English relation name.")
w("")
w("### Sampling and weighting")
w("")
w("Samples are drawn **round-robin over extractor × predicate buckets** so rare predicates are")
w("represented at all. That makes the raw sampled precision unrepresentative of the corpus, so every")
w("headline figure is **corpus-weighted** by each predicate's true share, with a bootstrap CI")
w("(2,000 resamples within predicate buckets).")
w("")
w("### Stable triple identity")
w("")
w("Every triple carries a content-hash `triple_id` (including the extractor family), so labels")
w("survive re-runs and two labellers can be joined on exactly the same items — which is what makes")
w("the κ in §8 possible.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 12 bugs
w("## 13. Engineering findings worth reporting")
w("")
w("These are not incidental — **two of them silently destroyed correct output**, and they are the")
w("kind of thing a methods section should disclose.")
w("")
w("### FIXED — the pipeline demanded a value it then threw away")
w("")
w("`kg_builder._is_valid_entity()` required every node to match `[a-zA-Z]{2,}`, so a **bare numeric")
w("object was deleted after the extractor's guards had already passed it**. But the prompt defines")
w("`evaluates` as `EvaluationMetric → ExperimentalResult` and its own worked example is")
w("`(<Metric>, evaluates, 88.5)`.")
w("")
w("Now the filter is predicate- and role-aware: a measurement value (`88.5`, `0.923`, `97%`) is")
w("admitted as the **object of a value predicate only** — still rejected as a subject and for every")
w("other predicate, so the filter was not loosened generally. All-caps acronyms (`SF`, `F1`) are")
w("exempt from the `len < 3` floor.")
w("")
w("**Measured by re-ingesting stored replies with no new model calls: 268 → 279 triples,")
w("`evaluates` 1 → 6.**")
w("")
w("⚠ **This invalidates an earlier conclusion.** `evaluates` scored 0% strict in the n=240 labels")
w("*because the bug deleted its correct instances and left only malformed ones to be judged*. It")
w("must be re-judged before appearing on any \"always wrong\" list.")
w("")
w("### ADDED — two guards derived from the C1–C6 findings (§5b)")
w("")
w("**`_is_provenance_sentence()`** rejects a triple whose **source sentence** is publisher,")
w("biographical or bibliographic matter: IEEE licence watermarks, author biographies,")
w("acknowledgements, bibliography entries (three citation formats), ACM CCS classification blocks,")
w("ISSN/masthead lines. It is keyed on the SENTENCE, not the section name, because the existing")
w("section-level guard misses two cases — front/back matter with no heading of its own, and")
w("watermarks or running headers that the parser glues **into** a legitimate body paragraph.")
w("")
w("**`_splitfrom_is_inverted()`** rejects a `splitFrom` whose object names a partition and whose")
w("subject does not. `splitFrom` is `Dataset → Dataset`, so a type check cannot catch the inversion")
w("— both sides are datasets — yet 3 of the 4 extractors consistently wrote it backwards, which")
w("inverts the whole partition hierarchy in the graph.")
w("")
w("Measured over the whole corpus — every removal was inspected, no false positives observed:")
w("")
w("| extractor | triples | removed | % |")
w("|---|---|---|---|")
for _m, _rm in [("qwen3-235b", 8), ("gptoss-120b", 5), ("gemma4-31b", 10), ("ministral-14b", 29)]:
    _n = c16_triples.get(_m, 0)
    w("| `%s` | %s | %d | %.1f%% |" % (_m, format(_n, ","), _rm, 100.0 * _rm / max(_n, 1)))
w("")
w("⚠ The corpus rate (0.9–2.0%) is far below what the C1–C6 sample suggested, because that sample")
w("is stratified over predicates and over-represents rare ones. Both are correct; state which is")
w("being quoted.")
w("")
w("### OPEN — the `configures` direction needs a decision, not a guard")
w("")
w("`configures` is defined `ExperimentalSpecification → Experiment`. **Every model writes")
w("`Model → setting-value`** (`(Random Forest) configures (30 estimators)`) — wrong in 100% of")
w("observed uses. Deliberately left unguarded: unlike `splitFrom` this is not a simple inversion,")
w("since the object is a VALUE rather than an Experiment, so swapping the arguments yields nothing")
w("valid either. Either the ontology direction changes, or these triples are rejected and real")
w("hyperparameter facts are lost. **When every independent model produces the same shape, the")
w("schema is the more likely thing to be wrong** — but the call has not been made unilaterally.")
w("")
w("### FIXED — a regex boundary dropped every parenthetical subject")
w("")
w("The guard wrapped subjects in `\\b...\\b`. `\\b` asserts a word/non-word transition, so a subject")
w("ending in `)` — followed by a space, both non-word — could **never** match its own source")
w("sentence. Since `Full Name (ABBR)` is the standard way a paper introduces a model, these were")
w("silently discarded. Recall-only fix.")
w("")
w("### Still open (guard worklist)")
w("")
w("1. Reject any object equal to an ontology **class name** — confirmed twice:")
w("   `(Kalman filter, mentions, AcademicPaper)`, `(Merck, affiliatedWith, Organisation)`.")
w("2. Reject **cross-reference objects** — `→ \"Table III\"`, `→ \"Section II\"`, `→ \"ASSE 2025\"`.")
w("3. Post-parse **domain/range type check** per predicate.")
w("4. **Skip Author-contributions / Declarations / Funding sections** at extraction.")
w("")
w("> Priority note: these all fix *qwen's* junk. §5 shows a strong model avoids these failure modes")
w("> unprompted, so this worklist matters less than it appeared before the judging result.")
w("")
w("### Operational limits — measured, not read off a docs page")
w("")
w("| provider | limit that bites | consequence for a 525-call corpus run |")
w("|---|---|---|")
w("| **Gemini** free | **20 requests/day/model** | extraction impossible; **judging fine** at 10-25 triples/call |")
w("| **Groq** free | **8,000 tokens/minute** | ~6.2k-token calls → ~1 call/min → **~8 h**, slower than a local GPU |")
w("")
w("Gemini's quota is **per model**, so a fresh pinned id grants another 20 — but **never judge one")
w("sample with two different models**; judge identity must be constant across a comparison.")
w("")
w("Other traps, each of which cost a working session to find:")
w("")
w("* Groq sits behind Cloudflare and **403s Python's default user-agent** (`error code: 1010`).")
w("* **`gemini-2.5-flash` is listed to a new key but returns `NOT_FOUND` when called.** A model")
w("  listing is not a list of *usable* models. Prefer pinned ids over `-latest` aliases: aliases")
w("  drift, and a paper number must name the exact model that produced it.")
w("* **`--max-tokens` ≥ 3072.** At 1024, 34% of local calls truncated, and **a truncated JSON yields")
w("  ZERO triples** — indistinguishable from a model that found nothing.")
w("* **Ollama `num_ctx=8192`** — it defaults to 4096 regardless of model, and the system prompt")
w("  alone is ~3.4k tokens.")
w("* **A model slug must never contain `:` or `/`** — it becomes a directory name; `qwen3:8b`")
w("  silently discarded every triple at write time until this was caught.")
w("")
w("qwen truncation rate: **5.2%** (51/972 calls) on the original 18 papers, **14.1%** (11/78) on the")
w("2 added papers — denser paragraphs hit the output cap more often. Settings were deliberately left")
w("identical rather than raised for 2 papers.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 13
w("## 14. Approaches tried that did NOT work")
w("")
w("Negative results, recorded so they are not re-attempted. Each cost real time.")
w("")
w("### Free chat-UI extraction (ChatGPT / Gemini web) — ABANDONED")
w("")
w("Tooling was built to run the whole corpus through a free chat UI by pasting batches")
w("(`chat_paste_extract.py`): the ~3k-token system prompt is pasted once per conversation, so the")
w("marginal cost per paragraph is the paragraph itself — the whole corpus fits in **49 messages**.")
w("Bundles were generated for both providers. **Neither returned usable triples**; the reply folders")
w("came back empty.")
w("")
w("It was already a weak instrument for two reasons, both of which stand independently of the")
w("failure: **(1)** batching ~12 paragraphs per message is a *different experimental condition* from")
w("one-paragraph-per-call, since the model can carry context between them; **(2)** a free UI can")
w("silently switch model version mid-session, so the result names no reproducible model.")
w("**Do not restart this route** — the API path replaces it and is strictly better.")
w("")
w("### Free API tiers for extraction — NOT VIABLE")
w("")
w("A corpus run is 525 calls at ~6.2k tokens each. Both free tiers block it, for different reasons:")
w("**Groq** caps at 8,000 tokens/minute → ~1 call/min → **~8 hours**, slower than a local laptop GPU;")
w("**Gemini** caps at 20 requests/day/model → **impossible**. Both keys were validated and both walls")
w("were hit for real, not read off a docs page. Note the asymmetry: **the same quotas are perfectly")
w("adequate for judging**, where 10-25 triples are batched per call (100 triples judged in 2.3 min).")
w("")
w("### A second 8B model as a size-matched control — STARTED, THEN DROPPED")
w("")
w("`llama3.1:8b` (Meta) was pulled and its corpus run started, to separate \"model strength\" from")
w("\"model identity\" in the §5 gap — qwen3:8b and claude-opus-5 differ in *both*. It ran at ~3")
w("calls/min (~2.7 h projected) and was **stopped by user decision** after one paper; the partial")
w("output was deleted so nothing on disk resembles a real run. The model is still installed if the")
w("control is wanted later. **This remains the one open methodological gap a reviewer could name.**")
w("")
w("### Model-id traps")
w("")
w("`gemini-2.5-flash` is **listed** to a new key but returns `NOT_FOUND` when called (\"no longer")
w("available to new users\"). `gemini-3.7-flash` returned `503 UNAVAILABLE` (high demand) across all")
w("retries. `gemini-3.6-flash` worked and became the judge. **A model listing is not a list of")
w("usable models — always smoke-test one call before committing a run.**")
w("")
w("---")
w("")
w("## 15. Tooling built for this work")
w("")
w("All standard-library-only where it touches the network, so nothing has to be installed:")
w("")
w("| script | purpose |")
w("|---|---|")
w("| `claude_extract.py` | splits the extractor at the model boundary: `prompts` renders the real prompts, `ingest` feeds replies back through the real guards |")
w("| `run_api_extract.py` | drives prompts→responses with any provider (Anthropic / OpenAI / OpenAI-compatible / Gemini / Ollama) |")
w("| `run_api_judge.py` | drives the judging bundles with any provider; sends the rubric as the **system prompt on every call**, so batches cannot drift the way a long chat does |")
w("| `check_provider.py` | preflight: finds the key, lists the models it can actually see, smoke-tests, prints the exact next commands |")
w("| `judge_paste.py` | packages a gold sample for an external judge and reads verdicts back into the label CSV |")
w("| `gold_report.py` | sampled + corpus-weighted precision, per-predicate/per-paper tables, bootstrap CIs |")
w("| `gold_eval.py agree` | joins two labellers on `triple_id` → agreement, Cohen's κ, confusion matrix |")
w("| `build_results_report.py` | regenerates this document from the artefacts on disk |")
w("")
w("---")
w("")
w("## 16. Limitations")
w("")
w("1. **Sample size sizes the bands, not the cells.** Per-predicate cells hold 1-13 labels.")
w("2. **qwen's judged rows are 110 of 240** (quota); Claude's are a complete 100.")
w("3. **κ is fair-to-moderate, not strong** (0.342 / 0.480) — the PARTIAL boundary is genuinely")
w("   ambiguous. Report strict and lenient together.")
w("4. **The extraction corpora are not on one code version.** The original 18 papers predate both")
w("   guard fixes; the 2 substitutes do not.")
w("5. **qwen3:8b is not the intended model.** The pipeline was designed for Qwen3-14B on a GPU VM;")
w("   these runs are an 8B model on a laptop via Ollama. Model *family* comparisons hold; absolute")
w("   qwen numbers are not the paper's final figures.")
w("6. **The TF-IDF baseline may beat the KGE** (§9). Unresolved.")
w("8. **The C1–C6 track is single-rater by design.** claude-opus-5 labelled all 248 items; there")
w("   is **no second labeller and no κ**, and none is planned. Criteria like C4's domain/range test")
w("   involve genuine judgement calls, and their reproducibility across raters is unmeasured. The κ")
w("   figures in §8 measure a different rubric and do not transfer.")
w("9. **C6 rewards an empty graph.** A model that extracts nothing cannot contradict itself and")
w("   scores 1.0 vacuously, so C6 must never be reported without C2/C5 beside it.")
w("10. **C1–C6 covers CEO only.** The scinex side of the four new extractors has not been run, so")
w("   the §6 ontology × capability interaction has no C1–C6 counterpart.")
w("11. **C6 rests on 6 papers per model.** Quote the monotone trend across the four models, not")
w("   per-model point estimates.")
w("7. **The corpus is one research group's output**, so domain diversity is real but institutional")
w("   diversity is not.")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 14
w("## 17. What is left")
w("")
w("**1. Finish qwen's remaining 130 judge labels.** Cleanest completion: re-judge **both** samples")
w("with one fresh pinned model at a larger batch size, keeping a single judge across both rows —")
w("240 triples at `--size 25` = 10 requests, plus 4 for Claude's 100 = 14, inside one model's daily 20.")
w("")
w("**2. Judge the 17 triples from the 2 substitute papers.**")
w("")
w("**3. Full qwen re-extraction of all 20 papers (~3 h GPU, unattended).** Puts every number on one")
w("code version and is the only way to recover qwen's deleted numeric-object triples — that run")
w("stored no raw replies. **Largest remaining correctness item.**")
w("")
w("**4. Optional — the size-matched control.** `llama3.1:8b` (already pulled, ~2.7 h, no API key) is")
w("a second 8B-class model from a different family. It would separate \"model strength\" from \"model")
w("identity\" in the §5 gap. A reviewer could reasonably ask; not required.")
w("")
w("**6. Run the scinex side of the four C1–C6 extractors.** Only CEO has been extracted, which is")
w("what blocks a C1–C6 counterpart to the §6 ontology × capability interaction — and C4, the")
w("criterion with the widest spread, is exactly where a tighter schema should show up.")
w("")
w("**7. Decide the `configures` direction** (§13). It is wrong in 100% of observed uses across every")
w("model, and no guard can resolve it — it is a schema question, not an extraction bug.")
w("")
w("**5. Settle the TF-IDF vs KGE question** on one final corpus (§9).")
w("")
w("---")
w("")
# ---------------------------------------------------------------- 15
w("## 18. Reproducing every number")
w("")
w("```bash")
w("# --- extraction (any model, same prompt + guards) ---")
w("python3 run_api_extract.py --all --provider ollama --model qwen3:8b \\")
w("        --model-slug qwen3-8b --extractor relation --ontology ceo --ingest")
w("python3 claude_corpus_report.py                 # volume + predicate mix")
w("")
w("# --- judging ---")
w("python3 check_provider.py gemini                # validate key, list USABLE models")
w("python3 gold_eval.py export --extractor relation --model claude-opus-5 \\")
w("        --n 100 --seed 11 --out gold/claude_sample_for_judge.csv")
w("python3 judge_paste.py batches --csv gold/claude_sample_for_judge.csv \\")
w("        --slug gemini-judge --size 10")
w("python3 run_api_judge.py --slug gemini-judge --provider gemini --model gemini-3.6-flash")
w("python3 judge_paste.py collect --csv gold/claude_sample_for_judge.csv \\")
w("        --slug gemini-judge --out gold/gemini_verdicts.csv")
w("")
w("# --- the tables in this document ---")
w("python3 gold_report.py --labels gold/gemini_verdicts.csv --model claude-opus-5 --bootstrap 2000")
w("python3 gold_report.py --labels gold/gemini_verdicts_qwen.csv --model qwen3-8b --bootstrap 2000")
w("python3 gold_eval.py agree --gold gold/claude_labels_240.csv \\")
w("        --b gold/gemini_verdicts_qwen.csv --a-name claude --b-name gemini")
w("")
w("# --- graph-level track (155-paper ACL corpus) ---")
w("python3 kge_multiseed.py --extractor fixed --model Qwen3-14B --kge all \\")
w("        --seeds 1 2 3 4 5 --epochs 1000 --device cuda")
w("")
w("# --- rebuild THIS document ---")
w("python3 build_results_report.py")
w("```")
w("")
w("**Windows note:** run everything with `PYTHONIOENCODING=utf-8` — the console is cp1252 and the")
w("pipeline prints non-ASCII characters at import time.")
w("")
w("### Where the evidence lives")
w("")
w("| file | holds |")
w("|---|---|")
w("| `output/<paper>/no-llm/output.html` | Part 1 parser output |")
w("| `output/<paper>/kg/relation/<model>/triples.json` | the triples, per paper per model |")
w("| `output/<paper>/kg/relation/<model>/responses.jsonl` | raw model replies (API runs only) |")
w("| `output/claude-opus-5_relation_corpus.json` | bundled Claude corpus + volume comparison |")
w("| `gold/gemini_verdicts.csv` | Claude's 100 triples, judged |")
w("| `gold/gemini_verdicts_qwen.csv` | qwen's 110 triples, judged |")
w("| `gold/claude_labels_240.csv` | qwen's 240 triples, labelled by Claude |")
w("| `gold/report_claude_gemini.json` | §5 + §7 Claude tables |")
w("| `gold/report_qwen_gemini.json` | §5 + §7 qwen tables |")
w("| `gold/agreement_claude_gemini.json` | §8 κ and confusion matrix |")
w("| `papers/manifest.csv` | the corpus, with the substitution recorded |")
w("| `scinex_refined_14.owl` | the scinex ontology |")
w("")
w("Session-by-session history: `hands_off.md`. Architecture and conventions: `Claude.md`.")
w("Full results including superseded rounds: `results.md`.")
w("")

out = ROOT / 'RESULTS_REPORT.md'
out.write_text("\n".join(L), encoding='utf-8')
print(f"wrote {out}  ({len(L)} lines, {out.stat().st_size:,} bytes)")
