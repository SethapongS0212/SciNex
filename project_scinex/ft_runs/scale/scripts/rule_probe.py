import csv, re, sys, glob
from collections import defaultdict
from pathlib import Path
R = Path('/home/dev1/project_scinex')
PTS = {'CORRECT': 1.0, 'PARTIAL': 0.5, 'INCORRECT': 0.0}
_types = {}
def etype(paper, subj):
    if paper not in _types:
        d = {}
        for f in glob.glob(str(R / 'output/acl_v2' / paper / '*.csv')):
            for row in csv.DictReader(open(f, encoding='utf-8-sig')):
                e = row.get('Entity', '').strip()
                if e: d[e.lower()] = row.get('NER_Type', '').strip()
                for a in (row.get('Aliases') or '').split(','):
                    if a.strip(): d.setdefault(a.strip().lower(), row.get('NER_Type', '').strip())
                ab = (row.get('Abbreviation') or '').strip()
                if ab: d.setdefault(ab.lower(), row.get('NER_Type', '').strip())
        _types[paper] = d
    return _types[paper].get(subj.lower(), '?')
STOP = set('the a an of in on for to and or with by is are was were be this that we our it as from at which can than not'.split())
CIT = re.compile(r'(proceedings|conference on|workshop on|journal|transactions|arxiv|pages?\s+\d|pp\.\s*\d|\d+\s*\(\d+(?:[-–]\d+)?\)\s*:\s*\d+|association for computational linguistics|in proc\b)', re.I)
YEAR = re.compile(r'\b(19[5-9]\d|20[0-2]\d)[a-z]?\b')
def is_citation(s):
    return bool(CIT.search(s) and YEAR.search(s)) or bool(re.match(r'^([A-Z][\w\'\-]+,\s+(?:[A-Z][\w\-]*\.?\s*)+(?:,|and)\s*)+', s))
def is_fragment(s):
    toks = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]", s)
    words = [t for t in toks if t.isalpha()]
    if len(words) < 4: return True
    stop = sum(t.lower() in STOP for t in words)
    nonword = sum(not t.isalpha() for t in toks)
    return stop < 2 or nonword / max(1, len(toks)) > 0.35
def rules(r, typ):
    p = r['predicate'].lower()
    out = []
    if p in ('addresses', 'achieves') and typ in ('Task',): out.append('task_subj_addr_ach')
    if p in ('addresses', 'achieves') and typ in ('Language',): out.append('lang_subj_addr_ach')
    if p in ('addresses', 'achieves') and typ in ('Resource',): out.append('resource_subj_addr_ach')
    if is_citation(r['source_sentence']): out.append('citation')
    if is_fragment(r['source_sentence']): out.append('fragment')
    return out
for rnd in sys.argv[1:]:
    rows = []
    for ont in ('ceo', 'scinex'):
        for f in glob.glob(str(R / f'ontology_eval/_blind/{rnd}/{ont}/by_model/*/verdicts.csv')):
            slug = Path(f).parent.name
            for r in csv.DictReader(open(f, encoding='utf-8')):
                if r['C4'] not in PTS: continue
                r['ont'], r['slug'] = ont, slug
                rows.append(r)
    print(f'=== {rnd}: {len(rows)} judged model-triples, C4 {100*sum(PTS[r["C4"]] for r in rows)/len(rows):.1f}')
    typc = defaultdict(list); hit = defaultdict(list); anyhit = []; keep = []
    for r in rows:
        t = etype(r['paper'], r['subject']); typc[(r['predicate'].lower() in ('addresses','achieves'), t)].append(PTS[r['C4']])
        rs = rules(r, t)
        for x in rs: hit[x].append(PTS[r['C4']])
        (anyhit if rs else keep).append(PTS[r['C4']])
    for k, v in sorted(typc.items(), key=lambda kv: -len(kv[1]))[:14]:
        print(f'  addr/ach={k[0]!s:5} type={k[1]:9} n={len(v):3} C4={100*sum(v)/len(v):5.1f}')
    for k, v in hit.items():
        print(f'  RULE {k:22} rejects n={len(v):3}  C4 of rejected {100*sum(v)/len(v):5.1f}  (correct {v.count(1.0)}, partial {v.count(0.5)})')
    print(f'  ALL rules: reject {len(anyhit)} ({100*len(anyhit)/len(rows):.0f}%), C4 rejected {100*sum(anyhit)/max(1,len(anyhit)):.1f}, C4 kept {100*sum(keep)/len(keep):.1f}')

print('\n=== conservative set: Resource/Language/Dataset subject on addresses/achieves + citation sentence')
for rnd in sys.argv[1:]:
    for ont in ('ceo', 'scinex', None):
        rows = []
        for o in (('ceo', 'scinex') if ont is None else (ont,)):
            for f in glob.glob(str(R / f'ontology_eval/_blind/{rnd}/{o}/by_model/*/verdicts.csv')):
                for r in csv.DictReader(open(f, encoding='utf-8')):
                    if r['C4'] in PTS: r['slug'] = Path(f).parent.name; rows.append(r)
        rej, kep = [], []
        bys = defaultdict(lambda: [[], []])
        for r in rows:
            t = etype(r['paper'], r['subject'])
            bad = (r['predicate'].lower() in ('addresses', 'achieves') and t in ('Resource', 'Language', 'Dataset')) or is_citation(r['source_sentence'])
            (rej if bad else kep).append(PTS[r['C4']])
            bys[r['slug']][1 if bad else 0].append(PTS[r['C4']])
        m = lambda v: 100 * sum(v) / max(1, len(v))
        print(f'{rnd:12} {str(ont):6} n={len(rows)} reject {len(rej)} ({100*len(rej)/len(rows):.0f}%)  C4 all {m(rej+kep):.1f} -> kept {m(kep):.1f}  (rejected {m(rej):.1f})')
        if ont is None:
            for s, (k, j) in sorted(bys.items()):
                print(f'      {s:45} keep {len(k)}/{len(k)+len(j)}  C4 {m(k+j):.1f} -> {m(k):.1f}')
