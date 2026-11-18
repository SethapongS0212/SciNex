import csv, json, glob
from pathlib import Path
R = Path('/home/dev1/project_scinex'); PTS = {'CORRECT': 1.0, 'PARTIAL': 0.5, 'INCORRECT': 0.0}
EX = {'ceo': 'fixed', 'scinex': 'fixed_scinex'}
out = {}
for ont in ('ceo', 'scinex'):
    for f in sorted(glob.glob(str(R / f'ontology_eval/_blind/scale-test/{ont}/by_model/*/verdicts.csv'))):
        slug = Path(f).parent.name; keep, drop = [], []
        cache = {}
        for r in csv.DictReader(open(f, encoding='utf-8')):
            p = r['paper']
            if p not in cache:
                tj = json.load(open(R / 'output' / p / 'kg' / EX[ont] / (slug + '-g3') / 'triples.json'))
                tl = tj['triples'] if isinstance(tj, dict) else tj
                cache[p] = {(t['subject'], t['predicate'].lower(), t['object'], t.get('source_sentence', '').strip()) for t in tl}
            key = (r['subject'], r['predicate'].lower(), r['object'], r['source_sentence'].strip())
            (keep if key in cache[p] else drop).append(PTS[r['C4']])
        m = lambda v: round(100 * sum(v) / max(1, len(v)), 1)
        strict = lambda v: round(100 * sum(x == 1.0 for x in v) / max(1, len(v)), 1)
        out[f'{ont}/{slug}'] = dict(n=len(keep) + len(drop), kept=len(keep), c4_all=m(keep + drop), c4_g3=m(keep),
                                    strict_g3=strict(keep), c4_removed=m(drop))
        print(f'{ont:6} {slug:42} kept {len(keep)}/{len(keep)+len(drop)}  C4 {m(keep+drop)} -> {m(keep)}  strict {strict(keep)}  removed C4 {m(drop)}')
json.dump(out, open(R / 'ft_runs/scale/blind_subset_g3.json', 'w'), indent=1)
