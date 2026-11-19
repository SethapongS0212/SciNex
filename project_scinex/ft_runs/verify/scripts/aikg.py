"""Answer-in-KG (judge-free Q&A proxy, results.md Table 27) for any slugs on run `scale`.
Share of the 10,402 questions whose gold answer is a node of the KG, with a paper-bootstrap 95% CI.
Usage (from project_scinex/): python ft_runs/verify/scripts/aikg.py <slug> [<slug> ...]"""
import sys, json, random
from pathlib import Path
sys.path.insert(0, '.')
import qa_eval as Q
qs = Q.read_jsonl(Q.run_dir('scale') / 'questions.jsonl')
papers = sorted({q['paper'] for q in qs})
out = {}
for ex in ('fixed', 'fixed_scinex'):
    for slug in sys.argv[1:]:
        spec = f'{ex}/{slug}'
        kgs = {p: Q.kg_triples(p, spec) for p in papers}
        per = {}
        for q in qs:
            per.setdefault(q['paper'], []).append(Q.answer_in_kg(q['answer'], kgs[q['paper']]))
        allx = [x for v in per.values() for x in v]
        rng = random.Random(0); ps = list(per); st = []
        for _ in range(1000):
            xs = [x for p in (rng.choice(ps) for _ in ps) for x in per[p]]
            st.append(sum(xs) / len(xs))
        st.sort()
        n_tr = sum(len(v) for v in kgs.values())
        out[spec] = {'answer_in_kg': round(100 * sum(allx) / len(allx), 1), 'ci': [round(100 * st[25], 1), round(100 * st[974], 1)], 'triples': n_tr}
        print(spec, out[spec], flush=True)
dst = Path('ft_runs/verify') / f'answer_in_kg_{sys.argv[-1]}.json'
json.dump(out, open(dst, 'w'), indent=1)
print('->', dst)
