import sys, json, random
sys.path.insert(0, '.')
import qa_eval as Q
qs = Q.read_jsonl(Q.run_dir('scale') / 'questions.jsonl')
T = 'qwen3-30b-a3b-v2-g2'; S = 'qwen3-8b-scale-q30b-g2-noont-me30'
out = {}
for ex in ('fixed', 'fixed_scinex'):
    for slug in (T, f'{S}-n764', f'{S}-nfull'):
        row = {}
        for v in ('', '-g3'):
            spec = f'{ex}/{slug}{v}'
            kgs = {p: Q.kg_triples(p, spec) for p in {q['paper'] for q in qs}}
            per = {}
            for q in qs:
                per.setdefault(q['paper'], []).append(Q.answer_in_kg(q['answer'], kgs[q['paper']]))
            allx = [x for v_ in per.values() for x in v_]
            rng = random.Random(0); papers = list(per); st = []
            for _ in range(1000):
                xs = [x for p in (rng.choice(papers) for _ in papers) for x in per[p]]
                st.append(sum(xs) / len(xs))
            st.sort()
            row[v or 'base'] = (round(100 * sum(allx) / len(allx), 1), round(100 * st[25], 1), round(100 * st[974], 1))
        out[f'{ex}/{slug}'] = row
        print(ex, slug, row)
json.dump(out, open('ft_runs/scale/answer_in_kg_g3.json', 'w'), indent=1)
