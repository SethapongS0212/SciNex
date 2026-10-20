"""Print only the sampled items that have no verdict yet, for targeted judging."""
import csv, re, sys
from pathlib import Path

model, ont = sys.argv[1], sys.argv[2]
d = Path("ontology_eval") / model / ont / "triples"
done = set()
v = d / "verdicts.csv"
if v.exists():
    done = {r["triple_id"] for r in csv.DictReader(open(v, encoding="utf-8"))}
blocks = []
for f in sorted(d.glob("*.txt")):
    txt = f.read_text(encoding="utf-8")
    for m in re.finditer(r"----- triple_id: (\w+) -----\n(.*?)(?=\n----- triple_id:|\Z)",
                         txt, re.S):
        if m.group(1) not in done:
            blocks.append(f"----- triple_id: {m.group(1)} -----\n{m.group(2).rstrip()}")
print(f"### {model} / {ont} — {len(blocks)} unjudged\n")
print("\n\n".join(blocks))
