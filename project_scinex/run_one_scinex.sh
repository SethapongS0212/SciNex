#!/usr/bin/env bash
# One model, scinex, over the 20-paper list. --resume makes this restart-safe.
#
# The repair step below is load-bearing: --resume rebuilds its done-set by parsing
# every line of responses.jsonl, so ONE truncated line (which is what you get if a
# run is killed mid-write) aborts the whole run with a JSONDecodeError before a
# single call is made. Dropping malformed lines first makes a restart always safe.
set -u
export PYTHONIOENCODING=utf-8
python3 - "$2" <<'PYEOF'
import json, glob, io, sys
slug = sys.argv[1]
for f in glob.glob(f'output/*/kg/relation_scinex/{slug}/responses.jsonl'):
    lines = io.open(f, encoding='utf-8').read().splitlines()
    good = []
    bad = 0
    for ln in lines:
        if not ln.strip():
            continue
        try:
            json.loads(ln)
            good.append(ln)
        except Exception:
            bad += 1
    if bad:
        io.open(f, 'w', encoding='utf-8').write('\n'.join(good) + '\n')
        print(f'  repaired {f}: dropped {bad} malformed line(s)')
PYEOF
python3 run_api_extract.py --paper $(cat papers_20.txt) \
  --provider openai-compat --base-url https://openrouter.ai/api/v1 \
  --model "$1" --model-slug "$2" \
  --extractor relation --ontology scinex --ontology-file scinex_refined_14.owl \
  --max-tokens 3072 --resume --ingest
echo "=== $2 COMPLETE (exit $?) ==="
