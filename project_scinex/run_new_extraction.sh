#!/usr/bin/env bash
# Run open -> enrich -> fixed -> pair extraction on the newly-parsed papers.
# Safe to re-run: each step is skipped if its output already exists.
# Pure GPU (no Semantic Scholar) -> no rate-limit issues. One model at a time.
set -u
cd /home/ubuntu/project_clean_9

MODEL="Qwen/Qwen3-14B"
MNT=512
LOG="logs/new_extraction_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

NEW="K16-1019 K19-1016 K19-1034 L14-1496 L14-1628 L16-1312 L16-1505 L16-1593 \
L18-1594 L18-1601 L18-1675 N13-1120 N15-1016 N19-1241 N19-5002 P17-2010 \
P17-2022 P18-1196 P18-2071 Q18-1018"

exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date) — log: $LOG ==="

# 1) OPEN extraction (feeds enrichment)
for id in $NEW; do
  if ls output/$id/kg/llm/*/triples.json >/dev/null 2>&1; then
    echo "[skip open]  $id — already has llm triples"; continue
  fi
  echo "[open]  $id  $(date +%H:%M:%S)"
  python3 kg_main.py --paper "$id" --extractor llm --model "$MODEL" --max-new-tokens $MNT
done

# 2) ENRICH entity CSVs from the open triples
for id in $NEW; do
  if [ -f output/acl/$id/Entity_${id}_enriched.csv ]; then
    echo "[skip enrich] $id — already enriched"; continue
  fi
  echo "[enrich]  $id"
  python3 enrich_entity_csv.py --paper "$id"
done

# 3) FIXED + PAIR extraction
for id in $NEW; do
  if ls output/$id/kg/fixed/*/triples.json >/dev/null 2>&1; then
    echo "[skip fixed] $id"; else
    echo "[fixed]  $id  $(date +%H:%M:%S)"
    python3 kg_main.py --paper "$id" --extractor fixed --model "$MODEL" --max-new-tokens $MNT
  fi
  if ls output/$id/kg/pair/*/triples.json >/dev/null 2>&1; then
    echo "[skip pair]  $id"; else
    echo "[pair]   $id  $(date +%H:%M:%S)"
    python3 kg_main.py --paper "$id" --extractor pair --model "$MODEL" --max-new-tokens $MNT
  fi
done

echo "=== DONE $(date) ==="
