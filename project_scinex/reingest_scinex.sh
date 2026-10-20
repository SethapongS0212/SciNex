#!/usr/bin/env bash
# Re-ingest scinex runs from stored raw replies. No model calls.
# Needed because the per-paper ingest crashed for some papers while duplicate
# runners were corrupting responses.jsonl; the responses themselves are fine.
set -u
export PYTHONIOENCODING=utf-8
slug="$1"
for p in $(cat papers_20.txt); do
  f="output/$p/kg/relation_scinex/$slug/responses.jsonl"
  [ -e "$f" ] || { echo "  $p: NO RESPONSES"; continue; }
  python3 claude_extract.py ingest --paper "$p" --extractor relation \
    --ontology scinex --ontology-file scinex_refined_14.owl \
    --model-slug "$slug" --responses "$f" 2>&1 | grep -E "kept|wrote" | sed "s|^|  $p: |"
done
