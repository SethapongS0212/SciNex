#!/usr/bin/env bash
# Re-ingest the CEO runs from stored raw replies through the CURRENT guards.
# No model calls: this only re-parses responses.jsonl that is already on disk.
# Purpose: the CEO runs predate the Session 29 guards while the scinex runs have
# them active, so a CEO-vs-scinex comparison would otherwise vary code version too.
set -u
export PYTHONIOENCODING=utf-8
for m in qwen3-235b gptoss-120b gemma4-31b ministral-14b; do
  for f in output/*/kg/relation/$m/responses.jsonl; do
    [ -e "$f" ] || continue
    paper=$(echo "$f" | cut -d/ -f2)
    python3 claude_extract.py ingest --paper "$paper" --extractor relation \
      --ontology ceo --model-slug "$m" --responses "$f" 2>&1 \
      | grep -E "kept|wrote" | sed "s|^|  $m/$paper: |"
  done
done
echo "CEO RE-INGEST COMPLETE"
