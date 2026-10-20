#!/usr/bin/env bash
set -u
export PYTHONIOENCODING=utf-8
slug="$1"; ont="$2"; dir="$3"; extra=""
[ "$ont" = "scinex" ] && extra="--ontology-file scinex_refined_14.owl"
for p in $(cat papers_20.txt); do
  f="output/$p/kg/$dir/$slug/responses.jsonl"
  [ -e "$f" ] || continue
  python3 claude_extract.py ingest --paper "$p" --extractor relation --ontology "$ont" $extra \
    --model-slug "$slug" --responses "$f" >/dev/null 2>&1
done
