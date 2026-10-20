#!/usr/bin/env bash
# scinex extraction, THE 20-PAPER LIST ONLY (substitutes aiabstract2025/routepred2023 excluded).
# qwen3-235b already has relation_scinex on disk, so only these three are run.
set -u
PAPERS=$(cat papers_20.txt)
export PYTHONIOENCODING=utf-8
run () {
  echo "=========== $2 ==========="
  python3 run_api_extract.py --paper $PAPERS \
    --provider openai-compat --base-url https://openrouter.ai/api/v1 \
    --model "$1" --model-slug "$2" \
    --extractor relation --ontology scinex --ontology-file scinex_refined_14.owl \
    --max-tokens 3072 --resume --ingest
  echo "=========== $2 done (exit $?) ==========="
}
run openai/gpt-oss-120b        gptoss-120b
run google/gemma-4-31b-it      gemma4-31b
run mistralai/ministral-14b-2512 ministral-14b
echo "ALL SCINEX RUNS COMPLETE"
