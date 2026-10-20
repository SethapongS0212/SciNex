#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_relation_extraction.sh — the 20-paper corpus, relation-only + fixed,
# both ontologies. RUN THIS ON THE VM (needs the GPU).
#
# Why this script exists: the Windows laptop cannot run the extractors at all.
# torch installs but its DLLs are blocked by a Windows Application Control
# policy (WinError 4551), and the laptop GPU is 8GB anyway — Qwen3-14B needs
# ~10GB at 4-bit. See hands_off.md §21.
#
#   bash run_relation_extraction.sh              # relation only (the ask)
#   bash run_relation_extraction.sh --with-fixed # relation + fixed, for the comparison
#
# GPU rules this script already obeys (Claude.md "GPU notes"):
#   - ONE extraction process at a time (the GPU holds one model copy; two → OOM)
#   - --max-new-tokens 4096, NOT 512. A 512 cap silently truncated 80 papers in
#     a previous run and cost a full re-extraction (hands_off §2l/§2m).
#   - Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — fatal on this vGPU.
# ---------------------------------------------------------------------------
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3-14B}"
MAXTOK="${MAXTOK:-4096}"
WITH_FIXED=0
[[ "${1:-}" == "--with-fixed" ]] && WITH_FIXED=1

# The 18 papers parsed and enriched so far. tripplanner2020 and linkpred2015 are
# still missing their PDFs; add them here once fetched, parsed and enriched.
PAPERS=(
  strabismus2026
  osmotic2026
  oxidecrack2025
  parkingyolo2023
  traveltime2022
  roadwaylight2018
  gamlprop2025
  vehiclemake2025
  csysguard2024
  reststop2018
  pesticide2025
  llamacorrupt2025
  videoseg2025
  ugmo2024
  eyelandmark2024
  textaug2023
  trafficspeed2021
  microwave2018
)

LOG="output/extraction_$(date +%Y%m%d_%H%M%S).log"
mkdir -p output
echo "model=$MODEL  max_new_tokens=$MAXTOK  papers=${#PAPERS[@]}  with_fixed=$WITH_FIXED" | tee "$LOG"

run_one () {  # $1=paper  $2=extractor  $3=ontology
  local paper="$1" ext="$2" onto="$3"
  local args=(--paper "$paper" --extractor "$ext" --model "$MODEL" --max-new-tokens "$MAXTOK")
  [[ "$onto" == "scinex" ]] && args+=(--ontology scinex)
  echo "" | tee -a "$LOG"
  echo "=== $paper | $ext | $onto | $(date +%H:%M:%S) ===" | tee -a "$LOG"
  if python3 kg_main.py "${args[@]}" >>"$LOG" 2>&1; then
    echo "    ok" | tee -a "$LOG"
  else
    # Keep going: one paper failing must not abandon the other six.
    echo "    !! FAILED (exit $?) — see $LOG" | tee -a "$LOG"
  fi
}

for p in "${PAPERS[@]}"; do
  run_one "$p" relation ceo
  run_one "$p" relation scinex
  if [[ $WITH_FIXED -eq 1 ]]; then
    run_one "$p" fixed ceo
    run_one "$p" fixed scinex
  fi
done

echo "" | tee -a "$LOG"
echo "=== triple counts ===" | tee -a "$LOG"
for p in "${PAPERS[@]}"; do
  for d in relation relation_scinex fixed fixed_scinex; do
    f=$(ls "output/$p/kg/$d"/*/triples.json 2>/dev/null | head -1)
    if [[ -n "$f" ]]; then
      n=$(python3 -c "import json,sys; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
      printf '  %-18s %-16s %s\n' "$p" "$d" "$n" | tee -a "$LOG"
    fi
  done
done

echo "" | tee -a "$LOG"
echo "Next: python3 kg_evaluate.py --all --extractor relation relation_scinex \\" | tee -a "$LOG"
echo "        --resume --summary-out output/eval/judge_corpus.json" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
