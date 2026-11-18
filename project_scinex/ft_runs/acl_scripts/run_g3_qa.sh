#!/usr/bin/env bash
# Session 40 (2026-11-18): Q&A for the guards-v3 KGs (`<slug>-g3`, make_guard_variant.py --guards v3)
# of the scale-up teacher and the four curve students, run `scale` (10,402 questions, 1,041 test
# papers), same vLLM Mistral answerer as results.md Table 25. GPU1, DEFAULT clocks (no sudo this
# session), so a thermal watchdog stops the server if the margin to the slowdown limit drops to
# <= 3 C; answers are resumable per qid, so a re-run continues. Log: scale_driver.log, g3_qa.log.
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; LOGS=/home/dev1/scinex_runs; GPU=1; PORT=8012
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] g3: $*" | tee -a "$LOGS/scale_driver.log"; }
T=qwen3-30b-a3b-v2-g2; S=qwen3-8b-scale-q30b-g2-noont-me30
QA=(--provider openai-compat --base-url http://127.0.0.1:$PORT/v1 --model mistral-small-24b --concurrency 24 --timeout 900)
SL=("$T" "$S-n764" "$S-n1500" "$S-n3000" "$S-nfull")
KGS=(); for s in "${SL[@]}"; do KGS+=("fixed/$s-g3" "fixed_scinex/$s-g3"); done

screen -dmS mistralg3 bash -c "$LOGS/serve_mistral.sh $GPU $PORT > $LOGS/mistral_server_g3.log 2>&1"
up=0; for _ in $(seq 120); do curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null && { up=1; break; }; sleep 10; done
[ "$up" = 1 ] || { say "Mistral server did not come up"; screen -S mistralg3 -X quit; exit 1; }
say "Mistral up on GPU$GPU; answering ${#KGS[@]} guards-v3 KGs"
( while sleep 20; do
    m=$(nvidia-smi -i $GPU --query-gpu=temperature.gpu.tlimit --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "$(date '+%T') margin=$m temp=$(nvidia-smi -i $GPU --query-gpu=temperature.gpu --format=csv,noheader)" >> "$LOGS/g3_thermal.log"
    if [ -n "$m" ] && [ "$m" -le 3 ] 2>/dev/null; then
      say "THERMAL STOP: GPU$GPU margin ${m} C — server stopped; re-run this script later to resume"
      screen -S mistralg3 -X quit; break; fi
  done ) & WD=$!
for pass in 1 2; do
  "$PY" qa_eval.py answer --run scale --context kg --kg "${KGS[@]}" "${QA[@]}" >> "$LOGS/g3_qa.log" 2>&1
done
kill $WD 2>/dev/null; screen -S mistralg3 -X quit 2>/dev/null; sleep 15
say "answers done; server stopped"
C=()
for o in fixed fixed_scinex; do
  for s in "${SL[@]}"; do C+=(--compare "kg__${o}__$s-g3" "kg__${o}__$s"); done
  C+=(--compare "kg__${o}__$S-nfull-g3" "kg__${o}__$S-n764-g3" --compare "kg__${o}__$S-nfull-g3" "kg__${o}__$T-g3")
done
"$PY" qa_eval.py grade --run scale >> "$LOGS/g3_qa.log" 2>&1
"$PY" qa_eval.py report --run scale "${C[@]}" > "$LOGS/g3_report.log" 2>&1 && cp qa_eval/scale/report.json qa_eval/scale/report_g3.json
mapfile -t TP < scale/test_papers.txt; G=(); for s in "${SL[@]}"; do G+=("$s-g3"); done
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs "${G[@]}" --papers "${TP[@]}" \
  --out ft_runs/scale/compare_g3.json > "$LOGS/g3_compare.log" 2>&1
say "=== g3 done: g3_report.log, g3_compare.log ==="
