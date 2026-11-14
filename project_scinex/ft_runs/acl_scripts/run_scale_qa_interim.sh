#!/usr/bin/env bash
# Session 40 (2026-11-14): INTERIM Q&A for the learning curve while GPU0 still trains `nfull`.
# GPU1 is idle from the end of `gen n3000` (~2026-11-15 02:00 server) until the curve driver's
# final Q&A step (~19:30). This uses it: Mistral server on GPU1 (port 8012, screen mistralsrv1),
# answers run `scale` + `acl-f0-vllm` for the n764 / n1500 / n3000 students, grades, and writes
# qa_eval/<run>/report_interim.json. `qa_eval.py answer` skips answered qids, so the curve driver's
# final pass only adds the `nfull` KG. Same server build, same greedy settings -> identical answers.
# Matches gpurestore's pattern (run_scale_*.sh), so GPU settings are restored only after this exits.
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] interim: $*" | tee -a "$LOGS/scale_driver.log"; }
T=qwen3-30b-a3b-v2-g2
S=qwen3-8b-scale-q30b-g2-noont-me30
QA=(--provider openai-compat --base-url http://127.0.0.1:8012/v1 --model mistral-small-24b --concurrency 32 --timeout 900)

say "waiting for gen n3000 (GPU1)"
until [ -f "$LOGS/scale_gen_$S-n3000.done" ]; do
  [ -f "$LOGS/scale_gen_$S-nfull.done" ] && { say "nfull already generated; the curve driver covers it — exit"; exit 0; }
  sleep 300
done
sleep 60
if [ -n "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ]; then
  say "GPU1 still busy — exit"; exit 1
fi

screen -dmS mistralsrv1 bash -c "$LOGS/serve_mistral.sh 1 8012 > $LOGS/mistral_server1.log 2>&1"
up=0; for _ in $(seq 120); do curl -s -m 2 http://127.0.0.1:8012/health > /dev/null && { up=1; break; }; sleep 10; done
[ "$up" = 1 ] || { say "Mistral server (GPU1) did not come up"; screen -S mistralsrv1 -X quit; exit 1; }
say "Mistral on GPU1 up; answering n764 / n1500 / n3000"

KGS=(); for n in 764 1500 3000; do KGS+=("fixed/$S-n$n" "fixed_scinex/$S-n$n"); done
for pass in 1 2; do   # the second pass re-asks questions a batched reply skipped
  "$PY" qa_eval.py answer --run scale --context kg --kg "${KGS[@]}" "${QA[@]}" >> "$LOGS/scale_qa_interim.log" 2>&1
  "$PY" qa_eval.py answer --run acl-f0-vllm --context kg --kg "${KGS[@]}" "${QA[@]}" >> "$LOGS/scale_qa_interim.log" 2>&1
done
screen -S mistralsrv1 -X quit 2>/dev/null; sleep 15
say "answers done; Mistral (GPU1) stopped"

C=()
for o in fixed fixed_scinex; do
  for n in 1500 3000; do C+=(--compare "kg__${o}__$S-n$n" "kg__${o}__$S-n764"); done
  C+=(--compare "kg__${o}__$S-n3000" "kg__${o}__$S-n1500")
  for n in 764 1500 3000; do C+=(--compare "kg__${o}__$S-n$n" "kg__${o}__$T"); done
done
for run in scale acl-f0-vllm; do
  "$PY" qa_eval.py grade --run $run >> "$LOGS/scale_qa_interim.log" 2>&1
  CC=(); for ((i=0; i<${#C[@]}; i+=3)); do
    [ -f "qa_eval/$run/answers/${C[i+1]}.jsonl" ] && [ -f "qa_eval/$run/answers/${C[i+2]}.jsonl" ] && CC+=("${C[@]:i:3}"); done
  "$PY" qa_eval.py report --run $run "${CC[@]}" > "$LOGS/scale_report_interim_$run.log" 2>&1 \
    && cp qa_eval/$run/report.json qa_eval/$run/report_interim.json
done
mapfile -t TP < scale/test_papers.txt
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs $T $S-n764 $S-n1500 $S-n3000 \
  --papers "${TP[@]}" --out ft_runs/scale/compare_interim.json > "$LOGS/scale_compare_interim.log" 2>&1
say "=== interim done: scale_report_interim_scale.log, scale_report_interim_acl-f0-vllm.log ==="
