#!/usr/bin/env bash
# Session 40 helper: second Mistral answerer on GPU0 (port 8013) for the LAST five guards-v3 KGs, in
# reverse order, so run_g3_qa.sh (GPU1) finds them answered and finishes ~2x sooner. Answers are
# per-qid resumable, so an overlap only duplicates rows (report: last write wins).
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
S=qwen3-8b-scale-q30b-g2-noont-me30
QA=(--provider openai-compat --base-url http://127.0.0.1:8013/v1 --model mistral-small-24b --concurrency 24 --timeout 900)
screen -dmS mistralg3b bash -c "$LOGS/serve_mistral.sh 0 8013 > $LOGS/mistral_server_g3b.log 2>&1"
for _ in $(seq 120); do curl -s -m 2 http://127.0.0.1:8013/health > /dev/null && break; sleep 10; done
echo "[$(date '+%F %T')] g3 helper: Mistral up on GPU0" >> $LOGS/scale_driver.log
for kg in fixed_scinex/$S-nfull-g3 fixed/$S-nfull-g3 fixed_scinex/$S-n3000-g3 fixed/$S-n3000-g3 fixed_scinex/$S-n1500-g3; do
  "$PY" qa_eval.py answer --run scale --context kg --kg "$kg" "${QA[@]}" >> "$LOGS/g3_qa_gpu0.log" 2>&1
done
screen -S mistralg3b -X quit
echo "[$(date '+%F %T')] g3 helper: done, GPU0 server stopped" >> $LOGS/scale_driver.log
