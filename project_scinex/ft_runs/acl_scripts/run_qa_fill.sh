#!/usr/bin/env bash
# Session 39 (2026-11-07): fill the 67 Q&A answers the answer model skipped in its replies
# (closedbook 30, oracle 2, CEO ft-me30 9, scinex teacher 18, scinex llama/phi4 noont 4 each),
# then grade + report with the Table 20 comparisons (-> qa_eval/acl-f0/report_table20.json).
# Waits until the teacher-v2 lanes of run_after_qa.sh are done (GPU1 is then free while the
# chain answers the v2 KGs on GPU0), and grades only after the chain has ended (one writer of
# grades.jsonl at a time). Resumes if re-run. Log: /home/dev1/scinex_runs/qa_fill.log
set -u
cd /home/dev1/project_scinex || exit 1
LOGS=/home/dev1/scinex_runs
PY=/home/dev1/envsam/bin/python
say() { echo "[$(date '+%F %T')] fill: $*" | tee -a "$LOGS/qa_fill.log" >> "$LOGS/qa_driver.log"; }
say "waiting for the teacher v2 lanes"
until grep -q "teacher v2 lanes:" "$LOGS/after_qa_driver.log"; do sleep 120; done
HF_BATCH=8 "$LOGS/run_qa.sh" 1 controls
HF_BATCH=8 "$LOGS/run_qa.sh" 1 kg fixed/qwen3-8b-acl-ft-me30-cv fixed_scinex/gptoss-20b \
  fixed_scinex/llama31-8b-acl-noont-me50-cv fixed_scinex/phi4-acl-noont-me50-cv
say "answers filled; waiting for the chain to end before grading"
while screen -ls | grep -qE '\.chain[[:space:]]'; do sleep 60; done
"$PY" qa_eval.py grade --run acl-f0 >> "$LOGS/qa_fill.log" 2>&1
C=()
for o in fixed fixed_scinex; do
  for pair in "noont-me50-cv ft-me50-cv" "noont-me30-cv ft-me30-cv" "noont-me50-cv base-noont" \
              "noont-me30-cv base-noont" "noont-me30-cv noont-me50-cv"; do
    set -- $pair; C+=(--compare "kg__${o}__qwen3-8b-acl-$1" "kg__${o}__qwen3-8b-acl-$2"); done
  for s in qwen3-8b-acl-noont-me50-cv qwen3-8b-acl-noont-me30-cv; do
    C+=(--compare "kg__${o}__$s" "kg__${o}__gptoss-20b"); done
  [ -f qa_eval/acl-f0/answers/kg__${o}__gptoss-20b-v2.jsonl ] && \
    C+=(--compare "kg__${o}__gptoss-20b-v2" "kg__${o}__gptoss-20b")
done
"$PY" qa_eval.py report --run acl-f0 "${C[@]}" >> "$LOGS/qa_fill.log" 2>&1 && \
  cp qa_eval/acl-f0/report.json qa_eval/acl-f0/report_table20.json
say "=== fill done: graded + reported (report_table20.json). If nothing else runs, RESTORE the GPU settings ==="
