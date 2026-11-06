#!/usr/bin/env bash
# MAIN EVALUATION (OBJECTIVE.md step 3): simulated Q&A against the 41 held-out ACL papers.
# Generator AND answerer: mistralai/Mistral-Small-24B-Instruct-2501, local, 4-bit, greedy —
# neither a teacher nor a student family, never Claude. Grading is deterministic (overlap).
# Questions come from the paragraphs the extractors saw (kg/fixed/gptoss-20b/prompts.jsonl).
# Usage: run_qa.sh <gpu> <step> [kg specs...]
#   run_qa.sh 0 questions                     generate the question set once (run acl-f0)
#   run_qa.sh 0 controls                      oracle + closedbook answers
#   run_qa.sh 0 kg fixed/<slug> fixed_scinex/<slug> ...   answer from those KGs
#   run_qa.sh 0 grade                         grade every answer file + report
# Every step resumes (qa_eval skips answered questions).
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=$1; STEP=$2; shift 2
RUN=acl-f0
QA=(--provider hf --model mistralai/Mistral-Small-24B-Instruct-2501 --hf-batch "${HF_BATCH:-4}")
say() { echo "[$(date '+%F %T')] qa gpu$GPU $*" | tee -a "$LOGS/qa_driver.log"; }
run() { say "$STEP started: $*"
  if CUDA_VISIBLE_DEVICES=$GPU "$PY" qa_eval.py "$@" >> "$LOGS/qa_$STEP.log" 2>&1; then say "$STEP finished"
  else say "$STEP FAILED (exit $?) - see $LOGS/qa_$STEP.log"; return 1; fi; }
case "$STEP" in
  questions) run generate --run "$RUN" --papers-from ft_data/acl/fold0/eval_papers.txt \
               --para-exdir fixed --para-slug gptoss-20b "${QA[@]}" ;;
  controls)  run answer --run "$RUN" --context oracle "${QA[@]}" && \
             run answer --run "$RUN" --context closedbook "${QA[@]}" ;;
  kg)        run answer --run "$RUN" --context kg --kg "$@" "${QA[@]}" ;;
  grade)     run grade --run "$RUN" && run report --run "$RUN" ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
