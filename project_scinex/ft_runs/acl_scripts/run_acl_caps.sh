#!/usr/bin/env bash
# Session 37 follow-up: milder empty-target caps (hands_off §37.10).
# The 30% cap over-corrected: the student extracted on 43.8% of paragraphs vs the teacher's 25.6%,
# and 55% of its CEO triples sat on paragraphs the teacher left empty. Two students bracket the
# teacher's coverage, one per GPU, same recipe as me30 (1 epoch, Qwen3-8B QLoRA):
#   GPU0  ft_data/acl_maxempty50/fold0  ->  qwen3-8b-acl-ft-me50-cv
#   GPU1  ft_data/acl_maxempty60/fold0  ->  qwen3-8b-acl-ft-me60-cv
# then ft_compare over base / teacher-only / me30 / me50 / me60 / teacher on the 41 eval papers.
# --save-steps 200: a crash costs minutes; re-running resumes (pipeline .done markers + checkpoints).
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/acl_driver.log"; }

step() {  # step <lane> <name> <logfile> <cmd...>
  local lane=$1 name=$2 log=$3; shift 3
  say "$lane: $name started (log $log)"
  if "$@" >> "$log" 2>&1; then
    say "$lane: $name finished"
  else
    say "$lane: $name FAILED (exit $?) - see $log"
    return 1
  fi
}

mapfile -t EVAL < ft_data/acl/fold0/eval_papers.txt

pipe() {  # pipe <gpu> <tag> <data> <steps...>
  local gpu=$1 tag=$2 data=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_ft_pipeline.py --base-model Qwen/Qwen3-8B \
    --prefix qwen3-8b-acl --fold 0 --extractor fixed --teacher gptoss-20b \
    --papers "${EVAL[@]}" --gen-args=--batch-size=8 --train-args="--epochs 1 --save-steps 200" \
    --tag "$tag" --data "$data" --steps "$@"
}

say "=== caps run start: me50 on GPU0, me60 on GPU1 ==="
step gpu0 "max-empty-50% train + ft_gen" "$LOGS/acl_caps_gpu0.log" \
  pipe 0 me50 ft_data/acl_maxempty50/fold0 train ft_gen & p0=$!
step gpu1 "max-empty-60% train + ft_gen" "$LOGS/acl_caps_gpu1.log" \
  pipe 1 me60 ft_data/acl_maxempty60/fold0 train ft_gen & p1=$!
wait $p0; r0=$?
wait $p1; r1=$?
say "caps lanes finished: me50 rc=$r0, me60 rc=$r1"

slugs=(qwen3-8b-acl-base-hf qwen3-8b-acl-ft-cv qwen3-8b-acl-ft-me30-cv)
for t in me50 me60; do
  [ -f "ft_runs/_pipeline_qwen3-8b-acl-f0-$t/ft_gen.done" ] && slugs+=("qwen3-8b-acl-ft-$t-cv")
done
step final "compare (all caps)" "$LOGS/acl_compare_caps.log" \
  "$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs "${slugs[@]}" gptoss-20b \
    --papers "${EVAL[@]}" --out ft_runs/qwen3-8b-acl-f0/compare_caps.json
say "=== caps run done. Table: $LOGS/acl_compare_caps.log ==="
