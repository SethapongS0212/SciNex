#!/usr/bin/env bash
# Phase 2 of run_acl.sh — ACL Anthology corpus, FIXED extractor, after the gpt-oss-20b teacher pass.
#
#   data   build_ft_data --extractor fixed, ACL-only split ft_data/acl/split.json (20 folds,
#          fold 0 = 41 eval papers), two variants:
#            ft_data/acl/fold0              teacher-only, the teacher's own empty/non-empty mix
#            ft_data/acl_maxempty30/fold0   --max-empty-frac 0.3 (the Session 36 recall fix)
#   GPU0   teacher-only: train (1 epoch) -> ft_gen                      slug qwen3-8b-acl-ft-cv
#   GPU1   max-empty-30%: train (1 epoch) -> base_gen -> ft_gen        slug qwen3-8b-acl-ft-me30-cv
#                                           (base slug qwen3-8b-acl-base-hf, shared)
#   then   ft_compare: base vs both finetunes vs teacher on the 41 eval papers
#
# 1 epoch, not the default 2: ~36k extraction examples vs the 619 of Session 36.
# Re-running resumes: run_ft_pipeline leaves ft_runs/_pipeline_<run>/*.done markers.
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

BUILD=(--extractor fixed --teacher gptoss-20b --papers-file papers_acl.txt
       --split ft_data/acl/split.json --folds 20 --eval-fold 0)
step data "build teacher-only" "$LOGS/acl_build.log" \
  "$PY" build_ft_data.py "${BUILD[@]}" --out ft_data/acl || exit 1
step data "build max-empty-30%" "$LOGS/acl_build.log" \
  "$PY" build_ft_data.py "${BUILD[@]}" --out ft_data/acl_maxempty30 --max-empty-frac 0.3 || exit 1
mapfile -t EVAL < ft_data/acl/fold0/eval_papers.txt
say "eval papers (fold 0): ${#EVAL[@]}"

pipe() {  # pipe <gpu> <tag or -> <data> <steps...>
  local gpu=$1 tag=$2 data=$3; shift 3
  local tagarg=()
  [ "$tag" != "-" ] && tagarg=(--tag "$tag")
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_ft_pipeline.py --base-model Qwen/Qwen3-8B \
    --prefix qwen3-8b-acl --fold 0 --extractor fixed --teacher gptoss-20b \
    --papers "${EVAL[@]}" --gen-args=--batch-size=8 --train-args="--epochs 1" \
    "${tagarg[@]}" --data "$data" --steps "$@"
}

lane0() {
  step gpu0 "teacher-only train+ft_gen" "$LOGS/acl_gpu0_teacheronly.log" \
    pipe 0 - ft_data/acl/fold0 train ft_gen
}

lane1() {
  step gpu1 "max-empty-30% train" "$LOGS/acl_gpu1_me30.log" \
    pipe 1 me30 ft_data/acl_maxempty30/fold0 train || return 1
  step gpu1 "base_gen (qwen3-8b-acl-base-hf)" "$LOGS/acl_gpu1_me30.log" \
    pipe 1 me30 ft_data/acl_maxempty30/fold0 base_gen || return 1
  step gpu1 "max-empty-30% ft_gen" "$LOGS/acl_gpu1_me30.log" \
    pipe 1 me30 ft_data/acl_maxempty30/fold0 ft_gen
}

lane0 & p0=$!
lane1 & p1=$!
wait $p0; r0=$?
wait $p1; r1=$?
say "phase 2 lanes finished: gpu0 rc=$r0, gpu1 rc=$r1"

also=()
[ -f ft_runs/_pipeline_qwen3-8b-acl-f0-me30/ft_gen.done ] && also=(--also-slugs qwen3-8b-acl-ft-me30-cv)
if [ -f ft_runs/_pipeline_qwen3-8b-acl-f0/base_gen.done ] && [ -f ft_runs/_pipeline_qwen3-8b-acl-f0/ft_gen.done ]; then
  step final "compare" "$LOGS/acl_compare.log" pipe 0 - ft_data/acl/fold0 compare "${also[@]}"
  say "=== phase 2 done. Table: $LOGS/acl_compare.log, ft_runs/qwen3-8b-acl-f0/compare.json ==="
else
  say "compare skipped: base_gen or the teacher-only ft_gen is not done - re-run to resume"
fi
