#!/usr/bin/env bash
# Continue the ACL run after the reboot that clears the GPU1 fault (hands_off §37.7).
#
# Why a reboot: once GPU1 dropped off the bus (2026-10-28 16:39), NO new process could start CUDA
# ("CUDA unknown error", even with CUDA_VISIBLE_DEVICES=0); only the already-running teacher-only
# training kept going. Launched once by an @reboot crontab entry, which this script removes.
#
# Remaining work, every step a run_ft_pipeline step with a .done marker (re-running resumes):
#   teacher-only  train (skipped if done) -> ft_gen                  qwen3-8b-acl-ft-cv
#   max-empty-30% train (restart, checkpoint every 200 steps) -> ft_gen   qwen3-8b-acl-ft-me30-cv
#   base_gen                                                         qwen3-8b-acl-base-hf
#   compare
# Two healthy GPUs: GPU0 teacher-only + base_gen, GPU1 max-empty-30%. One: all serially.
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

# one-shot: remove the @reboot entry that started us
crontab -l 2>/dev/null | grep -v run_acl_after_reboot | crontab -

mapfile -t EVAL < ft_data/acl/fold0/eval_papers.txt

pipe() {  # pipe <gpu> <tag or -> <data> <steps...>
  local gpu=$1 tag=$2 data=$3; shift 3
  local tagarg=()
  [ "$tag" != "-" ] && tagarg=(--tag "$tag")
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_ft_pipeline.py --base-model Qwen/Qwen3-8B \
    --prefix qwen3-8b-acl --fold 0 --extractor fixed --teacher gptoss-20b \
    --papers "${EVAL[@]}" --gen-args=--batch-size=8 --train-args="$TRAIN_ARGS" \
    "${tagarg[@]}" --data "$data" --steps "$@"
}

gpu_ok() {  # NVML can see the card, then a CUDA op completes on it within 2 minutes.
  # Never touch a card nvidia-smi cannot see: a lost GPU put the shared nvidia-uvm module into an
  # error state (every open of /dev/nvidia-uvm -> EIO), which is what blocked ALL new CUDA work.
  nvidia-smi -i "$1" --query-gpu=name --format=csv,noheader > /dev/null 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$1 timeout 120 "$PY" -c \
    "import torch; torch.zeros(1).cuda(); torch.cuda.synchronize()" > /dev/null 2>&1
}

teacher_only() {
  step "gpu$1" "teacher-only train (skipped if done) + ft_gen" "$LOGS/acl_after_reboot_gpu$1.log" \
    pipe "$1" - ft_data/acl/fold0 train ft_gen
}
me30() {
  step "gpu$1" "max-empty-30% train + ft_gen" "$LOGS/acl_after_reboot_gpu$1.log" \
    pipe "$1" me30 ft_data/acl_maxempty30/fold0 train ft_gen
}
basegen() {
  step "gpu$1" "base_gen (qwen3-8b-acl-base-hf)" "$LOGS/acl_after_reboot_gpu$1.log" \
    pipe "$1" me30 ft_data/acl_maxempty30/fold0 base_gen
}

ckpt_smoke() {  # does --save-steps save AND resume? 2 steps (checkpoint each), then resume to 3
  local g=$1 r=ft_runs/ckpt-smoke log=$LOGS/ckpt_smoke.log
  local a=(ft_train_qlora.py --data ft_data/acl_maxempty30/fold0 --base-model Qwen/Qwen3-8B
           --run-name ckpt-smoke --save-steps 1)
  rm -rf "$r"
  CUDA_VISIBLE_DEVICES=$g "$PY" "${a[@]}" --max-steps 2 >> "$log" 2>&1 || return 1
  ls -d "$r"/checkpoints/checkpoint-* > /dev/null 2>&1 || return 1
  rm -rf "$r/adapter" "$r/run_manifest.json"
  CUDA_VISIBLE_DEVICES=$g "$PY" "${a[@]}" --max-steps 3 >> "$log" 2>&1 || return 1
  grep -q "resuming from" "$log" || return 1
  rm -rf "$r"
}

sleep 60   # let the NVIDIA driver settle after boot
say "=== after-reboot continuation start ==="
ok0=0; ok1=0
gpu_ok 0 && ok0=1
gpu_ok 1 && ok1=1
say "GPU health after reboot: gpu0=$ok0 gpu1=$ok1"

# step checkpoints only if they demonstrably save AND resume here; otherwise the old behaviour
TRAIN_ARGS="--epochs 1"
sg=$([ $ok0 -eq 1 ] && echo 0 || echo 1)
if { [ $ok0 -eq 1 ] || [ $ok1 -eq 1 ]; } && ckpt_smoke "$sg"; then
  TRAIN_ARGS="--epochs 1 --save-steps 200"
  say "checkpoint smoke passed - training with --save-steps 200"
else
  say "checkpoint smoke FAILED or no GPU - training without step checkpoints (see $LOGS/ckpt_smoke.log)"
fi

if [ $ok0 -eq 1 ] && [ $ok1 -eq 1 ]; then
  ( teacher_only 0; basegen 0 ) & pa=$!
  me30 1 & pb=$!
  wait $pa; wait $pb
elif [ $ok0 -eq 1 ] || [ $ok1 -eq 1 ]; then
  g=$([ $ok0 -eq 1 ] && echo 0 || echo 1)
  say "only GPU$g usable - running everything serially on it"
  teacher_only "$g"; me30 "$g"; basegen "$g"
else
  say "no usable GPU after reboot - nothing run. Check nvidia-smi, then re-run this script."
  exit 1
fi

also=()
[ -f ft_runs/_pipeline_qwen3-8b-acl-f0-me30/ft_gen.done ] && also=(--also-slugs qwen3-8b-acl-ft-me30-cv)
if [ -f ft_runs/_pipeline_qwen3-8b-acl-f0/base_gen.done ] && [ -f ft_runs/_pipeline_qwen3-8b-acl-f0/ft_gen.done ]; then
  step final "compare" "$LOGS/acl_compare.log" pipe 0 - ft_data/acl/fold0 compare "${also[@]}"
  say "=== ACL run done. Table: $LOGS/acl_compare.log, ft_runs/qwen3-8b-acl-f0/compare.json ==="
else
  say "compare skipped: base_gen or the teacher-only ft_gen is not done - re-run this script to resume"
fi
