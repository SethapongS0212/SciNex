#!/usr/bin/env bash
# Session 39 (2026-11-12): scale-up curve point 0 = the 764 legacy training papers, Qwen3-30B
# teacher data under guards v2 (ft_data/acl_q30b_g2_noont_me30), Qwen3-8B QLoRA, 1 epoch, same
# recipe as every noont student. Generation happens later on the big test set (vLLM + LoRA).
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/scale_driver.log"; }
RUN=qwen3-8b-acl-f0-q30b-g2-noont-me30
if [ -f "ft_runs/$RUN/run_manifest.json" ] && [ -d "ft_runs/$RUN/adapter" ]; then say "p0: $RUN already trained"; exit 0; fi
say "p0: train $RUN on GPU0 started"
CUDA_VISIBLE_DEVICES=0 "$PY" ft_train_qlora.py --data ft_data/acl_q30b_g2_noont_me30/fold0 --base-model Qwen/Qwen3-8B \
  --run-name "$RUN" --epochs 1 --save-steps 200 >> "$LOGS/scale_train_$RUN.log" 2>&1 \
  && say "p0: train $RUN finished" || say "p0: train $RUN FAILED (exit $?)"
