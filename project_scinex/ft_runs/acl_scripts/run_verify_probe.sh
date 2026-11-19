#!/usr/bin/env bash
# Session 41 (2026-11-19): verifier probe on the 700 blind-judged triples. GPU 1 ONLY (GPU 0 = colleague).
# Default clocks (no sudo): a watchdog kills the run if GPU 1's margin to its limit drops to <= 5 C.
cd /home/dev1/project_scinex || exit 1
LOGS=/home/dev1/scinex_runs; VPY=/home/dev1/envvllm/bin/python; GPU=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
TAG=${1:-q30b}; MODEL=${2:-RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16}
CUDA_VISIBLE_DEVICES=$GPU "$VPY" verify_triples.py probe --rounds acl-f0-q30b scale-test --tag "$TAG" --model "$MODEL" \
  > "$LOGS/verify_probe_$TAG.log" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  m=$(nvidia-smi -i $GPU --query-gpu=temperature.gpu.tlimit --format=csv,noheader,nounits | tr -d ' ')
  echo "$(date '+%T') margin=$m temp=$(nvidia-smi -i $GPU --query-gpu=temperature.gpu,power.draw --format=csv,noheader)" >> "$LOGS/verify_thermal.log"
  if [ -n "$m" ] && [ "$m" -le 5 ]; then echo "THERMAL STOP margin $m" >> "$LOGS/verify_probe_$TAG.log"; kill $pid; sleep 5; kill -9 $pid 2>/dev/null; fi
  sleep 3
done
echo "probe exit" >> "$LOGS/verify_probe_$TAG.log"
