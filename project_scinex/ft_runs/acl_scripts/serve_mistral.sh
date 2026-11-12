#!/usr/bin/env bash
# Session 39: the Q&A model (Mistral-Small-24B-Instruct-2501, W4A16 RedHatAI build) behind a
# local OpenAI-compatible vLLM server, for qa_eval.py --provider openai-compat --concurrency N.
# Usage: serve_mistral.sh <gpu> <port>
set -u
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=$1 exec /home/dev1/envvllm/bin/vllm serve RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w4a16 \
  --served-model-name mistral-small-24b --port "$2" --host 127.0.0.1 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --seed 0
