#!/usr/bin/env bash
# Root waiter (Session 39, 2026-11-07), started with: sudo screen -dmS gpurestore bash <this file>
# Once the unattended drivers (run_after_qa.sh = screen chain, run_qa_fill.sh = screen qafill) have
# exited AND no CUDA process has run for 10 min, restore the GPU defaults (university box):
# stop the thermal guard, reset clock locks, power limit 575 W, persistence off.
LOG=/home/dev1/scinex_runs/gpu_restore.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
# Optional $1 = pgrep -f pattern of the drivers to wait for (bracket the dot so the pattern
# never matches this script's own command line), e.g. 'run_teacher2[.]sh'.
PAT=${1:-'run_after_qa[.]sh|run_qa_fill[.]sh'}
say "gpurestore: waiting for drivers matching $PAT to exit"
while pgrep -f "$PAT" > /dev/null; do sleep 120; done
say "drivers exited; waiting for 10 idle minutes on the GPUs"
quiet=0
while [ "$quiet" -lt 5 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
    quiet=$((quiet + 1)); else quiet=0; fi
  sleep 120
done
say "GPUs idle: restoring defaults"
screen -S gpuguard -X quit >> "$LOG" 2>&1
pkill -f 'gpu_thermal_guard[.]sh' >> "$LOG" 2>&1
nvidia-smi -i 0 -rgc >> "$LOG" 2>&1
nvidia-smi -i 1 -rgc >> "$LOG" 2>&1
nvidia-smi -pl 575 >> "$LOG" 2>&1
nvidia-smi -pm 0 >> "$LOG" 2>&1
nvidia-smi --query-gpu=index,persistence_mode,power.limit,clocks.max.sm,temperature.gpu --format=csv >> "$LOG" 2>&1
say "=== GPU settings RESTORED ==="
