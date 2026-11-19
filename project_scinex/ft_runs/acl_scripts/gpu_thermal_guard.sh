#!/usr/bin/env bash
# Thermal guard (Session 38, 2026-11-05). Runs as root in `screen -r gpuguard`.
# Every 30 s: if a GPU's margin to its max operating temperature (temperature.gpu.tlimit) is
# <= 2 C, lower that GPU's max graphics clock by 100 MHz (floor 1200). It never raises a clock.
# Undo by hand after the run: nvidia-smi -i 1 -rgc
# ⛔ Since 2026-11-19 GPU 0 belongs to a colleague: SciNex uses GPU 1 ONLY. This guard reads and
# changes GPU 1 only; never add GPU 0 back.
LOG=/home/dev1/scinex_runs/gpu_thermal_guard.log
GPU=1
declare -A CAP=([1]=1900)   # gpu1 1900 MHz = same speed as default, ~343 W / 77 C (Session 38)
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "guard start (GPU $GPU only): cap gpu$GPU=${CAP[$GPU]}"
while true; do
  while IFS=', ' read -r idx margin temp clk pw; do
    [ "$idx" = "$GPU" ] || continue
    if [ "$margin" -le 2 ] && [ "${CAP[$idx]}" -gt 1200 ]; then
      new=$(( ${CAP[$idx]} - 100 ))
      nvidia-smi -i "$idx" -lgc 0,"$new" >/dev/null 2>&1 && CAP[$idx]=$new
      say "gpu$idx margin ${margin}C temp ${temp}C clk ${clk}MHz ${pw}W -> max clock ${CAP[$idx]}"
    fi
  done < <(nvidia-smi -i "$GPU" --query-gpu=index,temperature.gpu.tlimit,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits)
  sleep 30
done
