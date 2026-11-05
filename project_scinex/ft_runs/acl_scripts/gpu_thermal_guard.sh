#!/usr/bin/env bash
# Thermal guard (Session 38, 2026-11-05). Runs as root in `screen -r gpuguard`.
# Every 30 s: if a GPU's margin to its max operating temperature (temperature.gpu.tlimit) is
# <= 2 C, lower that GPU's max graphics clock by 100 MHz (floor 1200). It never raises a clock.
# Undo by hand after the run: nvidia-smi -i N -rgc
LOG=/home/dev1/scinex_runs/gpu_thermal_guard.log
declare -A CAP=([0]=1650 [1]=1900)   # set by hand 12:5x: gpu0 1650 (1500 cost 30% speed), gpu1 1900
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "guard start: caps gpu0=${CAP[0]} gpu1=${CAP[1]}"
while true; do
  while IFS=', ' read -r idx margin temp clk pw; do
    [ -z "$idx" ] && continue
    if [ "$margin" -le 2 ] && [ "${CAP[$idx]}" -gt 1200 ]; then
      new=$(( ${CAP[$idx]} - 100 ))
      nvidia-smi -i "$idx" -lgc 0,"$new" >/dev/null 2>&1 && CAP[$idx]=$new
      say "gpu$idx margin ${margin}C temp ${temp}C clk ${clk}MHz ${pw}W -> max clock ${CAP[$idx]}"
    fi
  done < <(nvidia-smi --query-gpu=index,temperature.gpu.tlimit,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits)
  sleep 30
done
