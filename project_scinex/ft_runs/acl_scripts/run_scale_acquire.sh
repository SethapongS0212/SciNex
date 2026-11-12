#!/usr/bin/env bash
# Session 39 (2026-11-12): scale-up acquisition (CPU + network only): download -> parse ->
# CS-NER entity lists -> v2 lists, for scale/candidates.txt (7,500 seeded candidates).
# Resumable: every stage skips finished papers. Log: /home/dev1/scinex_runs/scale_acquire.log
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; LOGS=/home/dev1/scinex_runs
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/scale_acquire_driver.log"; }
L=scale/candidates.txt
say "=== scale acquisition start ($(wc -l < $L) candidates) ==="
"$PY" scale_acquire.py download --list $L --delay 1.0 >> "$LOGS/scale_acquire.log" 2>&1; say "download stage done"
"$PY" scale_acquire.py parse --list $L --workers 8 --timeout 600 >> "$LOGS/scale_acquire.log" 2>&1; say "parse stage done"
"$PY" scale_acquire.py enrich --list $L >> "$LOGS/scale_acquire.log" 2>&1; say "enrich stage done"
ok=$(for p in $(cat $L); do [ -f "output/acl/$p/Entity_${p}_enriched.csv" ] && echo $p; done)
"$PY" build_entity_v2.py --papers $ok --summary scale/entity_v2_summary.json >> "$LOGS/scale_acquire.log" 2>&1; say "v2 lists done"
"$PY" scale_acquire.py status --list $L | tee -a "$LOGS/scale_acquire_driver.log"
say "=== scale acquisition done ==="
