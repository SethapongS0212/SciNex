#!/usr/bin/env bash
# SciNex Part 2 on the 2x RTX 5090 box — ACL ANTHOLOGY corpus, FIXED extractor.
#
# The finetuning runs use the ACL Anthology papers (each has a fixed-entity CSV in
# output/acl/<id>/), not the 20-paper corpus (user, 2026-10-26). 805 papers (papers_acl.txt).
#
# Phase 1  teacher: openai/gpt-oss-20b (local, MXFP4, reasoning low, flex attention, batch 8)
#          through ft_generate --extractor fixed, both ontologies -> kg/fixed*/gptoss-20b/
#          GPU0 = odd lines of papers_acl.txt, GPU1 = even lines.  ~37,970 paragraphs.
# Phase 2  /home/dev1/scinex_runs/run_acl_phase2.sh, if it exists when phase 1 ends
#          (data build -> Qwen3-8B training -> eval generation -> compare).
#
# Re-running resumes: ft_generate --resume skips answered paragraphs per paper x ontology.
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/acl_driver.log"; }

teacher() {  # teacher <gpu> <paper list>
  CUDA_VISIBLE_DEVICES=$1 "$PY" ft_generate.py --base-model openai/gpt-oss-20b \
    --model-slug gptoss-20b --teacher gptoss-20b --extractor fixed --papers-from "$2" \
    --batch-size 8 --reasoning-effort low --attn-implementation flex_attention --ingest
}

lane() {  # lane <gpu>
  say "gpu$1: teacher gpt-oss-20b on $(wc -l < "$LOGS/acl_half$1.txt") papers started (log $LOGS/acl_teacher_gpu$1.log)"
  if teacher "$1" "$LOGS/acl_half$1.txt" >> "$LOGS/acl_teacher_gpu$1.log" 2>&1; then
    say "gpu$1: teacher finished"
  else
    say "gpu$1: teacher FAILED (exit $?) - see $LOGS/acl_teacher_gpu$1.log"
    return 1
  fi
}

say "=== run_acl phase 1 start ==="
lane 0 & p0=$!
lane 1 & p1=$!
wait $p0; r0=$?
wait $p1; r1=$?
say "phase 1 finished: gpu0 rc=$r0, gpu1 rc=$r1"
[ $r0 -eq 0 ] && [ $r1 -eq 0 ] || { say "phase 1 incomplete - re-run this script to resume"; exit 1; }

if [ -x "$LOGS/run_acl_phase2.sh" ]; then
  say "=== phase 2 start ==="
  "$LOGS/run_acl_phase2.sh"
  say "=== phase 2 exit $? ==="
else
  say "no run_acl_phase2.sh yet - stopping after the teacher run"
fi
