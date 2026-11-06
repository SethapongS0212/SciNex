#!/usr/bin/env bash
# Teacher v2 on the 41 held-out ACL papers (fold 0) — decides whether the full 805-paper v2
# teacher run (~30 h) is worth it. v2 = ft_generate --prompt-variant recall --entity-dir
# output/acl_v2 (cleaned lists + the papers' own model names), same paragraphs/para_ids as
# the v1 teacher (gptoss-20b). Output slug gptoss-20b-v2. Then ft_compare v1 vs v2.
# Usage: run_teacher_v2_test.sh <reasoning-effort low|medium>
# Re-running resumes (ft_generate --resume).
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs
EFFORT=${1:-low}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/teacher_v2_driver.log"; }

mapfile -t EVAL < ft_data/acl/fold0/eval_papers.txt
awk 'NR % 2 == 1' ft_data/acl/fold0/eval_papers.txt > "$LOGS/v2test_half0.txt"
awk 'NR % 2 == 0' ft_data/acl/fold0/eval_papers.txt > "$LOGS/v2test_half1.txt"

lane() {  # lane <gpu>
  say "gpu$1: teacher v2 ($EFFORT) on $(wc -l < "$LOGS/v2test_half$1.txt") papers started"
  if CUDA_VISIBLE_DEVICES=$1 "$PY" ft_generate.py --base-model openai/gpt-oss-20b \
      --model-slug gptoss-20b-v2 --teacher gptoss-20b --extractor fixed \
      --prompt-variant recall --entity-dir output/acl_v2 \
      --papers-from "$LOGS/v2test_half$1.txt" --batch-size 8 --reasoning-effort "$EFFORT" \
      --attn-implementation flex_attention --ingest >> "$LOGS/teacher_v2_gpu$1.log" 2>&1; then
    say "gpu$1: teacher v2 finished"
  else
    say "gpu$1: teacher v2 FAILED (exit $?) - see $LOGS/teacher_v2_gpu$1.log"; return 1
  fi
}

say "=== teacher v2 test run start (effort $EFFORT) ==="
lane 0 & p0=$!
lane 1 & p1=$!
wait $p0; r0=$?
wait $p1; r1=$?
say "lanes finished: gpu0 rc=$r0, gpu1 rc=$r1"
[ $r0 -eq 0 ] && [ $r1 -eq 0 ] || { say "incomplete - re-run to resume"; exit 1; }
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs gptoss-20b gptoss-20b-v2 \
  --papers "${EVAL[@]}" --out ft_runs/teacher_v2/compare_v2test.json > "$LOGS/teacher_v2_compare.log" 2>&1
say "=== teacher v2 test run done. Table: $LOGS/teacher_v2_compare.log ==="
