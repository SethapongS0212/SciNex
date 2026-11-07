#!/usr/bin/env bash
# Chain started 2026-11-07 (Session 38) so the GPUs keep working between sessions:
#   1. wait for the Q&A answer screens (qa0, qa1) to finish
#   2. grade + report the Q&A run acl-f0 (CPU)
#   3. TEACHER v2 = gpt-oss-20b with --prompt-variant recall + --entity-dir output/acl_v2 on ALL
#      805 ACL papers, both ontologies (slug gptoss-20b-v2). Each GPU does its share of the 41
#      eval papers FIRST, so the v2 KG for the test papers exists early. ~30 h.
#   4. ft_compare v1 vs v2 teacher on the 41 eval papers
#   5. Q&A answers from the v2 teacher's KGs (both ontologies) + grade + report
# Every step resumes if re-run (ft_generate --resume, qa_eval skips answered questions).
# Log: /home/dev1/scinex_runs/after_qa_driver.log
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/after_qa_driver.log"; }

say "=== chain start: waiting for qa0/qa1 ==="
while screen -ls | grep -qE '\.qa[01][[:space:]]'; do sleep 60; done
say "qa screens finished"
"$PY" qa_eval.py grade --run acl-f0 >> "$LOGS/qa_grade.log" 2>&1 && \
  "$PY" qa_eval.py report --run acl-f0 >> "$LOGS/qa_grade.log" 2>&1 && say "Q&A graded + reported (qa_eval/acl-f0/report.json)" \
  || say "Q&A grade/report FAILED - see $LOGS/qa_grade.log"

# paper lists: eval papers first on each GPU, then the remaining papers
EVALF=ft_data/acl/fold0/eval_papers.txt
awk 'NR % 2 == 1' "$EVALF" > "$LOGS/v2full_half0.txt"
awk 'NR % 2 == 0' "$EVALF" > "$LOGS/v2full_half1.txt"
grep -vxF -f "$EVALF" papers_acl.txt | awk 'NR % 2 == 1' >> "$LOGS/v2full_half0.txt"
grep -vxF -f "$EVALF" papers_acl.txt | awk 'NR % 2 == 0' >> "$LOGS/v2full_half1.txt"
say "teacher v2 lists: gpu0 $(wc -l < "$LOGS/v2full_half0.txt") papers, gpu1 $(wc -l < "$LOGS/v2full_half1.txt")"

lane() {  # lane <gpu>
  say "gpu$1: teacher v2 (all papers) started"
  if CUDA_VISIBLE_DEVICES=$1 "$PY" ft_generate.py --base-model openai/gpt-oss-20b \
      --model-slug gptoss-20b-v2 --teacher gptoss-20b --extractor fixed \
      --prompt-variant recall --entity-dir output/acl_v2 \
      --papers-from "$LOGS/v2full_half$1.txt" --batch-size 8 --reasoning-effort low \
      --attn-implementation flex_attention --ingest >> "$LOGS/teacher_v2full_gpu$1.log" 2>&1; then
    say "gpu$1: teacher v2 finished"
  else
    say "gpu$1: teacher v2 FAILED (exit $?) - see $LOGS/teacher_v2full_gpu$1.log"; return 1
  fi
}
lane 0 & p0=$!
lane 1 & p1=$!
wait $p0; r0=$?
wait $p1; r1=$?
say "teacher v2 lanes: gpu0 rc=$r0, gpu1 rc=$r1"
[ $r0 -eq 0 ] && [ $r1 -eq 0 ] || { say "teacher v2 incomplete - re-run this script to resume"; exit 1; }

mapfile -t EVAL < "$EVALF"
mkdir -p ft_runs/teacher_v2
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs gptoss-20b gptoss-20b-v2 \
  --papers "${EVAL[@]}" --out ft_runs/teacher_v2/compare_v2.json > "$LOGS/teacher_v2_compare.log" 2>&1
say "teacher v1 vs v2 compare: $LOGS/teacher_v2_compare.log"

HF_BATCH=8 "$LOGS/run_qa.sh" 0 kg fixed/gptoss-20b-v2 fixed_scinex/gptoss-20b-v2
"$PY" qa_eval.py grade --run acl-f0 >> "$LOGS/qa_grade.log" 2>&1 && \
  "$PY" qa_eval.py report --run acl-f0 >> "$LOGS/qa_grade.log" 2>&1
say "=== chain done: teacher v2 + its Q&A. GPUs idle - RESTORE the GPU settings if nothing else runs ==="
