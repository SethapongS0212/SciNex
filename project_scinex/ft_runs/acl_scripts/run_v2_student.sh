#!/usr/bin/env bash
# Session 39 (2026-11-08): the ontology-free student trained on TEACHER v2 data (HANDOFF step 2).
#   1. wait for the teacher-v2 lanes of run_after_qa.sh (exit if either lane failed)
#   2. build ft_data/acl_v2_noont_me30 (teacher gptoss-20b-v2, output/acl_v2 lists, noont prompt,
#      max 30% empty targets, ACL split fold 0) and sanity-check the manifest counts
#   3. GPU1, once run_qa_fill.sh has filled its answers: train Qwen3-8B (QLoRA, 1 epoch,
#      --save-steps 200), then generate on the 41 eval papers with the noont prompt + v2 lists
#   4. Q&A answers from its KGs on GPU1 (GPU0 belongs to run_teacher2.sh), wait for run_qa_fill.sh
#      to end, grade + report under flock $LOGS/qa_grade.lock (one grades.jsonl writer at a time) with the Table 20 comparisons plus the
#      v2 ones -> qa_eval/acl-f0/report_table20.json
#   5. ft_compare: v1 teacher, v2 teacher, v1 noont-me30 student, v2 noont-me30 student
# Every step is skipped if its output exists, so re-running resumes.
# Log: /home/dev1/scinex_runs/v2_student_driver.log
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/v2_student_driver.log"; }
EVALF=ft_data/acl/fold0/eval_papers.txt
DATA=ft_data/acl_v2_noont_me30
RUN=qwen3-8b-acl-f0-v2-noont-me30
SLUG=qwen3-8b-acl-v2-noont-me30-cv

say "=== v2 student chain start: waiting for the teacher v2 lanes ==="
until grep -q "teacher v2 lanes:" "$LOGS/after_qa_driver.log"; do sleep 120; done
if ! grep -q "teacher v2 lanes: gpu0 rc=0, gpu1 rc=0" "$LOGS/after_qa_driver.log"; then
  say "teacher v2 lanes did not both succeed - stop (re-run run_after_qa.sh first, then this)"; exit 1; fi

if [ ! -f "$DATA/fold0/manifest.json" ]; then
  say "build $DATA"
  "$PY" build_ft_data.py --extractor fixed --teacher gptoss-20b-v2 --entity-dir output/acl_v2 \
    --prompt-variant noont --papers-file papers_acl.txt --split ft_data/acl/split.json --folds 20 \
    --eval-fold 0 --max-empty-frac 0.3 --out "$DATA" > "$LOGS/v2_build_data.log" 2>&1 \
    || { say "build FAILED - see $LOGS/v2_build_data.log"; exit 1; }
fi
"$PY" - "$DATA/fold0/manifest.json" >> "$LOGS/v2_student_driver.log" 2>&1 <<'EOF' || { say "manifest sanity check FAILED - stop"; exit 1; }
import json, sys
m = json.load(open(sys.argv[1]))
c = m["counts"]
print(f"  manifest: teacher {m['teacher']}, variant {m['prompt_variant']}, cap {m['max_empty_frac']}, "
      f"train papers {len(m['train_papers'])}, eval {len(m['eval_papers'])}, counts {c}")
ok = (m["teacher"] == "gptoss-20b-v2" and m["prompt_variant"] == "noont" and len(m["eval_papers"]) == 41
      and c["extraction"] >= 8000 and c["ontology"] >= 1000 and c["vocabulary"] >= 5000)
sys.exit(0 if ok else 1)
EOF

say "waiting for run_qa_fill.sh to finish its GPU1 answers"
until grep -q "answers filled" "$LOGS/qa_fill.log"; do sleep 60; done

if [ -f "ft_runs/$RUN/run_manifest.json" ] && [ -d "ft_runs/$RUN/adapter" ]; then
  say "gpu1: $RUN already trained - skip"
else
  say "gpu1: train $RUN (Qwen/Qwen3-8B on $DATA/fold0) started"
  CUDA_VISIBLE_DEVICES=1 "$PY" ft_train_qlora.py --data "$DATA/fold0" --base-model Qwen/Qwen3-8B \
    --run-name "$RUN" --epochs 1 --save-steps 200 >> "$LOGS/v2_train_$RUN.log" 2>&1 \
    || { say "gpu1: train FAILED (exit $?) - see $LOGS/v2_train_$RUN.log"; exit 1; }
  say "gpu1: train $RUN finished"
fi

if [ -f "$LOGS/v2_gen_$SLUG.done" ]; then
  say "gpu1: gen $SLUG done before - skip"
else
  say "gpu1: gen $SLUG started"
  CUDA_VISIBLE_DEVICES=1 "$PY" ft_generate.py --adapter "ft_runs/$RUN" --model-slug "$SLUG" \
    --extractor fixed --teacher gptoss-20b-v2 --prompt-variant noont --entity-dir output/acl_v2 \
    --papers-from "$EVALF" --batch-size 8 --ingest >> "$LOGS/v2_gen_$SLUG.log" 2>&1 \
    || { say "gpu1: gen FAILED (exit $?) - see $LOGS/v2_gen_$SLUG.log"; exit 1; }
  touch "$LOGS/v2_gen_$SLUG.done"; say "gpu1: gen $SLUG finished"
fi

say "gpu1: Q&A answers from $SLUG"
HF_BATCH=8 "$LOGS/run_qa.sh" 1 kg "fixed/$SLUG" "fixed_scinex/$SLUG"
while pgrep -f 'run_qa_fill[.]sh' > /dev/null; do sleep 60; done
exec 9> "$LOGS/qa_grade.lock"; flock 9
"$PY" qa_eval.py grade --run acl-f0 >> "$LOGS/v2_student_qa.log" 2>&1
C=()
cmp() {  # add a comparison only when both answer files exist (report exits on unknown conditions)
  [ -f "qa_eval/acl-f0/answers/$1.jsonl" ] && [ -f "qa_eval/acl-f0/answers/$2.jsonl" ] && C+=(--compare "$1" "$2"); }
for o in fixed fixed_scinex; do
  for pair in "noont-me50-cv ft-me50-cv" "noont-me30-cv ft-me30-cv" "noont-me50-cv base-noont" \
              "noont-me30-cv base-noont" "noont-me30-cv noont-me50-cv" "v2-noont-me30-cv noont-me30-cv"; do
    set -- $pair; cmp "kg__${o}__qwen3-8b-acl-$1" "kg__${o}__qwen3-8b-acl-$2"; done
  for s in qwen3-8b-acl-noont-me50-cv qwen3-8b-acl-noont-me30-cv; do cmp "kg__${o}__$s" "kg__${o}__gptoss-20b"; done
  cmp "kg__${o}__gptoss-20b-v2" "kg__${o}__gptoss-20b"
  cmp "kg__${o}__$SLUG" "kg__${o}__gptoss-20b-v2"
  cmp "kg__${o}__qwen3-30b-a3b-v2" "kg__${o}__gptoss-20b-v2"
  cmp "kg__${o}__qwen3-8b-acl-q30b-noont-me30-cv" "kg__${o}__$SLUG"
done
"$PY" qa_eval.py report --run acl-f0 "${C[@]}" >> "$LOGS/v2_student_qa.log" 2>&1 && \
  cp qa_eval/acl-f0/report.json qa_eval/acl-f0/report_table20.json && say "Q&A graded + reported (report_table20.json)" \
  || say "Q&A grade/report FAILED - see $LOGS/v2_student_qa.log"
flock -u 9

mapfile -t EVAL < "$EVALF"
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b \
  --slugs gptoss-20b gptoss-20b-v2 qwen3-8b-acl-noont-me30-cv "$SLUG" \
  --papers "${EVAL[@]}" --out ft_runs/teacher_v2/compare_v2_student.json > "$LOGS/v2_student_compare.log" 2>&1
say "=== v2 student chain done. Tables: $LOGS/v2_student_qa.log, $LOGS/v2_student_compare.log ==="
