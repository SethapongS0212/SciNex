#!/usr/bin/env bash
# Session 39: grade + report the Q&A run acl-f0 with every Table 20 comparison whose answers exist
# (CPU only, seconds). Writes qa_eval/acl-f0/report_table20.json. Log: qa_report_all.log
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; LOGS=/home/dev1/scinex_runs
exec 9> "$LOGS/qa_grade.lock"; flock 9
"$PY" qa_eval.py grade --run acl-f0 > "$LOGS/qa_report_all.log" 2>&1
C=()
cmp() { [ -f "qa_eval/acl-f0/answers/$1.jsonl" ] && [ -f "qa_eval/acl-f0/answers/$2.jsonl" ] && C+=(--compare "$1" "$2"); }
Q=qwen3-8b-acl
for o in fixed fixed_scinex; do
  k="kg__${o}__"
  for pair in "noont-me50-cv ft-me50-cv" "noont-me30-cv ft-me30-cv" "noont-me50-cv base-noont" "noont-me30-cv base-noont" \
              "noont-me30-cv noont-me50-cv" "v2-noont-me30-cv noont-me30-cv" "q30b-noont-me30-cv v2-noont-me30-cv" \
              "q30b-noont-me30-cv base-hf" "q30b-noont-me30-cv base-noont"; do
    set -- $pair; cmp "$k$Q-$1" "$k$Q-$2"; done
  for s in $Q-noont-me50-cv $Q-noont-me30-cv; do cmp "$k$s" "${k}gptoss-20b"; done
  cmp "${k}gptoss-20b-v2" "${k}gptoss-20b"
  cmp "$k$Q-v2-noont-me30-cv" "${k}gptoss-20b-v2"
  cmp "${k}qwen3-30b-a3b-v2" "${k}gptoss-20b-v2"
  cmp "$k$Q-q30b-noont-me30-cv" "${k}qwen3-30b-a3b-v2"
  # guards v2 (Session 39): each KG against itself under v1, then the key pairs under v2
  for s in gptoss-20b gptoss-20b-v2 qwen3-30b-a3b-v2 $Q-base-hf $Q-base-noont $Q-noont-me50-cv \
           $Q-noont-me30-cv $Q-v2-noont-me30-cv $Q-q30b-noont-me30-cv; do cmp "$k$s-g2" "$k$s"; done
  cmp "$k$Q-q30b-noont-me30-cv-g2" "$k$Q-v2-noont-me30-cv-g2"
  cmp "$k$Q-q30b-noont-me30-cv-g2" "${k}qwen3-30b-a3b-v2-g2"
  cmp "$k$Q-q30b-noont-me30-cv-g2" "$k$Q-base-hf-g2"
  cmp "${k}qwen3-30b-a3b-v2-g2" "${k}gptoss-20b-v2-g2"
  cmp "$k$Q-v2-noont-me30-cv-g2" "${k}gptoss-20b-v2-g2"
done
"$PY" qa_eval.py report --run acl-f0 "${C[@]}" >> "$LOGS/qa_report_all.log" 2>&1 && cp qa_eval/acl-f0/report.json qa_eval/acl-f0/report_table20.json
flock -u 9
