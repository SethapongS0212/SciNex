#!/usr/bin/env bash
# Session 39 (2026-11-11): Q&A answers from the guards-v2 KGs (`<slug>-g2`, make_guard_variant.py),
# 9 slugs x 2 ontologies split over both GPUs, then qa_report_all.sh (grade + report with the
# v1-vs-v2 comparisons). Log: /home/dev1/scinex_runs/qa_g2_driver.log
set -u
LOGS=/home/dev1/scinex_runs
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/qa_g2_driver.log"; }
say "=== Q&A on guards-v2 KGs start ==="
HF_BATCH=8 "$LOGS/run_qa.sh" 0 kg fixed/qwen3-8b-acl-q30b-noont-me30-cv-g2 fixed_scinex/qwen3-8b-acl-q30b-noont-me30-cv-g2 \
  fixed/qwen3-30b-a3b-v2-g2 fixed_scinex/qwen3-30b-a3b-v2-g2 fixed/qwen3-8b-acl-base-hf-g2 fixed_scinex/qwen3-8b-acl-base-hf-g2 \
  fixed/qwen3-8b-acl-noont-me50-cv-g2 fixed_scinex/qwen3-8b-acl-noont-me50-cv-g2 fixed/gptoss-20b-g2 & p0=$!
HF_BATCH=8 "$LOGS/run_qa.sh" 1 kg fixed/qwen3-8b-acl-v2-noont-me30-cv-g2 fixed_scinex/qwen3-8b-acl-v2-noont-me30-cv-g2 \
  fixed/gptoss-20b-v2-g2 fixed_scinex/gptoss-20b-v2-g2 fixed/qwen3-8b-acl-noont-me30-cv-g2 fixed_scinex/qwen3-8b-acl-noont-me30-cv-g2 \
  fixed/qwen3-8b-acl-base-noont-g2 fixed_scinex/qwen3-8b-acl-base-noont-g2 fixed_scinex/gptoss-20b-g2 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
say "answer lanes rc $r0 / $r1"
"$LOGS/qa_report_all.sh" && say "graded + reported (report_table20.json, qa_report_all.log)" || say "grade/report FAILED"
say "=== Q&A on guards-v2 KGs done ==="
