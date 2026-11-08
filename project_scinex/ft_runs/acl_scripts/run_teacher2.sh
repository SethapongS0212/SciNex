#!/usr/bin/env bash
# Session 39 (2026-11-08): SECOND TEACHER = Qwen3-30B-A3B-Instruct-2507, 4-bit W4A16
# (RedHatAI/...-quantized.w4a16), served by vLLM from the separate venv /home/dev1/envvllm
# (ft_generate.py --backend vllm). Same setup as teacher v2 (gpt-oss-20b): --prompt-variant
# recall, --entity-dir output/acl_v2, paragraphs of the gptoss-20b run, max 3072 new tokens.
# GPU0 only (GPU1 belongs to run_v2_student.sh):
#   1. wait for run_after_qa.sh (the chain) to end, so GPU0 is free
#   2. SMOKE TEST: 2 training papers x 4 paragraphs, slug smoke-qwen30b (deleted afterwards):
#      valid-JSON rate + seconds per paragraph; stop if it fails
#   3. FULL RUN on all 805 papers x 2 ontologies, the 41 eval papers first (slug qwen3-30b-a3b-v2)
#   4. ft_compare teacher v1 / v2 / qwen30b on the eval papers
#   5. Q&A answers from its KGs (Mistral, HF backend, GPU0) + grade/report under the grade lock
#   6. ontology-free me30 student on ITS data (Qwen3-8B, same recipe as the v2 student, GPU0):
#      build ft_data/acl_q30b_noont_me30 -> train -> generate -> Q&A -> grade/report -> compare
# Every step resumes when re-run. Log: /home/dev1/scinex_runs/teacher2_driver.log
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
VPY=/home/dev1/envvllm/bin/python
LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/teacher2_driver.log"; }
MODEL=RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16
T2=qwen3-30b-a3b-v2
EVALF=ft_data/acl/fold0/eval_papers.txt
DATA=ft_data/acl_q30b_noont_me30
RUN=qwen3-8b-acl-f0-q30b-noont-me30
SLUG=qwen3-8b-acl-q30b-noont-me30-cv
VGEN=(--backend vllm --base-model "$MODEL" --teacher gptoss-20b --extractor fixed
      --prompt-variant recall --entity-dir output/acl_v2 --batch-size 64)

grade_report() {  # grade + report with every comparison whose answers exist, one writer at a time
  local label=$1
  exec 9> "$LOGS/qa_grade.lock"; flock 9
  "$PY" qa_eval.py grade --run acl-f0 >> "$LOGS/teacher2_qa.log" 2>&1
  local C=() o s pair
  cmp() { [ -f "qa_eval/acl-f0/answers/$1.jsonl" ] && [ -f "qa_eval/acl-f0/answers/$2.jsonl" ] && C+=(--compare "$1" "$2"); }
  for o in fixed fixed_scinex; do
    for pair in "noont-me50-cv ft-me50-cv" "noont-me30-cv ft-me30-cv" "noont-me50-cv base-noont" \
                "noont-me30-cv base-noont" "noont-me30-cv noont-me50-cv" "v2-noont-me30-cv noont-me30-cv" \
                "q30b-noont-me30-cv v2-noont-me30-cv"; do
      set -- $pair; cmp "kg__${o}__qwen3-8b-acl-$1" "kg__${o}__qwen3-8b-acl-$2"; done
    for s in qwen3-8b-acl-noont-me50-cv qwen3-8b-acl-noont-me30-cv; do cmp "kg__${o}__$s" "kg__${o}__gptoss-20b"; done
    cmp "kg__${o}__gptoss-20b-v2" "kg__${o}__gptoss-20b"
    cmp "kg__${o}__qwen3-8b-acl-v2-noont-me30-cv" "kg__${o}__gptoss-20b-v2"
    cmp "kg__${o}__$T2" "kg__${o}__gptoss-20b-v2"
    cmp "kg__${o}__$SLUG" "kg__${o}__$T2"
  done
  "$PY" qa_eval.py report --run acl-f0 "${C[@]}" >> "$LOGS/teacher2_qa.log" 2>&1 && \
    cp qa_eval/acl-f0/report.json qa_eval/acl-f0/report_table20.json && say "Q&A graded + reported ($label)" \
    || say "Q&A grade/report FAILED ($label) - see $LOGS/teacher2_qa.log"
  flock -u 9
}

say "=== teacher2 chain start: waiting for run_after_qa.sh (GPU0) ==="
while pgrep -f 'run_after_qa[.]sh' > /dev/null; do sleep 60; done
[ -x "$VPY" ] && "$VPY" -c "import vllm" 2>/dev/null || { say "vLLM venv not usable - stop"; exit 1; }

# 2. smoke test
if [ ! -f "$LOGS/teacher2_smoke.ok" ]; then
  SM=(); while read -r p; do SM+=("$p"); done < <(grep -vxF -f "$EVALF" papers_acl.txt | head -2)
  say "gpu0: smoke test on ${SM[*]} (4 paragraphs each, both ontologies)"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 "$VPY" ft_generate.py "${VGEN[@]}" --model-slug smoke-qwen30b \
    --papers "${SM[@]}" --limit 4 --no-resume >> "$LOGS/teacher2_smoke.log" 2>&1
  rc=$?; t1=$(date +%s)
  "$PY" - "${SM[@]}" >> "$LOGS/teacher2_driver.log" 2>&1 <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from ft_common import classify_raw
rows = []
for p in sys.argv[1:]:
    for ex in ("fixed", "fixed_scinex"):
        f = Path("output") / p / "kg" / ex / "smoke-qwen30b" / "responses.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in f.open(encoding="utf-8")]
ok = sum(classify_raw(r["raw"], api_run=False)[0] == "ok" for r in rows)
trip = sum(len((classify_raw(r["raw"], api_run=False)[1] or {}).get("triples", [])) for r in rows)
tok = sum(r.get("n_tokens", 0) for r in rows)
print(f"  smoke: {len(rows)} replies, {ok} valid JSON, {trip} triples proposed, "
      f"{sum(r.get('truncated', False) for r in rows)} truncated, {tok} tokens")
for r in rows[:2]:
    print("  sample:", r["raw"][:300].replace("\n", " "))
sys.exit(0 if rows and ok >= 0.8 * len(rows) else 1)
EOF
  vrc=$?
  say "smoke: ft_generate rc=$rc, check rc=$vrc, $((t1 - t0)) s wall (incl. model load + compile)"
  rm -rf output/*/kg/fixed/smoke-qwen30b output/*/kg/fixed_scinex/smoke-qwen30b
  [ $rc -eq 0 ] && [ $vrc -eq 0 ] || { say "smoke test FAILED - see $LOGS/teacher2_smoke.log; stop"; exit 1; }
  touch "$LOGS/teacher2_smoke.ok"
fi

# 3. full run, eval papers first
if [ ! -f "$LOGS/teacher2_full.done" ]; then
  cp "$EVALF" "$LOGS/teacher2_papers.txt"; grep -vxF -f "$EVALF" papers_acl.txt >> "$LOGS/teacher2_papers.txt"
  say "gpu0: $T2 on all $(wc -l < "$LOGS/teacher2_papers.txt") papers started"
  CUDA_VISIBLE_DEVICES=0 "$VPY" ft_generate.py "${VGEN[@]}" --model-slug "$T2" \
    --papers-from "$LOGS/teacher2_papers.txt" --ingest --ingest-python "$PY" >> "$LOGS/teacher2_full.log" 2>&1 \
    || { say "gpu0: $T2 FAILED (exit $?) - see $LOGS/teacher2_full.log (re-run resumes)"; exit 1; }
  touch "$LOGS/teacher2_full.done"; say "gpu0: $T2 finished"
fi

# 4. compare on the eval papers
mapfile -t EVAL < "$EVALF"
mkdir -p ft_runs/teacher2
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs gptoss-20b gptoss-20b-v2 "$T2" \
  --papers "${EVAL[@]}" --out ft_runs/teacher2/compare_teacher2.json > "$LOGS/teacher2_compare.log" 2>&1
say "teacher compare: $LOGS/teacher2_compare.log"

# 5. Q&A of the second teacher's KGs
HF_BATCH=8 "$LOGS/run_qa.sh" 0 kg "fixed/$T2" "fixed_scinex/$T2"
grade_report "teacher2 KGs"

# 6. ontology-free me30 student on the second teacher's data
if [ ! -f "$DATA/fold0/manifest.json" ]; then
  say "build $DATA"
  "$PY" build_ft_data.py --extractor fixed --teacher "$T2" --entity-dir output/acl_v2 \
    --prompt-variant noont --papers-file papers_acl.txt --split ft_data/acl/split.json --folds 20 \
    --eval-fold 0 --max-empty-frac 0.3 --out "$DATA" > "$LOGS/teacher2_build_data.log" 2>&1 \
    || { say "build FAILED - see $LOGS/teacher2_build_data.log"; exit 1; }
fi
"$PY" -c "
import json,sys; m=json.load(open('$DATA/fold0/manifest.json')); c=m['counts']
print('  manifest:', m['teacher'], m['prompt_variant'], m['max_empty_frac'], len(m['train_papers']), len(m['eval_papers']), c)
sys.exit(0 if m['teacher']=='$T2' and len(m['eval_papers'])==41 and c['extraction']>=5000 and c['ontology']>=800 and c['vocabulary']>=5000 else 1)
" >> "$LOGS/teacher2_driver.log" 2>&1 || { say "manifest sanity check FAILED - stop"; exit 1; }
if [ -f "ft_runs/$RUN/run_manifest.json" ] && [ -d "ft_runs/$RUN/adapter" ]; then
  say "gpu0: $RUN already trained - skip"
else
  say "gpu0: train $RUN started"
  CUDA_VISIBLE_DEVICES=0 "$PY" ft_train_qlora.py --data "$DATA/fold0" --base-model Qwen/Qwen3-8B \
    --run-name "$RUN" --epochs 1 --save-steps 200 >> "$LOGS/teacher2_train_$RUN.log" 2>&1 \
    || { say "gpu0: train FAILED (exit $?) - see $LOGS/teacher2_train_$RUN.log"; exit 1; }
  say "gpu0: train $RUN finished"
fi
if [ ! -f "$LOGS/teacher2_gen_$SLUG.done" ]; then
  say "gpu0: gen $SLUG started"
  CUDA_VISIBLE_DEVICES=0 "$PY" ft_generate.py --adapter "ft_runs/$RUN" --model-slug "$SLUG" \
    --extractor fixed --teacher "$T2" --prompt-variant noont --entity-dir output/acl_v2 \
    --papers-from "$EVALF" --batch-size 8 --ingest >> "$LOGS/teacher2_gen_$SLUG.log" 2>&1 \
    || { say "gpu0: gen FAILED (exit $?) - see $LOGS/teacher2_gen_$SLUG.log"; exit 1; }
  touch "$LOGS/teacher2_gen_$SLUG.done"; say "gpu0: gen $SLUG finished"
fi
HF_BATCH=8 "$LOGS/run_qa.sh" 0 kg "fixed/$SLUG" "fixed_scinex/$SLUG"
grade_report "q30b student"
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b \
  --slugs gptoss-20b gptoss-20b-v2 "$T2" qwen3-8b-acl-noont-me30-cv qwen3-8b-acl-v2-noont-me30-cv "$SLUG" \
  --papers "${EVAL[@]}" --out ft_runs/teacher2/compare_students.json > "$LOGS/teacher2_students_compare.log" 2>&1
say "=== teacher2 chain done. Tables: teacher2_compare.log, teacher2_students_compare.log, teacher2_qa.log ==="
