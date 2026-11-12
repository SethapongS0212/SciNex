#!/usr/bin/env bash
# Session 39 (2026-11-12): scale-up LEARNING CURVE + Q&A on the 1,041-paper test set.
# Waits for the teacher (scale_teacher.done) and curve point 0 (ft_runs/<p0>/adapter).
# Students: Qwen3-8B QLoRA, noont prompt, max 30% empty, 1 epoch, teacher data
# qwen3-30b-a3b-v2-g2 (guards v2), nested training sets of 764 (p0) / 1500 / 3000 / all papers.
# Generation on the test set via vLLM + LoRA with cross-paper batching; Q&A with the vLLM-served
# W4A16 Mistral (run `scale`, fresh questions, <= 10 per paper; plus run `acl-f0-vllm` for the 41
# legacy papers). Every step skips finished work, so re-running resumes.
#   GPU0: gen n764 -> Mistral (questions, controls, teacher answers) -> train full -> gen full
#   GPU1: train 1500 -> gen 1500 -> train 3000 -> gen 3000
#   then: Mistral answers for every student, grade, report. Log: scale_driver.log
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; VPY=/home/dev1/envvllm/bin/python; LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
say() { echo "[$(date '+%F %T')] curve: $*" | tee -a "$LOGS/scale_driver.log"; }
T=qwen3-30b-a3b-v2-g2
P0=qwen3-8b-acl-f0-q30b-g2-noont-me30
TEST=scale/test_papers.txt
QA=(--provider openai-compat --base-url http://127.0.0.1:8011/v1 --model mistral-small-24b --concurrency 32 --timeout 900)

slug_of() { echo "qwen3-8b-scale-q30b-g2-noont-me30-n$1"; }
run_of()  { [ "$1" = 764 ] && echo "$P0" || echo "qwen3-8b-scale-q30b-g2-noont-me30-n$1"; }

build() {  # build <N>  (N = 1500 | 3000 | full)
  local n=$1 corpus out=ft_data/scale_q30b_g2_noont_me30_n$1
  corpus=$([ "$n" = full ] && echo papers_scale.txt || echo scale/curve_$n.txt)
  [ -f "$out/fold0/manifest.json" ] && return 0
  say "build data n$n ($corpus)"
  "$PY" build_ft_data.py --extractor fixed --teacher $T --entity-dir output/acl_v2 --prompt-variant noont \
    --papers-file "$corpus" --split ft_data/scale/split.json --folds 2 --seed 2026 --eval-fold 0 \
    --max-empty-frac 0.3 --out "$out" > "$LOGS/scale_build_n$n.log" 2>&1 || { say "build n$n FAILED"; return 1; }
  "$PY" -c "import json;m=json.load(open('$out/fold0/manifest.json'));print('  n$n manifest:',len(m['train_papers']),'train,',len(m['eval_papers']),'eval,',m['counts'])" | tee -a "$LOGS/scale_driver.log"
}

train() {  # train <gpu> <N>
  local gpu=$1 n=$2 run; run=$(run_of $n)
  [ -f "ft_runs/$run/run_manifest.json" ] && [ -d "ft_runs/$run/adapter" ] && return 0
  build $n || return 1
  say "gpu$gpu: train $run started"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" ft_train_qlora.py --data ft_data/scale_q30b_g2_noont_me30_n$n/fold0 \
    --base-model Qwen/Qwen3-8B --run-name "$run" --epochs 1 --save-steps 200 >> "$LOGS/scale_train_$run.log" 2>&1 \
    && say "gpu$gpu: train $run finished" || { say "gpu$gpu: train $run FAILED"; return 1; }
}

gen() {  # gen <gpu> <N>
  local gpu=$1 n=$2 run slug; run=$(run_of $n); slug=$(slug_of $n)
  [ -f "$LOGS/scale_gen_$slug.done" ] && return 0
  say "gpu$gpu: gen $slug on $(wc -l < $TEST) test papers started"
  CUDA_VISIBLE_DEVICES=$gpu "$VPY" ft_generate.py --backend vllm --adapter "ft_runs/$run" --model-slug "$slug" \
    --extractor fixed --teacher $T --prompt-variant noont --entity-dir output/acl_v2 --guards v2 \
    --papers-from $TEST --vllm-group 16 --max-model-len 6144 --gpu-memory-utilization 0.92 \
    --ingest --ingest-python "$PY" >> "$LOGS/scale_gen_$slug.log" 2>&1 \
    && { touch "$LOGS/scale_gen_$slug.done"; say "gpu$gpu: gen $slug finished"; } || { say "gpu$gpu: gen $slug FAILED"; return 1; }
}

serve_start() {  # serve_start <gpu>
  screen -dmS mistralsrv bash -c "$LOGS/serve_mistral.sh $1 8011 > $LOGS/mistral_server.log 2>&1"
  for _ in $(seq 120); do curl -s -m 2 http://127.0.0.1:8011/health > /dev/null && return 0; sleep 10; done
  say "Mistral server did not come up"; return 1
}
serve_stop() { screen -S mistralsrv -X quit 2>/dev/null; sleep 15; }

say "waiting for the teacher and curve point 0"
until [ -f "$LOGS/scale_teacher.done" ]; do sleep 300; done
until [ -d "ft_runs/$P0/adapter" ]; do sleep 300; done

lane0() {
  gen 0 764
  if [ ! -f "$LOGS/scale_qa_questions.done" ]; then
    serve_start 0 && {
      say "questions for run scale"
      "$PY" qa_eval.py generate --run scale --papers-from $TEST --para-exdir fixed --para-slug $T \
        --per-paragraph 3 --max-per-paper 10 "${QA[@]}" >> "$LOGS/scale_qa.log" 2>&1
      "$PY" qa_eval.py generate --run scale --papers-from $TEST --para-exdir fixed --para-slug $T \
        --per-paragraph 3 --max-per-paper 10 "${QA[@]}" >> "$LOGS/scale_qa.log" 2>&1   # retry failed calls
      for c in oracle closedbook; do "$PY" qa_eval.py answer --run scale --context $c "${QA[@]}" >> "$LOGS/scale_qa.log" 2>&1; done
      "$PY" qa_eval.py answer --run scale --context kg --kg fixed/$T fixed_scinex/$T "${QA[@]}" >> "$LOGS/scale_qa.log" 2>&1
      touch "$LOGS/scale_qa_questions.done"; say "questions + controls + teacher answers done"; }
    serve_stop
  fi
  train 0 full && gen 0 full
}
lane1() { train 1 1500 && gen 1 1500 && train 1 3000 && gen 1 3000; }
lane0 & p0=$!; lane1 & p1=$!
wait $p0; wait $p1

serve_start 0 || exit 1
KGS=(); for n in 764 1500 3000 full; do s=$(slug_of $n); [ -f "$LOGS/scale_gen_$s.done" ] && KGS+=("fixed/$s" "fixed_scinex/$s"); done
for pass in 1 2; do   # the second pass re-asks questions a batched reply skipped
  "$PY" qa_eval.py answer --run scale --context kg --kg "${KGS[@]}" fixed/$T fixed_scinex/$T "${QA[@]}" >> "$LOGS/scale_qa.log" 2>&1
  "$PY" qa_eval.py answer --run acl-f0-vllm --context kg --kg "${KGS[@]}" "${QA[@]}" >> "$LOGS/scale_qa.log" 2>&1
done
serve_stop
C=()
for o in fixed fixed_scinex; do
  for n in 1500 3000 full; do C+=(--compare "kg__${o}__$(slug_of $n)" "kg__${o}__$(slug_of 764)"); done
  for n in 764 1500 3000 full; do C+=(--compare "kg__${o}__$(slug_of $n)" "kg__${o}__$T"); done
done
for run in scale acl-f0-vllm; do
  "$PY" qa_eval.py grade --run $run >> "$LOGS/scale_qa.log" 2>&1
  CC=(); for ((i=0; i<${#C[@]}; i+=3)); do
    [ -f "qa_eval/$run/answers/${C[i+1]}.jsonl" ] && [ -f "qa_eval/$run/answers/${C[i+2]}.jsonl" ] && CC+=("${C[@]:i:3}"); done
  "$PY" qa_eval.py report --run $run "${CC[@]}" > "$LOGS/scale_report_$run.log" 2>&1 && cp qa_eval/$run/report.json qa_eval/$run/report_scale.json
done
mkdir -p ft_runs/scale; mapfile -t TP < $TEST
SL=(); for n in 764 1500 3000 full; do SL+=("$(slug_of $n)"); done
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs $T "${SL[@]}" --papers "${TP[@]}" \
  --out ft_runs/scale/compare_curve.json > "$LOGS/scale_compare_curve.log" 2>&1
say "=== curve done: scale_report_scale.log, scale_report_acl-f0-vllm.log, scale_compare_curve.log ==="
