#!/usr/bin/env bash
# Session 41 (2026-11-19): option (a) "better training targets" = VERIFIER-FILTERED teacher targets.
# ⛔ GPU 1 ONLY (GPU 0 belongs to a colleague since 2026-11-19).
#   s1 score   Qwen3-30B verifier P(YES) for every teacher triple (805 ACL + 1,041 scale-test papers)
#   s2 variant teacher <T>-v50 (triples with P(YES) < 0.5 removed), re-ingested; ft_compare; answer-in-KG
#   s3 qa_t    Q&A on the teacher-v50 test KG (Mistral)                       [waits for the clock lock]
#   s4 build   training data = the n764 recipe on <T>-v50 (ACL split, fold 0, noont, max 30% empty)
#   s5 train   Qwen3-8B QLoRA 1 epoch (~5 h)                                  [waits for the clock lock]
#   s6 gen     student on the 1,041 test papers (vLLM + LoRA, ~3.3 h)
#   s7 qa_s    Q&A on the student KG; report vs n764 / teacher / teacher-v50; ft_compare
# Every GPU step WAITS FOR THE CLOCK LOCK (= persistence mode Enabled on GPU 1; the lock recipe sets -pm 1
# first): at default settings GPU 1 went 47 -> 85 C in 20 s at 575 W (2026-11-19 13:41). Each GPU step also
# runs under a thermal watchdog (3 s poll): margin <= 4 C -> kill the step and any process left on GPU 1,
# cool to margin >= 25 C, resume (all steps resume).
# Re-running the script resumes; finished steps leave <step>.done in $LOGS/verify/.
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; VPY=/home/dev1/envvllm/bin/python; LOGS=/home/dev1/scinex_runs
D=$LOGS/verify; mkdir -p "$D"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=1
GPU=1; PORT=8013; TAG=q30b; THR=0.5; SUF=-v50
T=qwen3-30b-a3b-v2-g2; TV=$T$SUF
P0=qwen3-8b-scale-q30b-g2-noont-me30-n764
RUN=qwen3-8b-acl-f0-q30b-g2v50-noont-me30; SLUG=qwen3-8b-scale-q30b-g2v50-noont-me30-n764
DATA=ft_data/acl_q30b_g2v50_noont_me30
TEST=scale/test_papers.txt; PAPERS=$D/papers_all.txt
QA=(--provider openai-compat --base-url http://127.0.0.1:$PORT/v1 --model mistral-small-24b --concurrency 24 --timeout 900)
say() { echo "[$(date '+%F %T')] verify: $*" | tee -a "$LOGS/scale_driver.log" >> "$D/chain.log"; }
margin() { nvidia-smi -i $GPU --query-gpu=temperature.gpu.tlimit --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
locked() { [ "$(nvidia-smi -i $GPU --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)" = "Enabled" ]; }
serve_start() {
  screen -dmS mistralv bash -c "$LOGS/serve_mistral.sh $GPU $PORT > $D/mistral_server.log 2>&1"
  for _ in $(seq 120); do curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null && return 0; sleep 10; done
  say "Mistral did not come up"; return 1
}
serve_stop() { screen -S mistralv -X quit 2>/dev/null; sleep 15; }

# ---- the steps (each run as `$0 <step>` in its own process group) ---------------------------------
s1() { "$VPY" verify_triples.py score --slugs $T --papers-from "$PAPERS" --tag $TAG >> "$D/s1_score.log" 2>&1; }
s2() {
  "$PY" verify_triples.py variant --slugs $T --papers-from "$PAPERS" --tag $TAG --threshold $THR --suffix=$SUF >> "$D/s2_variant.log" 2>&1 || return 1
  mapfile -t TP < $TEST
  "$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs $T $TV --papers "${TP[@]}" \
    --out ft_runs/verify/compare_teacher_v50.json > "$D/s2_compare.log" 2>&1
  "$PY" ft_runs/verify/scripts/aikg.py $T $TV $P0 > "$D/s2_aikg.log" 2>&1
}
s3() {
  serve_start || return 1
  for pass in 1 2; do "$PY" qa_eval.py answer --run scale --context kg --kg fixed/$TV fixed_scinex/$TV "${QA[@]}" >> "$D/s3_qa.log" 2>&1; done
  serve_stop
}
s4() {
  "$PY" build_ft_data.py --extractor fixed --teacher $TV --entity-dir output/acl_v2 --prompt-variant noont \
    --papers-file papers_acl.txt --split ft_data/acl/split.json --folds 20 --eval-fold 0 --max-empty-frac 0.3 \
    --out $DATA > "$D/s4_build.log" 2>&1
}
s5() {
  "$PY" ft_train_qlora.py --data $DATA/fold0 --base-model Qwen/Qwen3-8B --run-name $RUN --epochs 1 --save-steps 200 \
    >> "$D/s5_train.log" 2>&1
}
s6() {
  "$VPY" ft_generate.py --backend vllm --adapter "ft_runs/$RUN" --model-slug $SLUG --extractor fixed --teacher $T \
    --prompt-variant noont --entity-dir output/acl_v2 --guards v2 --papers-from $TEST --vllm-group 16 \
    --max-model-len 6144 --gpu-memory-utilization 0.92 --ingest --ingest-python "$PY" >> "$D/s6_gen.log" 2>&1
}
s7() {
  serve_start || return 1
  for pass in 1 2; do "$PY" qa_eval.py answer --run scale --context kg --kg fixed/$SLUG fixed_scinex/$SLUG fixed/$TV fixed_scinex/$TV "${QA[@]}" >> "$D/s7_qa.log" 2>&1; done
  serve_stop
  C=(); for o in fixed fixed_scinex; do
    C+=(--compare "kg__${o}__$SLUG" "kg__${o}__$P0" --compare "kg__${o}__$SLUG" "kg__${o}__$T"
        --compare "kg__${o}__$SLUG" "kg__${o}__$TV" --compare "kg__${o}__$TV" "kg__${o}__$T"); done
  "$PY" qa_eval.py grade --run scale >> "$D/s7_qa.log" 2>&1
  "$PY" qa_eval.py report --run scale "${C[@]}" > "$D/s7_report.log" 2>&1 && cp qa_eval/scale/report.json qa_eval/scale/report_v50.json
  mapfile -t TP < $TEST
  "$PY" ft_compare.py --extractor fixed --teacher gptoss-20b --slugs $T $TV $P0 $SLUG --papers "${TP[@]}" \
    --out ft_runs/verify/compare_v50.json > "$D/s7_compare.log" 2>&1
  "$PY" ft_runs/verify/scripts/aikg.py $T $TV $P0 $SLUG > "$D/s7_aikg.log" 2>&1
}
case "${1:-}" in s1|s2|s3|s4|s5|s6|s7) "$1"; exit $? ;; esac

# ---- the chain -------------------------------------------------------------------------------------
gpu1_clean() {  # kill anything still holding GPU 1 (only SciNex runs there; GPU 0 is never touched)
  local p; for p in $(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    say "killing leftover GPU $GPU process $p ($(ps -o comm= -p $p))"; kill -TERM $p 2>/dev/null; sleep 3; kill -KILL $p 2>/dev/null; done
}
guarded() {  # guarded <step>: run under the thermal watchdog, resume after cooling (max 8 tries)
  local step=$1 try rc
  [ -f "$D/$step.done" ] && return 0
  for try in 1 2 3 4 5 6 7 8; do
    say "$step: start (try $try)"
    setsid "$0" "$step" & local pid=$! hot=0
    while kill -0 $pid 2>/dev/null; do
      local m; m=$(margin)
      echo "$(date '+%F %T') $step margin=$m $(nvidia-smi -i $GPU --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader)" >> "$D/thermal.log"
      if [ -n "$m" ] && [ "$m" -le 4 ] 2>/dev/null; then
        hot=1; say "$step: THERMAL STOP (margin $m C) - killing, will resume after cooling"
        kill -TERM -- -$pid 2>/dev/null; screen -S mistralv -X quit 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null
        gpu1_clean; break
      fi
      sleep 3
    done
    wait $pid; rc=$?
    if [ $hot = 0 ]; then
      [ $rc = 0 ] && { touch "$D/$step.done"; say "$step: done"; return 0; }
      say "$step: FAILED (exit $rc) - see $D/${step}_*.log"; return 1
    fi
    until [ "$(margin)" -ge 25 ] 2>/dev/null; do sleep 30; done
    wait_lock "$step (retry)"
  done
  say "$step: gave up after 8 thermal stops"; return 1
}
wait_lock() { locked && return 0; say "waiting for the GPU $GPU clock lock (persistence Enabled) before $1"; until locked; do sleep 60; done; say "clock lock seen"; }

sort -u papers_acl.txt $TEST > "$PAPERS"
say "=== chain start: $(wc -l < "$PAPERS") papers, GPU $GPU only ==="
wait_lock s1; guarded s1 || exit 1
guarded s2 || exit 1
wait_lock s3; guarded s3 || exit 1
guarded s4 || exit 1
wait_lock s5; guarded s5 || exit 1
wait_lock s6; guarded s6 || exit 1
wait_lock s7; guarded s7 || exit 1
say "=== chain done: $D/s7_report.log, ft_runs/verify/compare_v50.json, $D/s7_aikg.log ==="
