#!/usr/bin/env bash
# ONTOLOGY-FREE students (Session 38; the professor's step 2: "finetune the LLM to understand
# the domain ontology and vocabulary"). Training data = ft_data/acl_noont_me{50,30}: identical
# to acl_maxempty{50,30} except that the extraction system prompt carries NO ontology (only
# its name; ft_common.noont_system_prompt). Inference uses the same prompt
# (ft_generate --prompt-variant noont), on the 41 held-out ACL papers of fold 0.
#
#   GPU0: qwen3-8b noont-me50 -> base qwen3-8b with the noont prompt (control: no finetuning)
#         -> llama-3.1-8b noont-me50 -> qwen3-4b noont-me50
#   GPU1: qwen3-8b noont-me30 -> gemma-3-12b noont-me50 -> phi-4 noont-me50
#   then ft_compare over all of them + the ontology-in-prompt me50/me30 students + teacher.
# Every step is skipped if its output exists, so re-running resumes. A failed training skips
# only that model's generation; the lane goes on.
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python
LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/noont_driver.log"; }
EVALF=ft_data/acl/fold0/eval_papers.txt
mapfile -t EVAL < "$EVALF"
GEMMA_T='re:.*language_model.*\.(q|k|v|o|gate|up|down)_proj'

train() {  # train <gpu> <run> <base> <data> [extra train args...]
  local gpu=$1 run=$2 base=$3 data=$4; shift 4
  if [ -f "ft_runs/$run/run_manifest.json" ] && [ -d "ft_runs/$run/adapter" ]; then
    say "gpu$gpu: $run already trained - skip"; return 0; fi
  say "gpu$gpu: train $run ($base on $data) started"
  if CUDA_VISIBLE_DEVICES=$gpu "$PY" ft_train_qlora.py --data "$data" --base-model "$base" \
      --run-name "$run" --epochs 1 --save-steps 200 "$@" >> "$LOGS/noont_train_$run.log" 2>&1; then
    say "gpu$gpu: train $run finished"
  else
    say "gpu$gpu: train $run FAILED (exit $?) - see $LOGS/noont_train_$run.log"; return 1; fi
}

gen() {  # gen <gpu> <slug> <model args...>
  local gpu=$1 slug=$2; shift 2
  if [ -f "$LOGS/noont_gen_$slug.done" ]; then say "gpu$gpu: gen $slug done before - skip"; return 0; fi
  say "gpu$gpu: gen $slug started"
  if CUDA_VISIBLE_DEVICES=$gpu "$PY" ft_generate.py "$@" --model-slug "$slug" --extractor fixed \
      --teacher gptoss-20b --prompt-variant noont --batch-size 8 --ingest \
      >> "$LOGS/noont_gen_$slug.log" 2>&1; then
    touch "$LOGS/noont_gen_$slug.done"; say "gpu$gpu: gen $slug finished"
  else
    say "gpu$gpu: gen $slug FAILED (exit $?) - see $LOGS/noont_gen_$slug.log"; return 1; fi
}

student() {  # student <gpu> <run> <slug> <base> <data> [extra train args...]
  local gpu=$1 run=$2 slug=$3 base=$4 data=$5; shift 5
  train "$gpu" "$run" "$base" "$data" "$@" && gen "$gpu" "$slug" --adapter "ft_runs/$run"
}

lane0() {
  student 0 qwen3-8b-acl-f0-noont-me50 qwen3-8b-acl-noont-me50-cv Qwen/Qwen3-8B ft_data/acl_noont_me50/fold0
  gen 0 qwen3-8b-acl-base-noont --base-model Qwen/Qwen3-8B --papers-from "$EVALF"
  student 0 llama31-8b-acl-f0-noont-me50 llama31-8b-acl-noont-me50-cv meta-llama/Meta-Llama-3.1-8B-Instruct ft_data/acl_noont_me50/fold0
  student 0 qwen3-4b-acl-f0-noont-me50 qwen3-4b-acl-noont-me50-cv Qwen/Qwen3-4B ft_data/acl_noont_me50/fold0
}
lane1() {
  student 1 qwen3-8b-acl-f0-noont-me30 qwen3-8b-acl-noont-me30-cv Qwen/Qwen3-8B ft_data/acl_noont_me30/fold0
  student 1 gemma3-12b-acl-f0-noont-me50 gemma3-12b-acl-noont-me50-cv google/gemma-3-12b-it ft_data/acl_noont_me50/fold0 --target-modules "$GEMMA_T"
  student 1 phi4-acl-f0-noont-me50 phi4-acl-noont-me50-cv microsoft/phi-4 ft_data/acl_noont_me50/fold0 --target-modules qkv_proj o_proj gate_up_proj down_proj
}

say "=== noont run start ==="
lane0 & p0=$!
lane1 & p1=$!
wait $p0; wait $p1
say "lanes finished"
slugs=()
for s in qwen3-8b-acl-noont-me50-cv qwen3-8b-acl-noont-me30-cv qwen3-8b-acl-base-noont llama31-8b-acl-noont-me50-cv \
         qwen3-4b-acl-noont-me50-cv gemma3-12b-acl-noont-me50-cv phi4-acl-noont-me50-cv; do
  [ -f "$LOGS/noont_gen_$s.done" ] && slugs+=("$s")
done
mkdir -p ft_runs/noont
"$PY" ft_compare.py --extractor fixed --teacher gptoss-20b \
  --slugs qwen3-8b-acl-base-hf qwen3-8b-acl-ft-me30-cv qwen3-8b-acl-ft-me50-cv "${slugs[@]}" gptoss-20b \
  --papers "${EVAL[@]}" --out ft_runs/noont/compare_noont.json > "$LOGS/noont_compare.log" 2>&1
say "=== noont run done (${#slugs[@]} students). Table: $LOGS/noont_compare.log ==="
