#!/usr/bin/env bash
# Session 39 (2026-11-12): scale-up TEACHER = Qwen3-30B-A3B W4A16 via vLLM on every new paper,
# same setup as the legacy qwen3-30b-a3b-v2(-g2) runs: recall prompt, v2 entity lists, guards v2,
# slug qwen3-30b-a3b-v2-g2 (the legacy papers already have that slug from make_guard_variant.py).
# Waits for acquisition and curve-point-0 training, fixes the split (scale_finalize.py), then two
# lanes (GPU0 / GPU1), each with half of the new TEST papers first, then half of the new train
# papers. Resumes (ft_generate skips answered paragraphs). Log: scale_driver.log
set -u
cd /home/dev1/project_scinex || exit 1
PY=/home/dev1/envsam/bin/python; VPY=/home/dev1/envvllm/bin/python; LOGS=/home/dev1/scinex_runs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/scale_driver.log"; }
say "teacher: waiting for acquisition and p0 training"
until grep -q "=== scale acquisition done ===" "$LOGS/scale_acquire_driver.log" 2>/dev/null; do sleep 120; done
while pgrep -f 'run_scale_p0[.]sh' > /dev/null; do sleep 120; done
if [ ! -f ft_data/scale/split.json ]; then
  "$PY" scale_finalize.py --n-test 1000 --curve 1500 3000 2>&1 | tee -a "$LOGS/scale_driver.log" \
    || { say "teacher: scale_finalize FAILED - stop"; exit 1; }
fi
NEWTEST=$(grep -vxF -f papers_acl.txt scale/test_papers.txt); NEWTRAIN=$(grep -vxF -f papers_acl.txt scale/train_papers.txt)
echo "$NEWTEST" | awk 'NR%2==1' > "$LOGS/scale_teacher_half0.txt"; echo "$NEWTEST" | awk 'NR%2==0' > "$LOGS/scale_teacher_half1.txt"
echo "$NEWTRAIN" | awk 'NR%2==1' >> "$LOGS/scale_teacher_half0.txt"; echo "$NEWTRAIN" | awk 'NR%2==0' >> "$LOGS/scale_teacher_half1.txt"
lane() {
  say "teacher gpu$1: $(wc -l < "$LOGS/scale_teacher_half$1.txt") papers started"
  if CUDA_VISIBLE_DEVICES=$1 "$VPY" ft_generate.py --backend vllm \
      --base-model RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16 --model-slug qwen3-30b-a3b-v2-g2 \
      --teacher gptoss-20b --extractor fixed --prompt-variant recall --entity-dir output/acl_v2 --guards v2 \
      --batch-size 64 --papers-from "$LOGS/scale_teacher_half$1.txt" --ingest --ingest-python "$PY" \
      >> "$LOGS/scale_teacher_gpu$1.log" 2>&1; then say "teacher gpu$1: finished"; else say "teacher gpu$1: FAILED (exit $?) - re-run resumes"; return 1; fi
}
lane 0 & p0=$!; lane 1 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 -eq 0 ] && [ $r1 -eq 0 ] && touch "$LOGS/scale_teacher.done" && say "=== teacher done on all new papers ===" || say "teacher incomplete (rc $r0/$r1)"
