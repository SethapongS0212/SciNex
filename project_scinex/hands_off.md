# Hands-Off Notes

## Session 36 — the unattended chain's post-mortem (a laptop sleep + MY deadline bug), then the real run starts (2026-10-23)

> User: *"continue on where we left off"*, and mid-session: *"i think we still have credits left to
> run the thing to run the model in openrouter, we might need to use that to run later after
> finishing it. notify me so we can work and discuss on openrouter afterwards"* (§36.6).

### 36.1 ⛔ POST-MORTEM: the §35.10 chain failed a step that had actually SUCCEEDED
The chain was launched at 13:25 on 2026-10-20 and its log jumps straight from `14:38:45` to
`2026-10-23 15:25:41  STEP B FAILED (torch not installed within 8 h)`. Two causes, compounding:
1. **The laptop slept at ~14:39 on 10-20 and woke at ~15:25 on 10-23.** `step_a` holds
   `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`, which prevents **idle** sleep only
   — it does **not** stop a lid close or a user-initiated sleep. The user went outside; the lid shut.
2. **⭐ My bug, and the expensive half:** `step_b` tested its absolute wall-clock deadline
   (`while time.time() < end`) **before** probing readiness, so on wake the loop fell straight
   through to `fail()` without ever re-checking. **torch had finished installing at ~14:38, before
   the sleep** — the step was already green. An 8 h deadline measured from the start cannot survive
   a 3-day pause.
**What was actually on disk on 10-23:** `torch 2.11.0+cu128` fully installed with all its
dependencies, `torch.cuda.is_available() True`, capability **(8, 9)**,
`NVIDIA GeForce RTX 4060 Laptop GPU`; `model-00001-of-00002.safetensors` **byte-exact**;
shard 2 stopped at 31 of 622 MB; and two 3-day-old `pip install torch` processes still alive, hung
on sockets the sleep had killed (they were killed by PID).

### 36.2 Three fixes to the chain (`ft_runs/chain_session35.py`)
1. **Probe before deadline, and count the deadline from the last PROGRESS, not from the start**
   (`step_b`): sleeping now only *pauses* the chain. Same bug class fixed in `wait_for_snapshot`,
   which now gives up only after 90 min with no new bytes.
2. **`pip_busy()` ignores a pip older than 90 minutes.** A pip hung on a dead socket would
   otherwise have made the chain wait on it forever.
3. `numpy==2.4.4` added to the dep list (torch warns `Failed to initialize NumPy` without it).
**Lesson, for any unattended runner on a laptop:** every wait must be *progress*-based, and the
readiness check must come first. Also expect sleep: a keep-awake request does not stop a lid close.

### 36.3 ⭐ The link is now ~6.8 MB/s — the reason for picking the 1.7B is GONE
Measured against huggingface.co on 10-23: **6,813,384 B/s**, i.e. **~20x the 145–470 KB/s of
10-20** (§35.5). At this speed `Qwen/Qwen3-4B-Instruct-2507` (8.05 GB) is a ~20-minute download, so
the bandwidth argument that made the user choose the smaller student no longer applies.
**The student for the run in flight is still `Qwen/Qwen3-1.7B` as decided** — but switching to (or
adding) the 4B would restore comparability with the §33/§34 numbers. Put to the user; open
decision 6 in `HANDOFF_NEXT_SESSION.md`.

### 36.4 Setup finished, stack verified on sm_89
Shard 2 downloaded to byte-exact (622,329,984 bytes; snapshot **3.80 GiB** total), the pinned deps
installed into the venv, and the whole stack checked:
`torch 2.11.0+cu128 | cuda True | cap (8, 9) | bnb 0.49.2 | peft 0.21.1 | transformers 5.5.4 |
numpy 2.4.4 | VRAM 8.0 GiB`.

### 36.5 ⭐ SMOKE TESTS PASS — and sm_89 does NOT hit the sm_120 attention trap
| smoke | result |
|---|---|
| `ft_generate --limit 2` (osmotic2026, CEO, batch 2) | 2 paragraphs in **0.7 min**, **peak VRAM 2.3 GB**; valid JSON in the right schema (para 0: a `designedFor` triple with a real `source_sentence`; para 1: a correct `{"triples": []}` abstention) |
| `ft_train_qlora --max-steps 2` (teacher-only data) | 2 steps in 28 s = **~14 s/step**, **peak VRAM 2.49 GB**, LoRA **17,432,576 trainable / 1,738,007,552 total (1.00%)**, loss 1.78, adapter written |
- **The §33.3 SDPA→MATH blow-up does not happen here.** On sm_120 one layer peaked at 7.27 GB;
  training the 1.7B on sm_89 peaked at **2.49 GB total**, so a real (flash) kernel is being used.
  `fast_attention` still picks deliberately, but this machine does not depend on the cuDNN work-around.
- ⚠ **The 14 s/step figure is pessimistic**: both smokes were run by hand from a windowless shell,
  i.e. subject to EcoQoS (§34.5). `run_ft_pipeline` opts itself and its children out, so the real
  rate should be better. At 14 s/step, 822 steps = ~3.2 h per variant.
- Both smoke artefacts (`output/osmotic2026/kg/relation/qwen3-1.7b-smoke/`,
  `ft_runs/qwen3-1.7b-smoke/`) were deleted; the chain re-runs its own copies of these two checks.

### 36.6 ▶ OPEN, user-raised: OpenRouter credit for the evaluation models
User, this session: credits are probably left on OpenRouter, to be used **after** the local run
finishes, and they want to be notified and to discuss it then. Where it would actually go — this is
exactly the project's one open evaluation gap:
- **Q&A (Part 3) needs an independent question writer + answerer** (open decision 1, §34.13). The
  pipeline default uses the base student itself, which makes every Q&A number a PIPELINE TEST, not a
  result. An OpenRouter model from outside the training data would make Part 3 reportable.
- **Second candidate: settling the qwen3-235b sentinel question** (§34.3/§34.10) — re-calling ~20
  sentinel paragraphs costs about $0.01 and would tell us whether its Table 10 coverage and C6 are
  real abstention or failed calls. Checking the OpenRouter activity log is free and comes first.
- ⛔ **Unchanged:** Claude is never called through an API here, and no OpenRouter credit is ever
  spent on Claude (§34.8). The C1/C3/C4 judge stays Claude-in-session via `blind_judge.py`.
- ⚠ Not needed for the run in flight: training, generation and `ft_compare` are entirely local, and
  Q&A grading is deterministic (§34.9).

### 36.7 Chain relaunched — gates green, the long runs are in flight
Relaunched hidden via WMI at **15:38** (same command as §35.10). This time every gate passed in
14 seconds, because the work was already done:
```
15:38:01  A ok: keep-awake held, EcoQoS opt-out applied
15:38:01  E ok: snapshot complete, 3.80 GiB, every file byte-exact
15:38:03  B ok: torch 2.11.0+cu128 importable
15:38:03  C ok: training/generation deps installed
15:38:15  D ok: stable torch on sm_89, bitsandbytes + peft import
15:38:15  F  (smoke generation running)
```
and then, confirmed on the log:
```
15:43:19  F ok: 2 paragraphs generated through the real extraction path (smoke slug deleted)
15:45:48  G ok: 2 optimizer steps ran and an adapter was written (smoke run deleted)
15:45:48  H starting: teacher-only variant
          base_gen -> kg/relation*/qwen3-1.7b-base-hf/  10 paper x ontology runs, 348 paragraphs
```
→ **H** (teacher-only: `base_gen → train → ft_gen → compare`) → **I** (consensus:
`train → ft_gen → compare`). Budget at the smoke rates: ~3 h training per variant plus ~1–2 h per
generation pass over 348 paragraphs (5 papers × 2 ontologies), three passes — call it 10–12 h.
Slugs the pipeline derives (checked against `run_ft_pipeline.py:98-99`, so the `blind_judge` command
in `HANDOFF_NEXT_SESSION.md` matches): `qwen3-1.7b-base-hf`, `qwen3-1.7b-ft-cv`,
`qwen3-1.7b-ft-cons-cv`; run dirs `ft_runs/qwen3-1.7b-f0` and `ft_runs/qwen3-1.7b-f0-cons`.
⚠ **If the lid closes again the chain pauses and resumes, but a CUDA context may not survive the
sleep**: a training step in flight can die with the GPU context, and only the completed pipeline
steps are protected by `.done` markers. Keep the laptop plugged in and open for the long runs, or
set the AC lid action to "do nothing".

### 36.8 ⛔ MEASURED: raising the generation batch 4 → 6 made it 2.85× SLOWER. Batch 4 is the setting.
A batch-size experiment, run and then **reverted**. Keep `--gen-args=--batch-size=4`.

**Why it was tried.** The first `base_gen` at batch 4 (the Session 33 setting) looked wasteful:
`csysguard2024/ceo` reached `12/57` in **8.0 min** (0.67 min/paragraph) at **peak VRAM 3.3 GB of 8**
with the GPU **39% busy at 16 W**. 4-bit decode at small batch is bound by weight bandwidth, so
batching usually comes close to free. KV cache here is **112 KiB/token** (28 layers × 8 KV heads ×
128 head_dim × 2 for K+V × 2 bytes), so the worst case — a p50 3.2k prompt plus the full 3,072-token
budget — projected 4.0 GiB of KV at batch 6 against 5.4 GiB at batch 8; 6 looked like the safe gain.

**What actually happened — a clean A/B on the SAME 12 paragraphs of the same paper:**
| setting | same 12 paragraphs | peak VRAM (script) | resident (nvidia-smi) |
|---|---|---|---|
| **batch 4** | **8.0 min** | 3.3 GB | 3.9 GB |
| batch 6 | **22.8 min** | 5.5 GB | **7.1 GB of 8** |
Per-batch, batch 6 went `6/57 (2.9 min)` → `12/57 (22.8 min)`: a single batch took **19.9 minutes**.
Over 24 paragraphs it averaged **1.27 min/paragraph vs 0.67** at batch 4. On 348 paragraphs that is
~7.4 h per pass instead of ~3.9 h — the "optimisation" would have cost ~10 h over three passes.

**Two mechanisms, and the data does not fully separate them:**
1. **VRAM pressure → WDDM paging (most likely dominant).** 7.1 GB resident of 8 GB is exactly where
   Windows starts paging VRAM to system RAM, which does **not** raise OOM — the run just crawls
   (the §33.3 trap). My 5.2 GiB estimate ignored the torch caching allocator, fragmentation and the
   desktop's own reservation, so the real figure was ~2 GB higher than predicted.
2. **Straggler gating.** A batch finishes only when its LONGEST sequence does, and base-model reply
   lengths are heavy-tailed (up to the full 3,072-token budget). The bigger the batch, the more
   likely one rambling reply holds up the rest. This cannot explain a 2.85× gap on its own.
**Lesson: on an 8 GB Windows card, predict VRAM from `nvidia-smi` residency, not from a KV-cache
calculation, and A/B any batch change on identical paragraphs before trusting it.** Batch 2 was not
tested; batch 4 is both measured-good here and the Session 33 setting, so it is what the runs use.

⚠ Both partial outputs from this experiment were deleted (12 paragraphs at batch 4, then 24 at
batch 6) so the whole corpus is generated at ONE batch size: with left padding and greedy decoding
the batch size can flip a token, and base / teacher-only / consensus must be compared on identical
settings. Cost of the experiment: ~40 min of run time.

⛔ **`--max-new-tokens` stays at the 3,072 default.** Lowering it would cap the slow batches, but the
base model rambles (~1k-token replies) while a finetuned model emits short JSON, so a tighter budget
would truncate the BASE model's replies into invalid JSON and flatter the finetuned one — biasing the
very comparison this run exists to make.

✔ `ft_generate` writes `responses.jsonl` **incrementally** (verified: 12 lines on disk at `12/57`)
and `--resume` is on by default, so a crash or a sleep costs at most one batch.

⚠ **Reading `ft_runs/_pipeline_qwen3-1.7b-f0/pipeline.log`:** it is appended across launches, so it
contains all three `base_gen` attempts interleaved — the first batch-4 try (12 paragraphs), the
batch-6 experiment (24), and the batch-4 run that counts (from **16:29:45**). Only the last one
produced the `output/*/kg/relation*/qwen3-1.7b-base-hf/` files on disk; the first two were deleted.
The batch-4 rate reproduced exactly on relaunch (`4/57 2.9 min`, `8/57 3.0 min`, peak 3.3 GB), which
is the evidence that the 2.85× gap was batch 6 and not drift.

**Honest budget for the whole chain, from these measurements:** base_gen 348 paragraphs ≈ 3.9 h ·
train 822 steps at ~9–14 s/step ≈ 2–3 h · ft_gen faster than base_gen (a finetuned model emits short
JSON instead of rambling) ≈ 1.5–2 h · then the consensus variant's train + ft_gen again ≈ 4–5 h.
**Total ≈ 13–17 h.** The lever that would actually shorten it is fewer eval papers (`--papers
strabismus2026 osmotic2026` = 148 paragraphs instead of 348), not a bigger batch.

### 36.9 ⭐ base_gen measured end to end, and the untuned Qwen3-1.7B BASELINE (fold 0, 5 eval papers)
`base_gen` ran 2026-10-23 16:29:45 → still finishing at 04:59 on 10-24, i.e. **~12.5 h for 311 of
348 paragraphs**. Per paper × ontology, from the pipeline log (cumulative timer differenced):
| job | paragraphs | minutes | min/paragraph |
|---|---|---|---|
| csysguard2024/ceo | 57 | 99.8 | 1.75 |
| linkpred2015/ceo | 29 | 89.6 | 3.09 |
| osmotic2026/ceo | 23 | 31.7 | 1.38 |
| strabismus2026/ceo | 51 | 179.3 | 3.52 |
| textaug2023/ceo | 14 | 35.7 | 2.55 |
| csysguard2024/scinex | 57 | 120.2 | 2.11 |
| linkpred2015/scinex | 29 | 64.4 | 2.22 |
| osmotic2026/scinex | 23 | 50.9 | 2.21 |
| strabismus2026/scinex | 28 of 51 | 76.0 | 2.71 |
**Real throughput: 2.40 min/paragraph at batch 4, peak VRAM 4.2 GB.**
⛔ **Correction to §36.8:** the 0.67 min/paragraph quoted there (and the "~3.9 h per pass" built on
it) came from the FIRST 12 paragraphs of the first paper and is **3.6× too optimistic**. A
generation pass over 348 paragraphs costs **~14 h**, not ~4 h. Within-job variance is large — single
batches take 19+ min when one reply runs to the full 3,072-token budget — so **any throughput figure
taken from the first few batches of a run is worthless**; difference the cumulative timer across
whole jobs instead.

**⭐ The baseline the finetuned models must beat** (`ft_compare.py --slugs qwen3-1.7b-base-hf
gptoss-120b`, the 5 fold-0 eval papers; **CEO is complete at 174/174 paragraphs**, scinex was still
running at 137/174 so its row is partial and NOT quotable):
| ontology | run | valid JSON | truncated | proposed → kept | kept rate | coverage | agree w/ teacher P / R |
|---|---|---|---|---|---|---|---|
| CEO | **qwen3-1.7b-base-hf** | 87.9% | **18** | 366 → 123 | **33.6%** | 20.7% | 13.8% / 8.9% |
| CEO | gptoss-120b (teacher) | 98.3% | 0 | 220 → 180 | **81.8%** | 42.0% | (teacher) |
| scinex *(partial, 137/174)* | qwen3-1.7b-base-hf | 88.3% | 15 | 329 → 120 | 36.5% | 19.0% | 6.7% / 5.6% |
| scinex | gptoss-120b (teacher) | 100.0% | 0 | 117 → 93 | 79.5% | 20.9% | (teacher) |
Reading it:
- **The untuned 1.7B lands at 33.6% kept, almost exactly where the untuned 4B did** (34.9% CEO in
  §33.5/§35.4, on 2 papers). The guards reject two thirds of what either base model proposes, and
  one less capable model generation did not change that much.
- **18 of 174 base replies (10.3%) hit the 3,072-token limit**; the teacher truncates 0. Those
  runaway replies are also what makes generation slow, and they are the concrete reason
  `--max-new-tokens` was NOT lowered (§36.8): a tighter budget would corrupt the base model's JSON
  and flatter the finetuned model in the very comparison being run.
- The base model proposes **more** than the teacher (366 vs 220) and keeps **less** (123 vs 180) —
  the over-extraction the finetuning is meant to fix, same shape as §33.5.

### 36.10 ⭐⭐ FIRST REAL FINETUNING RESULT — teacher-only Qwen3-1.7B: format and precision LEARNED, coverage COLLAPSED
Teacher-only variant finished at **13:02 on 2026-10-24** (`H ok`). Timings for the whole run:
`base_gen` 16:29:45 → 06:38 (**~14.1 h**, 348 paragraphs, 2.40 min each) · `train` 06:38 → 11:54
(**315.9 min**, 822 steps ≈ 23 s/step, loss **4.45 → ~0.10**, final points 0.03/0.08/0.14) ·
`ft_gen` 11:54 → 13:02 (**67 min**, 348 paragraphs, **0.19 min each — 12.5× faster than the base
model**, because the finetuned model stops rambling) · `compare` seconds.

**`ft_compare.py --slugs qwen3-1.7b-base-hf qwen3-1.7b-ft-cv gptoss-120b`, 5 fold-0 eval papers,
174 paragraphs per ontology (census, not a sample):**
| ontology | run | valid JSON | truncated | proposed → kept | kept rate | coverage | vs teacher P / R |
|---|---|---|---|---|---|---|---|
| CEO | qwen3-1.7b-base-hf | 87.9% | 18 | 366 → 123 | 33.6% | 20.7% | 13.8% / 8.9% |
| CEO | **qwen3-1.7b-ft-cv** | **100.0%** | **0** | 45 → 29 | **64.4%** | **4.6%** | **72.4%** / 12.8% |
| CEO | gptoss-120b (teacher) | 98.3% | 0 | 220 → 180 | 81.8% | 42.0% | (teacher) |
| scinex | qwen3-1.7b-base-hf | 87.9% | 20 | 434 → 155 | 35.7% | 21.3% | 6.5% / 7.5% |
| scinex | **qwen3-1.7b-ft-cv** | **100.0%** | **0** | 33 → 21 | **63.6%** | **3.4%** | **66.7%** / 15.1% |
| scinex | gptoss-120b (teacher) | 100.0% | 0 | 117 → 93 | 79.5% | 20.9% | (teacher) |

**What finetuning bought (both ontologies, same direction):**
1. **Output format: solved.** Valid JSON 87.9% → **100%**, truncated replies **18 → 0**. The base
   model's runaway generations are gone entirely.
2. **Guard pass rate: nearly doubled**, 33.6% → **64.4%** CEO (35.7% → 63.6% scinex), against the
   teacher's 81.8% / 79.5%. The invented predicates and malformed triples largely stop.
3. **Agreement with the teacher, when it does extract: 13.8% → 72.4% precision** (6.5% → 66.7%
   scinex). What it emits is mostly what the teacher would have emitted.

**⛔ What it cost — the §34.11 failure mode, worse than on the 4B:**
- **Coverage 20.7% → 4.6%** (teacher 42.0%): it returns `{"triples": []}` for **162 of 174
  paragraphs (93.1%)** on CEO and 165 of 174 (94.8%) on scinex. Not the §33.5b total collapse —
  it does extract on 12 and 9 paragraphs respectively — but close.
- **Recall against the teacher is only 12.8% / 15.1%**: it finds about an eighth of the teacher's
  triples. Absolute volume 29 kept triples vs the teacher's 180.
- Session 34's 4B kept 86% at 8-of-44 paragraphs; this 1.7B keeps 64% at 4.6% coverage. Same
  disease, further along.

**Cause, already identified before the run (§36.6): 402 of 619 extraction targets (65%) are
empty.** Token-level loss (the §33.5b fix) stops the model abstaining on *everything*, but it cannot
undo a 2:1 prior in favour of abstention. The student learned the teacher's precision and its
silence.

**Prepared fix, NOT yet run (open decision 5, the user's call):**
`build_ft_data.py --eval-fold 0 --max-empty-frac 0.3 --out ft_data/maxempty30` → built, giving
**310 extraction examples (164 CEO / 146 scinex) with 93 empty = 30%**, 309 empty examples capped
away; 2,976 train examples → **744 optimizer steps**. It halves the extraction examples but fixes
the abstain/extract prior. (Its `train.jsonl` is correctly ignored by the `ft_data/**` patterns from
§35.8.)
⚠ **The consensus variant, running now, will probably under-extract too:** its data is 380/619 = 61%
empty, only 3.6 points better than teacher-only. It is still worth finishing — teacher-only vs
consensus is the experiment of record — but the abstention prior, not the consensus additions, is
what dominates this student.

### 36.11 ⛔⛔ THE BIG ONE: the EcoQoS opt-out was applied to a TRAMPOLINE process — everything ran 8× slow
**Symptom.** The consensus training started at **48.4 s/step** (ETA 11 h) while the teacher-only
training had managed 23 s/step. `nvidia-smi` during it: **GPU util 0%, 9.75 W, SM clock 930 of
3105 MHz, "Idle: Active"**, temperature 54 °C and **no** throttle reason flagged — i.e. the GPU was
not throttled, it was **starved** by one CPU thread parked on an efficiency core. That is the §34.5
EcoQoS signature, which `run_ft_pipeline` was supposed to have fixed.

**Cause.** On this machine the venv was created by the Windows **Python Install Manager**, and
`.venv\Scripts\python.exe` is a **trampoline**: it re-execs the real interpreter
(`AppData\Local\Python\pythoncore-3.12-64\python.exe`) as a **child**. The process tree was:
```
31492 .venv launcher (chain)       -> 20980 pythoncore   REAL chain
  24828 .venv launcher             ->  5460 pythoncore   REAL run_ft_pipeline
    24948 .venv launcher           ->  8124 pythoncore   REAL ft_train_qlora  <- does the work
```
`run_ft_pipeline.py` called `no_power_throttling(proc.pid)` on **24948, the launcher** — a process
that does nothing — while 8124, the trainer, kept Windows' throttling. Session 34 never saw this
because that laptop's venv was a python.org install, where `.venv\Scripts\python.exe` IS the
interpreter.

**Fix, and the measurement.** Applying the opt-out to the real PIDs live (no restart, as in §34.5):
| | step time | GPU util | power | SM clock |
|---|---|---|---|---|
| throttled | **48.4 s** | 0% | 9.8 W | 930 MHz |
| opted out | **~9 s** (6–12 s, varies with example length) | 63–71% | 26–56 W | 1425–2475 MHz |
**≈8× on the worst steps**, and the consensus training's ETA fell from **11 h to ~2 h**.

**What it cost the runs already done** (speed only — greedy decoding is deterministic, so **no result
is invalidated and nothing needs re-running**):
- teacher-only `train`: **315.9 min**, would have been ~90–120 min;
- teacher-only `base_gen`: **~14.1 h** for 348 paragraphs (2.40 min each), should be a few hours;
- so the ~21 h spent to 13:00 on 10-24 was perhaps 4–8× more than necessary.

**Two fixes landed:**
1. **`run_ft_pipeline.no_power_throttling_tree(pid)`** (new) opts out the PID **and every
   descendant**, sweeping 3× with a 6 s settle because children appear a moment after launch; it
   uses `psutil` (pinned in `requirement.txt`) and falls back to a WMI parent/child query. `sh()`
   now calls it in a thread for every step it spawns. **Any future run gets this automatically.**
2. **`ft_runs/ecoqos_watchdog.py`** (new, machine-local) for a pipeline that is ALREADY running and
   therefore still holds the old code: it sweeps every python process whose command line mentions
   `chain_session35` / `run_ft_pipeline` / `ft_generate` / `ft_train_qlora` / `ft_compare` once a
   minute, so each newly spawned step is opted out within 60 s, and it exits by itself ~5 min after
   the chain disappears. Launched detached at 13:08; its first sweep opted out all 6 PIDs.

**Lesson, worth more than this run:** when a fix depends on a PID, verify the PID is the process
doing the work. `Get-CimInstance Win32_Process | Select ProcessId,ParentProcessId,CommandLine` shows
the trampoline immediately, and `nvidia-smi` reporting **0% util at <10 W with no throttle reason**
means the GPU is being starved by the CPU side, not throttled itself.

### 36.12 ⭐⭐⭐ BLIND C1/C3/C4 CENSUS, CEO — finetuning takes a 1.7B student from 29% to 85% relation correctness, i.e. TEACHER LEVEL
Run exactly as designed in §34.8: `blind_judge.py build` pooled every kept triple of all four
extractors on the 5 fold-0 eval papers — **355 model-triples → 315 unique items** (40 duplicates
judged once), shuffled, ids salted, `key.csv` **not opened until after `collect`** — judged in
session by claude-opus-5-5 across 13 batches, then `collect` + `report`.

| extractor | judged | C1 concept | C3 specificity | C4 relation | C4 strictly CORRECT |
|---|---|---|---|---|---|
| qwen3-1.7b-base-hf | 107 | 59.8 [52.3, 67.8] | 52.3 [43.0, 61.7] | **29.4** [22.0, 37.9] | 22.4 |
| **qwen3-1.7b-ft-cv** (teacher-only) | 27 | 90.7 [81.5, 98.1] | 81.5 [66.7, 92.6] | **83.3** [72.2, 94.4] | 74.1 |
| **qwen3-1.7b-ft-cons-cv** (consensus) | 41 | 90.2 [81.7, 97.6] | 87.8 [78.0, 97.6] | **85.4** [75.6, 93.9] | 78.0 |
| gptoss-120b (teacher) | 180 | 96.4 [94.4, 98.3] | 87.2 [82.2, 92.2] | **86.4** [83.1, 89.7] | 74.4 |

Verdict mix (the clearest way to read it):
| | C4 CORRECT | C4 PARTIAL | **C4 INCORRECT** |
|---|---|---|---|
| base | 24 | 15 | **68 = 63.6%** |
| teacher-only | 20 | 5 | **2 = 7.4%** |
| consensus | 32 | 6 | **3 = 7.3%** |
| teacher | 134 | 43 | **3 = 1.7%** |

**⭐ The headline: on per-triple quality the finetuned 1.7B students are statistically
indistinguishable from the 120B teacher.** C4 85.4 vs 86.4, C1 90.2 vs 96.4, and both finetuned
models beat the teacher on *strict* C4 (78.0 / 74.1 vs 74.4). Outright relation errors fall from
**63.6% of the base model's triples to ~7%**. A 1.7B student that has learned the ontology is a
different instrument from the same 1.7B model prompted.

**⚖ Teacher-only vs consensus — the experiment of record: NO significant difference in quality, but
consensus produces more.** C1 **−0.5 pts [−12.7, +12.4]**, C4 **+2.0 pts [−12.7, +16.8]** — both CIs
straddle zero at n=27 vs n=41, so quality is a tie. Consensus wins on everything countable:
**47 vs 29 kept triples (+62%)**, coverage **7.5% vs 4.6%**, guard kept rate **73.4% vs 64.4%**, C3
87.8 vs 81.5, strict C4 78.0 vs 74.1. **Recommendation: adopt the consensus variant** — same quality,
more of it — while noting the quality difference itself is not resolved at this sample size.

**⛔ The real limitation is unchanged, and the census makes it sharper:** the students are *precise
but nearly silent*. 29 and 47 kept triples against the teacher's 180, from 4.6–7.5% paragraph
coverage against 42.0%. High C1/C3/C4 on a handful of triples is not a usable knowledge graph. The
next lever is the already-built `--max-empty-frac 0.3` data (§36.10).

**Error types the census exposed** (useful for the next round of guards, and all from the base model
unless noted): metric-as-subject domain violations (`AUC evaluatedOn LTest`, `Decision Trees
evaluates F1 score`); hardware read as data (`cSysGuard trainedOn <Intel CPU>`, `evaluatedOn 16GB
DDR4 RAM`); hyperparameters as datasets (`max_features trainedOn sqrt`,
`use_pretrained_backbone trainedOn False`); **reference-list bleed** (`Hoonlor writtenBy Buyya`,
`Magableh affiliatedWith 'Int. Challis'`, `Snyder affiliatedWith <book title>`) — the boilerplate
guard is not catching merged bibliographies; **self-loops** (`Belvèze et al. affiliatedWith Belvèze
et al.`, `Decision Trees comparesAgainst Decision Trees`); **tautologies** (`SMOTE uses Synthetic
Minority Over-sampling Technique`); **fabricated objects absent from the sentence** (`RT-DETR
publishedIn Nature Machine Intelligence`, `Valente et al. affiliatedWith University of California,
San Francisco`); verb-phrase fragments as objects; and one **prompt-template leak**
(`RT1 evaluates '88.5 F1 on <Dataset>'` — the placeholder from the system prompt). The teacher's
43 PARTIALs are a different, milder failure: mostly loose typing (a metric or landmark as the object
of `uses`) and variant attribution (RT-DETR-L's score credited to RT-DETR).

⚠ **Caveats to state with these numbers:** single rater (claude-opus-5-5, in session, blind);
**not comparable with the claude-opus-5 C1–C6 tables** (Tables 10–12) which used a different judge;
n=27/41 for the finetuned models gives wide CIs; one fold and **CEO only** — the scinex census has
not been run; and the CI is a bootstrap over triples, which ignores that triples within a paper are
not independent, so read it as a lower bound on uncertainty.

### 36.13 Session close (2026-10-26): docs updated and everything pushed to GitHub
User: *"update all the md files. then upload the whole update and changes files to this github"*.

**Every markdown file reviewed and brought to the current state:**
| file | what changed |
|---|---|
| `hands_off.md` | §36.1–36.14 (this session): post-mortem, fixes, throughput, baseline, the result, the EcoQoS bug, the census |
| `Claude.md` | new STATUS block (2026-10-26) with the census table and the open recall problem |
| `HANDOFF_NEXT_SESSION.md` | rewritten: what is DONE, the result table, 5 ordered next steps with commands, decisions, traps |
| `results.md` | **Table 13** gained the consensus rows (+ corrected footnote); **Table 14** is the blind census; "what is missing" updated |
| `TASK.md` | the CURRENT TASK block replaced (it still described the abandoned RTX 5090 plan); Status line refreshed to 2026-10-26 |
| `SUMMARY_finetune_qa_2026-10-01.md` | new plain-language Session 36 section on top (the slide-ready summary for the user) |
| `SETUP_NEW_MACHINE.md` | new section: the EcoQoS trampoline trap + diagnostic, the lid/sleep warning, and the batch-size warning |
| `README.md` (root) and `project_scinex/README.md` | finetuning headline added to the status/headline blocks |
| `RESULTS_REPORT.md` | ⚠ banner: it is GENERATED and predates Part 2 — `build_results_report.py` has no Part 2 awareness, so the note points at `results.md` Tables 13–14. **Open item: teach the builder about Part 2 or that banner is the only hand-written block in a generated file.** |
| `PAPERS.md` | no change needed — the corpus is unchanged |

**Pushed to https://github.com/SethapongS0212/SciNex.** Public-repo audit before committing:
- The new `ontology_eval/_blind/ft-f0/ceo/batch_NN.txt` files quote **one source sentence per
  triple**. Checked against precedent: **64 equivalent triple-harness bundles with
  `SOURCE SENTENCE` lines are already tracked**, and `.gitignore` excludes only the
  *paragraph*-level bundles (`ontology_eval/*/*/paragraphs/*_batch_*.txt`), which print whole
  paragraphs. So committing the census is consistent with the established line.
- Per model-output directory: `responses.jsonl`, `triples.json`, `prompt_meta.json`,
  `system_prompt.txt`, `kg_stats.json` committed (the documented convention); `prompts.jsonl` and
  `kg.graphml` ignored. Verified with `git check-ignore` on every file of one directory.
- `ft_data/maxempty30/`: manifest + paper lists committed, `train.jsonl` and `parts/` ignored by the
  `ft_data/**` patterns fixed in §35.8.
- `ft_runs/qwen3-1.7b-f0{,-cons}/`: `run_manifest.json` committed, `adapter/` ignored.
- `req_no_torch.txt` added to `.gitignore` — it is derived from `requirement.txt` by a command the
  setup doc gives, so a committed copy could only go stale.
- `ft_runs/chain_session35.py` and `ft_runs/ecoqos_watchdog.py` committed. ⚠ The chain hardcodes this
  machine's paths (`C:\Users\cei008\...`); it is committed because §35.10/§36 describe it as the
  unattended runner, but it is machine-local and not meant to run elsewhere unedited.
- Confirmed absent from the commit: PDFs, `.env`/keys, parsed HTML, `prompts.jsonl`, adapters,
  `ft_data/**/train.jsonl`, pipeline `.done` markers.

### 36.14 Smoke steps are now skippable (`chain_smoke.done`)
The chain's own F/G children are **not** EcoQoS-opted-out — `run_ft_pipeline` opts out *its*
children (`run_ft_pipeline.py:116`), but the chain spawns F and G directly, so the smoke that took
0.7 min by hand took **4.7 min** under the chain. They now write
`ft_runs/_pipeline_qwen3-1.7b-f0/chain_smoke.done` and are skipped on a relaunch, which saves ~7
throttled minutes every time the chain restarts. The marker is gitignored by the existing
`ft_runs/_pipeline_*/*.done` rule. Delete it to force the smokes to run again (worth doing after any
change to the model, the data or the generation flags).

## Session 35 — new machine (RTX 4060 Laptop, 8 GB): clone set up, private-bundle data recovered from an older local copy, student = Qwen3-1.7B (2026-10-20)

> User: *"read all md files to catch up. then continue our process on here for now. this laptop is
> using 4070 so run the model that fits this laptop"*. The GPU is actually an **RTX 4060 Laptop GPU**
> (`nvidia-smi`: 8,188 MiB, compute capability **8.9**, driver 592.82 / CUDA 13.1), not a 4070 — the
> 8 GB ceiling is what matters and it is the same as the Session 33 laptop.
> **The RTX 5090 VM plan (§34.12) is NOT what happened; training runs on this laptop.**

### 35.1 The machine — Ada sm_89 is the FIRST machine here that needs no torch nightly
RTX 4060 Laptop GPU, 8 GB, **sm_89 (Ada)**, Windows 11 Home Single Language 10.0.26200.
- **Only Python 3.14.7 was installed** (via the new Python Install Manager, `py list`), and neither
  torch nor bitsandbytes publishes cp314 wheels. Installed the project's version with
  **`py install 3.12`** → 3.12.10, the same as Session 33. venv at `project_scinex/.venv`.
- **torch `2.11.0+cu128`, a STABLE release** (`--index-url .../whl/cu128`, cp312). sm_89 is covered by
  stable cu128 wheels, so **`--allow-dev-torch` is NOT needed on this machine** — unlike the sm_120
  RTX 5070/5090, which needed a nightly. The old "2.6.0+cu124" rule is still H100-vGPU-only.
- Windows, so the EcoQoS trap (§34.5) still applies to hidden launches; `run_ft_pipeline.py` already
  opts itself and its children out.
- The sm_120 SDPA→MATH attention trap (§33.3) is an sm_120/nightly finding; on sm_89 + stable torch
  flash SDPA is expected to be available. `ft_common.fast_attention(training=True)` covers either case
  — watch the `peak VRAM` / s/step lines in `pipeline.log` rather than assuming.

### 35.2 ⭐ The private bundle was not needed: an OLDER PROJECT COPY on this laptop held everything
`SciNex_private_bundle.zip` (§34.14) is not on this machine, but two pre-finetuning copies of the
project are: `C:\Users\cei008\Downloads\project_scinex` (its git log ends at *"Add PAPERS.md"*, i.e.
before Session 32) and `C:\Users\cei008\project_scinex` (an older ACL-corpus copy). The first one
carried every gitignored input. Copied into the clone (only gitignored paths; **`git status` stayed
clean afterwards**, verified):
| restored | count | from |
|---|---|---|
| corpus PDFs `papers/*.pdf` | **22** (the 20 + the 2 retired substitutes) | `papers/` |
| parsed papers `output/<id>/no-llm/` | **22** | `output/` |
| **stored prompts `output/<id>/kg/relation{,_scinex}/<model>/prompts.jsonl`** | **293** | `output/` |
| CS-NER gazetteer + per-paper entity CSVs `output/acl/` | 21 files, 2.1 MB | `output/acl/` |
So `enrich_entity_csv.py` did not have to be re-run and no PDF had to be re-downloaded (including
the 10 paywalled ones). **Check an older local copy before chasing the private bundle.**

### 35.3 ⭐ The Session 33 "3 usable training papers" limit is gone — 13 of 15 now contribute
The laptop in §33/§34 had no `prompts.jsonl` (gitignored, absent from the clone), so it rendered
paragraphs from HTML and lost `oxidecrack2025` (alignment 0.88 CEO / 0.71 scinex) and
`roadwaylight2018` (30 rendered vs 31 teacher paragraphs). With the teacher's stored prompts restored,
`load_paragraphs` prefers them (`ft_generate` passes `prefer_slugs=(teacher,)`), and
**every one of the 15 training papers has prompts == replies on BOTH ontologies** (11–45 paragraphs
each). Paragraph COUNTS pairing is not the same as the targets aligning, though: on the 0.9 alignment
floor  (both ontologies) and  CEO still drop out, so **13 of the 15
papers actually contribute** — see §35.6. The only count mismatch in fold 0 is `strabismus2026` scinex, 51 prompts vs **61** replies —
that is the known §30.8 double-writer incident (7 duplicated para_ids), which `read_replies` drops.
Also verified: for `strabismus2026` and `osmotic2026` the teacher's stored paragraph text is
**byte-identical to today's render** (51/51 and 23/23), so the stored `qwen3-4b-base-hf` replies, the
teacher's targets and any new run all sit on the same inputs.

### 35.4 Install check PASSES exactly (`SETUP_NEW_MACHINE.md` §7)
`ft_compare.py --slugs qwen3-4b-base-hf gptoss-120b --papers strabismus2026 osmotic2026`:
| ontology | run | proposed → kept | kept rate | expected |
|---|---|---|---|---|
| CEO | qwen3-4b-base-hf | 493 → 172 | 34.9% | 493 → 172 (34.9%) ✔ |
| CEO | gptoss-120b | 107 → 91 | 85.0% | 107 → 91 (85.0%) ✔ |
| scinex | qwen3-4b-base-hf | 464 → 164 | 35.3% | 464 → 164 (35.3%) ✔ |
| scinex | gptoss-120b | 70 → 57 | 81.4% | 70 → 57 (81.4%) ✔ |
So this machine reproduces the committed numbers from the stored replies.

### 35.5 ⛔ The binding constraint here is DOWNLOAD BANDWIDTH, not VRAM
Measured sustained throughput on this laptop's Wi-Fi: **~145–470 KB/s** (default route is the Wi-Fi
adapter, not the OpenVPN TAP, so it is the upstream link, not the VPN). Sizes that matter:
torch cu128 wheel **2.75 GB**; `Qwen/Qwen3-4B-Instruct-2507` weights **8.05 GB**;
`Qwen/Qwen3-1.7B` weights **3.88 GB** (3.28 GB + 0.59 GB shards). The documented 4B path therefore
meant ~20 h of downloading before any GPU work.
**Decision (user, 2026-10-20): student = `Qwen/Qwen3-1.7B`** ("smaller student") instead of the
Session 33 choice `Qwen/Qwen3-4B-Instruct-2507`.
- ⚠ **Not comparable with the §33/§34 numbers** (kept rate 38% → 86%, teacher 88%; 8 vs 21
  paragraphs): those are Qwen3-4B. A 1.7B student is weaker, so an under-extraction result on it
  cannot be separated from model size. State the model with every number.
- Qwen3-1.7B is a hybrid-thinking model; thinking is already off — `ft_common.render_chat` passes
  `enable_thinking=False`.
- The weights are fetched to **`C:\Users\cei008\hf_local\Qwen3-1.7B`** (a plain snapshot, OUTSIDE the
  repo so the public clone stays clean) and passed as `--base-model <that path>`; the HF id is
  `Qwen/Qwen3-1.7B`. Slugs follow the existing scheme: `qwen3-1.7b-base-hf`, `qwen3-1.7b-ft-cv`,
  `qwen3-1.7b-ft-cons-cv`.

### 35.6 ⭐ Both fold-0 datasets REBUILT on the full training set — and they are much bigger
Built here with the stored teacher prompts as the paragraph source for every paper
(`paragraph_source: output\<id>\kg\relation*\gptoss-120b\prompts.jsonl` in both manifests).
**13 of the 15 training papers contribute** (27 of 30 paper x ontology runs `ok`). Still excluded,
automatically, on the 0.9 alignment floor: `tripplanner2020` (0.89 CEO / 0.75 scinex) and
`vehiclemake2025` CEO (0.89) — the teacher's replies cite sentences that are not in the paragraph
they are attributed to. `oxidecrack2025` and `roadwaylight2018`, which the rendered-HTML build of
§33.4 lost, now align at **1.00** and are in.

| | Session 33/34 laptop (3 papers) | **here (13 papers)** |
|---|---|---|
| extraction examples | 168 (83 CEO / 85 scinex) | **619 (292 CEO / 327 scinex)** |
| teacher triples in the training targets | 192 | **495** |
| consensus additions (>=2 of gemma4-31b / qwen3-235b / ministral-14b) | +66 (+34%) | **+129 (+26%)** |
| teacher-empty paragraphs given a target by consensus | 7 | **22** |
| empty extraction targets | 87 -> 80 | **402 -> 380** |
| ontology / vocabulary cards | 700 / 1,612 | 700 / **1,966** |
| total train examples | 2,480 | **3,285** |
| optimizer steps (batch 1 x accum 8 x 2 epochs) | 620 | **822** |

Consensus skips this time: teacher already related the pair 4, relation conflict 12, `configures` 1,
sentence not in paragraph 0. `train.jsonl` sha256 — teacher-only `d0e3a2be…`, consensus `73dadf7c…`
(recorded in each `manifest.json`; the byte-identity regression of §34.4 no longer applies, since
both files are now built from 13 papers instead of 3).

⚠ **The empty-target share is now 402/619 = 65%** (was 87/168 = 52%). Empty targets are what drove
the §33.5b collapse and the §34.11 under-extraction, so this run is MORE exposed to that failure
mode, not less — the consensus variant only brings it to 380/619 = 61%. If the finetuned model
under-extracts again, `--max-empty-frac 0.3` (open decision 5) is the next lever, and it now has
more data to drop from.

### 35.7 Tokenizer dry run on Qwen3-1.7B — clean
`ft_train_qlora.py --data ft_data/fold0 --base-model C:\Users\cei008\hf_local\Qwen3-1.7B --dry-run`
(no GPU, no weights — only the tokenizer files, which download first):
| part | n | p50 | p95 | max | dropped (>6144) |
|---|---|---|---|---|---|
| extraction | 619 | 3,192 | 3,508 | **4,437** | 0 |
| ontology | 700 | 178 | 389 | 447 | 0 |
| vocabulary | 1,966 | 73 | 77 | 89 | 0 |
`chat template: {'template': 3285}` — every example split cleanly into prompt/target, **no
`fallback`** rows, so Qwen3-1.7B's template works with `ft_common.render_chat` as-is. `--max-seq-len
6144` keeps everything (the longest example is 4,437 tokens vs 3,821 on the 4B tokenizer).

### 35.8 ⛔ SECURITY/LICENSING FIX — `ft_data/consensus/fold0/train.jsonl` was NOT gitignored
`.gitignore` had `ft_data/*/train.jsonl`, which matches ONE directory level. The consensus variant
introduced in §34.4 writes one level deeper (`ft_data/consensus/fold0/train.jsonl`), so
`git check-ignore` did not match it and `git status` listed the file — **11 MB of full paragraphs
from the corpus, 10 of whose papers are paywalled, in a PUBLIC repo**, with
`SETUP_NEW_MACHINE.md` §8.4 telling the next session to run `git add -A`. The same applied to
`ft_data/consensus/fold0/parts/`.
- **Never committed** — verified with `git ls-files ft_data/` (only `manifest.json`,
  `eval_papers.txt`, `train_papers.txt`, `split.json` are tracked) and
  `git log --all -- ft_data/consensus/fold0/train.jsonl` (empty). A latent hole, not a leak.
- **Fixed:** the three patterns are now `ft_data/**/train.jsonl`, `ft_data/**/dev.jsonl`,
  `ft_data/**/parts/`, with a comment saying why. Re-verified with `git check-ignore -v` on both
  the nested and the flat path.
- **Lesson for any new nested output dir:** single-`*` ignore patterns silently stop matching. Check
  `git check-ignore -v <the new path>` whenever a script starts writing to a deeper directory.

### 35.10 ▶ UNATTENDED CHAIN LAUNCHED (user: *"start running what we can right now, i need to go outside"*)
Everything left was download-bound, so the remaining work was handed to one detached driver,
**`ft_runs/chain_session35.py`** (machine-local, untracked), started hidden via WMI at 13:25 so it
**outlives the assistant's session**:
```powershell
$repo = "C:\Users\cei008\Downloads\SciNex\project_scinex"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "`"$repo\.venv\Scripts\python.exe`" `"$repo\ft_runs\chain_session35.py`""
  CurrentDirectory = $repo; ProcessStartupInformation = $si }
```
Steps, each gated on the one before, logged to **`ft_runs/chain_session35.log`**:
| | step | note |
|---|---|---|
| A | keep-awake + EcoQoS opt-out | `SetThreadExecutionState` so the laptop cannot sleep mid-run, plus `run_ft_pipeline.no_power_throttling()` (§34.5) |
| B | wait for torch | polls; only runs `pip install torch` itself if NO pip is working, so it can never start a second 2.75 GB download |
| C | install the pinned training deps | minimal set — no pdfplumber/pymupdf/camelot/opencv/pandas/scipy; nothing on the train or generate path imports them |
| D | verify the stack | asserts CUDA up, **capability (8, 9)**, torch not a nightly, bitsandbytes + peft import |
| E | finish the Qwen3-1.7B snapshot | **runs in a thread from the start, in parallel with B/C**, since both are network-bound and the GPU is idle either way; `curl -C -` resume, **byte-exact** per-file size check against the HF `Content-Length` values |
| F | SMOKE 2 paragraphs | `ft_generate --limit 2` — proves 4-bit load, decode and VRAM headroom before committing to hours; throwaway slug deleted |
| G | SMOKE 2 optimizer steps | `ft_train_qlora --max-steps 2` — proves the training path; run dir deleted |
| H | teacher-only pipeline | `base_gen → train → ft_gen → compare` |
| I | consensus pipeline | `train → ft_gen → compare`, `--also-slugs qwen3-1.7b-ft-cv`, base_gen shared |

A failing step writes `STEP <x> FAILED` and the chain **stops instead of cascading**; an unexpected
exception writes `CHAIN CRASHED` plus the traceback. Re-launching resumes, because
`run_ft_pipeline`'s `.done` markers skip finished steps and `ft_generate --resume` is on by default.
- **Watch it:** `Get-Content ft_runs\chain_session35.log -Tail 20 -Wait`, and once step H starts,
  `ft_runs\_pipeline_qwen3-1.7b-f0\pipeline.log` for per-step s/step and peak VRAM.
- **Stop it:** kill the chain PID **and** its current child, by command line
  (`Get-CimInstance Win32_Process`) — stopping a parent does not kill the child (§34.5).
- ⚠ **One writer per file:** the chain's thread is the only thing allowed to `curl` into
  `C:\Users\cei008\hf_local\Qwen3-1.7B`. A second `curl -C -` on the same shard interleaves into one
  file and corrupts it. A leftover `curl` from the assistant's own earlier download had to be killed
  by PID first — stopping the task that spawned it did NOT kill it, the §34.5 trap again.
- State at launch: torch wheel ~870 MB of 2,626; shard 1 275 MB of 3,282; shard 2 9 MB of 594.

### 35.11 Status at the end of this session
**Done:** machine set up (Python 3.12.10, venv, stable torch 2.11.0+cu128 — §35.1) · all gitignored
inputs restored from an older local copy (§35.2) · 13 of 15 training papers now pair (§35.3) ·
install check green (§35.4) · both fold-0 datasets rebuilt, 619 extraction examples (§35.6) ·
tokenizer dry run clean, 822 steps (§35.7) · nested-`ft_data` gitignore hole fixed (§35.8) ·
`hands_off.md` / `Claude.md` / `HANDOFF_NEXT_SESSION.md` / `SETUP_NEW_MACHINE.md` / root `README.md`
updated.

**Not done when the session ended, and why:** nothing had been trained yet — the two large downloads
were still running (torch 2.75 GB at ~35%, Qwen3-1.7B weights 3.88 GB at ~280 MB), and no GPU work
can start before them. **The user left, so the rest was handed to the unattended chain in §35.10**,
which finishes the downloads, installs, verifies, smoke-tests and then runs both variants on its own.
Read `ft_runs/chain_session35.log` first on the next session: it says exactly how far it got.
- `beautifulsoup4`, `networkx`, `rdflib`, `transformers==5.5.4` and `jinja2` were installed into a
  side directory (`pip install --target`, put on `PYTHONPATH`) so the CPU-only steps — the install
  check, the paragraph comparison, both data builds and the tokenizer dry run — could all run while
  pip was still downloading torch. **Drop that `PYTHONPATH` once the venv has the real pinned set**
  (`req_no_torch.txt` + `requirements_kg.txt` + `requirements_ft.txt`); two transformers on one path
  is a debugging trap.
- The model is a plain directory snapshot at `C:\Users\cei008\hf_local\Qwen3-1.7B` (fetched with
  `curl`, so it could start before the venv existed), passed as `--base-model <path>`.

**Next:** finish the installs and confirm `torch.cuda.get_device_capability() == (8, 9)` → train
teacher-only, then consensus, on Qwen3-1.7B (822 steps each) → `ft_generate` on the 5 fold-0 eval
papers → `ft_compare` → blind judge census here (§34.8) with the `qwen3-1.7b-*` slugs. Exact commands
in `HANDOFF_NEXT_SESSION.md`. Budget for generation: **348 paragraphs per pass** (5 papers × 2
ontologies) × 3 passes (base, teacher-only, consensus); Session 33 took 186.7 min for 148 paragraphs
on a 4B at batch 4, so use `--gen-args=--batch-size=4` or higher for the 1.7B and expect hours per
pass. `ft_generate --resume` is ON by default, so an interrupted pass continues.

## Session 34 — why ONE teacher? Census of every stored model's replies as finetuning targets (2026-10-01)

> User: *"why did we only use one example result from models for finetuning? do we not have others
> also? if so we should use them"*. Offline analysis only: no model calls and no GPU. Every number comes from
> running the CURRENT `ft_common.guard_pass` (the one `build_ft_data.py` uses) over stored
> `responses.jsonl`; exact match = lowercased, whitespace-collapsed (s, p, o) on the same paragraph;
> loose = same predicate + subject and object substring either way.

### 34.1 Why one teacher (the Session 32 reasoning, now written down)
`build_ft_data.py` takes a single `--teacher` (default `gptoss-120b`). It was chosen on relation
correctness: C4 **86.2% CEO / 90.0% scinex** vs gemma4-31b 73.8 / 59.0, qwen3-235b 47.5 / 46.2,
ministral-14b 41.2 / 46.2 (n≈40, Claude-judged, §31). Distillation copies the teacher's errors, and
C4 errors (direction/typing) are exactly what finetuning is meant to fix. **But replies from every
model ARE on disk**, for all 20 papers × both ontologies (gemma3-12b CEO only): gemma4-31b,
qwen3-235b, ministral-14b, gemma3-12b, qwen3-8b-v2 (+ claude-opus-5, excluded).

### 34.2 Census — fold-0 training set (15 papers, i.e. what the VM build would see)
| CEO | paras w/ triples | kept | kept % | exact agree w/ gptoss | loose agree | paras where gptoss has nothing |
|---|---|---|---|---|---|---|
| gptoss-120b | 155 | 367 | 83.8 | — | — | — |
| gemma4-31b | 173 | 428 | 65.8 | 21.0% | 30.6% | 60 |
| qwen3-235b | 56 | 253 | 62.3 | 13.0% | 24.9% | 14 |
| ministral-14b | 258 | 1,025 | 47.5 | 9.2% | 14.0% | 117 |
| gemma3-12b | 294 | 1,147 | 62.2 | 5.7% | 9.2% | 151 |
| qwen3-8b-v2 | 210 | 932 | 37.1 | 8.0% | 10.8% | 84 |

| scinex | paras w/ triples | kept | kept % | exact agree | loose agree | gap-fill |
|---|---|---|---|---|---|---|
| gptoss-120b | 98 | 219 | 86.9 | — | — | — |
| gemma4-31b | 149 | 331 | 57.3 | 19.3% | 21.5% | 75 |
| qwen3-235b | 87 | 389 | 57.6 | 12.6% | 15.9% | 39 |
| ministral-14b | 260 | 1,007 | 45.6 | 6.1% | 9.2% | 172 |
| qwen3-8b-v2 | 206 | 910 | 37.6 | 3.1% | 4.5% | 126 |

**Consensus pool (exact match, ≥2 of the 4 line-up models on the same paragraph):** CEO **292**
triples on 124 paragraphs (2 votes 220 / 3 votes 58 / 4 votes 14), **143 of them absent from gptoss's
output**; scinex **232** on 96 (170 / 42 / 20), **124 absent from gptoss**. On the 3 papers usable on
the laptop: CEO 98 on 40 paragraphs (44 new), scinex 87 on 33 (56 new); gptoss alone there = CEO 125
triples on 48 paragraphs, scinex 67 on 33.

**Reading:** agreement with the teacher is low (exact 3–21%), so the other models give DIFFERENT
targets for the SAME paragraphs, not more examples of the same behaviour. Stacking them as extra
teachers would put conflicting answers on one input and pull the student toward over-extraction,
which is the base model's problem already (35% kept, §33). The consensus pool is the defensible way
to use them. ⚠ Agreement is not correctness (models can share an error), and gemma4-31b contributing
to training would disqualify it as the "independent" judge recommended in §32.6. Undecided, for the user.
gemma4-31b scinex has 8 conflicting duplicate reply lines (§30.8 pattern); `read_replies` drops them.

### 34.3 ⚠ SIDE FINDING, UNRESOLVED — most of qwen3-235b's "empty" replies are the FAILED-CALL string
Over the 22 dirs on disk (567 paragraphs), qwen3-235b CEO replies are **375 exactly
`{"triples": []}`** (the string `run_api_extract.py` writes when a call raises) vs **91 formatted
empties** (`{\n  "triples": []\n}`, the model's own abstention style) and 101 with content; scinex
319 / 93 / 155. Every other model: 0–3 such lines. The sentinels are interleaved with real replies,
not in contiguous blocks, so the files cannot tell an intermittent call failure from the model
occasionally writing a compact empty. But two whole papers are nearly all sentinel: tripplanner2020
23/23 and roadwaylight2018 30/31, plus textaug2023 12/14. **If these are failures, then qwen3-235b's
16.4% paragraph coverage (Table 10, §27.4), its "precise by declining" reading, and its C6 vacuous
passes (roadwaylight2018, textaug2023) come from failed calls, not abstention.** Its C1/C3/C4 are
unaffected (judged on triples it did produce). Settle it with the OpenRouter activity log (free) or by
re-calling a handful of sentinel paragraphs (costs credit → user's OK). For finetuning it does not
matter: `guard_pass` drops sentinels from API runs either way.

### 34.4 IMPLEMENTED — consensus-augmented targets (`build_ft_data.py --consensus-models`)
User: *"if more model means better finetuning then do it"*. Whether it IS better is the experiment
in §34.6. Rule: target = teacher's guard survivors + every triple that ≥ `--consensus-min` (2) of
gemma4-31b / qwen3-235b / ministral-14b kept on the same paragraph (exact normalised match). An
addition is skipped if the teacher already related that entity pair (any predicate, either
direction; protects the teacher's typing and direction), if it is `configures`, if its source
sentence is not in the paragraph, or if two additions on one pair disagree (a tie is dropped).
Only paragraphs with a usable teacher reply are used, so the paragraph count does not change. A consensus
model is skipped per paper on paragraph-count or alignment mismatch (ministral-14b on parkingyolo2023:
0.87 CEO / 0.90 scinex). Ontology and vocabulary cards are unchanged. Output defaults to `ft_data/consensus/`
so the teacher-only data cannot be overwritten. **Regression: the teacher-only rebuild is
byte-identical (train.jsonl sha256 a55cefec… = manifest).**
Local fold-0 build (3 usable train papers): **+66 triples on top of the teacher's 192 (+34%)**: CEO +35,
scinex +31; 7 teacher-empty paragraphs gained a target, so empty targets went 87 → 80. Skipped:
3 teacher-related pair, 12 relation conflict. Token max 4,034 (was 3,821), 0 dropped, still 2,480
examples / 620 steps.

### 34.5 ⛔ "Nothing is running" in §33.5c was WRONG — and the run was 4× slow because of Windows EcoQoS
At 07:58 the teacher-only retrain (`qwen3-4b-f0`, token-level loss) WAS running: PIDs 16912
(runner) / 12272 (trainer), started 06:47:42, i.e. the hidden relaunch after run #2's window-close
abort. It ran at ~40 s/step (ETA ~6 h) vs ~9 s/step for run #2 in a console. Cause: **Windows 11
power throttling (EcoQoS) of a windowless background process.** The trainer is bound by ONE CPU
thread feeding the GPU (CPU time ≈ wall time; GPU mostly P4/P5 at ~10% util), and EcoQoS parks it
on the i7-13620H's efficiency cores. `SetProcessInformation(ProcessPowerThrottling)` with
StateMask 0 on the live process → **~7 s/step immediately**, with no restart. Fixed in code:
`run_ft_pipeline.no_power_throttling()` opts out the runner and every child step.
`ft_runs/wait_then_run.py` does the same for an older runner's descendants. **Every hidden (WMI)
launch on this laptop needs this.**

### 34.6 ⏹ SUPERSEDED (killed before the consensus run started, §34.11). Was: teacher-only vs consensus, same fold, same base model
The plan below is now carried out on the RTX 5090 VM instead (§34.12). Kept for the design.
1. `qwen3-4b-f0` (teacher-only) continues: train → ft_gen (`qwen3-4b-ft-cv`) → compare. Its own
   Q&A step is deliberately skipped (`_pipeline_qwen3-4b-f0/qa.done` explains why).
2. Hidden waiter (`ft_runs/wait_then_run.py --pid 16912`, log `ft_runs/chain_qwen3-4b-f0-cons.log`)
   then runs `ft_runs/launch_qwen3-4b-f0-cons.cmd`: `run_ft_pipeline.py --tag cons --data
   ft_data/consensus/fold0 --also-slugs qwen3-4b-ft-cv` → train `ft_runs/qwen3-4b-f0-cons` → ft_gen
   `qwen3-4b-ft-cons-cv` → compare **base / ft-cv / ft-cons-cv / teacher** → Q&A with all four KGs
   on ONE question set (run `qwen3-4b-f0-cons-smoke`; base 4B as generator/answerer/grader = pipeline
   test only). New runner flags: `--data`, `--tag`, `--also-slugs`; base_gen is shared across tags.
Expected end ≈ 13:00 (2026-10-01). **Collapse check for both:** first replies in
`output/strabismus2026/kg/relation/<slug>/responses.jsonl` must not all be `{"triples": []}`.
⚠ ft_compare's "agreement with teacher" favours the teacher-only model BY CONSTRUCTION, because the
consensus model is trained to add triples the teacher lacks. Kept rate and coverage are fair;
correctness needs a judge or Q&A (§34.7).

### 34.7 Why a judge is still needed (answered for the user)
Judge-free numbers (valid JSON, guard kept rate, coverage) show triples are WELL-FORMED, not that
they are RIGHT; agreement-with-teacher measures imitation and is biased between the two variants.
Q&A uses an LLM as question writer (code-verified), answerer (unavoidable; it IS the test), and
grader (only for non-exact answers; could be made deterministic with token-F1). Q&A cannot see
DIRECTION errors (the answerer reads triples loosely), and direction/typing (C4) is what
finetuning targets. So C4 still needs a triple-level check. With consensus training, all four
line-up models feed the training data, so an independent judge must come from OUTSIDE the line-up
(non-Claude), or be a human-labelled sample (`gold_eval.py export`). → The user chose Claude in
session, blind (§34.8).

### 34.8 DECIDED (user, 2026-10-01): judge = Claude Opus 5.5 IN SESSION, for C1/C3/C4 only
Reverses §32.1 for judging ONLY. Claude stays out as extractor, teacher and training data, and
is never called through an API. Claude is the one model in no part of the training data, so it is
independent of teacher, consensus voters and student. Caveats to state: (a) the judge is
**claude-opus-5-5**, NOT the claude-opus-5 that judged Tables 10–12, so never mix those numbers with
these; (b) single rater, as decided in Session 29; (c) the judge co-designed the consensus variant,
so judging is **BLIND**: new `blind_judge.py` pools every kept triple of every extractor on the
held-out papers, de-duplicates (identical triple = one verdict for all models), shuffles, salts the
ids, and keeps the model mapping in `key.csv`, which the judge does not open before `collect`. It is a
census, not a sample (n≈40 is too noisy, §31.3). The triple set = exactly what `ft_compare` counts
as kept (today's guards over stored replies, conflicting duplicates dropped); verified equal: CEO
base 172 / teacher 91, scinex 164 / 57. `build` → judge writes `replies/reply_NN.txt` → `status` →
`collect` → `report --compare A B` (C1/C3/C4 with bootstrap CI over triples). Tested end to end on
DUMMY verdicts (deleted). **Not yet run for real:** it needs both finetuned KGs, which now come
from the RTX 5090 VM (§34.12). Size on the laptop's 2 papers: base + teacher alone = 257 CEO / 220
scinex unique items; the 5090 run covers all 5 fold-0 papers, so expect more. Batches of 25; do CEO first.
```
python blind_judge.py build --run ft-f0 --ontology ceo \
       --papers csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023 \
       --slugs qwen3-8b-base-hf qwen3-8b-ft-cv qwen3-8b-ft-cons-cv gptoss-120b
```
`build` refuses a slug/paper with no stored replies (`--allow-missing` overrides). It reads
`output/<paper>/kg/relation{,_scinex}/<slug>/responses.jsonl` + `prompt_meta.json`, so those must be
copied back from the 5090 VM.

### 34.9 Q&A grading is now DETERMINISTIC by default (`qa_eval.py grade --method overlap`)
SQuAD-style token overlap over content words: CORRECT if F1 ≥ 0.8 or (all reference words present
and precision ≥ 0.5); PARTIAL if all reference words present or F1 ≥ 0.4; else INCORRECT.
Thresholds `--f1-correct/--f1-partial`. Grades store f1/precision/recall. `report` gains a
**token F1** column (the standard continuous Q&A score). `--method llm` keeps the old blind LLM
grader. Mixing methods in one run is refused. Known bias: stricter than a human on verbose answers
("average accuracy of 83.0% on X" vs "83.0%" → PARTIAL). Tested on a synthetic run (deleted).
`run_ft_pipeline`'s qa step calls `grade` with `--provider/--model`; they are accepted and ignored
under `--method overlap`. **Q&A now needs an LLM only to write questions (code-verified) and to
answer them.**

### 34.10 How to settle the qwen3-235b sentinel question (§34.3), for the user
OpenRouter id `qwen/qwen3-235b-a22b-2507`, runs in Session 27 (2026-09-03 → 09-05), on the
professor's key. In the OpenRouter Activity view, count that model's completed requests in that
window. If every call succeeded, expect ≈1,134 (567 CEO + 567 scinex); if the sentinels were
failures, ≈440 (192 CEO + 248 scinex). Also look for hundreds of completions with ~5–10 output
tokens (a genuine compact `{"triples": []}`), plus error rows (429/5xx/timeouts). Alternative:
re-call ~20 sentinel paragraphs (≈$0.01), which needs a key on the machine and the user's OK.

### 34.11 ⏹ ALL LOCAL RUNS KILLED (user, ~09:20): moving to a DGX Spark (later changed to an RTX 5090 VM, §34.12)
User: *"kill all tasks running in the background, i will run it in dgx spark instead"*. Killed the
waiter (first, so it could not start the consensus run), the runner, the launcher cmd and
`ft_generate`. Verified: no python processes, GPU 0 MiB. **The consensus variant never started**
(no `ft_runs/qwen3-4b-f0-cons/`).
* **Teacher-only Qwen3-4B adapter FINISHED training** at 09:11 (144.3 min wall, of which the first
  ~70 min were EcoQoS-throttled) → `ft_runs/qwen3-4b-f0/adapter` + `run_manifest.json`.
* Its extraction was killed after **44 of 51 strabismus2026 CEO paragraphs**
  (`output/strabismus2026/kg/relation/qwen3-4b-ft-cv/responses.jsonl`, partial and NOT ingested).
* **⭐ EARLY SIGNAL (1 paper, CEO, same 44 paragraphs; not a result):** the loss fix WORKED, with no
  collapse. Abstract: 7 kept of 7 proposed (teacher 7). Kept rate base **38%** (117/308) →
  finetuned **86%** (25/29) vs teacher **88%** (53/60). **But it under-extracts:** triples in 8
  paragraphs vs the teacher's 21 (base 32). It learned the teacher's precision and over-learned
  abstention. That is exactly what the consensus variant (fewer empty targets) and
  `--max-empty-frac` address.
* Leftover markers: `ft_runs/_pipeline_qwen3-4b-f0/{base_gen,train,qa}.done`. `qa.done` is a
  deliberate skip (§34.6). **Do not copy `ft_runs/` to another machine**, or the markers skip steps.

### 34.12 ▶ NEXT: the RTX 5090 VM run (user runs it)
**⚠ TARGET CHANGED 2026-10-20 (user): "we will change from running in dgx to running on rtx5090 vm
instead".** The DGX Spark plan of 2026-10-01 is dropped; this section is the 5090 plan. Status at
that point: no local process running, laptop files unchanged since 2026-10-01 09:12, and no
results from any other machine. If the user reports 5090 results, they supersede this section.
* **Copy to the VM:** the code, `ft_data/` (incl. `split.json`, the sticky fold file; never
  regenerate it), and `output/` (stored model replies + parsed papers). **Do NOT copy `ft_runs/`**
  (its `.done` markers would skip training).
* **The GPU: RTX 5090 = 32 GB, Blackwell sm_120, the SAME architecture as the laptop's RTX 5070
  (§33.1), so the laptop's findings transfer:**
  - **torch must be a CUDA 12.8+ build.** ⛔ The old VM rule "stable torch 2.6.0+cu124, never a
    nightly" (`Claude.md` GPU notes, `requirements_ft.txt`) does NOT work here: cu124 wheels have no
    sm_120 kernels. Use a stable cu128-or-newer build if one installs; pass `--allow-dev-torch` ONLY
    if the torch you end up with is a dev/nightly build (`check_environment` refuses those otherwise).
    The laptop ran `2.12.0.dev…+cu128` + bitsandbytes **0.49.2** successfully on sm_120.
  - `pip install -r requirements_ft.txt` with that torch ALREADY installed; check
    `torch.__version__` is unchanged afterwards. Then:
    `python3 -c "import torch, bitsandbytes; print(torch.__version__, torch.cuda.get_device_capability())"`
    should print capability (12, 0).
  - The laptop's attention trap (SDPA silently falling back to the MATH kernel for Qwen3's
    grouped-query attention) may recur on sm_120. `ft_common.fast_attention(training=True)` already
    forces cuDNN for training only. Watch `peak VRAM` and s/step in the step lines of `pipeline.log`.
  - 32 GB is ample for Qwen3-8B QLoRA (the 4B peaked at 4.6 GB on the laptop), and likely for
    Ministral-14B later. Generation batch can go above the laptop's 4 (`--gen-args=--batch-size=8`);
    halve it if VRAM peaks near the limit.
* **The VM:** if Linux, use `python3` and run inside `tmux`/`nohup` so an SSH drop doesn't kill
  training. If it is a **vGPU**, the old VM notes apply: never set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `nvidia-smi` utilisation may read 0% while
  busy (judge by the log advancing). If it is a Windows VM, the EcoQoS opt-out is automatic in
  `run_ft_pipeline.py`. Launch hidden per the laptop notes. Qwen3-8B's thinking mode is already
  off (`ft_common.render_chat` passes `enable_thinking=False`).
* **Rebuild the data there if all 20 papers are parsed** (`output/<id>/no-llm/output.html`, or the
  teacher's `prompts.jsonl`). The laptop could only use 3 training papers; fold 0 has 15. Check
  every train paper × ontology says `ok`. On the laptop, oxidecrack2025 (misaligned) and
  roadwaylight2018 (paragraph count) were excluded; they may align on the machine where the
  teacher ran.
```
python3 build_ft_data.py --eval-fold 0
python3 build_ft_data.py --eval-fold 0 --consensus-models gemma4-31b qwen3-235b ministral-14b
python3 run_ft_pipeline.py --base-model Qwen/Qwen3-8B --prefix qwen3-8b --fold 0 \
    --papers csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023 \
    --gen-args=--batch-size=8 --steps base_gen train ft_gen compare
python3 run_ft_pipeline.py --base-model Qwen/Qwen3-8B --prefix qwen3-8b --fold 0 \
    --papers csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023 \
    --gen-args=--batch-size=8 --tag cons --data ft_data/consensus/fold0 --also-slugs qwen3-8b-ft-cv
```
(add `--allow-dev-torch` to both only if the VM's torch is a dev build)
  Slugs produced: `qwen3-8b-base-hf`, `qwen3-8b-ft-cv` (teacher-only), `qwen3-8b-ft-cons-cv`
  (consensus). The second run's `compare` and `qa` cover base / both finetunes / teacher. Its Q&A
  uses the base Qwen3-8B as question writer + answerer (pipeline default), so it is a PIPELINE
  TEST; real Q&A needs an independent writer/answerer (open, §34.13).
* **First checks on the results:** (1) collapse: the finetuned `responses.jsonl` must not be all
  `{"triples": []}`; (2) the §34.11 under-extraction: paragraphs with kept triples, finetuned vs
  teacher; (3) `ft_runs/qwen3-8b-f0-cons/compare.json`. Agreement-with-teacher favours the
  teacher-only model by construction.
* **Then bring back** `output/<5 fold-0 papers>/kg/relation{,_scinex}/qwen3-8b-*/` and run the blind
  judge here (§34.8 command), CEO first, then scinex.
* After fold 0: 4-fold CV with the winning variant (`--eval-fold 1..3`, the SAME `--model-slug`s,
  so every paper is scored by an adapter that never saw it).

### 34.14 Project moved to GitHub: https://github.com/SethapongS0212/SciNex (2026-10-20)
User: *"update this github to be up to date. make sure to include everything and wrote the
instructions also since i will need to move work to another laptop"*.
* **The project lives in `project_scinex/` of that PUBLIC repo** (the layout the repo already had).
  Before this, the repo held a July 17–19 web-upload snapshot (72 files). This laptop's folder
  (`Downloads/llm-as-a-judge-on-triples-main/…`) was not a git repo; git was installed with winget, the
  repo cloned to `Downloads/SciNex`, and the current project copied over it (no file deleted except
  the ones below).
* **⛔ SECURITY: the July snapshot contained a credential file.** It is deleted from the current tree,
  but deletion does not remove it from git history; the user was told in session to revoke that
  key (details deliberately not repeated in this public file). Also removed from the tree: `BERT.pdf` (no-PDF
  policy), two `.log` files, `acl_title_index.json.corrupt.bak`, and a root-level
  `fixed_extractor.py`. That was an OLDER copy (1,035 lines, without the Session 25–29 guards); the real
  one is `kg_extraction/fixed_extractor.py` (1,268 lines), which the last July commit had deleted and
  this sync restores. Kept although absent here: `kg_masked_node_pipeline.py`.
* **"Everything" = everything except what the public-repo `.gitignore` blocks:** secrets, PDFs, and
  files holding full paper text (parsed HTML, prompts.jsonl, ft_data train/parts). Two rules were
  added this session: `ft_runs/_pipeline_*/*.done` (a clone would otherwise skip pipeline steps) and
  `ontology_eval/*/*/paragraphs/*_batch_*.txt` (C2/C5 bundles print whole paragraphs). Adapters are
  excluded too (126 MB > GitHub's 100 MB limit).
* **The excluded-but-needed files went into `Downloads/SciNex_private_bundle.zip`** (for a private
  transfer; never public): 7 OA PDFs, parsed HTML, `output/acl/`, both ft datasets, the teacher-only
  Qwen3-4B adapter, and the C2/C5 bundles, listed in its `BUNDLE_CONTENTS.txt`.
* **New `SETUP_NEW_MACHINE.md`**: install order (torch first; RTX 50-series needs CUDA 12.8+), keys,
  restore from the bundle or regenerate, an install check with expected numbers (ft_compare: CEO
  base 493 → 172, teacher 107 → 91), and how to continue. README "Start here" table links it.
* **From now on, work inside a clone of SciNex** (`project_scinex/`): `git pull` first; commit +
  push after each session, checking `git status` for PDFs/keys/full text (public repo).

### 34.13 Open decisions for the user (as of end of Session 34)
1. Q&A question writer / answerer: the pipeline default (base student) is a pipeline test only.
   An independent model is needed for reportable Q&A; Claude is NOT used there (its only role is
   the blind triple judge). Needs a choice and possibly API spend.
2. qwen3-235b sentinel check (§34.10), which affects the EXISTING Table 10 coverage/C6, not finetuning.
3. `configures` direction (§29.5); training still excludes it.
4. `--verdict-filter` stays off.
5. If the 5090 run under-extracts again: try `--max-empty-frac 0.3` as a third variant.

## Session 33 — first LOCAL finetune test on the 8 GB laptop: Qwen3-4B, fold 0, 2 held-out papers (2026-10-01)

> User: *"is qwen 3 8b able to run on 8gb vram? i need to run local for now, if not, change to a
> suitable model … you can just extract a few papers for this run as a test"*. This is a
> **PIPELINE TEST**, not a result: 3 training papers, 2 eval papers, no independent judge.

### 33.1 Machine (verified — NOT the laptop the older notes describe)
RTX 5070 **Laptop** GPU, **8 GB**, Blackwell **sm_120**; 15.6 GB RAM; Windows 11; Python 3.12.10;
torch **2.12.0.dev20260408+cu128** (a nightly — sm_120 needs cu128; `--allow-dev-torch` is passed on
this machine only); transformers 5.5.4; bitsandbytes 0.49.2. Added `peft 0.21.1` + `rdflib 7.6.0`
with `--no-deps` so pip could not touch torch. **No Ollama, no `.env` keys** on this machine.

### 33.2 Model choice: Qwen3-8B does NOT fit for training at 8 GB → `Qwen/Qwen3-4B-Instruct-2507`
Qwen3-8B *runs* at 8 GB (4-bit inference) but cannot be *trained*: 4-bit weights + unquantised
151,936 x 4,096 embeddings ≈ 6 GB before activations. **Qwen3-4B-Instruct-2507** (Apache-2.0, same
family, non-thinking instruct) loads at 2.62 GB. Training examples are ≤3,821 tokens with this
tokenizer (system prompt ≈ 3.1k), so nothing is dropped at `--max-seq-len 6144`.

### 33.3 ⛔ Two memory traps, both found by measurement — both now fixed in code
1. **SDPA silently fell back to the MATH attention kernel.** On this Windows nightly the FLASH and
   MEM-EFFICIENT kernels report "No available kernel" for grouped-query attention, so every Qwen3
   attention layer ran MATH: **7.27 GB peak for ONE layer at 3.8k tokens**, vs **0.34 GB with cuDNN**
   (0.33 GB mem-efficient without GQA). Symptom: 100% GPU util at ~40 W and no training step in 10 min
   — Windows pages VRAM to system RAM instead of raising OOM. Fix: `ft_common.fast_attention()`.
   After the fix one forward+backward on the longest example = **6–8 s, 5.10 GB peak**.
   **⚠ But cuDNN is WRONG for generation:** it builds an execution plan per shape and the key length
   grows every decoded token, so it re-plans each step — the first overnight launch decoded at
   **5.7 tok/s vs 18 tok/s** (batch 4) and was killed after one batch. So `fast_attention(training=True)`
   = flash → cuDNN → mem-efficient → math (training only); the default for generation/Q&A = flash →
   mem-efficient → math (MATH's prefill peak is fine without a backward pass: 5.3 GB at batch 4).
   Re-measured: identical token counts to the pre-fix run, 3.6 min per 4-paragraph batch.
2. **peft's `prepare_model_for_kbit_training` upcasts all non-quantised weights to fp32** — doubles
   Qwen3-4B's tied embedding (0.8 → 1.6 GB). Default is now a lean bf16 path; `--upcast-fp32` restores
   the peft helper (fine on the 20 GB VM).
Also: `ft_train_qlora.py` now computes logits only over the target span (`logits_to_keep`) instead
of all ~3.8k positions x 152k vocab; prints one line per step with ETA and peak VRAM, and warns near
the VRAM limit.

### 33.4 Data for this run
* **PDFs:** the 7 `oa` papers of the EXISTING corpus re-downloaded to `papers/` (no new papers; the
  other 13 need the manual/bulk PDFs). Nature returns an HTML bot page to Python's urllib — `curl.exe`
  works; gamlprop2025 failed urllib's certificate store — also fine with `curl.exe`.
* **Parsing must reuse the committed `citation_network.json`.** The teacher's paragraphs contain
  concept-mark spacing artefacts (`time-series analysis ,`, `( esotropia , exotropia`) — they were
  parsed WITH citation concepts. `main.py --no-llm` loads the cached network (no S2 calls, nothing
  overwritten). gamlprop2025 had no network (not in S2) → parsed `--no-citations` into `no-llm/`.
  ⚠ `--no-citations` alone writes to `no-llm_no-citations/`, not `no-llm/`.
* **The pairing checks earned their keep:** `oxidecrack2025` is GENUINELY misaligned (teacher reply 29
  cites sentences that today's render puts in paragraph 28 — one boundary shifted; 0.88 CEO / 0.71
  scinex) and `roadwaylight2018` renders 30 paragraphs vs the teacher's 31. Both excluded
  automatically. Usable training papers: **gamlprop2025, parkingyolo2023, traveltime2022**.
* `ft_data/fold0/train.jsonl`: **2,480** examples — extraction **168** (83 CEO / 85 scinex; 87 empty
  targets), ontology **700**, vocabulary **1,612**. Eval (held out): **strabismus2026, osmotic2026**.
* Gazetteer restored from `output.zip` to `output/acl/csner_gazetteer.csv` (its canonical path).

### 33.5 Smoke results (before the overnight run)
* Base Qwen3-4B, 4 paragraphs of osmotic2026 (CEO): **42 triples proposed, 5 kept by the guards**
  (it invents predicates such as `definedAs`, `classifiedInto`). ~1,000 tokens per reply.
* Decode speed (bnb 4-bit): ~8 tok/s per sequence; batch 4 → ~18 tok/s aggregate, 5.3 GB peak.
* 2-step QLoRA through the real trainer: 16 long examples in 1.3 min, 4.18 GB peak, adapter saved.

### 33.5b ⛔ FIRST TRAINING RUN COLLAPSED — the adapter answered `{"triples": []}` to EVERYTHING
Base extraction finished cleanly (186.7 min). Training #1 (83.9 min, loss 3.31 → 0.14) produced an
adapter that returned `{"triples": []}` for **40/40** held-out paragraphs — and, probed directly, for
**5/5 TRAINING paragraphs whose target had triples**. So not overfitting: the abstain decision won.
**Cause (mine): per-EXAMPLE loss averaging.** A 7-token empty target weighed as much as a 300-token
extraction, so its abstain token carried ~1/7 of an example's weight vs ~1/300 for "start a triple";
with 87 empty vs 81 non-empty targets the model learned to always abstain. **Fix:** token-level
normalisation across the accumulation window (`model_accepts_loss_kwargs = True`, loss = token sum /
`num_items_in_batch`). Collapsed adapter kept as `ft_runs/qwen3-4b-f0-collapsed-exampleweighted/`.
Retrained from 06:35.
**Also fixed while waiting:** `{"triples": []}` is `run_api_extract`'s FAILED-CALL marker, but a local
model prints the identical string as a genuine answer (21 of the base model's 148 replies). The
marker is now honoured only for API runs (`ft_common.is_api_run`: `ft_generate` runs record
`base_model` in `prompt_meta.json`); ft_compare had been reporting the base model at 86.5% valid JSON
instead of 100%.

### 33.5c ⏸ FINAL STATE — no usable finetuned model yet
Retraining #2 (token-level loss) ran 06:35–06:46 and reached **step 72/620** (losses 0.12–0.52, peak
4.30 GB) before aborting with `forrtl: error (200): program aborting due to window-CLOSE event` — the
Intel Fortran runtime in numpy/scipy exits when the launcher's console window is closed. The launcher
now sets `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`; launch it hidden (command in `HANDOFF_NEXT_SESSION.md`).
**The user then stopped the work** ("you dont have to run the extraction anymore"). Nothing is running.
Resume = relaunch `ft_runs\launch_qwen3-4b-f0.cmd` (base extraction is marked done). Q&A was never
reached. Written this session: `SUMMARY_finetune_qa_2026-10-01.md` (slide-ready summary for the user)
and `HANDOFF_NEXT_SESSION.md` (what to do next).

### 33.6 The overnight run
`ft_runs/launch_qwen3-4b-f0.cmd` (launched detached via WMI so it outlives the assistant's session;
double-click it to resume — finished steps are skipped) → `run_ft_pipeline.py`:
`base_gen` (slug `qwen3-4b-base-hf`) → `train` (`ft_runs/qwen3-4b-f0`) → `ft_gen` (`qwen3-4b-ft-cv`)
→ `compare` (`ft_compare.py`, judge-free) → `qa` (`qa_eval.py`, run `qwen3-4b-f0-smoke`, local `hf`
provider). Log: `ft_runs/_pipeline_qwen3-4b-f0/pipeline.log`. The pipeline holds a Windows
keep-awake request while it runs. ⚠ The Q&A step uses the base Qwen3-4B as generator/answerer/grader
— same family as the student — so its numbers test the PIPELINE, not the model.

## Session 32 — Part 2 finetuning + Part 3 Q&A evaluation: CODE BUILT, NOTHING RUN (2026-10-01)

> **User, mid-session: "dont run the model or anything yet, just finish the fine tune and q and a
> evaluation first".** No model was loaded, no training started, no API called. Only offline checks
> (py_compile, a `build_ft_data.py --dry-run`, a guard-parity scan of stored replies) were run.

### 32.1 Decisions carried in (from Sam's handoff — do not re-open)
Claude is OUT of the pipeline (no extractor, judge, teacher; exclude every `claude-opus-5` output from
training data) · finetuning before Q&A · corpus size ON HOLD · no Qwen3-14B · line-up qwen3-235b /
gptoss-120b / gemma4-31b / ministral-14b (+ qwen3:8b as the first student) · never OpenRouter on
Claude · C1-C6 single-rater · `configures` direction still Sam's decision (§29.5).

### 32.2 The teacher replies — verified on disk
`output/<paper>/kg/relation{,_scinex}/gptoss-120b/responses.jsonl`, one line per paragraph,
`{"para_id": N, "raw": "<the model's JSON reply>"}`. gptoss ran **relation mode only** (no `fixed`).
Offline scan over the 20 papers:

| | lines | paragraphs | ok | empty content | failed-call sentinel | unparseable | duplicated para_ids |
|---|---|---|---|---|---|---|---|
| CEO | 528 | 528 | 520 (250 with triples) | 7 | 0 | 1 | 0 |
| scinex | 538 | 528 | 522 (155 with triples) | 0 | 3 | 1 | **7, all strabismus2026** (5 identical, 2 conflicting) |

**Guard parity:** running the current `_parse_fixed_output` + `kg_builder._is_valid_entity` over the
stored replies reproduces `triples.json` **exactly** — CEO 547 = 547 on disk, and every scinex paper
except `strabismus2026`, where ingesting every line (duplicates included) reproduces the on-disk 43
and de-duplicating gives 32. **Side finding:** gptoss-120b's scinex volume **323 is inflated by the
§30.8 double-writer incident; the true count is 312–323** (ratio 0.57–0.59×). Caveat added to
`results.md` Table 12.

⚠ `prompts.jsonl`, `no-llm/output.html` and `output/acl/` are **gitignored** — this GitHub checkout
has none of them (`output.zip` holds only the gazetteer + one stale `no-llm` parse). Extraction pairs
can only be built where the pipeline ran. This machine also differs from the laptop the docs
describe: Python **3.12.10** with torch + transformers installed, no peft / rdflib / spaCy.

### 32.3 Files added
| file | what |
|---|---|
| `ft_common.py` | shared: sticky fold split, paragraph loading (stored `prompts.jsonl` → else rendered from HTML with `claude_extract`'s exact logic), defensive reply reader, guard pass, the ONE chat-template helper used by training and inference, Claude refusal |
| `build_ft_data.py` | the three data parts as chat JSONL → `ft_data/fold<k>/{train.jsonl, parts/, manifest.json, eval_papers.txt}` + `ft_data/split.json` |
| `ft_train_qlora.py` | QLoRA (4-bit NF4, LoRA r16 all attention+MLP projections, loss on assistant tokens only), `--dry-run` = tokenizer stats only |
| `ft_generate.py` | base or base+adapter through the SAME relation-extraction path → `responses.jsonl` → `claude_extract.py ingest`; refuses papers the adapter trained on |
| `qa_eval.py` | Part 3: `generate` / `answer` / `grade` / `report` |
| `requirements_ft.txt` | transformers ≥4.51, peft, bitsandbytes, accelerate, rdflib |

### 32.4 Design decisions, and why
* **Split = sticky folds, default K=4.** Every paper gets a fold once (`ft_data/split.json`); new papers
  go to the smallest fold (optionally within a stratum, `--strata-file`) and nobody ever moves. So it
  works at any corpus size and growing the corpus can never turn a training paper into an eval paper.
  Fold 0 = `csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023`. Disjointness is
  enforced **three times**: build (abort if any example carries an eval paper), train (re-checks the
  manifest, the split on disk and the train.jsonl sha256), generate (refuses any paper in the adapter's
  `train_papers`). Changing `--folds`/`--seed` on an existing split is refused.
* **What counts as held-out data:** nothing from an eval paper — not its paragraphs, triples, source
  sentences (the ontology cards use train-paper sentences only) or entity list. The CS-NER gazetteer
  IS used whole: it is a public, paper-independent resource.
* **Extraction targets = the guard survivors**, re-serialised in the extractor's own output contract
  with canonical camelCase predicates (the guard lowercases them). A paragraph whose target can't be
  trusted is **dropped, never edited**: empty-content replies, failed-call sentinels, unparseable
  replies, conflicting duplicates, and (by default) any paragraph where the teacher used `configures`.
  Teacher abstentions and all-rejected replies become `{"triples": []}` targets — gptoss abstains on
  ~52% of CEO paragraphs, and that restraint is part of why it is precise (§27.4).
* **Pairing safety:** when paragraphs are rendered rather than read from the teacher's own
  `prompts.jsonl`, the paper is used only if the paragraph count matches the teacher run's
  `n_paragraphs` AND ≥90% of the teacher's source sentences are found in the paragraph they are
  attributed to. Otherwise every target would be attached to the wrong input, silently.
* **`--verdict-filter` defaults OFF — conflicts with the handoff, flagged for Sam.** The handoff asks
  to drop triples with bad C1/C4 verdicts, but every C1-C6 verdict on disk was made by
  claude-opus-5, and Claude is out of the pipeline. Turning it on (`incorrect` or `partial`) puts a
  Claude judgement into data selection. It touches ~40 sampled triples per ontology at most.
* **`--configures exclude` default** — no example teaches either direction (paragraphs skipped, no
  cards). `--configures keep` once Sam decides.
* **Ontology cards:** 3 definition phrasings per relation; direction judgements (as extracted = yes,
  subject/object swapped = no) ONLY where a swap is unambiguously wrong — domain ≠ range, or
  `splitFrom` where `_splitfrom_is_inverted` confirms the swapped form; relation-choice items.
  ⚠ Positives are teacher output, so they carry the teacher's error rate (gptoss C4 86.2% CEO / 90.0%
  scinex, n≈40, Claude-judged).
* **Train/infer prompt parity:** the system prompt is today's code (`--system-prompt current`; scinex
  needs rdflib), identical to what `ft_generate.py` sends. The baseline must go through
  `ft_generate.py` too — comparing an HF finetune against the old Ollama `qwen3-8b-v2` run would
  confound the adapter with the runtime (GGUF vs NF4, different templating).
* **Overlong examples are dropped, never truncated** (`--max-seq-len 6144`; the system prompt alone is
  ~3.4k tokens) — a truncated JSON target teaches broken JSON.

### 32.5 Local dry run (offline, `--system-prompt stored`)
Split 5/5/5/5. Ontology part **700** cards (141 definition, 167 direction-pos, 167 direction-neg, 225
relation-choice); vocabulary **1,612** type cards (gazetteer, count ≥ 2, ≤300 per type). Extraction
**0** — no paragraphs on this machine, as expected (§32.2).

### 32.6 Evaluation plan — finetuned vs base
1. Base and FT through `ft_generate.py` on the held-out papers; `--ingest` runs the normal guards.
2. **Census, not a sample**: `ontology_eval.py triples --n 100000 --papers-file
   ft_data/fold0/eval_papers.txt` judges every held-out triple. n≈40 moved scores ±10 pts (§31.3); a
   base-vs-FT delta would drown in it.
3. **Judge: not Claude, not the teacher, not the student's family.** For a qwen3:8b student trained on
   gptoss: **gemma-4-31b** is the only line-up model that is neither. Same judge for base, FT and the
   teacher ceiling (gptoss's own KG on the same papers). Driven by `run_api_judge.py --dest
   ontology_eval/<slug>/ceo --slug triples --rubric triples`. **Needs OpenRouter spend → Sam's OK.**
   ⚠ These numbers are NOT comparable with Tables 11/12 (Claude-judged): compare only within one judge.
4. Also report volume and paragraph coverage (exact, no judge), and the Q&A result.
5. Cross-validation: folds 1-3 with the SAME `--model-slug`, so every paper is scored by the one
   adapter that never saw it; then score all 20 under one slug.

### 32.7 Q&A evaluation design (`qa_eval.py`)
Questions are written from the real paragraphs by a generator model and **checked in code**: answer
copied verbatim from the paragraph, evidence sentence present, answer not in the question, no generic
"the proposed model", no boilerplate (`_is_provenance_sentence`), deduped; ≤3 per paragraph, ≤30 per
paper round-robin over paragraphs. The same question set is reused for every KG (paired). Conditions:
`kg` (triples only, **no source sentences** — they would leak the text; whole paper graph if ≤300
triples), `oracle` (the source paragraph — ceiling), `closedbook` (nothing — floor, since these are
published papers). Grading is blind and deduplicated: one verdict per (question, normalised answer)
whichever condition produced it; exact matches and abstentions need no call. Report: accuracy with a
**paper-cluster bootstrap CI**, strict share, precision on answered, per relation/paper,
**`answer_in_kg`** (deterministic, no LLM: is the gold answer a node of the graph?), **`grounded`**
(do correct KG answers come from the triples they cite, or from the answerer's own knowledge?), and
paired `--compare A B`. A missing KG run aborts rather than scoring as empty. Claude refused as
generator/answerer/grader/KG; `anthropic` is not even a provider option.

### 32.8 ▶ OPEN — needs Sam
1. **Confirm the base model** before any training (default suggestion `Qwen/Qwen3-8B`).
2. **Judge + Q&A models, and the OpenRouter spend** — recommendation: gemma-4-31b for the C1/C3/C4
   judge and for Q&A generator/answerer/grader (not the teacher, not the student's family).
3. **`--verdict-filter`** on or off, given the verdicts are Claude's (§32.4).
4. `configures` direction (§29.5) — decides `--configures keep`.
5. Verify the Hugging Face ids for Ministral-14B / Gemma-4-31B before their runs; Ministral 3 may load
   through a multimodal class, which `ft_train_qlora.py` does not handle yet.

## Session 31 — junk filter applied, samples REDRAWN, and n≈40 shown to be the binding constraint (2026-09-08)

> **⭐ START HERE. The headline of this session is a METHODOLOGICAL result, not a model result:
> at n≈40 a fresh draw of the same corpus moves scores by up to 10 points and can flip a close
> ranking. Everything below is downstream of that.**

### 31.1 What was done

1. **Extended `_is_provenance_sentence`** with the boilerplate classes the scinex round exposed
   (§30.14): fictional UX personas, CRediT lines, declarations, data-availability statements,
   affiliation blocks with postal codes, and author bios lacking "degree"/"currently".
   Added `_looks_like_table_debris()` — rejects a subject/object carrying ≥3 BARE standalone
   integers, which is the signature of a flattened table row (`(CH3)2...NCl 27 30 ...NBr 31`).
   Threshold 3 so `Llama-3.2-1B-Instruct`, `2-butyne`, `ResNet50`, `MOT17` and `1 x 1 convolution`
   all pass. **22/22 unit cases pass: 7 new patterns caught, 4 old ones still caught, 8 real-science
   sentences untouched.**
2. **Re-ingested all 8 runs from stored replies — zero API calls.**
3. **Redrew and re-judged the 5 affected samples** (146 new judgements).

### 31.2 ⭐ THE FILTER BARELY MATTERS — AND ONLY ONE MODEL WAS AFFECTED

Removed **40 triples of 6,054 (0.7%)**, and **every single one came from ministral-14b**
(12 CEO, 28 scinex). qwen3-235b, gptoss-120b and gemma4-31b lost **nothing** — they decline
that boilerplate unprompted. Worth reporting: boilerplate mining is not a general property of
extraction, it is a property of the weakest extractor in the set.

### 31.3 ⛔⛔ THE REDRAW MOVED SCORES BY UP TO 10 POINTS AND FLIPPED A RANKING

| model | C1 | C3 | C4 |
|---|---|---|---|
| qwen3-235b | 79.5 -> **75.0** | 61.5 -> 60.0 | 44.9 -> 47.5 |
| gptoss-120b | 89.7 -> 88.8 | 74.4 -> **80.0** | 84.6 -> 86.2 |
| gemma4-31b | 85.1 -> **93.8** | 73.0 -> **80.0** | 63.5 -> **73.8** |
| ministral-14b | 65.4 -> 70.0 | 56.4 -> 52.5 | 39.7 -> 41.2 |

**A prediction made before the redraw — "±4.5 points, no ordering change" — was WRONG.** Movement
reached **+10.3 points**, and the **C1 ranking flipped at the top**: gemma4-31b (93.8) now leads
gptoss-120b (88.8), reversing the earlier order.

**This is NOT the filter's doing.** Removing 0.7% of a corpus cannot move a score ten points. It is
**sampling variance**: the sampler round-robins over predicate buckets, so disturbing the pool at all
reshuffles the draw, and two draws of the same corpus at n≈40 differ by roughly ±10 points.

**⚠ THE RULE THAT FOLLOWS, APPLY IT EVERYWHERE:** at n≈40, **any gap smaller than ~10 points between
two models is not distinguishable.** Quote rankings only where the gap is comfortably larger, and
never quote a C1 ordering — its whole spread is ~24 points across four models, so neighbouring pairs
are inside the noise.

**What survives:** the **C4 ranking is unchanged** (gptoss-120b > gemma4-31b > qwen3-235b >
ministral-14b), and C4's spread (~45 pts) remains far wider than C1's (~24 pts), so
"models differ on typing more than on finding concepts" holds comfortably.

### 31.4 FINAL NUMBERS — 20-paper corpus, post-filter, post-redraw

| model | CEO vol | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|
| gptoss-120b | 547 | 88.8 | 34.4 | 80.0 | **86.2** | 31.2 | 25.0 |
| gemma4-31b | 615 | **93.8** | 28.6 | 80.0 | 73.8 | 21.4 | 33.3 |
| qwen3-235b | 367 | 75.0 | 31.2 | 60.0 | 47.5 | 28.1 | **50.0** |
| ministral-14b | 1,536 | 70.0 | 26.7 | 52.5 | 41.2 | **34.4** | 8.3 |

scinex, triples harness only:

| model | scinex vol | C1 | C3 | C4 | dC1 | dC3 | dC4 | vol ratio |
|---|---|---|---|---|---|---|---|---|
| ministral-14b | 1,538 | 72.5 | 60.0 | 46.2 | +2.5 | +7.5 | +5.0 | 1.00x |
| qwen3-235b | 563 | 72.5 | 57.5 | 46.2 | -2.5 | -2.5 | -1.2 | 1.53x |
| gemma4-31b | 525 | 88.5 | 79.5 | 59.0 | -5.3 | -0.5 | **-14.8** | 0.85x |
| gptoss-120b | 323 | 86.2 | 72.5 | **90.0** | -2.5 | -7.5 | +3.7 | 0.59x |

**⚠ Most of these deltas are inside the ±10 pt noise band.** Only gemma4-31b's **-14.8** on C4 and
possibly ministral's +7.5 on C3 sit outside it. The earlier conclusion — *the ontology effect is
model-specific and does not order by capability* — still stands, but state it as **"no consistent
ontology effect was detectable at this sample size"**, which is what the data actually support.

**The volume ratio still falls monotonically with capability** (1.00x, 1.53x, 0.85x, 0.59x ordered
by C4 — not monotone; ordered by C4 it is 1.00, 1.53, 0.85, 0.59). Volume counts are exact, not
sampled, so that column carries no sampling noise — but it is also not monotone, so do not claim it is.

### 31.5 ⛔ GUARD GAPS STILL OPEN — found by judging the redrawn items

The extended guard closed the §30.14 list, and judging the new draws immediately exposed four more:

| gap | example that slipped through |
|---|---|
| author bio, third form | `He is a Machine Learning Engineer with Kasikorn Business-Technology Group, Thailand.` — no "degree", no "currently", no "From YYYY to YYYY". Needs `(he\|she\|they) is a/an <role> with <Org>` |
| bibliography, third form | `Chen, Xuzhan... "Facial landmark detection based on cascade neural network." Journal of Physics: Conference Series.` — no year:pages tail, no vol./pp., no doi |
| CRediT split by OCR | `Mansouri: Writing – re view & editing.` — the OCR splits "review" into "re view", so `writing - review` never matches |
| `splitFrom` guard, known limit | `(training set) splitFrom (five segments)` — the guard only fires when the OBJECT names a partition and the subject does not; here the subject is the partition word |

**Do NOT patch these mid-measurement.** That is what forced this session's 146-judgement redraw.
Batch them for a single guard round, then re-ingest and redraw once.

### 31.6 ⭐ `configures` — THREE CORRECT USES EXIST, ALL FROM ONE MODEL

The user asked to ignore the `configures` ontology question, so nothing was changed. Recording the
evidence for whenever it is revisited: across the whole study exactly **three** correctly-directed
`configures` triples exist (Specification -> Experiment), and **all three are ministral-14b's**:
`(LLM parameters) -> (Experiment 1)`, `(mutation probability m) -> (mutation operation)`,
`(K-Fold Cross-Validation) -> (each experiment)`. Every other instance from every model is
`Model -> setting-value`. So the direction is not unlearnable — one model gets it right sometimes,
inconsistently, and the same model also produces the inverted form elsewhere.

### 31.7 The 8B question, settled

The user asked whether a Gemma 8B was in the fixed line-up. **It was not** — the Session 27 list is
Qwen3-235B / GPT-OSS-120B / **Gemma-4-31B** / Ministral-14B. A `gemma3-12b` exists on disk from
Session 26 (22-paper frame, CEO only) and is not part of this experiment. No Gemma is installed
locally, so no local Gemma run was ever pending.

**qwen3-8b DID run everything, via the local Ollama build:** `relation/qwen3-8b-v2` and
`relation_scinex/qwen3-8b-v2` are both **20/20 papers with stored replies** (1,355 and 1,325
triples). v1 (`qwen3-8b`) sits at 18/20 with NO stored replies and stays frozen as the §26.14
"before" baseline. **qwen3-8b-v2 could be judged for free** — the extraction exists and judging is
in-session — giving a fifth point on the capability curve. Outside the fixed line-up, so not done.

### 31.8 Tooling changed this session

* `_is_provenance_sentence` extended; `_looks_like_table_debris` added.
* **`ontology_eval.py collect` is now INDEX-AWARE.** It previously read every file in `replies/` and
  deduped by id, so after a redraw a verdict for a no-longer-sampled triple silently survived into
  `verdicts.csv` and got scored. It now restricts to the ids in `index.csv`, reports how many stale
  verdicts it dropped, and **warns about sampled items that are still unjudged** — which is what made
  the 146-item redraw tractable.
* `show_unjudged.py` (new) prints only the sampled items lacking a verdict, so a redraw costs
  judging time proportional to what actually changed.
* `reingest_ceo.sh`, `reingest_scinex.sh`, `reingest_one.sh` — re-ingest from stored replies, no calls.

### 31.9 ▶ WHAT IS LEFT

1. **Sample size is now the top issue.** Every per-model comparison narrower than ~10 points is
   undecidable. Raising n to ~120 per cell would be the single highest-value change; it costs
   judging time only, no money.
2. **scinex C2/C5 deliberately skipped** — user: *"we want correctness not how importance."*
   scinex C6 (contradiction, arguably correctness) is still unjudged: 8 batches if wanted.
3. **C2/C5/C6 were NOT redrawn** this session — only the triples harness was affected by the filter.
   ministral's C6 triple sets did change slightly, so its C6 (8.3%) is the least trustworthy figure.
4. Guard gaps in §31.5, batched for one round.
5. `configures` — ignored by user decision; evidence preserved in §31.6.
6. `results.md` still lacks the scinex tables (RESULTS_REPORT.md has them).

## Session 30 — ⛔ CORPUS CORRECTED TO 20 PAPERS; Qwen3-14B retired; scinex run launched (2026-09-07)

> **⭐ READ THIS BEFORE QUOTING ANY NUMBER. The corpus is 20 papers, not 22.**

### 30.1 ⛔⛔ THE CORPUS IS THE USER'S 20-PAPER LIST. NOT 22. NOT "18 + 2 SUBSTITUTES".

User, this session: *"arent we supposed to run only 20 papers that are all the list i gave you
before. not 22"*. They are right, and `papers/manifest.csv` proves it:

* `aiabstract2025` — note reads **`SUBSTITUTE for tripplanner2020`**
* `routepred2023`  — note reads **`SUBSTITUTE for linkpred2015`**

Those two were added in Session 26 **only because** `tripplanner2020` and `linkpred2015` were
paywalled and unobtainable. In Session 27 **the user supplied both missing PDFs** and they parsed
clean — at which point the substitutes should have been RETIRED. Instead they were kept *alongside*
the recovered originals, and the corpus was written up as 22. That was the error.

**The corpus = the 20 papers on the user's list**, which INCLUDES `tripplanner2020` and
`linkpred2015` and EXCLUDES `aiabstract2025` and `routepred2023`. All 20 are parsed and on disk.
The canonical id list is now **`papers_20.txt`**, generated from the manifest by excluding rows
whose note starts with `SUBSTITUTE for`. **Do not filter on the substring "SUBSTITUTE"** — the two
recovered papers say "SUBSTITUTED" in their own notes and a naive filter drops them too (it did,
first try, and produced a list of 18).

`papers/manifest.csv` is updated: the recovered pair is `access=manual_user` and marked in-corpus;
the substitutes are `access=retired` and marked NOT in the corpus. Their parsed output stays on
disk — nothing is deleted, it is simply out of scope.

### 30.2 How far the 22-paper framing leaked into the C1-C6 results

> **⚠ CORRECTED LATER THE SAME SESSION — the first version of this table was wrong.**
> It was computed from `verdicts.csv`, which for the **triples** harness has columns
> `triple_id,C1,C3,C4,...` and **no `paper` column**. `row.get('paper','')` therefore returned
> `''` for every row, matched nothing, and reported a confident "0 of 40 — CLEAN". The paper id
> lives in **`index.csv`** (`triple_id,paper,subject,predicate,object,source_sentence`), which is
> what the corrected figures below join against. **Lesson: a contamination count of exactly zero
> across four independent models is a red flag, not a clean bill of health — check that the column
> you are filtering on exists.**

Judged items drawn from the two substitute papers, joined `verdicts.csv` x `index.csv`:

| harness | criteria | qwen3-235b | gptoss-120b | gemma4-31b | ministral-14b |
|---|---|---|---|---|---|
| triples | C1, C3, C4 | 1/40 | 1/40 | **3/40** | 1/40 |
| paragraphs | C2, C5 | 0/16 | 0/16 | **2/16** | 1/16 |
| papers | C6 | **2/6** | **2/6** | **2/6** | **2/6** |

### 30.2b ⭐ THE SCORES BARELY MOVE WHEN RESTRICTED TO THE 20 PAPERS

Recomputed with substitute items dropped (`as judged -> restricted`):

| model | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| qwen3-235b | 78.8->79.5 | 31.2->31.2 | 60.0->61.5 | 45.0->44.9 | 28.1->28.1 | 50.0->50.0 |
| gptoss-120b | 88.8->89.7 | 34.4->34.4 | 75.0->74.4 | 82.5->**84.6** | 31.2->31.2 | 25.0->**12.5** |
| gemma4-31b | 85.0->85.1 | 28.1->28.6 | 72.5->73.0 | 62.5->63.5 | 25.0->21.4 | 16.7->**12.5** |
| ministral-14b | 63.7->65.4 | 31.2->26.7 | 55.0->56.4 | 38.8->39.7 | 34.4->30.0 | 0.0->0.0 |

**Every headline finding survives, and the rank order is unchanged:**
* **C4 still spans ~2.1x** (39.7% ministral -> 84.6% gptoss) while C1 spans only ~1.4x. Models still
  differ on TYPING, not on concept identification.
* **C2/C5 are still flat** across a 4.3x volume range (band 21.4-34.4%). Volume still does not buy
  completeness.
* **C6 still falls as volume rises** — but see the caveat below.

**⚠ TWO CLAIMS MUST BE WEAKENED:**
1. **C6 "perfect rank correlation with volume" is now a TIE, not a strict ordering:**
   qwen3-235b 50.0 (395 triples) > gptoss-120b **12.5** (583) = gemma4-31b **12.5** (684) >
   ministral-14b 0.0 (1,709). Say **"monotone non-increasing"**, not "perfect rank correlation".
2. **C6 n drops from 6 to 4 papers per model.** That is too small to quote at all. **The C6 redraw
   is mandatory**, not optional.

C1/C3/C4 and C2/C5 do **not** need re-judging: dropping 1-3 items moves them by at most 4.5 points
and changes no ordering. Only C6 does.

### 30.9 ⭐ C6 REDRAWN ON THE 20-PAPER CORPUS — AND IT BREAKS A HEADLINE CLAIM

C6 was redrawn (the substitutes were 2 of 6 papers per model) and re-judged in session, on the
CEO triples **after** the guard re-ingest. `ontology_eval` now carries a corpus allowlist that
**defaults to `papers_20.txt`**, so the substitutes cannot be redrawn by forgetting a flag.

| model | triples (20 papers) | C6 before redraw | C6 after redraw |
|---|---|---|---|
| qwen3-235b | 367 | 50.0% | **50.0%** |
| gptoss-120b | 547 | 25.0% | **25.0%** |
| gemma4-31b | 615 | 16.7% | **33.3%** |
| ministral-14b | 1,548 | 0.0% | **8.3%** |

**⛔ THE MONOTONICITY CLAIM DOES NOT SURVIVE.** Ordered by volume:
367 -> 50.0%, 547 -> 25.0%, **615 -> 33.3%**, 1,548 -> 8.3%. gemma4-31b has *more* triples than
gptoss-120b and a *higher* C6, so the sequence is **not monotone non-increasing**.
**Spearman rho = -0.80**, not -1.00.

Report it as: **"C6 falls as extraction volume rises (Spearman rho = -0.80 over four extractors),
though not strictly monotonically."** Do NOT write "perfect rank correlation" — that claim came from
the 22-paper draw and died with it. With only four models one swap moves rho from -1.00 to -0.80, so
the direction is the claim, not the ordering.

**Why gemma4 moved most (16.7 -> 33.3):** its redrawn sample replaced two substitute papers with
`microwave2018` (scored CONSISTENT — genuinely coherent, 11 triples) and `roadwaylight2018`
(MINOR_ISSUES). The old draw had caught it on two of its worst sets.

**The vacuous-pass artifact is now visible in the data:** qwen3-235b's redrawn sample contains
**two papers with zero triples** (`roadwaylight2018`, `textaug2023`), both scoring CONSISTENT = 1.0.
That is 2 of its 6 C6 points earned by extracting nothing, and it extracts from only 16.4% of
paragraphs. **Never quote C6 without C2/C5 beside it.**

Notable individual findings from the redraw:
* **The only correctly-directed `configures` in the entire study**: ministral-14b's
  `(LLM parameters) --configures--> (Experiment 1)` on `llamacorrupt2025` — Specification ->
  Experiment, exactly as defined. Every other instance across every model is Model -> setting-value.
* **Two models independently collapsed the same contrast**: gptoss-120b and gemma4-31b both assert
  `(cSysGuard) --uses--> (simple averaging)` when the source says cSysGuard uses stacking *rather
  than* simple/weight averaging. A shared failure mode on contrastive sentences, not a per-model quirk.
* **gemma4-31b and gptoss-120b both got all three `splitFrom` partitions right** on traveltime2022,
  where most models in the study inverted them.

### 30.12 ⭐⭐ SCINEX JUDGED (triples harness) — THE SESSION 26 CAPABILITY STORY DOES NOT REPLICATE

scinex extraction is complete for qwen3-235b, gemma4-31b and ministral-14b over the 20-paper corpus
(gptoss-120b still running at time of writing). All four CEO runs and all finished scinex runs are
on ONE code version — every one re-ingested through the Session 29 guards from stored replies.

**Volumes (20-paper corpus):**

| model | CEO | scinex | scinex/CEO |
|---|---|---|---|
| qwen3-235b | 367 | 563 | 1.53x |
| gemma4-31b | 615 | 525 | 0.85x |
| ministral-14b | 1,548 | 1,566 | 1.01x |

**C1/C3/C4 deltas, CEO -> scinex, n≈40 per cell, ordered by CEO capability (C4):**

| model | CEO C4 | dC1 | dC3 | dC4 | vol ratio |
|---|---|---|---|---|---|
| ministral-14b | 39.7% (weakest) | **+5.9** | **+6.1** | **+5.3** | 1.01x |
| qwen3-235b | 44.9% | **-7.0** | -4.0 | +1.4 | 1.53x |
| gemma4-31b | 63.5% (strongest) | +3.3 | +6.5 | **-4.5** | 0.85x |

**⛔ THE §26.16 INTERACTION DOES NOT REPRODUCE HERE.** Session 26 found scinex to be a liability for
a weak extractor and an asset for a capable one (8B: -26.0 pts; 235B: +5.9 pts). Under C1-C6 that
ordering is absent: the WEAKEST model gains on all three criteria, the strongest gains on concepts
but LOSES on relation correctness, and the middle model loses on concepts. There is no monotone
relationship with capability in either direction.

**State it as: under C1-C6 the ontology effect is MODEL-SPECIFIC and does not order by extraction
quality.** Two caveats that stop this being a refutation of §26.16:
1. n≈40 per cell.
2. §26.16 measured a different quantity (single-rubric strict precision) and its capability contrast
   was WITHIN one model family (qwen3:8b vs qwen3-235b). These three are three different families,
   so family effects and ontology effects are confounded. **This is a failure to reproduce the
   pattern on a different axis, not a refutation of the original measurement.**

The volume ratio also fails to track capability: gemma4-31b has much better CEO C4 than qwen3-235b
(63.5% vs 44.9%) yet the LOWEST ratio (0.85x vs 1.53x).

### 30.13 ⭐ WHAT SCINEX BUYS AND WHAT IT COSTS, from the per-triple notes

**scinex fixes CEO failures — same sentence, better relation available:**
* `(YOLOv8) SPPF`: the CEO run chose `employs` (typed Organisation -> Person) and scored INCORRECT;
  the scinex run chose `uses` and scored CORRECT.
* `splitFrom`: gemma4-31b inverted it under CEO, got it right under scinex.
* `producedBy` (artifact -> process) gives a direction CEO's `produces` kept inverting. ministral's
  `(GBipt) producedBy (transforming RDF dataset into a bipartite graph)` is exemplary.
* `achievesResult` (Model -> ExperimentalResult) is a genuine refinement of CEO's overloaded
  `achieves` — **but only when the model reaches for it.** gemma4-31b kept using plain `achieves`
  for metric outcomes and lost the benefit.

**scinex adds failure modes CEO structurally cannot have:**
* **`extractedFrom` was misused FIVE times** — range is AcademicPaper, and it was pointed at a
  dataset, an architecture component, OpenStreetMap, a vehicle fleet and a data-availability
  statement. Its name reads generically enough that the English always seems to fit. CEO has no
  equivalent relation, so it cannot make this error at all.
* `mentions` (AcademicPaper -> ?) attracted a reversed use: `(SP) mentions (Jeung et al. (2008))`
  where the sentence says the SF/SP hybrid was mentioned IN that work.

### 30.14 ⛔ GUARD GAP — `_is_provenance_sentence` KEYS ON SECTION NAMES AND MISSES SENTENCES

The scinex judging surfaced boilerplate classes the Session 29 guard does not catch, because the
Session 26 boilerplate guard matches SECTION NAMES and these appear as sentences inside otherwise
legitimate sections:

| what was mined | example |
|---|---|
| **a FICTIONAL UX PERSONA** | `Persona: Dan is a senior salesman working in a company in Europe.` -> `(Dan) affiliatedWith (a company in Europe)` |
| a CRediT contribution line | `Jinor ose: Writing - review & editing, Validation.` -> `(Jinor ose) writes (this study)` |
| a declarations statement | `All authors have read and agreed to the published version...` |
| a data-availability statement | `The data presented ... are available on request` |
| an affiliation block | `a Faculty of Engineering, KMITL, Bangkok 10520, Thailand` |
| **OCR TABLE DEBRIS** | subject = `(CH3)2(CH2(C6H5))(C2H4OH)NCl 27 30 ... NBr 31` — two chemical formulas with their table row numbers |
| an author bio WITHOUT the keywords | `From 2022 to 2023, he was a Software Engineer with RentSpree.` — no 'degree', no 'currently', so the existing bio pattern misses it |

**The persona case is the most striking: an invented person entering the knowledge graph as a real
agent.** No guard in this pipeline anticipates fictional entities from UX/requirements sections.

Worth noting the three CORRECTLY-typed `employs` triples in the whole study all come from author
biographies — i.e. the one relation the models get right is the one whose evidence should have been
excluded.

### 30.15 ⛔ TWO HARNESS BUGS FOUND WHILE PREPARING THE SCINEX BUNDLES

1. **`_schema_for` silently fell back to the CEO schema for scinex.** It imported
   `_load_scinex_schema` from `fixed_extractor` — **a function that does not exist** — and a bare
   `except` swallowed the ImportError. Every scinex bundle printed
   `(predicate not in this ontology)` for scinex-only relations, which would have collapsed C4 into
   a plausibility check on the English relation name — precisely the failure §27.3 design decision 4
   exists to prevent. The real loader is `kg_extraction.ontology_loader.load_ontology`, returning
   `(relations, schema)`. **Now fixed AND made to raise rather than fall back**, because a silent CEO
   fallback invalidates every scinex C4 verdict. Caught only by reading a bundle before judging it.
2. **`triple_id` collisions shrink the sample.** The id hashes only
   `(slug, subject, predicate, object)`, so an identical triple drawn from two DIFFERENT source
   sentences is one id and deduplicates on `collect`. gemma4-31b's scinex sample came back n=39, not
   40. Small, but the harness can silently under-deliver its requested sample size.


### 30.16 ⚠ MINOR STALENESS: the CEO samples were drawn BEFORE the guard re-ingest

Order of operations this session was: CEO judged (Session 27-28) -> Session 29 guards written ->
CEO re-ingested through those guards. So a few already-judged CEO triples have since been removed
from the corpus by `_is_provenance_sentence` / `_splitfrom_is_inverted`.

| model | judged (20-paper) | since removed | C4 as reported | C4 excluding removed |
|---|---|---|---|---|
| qwen3-235b | 39 | 2 | 44.9% | 47.3% |
| gptoss-120b | 39 | 2 | 84.6% | 83.8% |
| gemma4-31b | 37 | 4 | 63.5% | 65.2% |
| ministral-14b | 39 | 4 | 39.7% | 40.0% |

**Not material — the shift is between -0.8 and +2.4 points and changes no ordering.** Note the sign
is MIXED: gptoss-120b goes DOWN when the removed triples are excluded, so the guards were not
removing uniformly-bad triples in these samples.

**Reported figures describe the corpus AS SAMPLED (pre-guard).** Excluding the removed items would
not give a clean post-guard measurement either, because the sample itself was drawn from the
pre-guard pool — only a redraw would. Given the size of the effect, a redraw is not worth the
judging time; record the caveat instead. The scinex samples do NOT have this problem: they were
drawn after the guards, from the re-ingested corpus.

### 30.3 Other standing instructions from the user this session

* **⛔ Qwen3-14B IS RETIRED. Do not plan, schedule, or reference a VM re-run with it.** The old
  "re-run everything on the VM with Qwen3-14B" item is DEAD and has been struck from `TASK.md`.
  Work continues with the Session 27 model line-up only: **qwen3-235b, gptoss-120b, gemma4-31b,
  ministral-14b**.
* **⛔ Judging is IN-SESSION by claude-opus-5 only. Never an OpenRouter judge.** OpenRouter credit
  is for EXTRACTION only. (Unchanged, restated by the user.)
* **C1-C6 is single-rater by design** (Session 29) — a stated limitation, not an open item.
* The user noted a Gemma model could be run locally via Ollama. **Not actioned: no Gemma is pulled**
  (`ollama list` shows only `llama3.1:8b` and `qwen3:8b`), and `gemma-4-31b` cannot fit the 8 GB
  laptop GPU. Needs one clarifying answer before pulling several GB — see §30.6.

### 30.4 scinex extraction launched — 3 models, 20 papers

`relation_scinex/qwen3-235b` **already exists** (600 triples, from Session 27), so only three models
needed running. Runner: **`run_scinex_20.sh`** (`--paper $(cat papers_20.txt)`, so the substitutes
are never touched), log: `scinex_20_extraction.log`.

    openai/gpt-oss-120b          -> relation_scinex/gptoss-120b
    google/gemma-4-31b-it        -> relation_scinex/gemma4-31b
    mistralai/ministral-14b-2512 -> relation_scinex/ministral-14b

OpenRouter balance at launch: **$1.2725 used of $3.00, $1.7275 remaining.** Estimated ~$0.6-0.7 for
all three (Session 27 cost ~$0.23 per model-run over 22 papers).

### 30.5 ⚠ A CONFOUND THAT MUST BE CLOSED BEFORE COMPARING CEO vs scinex

The CEO extractions were produced in Session 27, **before** the Session 29 guards
(`_is_provenance_sentence`, `_splitfrom_is_inverted`). The scinex runs launched today have those
guards **active**. Comparing them directly would vary ontology AND code version at once.

**Fix, and it costs nothing:** all 22 CEO papers x 4 models have `responses.jsonl` on disk, so CEO
can be **re-ingested through the current guards with no new model calls** — the same trick that took
Claude's corpus 268 -> 279 in Session 26. Do this before any CEO-vs-scinex claim.

### 30.6 ▶ NEXT, in order

1. Wait out `scinex_20_extraction.log`; verify triple counts for all three models.
2. **Re-ingest CEO from stored replies** so both ontologies sit on one code version (§30.5).
3. Regenerate C1-C6 bundles **restricted to the 20 papers**, for CEO and scinex.
4. **Redraw and re-judge C6** (§30.2) — mandatory. Re-judge C2/C5 for gemma4/ministral if cheap.
5. Rebuild `RESULTS_REPORT.md`; correct every "22 papers" to 20 across the docs.
6. Ask the user which Gemma they mean for the local run (§30.3).

### 30.10 ⚠ THE PER-PAPER INGEST FAILS SILENTLY — ALWAYS RE-INGEST AFTER A RUN

`run_api_extract --ingest` ingests after each paper. When a paper's `responses.jsonl` contains a
malformed line (which the duplicate-runner incident in §30.8 produced), that paper's ingest raises a
`JSONDecodeError` and is skipped — **but the run continues to the next paper and exits 0**. The
extraction looks successful and the log ends with "COMPLETE".

Result for gemma4-31b's scinex run: **559 responses saved, but only 15 of 20 papers had a
`triples.json`** — a 30% silent loss that a triple count alone would have reported as "the model
just extracts less on scinex".

**Always verify papers-with-triples, not just responses, and re-ingest from stored replies before
using any run.** `reingest_scinex.sh <slug>` and `reingest_ceo.sh` do this with no model calls:

    bash reingest_scinex.sh gemma4-31b     # 15/20 -> 20/20 papers, 349 -> 525 triples

Also re-ingested `relation_scinex/qwen3-235b` (a Session 27 run that predated the Session 29
guards): 567 -> 563 triples. **Every CEO and scinex run is now on one code version**, which is the
precondition for any CEO-vs-scinex comparison.

### 30.11 ⭐ EARLY RESULT — the scinex/CEO VOLUME RATIO IS CAPABILITY-DEPENDENT, and it replicates

Over the 20-paper corpus, all runs on the current guards:

| model | CEO | scinex | scinex/CEO |
|---|---|---|---|
| qwen3-235b | 367 | 563 | **1.53x** |
| gemma4-31b | 615 | 525 | **0.85x** |

This **reproduces the Session 26 finding** (§26.16) that the ratio flips with capability: measured
there as 0.94x at qwen3:8b and **1.48x** at qwen3-235b. The 1.53x here is an independent
re-measurement of the same model on a corrected corpus and a newer code version, and lands within
0.05 of the original. gemma4-31b sitting below 1.0 places it on the weak side of that curve on the
volume axis, despite its strong C1/C3/C4 scores — worth watching when its scinex triples are judged.

⚠ Volume is not quality: Session 26's interaction result was that scinex wins on BOTH axes only at
the higher capability level. The judged C1-C6 numbers decide that, not these counts.

### 30.8 ⛔⛔ INCIDENT: STOPPING A BACKGROUND TASK DOES NOT KILL ITS PYTHON CHILD

**This corrupted data and double-spent OpenRouter credit. Read before backgrounding anything.**

The first scinex attempt used one sequential bash runner (`run_scinex_20.sh`, models one after
another). To parallelise, that task was stopped with `TaskStop` and three per-model runners were
launched instead. `TaskStop` reported success — **and the orphaned bash wrapper and its `python
run_api_extract.py` child both kept running.** The orphan then walked its own loop to the next
model, so within fifteen minutes there were **two processes extracting `gptoss-120b` and two
extracting `gemma4-31b`, writing the same `responses.jsonl` files**.

**How it showed up:** `ontology_eval`-style JSON errors during the per-paper ingest, and lines in
`responses.jsonl` that began mid-sentence — `', it can be observed that at higher values, DT, RF...'`
— with no `{"para_id":` prefix. That is the signature of **two processes appending to one file**:
`responses.jsonl` is opened in append mode and a record can be split by another writer's write.
13 of 1,606 lines (0.8%) were mangled this way.

**Why it was nearly invisible:** `ps aux | grep run_api_extract` under Git Bash returned **nothing**
even while the processes were demonstrably making API calls and writing files. Do not trust `ps`
here. The reliable check is PowerShell:

    Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
      ForEach-Object { if ($_.CommandLine -match '--model-slug\s+(\S+)') { $Matches[1] } } |
      Group-Object | Where-Object { $_.Count -gt 2 }

(Each real runner shows **two** entries — a WindowsApps launcher stub plus the real interpreter — so
"more than 2 entries for one slug" is the duplicate condition, not "more than 1".)

**Cleanup that worked:** kill the duplicate PIDs by creation time, then kill every
`bash ... run_scinex_20.sh` wrapper — otherwise the wrapper immediately spawns the *next* model and
recreates the collision (it did: killing the wrappers spawned a duplicate `ministral-14b`, which
then also had to be killed).

**Rules going forward:**
1. **Never background a script that loops over models.** One background task per model, so an
   orphan can only ever duplicate itself and is trivially identifiable.
2. **After any `TaskStop`, verify with PowerShell that the child is actually gone.** The task
   status is not evidence.
3. **Treat mid-sentence lines in a `.jsonl` as a concurrency signal**, not a model failure.
4. `--resume` rebuilds its done-set by parsing every line of `responses.jsonl`, so **one malformed
   line aborts the entire run before a single call is made** (`JSONDecodeError`). `run_one_scinex.sh`
   now repairs the file first; keep that step.

**Cost of the incident:** roughly \$0.05-0.10 of duplicated calls and 13 mangled responses, all
recoverable — the affected paragraphs simply get re-called once the malformed lines are dropped.

### 30.7 A smoke-test result that looked alarming and was not

The scinex smoke test returned **0 triples kept, 7 rejected**. Cause was neither the new guards nor
the ontology: the two pre-existing guards fired correctly — the `designedFor` intent check, and the
subject-presence rule, because the model wrote `U-GMo` as the subject where the source sentence says
"The system" (it resolved the coreference, which relation-mode forbids). Both new guards returned
False on all 7. **A 2-paragraph smoke test is too small to read as a failure signal.**

## Session 29 — C1-C6 declared single-rater; two new guards built and measured (2026-09-07)

### 29.1 ⛔ DECIDED BY THE USER: C1-C6 IS SINGLE-RATER. DO NOT RE-PROPOSE A SECOND JUDGE.

claude-opus-5 in-session is the judge of record for all 248 C1-C6 judgements. **No second labeller,
no kappa, none planned.** Report it as a *stated limitation* of the track, never as an unfinished
item. The old-rubric kappa (results.md Table 9) measures a different rubric and does not transfer.
`results.md` caveat rewritten accordingly.

`run_api_judge.py` was extended before that decision and the change is **kept**, because it fixed a
latent bug rather than adding a judge: it hardcoded the OLD CORRECT/PARTIAL/INCORRECT rubric, so it
could not drive `ontology_eval` bundles at all. It now takes `--rubric {gold,triples,paragraphs,papers}`
(default `gold`, so existing invocations are unchanged) and prints the matching `collect` command.

### 29.2 Two new guards in `kg_extraction/fixed_extractor.py`, from the C1-C6 findings

**(a) Provenance guard — `_is_provenance_sentence(sentence)`.** Rejects a triple whose SOURCE
SENTENCE is publisher/biographical/bibliographic matter. This is the fix for §28.5: triples that
score CORRECT on C1, C3 AND C4 and are still worthless, because the per-triple criteria judge a
triple against its source sentence and never ask whether that sentence belonged to the paper.
Keyed on the SENTENCE, not the section name, because `_is_boilerplate_section` misses two cases:
front/back matter with no heading of its own, and IEEE watermarks/running headers that the parser
glues INTO a body paragraph (so the paragraph's section is legitimate). Covers IEEE licence
watermarks, author biographies, acknowledgements, bibliography entries (three citation formats),
ACM CCS classification blocks, licence/ISSN/masthead lines.

**(b) splitFrom direction guard — `_splitfrom_is_inverted(subject, object)`.** splitFrom is
Dataset->Dataset, so a type check cannot catch an inversion - both sides are datasets. The derived
partition must be the SUBJECT. 3 of the 4 extractors consistently wrote it backwards (gptoss-120b
was the only one that did not). Rejects only the unambiguous case: object names a partition, subject
does not. When both or neither look like partitions the triple is left alone.

### 29.3 Measured impact (whole corpus, all four models)

| model | triples | removed | % |
|---|---|---|---|
| qwen3-235b | 395 | 8 | 2.0% |
| gptoss-120b | 583 | 5 | 0.9% |
| gemma4-31b | 684 | 10 | 1.5% |
| ministral-14b | 1,709 | 29 | 1.7% |

Every removal was eyeballed; all are genuine defects, no false positives observed. Examples:
`(KMITL, affiliatedWith, UniNet)` from an IEEE licence watermark; `(SOKENDAI, locatedIn, Japan)` and
four `(RATHACHAI CHAWUTHAI, affiliatedWith, <university>)` from one author biography; two
`publishedIn` triples from bibliography entries; `(LExist, splitFrom, LTrain)` inverted.

**⚠ The corpus rate (0.9-2.0%) is much lower than the judged sample suggested** because the C1-C6
sample is stratified over predicates and deliberately over-represents rare ones - the same
sampled-vs-weighted gap as the old track (sampled strict 20.2% vs weighted 34.7%). Both numbers are
right; state which one is being quoted.

### 29.4 ⛔ THREE TRAPS FOUND WHILE WRITING THESE GUARDS

1. **Backslashes are collapsed when a Python heredoc is piped through the Bash tool.**
   **⚠ SECOND OCCURRENCE — the same trap is recorded in the tooling note at the end of
   Session 26 §11.** It also silently mangled the sentence in this very note that described
   it, which is how thoroughly invisible it is. Writing
   `\b` into a regex produced a literal **backspace byte (0x08)** in the file, not `(backslash-b)`, so 15
   word-boundary anchors were silently dead - the patterns compiled fine and just failed to match.
   `\s` survived only because it is an *invalid* escape that Python leaves alone. **Build the
   backslash with `chr(92)`** when generating regex source this way, and always verify with
   `repr(open(f,'rb').read())`, never with grep - a terminal renders 0x08 by erasing the previous
   character, so the corruption is invisible.
2. **A global `_re.IGNORECASE` makes `[a-z-]` match uppercase**, which silently disabled the
   lowercase-only lookarounds in `_PARTITION_RE` - `LTrain` was then treated exactly like
   `constraint`. Fixed by scoping case-insensitivity to the word with `(?i:...)` and leaving the
   lookarounds case-sensitive.
3. `python3 run_api_judge.py --help` **crashes on this laptop** without `PYTHONIOENCODING=utf-8`
   (the docstring contains an arrow; the console is cp1252). Known trap, now hit by one more file.

### 29.5 ▶ OPEN: `configures` direction needs a HUMAN DECISION, not a guard

`configures` is defined ExperimentalSpecification -> Experiment. **Every model writes
Model -> setting-value** (`(Random Forest) configures (30 estimators)`), wrong in 100% of observed
uses. This is deliberately NOT guarded, because unlike `splitFrom` it is not a simple inversion -
the object is a VALUE, not an Experiment, so swapping the arguments does not produce a valid triple
either. Two options, and the choice is the user's:
* **(a) Change the ontology** so `configures` reads Model/Experiment -> ExperimentalSpecification.
  Matches what every model naturally produces and what the prompt's own examples imply, and would
  turn ~12 rejected triples per dense paragraph into valid ones.
* **(b) Reject the inverted form.** Ontology-faithful, but deletes genuinely useful hyperparameter
  facts - ministral's `roadwaylight2018#21` alone would lose 9 of its 11 triples.
The evidence points at (a): when every independent model produces the same shape, the schema is the
more likely thing to be wrong. Not acted on unilaterally.

### 29.6 Still open after this session

1. **scinex side of the four models** - CEO only. Needs OpenRouter spend (~$0.9 of the $1.73 left);
   **the user asked to discuss OpenRouter spending before any is committed.** Not started.
2. `RESULTS_REPORT.md` regeneration + a C1-C6 section in `build_results_report.py`.
3. `configures` decision (§29.5).
4. Older judged samples (claude-opus-5, gemma3-12b, qwen3-8b-v2) still describe 20 papers, not 22.
5. **Every number on disk predates these two guards** - as with every previous guard round.

## Session 28 — C1-C6 JUDGING COMPLETE: all 40 batches, 248 judgements, four extractors (2026-09-05)

> **⭐ START HERE. The C1-C6 evaluation Session 27 set up is now FINISHED.**
> Numbers and tables: `results.md` "C1-C6 ONTOLOGY-QUALITY EVALUATION" (Table 10 + Findings 1-6).

### 28.1 What was done

Judged every remaining bundle in `ontology_eval/` in-session (never via OpenRouter, per the standing
constraint). Went from 3/40 batches to **40/40**:

| model | triples (C1/C3/C4) | paragraphs (C2/C5) | papers (C6) |
|---|---|---|---|
| qwen3-235b | 4/4 | 4/4 | 2/2 |
| gptoss-120b | 4/4 | 4/4 | 2/2 |
| gemma4-31b | 4/4 | 4/4 | 2/2 |
| ministral-14b | 4/4 | 4/4 | 2/2 |

Full sample sizes as the user asked: **40 triples + 16 paragraphs + 6 papers per model = 248
judgements.** All collected to `verdicts.csv`. Reproduce the table with:
`python3 ontology_eval.py report --model qwen3-235b gptoss-120b gemma4-31b ministral-14b`

### 28.2 THE RESULT

| extractor | triples | para coverage | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|
| qwen3-235b | 395 | 16.4% | 78.8% | 31.2% | 60.0% | 45.0% | 28.1% | **50.0%** |
| **gptoss-120b** | 583 | 42.9% | **88.8%** | **34.4%** | **75.0%** | **82.5%** | 31.2% | 25.0% |
| gemma4-31b | 684 | 51.7% | 85.0% | 28.1% | 72.5% | 62.5% | 25.0% | 16.7% |
| ministral-14b | **1,709** | **69.1%** | 63.7% | 31.2% | 55.0% | 38.8% | **34.4%** | **0.0%** |

Triple counts verified from disk (`output/*/kg/relation/<slug>/triples.json`, 22/22 papers each).

**Three findings that only exist because the criteria were split:**

1. **Models differ on TYPING, not on concept identification.** C1 spans 1.4x (63.7-88.8), **C4 spans
   2.1x (38.8-82.5)**. The concepts are mostly real and present; domain/range conformance is what
   separates good from bad. The old single rubric collapsed these into one number.

2. **VOLUME DOES NOT BUY COMPLETENESS - the headline.** ministral-14b makes **4.3x** qwen3-235b's
   triples and fires on **4.2x** more paragraphs, and its completeness is **flat**: C2 31.2% vs
   31.2% (identical), C5 34.4% vs 28.1% (+6.3 pts). All four models sit in a 25-34% band on both.
   The extra volume goes on restating captured facts, enumerating pairwise combinations, and mining
   boilerplate - not on the paragraphs being missed. Every extractor leaves ~2/3 of each paper
   unrecorded, and they leave *different* thirds.

3. **C6 falls monotonically with volume, perfect rank correlation.** 395 triples -> 50.0%;
   583 -> 25.0%; 684 -> 16.7%; 1,709 -> **0.0% (all 6 papers CONTRADICTORY)**. More triples = more
   chances to conflict, concentrated in three mechanisms: superlatives with the scope qualifier
   stripped, metric nodes bound to two values because the class label was dropped, and
   surface-variant splitting letting incompatible claims attach to what should be one node.

### 28.3 ⚠ C6 SCORING ARTIFACT - do not report C6 alone

An empty triple set cannot contradict itself, so it scores CONSISTENT (1.0) **vacuously**.
qwen3-235b's `textaug2023` (0 triples) is exactly that and is 1 of the 6 papers behind its 50.0%.
**C6 systematically rewards extracting nothing** and is only interpretable next to C2/C5. Flagged in
the reply file itself so the artifact is auditable, not buried.

### 28.4 Predicate failures that are systematic, not per-model

* `employs` (Organisation->Person) - matched on the English verb for method-uses-method. **The only
  two correct uses in 160 judged triples were ministral's, and both came from AUTHOR BIOGRAPHIES.**
* `configures` (ExperimentalSpecification->Experiment) - **direction wrong in 100% of observed uses**
  (gemma4 x2, ministral x2, plus 12 more inside judged paragraphs). Every model writes Model->setting.
  Either add a direction guard or reconsider the ontology's direction here.
* `splitFrom` - gptoss correct 2/2; gemma4, ministral and qwen3-235b all invert it (ministral 3/3
  wrong in gamlprop2025, 4x in linkpred2015).
* `evaluatedOn` - hardware (Raspberry Pi 3 Model B), other models, and evaluation protocols placed in
  the Dataset slot.
* `(HDBSCAN clustering) uses (route data preprocessing)` - **inverted identically by all models that
  produced it**. A shared failure mode, not a quirk.

### 28.5 ⭐ Boilerplate is mined as content, and C1-C4 structurally cannot see it

Triples scoring CORRECT on all three per-triple criteria, drawn from: author biographies
(`(SOKENDAI) locatedIn (Japan)`), IEEE download watermarks (`(KMITL) affiliatedWith (UniNet)`),
bibliographies, acknowledgements (`(Assoc. Pannawit Samatthiyadikun) affiliatedWith (data analysis)`
x4), and the ACM CCS classification block. The Session 26 boilerplate guard covers
CRediT/funding/declarations but **not** these. **A triple can be perfect on C1, C3 and C4 and still
be worthless** - the criteria judge a triple against its source sentence and never ask whether that
sentence belonged to the paper. Concrete guard worklist: skip author-bio blocks, IEEE/publisher
watermarks, bibliography entries, acknowledgements, and ACM CCS blocks at extraction time.

### 28.6 The dissociation, in one paragraph

ministral-14b on `roadwaylight2018#21`: **11 triples**, 8 of ~10 important concepts, C2/C5 both
PARTIAL - the highest-yield paragraph in the study. **All nine of its `configures` triples have the
ontology direction wrong.** High completeness, low correctness, same paragraph. That is exactly what
C1-C6 was adopted to expose and what a single precision number cannot represent.

### 28.7 Judging conventions used (apply these if the sample is extended)

Recorded so a later round is comparable, since the rubric does not fix them:
* **C2/C5 boundary**: COMPLETE = nothing important missed (incl. genuinely empty paragraphs);
  PARTIAL = roughly >=1/3 of the KG-worthy items captured, OR a thin paragraph whose one or two
  items were missed; POOR = <1/3 captured, or nothing extracted from a content-rich paragraph.
* **C1 vs C3**: a vague placeholder ("our method", "test set", "system", "bespoke dataset") is
  C1 PARTIAL *and* C3 TOO_BROAD - the same defect seen as identity and as granularity. C4 is then
  judged on relation + typing alone.
* **C4 PARTIAL vs INCORRECT**: PARTIAL = supported but loosely typed (one slot off, or implied
  rather than stated); INCORRECT = unsupported, direction swapped, or BOTH slots violating
  domain/range.
* **C6 CONTRADICTORY** requires (a) two assertions that cannot both be true, or (b) one predicate
  used with mutually incompatible ranges in the same paper. Duplication and surface-variant
  splitting alone = MINOR_ISSUES.

### 28.8 What is still open

1. **No second labeller for C1-C6, so no kappa.** Single judge (claude-opus-5). The old-rubric kappa
   (results.md Table 9) does not transfer. This is the largest methodological gap.
2. **CEO only** - the scinex side of these four extractors is still not run, so the Session 26
   ontology x capability interaction has no C1-C6 counterpart.
3. **n=6 papers for C6.** The monotone trend across four models (24 papers) is the claim; per-model
   point estimates are not quotable.
4. **`RESULTS_REPORT.md` is still stale** and `build_results_report.py` still has no C1-C6 section.
   Now unblocked - the numbers exist.
5. Older judged samples (claude-opus-5, gemma3-12b, qwen3-8b-v2) still describe 20 papers, not 22.

## Session 27 — NEW EVALUATION FRAMEWORK (C1-C6); corpus 20 -> 22; four new extractors (2026-09-03/05)

> **⭐ START HERE. This session changed the evaluation criteria and the corpus.**
> Read §27.1 (what changed), §27.2 (state on disk), §27.6 (exactly what to do next).

### 27.1 Two directives from the user, both structural

**(a) The evaluation criteria changed.** The old CORRECT/PARTIAL/INCORRECT rubric is **superseded**
by a six-criterion ontology-quality framework the user supplied:

| | criterion | meaning | reference |
|---|---|---|---|
| C1 | Concept Correctness | concepts are semantically correct and evidenced in the text | Zhang, Conia & Rago, IJCNLP-AACL 2025 |
| C2 | Concept Completeness | the ontology covers the important concepts in the text | Wilson et al., Semantic Web 14(6) 2023 |
| C3 | Concept Specificity | concepts sit at an appropriate granularity | Zhang et al. 2025 |
| C4 | Relation Correctness | the relation holds per the text and fits domain -> range | Zhang et al. 2025 |
| C5 | Relation Completeness | the important relations in the text are all extracted | Wilson et al. 2023 |
| C6 | Semantic Consistency | concepts/hierarchy/relations do not contradict | Wilson et al. 2023 |

**(b) The model line-up is fixed by the user:** extract with **Qwen3-235B, GPT-OSS-120B,
Gemma-4-31B, Ministral-14B** (all via OpenRouter), and **judge all of them with claude-opus-5**.

> ### ⛔⛔ HARD CONSTRAINT — DO NOT SPEND OPENROUTER CREDIT ON CLAUDE
> The user was explicit: *"claude opus 5 that we will run as a judge will be through my claude.
> NOT FROM OPEN ROUTER. DO NOT SPEND ANY MONEY OF OPENROUTER ON CLAUDE."*
> **Judging is done BY THE ASSISTANT IN-SESSION** — read the batch file, write the reply file.
> `anthropic/*` models exist on OpenRouter and must never be called.

### 27.2 State on disk

**Corpus is now 22 papers.** The user supplied the two previously-unobtainable PDFs
(`bulk-download/trip-planner.pdf`, `bulk-download/LPII.pdf`); both verified as the right papers,
copied to `papers/tripplanner2020.pdf` and `papers/linkpred2015.pdf`, parsed clean
(14 and 17 sections, titles manifest-verified). So the corpus is **the 20 papers from the original
list + the 2 substitutes** (`aiabstract2025`, `routepred2023`) added when those two were unobtainable.
Describe it that way; do not write "20 papers".

| extraction run | papers | triples |
|---|---|---|
| `relation/gemma3-12b` | 22/22 | 1,930 |
| `relation/ministral-14b` | 22/22 | **1,709** |
| `relation/qwen3-8b-v2` | 22/22 | 1,444 |
| `relation_scinex/qwen3-8b-v2` | 22/22 | 1,389 |
| `relation/qwen3-8b` | **20/22 — FROZEN BY DESIGN** | 1,185 |
| `relation_scinex/qwen3-8b` | **20/22 — FROZEN BY DESIGN** | 1,117 |
| `relation/gemma4-31b` | 22/22 | **684** |
| `relation_scinex/qwen3-235b` | 22/22 | 600 |
| `relation/gptoss-120b` | 22/22 | **583** |
| `relation/qwen3-235b` | 22/22 | 395 |
| `relation/claude-opus-5` | 22/22 | 315 |

**⚠ Why `qwen3-8b` (v1) must stay at 20 papers:** it is the "before" half of the §26.14 engineering
result (+8.3 pts from guards + bug fixes). Extending it would require running the *current* code,
which is exactly what that baseline predates, and would destroy the measurement. Report it as a
paired 20-paper comparison against v2 restricted to the same 20 papers.

**OpenRouter: \$1.27 spent of \$3.00, \$1.73 remaining.** All of it on extraction, none on Claude.

### 27.3 New tool: `ontology_eval.py` — the C1-C6 harness

**THREE harnesses, because the criteria do not share a unit of analysis.** This is the key design
point and it is not optional:

```
C1, C3, C4   judged PER TRIPLE     against its source sentence        (precision-like)
C2, C5       judged PER PARAGRAPH  against every triple drawn from it (recall-like)
C6           judged PER PAPER      against that paper's whole triple set
```

A completeness question cannot be asked of a single triple — what is missing is by definition not
in front of the judge. **Everything this project measured before Session 27 was precision only;
C2/C5 are a genuinely new axis.**

```bash
python3 ontology_eval.py triples    --model <slug> --n 40 --size 10   # C1, C3, C4
python3 ontology_eval.py paragraphs --model <slug> --n 16 --size 4    # C2, C5
python3 ontology_eval.py papers     --model <slug> --n 6  --size 3    # C6
#   -> assistant reads ontology_eval/<slug>/ceo/<kind>/NN_batch_NN.txt
#   -> assistant writes    .../replies/reply_NN_batch_NN.txt
python3 ontology_eval.py collect --model <slug> --kind triples
python3 ontology_eval.py report  --model qwen3-235b gptoss-120b gemma4-31b ministral-14b
```

**Four design decisions, each with a reason — do not silently revert them:**

1. **C3 is NOT scored ordinally.** `TOO_BROAD` and `TOO_NARROW` are two ways of being wrong, not a
   better and a worse one, so both score 0.0 (`POINTS_BY_CRIT`). Scoring one at 0.5 would assert
   that over-general concepts are half-acceptable. The direction is kept in the label for diagnosis.
2. **Paragraph matching normalises whitespace.** Models re-wrap the sentence they quote, so a raw
   substring match found **0 of 12** triples on a test paper; `_norm()` finds **12 of 12**, 97.7%
   corpus-wide.
3. **C2/C5 sampling is STRATIFIED on "did this paragraph produce any triple", then reweighted.**
   Sparse extractors yield nothing from most of the corpus, so a pure random draw wastes the sample
   (seed 11 drew 0 of 20 for qwen3-235b). `strata.json` records the true shares.
4. **Predicate definitions are injected from the ontology** (`_schema_for`). Without them the
   bundles printed `(not recorded)` and C4 collapsed into "does this English relation sound
   plausible?" — losing the domain/range half of the criterion.

### 27.4 ⭐ A finding that already exists, before any judging

Paragraph coverage — the share of the 567 paragraphs from which a model produced **at least one**
triple — varies 4x and runs **opposite** to precision:

| model | paragraphs producing >=1 triple | silently skipped |
|---|---|---|
| qwen3-235b | 16.4% (93/567) | **84%** |
| gptoss-120b | 42.9% (243/567) | 57% |
| gemma4-31b | 51.7% (293/567) | 48% |
| ministral-14b | 69.1% (392/567) | 31% |

**The precise models are precise partly by declining to extract at all.** C2/C5 exist to price that,
and this is the strongest reason the new criteria were worth adopting.

### 27.5 ⭐ First judged result (partial): qwen3-235b, 30 of 40 triples

| criterion | score (n=10, batch 1 only via `report`) |
|---|---|
| C1 Concept Correctness | 75.0% |
| C3 Concept Specificity | 60.0% |
| C4 Relation Correctness | **45.0%** |

**C4 is far weaker than C1, and the split is the point.** The concepts are usually real and present;
the *typing* is what fails. Three recurring failure modes, all invisible under the old single rubric:

* **Direction reversed** — `(RT-DETR, uses, object detection)` where the sentence says "object
  detection **using** RT-DETR"; `(time-series data, splitFrom, training set)`;
  `(University of Washington, affiliatedWith, Joseph Redmon)` (Agent->Organisation is the defined
  direction); `(dataset, splitFrom, training and testing sets)`.
* **Domain/range violation** — `(dataset, comprises, video footage)` when `comprises` requires BOTH
  sides to be processes; `(RT-DETR, employs, IoU-aware query selection)` when `employs` is defined
  `Organisation -> Person`; `(YOLOv8, cites, Ultralytics)` when `cites` is Paper->Paper.
* **Role confusion** — `(performance, evaluates, BERT's F1 score)` puts the metric in the object slot.

Contrast `(accuracy, evaluates, 0.974)`, which is exactly the intended shape and scores CORRECT on
all three.

### 27.6 ⭐ EXACTLY WHERE TO RESUME

**Judging is 3 of 40 batches done.** Everything else is extracted and ready.

| model | triples (C1/C3/C4) | paragraphs (C2/C5) | papers (C6) |
|---|---|---|---|
| qwen3-235b | **3/4 done** | 0/4 | 0/2 |
| gptoss-120b | 0/4 | 0/4 | 0/2 |
| gemma4-31b | 0/4 | 0/4 | 0/2 |
| ministral-14b | 0/4 | 0/4 | 0/2 |

**Next action:** judge `ontology_eval/qwen3-235b/ceo/triples/04_batch_04.txt`, then work through the
remaining models. Bundles for all four models and all three harnesses are **already generated** — no
regeneration needed unless the sample size changes.

Procedure per batch: `cat` the batch file, judge each item against the rubric printed in batch 01,
write `replies/reply_<same name>.txt` as one JSON object, then `collect` and `report`.

**The user asked for full sample sizes on every criterion** (not a reduced set), i.e. 40 triples +
16 paragraphs + 6 papers per model = 160 + 64 + 24 judgements.

### 27.7 Open items beyond the judging

1. **Judged samples for the OLDER runs describe 20 papers, not 22.** `claude-opus-5`, `gemma3-12b`,
   `qwen3-8b-v2` etc. were sampled before the corpus grew. Redraw before mixing old strict/lenient
   numbers with new C1-C6 numbers, or state the difference.
2. **`RESULTS_REPORT.md` is stale** — it documents the 20-paper corpus and the old rubric, and knows
   nothing about the four new extractors or C1-C6. Regenerate with `build_results_report.py` after
   the C1-C6 numbers exist, and add a C1-C6 section to the builder.
3. **The scinex side of the new models is not run** — only CEO. The §26.16 ontology x capability
   interaction currently rests on qwen3-8b-v2 and qwen3-235b.
4. **KGE / TF-IDF question still unresolved** (§10 of RESULTS_REPORT) — different corpus, separate
   decision about whether that track stays in the paper.

### 27.8 Do NOT re-derive these

* **No OpenRouter spend on Claude** (§27.1). Judging is in-session.
* `--model-slug` must never contain `:` or `/`.
* **Ollama: `"think": False` is load-bearing.** Without it qwen3 spends the whole `num_predict`
  budget reasoning and returns **empty content** — 93/515 empty replies cost a 7-hour run in §26.14.
* **Reasoning models generally**: a budget consumed by reasoning yields empty output that is
  indistinguishable from "the model found nothing". Check for empty content + populated reasoning
  before believing a null result. (Hit 3x: gpt-oss at max_tokens=64, qwen3 at 3072.)
* `claude_corpus_report.py --out` now defaults per model slug; it used to clobber Claude's JSON.
* Key resolution in `run_api_extract.py` is **endpoint-aware** — a fixed preference order sent the
  Groq key to OpenRouter and 401'd.
* Model ids: **Gemma 4 is `google/gemma-4-31b-it` (31B, not 32B)**; Ministral is
  `mistralai/ministral-14b-2512`.

---

# Hands-Off Notes (earlier sessions) — Model Swap Session

**Date:** 2026-06-09
**Context:** Switching the KG extraction LLM off Qwen3-32B because the VM's GPU (20GB VRAM) can't run it.

> Read `Claude.md` first for the full project overview. This file only covers the model-swap work from this session.

---

## What we changed

### 1. Model swapped: Qwen3-32B → Qwen3-14B
We tried several models before landing on Qwen3-14B:

| Model | Result |
|---|---|
| Qwen3-32B (original) | OOM — too big for 20GB |
| google/gemma-3-27b-it | Rejected — HuggingFace repo is **gated** (needs license acceptance) |
| Qwen3-30B-A3B | Loaded but **froze during generation** — MoE loads all 30B params into VRAM (~19.3/19.8GB used), leaving no room for KV cache/generation |
| **Qwen/Qwen3-14B** | **CURRENT** — ~7–8GB at 4-bit, ~12GB headroom. Largest practical Qwen3 dense model for this GPU |

Changed in:
- `kg_extraction/llm_extractor.py` — `MODEL_NAME = "Qwen/Qwen3-14B"`
- `kg_extraction/fixed_extractor.py` — `DEFAULT_MODEL = "Qwen/Qwen3-14B"`
- `Claude.md` — updated all model references + run commands + added a "GPU notes" section

### 2. Fragmentation fix
Hit `torch.OutOfMemoryError` that was actually fragmentation (only 8.5GB allocated but reserved-unallocated was 10GB). Added to the top of **both** extractors:
```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

### 3. Greedy decoding fix
Qwen3's model `generation_config` defaults to `do_sample=True`, which overrode our pipeline's `do_sample=False` at runtime (and threw `top_k not valid` warnings). This is bad for deterministic structured extraction and was suspected in the 30B freeze. Added after `model.eval()` in **both** extractors:
```python
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None
```

### 4. CPU-offload "freeze" fix (2026-06-10)
A `kg_main.py --extractor llm` run appeared frozen "after calling the model". It was **not** a hang: `device_map="auto"` silently offloaded layers to CPU on the 20GB vGPU (H100-20C), so a 14B generation was running on CPU (276% CPU, 37GB RAM, only ~449MiB VRAM) — effectively never finishing, and it blocked the foreground terminal.

Fix — force the whole model onto GPU 0 so it either loads fully on-GPU or fails loudly (never silently offloads):
```python
device_map={"": 0},   # was device_map="auto"
```
Changed in **both** `kg_extraction/llm_extractor.py` (line ~291) and `kg_extraction/fixed_extractor.py` (line ~393).

Verified with Qwen3-14B: loads in ~34s, all layers on GPU 0, **10GB / 20GB VRAM** (~10GB headroom for KV cache), generation completes in ~9s. The `['temperature'] not valid` warning is harmless (greedy-decode config being ignored), unrelated to the freeze.

> If a future run looks frozen again: `nvidia-smi`. Low VRAM (<2GB) + high CPU/RAM on the python PID = CPU offload, not a hang. Kill the PID to free the terminal.

### 5. expandable_segments removed — fatal on vGPU (2026-06-10)
After fix #4, `kg_main.py` hit `RuntimeError: CUDA driver error: operation not supported` during checkpoint loading. Cause: the session-3 "fragmentation fix" `os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"` uses CUDA **virtual-memory** APIs that vGPUs (H100-20C here) don't support — it fails on *any* CUDA allocation, even `torch.zeros(1000, device='cuda')`. (Standalone test scripts passed only because they didn't import the extractors / set the flag.)

Fix: **removed** the `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...)` line (and now-unused `import os`) from both `kg_extraction/llm_extractor.py` and `kg_extraction/fixed_extractor.py`. Verified: importing the extractor then allocating on CUDA now works. Do NOT re-add expandable_segments on this hardware. Also remove the matching note in `Claude.md` (line ~112).

---

## Current problem / state
- **Freeze is fixed (see #4 above). Model now verified to load + generate on GPU.** Still pending: a full end-to-end pipeline run producing/evaluating triples.
- Qwen3-14B is **not in the HF cache yet** — first run will download (~28GB) before loading. Don't mistake the download/load wait for a freeze (see Claude.md: loading 14B into VRAM takes a few min).
- The previous Qwen3-30B-A3B process needs to be **killed** if still running.

## Current work
- Run the pipeline once with Qwen3-14B and confirm:
  1. Model loads without OOM
  2. Generation completes (no freeze) and produces triples
  3. Output quality is acceptable vs. the old Qwen3-32B BERT results (was 80% precise / 85% lenient precision, 30 triples)

Test command:
```bash
python kg_main.py --paper BERT --extractor fixed --entity-csv Entity_-_BERTv2.csv --model Qwen/Qwen3-14B
```

---

## Session 26 — GPT/Gemini chat route abandoned; two pipeline bugs FIXED and measured; more models via API + a second local family (2026-08-28)

**User's report:** ChatGPT and Gemini could not produce usable triples through the chat-paste route.
`chat_upload/gpt/replies/` and `chat_upload/gemini/replies/` are both **empty** — nothing came back.
**Do not spend more time on the free-chat-UI route.** §25.4d already flagged it as a sanity check
rather than a paper number (batching changes the experimental condition, and the model version is not
controllable); it is now also the route that failed in practice. The API route replaces it and is
strictly better: a pinned model id, one paragraph per call (so it *is* comparable to the qwen3:8b and
claude-opus-5 runs), and no pasting.

### 1. ✅ FIXED — `kg_builder._is_valid_entity()` deleted correct triples (§25.4b worklist item 5)

The fix from §25.4b is applied. The signature is now
`_is_valid_entity(text, predicate=None, role="subject")`:

* a **measurement value** (`_VALUE_RE` — `88.5`, `0.923`, `14.76 ms`, `97%`, `1.2e-4`) is accepted as
  the **object** of a **value predicate** (`evaluates`, `achieves`, `reports`) *only*. A bare number is
  still rejected as a subject, and still rejected as the object of every other predicate — the filter
  was not loosened generally, which is what makes this safe to apply retroactively.
* all-caps acronyms (`SF`, `SD`, `F1`) are exempt from the `len < 3` floor.

Verified on 11 hand-checked cases (including the negative ones: `(0.923, uses, …)` still rejected,
`88.5` as a *subject* still rejected).

**Measured, not assumed.** Claude's run stores its raw replies, so it was re-ingested through the
corrected builder with **no new model calls** — `claude_extract.py ingest` over all 18 papers:

| | before | after |
|---|---|---|
| claude-opus-5 triples | 268 | **279** (+11) |
| `evaluates` | 1 | **6** |
| `comprises` | 25 | **28** |

The recovered triples are precisely the ones the bug was built to delete:
`(precision, evaluates, 0.923)`, `(accuracy, evaluates, 0.974)`, `(IoU, evaluates, 0.612)`,
`(RMSE, evaluates, 14.76)`, `(PNext, comprises, SF/SD/SP)`, `(the XGB algorithm, comparesAgainst, RF)`.

**⚠ Consequence for the gold set:** `evaluates` sits at 0% strict in the n=240 labels *because the bug
deleted its correct instances and left only the malformed wordy ones to be judged*. It must be
**re-judged** before it stays on the "always wrong" list. The other 10 zero-strict predicates are
unaffected.

**⚠ qwen3:8b's 1,101 triples cannot be recovered this way.** That run went through `kg_main`
(`run_local.py`), which never stored raw replies — only `triples.json`. Confirmed: the only
`responses.jsonl` on disk belong to `claude-opus-5` and `llama31-8b`. Its equivalent losses need a
re-extraction. **The two columns are no longer strictly like-for-like on this axis** — say so when
reporting.

`(root-mean-square error (RMSE), evaluates, 9.23)` also came back, which is the §25.5
`_contains_term()` parenthetical-subject fix showing up in output for the first time (worklist item 6,
previously unmeasured).

### 2. Second local model — llama3.1:8b (Meta), full corpus, IN FLIGHT at session close

Family diversity is the point: qwen3:8b (Alibaba) and claude-opus-5 (Anthropic) are the only two
extractors so far. `llama3.1:8b` is a third family and needs no key.

```bash
export PYTHONIOENCODING=utf-8
python run_api_extract.py --all --provider ollama --model llama3.1:8b        --model-slug llama31-8b --extractor relation --ontology ceo        --max-tokens 3072 --ingest          # log: llama31_extraction.log
```

Smoke-tested first on 3 paragraphs of ugmo2024 (19 triples emitted, 5 kept) and the smoke directory
deleted. **Measured rate: ~3 calls/min → ~2.7 h for 486 paragraphs.** Slower than the smoke test
suggested because llama3.1 is verbose. It resumes (`--resume` is the default): re-running the same
command picks up from `responses.jsonl` if it is interrupted.
**Check on it with:** `tail -3 llama31_extraction.log` and
`wc -l output/*/kg/relation/llama31-8b/responses.jsonl`, then
`python3 claude_corpus_report.py --model-slug llama31-8b`.

### 3. The API route is now the fast lane — and both keys are free

`run_api_extract.py` was already model-agnostic; three things were added so it can be pointed at a
new provider in one command.

* **`.env` is now loaded** (stdlib, no python-dotenv). Keys go in `.env`, never on the command line.
* **Per-provider key resolution** — `_key()` tries `OPENAI_COMPAT_API_KEY`, `GROQ_API_KEY`,
  `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY`, then `OPENAI_API_KEY`. So a Groq key
  and a Gemini key coexist in one file with no shell juggling.
* **`check_provider.py` (new)** — validates a key *before* committing 486 calls: finds the key, lists
  the models the key can actually see (**looked up, not guessed — model ids drift**), picks the
  strongest available, and prints the exact smoke-test + full-corpus + report commands.

```bash
python3 check_provider.py groq        # or: gemini / openai
```

Why these two providers: **Groq** hosts `openai/gpt-oss-120b` — a real OpenAI-family model with a
pinned id, which is the defensible replacement for the ChatGPT run that failed — plus
`llama-3.3-70b-versatile` and `kimi-k2`. Free tier ~1,000 requests/day, and it is fast enough to do
the whole corpus in well under an hour. **Google AI Studio** gives the genuine Gemini number.
486 calls fit inside both free daily allowances.

### 4. Files

* `kg_extraction/kg_builder.py` — `_is_valid_entity()` predicate/role aware; `_VALUE_RE`,
  `_VALUE_PREDICATES`, `_is_value()` added.
* `run_api_extract.py` — `_load_dotenv()`, `_key()`, wired into `main()`.
* `check_provider.py` — **new**, key/model preflight.
* `.env` — commented placeholders for `GROQ_API_KEY` / `GEMINI_API_KEY`.
* `output_backup_claude_preingest/` — the 268-triple `triples.json` files, kept so the re-ingest is
  reversible and the delta is auditable.
* `llama31_extraction.log` — the in-flight run.

### 5. NOT done / next session

1. **llama3.1:8b run** — confirm it finished (18 papers), then
   `python3 claude_corpus_report.py --model-slug llama31-8b` and add a column to Table 7.
2. **Groq + Gemini runs** — `check_provider.py <provider>` then the two commands it prints.
3. **Re-judge `evaluates`** — its 0% strict was an artefact of the builder bug (§26.1).
4. **Re-extract qwen3:8b** if its column needs to be like-for-like after the two guard fixes (~2.7 h
   GPU, and it would supersede every §23–§25 number for that model).
5. Guard worklist items (a)-(d) from §25.7 are still open: ontology-class-name objects,
   cross-reference objects, post-parse domain/range check, skip CRediT/Funding sections.
6. **Judging is still the bottleneck for every model except qwen3:8b.** With three or four extractor
   families on disk, one judge pass covering all of them (a model in none of those families) is worth
   more than another extraction run.


### 6. ⭐⭐⭐ THE BIG ONE — both extractors judged by ONE judge, and the first κ

The chat-UI judging route was replaced with an API driver and **it worked in minutes, not days.**

**`run_api_judge.py` (new)** drives the existing `judge_paste.py` bundles over HTTP and writes the
replies where `judge_paste.py collect` already looks, so nothing downstream changed. One deliberate
improvement over the chat route: a chat conversation carries the JUDGING RULES from its first
message and drifts as context grows — here the rubric is the **system prompt on every call**, so
every batch is judged under byte-identical instructions.

```bash
python3 run_api_judge.py --slug gemini-judge --provider gemini --model gemini-3.6-flash
python3 judge_paste.py collect --csv gold/claude_sample_for_judge.csv --slug gemini-judge \
        --out gold/gemini_verdicts.csv
python3 gold_report.py --labels gold/gemini_verdicts.csv --model claude-opus-5 --bootstrap 2000
```

**Result — claude-opus-5 relation/CEO, n=100, judged by gemini-3.6-flash:
weighted strict 89.1% [82.3, 94.5], lenient 97.9%. 86 CORRECT / 13 PARTIAL / 1 INCORRECT.**

Then the confound was closed: the n=240 qwen numbers were judged by **Claude**, so a Claude-vs-qwen
comparison across them mixed extractor and judge. The same 240 qwen ids were re-judged by
**gemini-3.6-flash** (verdicts blanked first — `cmd_batches` never emits the verdict column, checked).
110 of 240 came back before the quota wall (§26.7):

| extractor | ontology | n | weighted strict | lenient |
|---|---|---|---|---|
| claude-opus-5 | CEO | 100 | **89.1%** | 97.9% |
| qwen3:8b | CEO | 58 | 25.6% | 50.4% |
| qwen3:8b | scinex | 52 | 11.7% | 34.2% |

Same paragraphs, same prompt, same guards, same judge — **only the model differs.** ~3.5x.
Every predicate at 0% strict for qwen scores >=66% for Claude *or Claude never emits it*:
`affiliatedWith`, `employs`, `configures`, `publishedIn`, `cites` are **absent** from Claude's output.

**And the by-product is the κ this project has wanted since §19:** Claude and Gemini both labelled
the same 110 triples, independently. Raw agreement 59.1%, **Cohen's κ 0.342 (3-class)**;
CORRECT-vs-not agreement 83.6%, **κ 0.480 (binary)**. The confusion matrix says the disagreement is
almost entirely the PARTIAL boundary — Claude's PARTIAL → Gemini's INCORRECT **25 times**, while
joint INCORRECT is 44 and Gemini never promoted a Claude-INCORRECT to CORRECT. **That is the
empirical case for reporting strict AND lenient rather than one number.** It also makes Table 8
stronger: Gemini is the harsher judge (it scores qwen 13.6% where Claude scored 24.5%) and Claude's
extraction still earns 89.1% from it.

Files: `gold/gemini_verdicts.csv` (Claude's 100), `gold/gemini_verdicts_qwen.csv` (qwen's 110),
`gold/claude_labels_240.csv` (Claude's 240 merged, for `agree`), `gold/qwen_sample_for_judge.csv`
(blank-verdict 240), `gold/report_claude_gemini.json`, `gold/report_qwen_gemini.json`,
`gold/agreement_claude_gemini.json`.

### 7. ⛔ FREE-TIER LIMITS — measured, do not re-derive

| provider | limit that bites | what it means for a 486-call corpus run |
|---|---|---|
| **Gemini** free | **20 requests/day/model** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`) | Extraction is impossible (486 calls). **Judging is fine** — batch 10-25 triples per call. |
| **Groq** free | **8,000 tokens/minute** (`on_demand` tier) | Our calls are ~6.2k tokens → ~1 call/min → **~8 h**. Slower than the local GPU. Not viable. |

Both were hit for real, not read off a docs page. The Gemini quota is **per model**, so switching to
another pinned id (`gemini-3.7-flash`, `gemini-3.5-flash`, ...) grants a fresh 20 — but **do not judge
one sample with two different models**; judge identity must be constant across a comparison.
**To finish the remaining 130 qwen labels:** rebuild at a bigger batch size so the whole set fits in
one model's daily allowance —
`python3 judge_paste.py batches --csv gold/qwen_sample_for_judge.csv --slug gemini-judge-qwen2 --size 25`
(10 requests), then `run_api_judge.py --slug gemini-judge-qwen2 --provider gemini --model <fresh id>`.
Re-judging the Claude 100 with that same fresh id costs 4 more requests and keeps both rows on one judge.

Two API fixes needed along the way, both real:
* **Groq sits behind Cloudflare and 403s Python's default user agent** (`error code: 1010`).
  `USER_AGENT` is now sent by all four adapters in `run_api_extract.py` and by `check_provider.py`.
* **`gemini-2.5-flash` is LISTED to a new key but returns `NOT_FOUND` when called** ("no longer
  available to new users"). The model list is not a list of usable models — hence the smoke test.
  `check_provider.py` now prefers pinned `gemini-3.x` ids over the `-latest` aliases (aliases drift,
  and a paper number must name the exact model that produced it).

### 8. Corpus 18 → 20, by SUBSTITUTION (user's call — state this in the write-up)

`tripplanner2020` and `linkpred2015` are **not obtainable** — both paywalled Springer chapters, and
**neither is in `bulk-download/`** (checked by title-Jaccard against every file: the 8 bulk-download
files that match a corpus paper all match at 1.0, and neither missing title matches anything).
On the user's instruction ("just pick 2 that hasn't been used"), two unused papers **by the same
group** were substituted from `bulk-download/`:

| new id | title | venue |
|---|---|---|
| `aiabstract2025` | Detecting AI-Generated Scientific Abstracts Using Galactica and Graph Neural Networks | ISCON 2025 (IEEE) |
| `routepred2023` | Route Prediction from GPS Trajectory and Road Data | ICCAE 2023 (IEEE) |

**⚠ The corpus is therefore no longer "the 20 papers on the user's citation list" — it is 18 of them
plus 2 substitutes.** `papers/manifest.csv` records this: the 2 originals are marked
`access=unobtainable` with a `note`, and the 2 substitutes carry `SUBSTITUTE for <id>`. Say it plainly
in the paper's corpus description rather than quietly reporting "20 papers".

Both parsed clean on the first try, titles verified, **and `BERT.pdf` re-checked as a regression
(44 sections / 7,254 body words, unchanged)**:

| id | sections | body words | headings | enriched entities |
|---|---|---|---|---|
| `aiabstract2025` | 9 | 4,160 | Abstract...References, all found | **171** |
| `routepred2023` | 7 | 2,865 | Abstract...References, all found | 41 |

171 entities for the NLP/GNN paper is at the top of the corpus range — it is a CS paper, so CS-NER
covers it well, which is the §22.4 CS/non-CS split showing up again.

**⚠ Consistency caveat:** these 2 papers are extracted with the §26.1 builder fix applied, while the
other 18 qwen papers predate it. The only way to make all 20 consistent is a full qwen re-extraction
(~3 h GPU). Recommended before the paper's final numbers; noted here so the difference is not
discovered later as a mystery.

**✅ BOTH MODELS HAVE NOW EXTRACTED BOTH NEW PAPERS — the corpus is complete at 20/20.**

| extractor / model | triples | papers |
|---|---|---|
| `relation/qwen3-8b` (CEO) | **1,185** | 20 |
| `relation_scinex/qwen3-8b` | **1,117** | 20 |
| `relation/claude-opus-5` (CEO) | **296** | 20 |

New-paper contributions — qwen: `aiabstract2025` 47 CEO / 57 scinex, `routepred2023` 37 / 29
(4/4 runs ok, 78 requests, **11 truncated = 14.1%** vs 5.2% on the original 18 — denser paragraphs
hit the 3,072-token output cap more often; settings were left identical to the other 18 on purpose).
Claude: 12 and 5, **all 17 passing the guards with zero rejections**.

The **25% volume ratio held on 39 paragraphs neither model had seen** (Claude 0.56 triples/paragraph
vs qwen 2.26). That is worth stating in the paper: the volume gap is a property of the models, not
an artefact of the original paper selection.

### 9b. Negative results from this session — recorded so they are not re-attempted

* **Free chat-UI extraction (ChatGPT / Gemini web) — ABANDONED.** Bundles were built for both
  (`chat_upload/{gpt,gemini}/`, 49 files each) and **neither returned usable triples**; both
  `replies/` folders came back empty. Already weak for two reasons that stand regardless: batching
  ~12 paragraphs per message is a *different experimental condition* from one-paragraph-per-call,
  and a free UI can silently switch model version mid-session. **Do not restart this route.**
* **Free API tiers for extraction — NOT VIABLE (measured).** Groq 8,000 tok/min → ~1 call/min →
  ~8 h for 525 calls, slower than the laptop GPU. Gemini 20 requests/day/model → impossible.
  Note the asymmetry: **the same quotas are fine for judging** (10-25 triples batched per call).
* **`llama3.1:8b` size-matched control — started, stopped by the user after 1 paper.** ~3 calls/min
  (~2.7 h projected). Partial output deleted, log renamed `llama31_extraction_ABANDONED.log`.
  Model still installed. **This is the one open methodological gap a reviewer could name** —
  qwen3:8b and claude-opus-5 differ in both size and family, so "strength" and "identity" are
  currently confounded in the §26.6 result.
* **Model-id traps:** `gemini-2.5-flash` is LISTED to a new key but returns `NOT_FOUND` when called;
  `gemini-3.7-flash` returned `503 UNAVAILABLE` on every retry. `gemini-3.6-flash` worked and became
  the judge. **A model listing is not a list of usable models — always smoke-test one call.**

### 9c. New artefact: `RESULTS_REPORT.md` (+ `build_results_report.py`)

A single self-contained document for hand-off / slide generation / the paper writer: project
overview, pipeline, the constraint ladder, both ontologies, both evaluation tracks, results,
methodology, engineering findings, negative results, limitations, next steps, reproduction commands.
**It is generated, not written** — `build_results_report.py` reads every per-triple number straight
from `output/` and `gold/`, so the document cannot drift from the artefacts. Re-run the builder after
any new result instead of editing the markdown. `README.md` now points at it.


### 10. ⭐⭐ SECOND JUDGE — the headline survives an independent cross-check (2026-08-28)

The κ in §26.6 is on *qwen's* triples. Claude's own 89.0% still rested on **one** judge. Closed now.

**Correcting an earlier call of mine:** Groq was written off after its 8,000 tok/min limit made
*extraction* infeasible (525 calls x ~6.2k tokens). **Judging batches are ~1,400 tokens** — the whole
100-triple bundle is ~13.7k — so the limit barely touches judging. The `judge_upload/gpt-judge/`
bundle built for the dead chat route was reusable as-is.

```bash
python3 run_api_judge.py --slug gpt-judge --provider openai-compat         --base-url https://api.groq.com/openai/v1 --model openai/gpt-oss-120b         --max-tokens 3072 --sleep 40 --retries 5 --backoff 15     # 10 batches, 6.9 min, 0 failures
python3 judge_paste.py collect --csv gold/claude_sample_for_judge.csv --slug gpt-judge         --out gold/gptoss_verdicts.csv
```

`--sleep 40` is load-bearing: Groq reserves `input + max_tokens` against the TPM budget, so
~1.6k + 3072 = ~4.7k per call means roughly one call per minute.

| judge | family | weighted strict | lenient | sampled strict |
|---|---|---|---|---|
| `gemini-3.6-flash` | Google | **89.0%** | 97.9% | 86.0% |
| `openai/gpt-oss-120b` | OpenAI | **82.8%** | 92.6% | 85.0% |

**Two independent judges, near-identical sampled precision (86.0 vs 85.0).** Report the headline as a
**range, 82.8-89.0% weighted strict**, not a point estimate. Still 3-4x qwen3:8b under one protocol.

**Second κ, and it is much stronger than the first:**

| | Claude's triples (gemini vs gpt-oss) | qwen's triples (claude vs gemini) |
|---|---|---|
| raw agreement | **86.0%** | 59.1% |
| κ (3-class) | **0.457** | 0.342 |
| binary agreement | **91.0%** | 83.6% |
| **κ (binary)** | **0.637** (substantial) | 0.480 (moderate) |

81 of 100 called CORRECT by both. **Judges agree far more about good extraction than about bad
extraction** — a triple that clearly restates its source sentence leaves nothing to disagree about,
while a loosely-typed one lands in the PARTIAL zone. **Low inter-rater agreement is itself a symptom
of poor extraction.** Worth reporting as a finding, not just a methodology footnote.

Files: `gold/gptoss_verdicts.csv`, `gold/report_claude_gptoss.json`,
`gold/agreement_claude_twojudges.json`.


### 11. ✅ GUARD WORKLIST CLOSED — three of the four open items implemented and measured

The §25.7 / §26.9 worklist items (a), (b) and (d) are done. All three are **post-parse code**, so
they apply to every model equally and cannot be prompt-engineered away.

**(a) Ontology class-name guard.** `_schema_class_names()` derives the ontology's CLASS names from
the schema's own `domain → range` strings (23 for CEO) and any object equal to one is rejected.
This was the §25.2 "schema placeholder leak", confirmed on two different predicates, so it rejects
**any** class name rather than patching one relation. Real hits on the corpus:
`(training set, splitFrom, dataset)`, `(cross-validation, uses, model)`,
`(APPLICATION SIMULATION, produces, dataset)`.

**(b) Cross-reference guard.** Objects that point into the paper rather than naming a thing.
Split into two word classes after a false-positive check:

| class | words | rule |
|---|---|---|
| pointer words | table, figure/fig, section, appendix, equation/eq | cross-reference **even without a number** |
| ordinary nouns | algorithm, scheme, listing, chapter, paragraph, step | only when **numbered** |

The boundary is `(?![\w-])`, **not** `\b` — with `\b`, *figure-ground segmentation* and
*Table-driven parser* were both wrongly rejected. Verified on 19 cases (10 reject / 9 keep).
Real hits: `(the refined model, evaluatedOn, Figure 4)`, `(XGB algorithm, evaluatedOn, Table 4)`,
`(dataset, extractedFrom, Section B. DATA PREPROCESSING)`.

**(d) Boilerplate-section skip.** `_is_boilerplate_section()` — author contributions / CRediT /
declarations / competing interests / funding / data & code availability / ethics / consent /
supplementary / abbreviations. Kept as a **separate predicate from `_is_garbled_section`** so the two
reasons for skipping stay distinguishable in the logs, then OR-ed into it so every call site inherits
it. Why it matters: *"A.P. and R.C. conceived the study"* is grammatically a perfect
`(Agent, verb, Thing)` sentence, which is why CRediT blocks produced nearly all the `affiliatedWith`
/ `employs` / `supports` junk. Verified: 9 boilerplate headings skipped, 11 real section headings
kept.

**Measured against the corpus on disk** (what these guards would have removed):

| extractor / model | triples | would be removed | % |
|---|---|---|---|
| `relation/qwen3-8b` | 1,185 | 40 | **3.4%** |
| `relation_scinex/qwen3-8b` | 1,117 | 32 | **2.9%** |
| `relation/claude-opus-5` | 296 | 2 | **0.7%** |

**The same asymmetry as everywhere else**: the guards are worth ~5x more to the weak extractor than
to the strong one. A strong model was already declining these on its own.

⚠ **Every number on disk predates these guards.** They take effect on the next extraction run
(worklist item 3, the full qwen re-extraction). Item (c) — a general post-parse domain/range type
check — remains open; it needs a type inference for arbitrary object strings, which is a bigger job
than the other three and has much less evidence behind it.

⚠ **A tooling note worth remembering:** writing regexes through a bash heredoc silently converted
`\b` into literal backspace characters (0x08) inside the source file. Caught by a behaviour test
(every cross-reference case returned False), not by the parser — the file still imported fine. When
patching regex source, write the patch to a file and run it, or verify with a behaviour test rather
than by reading the diff.


### 12. ⭐⭐ COMPLETE SINGLE-JUDGE COMPARISON + a κ interpretation CORRECTED (2026-08-28)

`openai/gpt-oss-120b` judged the **full 240 qwen sample** too (24 batches, 18.2 min, 0 failures, via
Groq at `--sleep 40`). One judge, both extractors, full samples, identical protocol:

| extractor | ontology | n | weighted strict | lenient |
|---|---|---|---|---|
| **claude-opus-5** | CEO | 100 | **82.8%** | 92.6% |
| qwen3:8b | CEO | 114 | 29.8% | 64.0% |
| qwen3:8b | scinex | 126 | 18.3% | 40.6% |

**No partial-sample caveat on this row set** — it supersedes the gemini rows (n=58/52, quota-capped)
as the version to quote. Both judges independently put Claude at ~3x qwen.

#### ⛔ A κ claim from earlier this session is WITHDRAWN

Earlier I wrote that "judges agree far more about good extraction than about bad extraction",
citing κ 0.342 (on qwen) rising to 0.457 (on Claude). **That compared two different judge PAIRS** —
claude-vs-gemini against gemini-vs-gpt-oss — so the pair changed at the same time as the extractor.
It was a confound, not a finding.

| judge pair | on | raw | κ (3-class) | binary | κ (binary) |
|---|---|---|---|---|---|
| claude vs gemini | qwen's 110 | 59.1% | 0.342 | 83.6% | 0.480 |
| **gemini vs gpt-oss** | **qwen's 110** | **75.5%** | **0.552** | 91.8% | 0.662 |
| **gemini vs gpt-oss** | **claude's 100** | **86.0%** | **0.457** | 91.0% | 0.637 |

Holding the pair constant, the correct reading is:

1. **Raw agreement IS higher on the good extraction** (86.0% vs 75.5%).
2. **But κ is LOWER there** (0.457 vs 0.552) — the **kappa paradox**. Claude's triples are
   overwhelmingly CORRECT (81/100 joint-CORRECT), so the marginals are skewed, chance agreement is
   high, and κ is penalised even as raw agreement rises. **κ is not comparable across samples with
   different class balance.** Always report it with the raw agreement and the confusion matrix.
3. **Claude was the outlier labeller, not Gemini.** It gave qwen's triples 24.5% precision where
   gemini gave 13.6% and gpt-oss 14.5%; the two non-Claude judges agree with each other (0.552) far
   more than either agrees with Claude (0.342). **Prefer the gemini/gpt-oss figures**, and treat the
   n=240 Claude-labelled gold set as the lenient end of the range.

Files: `gold/gptoss_verdicts_qwen.csv`, `gold/report_qwen_gptoss.json`,
`gold/agreement_qwen_twojudges.json`.


### 13. ⭐⭐⭐ THE CAPABILITY CURVE — four extractors, one judge (2026-08-28)

The professor supplied an **OpenRouter** key ($3 credit, paid tier). OpenRouter proxies ~419
models behind one OpenAI-compatible endpoint, so `run_api_extract.py --provider openai-compat
--base-url https://openrouter.ai/api/v1` drives it with **no new code**. Two full corpus runs cost
**$0.34 of $3** — the whole budget is ~10-30 corpus runs.

Judged by `openai/gpt-oss-120b` (OpenAI family) — chosen because it is the **only** judge
family-independent of all four extractors, and gemma disqualifies gemini (both Google).
Via OpenRouter it needs no `--sleep`: 100 triples judged in ~6 min, ~$0.001.

| extractor | family | scale | triples | /para | n | **strict** | lenient | junk share |
|---|---|---|---|---|---|---|---|---|
| gemma-3-12b | Google | 12B dense | 1,789 | 3.47 | 100 | **29.2%** | 52.1% | 20.1% |
| qwen3:8b | Alibaba | 8B dense | 1,185 | 2.26 | 114 | **29.8%** | 64.0% | 12.7% |
| qwen3-235b | Alibaba | 235B MoE | 393 | 0.76 | 100 | **48.8%** | 60.9% | 6.4% |
| claude-opus-5 | Anthropic | frontier | 296 | 0.56 | 100 | **82.8%** | 92.6% | 0.7% |

**1. Precision and volume are inversely ordered, without exception.** Rank by strict precision and
you have also ranked, in reverse, triples per paragraph (3.47 → 0.56) and junk-predicate share
(20.1% → 0.7%).

**2. ✅ THE CONFOUND IS CLOSED.** qwen3:8b vs qwen3-235b holds vendor, lineage and tokenizer fixed
and varies only scale (~29x total params; 8B dense vs 235B MoE / 22B active): strict **29.8% →
48.8%**, volume **2.26 → 0.76**, junk **12.7% → 6.4%**. Until now the headline compared an 8B
Alibaba model against a frontier Anthropic one, so strength and vendor moved together. They no
longer do. **This was listed as the one open methodological gap in §26.9b — it is now closed.**

**3. ⛔ BUT PARAMETER COUNT DOES NOT TRANSFER ACROSS FAMILIES.** gemma-3-12b is 50% larger than
qwen3:8b and is **not better** (29.2% vs 29.8% strict), while emitting the most triples of any model
tested and carrying the largest junk share. **So the axis is capability, not parameters** — and
gemma is the negative control that licenses saying so. Do not write "bigger model → better
extraction"; write "more capable extractor → fewer, better-typed triples".

⚠ **Asymmetry to disclose:** `qwen3-235b` and `gemma-3-12b` were prompted *after* the
boilerplate-section guard (§26.11) landed, so they saw **515** paragraphs where the older runs saw
**525**. Verified by diffing the rendered prompts: the 10 missing ones are exactly the
Author-contributions / Declarations / Funding / Data-availability sections. This is also the guard's
**live validation on real data** — it removed precisely what it was designed to remove. It mildly
*helps* the two newer models, so state it.

**Two bugs found and fixed doing this:**
* **Key resolution was order-based, not endpoint-based.** With several provider keys in `.env`,
  `_openai()` walked a fixed preference list and sent the **Groq key to OpenRouter** →
  `401 Missing Authentication header`. Now `_HOST_KEYS` maps a base-URL fragment to its env var.
* **`claude_corpus_report.py --out` defaults to `output/claude_relation_corpus.json`** regardless of
  `--model-slug`, so running it for a new model **clobbers Claude's corpus JSON**. Always pass
  `--out`. Per-model files now exist for claude-opus-5, qwen3-235b and gemma3-12b.

**Also worth remembering:** `gpt-oss-120b` spends completion tokens on internal reasoning *before*
content. At `max_tokens=64` a test returned `content: None` — a too-small budget yields **empty
verdicts that look like a model finding nothing**. Judge at `--max-tokens 3072`.

Files: `gold/gptoss_verdicts_qwen3-235b.csv`, `gold/gptoss_verdicts_gemma3-12b.csv`,
`gold/report_qwen3-235b_gptoss.json`, `gold/report_gemma3-12b_gptoss.json`,
`output/{qwen3-235b,gemma3-12b}_relation_corpus.json`.


### 14. ⭐⭐ PIPELINE ENGINEERING MEASURED ON ONE MODEL — +8.3 pts strict (2026-09-02)

`qwen3-8b-v2` re-runs the identical local Ollama build over the identical paragraphs. **Only the
code changed**: the three post-parse guards (§26.11) plus the two bug fixes (§26.1, §25.5).

| | pre-fix | +guards & fixes | change |
|---|---|---|---|
| weighted strict | 29.8% | **38.1%** | **+8.3 pts (+28% rel.)** |
| weighted lenient | 64.0% | 66.1% | +2.1 pts |
| triples | 1,185 | **1,357** | **+14.5%** |
| junk-predicate share | 12.7% | 11.5% | −1.2 pts |

**Precision and volume rose together — not a precision/recall trade.** Per-predicate movement matches
each fix one for one: `evaluates` 13 → **48** (+269%, the numeric-object bug), `comparesAgainst` +53%,
`comprises` +50%, `achieves` +25%; against `affiliatedWith` 29 → **13** (−55%) and `employs` −38%
(boilerplate-section guard), `evaluatedOn` −27% (cross-reference guard). Nothing else moved materially.

**The paper now has two independent levers, and they are additive:** model choice 29.8% → 82.8%,
pipeline engineering 29.8% → 38.1% on a fixed model.

### ⛔ THE 7-HOUR MISTAKE — a third instance of one failure class

The first `qwen3-8b-v2` attempt ran 428 min and produced 495 triples. It nearly went into the report
as "the guards removed 58% of triples". It was wrong, and the tell was that the guards had rejected
**zero**.

**93 of 515 replies (18%) came back completely empty** — median length 0 chars, every one inside an
unclosed `<think>`. Ollama returns reasoning in a **separate `thinking` field**; `run_api_extract`'s
adapter read only `content`. qwen3 spent the whole 3,072-token budget reasoning and returned nothing.
`kg_extraction/ollama_backend.py:91` had always set `"think": False`; the API adapter never did — so
the run was a **different experimental condition**, not a corrected version.

Fixed: `_ollama()` sends `"think": False`, and `_ollama_text()` falls back to the `thinking` field if
a build ignores the flag. Re-run: **0 empty replies, 6.0% non-JSON (vs 5.2% originally), 163 min
instead of 428** — most of the first run was generating reasoning that was then discarded.

> **⚠ GENERALISE THIS — it has now bitten three times.** A reasoning model separates reasoning from
> content, and **a token budget consumed by reasoning yields empty output that is indistinguishable
> from "the model found nothing"**. Seen with `gpt-oss-120b` at `max_tokens=64` (content `None`) and
> with `qwen3:8b` at 3,072 (93 empty replies). **Whenever a model "finds nothing", check for an empty
> content field with a populated reasoning field BEFORE believing the result.**

The invalid log is kept as `extract_qwen8b_v2_INVALID_thinking_on.log`.


### 16. ⭐⭐⭐ ONTOLOGY x CAPABILITY — THE RANKING REVERSES (2026-09-03)

Research question 2 finally has a real answer, and it is not the one on record.

**The earlier conclusion was "CEO vs scinex is a tie"** (§25.1: corpus-weighted 34.7% vs 39.6%,
difference −4.9 pts, 95% CI [−22.5, +13.9]). **Every measurement behind it was taken on qwen3:8b** —
the weakest extractor in the study. Running both ontologies at two capability levels shows the tie
was an artefact of the measurement model.

All four cells: same paragraphs, same guards, same code version, same judge (`openai/gpt-oss-120b`),
**n=100 each**.

| | CEO | scinex | scinex − CEO |
|---|---|---|---|
| **qwen3:8b (8B)** | 38.1% | **12.1%** | **−26.0 pts** → CEO wins |
| **qwen3-235b (235B MoE)** | 48.8% | **54.7%** | **+5.9 pts** → **scinex wins** |

**Interaction: +31.9 points. The ranking inverts, it does not merely narrow.**

Volume moves the same way, so this is not precision bought with recall:

| | CEO | scinex | ratio |
|---|---|---|---|
| qwen3:8b | 1,357 | 1,280 | **0.94x** |
| qwen3-235b | 393 | 582 | **1.48x** |

At 235B scinex wins on **both** axes — higher precision *and* ~48% more triples. The 0.94x ratio at
8B reproduces exactly across the guards change (pre-guards 1,117/1,185 = 0.94), so the volume effect
is a property of the model×ontology pairing, not of the code version.

**Mechanism:** scinex is the richer schema (27 relations vs 22, tighter domain/range). Extra
structure only helps a model that can satisfy it — for a weak extractor it is more surface area to
get wrong (scinex is qwen3:8b's *worst* configuration of anything tested); for a capable one the
tighter typing becomes an asset.

**This also explains §25.1's instability.** CEO beat scinex in gold round 1 and scinex beat CEO in a
disjoint round 2. Both were measured on qwen3:8b, where the schema is not the binding constraint —
so the comparison was noise-dominated and flipped between draws.

⚠ **What to claim.** The **sign** is robust: it holds pre-guards (−11.5 pts at 8B) and post-guards
(−26.0). The **magnitude at 8B moves with the sample**, so report the direction and the interaction,
not the 8B gap to a decimal. Sanity-checked rather than assumed: the 8B scinex verdict mix is
15 CORRECT / 28 PARTIAL / 57 INCORRECT, and an independent judge on a different sample gave a
comparable 11.5%.

**Cost: $0.16** (the 235B scinex run via OpenRouter). The 8B half ran locally, free.

### 16b. ⛔ Guard gap found by reading the errors — cross-reference SUBJECTS

An INCORRECT example in the 8B scinex sample was `(Section 3, reports, ...)`. The §26.11
cross-reference guard only checked **objects**. Measured across the whole corpus:

| model | cross-ref subjects |
|---|---|
| gemma-3-12b | 15 / 1,789 (0.84%) |
| qwen3:8b | 11 / 1,357 (0.81%) |
| qwen3-235b | 1 / 393 (0.25%) |
| claude-opus-5 | 0 |

Small, but concentrated in exactly the weaker extractors — the same pattern as every other defect.
`_looks_like_cross_reference()` is now applied to `canonical_subj` as well as `obj`. Verified:
`Section 5` rejected; `Table-driven parser` and `sectional analysis` still kept.

### 17. Not done / next

1. **Finish the 130 remaining qwen judge labels** — recipe in §26.7. Then Table 8's qwen row is final.
2. **Judge the 2 new papers** once qwen and Claude have both extracted them.
3. **Full qwen re-extraction** of all 20 papers with the §26.1 + §25.5 fixes, so every number is on
   one code version (and it would let `evaluates` be re-judged fairly).
4. Guard worklist (a)-(d) from §25.7 — still open, and Table 8 says they matter much less for a
   strong model than for a weak one.
5. `llama3.1:8b` was pulled and its corpus run **killed at the user's request** after 1 paper.
   The partial output directory was **deleted** and the log renamed
   `llama31_extraction_ABANDONED.log`, so nothing on disk can be mistaken for a real run. The model
   is still in Ollama if the size-matched control (an 8B from a different family, to separate "model
   strength" from "model identity" in the Table 8 gap) is ever wanted — ~2.7 h, no key needed.


---

## Session 25 — Gold set 80 → 240; the CEO-beats-scinex headline does NOT replicate; Claude run as the extractor over the FULL corpus; replication tooling for any model; three real bugs (2026-08-27 → 2026-08-28)

Two asks from the user: (1) keep judging — extend the gold labels, (2) find out whether Claude can
*produce* triples the same way the local model does, using the same prompt and constraints. Both done.

### 1. ⭐⭐ HEADLINE CORRECTION — Session 24's "CEO beats scinex" is WITHDRAWN

A second, **disjoint** gold sample of **160** triples was drawn (`gold_eval.py export --n 300 --seed 7`,
then the 37 ids already labelled in round 1 removed, then round-robin over extractor × predicate down
to 160) and labelled by Claude under the same rubric → `gold/sample_round2_claude.csv`.
Combined gold set is now **240 labels** (54 CORRECT / 75 PARTIAL / 111 INCORRECT).

| gold set | CEO weighted strict | scinex weighted strict |
|---|---|---|
| round 1 (n=80) | **46.2%** | 34.4% |
| round 2 (n=160, disjoint) | 28.9% | **41.7%** |
| **combined (n=240)** | **34.7%** | **39.6%** |

Bootstrap (2,000 resamples within predicate buckets, corpus weights fixed):
CEO **34.8% [21.7, 48.8]**, scinex **39.7% [27.4, 51.9]**, difference (CEO − scinex)
**−4.9 pts, 95% CI [−22.5, +13.9]**, `P(CEO > scinex) = 0.30`.

**The advantage reversed sign on a disjoint draw and the combined interval straddles zero.** At this
sample size the two ontologies are indistinguishable on per-triple precision. Do not put "CEO beats
scinex" in the paper.

**Why round 1 looked decisive, mechanically:** CEO's weighted number is dominated by `addresses`
(26.1% of the corpus) and `uses` (24.9%) — half the corpus mass estimated from ~8 labels in round 1.
Both cells fell in round 2 (`addresses` 2/6, `uses` 2/6 combined) and dragged the weighted figure
down with them. The Session-24 note warned that per-predicate cells rested on ~4 samples; this is
that warning coming true, which is worth stating plainly in the write-up rather than burying.

**What DOES survive:** CEO keeps a consistent **lenient** edge (66.8% vs 59.9% weighted). CEO's
errors are more often "on topic, loosely typed"; scinex's are more often flatly wrong. Also unchanged:
the KGE/citation-prediction ontology tie — different axis, untouched.

### 2. Per-predicate findings that are stable across both rounds

Only these are worth acting on; single-round cells are noise.

- **Genuinely good:** `achieves` (CEO 6/6, scinex 4/6) — the metric/outcome backbone of the graph.
  `comprises` under scinex 5/6, `comparesAgainst` CEO 4/6.
- **Zero in both rounds AND both ontologies:** `employs`, `evaluates`, `configures`, `evaluatedOn`,
  `trainedOn`, `cites`, `affiliatedWith`, `supports`. These are systematic, not sampling.
- `trainedOn` is the benign failure — nearly all PARTIAL, object is a generic phrase
  ("dataset of car images") instead of a named corpus.
- **The schema placeholder leak RECURRED**: `(Merck, affiliatedWith, Organisation)` and
  `(FMC Corporation, affiliatedWith, Organisation)` — the literal ontology class name as an object
  value, now confirmed on a second predicate. The fix must reject **any** object matching an ontology
  class name; patching `mentions` alone would have missed these.
- **New error class — cross-reference objects** (6/240): `extractedFrom → "Section II DATA
  PREPARATION"`, `evaluatedOn → "Table III"`, `evaluatedOn → "ASSE 2025"` (a running header).
  Prompt rule 5 already forbids section titles as objects; nothing enforces it post-parse.

### 3. New tooling: `gold_report.py`

Session 24's weighting was ad hoc in-session. It is now a script, and it **reproduces the round-1
numbers exactly** (46.2 / 34.4), so the method is confirmed rather than re-derived:

```bash
python3 gold_report.py --labels gold/sample_claude.csv gold/sample_round2_claude.csv \
        --bootstrap 2000 --out gold/report_combined.json
```

Prints sampled + corpus-weighted precision per extractor, per-predicate and per-paper tables, and a
bootstrap CI on both the weighted figures and their difference. Accepts any filled label CSV, so a
second labeller (or a local judge) drops straight in.

Per-paper spread (pooled, n=240) is wide: osmotic2026 / ugmo2024 50% strict down to
trafficspeed2021 / textaug2023 / microwave2018 / reststop2018 at 0%. **Do not read anything into
this.** Checked against parse quality and it does NOT track it — the 0% papers have ordinary body
lengths (2,472 / 2,476 / 3,114 / 3,868 words) and ordinary yields (45–80 triples). What they have in
common is 4–9 gold labels each, so 0% is what a handful of unlucky draws looks like. Per-paper
precision needs its own stratification before it means anything.

### 4. ⭐ Claude run as the extraction backend — `claude_extract.py`

The question was whether Claude can produce triples "like this model using the same things". It can,
and the way it was wired matters: **`claude_extract.py` splits the extractor at the model boundary**
instead of reimplementing it.

- `prompts` — walks the paper exactly as `kg_main` does (parse_html → split_into_sentences →
  `_sentences_to_paragraphs` → `_is_garbled_section` skip) and writes the **real** system prompt
  (`_make_fixed_system_prompt`, 11,908 chars) plus one user prompt per paragraph to `prompts.jsonl`.
- `ingest` — takes the replies back as `{"para_id", "raw"}` JSONL and runs them through the **same**
  `_parse_fixed_output()` guards and the **same** `KnowledgeGraphBuilder.save()`.

So prompt, ontology, guards, dedup and output format are shared code, not copies — the only variable
is the model. Works for `--extractor relation|fixed` × `--ontology ceo|scinex`.

**FULL CORPUS RUN COMPLETE — all 18 papers, 486 paragraphs, CEO ontology, relation mode.**
284 triples emitted → **268 stored** (16 lost to guards + the graph-builder entity filter).
qwen3:8b produced **1,101** on the identical input.

| | qwen3:8b | claude-opus-5 |
|---|---|---|
| triples stored | **1,101** | **268** (24%) |
| distinct predicates | 21 | 14 |
| triples per paragraph | 2.27 | 0.55 |

**The missing volume is concentrated in the predicates that are always wrong.** Share of each
model's corpus held by the 11 predicates scoring **0% strict** in the n=240 gold set:
**qwen3:8b 22.7% vs claude-opus-5 6.0%** (16 of 268). Per predicate, qwen → Claude counts:
`affiliatedWith` 29→**0**, `employs` 26→**0**, `publishedIn` 8→**0**, `configures` 8→**0**,
`cites` 4→**0**, `reports` 2→**0**, `motivates` 1→**0**, `trainedOn` 54→**1**, `evaluates` 12→**1**.
The two predicates scoring ≥50% strict (`achieves`, `comparesAgainst`) go the other way:
**16.9% of qwen's corpus → 29.5% of Claude's.**

Concretely: the CRediT author-contribution blocks, the Funding paragraph and the running-header
lines that generated `affiliatedWith`/`supports`/`locatedIn` junk all return `{"triples": []}` —
the §24.3 "skip these sections" fix turns out to be something a stronger model does on its own.

Rebuild the table any time with `python3 claude_corpus_report.py`; corpus JSON at
`output/claude_relation_corpus.json`.

### 4b. ⛔ A SECOND pipeline bug, found by running the full corpus: `evaluates` cannot ever be correct

`kg_builder._is_valid_entity()` requires every node to match `[a-zA-Z]{2,}`, so a **bare numeric
object is deleted after the guards pass**. The prompt defines `evaluates` as
`EvaluationMetric → ExperimentalResult` and its own GOOD example is `(<Metric>, evaluates, 88.5)`.
The pipeline therefore **demands a value it then throws away**. Measured on this run:
`(precision, evaluates, 0.923)`, `(accuracy, evaluates, 0.974)`, `(IoU, evaluates, 0.612)`,
`(RMSE, evaluates, 14.76)`, `(root-mean-square error, evaluates, 9.23)` were all produced and all
silently dropped — only the malformed wordy ones survive to be judged, which is exactly why
`evaluates` sits at **0% strict** in §25.2. The same filter's `len < 3` rule deleted
`(PNext, comprises, SF)` / `SD` / `SP` in reststop2018.
**Fix: permit numeric objects for `evaluates`/`achieves`, and exempt all-caps acronyms from the
length floor.** NOT applied — it changes graph content and every number on disk predates it.

> **⚠ These triples are NOT judged, and Claude must not judge them.** Extractor and judge have to be
> different models or the evaluation is self-scoring. Use `run_local_judge.py` (qwen2.5) or a human
> pass on `output/videoseg2025/kg/relation/claude-opus-5/triples.json`. The existing 240 gold labels
> are unaffected — they cover qwen3:8b triples only.

### 4c. Replicating the run with ANY model — `run_api_extract.py` (2026-08-28)

The prompt/response protocol was already model-agnostic; what was missing was the driver.
`run_api_extract.py` fills it, so the same experiment can be repeated across models and the results
land side by side under different `--model-slug` directories.

```
claude_extract.py prompts  →  run_api_extract.py  →  claude_extract.py ingest
   (renders the real prompt)   (calls any model)      (same guards + KG builder)
```

Providers: `anthropic`, `openai`, `openai-compat` (OpenRouter / Together / DeepSeek / vLLM /
LM Studio via `--base-url`), `gemini`, `ollama`. **Standard library only** — nothing to install.
Keys come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`; Ollama needs none.
Flags that matter: `--resume` (default), `--limit N` (smoke test), `--sleep` (rate limits),
`--ingest` (chain the ingest), `--all`, `--retries`. Temperature is pinned to 0; a 400 that
complains about `temperature`/`max_tokens` is retried with the offending field removed, so
reasoning-tier models work without special-casing.

One full corpus run = **486 calls**, ~2,977-token system prompt each, ~159k tokens of paragraphs,
~24k tokens out. That is **~1.6M input tokens uncached**, or **~162k + 486 cache reads** if the
provider supports prompt caching — the system prompt is byte-identical on every call, so caching
it is the single biggest lever.

**Verified end to end** on 2026-08-28 by driving qwen3:8b through the HTTP path
(`--provider ollama --model qwen3:8b --model-slug qwen3-8b-api --limit 2`): prompts rendered,
2 calls answered in 2.8 min, ingest produced 1 triple. The smoke-test directory was **deleted
afterwards** — a partial extraction left on disk is exactly the kind of thing that gets mistaken
for a real run later.

**⛔ Bug this exposed and fixed:** `cmd_ingest` computed its "rejected by the guards" tally with a
bare `json.loads(raw)`, which crashed the whole ingest on any reply that is not clean JSON —
markdown fences, a `<think>` block, a truncated string. Claude's replies were always clean so it
never fired; qwen3:8b's first reply killed it immediately. Now wrapped, with a
`N replies were not plain JSON … / N yielded nothing` line so a badly-behaved model is visible
rather than silent. `_parse_fixed_output` already handled all three cases.

Also fixed: `--model-slug` now flows into the extractor object, so the log says the model actually
being run instead of always `claude-opus-5`.

### 4d. Free chat UIs (ChatGPT / Gemini / Claude.ai) — `chat_paste_extract.py` (2026-08-28)

No API key needed. **Batching is what makes it viable:** in a chat UI the ~3k-token system prompt is
pasted ONCE per conversation and applies to every later message, so the marginal cost per paragraph
is the paragraph itself (~330 tokens). The binding constraint on a free tier is the number of
MESSAGES, not tokens.

- `batches` writes `output/<paper>/kg/<extractor>/<slug>/chat_batches/batch_NN.txt`
  (batch_01 carries the SYSTEM RULES block; `--repeat-system` puts them in every file).
- `collect` reads `reply_NN.txt` back and writes `responses.jsonl` → normal `claude_extract.py ingest`.

**Whole corpus at `--size 12`: 49 messages, ~196k tokens of pasting, ~4k tokens per message.**
Per paper: 1 (ugmo2024) to 5 (csysguard2024, strabismus2026). Suggested cadence: a fresh
conversation every ~4 batches (~23k tokens of accumulated context) so quality does not drift.
Ready-made batches for all 18 papers are on disk under the `gpt5-chat` slug; regenerate for any
other model with one command.

`collect` is written for what chat UIs actually return: markdown fences, a chatty preamble and
sign-off, and truncation are all tolerated (a regex salvages whole `{"para_id": …}` objects from a
broken tail). It reports exactly which para_ids are missing and merges re-pasted batches, so an
interrupted session is resumable. **Verified end to end** with a simulated fenced + chatty reply:
3/11 paragraphs collected, missing ids listed, ingest produced 2 triples through the normal guards.

**⚠ Two caveats — state both if any chat-UI number is reported.**
1. **Batching is a different experimental condition.** The API/local runs send one paragraph per
   call; here the model sees ~12 at once and can carry context between them. Not a reproduction of
   the qwen3:8b / claude-opus-5 runs. `--size 1` restores strict comparability but costs 486
   messages, which no free tier will allow.
2. **You do not control the model version in a free UI.** It can silently fall back to a smaller
   model when limits are hit, mid-session. Fine for a cheap sanity check; **not** a defensible
   number for the paper. For that, use `run_api_extract.py` with a pinned model id.

### 4e. Getting Claude's triples judged by another model — `judge_paste.py` (2026-08-28)

Claude extracted them, so Claude cannot judge them. `judge_paste.py` packages a gold sample for a
chat-UI judge and reads the verdicts back into the CSV format `gold_report.py` and
`gold_eval.py agree` already consume.

```
gold_eval.py export → judge_paste.py batches → (paste) → judge_paste.py collect → gold_report.py
```

Sample: `gold_eval.py export --extractor relation --model claude-opus-5 --n 100 --seed 11`
→ `gold/claude_sample_for_judge.csv`, stratified over all 14 predicates Claude used.
Bundles built for two judges: `judge_upload/gpt-judge/` and `judge_upload/gemini-judge/` —
**10 files, 100 triples, ~13.7k tokens of pasting each.** Each batch shows the triple, the
predicate's ontology definition and the source sentence; batch 01 carries the JUDGING RULES
(the same CORRECT/PARTIAL/INCORRECT rubric used for the n=240 gold set, plus the tie-breakers
that decide most hard cases).

`collect` tolerates fences/preamble/truncation, reports missing triple_ids, and merges re-pastes.
**Verified end to end** with a simulated chatty reply: 2/100 collected, 98 listed as missing,
`gold_report.py --labels … --model claude-opus-5` read the result. Test artifacts deleted.

Once both judges return, `gold_eval.py agree --gold gold/gpt_verdicts.csv --b gold/gemini_verdicts.csv`
gives the **first κ this project has ever had** — two independent labellers on the same ids.

### Rules for a replication run

1. `--model-slug` must not contain `:` or `/` (Windows path; `qwen3:8b` silently discarded every
   triple at write time until §23 caught it). The script refuses them.
2. `--max-tokens` ≥ 3072. At 1024, 34% of local calls truncated, and a truncated JSON yields ZERO
   triples — which looks identical to a model that found nothing.
3. Temperature 0, and re-render prompts per slug (deterministic — same 486 paragraphs every time).
4. **A model must never judge its own extraction.** Extract with one family, judge with another.

### 5. ⛔ REAL BUG FOUND AND FIXED — `_subject_in_sentence()` dropped every parenthetical subject

Found because three of Claude's triples with the subject `Segment Anything Model (SAM2)` were rejected
even though the string is verbatim in the source sentence.

```python
# before
_re.search(r'\b' + _re.escape(canonical.lower()) + r'\b', src_lower)
```

`\b` asserts a word/non-word transition. A subject ending in `)` is followed by a space in the
sentence — both non-word — so the trailing `\b` can never match. **Any subject ending in a
non-word character was silently discarded**, and "Full Name (ABBR)" is the standard way a paper
introduces a model. Evidence: **2 of 2,132** triples in the whole qwen3-8b corpus have a subject
ending in `)`.

Fixed with a new `_contains_term()` that applies `(?<!\w)` / `(?!\w)` only on the alphanumeric side.
Verified: `Segment Anything Model (SAM2)` now matches, `SAM` still does **not** match inside `SAM2`,
`yolov` still does not match inside `YOLOv8`. This is a **recall** fix — it can only add triples.
All numbers in `results.md` were produced with the buggy guard; the gain is unmeasured until
re-extraction.

### 6. Files

- `gold/sample_round2.csv` — the 160-row disjoint sample, blank verdicts (for a second labeller).
- `gold/sample_round2_claude.csv` — the same 160 rows labelled, with a reason on every non-CORRECT row.
- `gold/report_combined.json` — the n=240 report (per-extractor, per-predicate, per-paper, bootstrap).
- `gold_report.py` — new, reproduces every table above.
- `claude_extract.py` — new, Claude (or any API model) as the extraction backend.
- `kg_extraction/fixed_extractor.py` — `_contains_term()` added, `_subject_in_sentence()` fixed.
- `output/videoseg2025/kg/relation/claude-opus-5/` — prompts, responses, triples, graphml.

### 7. NOT done / next session — ⭐ START HERE

**Waiting on Sam (nothing blocks these on my side):**

1. **Two PDFs are still missing** — the corpus is 18/20. Drop them in as
   `papers/tripplanner2020.pdf` ("A Recommender System for Trip Planners") and
   `papers/linkpred2015.pdf` ("Link prediction in linked data of interspecies interactions").
   The filename IS the paper id; a wrong name silently forks the output tree. Then:
   `main.py <pdf> --no-llm` → check the parsed Title against `papers/manifest.csv` →
   `claude_extract.py prompts` → extract → `ingest` → `claude_corpus_report.py`.
   Note when reporting: that makes 20 papers for claude-opus-5 but qwen3:8b stays at 18 (needs GPU).
2. **GPT / Gemini extraction run** — bundles are built and ready to paste:
   `chat_upload/gpt/` and `chat_upload/gemini/` (49 files each, files 01–21 = the six priority
   papers). When the `replies/` folders come back:
   `chat_paste_extract.py collect --all --model-slug gpt --flat-dir chat_upload/gpt/replies`
   → `claude_extract.py ingest` per paper → `claude_corpus_report.py --model-slug gpt`.
   Use the **reasoning/thinking flagship** on both sides, web search OFF, model name recorded in
   `MODEL_USED.txt`.
3. **Judging Claude's triples** — bundles ready: `judge_upload/gpt-judge/` and
   `judge_upload/gemini-judge/` (10 files, 100 triples, sample in
   `gold/claude_sample_for_judge.csv`). When replies come back:
   `judge_paste.py collect --csv gold/claude_sample_for_judge.csv --slug gpt-judge
   --out gold/gpt_verdicts.csv` → `gold_report.py --labels gold/gpt_verdicts.csv
   --model claude-opus-5`. **Claude must never judge Claude's own extraction.**
   With BOTH judges back: `gold_eval.py agree --gold gold/gpt_verdicts.csv
   --b gold/gemini_verdicts.csv` = **the first κ this project has ever had.**

**Code work, unblocked:**

4. **Apply the guard worklist** (four items, one confirmed twice): (a) reject any object equal to an
   ontology class name; (b) reject section/table/figure/running-header cross-references as objects;
   (c) post-parse domain/range type check per predicate; (d) skip Author-contributions /
   Declarations / Funding sections at extraction. §24.3 + §25.2.
5. **Fix `kg_builder._is_valid_entity()`** — bare numeric objects are deleted after the guards pass,
   so a correct `(<Metric>, evaluates, 88.5)` can never reach triples.json. That is *the* reason
   `evaluates` scores 0%. Also exempt all-caps acronyms from the `len < 3` floor. §25.4b.
6. **Re-extract at least one paper** to size what the `_contains_term()` fix (§25.5) recovers —
   currently unmeasured.

**Needs the VM (blocked on Sam's laptop):**

7. Run `fixed` extraction for the fixed-vs-relation comparison across the CS/non-CS split.
8. **Re-run everything with Qwen3-14B.** Every number in §23–§25 is `qwen3:8b` on a laptop and is
   NOT the paper's figure.

**Do NOT re-derive these (they cost a session each to find):**
`--model-slug` must never contain `:` or `/` · `--max-tokens` ≥ 3072 (1024 truncated 34% of calls,
and a truncated JSON yields ZERO triples) · Ollama `num_ctx=8192` · no torch on the laptop
(Smart App Control) · scipy pinned at 1.17.1 · `nvidia-smi` util% is broken on the vGPU.

---

## Session 24 — Claude judged 80 triples; CEO BEATS scinex once corpus-weighted; guard worklist identified (2026-08-27)

**The user asked Claude (this assistant) to act as the judge** instead of a local model. That is a
legitimate independent judge — a different model family from the extractor's `qwen3:8b`, which is the
independence requirement §23 set out. Verdicts are saved as human-equivalent **gold labels** in
`gold/sample_claude.csv` (80 rows, `verdict` + `note` filled), so any future judge can be scored
against them with `gold_eval.py agree --gold gold/sample_claude.csv --b <other>.csv`.

Sample: `gold_eval.py export --extractor relation relation_scinex --model qwen3-8b --n 80`
(seed 42, stratified round-robin over extractor × predicate). Rubric as printed by `export`.

### 1. ⭐⭐ HEADLINE: the ontology comparison INVERTS depending on how you weight it

| ontology | sampled strict | **corpus-weighted strict** | **corpus-weighted lenient** |
|---|---|---|---|
| **CEO** (`relation`) | 28.9% | **46.2%** | **86.3%** |
| **scinex** (`relation_scinex`) | 28.6% | **34.4%** | **65.6%** |

- **On the stratified sample the two are TIED** (28.9% vs 28.6%, n=38/42).
- **Re-weighted to the actual corpus predicate distribution, CEO wins clearly** — +11.8 points
  strict, +20.7 lenient. Weighting covers 100% of triples in both ontologies.

**Why, and why this matters.** §23.5 found scinex spreads predicate mass more evenly (top-2 = 38% vs
CEO's 51%) and I flagged it as *possibly* the first evidence separating the two ontologies — with the
explicit caveat that **finer-grained is only better if it is correct**. It is not. The relations
scinex redistributes mass *into* are exactly the ones that score **0% strict**: `configures` (0/4),
`evaluates` (0/4), `employs` (0/4), `supports` (0/4). CEO concentrates in `addresses`/`uses`/
`achieves`, which score 50–100%. So scinex's "finer granularity" is largely the model **reaching for
a specific predicate and getting it wrong**.

> **⚠ Report the weighted numbers, and report the weighting.** A stratified sample deliberately
> over-represents rare predicates; quoting its raw precision as corpus precision understates both
> ontologies (28.7% vs a true ~46%/~34%). Conversely, quoting weighted precision without saying it is
> reweighted hides that per-predicate estimates rest on ~4 samples each. **Both numbers, always, with
> the method stated.** The wide per-predicate error bars are the main reason a bigger gold set is the
> top priority.

### 2. Precision by predicate (the guard worklist)

**Perfect / good:** `achieves` 4/4 (100%), `designedFor` 3/4 (75%).
**Middling (50%):** `addresses`, `comparesAgainst`, `comprises`, `produces`, `splitFrom`, `uses`.
**ZERO strict-correct** — every one of these needs a guard:

| predicate | n | C | P | I | the failure |
|---|---|---|---|---|---|
| `employs` | 4 | 0 | 0 | 4 | CEO/scinex define it `Organisation → Person`; the model uses it for *method-uses-method* every single time |
| `affiliatedWith` | 3 | 0 | 0 | 3 | mined from **CRediT author-contribution statements** — `(Rathachai Chawuthai, affiliatedWith, Formal analysis)` |
| `configures` | 4 | 0 | 1 | 3 | object is a metric or a value, not an `Experiment`; often direction-inverted |
| `evaluates` | 4 | 0 | 1 | 3 | self-referential (`root mean square error → RMSE`) or swapped (metric ↔ result) |
| `supports` | 4 | 0 | 1 | 3 | subject is a tool/process, not an `ExperimentalMeasurement`; frequently inverted |
| `evaluatedOn` | 4 | 0 | 2 | 2 | object is a metric (`RMSE`) or a figure (`Table II`), not a Dataset |
| `trainedOn` | 4 | 0 | 4 | 0 | **all PARTIAL** — object is generic ("dataset of car images"), never a named corpus |
| `publishedIn` | 2 | 0 | 0 | 2 | used for arbitrary noun phrases; never a venue |
| `cites` | 2 | 0 | 0 | 2 | subject is a method or the cited authors, not the citing paper |
| `mentions` | 2 | 0 | 0 | 2 | see the placeholder leak below |
| `encompasses` | 2 | 0 | 0 | 2 | chemistry taxonomy (`Hydrocarbons → alkynes`) mapped onto `ResearchDomain → ResearchTask` |

### 3. Four concrete, fixable failure modes

1. **Domain/range violations dominate.** The single largest error class. `employs`, `affiliatedWith`,
   `locatedIn`, `publishedIn` are being used as generic verbs whenever their English name sounds
   right, ignoring the declared types. **Fix: a post-parse type check per predicate**, in the same
   place as the existing IS-A and direction guards.
2. **Direction inversions.** `(data, splitFrom, test set)` — backwards. `(road safety, supports,
   the overall results)` — backwards. `(Errors, evaluates, RMSE)` — backwards. `(ANN, configures,
   tolerance…)` — backwards. **Fix: extend the existing direction guards to `splitFrom`, `supports`,
   `evaluates`, `configures`.**
3. **⛔ SCHEMA PLACEHOLDER LEAK — a real bug.** Two triples have the literal string `AcademicPaper`
   as their object: `(Kalman filter, mentions, AcademicPaper)`, `(Butterworth low-pass filter,
   mentions, AcademicPaper)`. The scinex definitions for `mentions`/`extractedFrom` contain a `?`
   placeholder (`AcademicPaper → ?`), and the model is copying the type name out of the prompt as if
   it were a value. **Fix: reject any object that exactly matches an ontology class name.** Cheap,
   and it is unambiguously wrong output.
4. **Author-contribution sections are being mined.** oxidecrack2025's CRediT statement produced
   `affiliatedWith` triples. **Fix: skip `Author contributions` / `Declarations` / `Acknowledgements`
   sections at extraction time** — they contain no research content. This is a `kg_main`/`html_parser`
   change, not a prompt change, and it will also slightly raise yield-per-useful-paragraph.

### 4. What is genuinely good

23/80 fully correct and 25/80 partial — and the good ones are the ones that matter for a KG:

```
(Random Forest model, achieves,        accuracy of 0.95)
(LSTM,                comparesAgainst, ARIMA)
(Genetic Algorithm,   comprises,       Selection)
(RT-DETR,             uses,            IoU-aware query selection)
(test set,            splitFrom,       dataset)
(CNN model,           designedFor,     vehicle makes classification)
```

`achieves` at 100% and `designedFor` at 75% mean the metric-outcome and model-purpose backbone of
the graph is sound. The corpus-weighted **lenient** figure for CEO — **86.3%** — says most triples
are at least defensibly on-topic; the strict figure (46.2%) says roughly half need the ontology
conformance work above.

### 5. Files

- **`gold/sample_claude.csv`** — 80 gold labels with per-row reasons. This is the reference set for
  validating any future judge (`gold_eval.py agree --gold gold/sample_claude.csv --b judge.csv`).
- Analysis was ad-hoc over `gold/sample_claude.csv` + `output/*/kg/relation*/qwen3-8b/triples.json`;
  no new scripts were added.

### 6. NOT done / next session

1. **⚠ n=80 is a pilot, not a measurement.** Per-predicate cells hold ~4 samples. Before any of this
   goes in the paper, extend the gold set to ~150–200 (`gold_eval.py export --n 200 --seed 7` gives a
   disjoint draw) — especially for the zero-scoring predicates.
2. **Apply the four §24.3 fixes**, re-extract, re-judge **on the same triple ids** (they are content
   hashes, so unchanged triples keep their labels and `agree` reports how many dropped out).
3. **Run `fixed`** (~5.5h local) for the fixed-vs-relation comparison on the CS/non-CS split.
4. **Re-run everything on the VM with Qwen3-14B.** All of §23–§24 is `qwen3:8b` on a laptop; a
   stronger extractor may not make these mistakes at all, and the paper's numbers must come from the
   project-standard model.
5. Optional and cheap: an API run (`claude-opus-5`, ~$8 with prompt caching + Batch) as a quality
   ceiling — see §23 discussion. If Claude extracts, Claude must **not** also judge.

---

## Session 23 — Local extraction WORKS via Ollama; three config bugs found and fixed; first triples exist (2026-08-27)

**The laptop can now run the extractors after all** — not through torch (still blocked, see §22.3) but
through a signed **Ollama** binary. First triples for this corpus now exist on disk.

### 1. The Ollama backend

Windows 11 Smart App Control blocks torch's unsigned DLLs, but Ollama ships signed binaries and
brings its own runtime, so it installs and runs normally:
`winget install Ollama.Ollama --source winget` (v0.33.0), then `ollama pull qwen3:8b` (5.2GB).

The extractors reach their model through exactly two attributes —
`self._tokenizer.apply_chat_template(messages, …)` then `self._pipeline(prompt)` — so shimming those
two is enough; **the prompt, ontology, guards, kg_builder and output layout are all untouched.**

- **`kg_extraction/ollama_backend.py`** — `_ChatTemplateShim` (identity: Ollama's `/api/chat` wants
  the message list, and the server applies the model's real template), `_OllamaPipeline` (POSTs to
  `/api/chat`, `temperature=0` for greedy to match the transformers path, `think:false`), plus
  `patch_extractors()` and `patch_judge()`. Patches the **class**, since `kg_main`/`kg_evaluate`
  construct their own instances. Also replaces the `device` property — the real one does
  `import torch` inside `except ImportError`, which no longer catches anything now that torch raises
  `OSError`.
- **`run_local.py`** — drives the ordinary `kg_main` pipeline through that backend.
- **`run_local_judge.py`** — same for `kg_evaluate`'s `LLMJudge`. Judge model defaults to
  **`qwen2.5:7b-instruct`**, deliberately a **different family** from the extractor's `qwen3:8b`: a
  model grading its own output is not an independent check.
- **`count_triples.py`** — counts per paper/extractor + predicate breakdown.

### 2. ⚠ Three config bugs, all found by measurement. Do not re-derive these.

| # | symptom | root cause | fix |
|---|---|---|---|
| 1 | model loaded, GPU 0%, nothing generated, looked like a hang | **Ollama defaults to `num_ctx=4096` regardless of what the model supports.** The relation/fixed system prompt is ~13.6k chars ≈ **3.4k tokens** before the paragraph is added, so the prompt crowded out the answer | set `num_ctx` explicitly |
| 2 | ~110s per call, `ollama ps` showed `20%/80% CPU/GPU` | overcorrecting to `num_ctx=16384` pushed the footprint to **7.8GB**, past the 8GB VRAM, so Ollama spilled to CPU | **`num_ctx=8192` → 6.2GB, 100% GPU** (measured; 6144 → 5.9GB also fine, 16384 → spill) |
| 3 | **34% of calls produced 0 triples** | `--max-new-tokens 1024` truncated the JSON mid-object, and a truncated JSON parses to **zero** triples — the whole call is wasted. Each triple echoes its `source_sentence` verbatim, so a triple off an abstract costs **~500 tokens**, not the ~80 a bare triple would | **`--max-new-tokens 3072`** → cap-hit rate 34% → **6%**, yield 0.86 → **2.77 triples/call** |

**And one that would have silently destroyed the whole run:**

> **⛔ `qwen3:8b` contains a colon, which is ILLEGAL in a Windows path.** `kg_main` builds
> `output/<paper>/kg/<extractor>/<model>/` from the `--model` string, so every triple was extracted
> correctly and then thrown away with `NotADirectoryError` at write time. A probe run on one small
> paper reported `rc=1` after producing 31 perfectly good triples and writing nothing.
> **Fix:** `run_local.py` passes a slug (`qwen3-8b`) to `kg_main` for pathing while the backend keeps
> the real Ollama name — `kg_main` only uses `--model` for the output path and logging, and the
> patched `_load` ignores it. Same fix in `run_local_judge.py`.
> **Lesson: probe one small paper end-to-end before launching a multi-hour batch.** This surfaced in
> 4 minutes what would otherwise have surfaced after ~6 hours with an empty output tree.

**Throughput after tuning:** ~26–30s per generation, 100% GPU, ~2.8 triples per call.
Full run = 18 papers × 2 ontologies ≈ 820 generations ≈ **6–7 hours** on the RTX 4060.

### 3. First quality read (strabismus2026, CEO, 131 triples)

**Working — and it validates the whole reason `relation` exists.** Top subjects are `RT-DETR` (27),
`SMOTE` (8), `Random Forest` (6). **None of those are in that paper's CS-NER entity list**, which held
only generic ML vocabulary. Free subjects recover exactly what the curated list misses on non-CS
papers. Good triples look like:
`(RT-DETR, achieves, intersection over union of 0.62)`,
`(Random Forest classifier, achieves, accuracy of 0.95)`,
`(two-stage system, comprises, Real-Time Detection Transformer (RT-DETR))`.

**Three defects already visible — this is the worklist for the guard/prompt round:**
1. **Subject fragmentation.** `RT-DETR`(27) / `Random Forest model`(8) / `Random Forest`(6) /
   `Random Forest classifier`(5) are one entity split four ways. `fixed` canonicalises against the
   CSV; `relation` has **no canonicalisation step at all**. This is the concrete price of free
   subjects and it matters for anything graph-shaped downstream.
2. **Vague objects.** The subject must appear literally in the source sentence; **the object has no
   quality guard**. Hence `(YOLOv8, produces, result)`, `(… designedFor, use case)`,
   `(YOLOv8, evaluatedOn, Fig 4)` — the last is a domain/range violation the judge should catch.
3. **Junk subjects on thin papers.** ugmo2024's top subject is `degree conferral` (8 triples), a
   phrase from the paper, not a research entity.

**Extraction quality tracks parse quality:** strabismus2026 (34 sections, 8k body words) → 131
triples; ugmo2024 (3 sections, 4 paragraphs) → 19, several junk. Exact duplicates are negligible (1
per paper).

### 5. ✅ RUN COMPLETE — 36/36, 0 failures, 2,132 triples (2026-08-27)

`run_local.py --all --extractor relation --both-ontologies`, qwen3:8b via Ollama, ~5.5 h,
~965 generations at ~21s each. **Every one of the 36 runs succeeded.**
Output: `output/<paper>/kg/relation{,_scinex}/qwen3-8b/triples.json`.

| | CEO (`relation`) | scinex (`relation_scinex`) |
|---|---|---|
| triples | **1,101** | **1,031** |
| distinct predicates | 21 | 24 |
| **top-2 predicate concentration** | **51%** | **38%** |

> **Count reconciliation:** the log's `[PARSED]` lines sum to 2,351 but the JSON files hold 2,132.
> `postprocess_triples` did **not** run (no such log line); the gap is **graph-level dedup** —
> `kg_builder` stores edges, so an identical (S,P,O) emitted from two paragraphs collapses to one.
> ~219 triples (9.3%) were exact duplicates. The per-run `Triples :` log lines sum to exactly 2,132.

**⭐ THE FINDING: scinex spreads predicate mass far more evenly than CEO.**
CEO collapses into two relations — `addresses` 26.1% + `uses` 24.9% = **51% of all triples**. Under
scinex `uses` falls to 8.8% and the mass redistributes into more specific relations:
`comparesAgainst` 4.2%→**9.1%**, `configures` 0→**6.2%**, `produces` →**4.2%**, `trainedOn`
4.9%→**6.0%**. Total yield is essentially unchanged (ratio **0.94**), so scinex is not extracting
*less* — it is extracting the **same volume at finer granularity**. The effect was visible at 3
papers and held all the way to 18.

**Why this matters:** the KGE/citation-prediction work found CEO and scinex **tied**, and
`results.md` flagged that citation prediction was too blunt an instrument to separate them. This is
the first measurement that does. **⚠ But finer-grained is only better if it is CORRECT** — the 64
`configures` triples where CEO said `uses` could be a better fit or a worse one. **Do not report this
as a scinex win until the judge has scored both.** That is the single highest-value next step.

**Object quality — my earlier warning (§23.3) was OVERSTATED; corrected here.** Measured over all
1,101 CEO triples: figure/table-reference objects (`evaluatedOn → Fig 4`) **13 (1.2%)**, generic
objects (`produces → result`) **4 (0.4%)**, clean remainder **1,084 (98.4%)**. I had generalised
from two bad examples in ugmo2024, the weakest-parsed paper in the corpus. Combined with subject
fragmentation measuring only **6.2%** (§23 addendum), the structural defects of relation mode are
**materially smaller** than the single-paper glance suggested. Free subjects look like a good trade.

**CS vs non-CS (CEO):**

| group | papers | triples | paragraphs | triples/para |
|---|---|---|---|---|
| CS | 13 | 748 | 251 | **2.98** |
| non-CS | 5 | 353 | 159 | **2.22** |

CS papers yield ~34% more per paragraph. **Note this is `relation` mode, where subjects are free** —
so the gap is genuine information density, *not* entity-list coverage. The coverage question only
bites in `fixed` mode, which is precisely why the `fixed` run is the informative next experiment.

**Yield tracks parse quality throughout** (triples/paragraph): videoseg2025 8.29, microwave2018 5.80,
vehiclemake2025 4.80, ugmo2024 4.75 … reststop2018 1.59, llamacorrupt2025 1.09.
Per-paper CEO/scinex ratios range 0.70 (roadwaylight2018) to 1.67 (llamacorrupt2025).

**NEXT, in this order:**
1. **Judge** — `python run_local_judge.py --extractor relation relation_scinex --max-per-paper 20`
   (~700 judgements, ~2h). Needs `ollama pull qwen2.5:7b-instruct` first — a **different family**
   from the extractor. This converts "scinex uses finer predicates" into "…correctly".
2. **`gold_eval.py export --n 150`** → human labels → `agree` → **read κ before quoting any judge
   number**.
3. **Then `fixed`** (~5.5h) for the fixed-vs-relation comparison across the CS/non-CS split.


---

## Session 22 — Corpus 7 → 18 papers, all parsed/enriched/planned; 3 more parser fixes; scipy DLL incident (2026-08-26/27)

### 1. The corpus is now 18 of the user's 20 papers

The user supplied a `bulk-download/` folder of 20 PDFs. **It is NOT the same 20 papers** — only 8
overlap. Checked the user's own pasted citation list against `papers/manifest.csv`: **all 20 matched,
0 discrepancies**, so the manifest was and remains authoritative.

- **8 of the 13 previously-unobtainable papers came from `bulk-download/`** (all of them were
  `oa_blocked` or `paywalled`): vehiclemake2025, csysguard2024, videoseg2025, ugmo2024,
  eyelandmark2024, textaug2023, trafficspeed2021, microwave2018.
- **3 more the user downloaded by hand** (found already correctly named inside `bulk-download/`):
  reststop2018, pesticide2025, llamacorrupt2025.
- **The other 12 files in `bulk-download/` are different papers by the same group** (Twitter traffic
  incidents, HDBSCAN congestion, bus-route ASP, GPS-tracker defects, MeddyCall, SrRL, …). The user's
  list confirms they are **not** part of this corpus — ignored, not imported.
- **Still missing (2):** `tripplanner2020` (ACIIDS/Springer, `10.1007/978-981-15-3380-8_43`) and
  `linkpred2015` (JIST/Springer, `10.1007/978-3-319-15615-6_9`).
- `manifest.csv` `access` column now reads: 7 `oa`, 11 `manual_bulk`, 2 `paywalled`.

**All 18 parsed with `--no-llm` and title-verified against the manifest — 18/18 correct.** Body words
1,985–9,147; body-capture rate spot-checked against raw PDF text at **77–102%**, so no paper is
silently losing content. References are *dropped* rather than leaking into body on the low-`ref`
papers (trafficspeed2021, ugmo2024) — harmless, arguably desirable, for triple extraction.

### 2. Three more parser fixes (all regression-checked on BERT.pdf)

1. **`filter_layout_noise` deleted the first line of a title.** The `b["y"] < 40` header-zone cut is
   right for pages 2+, but IEEE conference templates set the title at **y≈24 on page 1** —
   trafficspeed2021 lost *"Spatial-Temporal Traffic Speed Prediction on"* before `extract_title` ever
   saw it, leaving only *"Thailand Roads"* (2 words → failed `_title_final_ok` → fell back to the
   length heuristic → picked the author line `'Rathachai Chawuthai *'`). Fix: `top_cut = 0` on page 0,
   40 elsewhere.
2. **`extract_title` joined wrapped title lines in the wrong order.** With fix 1 applied, the title
   came out *"Thailand Roads Spatial-Temporal Traffic Speed Prediction on"* — by the time
   `extract_title` runs, blocks have been through column clustering, which can interleave a wrapped
   title. Candidates within 5% of max font size are now sorted by `(y, x)` before joining.
3. **Publisher furniture at title font size.** `'Measurement Development of machine learning…'` and
   `'ScienceDirect A Hybrid Method…'` — the Elsevier masthead and the ScienceDirect brand are set at
   the same size as the title, so the largest-font filter kept them. Added `sciencedirect`,
   `procedia`, and `measurement\s*$` to `_NON_TITLE_RE` (the last anchored to end-of-block so a real
   title beginning *"Measurement of …"* is not rejected).

BERT.pdf unchanged throughout: 44 sections / 66 paragraphs / 7,254 body words.

### 3. ⚠ INCIDENT — installing the ML stack broke the parser via Smart App Control. Read this before installing anything on the laptop.

Trying to run extraction locally (user chose the local option), I installed
`torch 2.13.0+cu126` + transformers + accelerate + bitsandbytes. Two consequences:

- **torch never worked** — `OSError [WinError 4551] An Application Control policy has blocked this
  file` on `torch\lib\shm.dll`. Confirmed the cause: **Windows 11 Smart App Control is ON**
  (`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState = 1`). This is
  the consumer feature on Home editions, not corporate WDAC. **Turning it off is one-way — Windows
  will not let you turn it back on without a reinstall — so it was not done and should not be.**
- **Worse: it took the parser down with it.** Smart App Control then also began blocking
  `scipy/interpolate/_rbfinterp_pythran.cp314-win_amd64.pyd`, which `sklearn` imports and
  `parser/layout.py` needs for column detection. **Every paper stopped parsing, including ones that
  had parsed fine hours earlier.** The blocked `.pyd` was dated Aug 25 and *never modified* — pip did
  not touch it; installing a pile of unsigned binaries appears to have made SAC tighten on that
  Python environment.
- **Fix:** uninstalled torch/transformers/accelerate/bitsandbytes (unusable anyway — uninstalling
  alone did NOT clear the block), then **downgraded `scipy` 1.18.1 → 1.17.1**, whose DLLs are not
  blocked. Parser restored and verified byte-identical on BERT.pdf. **`scipy==1.17.1` is now a
  requirement of the laptop environment; do not upgrade it.**
- `rdflib` was installed and kept — pure Python, works, and `ontology_loader` needs it for the scinex
  OWL.

### 4. Enrichment + coverage for all 18 — and the finding that actually matters

`enrich_entity_csv.py --source csner` on all 18 → 35–169 entities each.
`entity_coverage.py --plan output/corpus_plan.json` → **18/18 `fixed`, 0 `relation`** at the default
threshold of 15.

**But the composition finding from §20.6 now splits the corpus cleanly in two**, which makes the
fixed-vs-relation comparison far more interesting than it was:

- **The non-CS papers still get only generic ML vocabulary.** osmotic2026 (chemistry), oxidecrack2025
  (materials), strabismus2026 (clinical): `Random Forest`, `SMOTE`, `kNN`, `training`, `Learning`,
  `Input` — and **zero** domain terms (no "alkyl ammonium salts", no "oxide scale", no "horizontal
  strabismus").
- **The new CS papers get real, paper-specific entities**, exactly as predicted — CS-NER is annotated
  over CS/NLP papers:
  - textaug2023: `Text Classification`, `Data Augmentation`, `Word Embedding`, `Long Short-Term Memory`
  - videoseg2025: `Video Segmentation`, `Spatial Pyramid Pooling`, `Instance Segmentation`
  - vehiclemake2025: `License Plate Recognition`, `Batch Normalization`, `Inception Module`
  - llamacorrupt2025: `Large Language Models`, `Natural Language Generation`
  - csysguard2024: 169 entities, 46% multi-word

**So the corpus now contains a natural controlled contrast**: ~11 CS papers where the entity list is
genuinely informative, and ~7 non-CS papers where it is nearly vacuous. `fixed` vs `relation` should
diverge sharply between those two halves, and that split is a more defensible result than a single
corpus-wide precision number. **Report the two halves separately.**

### 5. Runners updated

`run_relation_extraction.sh` (VM) and `run_local.py` (Ollama) both carry the full 18-paper list.
`count_triples.py` unchanged. Extraction still has NOT run — it needs either the VM or a local Ollama
install.

**Files changed:** `parser/layout.py` (page-0 header-cut exemption), `parser/structure_builder.py`
(title reading-order sort, `_NON_TITLE_RE` publisher furniture), `papers/manifest.csv` (access
column), `run_relation_extraction.sh`, `run_local.py`. New on disk: 11 × `output/<id>/no-llm/`,
18 × `output/acl/<id>/Entity_<id>_enriched.csv`, `output/corpus_plan.json`, `papers/*.pdf` (7→18).

### NOT done / next session

1. **Extraction.** Either `bash run_relation_extraction.sh --with-fixed` on the VM (Qwen3-14B, the
   real numbers) or, locally, install Ollama + `ollama pull qwen3:8b` then
   `python run_local.py --all --extractor relation --also-fixed --both-ontologies` (preliminary
   numbers, different model). The laptop cannot use the transformers path at all — see §3.
2. Judge → gold sample → κ, unchanged from §19.
3. `tripplanner2020` and `linkpred2015` PDFs.

---

## Session 21 — Re-parse confirmed reproducible; local extraction BLOCKED by Windows Application Control; relation path dry-run-validated (2026-08-26)

### 1. Re-parsed all 7 papers — byte-identical to Session 20. Parsing is settled.

Re-ran `main.py papers/<id>.pdf --no-llm` over all 7, after backing the previous output up to
`output_backup_20260826/`. Every paper reproduced its Session-20 numbers **exactly** (sections,
paragraphs, body words, ref words all identical), and the 6 citation networks came back unchanged
(osmotic 10, oxidecrack 20, parkingyolo 20, roadwaylight 5, strabismus 11, traveltime 16 nodes;
gamlprop2025 still has none — not in Semantic Scholar). The parser is deterministic and done.

> The backup exists because a re-fetch could have degraded the citation networks under S2 rate
> limiting. It didn't. `output_backup_20260826/` can be deleted whenever.

### 2. ⛔ Extraction cannot run on the Windows laptop — Windows Application Control blocks torch

The user asked for relation-only extraction to be run here. It cannot be, for two independent
reasons, the second of which is fatal:

1. **The laptop GPU is an RTX 4060 Laptop with 8GB VRAM.** Qwen3-14B at 4-bit needs ~10GB. The
   project's standard model does not fit regardless.
2. **torch cannot load at all.** Installed cleanly (`torch 2.13.0+cu126` — the cu124 index the VM
   uses has no cp314 wheels; **cu126 does**, and PyPI shows torch up to 2.13 for Python 3.14). But
   every import dies with:
   `OSError: [WinError 4551] An Application Control policy has blocked this file. Error loading
   ...\torch\lib\shm.dll`
   That is WDAC/AppLocker — a corporate security control on a managed device. **I did not attempt to
   work around it and neither should a future session**; it needs IT to allowlist the DLLs, or the
   work goes to the VM.

`transformers 5.15.1` installed fine; `accelerate` and `bitsandbytes` install but fail on import for
the same reason (they import torch). **`rdflib` was also installed** (pure Python, unaffected) — it
was missing, and `kg_extraction/ontology_loader.load_ontology` needs it for the scinex OWL.

> **The broken torch install is still on disk (~2.5GB).** It does NOT break anything —
> `gold_eval`, `kg_evaluate`, `entity_coverage`, `enrich_entity_csv`, `kg_main` and
> `kg_extraction` were re-checked after installing and all still import (they import torch lazily,
> inside the model-loading functions). Left in place in case IT allowlists it later. Uninstall with
> `python -m pip uninstall torch transformers accelerate bitsandbytes` if it gets in the way.

### 3. What was validated instead: the relation-only path, dry-run without a model

`relation_extractor.py` had never been executed. Everything *after* generation is pure Python, so it
can be exercised with synthetic model output and no GPU. Harness:
`scratchpad/relation_dryrun.py` (kept out of the repo — it is a diagnostic, not a test suite).
**All checks pass:**

- **Prompt construction** — the free-subject system prompt builds (11,908 chars for CEO), contains
  the "Subjects are NOT fixed" block, and **no fixed-subject "entity CSV" wording leaks into it**
  (this was the specific risk Session 19 guarded with `self._subject_prompt`).
- **`_subject_in_sentence(..., entity_set=None)`** — the free-subject literal-match path behaves:
  `Random Forest` ✓, `SMOTE` ✓, `random forest` ✓ (case-insensitive), `XGBoost` ✗ (not in sentence),
  `Forest Random` ✗ (reordered).
- **Guards** — fed 6 synthetic candidates, kept exactly the 2 valid ones. Correctly dropped: a
  predicate outside the ontology (`isGreatAt`, no close match), a subject absent from the source
  sentence (`XGBoost`), a generic subject (`the proposed method`), and an empty object.
- **Malformed-input resilience** — empty output, a `<think>` block truncated by the token limit,
  non-JSON prose, and ```-fenced JSON all handled without crashing; the fenced case parses correctly.
- **scinex OWL** — loads 27 relations with full domain→range definitions, and the relation-mode
  scinex system prompt builds (12,663 chars).

**Two things that looked like bugs and are not** — recording them so nobody re-investigates:

- **Predicates are stored lowercased** (`fixed_extractor.py:947` emits `pred.lower()`), so triples
  carry `trainedon`, not `trainedOn`. Checked both consumers: `kg_evaluate._judge_triple` matches
  with `rel.lower() == pred`, and `gold_eval` builds `{k.lower(): v}` before lookup. **Both are
  case-insensitive, so the judge does get its predicate definition.** Cosmetic only — but paper-ready
  tables will show `trainedon`/`evaluatedon`/`comparesagainst` unless prettified at report time.
- A triple whose `source_sentence` is very short is dropped as `low-quality source`. That is a
  deliberate guard, not a parse failure; an early version of the harness tripped over it with a
  4-word test sentence.

### 4. NEW `run_relation_extraction.sh` — the VM runner

Project root, syntax-checked (`bash -n`), and every flag it passes verified against
`kg_main.py --help`. Runs relation-only over both ontologies for all 7 papers
(`--with-fixed` adds the fixed runs for the comparison), one process at a time, at
`--max-new-tokens 4096`, continuing past a failed paper instead of abandoning the batch, then prints
a triple-count table and the follow-on `kg_evaluate.py` command. Add the other 13 paper ids to
`PAPERS=(...)` once their PDFs are fetched, parsed and enriched.

### 5. Still no triples, therefore still no evaluation

The user asked for an assessment of extraction quality. **That assessment cannot be made yet** — no
triples exist for this corpus under any extractor. Nothing in `results.md` changes. The extraction
must run on the VM first; §3 raises the confidence that it will run without crashing, but says
nothing about output quality.

**Files:** new `run_relation_extraction.sh`. Installed (laptop only, non-functional except rdflib):
`torch 2.13.0+cu126`, `transformers 5.15.1`, `accelerate`, `bitsandbytes`, `rdflib`.
New on disk: `output_backup_20260826/` (deletable).

### NOT done / next session

1. **Run `bash run_relation_extraction.sh --with-fixed` on the VM.** This is the blocking step for
   everything downstream.
2. Then `kg_evaluate.py --all --extractor relation relation_scinex fixed fixed_scinex --resume
   --summary-out output/eval/judge_corpus.json`, `gold_eval.py export --n 150`, human labels,
   `gold_eval.py agree` — and read κ **before** quoting any judge number.
3. The 13 missing PDFs (3 OA-but-bot-blocked, 10 paywalled) still need a human with institutional
   access.

---

## Session 20 — Parser leftovers FIXED; all 7 corpus papers re-parsed with `--no-llm`; enrich + coverage done (2026-08-25)

Picks up directly from §10. The three parser items §10 left open are done, every corpus paper has
been re-parsed through the real `main.py --no-llm` pipeline (not just the `parse_check` harness),
and the two no-GPU stages after it (entity enrichment, coverage decision) have been run. **Still no
triples extracted — that is the next step and it needs the VM's GPU.**

### 1. The 2 MDPI titles — FIXED. 8/8 titles now correct and manifest-verified.

Both were diagnosed exactly right in §10, and both fixes are in `_title_block_ok`
(`parser/structure_builder.py`):

- **traveltime2022** — the masthead block is literally `'applied   sciences'` with multiple internal
  spaces, so `_NON_TITLE_RE`'s `applied sciences\b` never matched and the 19.02pt masthead outranked
  the 17.93pt title. Fix: `text = re.sub(r"\s+", " ", text.strip())` **before** the `_NON_TITLE_RE`
  test, so every furniture alternative sees normalised text.
- **parkingyolo2023** — the author-list guard `\d\s*(?:,|and)\s+[A-Z][a-z]+\s+[A-Z]` matched
  **"8 and Tracking A"** inside *"…Using YOLOv8 and Tracking Algorithms"* and rejected the real
  title, leaving a `Citation:` fragment to win. Fix: that alternative is now
  `(?:^|\s)[A-Z][a-z]+\s+\d\s*(?:,|and\s)` — the superscript digit must be **its own token** after a
  surname, which a digit inside a model name cannot be. Also added a citation-continuation reject
  (`[A-Z]\.\s*;|[A-Z][a-z]+,\s*[A-Z]\.`) for wrapped `Citation:` lines, which `_NON_TITLE_RE` only
  caught on the first line.

All 8 titles (BERT + the 7 corpus papers) now match `papers/manifest.csv` exactly. **This also
confirms the SM1674 download is the right article** — roadwaylight2018's parsed title matches the
manifest, closing the "a DOI suffix is not the publisher's PDF number" worry from §5.

### 2. Nature heading-splitting — IMPLEMENTED. This was the big one.

§10 designed this but did not write it. Implemented in `parser/layout.py` as
`_page_lines` + `_split_glued_headings`, replacing the old `_page_span_sizes`:

- `_page_lines(page)` returns per-LINE `(bbox, text, size, font, styled, uniform)` from
  `get_text("dict")`. `styled` = the line's **dominant** span (the one with the most characters, so a
  superscript citation marker can't define the line's style) is bold or italic — PyMuPDF span flags,
  bit 4 = bold, bit 1 = italic. This is now computed on **every page**, not just pages 0–1.
- `_split_glued_headings` splits a block when its first line is a heading and the rest is body:
  lead line **uniform in style**, ≤8 words, doesn't end in `.?!,;`, ≥100 chars of body follow, and
  either the lead is **≥0.5pt larger** than the body's dominant size **or** it is a bold/italic face
  change against an unstyled body. The split-off block carries `is_heading_hint=True`.
- `merge_blocks` now refuses to merge into or out of an `is_heading_hint` block — the split heading
  sits directly above its paragraph at the same indent, exactly the shape `merge_blocks` joins.
- `build_structure` honours the hint **before** `is_garbage` (which would otherwise drop a 2-word
  heading) and before `is_heading` (whose vocabulary can't cover "Related works", "Object
  detections", "Final prediction models").

**Font evidence, not a text rule** — the whole point is that Nature glues the heading into the same
PDF block as the paragraph, so no amount of vocabulary work could ever have found it.

**Why it matters beyond tidiness:** `source_meta["section"]` is fed to the extractor prompt and
checked by `_is_garbled_section`. Before this, every triple from the 3 Scientific Reports papers
claimed to come from "Abstract".

**Guard added while testing (real regression, caught on BERT):** the split turned BERT's byline into
a section called `Jacob Devlin`, because the first author is set larger than the rest of the author
list — front matter has the same shape as a run-in heading. `_looks_like_paragraph()` now requires
the body to read like prose (no `@`, at least one sentence terminator). A heading only ever runs into
prose; a byline runs into names, affiliations and an e-mail.

**A/B measurement (same parser, `_split_glued_headings` monkeypatched to a no-op vs live):**

| paper | paras OFF → ON | body words OFF → ON |
|---|---|---|
| strabismus2026 | 4 → 49 | 5,447 → 6,619 |
| oxidecrack2025 | 5 → 31 | 3,699 → 4,197 |
| BERT.pdf (regression) | 48 → 58 | 5,589 → 5,574 |

The Sci Rep papers previously held their whole body in 4–5 giant blocks. The word **gain** is real
content that the giant merged blocks had been getting suppressed by the paragraph-quality/garbage
filters; verified by diffing the two body-text sets, and paragraph-level duplicate count is **0** on
every paper (the split partitions its source block, it never copies). BERT's 15-word **loss** is the
heading text itself moving out of paragraphs into headings — no content lost.

**BERT gained its genuine bold run-in headings** (`Model Architecture`, `Input/Output
Representations`, `Task #1: Masked LM`, and the GLUE task names `MNLI`/`QQP`/`QNLI`/`SST-2`/`CoLA`/
`STS-B`/`MRPC`/`RTE`/`WNLI` in Appendix B.1): 32 → 44 sections. Two cosmetic imperfections remain,
both harmless: `Task #2:` is truncated (the heading wraps mid-line in the PDF), and one oxidecrack
block keeps a nested heading inline (`'Classification analytics Logistic Regression (LR) In order
to…'`).

> ⚠ **BERT's parse changed** (32→44 sections, 7,271→7,254 body words). Nothing re-parses the frozen
> 315-paper ACL corpus, so its extraction results are unaffected — but if that corpus is ever
> re-parsed, section metadata will shift and the KGE numbers would need re-checking.

### 3. roadwaylight2018 running-header junk headings — FIXED

`"2268 Sensors and Materials, Vol. 30, No. 10 (2018)"` satisfied the numeric-heading rule
`^\d+(\.\d+)*\s+[A-Z]` on every page, fragmenting the document into one junk section per page. New
`_is_running_header()` in `structure_builder.py`, consulted at the top of the numeric branch:
reject when the leading number is > 40 (`_MAX_SECTION_NUMBER` — real section numbers are small, a
number in the hundreds is a page number) **or** the line carries a bibliographic marker
(`Vol.|No.|pp.|ISSN|ISBN|DOI|(YYYY)`). Unit-checked: the running header is rejected while
`1 Introduction`, `2.1 Attention Model`, `3. Interpretation`, `4 Results and Discussion` and
`12 Related Work` all still pass.

### 4. Re-parsed all 7 with `main.py --no-llm` — the stale `output/*/fast/*` problem is cleared

Run as `PYTHONPATH=. PYTHONIOENCODING=utf-8 python main.py papers/<id>.pdf --no-llm`, one at a time.
Output now at `output/<id>/no-llm/output.{html,json}` — this is what `enrich_entity_csv.py` and
`kg_main.py` read.

| paper | sections | paras | figs | body words | ref words |
|---|---|---|---|---|---|
| strabismus2026 | 34 | 52 | 7 | 8,015 | 724 |
| traveltime2022 | 26 | 39 | 8 | 7,886 | 303 |
| oxidecrack2025 | 21 | 35 | 9 | 5,671 | 1,927 |
| parkingyolo2023 | 15 | 23 | 15 | 4,185 | 1,206 |
| roadwaylight2018 | 24 | 33 | 10 | 4,150 | 528 |
| osmotic2026 | 16 | 23 | 12 | 3,450 | 1,194 |
| gamlprop2025 | 12 | 13 | 2 | 1,985 | 334 |

Every paper has real body text and a correct title; **the three "bibliography only" papers from §9
are fully recovered.** Section hierarchies are now genuinely usable — traveltime2022 comes out with
its full `1. Introduction / 2. Literature Review / 2.1 … / 3.3.4 …` tree, strabismus2026 with 34
Nature-style sections.

**Citation networks fetched for 6/7** (`output/<id>/citation_network.json`, 5–20 nodes):
osmotic2026 10, oxidecrack2025 20, parkingyolo2023 20, roadwaylight2018 5, strabismus2026 11,
traveltime2022 16. **gamlprop2025 got none** — Chemical Engineering Transactions is not indexed in
Semantic Scholar. Irrelevant to this task (per-triple accuracy), relevant if this corpus is ever fed
to the global graph or KGE.

> ⏱ **Timing note:** the S2 citation fetch dominates the runtime — roadwaylight2018 alone ran >5 min
> and stalled a naive 10-min loop over all 7. Parse with `--no-citations` if you only need the HTML,
> or run the papers in the background one at a time.

### 5. Enrichment + coverage — all 7 papers say `fixed`, but read §6 before believing it

`enrich_entity_csv.py --paper <id> --source csner` ran clean on all 7 (it does work for non-ACL
papers, as §4 predicted from reading the code — 0 title seeds, entities from the gazetteer∩body
intersection). `entity_coverage.py --plan output/corpus_plan.json`:

| paper | entities | decision |
|---|---|---|
| strabismus2026 | 159 | fixed |
| parkingyolo2023 | 91 | fixed |
| traveltime2022 | 86 | fixed |
| oxidecrack2025 | 63 | fixed |
| osmotic2026 | 59 | fixed |
| roadwaylight2018 | 42 | fixed |
| gamlprop2025 | 36 | fixed |

**7 → fixed, 0 → relation** at the default threshold of 15. Written to `output/corpus_plan.json`.

### 6. ⚠ The coverage *count* passes but the coverage *composition* is exactly the domain-shift
problem TASK.md §3 predicted — do not read "7/7 fixed" as "domain shift solved"

Inspecting the enriched CSVs, every entity is **ML-methodology vocabulary**; the domain vocabulary is
absent:

- osmotic2026 (chemistry): `Algorithm, predictive, Learning, training, Machine Learning, Gaussian
  Process, Models, NCL, kNN, Compounds, Information, analysis, Experimental Data, Input` — **no**
  "alkyl ammonium salts", no "osmotic coefficient", no "activity coefficient".
- oxidecrack2025 (materials): `Features, Learning, Machine Learning, Classification, Random Forest,
  SMOTE, Logistic Regression, Regression, decision tree, Neural Network, training, Input, kNN,
  Evaluation` — **no** "oxide scale", no "high temperature oxidation".
- strabismus2026 (clinical): `Learning, Images, detection, Classification, Gaze, Image, Features,
  training, Deep Learning, Random Forest, SMOTE, Machine Learning, analysis, Models` — **no**
  "horizontal strabismus", no "orthotropia", no "RT-DETR".

So `fixed` on this corpus will describe **the ML half of each paper and systematically miss the
domain half** — and the CS-NER quality gate's known weak spot shows up too (`Learning`, `Input`,
`Information`, `analysis`, `predictive`, `Models` are generic single tokens). `relation` (free
subjects, relation ∈ ontology) is the exact complement of that blind spot.

**Recommendation for next session: run BOTH `fixed` and `relation` on all 7, per ontology, and
compare.** `entity_coverage.py`'s count-based decision was designed to catch a paper that gets
*nothing*; it cannot see that 59 entities are all the wrong 59. This is a finding about the corpus,
not a bug in the script — but the script's verdict should not be the last word here.

### 7. Import check — cleared (first Session-19 Python ever executed)

`gold_eval`, `kg_evaluate`, `entity_coverage`, `enrich_entity_csv`, `kg_main`, `kg_extraction` all
import clean on Python 3.14.7 locally; `kg_main.py --help` and `gold_eval.py --help` both render
(the `--extractor {…,relation,…}` and `--ontology {ceo,scinex}` flags are present as designed).
**Local Python has no torch/transformers**, so this is an import/CLI check only — the extractors'
runtime paths are still unexercised.

**Files changed:** `parser/structure_builder.py` (title guards, `_is_running_header`,
`is_heading_hint` handling in `build_structure`), `parser/layout.py` (`_page_lines`, `_lines_in`,
`_annotate_font_sizes` reworked to lines, `_looks_like_paragraph`, `_split_glued_headings`,
`merge_blocks` guard, all-pages annotation). New on disk: `output/<id>/no-llm/*` ×7,
`output/<id>/citation_network.json` ×6, `output/acl/<id>/Entity_<id>_enriched.csv` ×7,
`output/corpus_plan.json`.

### NOT done / next session

1. **Extraction — needs the VM GPU.** `for p in <the 7>; do kg_main.py --paper $p --extractor fixed
   --model Qwen/Qwen3-14B; kg_main.py --paper $p --extractor fixed --ontology scinex --model
   Qwen/Qwen3-14B; done` — and per §6, the same loop with `--extractor relation`. Use
   `--max-new-tokens 4096` (NOT 512 — that truncation cost 80 papers a re-run, §2l/§2m) and run one
   process at a time.
2. **The 13 missing PDFs** — 3 OA-but-bot-blocked (IEEE Xplore ×2, ScienceDirect ×1), 10 paywalled.
   Unchanged; needs a human with institutional access. The 7 on disk are enough to start the
   judge/κ loop.
3. Then §19's stages 5–8 unchanged: `kg_evaluate.py --resume --summary-out` → `gold_eval.py export
   --n 150` → human labels → `gold_eval.py agree` → read κ **before** quoting any judge number →
   `gold_eval.py errors` → fix guards → re-extract → re-measure.
4. Cosmetic parser leftovers, all verified harmless: `Task #2:` truncated in BERT; one oxidecrack
   block keeps a nested run-in heading inline; traveltime2022 still emits `References` before
   `5. Conclusions` (two-column reading-order artifact, content intact); the pre-existing ALL-CAPS
   false positives (`NER MNLI`, `BERT BERT`) are still deliberately untouched.

---

## Session 19 — New task: fixed-extraction accuracy via LLM-as-Judge + human agreement (2026-08-13)

**User's new direction:** stop broadening, go deep on quality. Focus **solely on fixed extraction**, on a **new corpus of ~20 handpicked research papers**, extracted under **both ontologies (CEO `fixed` + scinex `fixed_scinex`) and compared**. Judge the triples with an LLM, **validate that judge against human labels**, then iterate the extractor prompt/guards for the best accuracy achievable. Judge model = **Qwen2.5-7B-Instruct for now** (user may bring a stronger model or Claude in as a second judge later; gold sample size left to my discretion).

**1. `kg_evaluate.py` un-retired and upgraded** (it was declared retired in Session 12 in favour of the KGE citation-prediction eval; that eval is unaffected and its numbers still stand).
- `triple_id(triple, extractor)` — sha1 over extractor-family|paper|subject|predicate|object|source_sentence[:300], 12 hex. Content-keyed so human labels rejoin the same triples across judge re-runs, and so CEO/scinex triples that happen to be byte-identical stay distinct.
- Per-paper **entity-CSV alias auto-resolution** (`load_aliases_for`, reusing `kg_main.resolve_entity_csv`). Previously one global `--entity-csv` was applied to every paper — wrong on a multi-paper corpus, and the judge then penalised abbreviations it had never been told about. `--entity-csv` still overrides.
- **`find_extractor_dirs` now matches the extractor family exactly** (`k.split("/")[0] in wanted`). The old `any(f in k ...)` substring test meant `--extractor fixed` silently also judged `fixed_scinex` — which would have quietly merged the two ontologies in this very comparison.
- `compute_summary` gains **`by_predicate`** (n / verdict counts / precision per relation).
- **The judge now sees the predicate's ontology definition** (`schema_for()`: CEO from `fixed_extractor._CEO_SCHEMA`, scinex parsed from the OWL via `ontology_loader.load_ontology`, cached per extractor family; `--ontology-file`, default `scinex_refined_14.owl`). Without it the judge scored a predicate by the everyday meaning of its English name, so a CEO-vs-scinex comparison would have measured nothing but name plausibility. `JUDGE_SYSTEM_PROMPT` gained a matching "ontology conformance" rule: fits definition → CORRECT, loose fit / wrong range type → PARTIAL, domain↔range swapped → INCORRECT. **This changes what "precision" means** — it is now faithfulness AND ontology conformance, so these numbers are not comparable to the old 80%/85% BERT figures.
- New flags: `--resume` (reuse verdicts already in `evaluation.json`, matched by triple id — only judge what's new), `--max-per-paper N` (deterministic subsample by id, same N triples every re-run), `--ontology-file`, `--summary-out JSON` (corpus-level report: per extractor, per predicate, per paper). Final printed table now has a CORPUS block under the per-paper rows.

**2. NEW `gold_eval.py` — the human side of the loop.** Three subcommands:
- `export` — sample triples → CSV (`triple_id, paper, extractor, model, subject, predicate, object, object_type, predicate_definition, section, source_sentence, verdict, note`), `verdict` blank. `predicate_definition` is filled from the SAME `schema_for()` the judge uses, and `export` prints the verdict rubric — if the human labels by intuition while the judge labels by domain/range, low κ would mean "different rubrics", not "bad judge". **Stratified round-robin over (extractor × predicate) buckets**, not proportional: proportional sampling would spend the whole budget on `uses`/`achieves` and never surface the rare predicates, which is exactly where the guards are weakest. **The judge's verdict is deliberately NOT in the export** — anchoring the human on it would inflate agreement. Written utf-8-sig so Excel opens it cleanly.
- `agree` — joins the filled CSV back by `triple_id`: raw agreement, **Cohen's κ** (3-class and a CORRECT/NOT_CORRECT binary collapse), confusion matrix, per-extractor and per-predicate tables (each with both sides' precision), and a full disagreement dump (printed + saved to `--out` JSON). `--b other.csv` compares **two label files** instead of labels-vs-judge — this is the path for "Claude as a second judge", annotator A vs B, or 7B vs 14B judge, with no new code. Kappa is pure-python (no sklearn dependency).
- `errors` — the judge's PARTIAL/INCORRECT verdicts grouped by (extractor, predicate), ranked worst-first, with examples incl. the judge's reason. This is the worklist that drives the next round of `fixed_extractor.py` prompt/guard edits.
- Verdict parsing accepts C/P/I, Y/N, 1/0 shorthand — humans do not type "INCORRECT" 150 times.

**3. Environment reality check (important for whoever picks this up):** the Windows laptop this was written on has **no Python at all** (only the `WindowsApps` Store stub — `python`/`python3` resolve to it and fail) and **no `output/` directory** in `Downloads/project_scinex`. So this session is **code-only: nothing was executed, not even `py_compile`.** First action on the VM: `python3 -c "import gold_eval, kg_evaluate"` to catch anything I got wrong by eye.

**4. Per-paper chain for a handpicked (non-ACL) paper** — verified by reading the code, not by running it: `enrich_entity_csv.py --source csner` works for **any parsed paper**, not just CS-NER/ACL ones (`discover_paper_ids` accepts any `output/<id>/no-llm/output.html`; the title-seed CSV is optional — `load_seed_entities` on a missing file just returns no seeds, and the gazetteer∩text intersection supplies the entity pool). So: `main.py <pdf> --no-llm` → `enrich_entity_csv.py --paper <id> --source csner` → `kg_main.py --paper <id> --extractor fixed [--ontology scinex]`.

**8. NEW `TASK.md`** — the brief for this whole task in one place: goal and what "done" looks like, what is explicitly out of scope (KGE results frozen, 315-corpus untouched, fixed extraction only), the corpus + access table, why the `fixed`/`relation` split exists, the 8 pipeline stages with exact commands, the verdict rubric, what to report (incl. κ reading and the rule that coverage failures never get folded into a precision number), the file map, the GPU constraints, and a live checklist. `Claude.md`'s status block now points at it.

**9. ⚠⚠ PARSING AUDIT of the first 7 corpus papers (2026-08-25) — 3 of 6 parsed papers contain NO body text at all. This blocks the whole task; do not extract until it's fixed.**
The user ran `main.py` on the downloaded PDFs (mode `fast`) and asked what went wrong. Measured from `output/<id>/fast/output.{html,json}` — body words = total minus everything from the `References` heading onward:

| paper | venue style | sections | body words | ref words | verdict |
|---|---|---|---|---|---|
| traveltime2022 | MDPI, numbered | 25 | 7,487 | 1,259 | OK |
| parkingyolo2023 | MDPI, numbered | 14 | 4,826 | 1,207 | OK |
| gamlprop2025 | CET, numbered | 10 | 2,124 | 335 | OK |
| **osmotic2026** | **Sci Rep, unnumbered** | **1** | **366** | 1,533 | **BODY LOST** |
| **oxidecrack2025** | **Sci Rep, unnumbered** | **1** | **362** | 2,186 | **BODY LOST** |
| **strabismus2026** | **Sci Rep, unnumbered** | **1** | **362** | 1,020 | **BODY LOST** |
| **roadwaylight2018** | Sensors & Materials | — | — | — | **NO OUTPUT AT ALL** |

- **Root cause of the three total losses:** `is_heading()` (`parser/structure_builder.py:117`) only recognises **numbered** (`1 X`, `1. X`, `2.1 X`), **roman**, **appendix-lettered** and **ALL-CAPS** headings. Nature/Scientific Reports uses plain title-case headings — `Introduction`, `Results`, `Discussion`, `Methods` — which match **none** of those rules. So the only "heading" ever detected is `References` (via `is_references`, a separate `startswith("references")` check at line 47), and `build_structure`'s `if not current_section: continue` (line ~558) silently discards **every block before the first heading** — i.e. the entire paper. What survives is the bibliography: refs are **73–85%** of those documents. Those ~362 "body words" are HTML/CSS boilerplate, not content. **Extraction on these three would mine the reference list**, and the near-zero triple count would look like an entity-coverage or quality problem when it is a parsing problem — exactly the confusion `TASK.md §6` says must never be folded together.
- **The abstract is dropped from ALL SIX papers**, the good ones included — same `if not current_section` rule, since the abstract precedes the first numbered heading. `grep -i abstract` on every output.html matches only the `.ltx_abstract` CSS class, never any text. Abstracts are dense in exactly the entities/relations this task wants.
- **Every title is wrong on all six**: `applied sciences`, `sensors`, `CHEMICAL ENGINEERING TRANSACTIONS` (journal mastheads), `Keywords  Machine learning, …` (keyword line), `Scientific Reports | (2026) 16:18278 …` (DOI footer). `extract_title()` (line 289) scores the first 20 blocks by **length alone** (<12 words → +2) and `max()` returns the first top-scorer, so it reliably picks the masthead. **No font-size signal is available to fix this**: `parser/layout.py:181` uses `page.get_text("blocks")`, which yields only `text`+`bbox` — sizes/flags require `get_text("dict")` spans.
- **roadwaylight2018:** `output/roadwaylight2018/fast/` exists but is **empty** — `main.py` created the dir and then failed. The PDF is fine (57 `/Font`, 12 `/FontFile` → real text layer, not a scan). `output/strabismus2026/no-llm/` is also empty, suggesting a second aborted run. **Needs a re-run on the VM to capture the traceback — the failure is not reproducible here (no Python).**
- Minor: in traveltime2022 the `References` section is emitted **before** `5. Conclusions` — two-column reading-order artifact on the last page. Content is intact, only ordering is off.
- **→ SUPERSEDED BY §10: most of this is now FIXED and verified. Read §10 for current state.**
- **Fix design (as diagnosed; see §10 for what was actually implemented):** (a) add an unnumbered title-case heading rule, safest as a known-section vocabulary (`Introduction|Results|Discussion|Methods|Materials and Methods|Conclusion[s]|Related Work|Background|Data availability|Acknowledg(e)ments|Supplementary…`) matched case-insensitively on a short standalone block — a vocabulary list can't fire on body prose the way a generic "short title-case line" rule would; (b) capture pre-first-heading content into an implicit `Abstract`/frontmatter section instead of dropping it; (c) switch `layout.py` to `get_text("dict")` so title detection can use max font size on page 1, with the current length heuristic as fallback. **Any change here must be regression-checked against the known-good ACL papers** (the J13-4001 / D17-1028 precedent in this file) before the 315-corpus results are trusted again.

**10. PARSER FIXED (2026-08-25, same session) — all 7 corpus papers now parse with real body text. Verified by running the parser locally, not by reading code.**

**⚠ First, a correction to a standing assumption in these notes: this laptop DOES have Python.** `python`/`python3`/`py` → **Python 3.14.7**, with `pymupdf`, `pdfplumber`, `ftfy`, `bs4`, `networkx`, `pandas` already installed (only `lxml` missing). So Part-1 parsing can be developed and tested **locally**; only the GPU/LLM stages need the VM. Two local gotchas: run from the project root with `PYTHONPATH=.`, and set **`PYTHONIOENCODING=utf-8`** — the Windows console is cp1252 and `citation/network.py:69` prints a `✔`, which otherwise kills the process with `UnicodeEncodeError` before any work starts.

**Results — measured with `scratchpad/parse_stats.py` (title / sections / body words / ref words), body = every paragraph outside the References section:**

| paper | body words BEFORE | body words AFTER | title AFTER |
|---|---|---|---|
| BERT.pdf (ACL regression) | 7,111 | **7,271** | ✅ now full ("…for Language Understanding" — was truncated before) |
| traveltime2022 | 7,771 | **7,886** | ❌ still wrong (see below) |
| parkingyolo2023 | 4,020 | **4,185** | ❌ still wrong (see below) |
| gamlprop2025 | 1,717 | **1,985** | ✅ correct |
| strabismus2026 | **0** | **6,456** | ✅ correct |
| osmotic2026 | **0** | **3,077** | ✅ correct |
| oxidecrack2025 | **0** | **4,978** | ✅ correct |
| roadwaylight2018 | **CRASH** | **4,116** | ✅ correct |

Every paper gained words (the abstract, previously dropped everywhere), no paper lost any — BERT is the regression check for the frozen 315-paper ACL corpus and its 32 sections/headings are unchanged apart from a new leading `Abstract` section.

**What was changed (4 fixes):**
1. **Crash — `parser/structure_builder.py` `_reading_order_key`.** roadwaylight2018's 13 detected tables all had `page=None, y=None` (no `Table N:` caption matched, and `assign_table_positions`' no-caption fallback set `y` but never `page`), so the reading-order `sorted()` compared `None < int` → `TypeError`, aborting the parse and leaving an empty output dir. Now None-safe (`block.get("page") or 0`), and both `assign_table_positions` fallbacks set `page`.
2. **Unnumbered headings — `is_heading`.** Added `_SECTION_VOCAB`, a closed set of ~55 section names matched case-insensitively via `_section_vocab_key` (strips numbering and trailing `:`/`.`). A **closed vocabulary, not a general "short title-case line" rule** — the general rule would fire on body prose. This is what un-broke the three Scientific Reports papers.
3. **Abstract rescue — `build_structure`.** Blocks before the first heading are now buffered (`pre_heading_texts`) instead of dropped, then filtered by `looks_like_abstract_prose()` (≥40 words, ≥2 sentences, ≥80% letters, no affiliation/copyright/email markers) and inserted as a leading `Abstract` section. It doubles as a **safety net**: if heading detection ever fails completely again, the body lands in `Abstract` rather than vanishing. Also added a loud warning when a document ends up with only a `References` section.
4. **Title — font size.** `parser/layout.py` now annotates each block with the max font size of its spans (`_page_spans` + `_annotate_font_sizes`, using `get_text("dict")`; `merge_blocks` propagates the max and `extract_blocks` annotates the first 2 pages). `extract_title` takes the largest-font blocks on page 1, joins those within 5% of the max (titles wrap across blocks), strips journal article-type labels (`_TITLE_LABEL_RE`: "Article …", "Review …"), and rejects front-matter furniture (`_NON_TITLE_RE`: mastheads, DOI, ISSN, dates, "Citation:", emails). Old length-only heuristic kept as fallback.

**⚠ STILL BROKEN — 2 of 8 titles, both MDPI, both diagnosed exactly (this is the first thing to fix next session; ~2 small edits):**
- **traveltime2022** → `'Article Travel Time Prediction on Long-Distance Road Segments'`. The masthead block is literally `'applied   sciences'` **with multiple internal spaces**, so `_NON_TITLE_RE`'s `applied sciences\b` never matches and the masthead (19.02pt) outranks the real title (17.93pt). **Fix: collapse internal whitespace (`re.sub(r"\s+"," ",text)`) BEFORE the `_NON_TITLE_RE` test in `_title_block_ok`.**
- **parkingyolo2023** → `'M.P.; Chawuthai, R. Parking Time'` (a fragment of the `Citation:` block). The real title block is rejected by my author-list guard in `_title_block_ok`: the regex `\d\s*(?:,|and)\s+[A-Z][a-z]+\s+[A-Z]` matches **"8 and Tracking A"** inside *"…Using YOLOv**8 and Tracking A**lgorithms"*. **Fix: narrow that alternative to real affiliation markers (e.g. require `\d\s*,\s*\*` / `[A-Z][a-z]+\s+[A-Z]\.?\s+\d\s*,`) so a digit inside a model name can't trip it.** Also add `Citation:`-style blocks to the reject list (partially there — it fires on the first block but not on wrapped continuation lines).

**⚠ ALSO STILL OPEN (lower priority, all verified present):**
- **The 3 Scientific Reports papers have their body, but as ONE `Abstract` section** (sections = `Abstract | References | Declarations`). Nature glues the heading onto the front of the paragraph **in the same block at a larger font** — e.g. a single block reading *"Literature reviews The following review includes both related work and tech…"* (11.0pt lead, 9.0pt body). No vocabulary rule can catch that. **The fix, already designed but NOT written: split a block when its leading spans are ≥0.5pt larger than the block's dominant size, the lead is ≤8 words and doesn't end in sentence punctuation, and ≥100 chars of body follow; mark the split-off block `is_heading_hint=True` and have `is_heading` honour that flag.** This needs span-level data on **all** pages (currently annotated only for pages 0–1) and a guard in `merge_blocks` so the split isn't immediately re-merged. **This matters beyond tidiness: `source_meta["section"]` is fed to the extractor prompt and checked by `_is_garbled_section`, so every triple from those papers currently claims to come from "Abstract".**
- **roadwaylight2018 has 6 junk sections from running page headers** — `"2268 Sensors and Materials, Vol. 30, No. 10 (2018)"` matches the numeric-heading rule `^\d+(\.\d+)*\s+[A-Z]`. **Fix: require the leading section number ≤ ~40 and reject text matching `Vol\.|No\.|pp\.|ISSN|\(\d{4}\)`.** Content is intact, only fragmented.
- **Pre-existing ALL-CAPS false-positive headings, deliberately NOT touched**: BERT gets `NER MNLI` and `BERT BERT` (table-header fragments), gamlprop2025 gets `CHEMICAL ENGINEERING TRANSACTIONS`. These predate this session and the ALL-CAPS rule is used by the frozen 315-paper corpus, so changing it needs its own regression check.
- **traveltime2022 still emits `References` before `5. Conclusions`** — two-column reading-order artifact; content intact.

**⚠ `output/*/fast/output.html` on disk is STALE** — produced by the old parser (3 papers with no body, 1 missing). **Everything must be re-parsed before any extraction**, and note those were run in `fast` mode while the pipeline convention is `--no-llm`.

**Test harness: `parse_check.py`** (now in the project root). Run:
`PYTHONPATH=. PYTHONIOENCODING=utf-8 python parse_check.py BERT.pdf papers/*.pdf --out after.json`
It prints title / sections / paragraphs / tables / figures / body words / ref words / heading list per PDF and dumps JSON for diffing. **Always include `BERT.pdf` — it is the ACL regression check for the frozen corpus.**

**NOT done / next session:**
0. **Sanity-import everything first**: `python3 -c "import gold_eval, kg_evaluate, entity_coverage, kg_extraction"` and `python3 kg_main.py --help` (the `kg_main` changes are still unexercised — `entity_coverage` imports it lazily, so the import check alone won't catch them).
0b. **Finish the parser: the 2 MDPI title fixes above (quick), then the Nature heading-split (bigger, matters for `section` metadata).** Re-run the harness incl. `BERT.pdf` after each change.
0c. **Re-parse every paper with `--no-llm`** once the parser is settled, and check each `Title:` against `papers/manifest.csv`.
1. Copy `papers/` to the VM, get the 13 missing PDFs, then parse→enrich→coverage→extract for both ontologies. **Check each parsed "Title:" against `papers/manifest.csv`** — that is the only verification that the downloaded PDF is the intended paper.
2. `kg_evaluate.py --all --extractor fixed fixed_scinex --resume --summary-out output/eval/judge_corpus.json` — baseline judge numbers.
3. `gold_eval.py export --n 150` → user labels → `gold_eval.py agree --out gold/agreement.json`. **Read κ before trusting any judge number**: κ ≥ 0.6 = judge usable as a proxy; below that, fix the judge prompt (`JUDGE_SYSTEM_PROMPT`) before touching the extractor.
4. Then the improvement loop off `gold_eval.py errors` + the human-flagged failures, re-extract, re-judge, re-measure on the SAME gold ids (they survive re-extraction only if the triple text is unchanged — expect some ids to drop out; `agree` reports unmatched counts).

**5. The corpus arrived later the same day — 20 papers by R. Chawuthai's group** (applied ML: ophthalmology, chemistry, materials, traffic/ITS, cloud, CV, NLP). The user gave citations, not files, and asked me to fetch the open-access ones automatically.
- **`papers/manifest.csv`** (new) — all 20 with paper_id, year, venue, DOI, access status, PDF URL. DOIs came from the author's own publication page (`https://rathachai.creatier.pro/publications`), which was faster and more reliable than 20 separate searches.
- **`fetch_corpus.sh`** (new) — downloads every `oa` row into `papers/<paper_id>.pdf`, idempotent, and **deletes anything that isn't `%PDF`** (publisher bot-blocks return a 400-byte HTML "Access Denied" that `main.py` would happily "parse" into garbage). Prints the manual-download list for the rest. **Verified working locally: 7/7 OA papers downloaded.**
- **Got (7):** strabismus2026, osmotic2026, oxidecrack2025 (Nature/Sci Rep — plain `…/articles/<id>.pdf` works), parkingyolo2023, traveltime2022 (MDPI — **www.mdpi.com is Akamai-blocked; `mdpi-res.com/d_attachment/...` is not**), roadwaylight2018 (Sensors & Materials), gamlprop2025 (Chem Eng Transactions).
- **Blocked (13):** IEEE Xplore (Akamai `APM_DO_NOT_TOUCH` script) and ScienceDirect (JS shell) block scripted downloads even for their OA articles — vehiclemake2025, csysguard2024, reststop2018 are OA but must be fetched by hand. The other 10 are genuinely paywalled.
- **⚠ Trap worth remembering: a DOI suffix is not the publisher's PDF number.** I guessed `SM1842.pdf` from DOI `10.18494/SAM.2018.1842`; it returned a valid PDF of a *different* article. The real file is `SM1674.pdf`. Always resolve the article page. Since PDF text streams are compressed, `strings`/`grep` can't verify a downloaded PDF's title — **the real check is `main.py`'s parsed "Title:" log line; verify all 7 against manifest.csv on the first parse run.**

**6. NEW `kg_extraction/relation_extractor.py` — relation-only extraction, per the user's fallback instruction** ("try to check it from everywhere, but if there's none then we change the fixed extraction to fix only relation").
- The problem it solves: CS-NER is annotated over CS/NLP papers. The chemistry/materials/clinical papers in this corpus will intersect the gazetteer thinly, so `fixed` would report near-zero triples for **coverage** reasons that look like **quality** reasons.
- `RelationOnlyExtractor(FixedTripleExtractor)`: relation ∈ ontology, subject free but **must appear literally in the source sentence** (`_subject_in_sentence` with `entity_set=None` falls back to a literal word-boundary match). Every other guard is inherited unchanged, so `fixed` vs `relation` differs in exactly one dimension.
- Supporting refactors in `fixed_extractor.py`: `_parse_fixed_output(raw, entity_set=None, ...)` skips the CSV-canonicalisation chain (all the later `canonical_subj is None` fallbacks are naturally skipped once the subject is taken as written) and caps free subjects at 8 words; `_make_fixed_system_prompt` gained `subject_block`/`subject_rule`/`subject_field` params (defaults = the existing fixed wording, so `fixed` output is unchanged); `self._subject_prompt` is stored so `update_relations()` can't silently revert a relation-only prompt to fixed-subject wording; the init log no longer does `len(None)`.
- `kg_main.py`: `--extractor relation`, `_ontology_for(base)` helper shared by fixed and relation (scinex → `relation_scinex`), model-slug subdir list extended. Relation is deliberately NOT in `ENTITY_EXTRACTORS` (so it runs when no CSV exists) and NOT in `GLOBAL_GRAPH_EXTRACTORS` (the global graph's entity nodes assume CSV-canonicalised entities).
- `kg_evaluate.schema_for` maps `relation`→CEO and `relation_scinex`→OWL, so judged relation-mode triples still get predicate definitions.

**7. NEW `entity_coverage.py`** — reads only (no GPU): per paper, resolves the entity CSV the extractor would use, counts TP entities vs all rows, and splits the corpus into `fixed` (≥ `--min-entities`, default 15) vs `relation`. Prints the ready-to-run bash loops for both groups and optionally writes `output/corpus_plan.json`. It also reminds you to rule out the two boring causes (enrichment never ran; PDF never parsed) before believing "the gazetteer doesn't cover this domain".

**Files:** new `gold_eval.py`, `fetch_corpus.sh`, `papers/manifest.csv`, `entity_coverage.py`, `kg_extraction/relation_extractor.py`; edited `kg_evaluate.py`, `kg_main.py`, `kg_extraction/fixed_extractor.py`, `kg_extraction/__init__.py`, `Claude.md`, `hands_off.md`, `results.md` (pointer only, no numbers changed). **Only `fetch_corpus.sh` has actually been executed** — everything Python is unrun (no interpreter on this machine).

---

## Session 18 — Persistent global KG (paper big-nodes + augmented entity nodes + real cites edges) (2026-07-17)

**New direction from the user.** Reframe the KG: each paper = a **big node**; its fixed-extraction entities = **augmented nodes** that exist to help link prediction between big nodes. Instead of each extraction returning one isolated KG, every extracted paper is **added into one growing global graph** until the whole corpus lives in a single space. Explicit design choice (user): **citation edges become REAL paper→paper edges in this graph** — a deliberate break from `kg_transe_pipeline.py`'s methodology where citations are held out as eval-only ground truth (that pipeline is untouched; all prior KGE numbers remain valid). Future work (parts 2-3, NOT in this session): masked-node training (cut 1-2 big nodes, predict the missing link) and a citation-only ablation graph (no augmented nodes) evaluated the same way.

**1. NEW `kg_extraction/global_graph.py` — persistent incremental cross-paper KG.**
- Store: `output/global_kg/<extractor>/<model_slug>/graph.graphml` + `meta.json` (extractor ∈ {fixed, fixed_scinex}, mirroring the per-paper `kg/<extractor>/<model>/` convention). MultiDiGraph.
- Node model: `paper:<id>` big-nodes (`type="paper"`, title/year attrs) — entity nodes (`type="entity"`, produced by the SAME `KnowledgeGraphBuilder` used per paper, so normalisation/canonicalisation is identical) — `paper_stub` nodes for citation neighbors not yet in the corpus.
- Edges: entity→predicate→entity (from triples, deduped by relation|paper|section key), `paper→mentions→entity`, and `paper→cites→paper` (from `citation_network.json` edges; source cites target per `citation/network.py`).
- **Stub resolution (the key identity problem):** citation edges reference neighbors by raw S2 id. A neighbor gets a `paper:<s2_id>` stub with its title; `meta["s2_to_paper"]` accumulates the s2→acl mapping as papers merge; when a previously-stubbed paper is later extracted for real, its stub is `nx.relabel_nodes`-merged into `paper:<acl_id>` and all previously-attached edges survive. Merge order doesn't matter.
- Idempotency: `meta["merged_papers"][id].n_triples` — re-merging an unchanged paper is a no-op; a changed triple count re-merges automatically.
- `connectivity_report(graph, ids)` — per paper: # cites edges, # other papers reachable via shared entities, isolated yes/no.

**2. `kg_main.py` — auto-merge wired in (fixed/fixed_scinex only, `GLOBAL_GRAPH_EXTRACTORS`).**
- After each successful fixed extraction the paper merges into the in-memory global graph; **also on the `--skip-existing` path** (reads the existing `triples.json`) — so ONE `--all --skip-existing` pass over the current 155-paper corpus backfills the entire global graph WITHOUT re-running the LLM.
- Save strategy: in-memory merges, checkpoint every `--global-save-every` N papers (default 25) + final save in a `finally:` (partial batches persist — this VM has a history of killed runs). New flags: `--global-graph-dir` (default `output/global_kg`), `--no-global-merge`, `--global-save-every`. Per-paper triples.json/kg.graphml outputs unchanged.

**3. NEW `expand_and_validate.py` — the smoke test the user asked for.** Uses ONE seed paper's `citation_network.json` to download new papers from ACL Anthology (reusing `citation_expand_pipeline.py`'s resolve/download + `run_acl_batch.run_paper` parse + `enrich_entity_csv --source csner` + `kg_main --skip-existing` extraction), then prints a connectivity report on exactly the new ids: present in graph? # cites edges? # shared-entity papers? isolated (=bug signal)? Flags to resume mid-chain: `--skip-download/--skip-parse/--skip-extract`.
- **⚠ Seed must be an ACL-Anthology-pulled paper (user instruction), NOT the manually-added "BERT" folder** — BERT has no ACL id and its S2 network historically pulled wrong papers (Sentence-BERT, see Known Bugs).
- **⚠ Seed choice matters — `2020.acl-main.185` FAILED with 0/8 resolvable (tried live 2026-07-17):** its 20 neighbors = 10 recent citing papers (2024-25, not on ACL Anthology) + 10 references that are mostly arXiv/ICLR (Seq2Sick, HotFlip-with-mismatched-title, Show-and-Fool…). Each unresolvable title costs ~12s of DBLP/S2 fallback → 4 min for nothing. **Fix: pre-scan seeds OFFLINE against `acl_title_index.json`** (scratchpad script `find_seed.py`, no network): best seeds on the current local corpus = **`D13-1109` (10 new locally-index-resolvable ACL neighbors: J17-2001, D15-1163, Q13-1035, N13-1056, P12-1017, P11-1002…)** and `W14-5318` (10, mostly W-workshop); also E14-4024 (9), W19-8615 / P19-1117 / P19-1454 (7 each). Old-style-id seeds (P/D/N/W 11-19) resolve from the local index instantly; recent *-main seeds mostly don't (their citers are post-index arXiv papers).

**4. LOCAL smoke-test constraints (user is running this on their Windows machine, NOT the VM):** local GPU tops out around a 7B model → use the already-cached judge model as the extractor: `--model Qwen/Qwen2.5-7B-Instruct` (works fine — the extractor's Qwen3-specific bits, `/no_think` prefix + `<think>`-stripping, are no-ops on Qwen2.5). Outputs land under `kg/fixed/Qwen2.5-7B-Instruct/` + `output/global_kg/fixed/Qwen2.5-7B-Instruct/`, cleanly ISOLATED from the VM's Qwen3-14B data — triple quality is below corpus standard but irrelevant for validating merge logic. Also learned: `expand_and_validate.py` is the FULL chain (step 4 loads the LLM ~10GB for 14B), so on a small GPU either pass a small `--model` or use `--skip-extract`. **The command in flight when the session closed:**
```bash
python expand_and_validate.py --seed D13-1109 --count 8 --model Qwen/Qwen2.5-7B-Instruct
```
(User was about to run this locally; outcome unknown — check `output/global_kg/fixed/Qwen2.5-7B-Instruct/` and the printed connectivity report next session.)

**Verified this session (Windows laptop, no GPU — code-level only):** `py_compile` clean on all touched files; synthetic unit test of `global_graph.py` PASSED (merge A→stub for B created → idempotent re-merge skipped → B extracted → stub resolved, cites edge survived relabel → graphml round-trip counts identical → connectivity report correct → changed triples re-merge). `kg_main.py --help` + `expand_and_validate.py --help` import/parse clean.

**NOT yet done / next session:**
1. **Smoke test:** locally in flight as `python expand_and_validate.py --seed D13-1109 --count 8 --model Qwen/Qwen2.5-7B-Instruct` (see #4 — outcome unknown, check first thing). On the VM the same test runs with the default `--model Qwen/Qwen3-14B`. Want: 0 isolated new papers in the report.
2. **Full-corpus backfill (no LLM cost):** `python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B --skip-existing` → populates `output/global_kg/fixed/Qwen3-14B/` from the existing triples (repeat with `--ontology scinex` for the scinex graph if wanted).
3. Then parts 2-3: masked-big-node link-prediction training on the global graph + citation-only ablation (no augmented nodes) — the actual experiment this foundation is for.

**Files:** new `kg_extraction/global_graph.py`, new `expand_and_validate.py`, edited `kg_main.py` + `kg_extraction/__init__.py`. No changes to `kg_transe_pipeline.py`/`kg_evaluate.py`/any output format; `results.md` unaffected (no result numbers changed).

---

## Session 17 — Corpus → 102 papers, fetch-arbitrary-ACL validated, KGE precision FIXED (0.05→0.39) (2026-06-17/18)

**1. Corpus expanded 54 → 102 parsed papers.**
- Ran `run_acl_batch.py --timeout 1200` with the S2 key live → the 40 unparsed CS-NER papers parsed (9× citation-bearing). `network.py` `load_dotenv()` now loads `.env` from the PROJECT ROOT (absolute path off `__file__`), fixing a CWD-dependent 403.
- **CS-NER ACL source is now exhausted** (93/97 downloaded; the few left are malformed ids). To grow further we fetch ARBITRARY ACL Anthology papers — the fixed extractor's entity list (CS-NER gazetteer ∩ paper text) works for ANY parsed paper, not just CS-NER-annotated ones.
- **Validated the fetch-arbitrary-ACL chain:** picked 10 recent main-conf papers (ACL/EMNLP/NAACL 2019–2022) from `acl_title_index.json` (81k entries, title→id), downloaded via `acl_pipeline.download_pdf`, parsed → **9/10 succeeded** (P19-1211, 2020.acl-main.321, P19-1648, 2020.acl-main.513, 2022.emnlp-main.716, 2021.emnlp-main.298, 2021.emnlp-main.669, 2022.naacl-main.238, 2021.naacl-main.50). Then enrich (`--source csner`) + fixed extraction ran on the new papers. Scaling to ~50/100 is the same flow with a bigger selection.
- **Stragglers:** D19-1395, S16-1185 fail on `timeout after 1200s` (S2 citation rate-limit, not a bug). Retry with `--rerun-failed --timeout 1800` or parse `--no-citations`.

**2. KGE precision problem diagnosed and FIXED — the big win.** First run on the 79-paper corpus was poor (ComplEx MRR 0.101, RotatE 0.053; the diagnostic `gt_ranks` showed **90% of true citation neighbors ranked 100+**, but only 6% null → NOT a coverage problem). Root cause: **representation/vocabulary mismatch** — query (corpus) papers were represented ONLY by their CEO entities while candidate papers carried abstract `concepts`; cosine ranking was near-random. Specifically the `root_concepts` block in `build_unified_graph` was an **unimplemented empty comment**, so seed papers had zero concept content.

Four changes in `kg_transe_pipeline.py`:
- **Seed concepts implemented** — seed paper now gets `paper→mentions→concept` edges from its `root_concepts` (loaded into `paper_meta`). Puts query + candidate papers in the SAME concept vocabulary. *(This was the structural unlock.)*
- **Multi-negative sampling** — new `--neg-ratio` (default **10**); tiles each positive against 10 corruptions (all 3 model losses are element-wise margin, so tiling works unchanged).
- **`--dim` 64 → 128**, **`--epochs` 500 → 1000** (new defaults).

**Result (same 79-paper eval, single run each):**

| Model   | MRR before | MRR after | Hits@1 | Hits@10 |
|---------|-----------|-----------|--------|---------|
| TransE  | 0.052 | 0.254 | 0.165 | 0.430 |
| ComplEx | 0.101 | 0.362 | 0.253 | 0.582 |
| **RotatE** | 0.053 | **0.387** | **0.266** | **0.608** |

→ ~4–7× jump. Hits@10 ≈ 0.61 (real citation in top-10 for ~60% of papers), Hits@1 ≈ 0.27. Rank-100+ dropped 90% → 65%. **This is now a genuinely useful result, not just above-random.**

**2b. MULTI-SEED CONFIRMS it (5 seeds, 79 papers, dim128/1000ep/neg10, `output/kge_multiseed_summary.json`):**

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            |
|---------|---------------|---------------|---------------|----------------|
| TransE  | 0.170 ± 0.019 | 0.316 ± 0.009 | 0.433 ± 0.016 | 0.259 ± 0.010  |
| ComplEx | 0.213 ± 0.052 | 0.499 ± 0.051 | 0.592 ± 0.053 | 0.341 ± 0.044  |
| **RotatE** | **0.246 ± 0.033** | 0.489 ± 0.039 | **0.608 ± 0.022** | **0.364 ± 0.031** |

→ **RotatE is best and settled** — leads MRR/Hits@1/Hits@10, wins 4/5 seeds on MRR, lowest variance (ComplEx only edges Hits@5). **Verdict FLIPPED from the old ComplEx (37-paper, pre-fix MRR 0.157) to RotatE (MRR 0.364 ± 0.031).** RotatE & ComplEx overlap within ~1 std but RotatE wins on more metrics + more seeds + stability. Reportable: "RotatE MRR 0.364 ± 0.031, Hits@10 0.608 ± 0.022 (5 seeds, 79 papers)". The single-run 0.387 sits inside the seed range (max 0.416) → holds up.

**3. K19-1053 parser bug fixed** (see 4b in Session 16): `html_generator.py` `_subsub` seed `{}`→`0` + coerce. Re-parsed clean.

**2c. PRECISION PUSH #1 — empty-concept backfill (IMPLEMENTED 2026-06-19).** ~27% (386) of citation-network candidate nodes had empty `concepts` (S2 returned no abstract) → isolated random-embedding nodes. Fix in `build_unified_graph`: when a node's `concepts` is empty, fall back to `extract_concepts(title)` (imported from `citation.network`). All 386 recover content from their titles (0 still empty); no json files rewritten — done at graph-build time. **Baseline to beat = the Session-17 multiseed RotatE MRR 0.364 ± 0.031.**

**RESULT — #1 is a clear win (5 seeds, 1000ep, post-backfill `output/kge_multiseed_summary.json`):**

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            | (MRR was) |
|---------|---------------|---------------|---------------|----------------|-----------|
| TransE  | 0.187 ± 0.021 | 0.349 ± 0.007 | 0.461 ± 0.014 | 0.287 ± 0.010  | 0.259 |
| ComplEx | 0.263 ± 0.016 | 0.504 ± 0.039 | 0.623 ± 0.033 | 0.386 ± 0.015  | 0.341 |
| **RotatE** | **0.296 ± 0.019** | **0.554 ± 0.049** | **0.678 ± 0.028** | **0.423 ± 0.014** | 0.364 |

→ All three models up; **RotatE still best and now DECISIVE** (MRR 0.423 vs ComplEx 0.386 — gap ≈ 2.5× their stds, no longer overlapping). RotatE MRR +16%, Hits@1 0.246→0.296, Hits@10 0.608→0.678 (real citation in top-10 for ~68% of papers). **Variance also halved** (RotatE ±0.031→±0.014; ComplEx ±0.044→±0.015) — backfilling the random-embedding nodes removed ranking noise. **New reportable headline: RotatE MRR 0.423 ± 0.014, Hits@10 0.678 ± 0.028, Hits@1 0.296 (5 seeds, 79 papers).**

**2d. PERF — vectorized negative sampling (IMPLEMENTED 2026-06-19, REQUIRED fix).** With `--neg-ratio 10` the old per-element Python `_corrupt` loop (with `.item()` calls) made a 5-seed/3-model/1000ep multiseed take **~16 HOURS** (CPU-bound, GPU idle). Rewrote `train_kge`'s negative sampling as pure on-GPU torch ops (`torch.randint`/`torch.rand` + `torch.where`, triples kept on-device) with a seeded `torch.Generator` for reproducibility. **~13× faster: 100 epochs 6.5min → 30s; full multiseed ~16h → ~75min.** `_corrupt` is now dead code (left in place). NOTE: the torch-generator stream differs from the old `random.Random` stream, so seed N won't reproduce pre-vectorization seed-N numbers exactly — still fully reproducible going forward. Loss converges by ~100-200 epochs now, so 1000ep is generous (kept for comparability with the 0.364 baseline).

**2e. PRECISION PUSH #4 — vocabulary normalization (IMPLEMENTED 2026-06-19).** `_norm_entity` was just `strip().lower()`; upgraded to also collapse whitespace, strip surrounding punctuation, and **singularize the head (last) word** via a dependency-free `_singularize` + `_NON_PLURAL` exception set (bias/series/species/physics/analysis…). Applied uniformly to BOTH KG entities and paper concepts, so `Language Models`↔`language model`, `embeddings`↔`embedding`, `datasets`↔`dataset` now link. Entity-side merge is modest (~65 forms, 2%); the intended win is cross-linking entity↔concept singular/plural. **Baseline to beat = post-#1 RotatE MRR 0.423 ± 0.014.**

**RESULT — #4 REGRESSED the best models → REVERTED.** Post-#4: TransE MRR 0.303±0.008 (+0.016), ComplEx 0.351±0.051 (−0.035, variance ↑), **RotatE 0.386±0.039 (−0.037, variance ↑)**. Singularization over-merged distinct entities/concepts, adding spurious links + ranking noise (only TransE, the weakest, nudged up). `_norm_entity` reverted to plain `strip().lower()` (the validated post-#1 state; a comment in-code warns not to re-add singularization without an ablation). Re-ran multiseed to restore the canonical 0.423 summary. **Lesson: aggressive surface-form merging hurts — the entities/concepts carry signal in their exact forms.**

**2f. PRECISION PUSH #2 — richer seed content from full text (IMPLEMENTED 2026-06-19).** Seed/query papers were represented by only ~15 abstract `root_concepts` (often inaccurate — e.g. BERT's root_concepts were actually SBERT's, wrong S2 abstract). Now `load_all_data` also reads each paper's `no-llm/output.json`, concatenates section text, and `extract_concepts(body, top_n=50)` → stored as `fulltext_concepts`; `build_unified_graph` adds the UNION of root+fulltext concepts as seed `mentions` edges. Graph grew 26.3k→30.6k triples. **CONFIRMED by multiseed (5 seeds, 1000ep) — #2 is a BIG win, even bigger than #1:**

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            | (MRR post-#1) |
|---------|---------------|---------------|---------------|----------------|---------------|
| TransE  | 0.309 ± 0.017 | 0.562 ± 0.017 | 0.676 ± 0.033 | 0.426 ± 0.013  | 0.287 |
| ComplEx | 0.284 ± 0.058 | 0.587 ± 0.038 | 0.701 ± 0.060 | 0.422 ± 0.049  | 0.386 |
| **RotatE** | **0.359 ± 0.033** | **0.684 ± 0.043** | **0.780 ± 0.026** | **0.505 ± 0.025** | 0.423 |

→ **RotatE MRR 0.423→0.505 (+19%), Hits@10 0.678→0.780 (real citation in top-10 for 78% of papers), Hits@1 0.296→0.359.** Tight variance (all 5 seeds ∈ [0.47,0.54]). RotatE decisively best (MRR 0.505 vs ComplEx 0.422). NOTE: #4 (vocab normalization) stays REVERTED; this is separate/additive.

**2g. TEMPORAL-FILTERED second metric added (IMPLEMENTED 2026-06-19, user request).** A paper can't cite the future, so added a second eval: rank only candidates with year ≤ query year, GT = neighbors ≤ query year ("papers it cited" / reference-prediction). Reported ALONGSIDE the bidirectional metric (not replacing it) — `evaluate()` adds flat `filt_hits@{1,5,10}`/`filt_mrr` keys + `filt_best_rank` per paper; `kge_multiseed.py` METRICS extended so both aggregate. Years come from `citation_network.json` (`year` field, stored in paper_meta + node_year map). NOTE: the bidirectional GT is ~50% newer "citing" papers, so the filtered task is SMALLER (n≈56 vs 79 — 23 papers have no in-corpus older refs) and NOT 1:1 comparable. Preliminary single run (rotate 200ep): bidirectional MRR 0.468 / filtered MRR 0.517.

**2h. NEW TOOL `fetch_more_papers.py` (scale the corpus).** Selects recent main-conf ACL/EMNLP/NAACL long papers (old-style P/D/N 18-19 + new-style 2018-2022 *-main) from `acl_title_index.json` (6,864 candidates not yet downloaded) excluding what we have, downloads PDFs to `output/acl/pdfs/`. `--count N` (default 50), reproducible `--seed`. Full scale chain (run in screen): `python3 fetch_more_papers.py --count 50 && python3 run_acl_batch.py --timeout 1200 && python3 enrich_entity_csv.py --all --source csner && python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B --skip-existing`. ⚠ Last step is GPU (Qwen 10GB) — do NOT run concurrently with a KGE multiseed/other GPU job on the 20GB vGPU.

**2i. SCALE-UP to 155 parsed (102→155, +53 main-conf papers) + enrich discovery BUG FIXED.** `fetch_more_papers.py --count 50` + parse ran fine (155 parsed). BUT `enrich_entity_csv.discover_paper_ids` only iterated `output/acl/<id>/` folders, and fetched papers are PDF-only (no per-paper acl folder) → enrich silently skipped them → no entity CSV → fixed extraction skipped them (fixed-triples stuck at 94 despite 155 parsed). **Fixed:** `discover_paper_ids` now also includes ANY parsed paper (`output/<id>/no-llm/output.html`) for csner/iter sources (write path already mkdir'd output/acl/<id>/). Verified 93→155 discovered. **TODO: still need to run `enrich --all --source csner` + `kg_main --all --extractor fixed --skip-existing` to actually create triples for the ~62 new papers, then re-run KGE on the bigger corpus.**

**2j. SCALED CORPUS KGE — 155 papers, 5 seeds (2026-06-20, `output/kge_multiseed_summary.json`).** Fixed extraction finished (155 papers w/ triples). Both metrics:

| Model   | Bidir MRR     | Bidir H@10 | Filt MRR      | Filt H@10 |
|---------|---------------|------------|---------------|-----------|
| TransE  | 0.315 ± 0.008 | 0.567      | 0.334 ± 0.006 | 0.583 |
| **ComplEx** | **0.432 ± 0.018** | **0.742** | **0.456 ± 0.027** | 0.760 |
| RotatE  | 0.403 ± 0.021 | 0.708      | 0.423 ± 0.025 | 0.760 |

→ **Best model FLIPPED back to ComplEx at 155 papers** (was RotatE at 79). ComplEx & RotatE are close throughout (~0.03 apart); ComplEx scales better (0.386→0.432 from 79→155 while RotatE dropped 0.505→0.403). **Absolute scores dropped vs the 79-paper run — EXPECTED (2× corpus = ~2× candidate pool = harder ranking); report lift-over-random.** Temporal filter gives a small consistent lift (+~0.02 MRR, filt H@10 0.76) — confirms the "papers it cited" task is slightly cleaner. **Headline: ComplEx MRR 0.432 (bidir) / 0.456 (temporal-filtered), Hits@10 ~0.74-0.76, 155 papers, 5 seeds.**

**2k. SECOND ONTOLOGY (scinex) added for fixed extraction — both kept, for comparison (2026-06-21).** New refined ontology `scinex_refined_14.owl` (Turtle, "Core Experiment Ontology / scinex" v0.7.0, 47 classes, 27 object properties). Added as a SELECTABLE ontology without touching the hardcoded CEO path:
- NEW `kg_extraction/ontology_loader.py` — `load_ontology(owl) -> (relations, schema)` via rdflib (installed). Builds schema hints `Domain → Range | <first sentence of rdfs:comment>`, expands `owl:unionOf` domains.
- `fixed_extractor.py`: `_make_fixed_system_prompt(relations, schema=None)` + `FixedTripleExtractor(..., schema=None)` — default None = hardcoded `_CEO_SCHEMA` (CEO unchanged); scinex passes its own.
- `kg_main.py`: `--ontology {ceo,scinex}` (default ceo) + `--ontology-file` (default scinex_refined_14.owl). scinex routes to a SEPARATE output dir **`kg/fixed_scinex/<model>/`** (CEO stays `kg/fixed/`), so both coexist. `ENTITY_EXTRACTORS` + model-slug path updated to include `fixed_scinex`.
- KGE: `--extractor fixed_scinex` added to `kg_transe_pipeline.py` + `kge_multiseed.py` (path is generic: `kg/<extractor>/<model>/triples.json`).
- scinex relations = CEO + 5 new (achievesResult, extractedFrom, mentions, producedBy, writes); domain/range refined. CEO-specific validation guards in the parser still fire on shared relation names (fine for comparison); new relations get generic validation. Verified: loads 27 rels, builds scinex prompt, CEO default untouched.
- **TODO (user): run scinex fixed extraction → `kg_main.py --all --extractor fixed --ontology scinex --model Qwen/Qwen3-14B --skip-existing` (GPU, ~hours, same entity CSVs); then KGE `--extractor fixed_scinex` (save to a DIFFERENT summary file) and compare vs CEO `--extractor fixed`.**

**2l. ✅ RESOLVED in 2m — scinex extraction was TRUNCATED by --max-new-tokens 512, 80 papers re-extracted (2026-06-22; verified fixed 2026-06-23).** scinex fixed extraction finished (155 papers, 152 w/ triples, 4922 total). BUT the run used a MIX of token caps: papers #1-75 at 4096 (default), #76-155 at 512 (a mid-run speedup that backfired). Truncation check vs CEO at the #75 boundary: **4096 batch ratio scinex/CEO = 1.03 (clean, scinex≈CEO); 512 batch ratio = 0.40 (60% of triples LOST).** So the comparison is invalid until the 80 truncated papers are redone. The 3 zero-triple papers (L16-1593, W14-5502 zero in CEO too; N19-5002 CEO=4) are a side-effect / benign. **LESSON: do NOT use --max-new-tokens 512 for fixed extraction — triple-rich paragraphs need more; the Claude.md "use 512 for snappy runs" note is about SPEED not completeness. Use the 4096 default (or ≥2048) for real extraction.** **FIX (TODO): delete the latest-80-by-mtime fixed_scinex triples, re-run `kg_main.py --all --extractor fixed --ontology scinex --model Qwen/Qwen3-14B --skip-existing` (no --max-new-tokens → 4096), then KGE compare. Encouraging: at equal 4096 cap scinex yields ≈ CEO triple count (1.03×).**

**2m. ✅ scinex truncation FIXED + VERIFIED — re-extraction is complete and clean (2026-06-23).** Verified the 2l TODO was carried out. State on disk now:
- **155/155 parsed papers have `kg/fixed_scinex/Qwen3-14B/triples.json`** — full coverage, none missing.
- **scinex total = 6,776 triples vs CEO `fixed` = 6,574 → ratio 1.031** (was 4922 / ratio 0.40 on the bad 512 batch). This is the clean scinex≈CEO ratio 2l predicted at the 4096 cap → **the CEO-vs-scinex comparison is now VALID.**
- **mtime distribution confirms the re-run:** 75 files dated 06-21 (original clean #1-75 @4096) + 80 files dated 06-22/06-23 (= exactly the 80 truncated #76-155, deleted and re-extracted @4096). 75+80=155.
- **Zero-triple papers = 2, both benign:** L16-1593 and W14-5502 — **both also 0 in CEO `fixed`**, so it's the paper content, not a scinex/truncation artifact. (N19-5002, flagged in 2l at CEO=4, is now scinex=3 — no longer a concern.)
- **4 papers run scinex < CEO (ratio<0.6): D19-1408 (37 vs 79), D19-1528 (23 vs 47), E17-1082 (14 vs 30), N18-1013 (49 vs 82). Checked — these are NOT truncated:** all valid JSON (cleanly closed), and triples span from "1 Introduction" through late sections (5.3.3 / Conclusion / 6.1 / 3.4). The first 3 are from the 06-21 clean batch; N18-1013 was re-extracted 06-22. The lower count is a **genuine ontology difference** (scinex's refined domain/range + validation produce fewer triples per section than CEO), not a token cutoff. No action needed.
- **CONCLUSION: scinex extraction is done across the whole corpus and ready for KGE comparison (`--extractor fixed_scinex`, save to a separate summary file) vs CEO `--extractor fixed`.**

**2n. scinex KGE multiseed — SET UP, user runs it (2026-06-23).** Everything is staged so the scinex eval can run without disturbing the CEO baseline:
- **CEO baseline PRESERVED.** Canonical CEO multiseed = `output/kge_multiseed_summary.json` (155 papers, seeds 1-5, 1000ep) + per-seed JSONs in `output/kge_seedruns/`. A labeled safety copy was made at **`output/kge_multiseed_summary_ceo.json`** (identical). DO NOT overwrite these.
- **scinex run writes to SEPARATE paths** (`--save-summary output/kge_multiseed_summary_scinex.json`, `--run-dir output/kge_seedruns_scinex`). The default save path is the CEO file, so these two flags are mandatory or the CEO result is clobbered.
- **Exact command (matches CEO params for a fair comparison — seeds 1-5, 1000ep, kge all, dim128 default):**
  ```bash
  python3 kge_multiseed.py --extractor fixed_scinex --model Qwen3-14B --kge all \
    --seeds 1 2 3 4 5 --epochs 1000 --device cuda \
    --run-dir output/kge_seedruns_scinex \
    --save-summary output/kge_multiseed_summary_scinex.json
  ```
  GPU job (~10GB, ~75 min). One GPU job at a time. A 1000ep/dim128 smoke confirmed it loads cleanly (6776 triples / 153 papers, 2205 held-out citation edges, 25 relations incl. scinex's achievesresult/mentions/extractedfrom/producedby, training on cuda) before being stopped for the user to run.
- **AFTER it finishes:** compare `kge_multiseed_summary_scinex.json` (scinex) vs `kge_multiseed_summary.json` (CEO). CEO baseline = ComplEx bidir MRR 0.432 / filt MRR 0.456 (full CEO table §2o). **DONE — results in §2p below.**

**2p. ✅ scinex KGE multiseed COMPLETE + scinex-vs-CEO comparison (2026-06-23, `output/kge_multiseed_summary_scinex.json`; 155 papers, seeds 1-5, 1000ep, dim128, neg10 — identical params to CEO).**

scinex per-model results (mean ± std):

| Model | Bidir MRR | Bidir H@1 | Bidir H@5 | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|---|---|
| TransE  | 0.325 ± 0.005 | 0.202 ± 0.013 | 0.441 ± 0.012 | 0.565 ± 0.016 | 0.330 ± 0.006 | 0.587 ± 0.014 |
| **ComplEx** | **0.441 ± 0.028** | **0.297 ± 0.043** | **0.618 ± 0.016** | **0.744 ± 0.015** | **0.472 ± 0.041** | **0.774 ± 0.022** |
| RotatE  | 0.391 ± 0.033 | 0.253 ± 0.039 | 0.552 ± 0.041 | 0.703 ± 0.027 | 0.407 ± 0.019 | 0.727 ± 0.051 |

**scinex − CEO deltas (bidir MRR):** TransE +0.010, ComplEx +0.009, RotatE −0.011. **ComplEx Δ across all metrics: +0.009 MRR, +0.012 H@1, +0.020 H@5, +0.002 H@10, +0.016 filt-MRR, +0.015 filt-H@10.**

**2p-bis. NO-FILTER vs YEAR-FILTERED comparison (both MULTISEED, 5 seeds, 155 papers, 1000ep) — the temporal filter helps everywhere.** Each seed's run computes BOTH metrics on the same trained model; table below is the 5-seed mean (full per-seed values in the summary JSONs):

| Ontology | Model | MRR no-filter | MRR filtered | ΔMRR | H@10 no-filter | H@10 filtered | ΔH@10 |
|---|---|---|---|---|---|---|---|
| CEO | TransE | 0.315 | 0.334 | +0.019 | 0.567 | 0.583 | +0.017 |
| CEO | **ComplEx** | 0.432 | **0.456** | +0.024 | 0.742 | **0.760** | +0.017 |
| CEO | RotatE | 0.403 | 0.423 | +0.020 | 0.708 | 0.760 | +0.052 |
| scinex | TransE | 0.325 | 0.330 | +0.005 | 0.565 | 0.587 | +0.022 |
| scinex | **ComplEx** | 0.441 | **0.472** | +0.031 | 0.744 | **0.774** | +0.030 |
| scinex | RotatE | 0.391 | 0.407 | +0.016 | 0.703 | 0.727 | +0.024 |

→ **Year filter improves every model × both ontologies: +0.02-0.03 MRR, up to +0.05 H@10.** Best cell overall = **scinex ComplEx filtered: MRR 0.472, H@10 0.774.** ⚠ **NOT a strictly 1:1 lift** — the filtered task runs on a SMALLER/different test set (drops papers with no in-corpus older refs, n≈56 vs ~79), so the gain is partly cleaner-task + partly easier-subset. Report BOTH; lead with filtered as the "reference-prediction (temporal)" task (more correct setup) but disclose the reduced test set. Both metrics are 5-seed multiseed (NOT single-run).

**VERDICT: the two ontologies are statistically INDISTINGUISHABLE on citation prediction.** Every delta is smaller than the seed-to-seed std (ComplEx +0.009 MRR vs ±0.028 std → error bars fully overlap). ComplEx is best under BOTH ontologies. scinex shows a small CONSISTENT edge on ComplEx (best on all 4 ComplEx metrics, biggest = filt-MRR +0.016) but it's within noise; RotatE is the one model slightly worse under scinex. **Defensible paper claim: the refined scinex ontology (47 classes / 27 object properties, +5 relations vs CEO) MATCHES CEO's downstream citation-prediction quality and does not degrade it — with a small non-significant improvement on the best model. NOT "scinex wins".** At near-equal triple count (scinex 6776 vs CEO 6574, 1.03×) the ontology choice is roughly neutral for this metric.

**2q. ⭐ BIG WIN — self-adversarial loss (Sun et al. 2019) on RotatE: MRR 0.40→0.57 (2026-06-23).** Implemented self-adversarial negative sampling + logsigmoid loss as an OPT-IN training objective; the historical margin-ranking loss stays the default so all prior baselines reproduce.
- **Code:** `kg_transe_pipeline.py` — new flags `--loss {margin,adv}` (default margin), `--gamma` (γ offset, default 9.0), `--adv-temp` (α, default 1.0). The adv branch in `train_kge` keeps negatives grouped per-positive `[B,K]`, softmax-weights hard negatives by α (detached), uses `−logσ(γ+s_pos) − Σ wᵢ logσ(−(γ+s_negᵢ))` where `s=forward()` (higher=better). Margin path is byte-identical to before. `kge_multiseed.py` also got `--loss/--gamma/--adv-temp` and passes them through (REQUIRED — without this the multiseed silently runs margin).
- **Tuning (single seed=1, 600ep, CEO/fixed):** RotatE needs a LARGE γ (distance-model scale): **γ=24 → bidir MRR 0.548**; γ=9 broke it (0.05). ComplEx adv was WORSE than its own margin (best 0.357 @ γ12 vs 0.432 margin) and *declines with training* (cosine-sim eval ↔ adv objective mismatch — its 20-epoch 0.485 was an underfit blip). TransE adv broke (0.012). **So self-adversarial is RotatE-only here** — it's RotatE's native objective; margin remains best for ComplEx/TransE.
- **CONFIRMED multiseed (5 seeds, 1000ep, γ=24, α=1.0, dim128; `output/kge_multiseed_summary_adv_{ceo,scinex}.json`, per-seed in `output/kge_seedruns_adv_{ceo,scinex}/`):**

| Config | Bidir MRR | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|
| CEO RotatE **margin** (old) | 0.403 ± 0.021 | 0.708 | 0.423 ± 0.025 | 0.760 |
| **CEO RotatE adv** | **0.566 ± 0.026** | **0.847** | **0.555 ± 0.029** | **0.855** |
| scinex RotatE **margin** (old) | 0.391 ± 0.033 | 0.703 | 0.407 ± 0.019 | 0.727 |
| **scinex RotatE adv** | **0.568 ± 0.022** | **0.838** | **0.575 ± 0.018** | **0.844** |

→ **RotatE adv vs margin: CEO +0.164 bidir / +0.133 filt; scinex +0.177 / +0.168.** Tight std (±0.02-0.03 over 5 seeds → not seed-luck). **NEW HEADLINE BEST = RotatE + self-adversarial: MRR ~0.57, Hits@10 ~0.84-0.85** (vs the old ComplEx-margin best 0.432/0.456, H@10 0.74). Legitimate method improvement (the model's native training objective), NOT eval tuning. **Ontology comparison still ≈ TIE at the higher level** (scinex filt MRR 0.575 vs CEO 0.555 — within std; scinex edges ahead but not significant). Run cmd (both ontologies, ~50-60 min total): `kge_multiseed.py --kge rotate --loss adv --gamma 24 --adv-temp 1.0 --seeds 1 2 3 4 5 --epochs 1000` with `--extractor {fixed,fixed_scinex}` + separate `--run-dir`/`--save-summary`. **TODO maybe: γ tuning was coarse (18/24/30 tried, 24 best single-seed); a finer sweep or ComplEx-with-adv-and-score-based-eval could squeeze more, but 0.57 is already a strong, defensible headline.**

**2t. CITATION-EDGE DIRECTION FIX — `filt_*` is now TRUE reference prediction, not a year proxy (2026-06-28).** Problem (raised by user/professor): the old temporal metric used `year ≤ query year` as a PROXY for citation direction, which is inconsistent — the same edge "4 cites 1" was counted as a link for query=1 (where it's a newer *citer*, dropped by the year filter) AND query=4 (where it's an older *reference*). Year is also a bad proxy (same-year cites, preprint-vs-pub mismatches).
- **Fix in `evaluate()` (`kg_transe_pipeline.py`):** `filt_*` now uses the ACTUAL edge direction. Citation-network edges are `{source, target, type:'cites'}` = source cites target; `relation` field confirms `cited_by` = reference (older), `citing` = citer (newer). GT for query P = papers P actually cites = edges where `source == P.s2_id` → targets. Ranked against the FULL candidate pool (no year filtering — references are naturally older, future papers are just distractors). This is the consistent directional "what does this paper cite" task.
- **Effect (text baseline, title+abstract, test split):** filt MRR 0.740→**0.524**, filt H@10 0.955→**0.850** — lower but HONEST (old was inflated by the shrunk ≤year candidate pool + direction-agnostic GT). Bidirectional `mrr`/`hits@*` UNCHANGED (still undirected "relatedness": any citation neighbor).
- **⚠ Supersedes all prior `filt_*` numbers** (§2o–2s year-based filt are stale; bidir there still valid). Re-run KGE on the final corpus to get fresh directional numbers. `kge_multiseed.py` needs no change (same keys). `node_year` map in evaluate() is now unused by filt (left in place, harmless).
- **Two metrics now, both kept:** `mrr`/`hits@*` = bidirectional relatedness (undirected); `filt_*` = directional reference prediction (edge-direction, the defensible "papers it cites" task).

**2s. ⚠ TEXT-SIMILARITY BASELINES added — and they look STRONGER than KGE (2026-06-28).** User request: baselines that embed paper TEXT and predict citations by cosine similarity, to test whether KG embeddings actually beat naive text matching. Built `text_baseline.py` (CPU-only, TF-IDF via scikit-learn — sentence-transformers NOT installed and avoided to not disturb the pinned torch). Two baselines: **title** and **title+abstract**. Title/abstract pulled from `citation_network.json` (query = root title/abstract keyed by FOLDER id — NOT root `paper_id`, which is an S2 hash; candidates = `nodes[s2]` title/abstract; output.json title fallback). Reuses `kg_transe_pipeline.evaluate(predict_fn=...)` (new `predict_fn` hook) so candidate pool / ground-truth / val-test split / metrics are IDENTICAL to the KGE eval → directly comparable.
- **PRELIMINARY result (test split, current IN-FLUX 315-paper corpus, `output/text_baseline_test.json`):** title TF-IDF MRR 0.606 / H@10 0.829; **title+abstract TF-IDF MRR 0.703 / H@10 0.904 / filt MRR 0.740 / filt H@10 0.955.**
- **⚠ This BEATS the KGE headline** (RotatE+adv ~0.60 MRR / 0.85 H@10) — and on a LARGER, harder corpus, so the gap is likely real. **Implication for the paper: a simple lexical baseline over abstracts outperforms the KG embedding → cannot claim "KGE is best" naively. Reframe needed** (e.g. "KG structure alone, without abstract text, approaches a strong content baseline"; or combine KG+text; or position KGE as complementary). NOT fatal, but strategic.
- **NOT yet apples-to-apples:** text ran on 315-corpus (146 test); KGE §2r was 155-corpus (66 test). **DEFINITIVE comparison = run BOTH on the FINAL corpus after extraction finishes + KGE re-run** (`--split-seed 42` guarantees identical test papers). Bug fixed during build: corpus papers were initially keyed by S2-hash `paper_id` → empty query text → near-random; now keyed by folder id (100% text coverage).
- Run: `python3 text_baseline.py --extractor fixed --model Qwen3-14B --text-source both --eval-split test --save-results output/text_baseline_<corpus>.json`

**2r. ⭐⭐ DEFENSIBLE HEADLINE — validation/test split + RotatE adv on TEST (2026-06-24).** Closed the "hyperparameters tuned on the test set" rigor gap. Implemented a query-paper val/test split (`--eval-split {all,val,test}`, `--val-frac 0.5`, `--split-seed 42` — split fixed by split_seed ONLY, independent of training --seed, so val/test membership is identical across configs/seeds; default `all` = old behaviour). `evaluate()` in `kg_transe_pipeline.py` partitions `papers_to_eval`; `kge_multiseed.py` passes the flags through. 132 eval papers → 66 val + 66 test (disjoint).
- **γ selected on VALIDATION only** (RotatE adv, seed1, 600ep, `output/val_tune/`): γ16→0.522, γ20→0.580, γ24→0.567, **γ28→0.628 (peak)**, γ32→0.619. Curve turns over → **val-selected γ = 28** (the earlier γ=24 was picked on the full set; 28 is the clean choice).
- **FINAL multiseed on the disjoint TEST split (66 papers, RotatE, 5 seeds, 1000ep, γ=28; `output/kge_multiseed_summary_test_{ceo,scinex}_{margin,adv}.json`):**

| Config | Bidir MRR | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|
| CEO RotatE margin | 0.406 ± 0.053 | 0.727 | 0.418 ± 0.036 | 0.759 |
| **CEO RotatE adv** | **0.598 ± 0.041** | **0.861** | **0.594 ± 0.027** | 0.844 |
| scinex RotatE margin | 0.391 ± 0.043 | 0.712 | 0.417 ± 0.027 | 0.737 |
| **scinex RotatE adv** | **0.599 ± 0.023** | 0.845 | **0.606 ± 0.054** | 0.844 |

→ **adv vs margin on the SAME test split: CEO +0.19 bidir / +0.18 filt; scinex +0.21 / +0.19.** The self-adversarial win SURVIVES clean model selection (actually a touch higher, ~0.60 vs the §2q full-set 0.57, since γ=28>24 + this test half runs slightly high). **DEFENSIBLE PAPER HEADLINE: RotatE + self-adversarial, MRR ~0.60, Hits@10 ~0.85, 5 seeds, with γ selected on a disjoint validation set (no test-set tuning).** Ontology comparison STILL a tie (CEO filt 0.594 vs scinex 0.606, within std). NOTE: test-split numbers are on 66 papers (half the corpus) — that's the cost of an honest held-out split; the §2q all-papers numbers (155, MRR 0.57) remain valid as the "evaluated on all papers" figure, but §2r is the one to report as the primary result because its hyperparameters weren't chosen on the eval data.

**2o. CEO BASELINE — full multiseed numbers for the writeup (155 papers, 5 seeds, 1000ep, dim128, neg10; `output/kge_multiseed_summary.json`).** Exact mean ± std, both metrics (bidirectional = predict any citation neighbor; temporal-filtered `filt_*` = predict only older papers it could have cited):

| Model | Bidir MRR | Bidir H@1 | Bidir H@5 | Bidir H@10 | Filt MRR | Filt H@10 |
|---|---|---|---|---|---|---|
| TransE  | 0.315 ± 0.008 | 0.197 ± 0.011 | 0.429 ± 0.007 | 0.567 ± 0.021 | 0.334 ± 0.006 | 0.583 ± 0.018 |
| **ComplEx** | **0.432 ± 0.018** | **0.285 ± 0.029** | **0.598 ± 0.028** | **0.742 ± 0.029** | **0.456 ± 0.027** | **0.760 ± 0.028** |
| RotatE  | 0.403 ± 0.021 | 0.259 ± 0.033 | 0.573 ± 0.017 | 0.708 ± 0.011 | 0.423 ± 0.025 | 0.760 ± 0.025 |

→ **ComplEx is best for CEO at 155 papers.** Random baseline ≈ 0.2% Hits@1 over ~500 candidate papers → report lift-over-random, not the absolute number (1.0 is not the target for unsupervised citation prediction). **scinex equivalent table = §2p (done; scinex ≈ CEO, indistinguishable).**

**Open items next session:**
00. **`results.md` (project root) = consolidated PAPER-READY results** (final tables, methodology, findings, caveats, reproduction, scaling decision). Generated 2026-06-24. Hand THIS to the paper/abstract writer; keep it in sync if numbers change.
0. **DONE: scinex extracted (§2m), KGE run (§2p) — scinex ≈ CEO on citation prediction, statistically indistinguishable.** The CEO-vs-scinex ontology comparison is complete.
0b. **⭐⭐ DEFENSIBLE HEADLINE (§2r): RotatE + self-adversarial, γ selected on a held-out VALIDATION set, reported on disjoint TEST = MRR ~0.60 / Hits@10 ~0.85 (5 seeds, both ontologies).** This is the number to put in the paper (clean model selection, no test-set tuning). Self-adv helps RotatE only. **The method/experiments are DONE — next is write-up.**
1. **CURRENT BEST = RotatE self-adversarial (γ=28), TEST split: CEO bidir 0.598 / filt 0.594, scinex bidir 0.599 / filt 0.606, Hits@10 ~0.85 (§2r, `output/kge_multiseed_summary_test_*`).** All-papers version (§2q, γ=24, no val/test split): CEO/scinex MRR ~0.57. Margin reference: ComplEx 0.432 (§2o). Ontology = tie throughout.
2. **Precision arc: ~0.05 → 0.36 (seed concepts+tuning) → 0.423 (#1 empty-concept backfill) → 0.505 (#2 full-text seed content).** Wins: #1, #2. Negative/reverted: #4 vocab normalization. Untried: self-adversarial negatives (RotatE technique, principled but loss-rewrite); tighter eval pool (risks gaming — skip).
3. **Recommended next: scale papers OR write up.** RotatE MRR 0.51 / Hits@10 0.78 is strong & defensible. If scaling: fetch more ACL via `acl_title_index.json` (validated chain) → parse → enrich → fixed → re-run KGE; expect absolute scores to dip with a bigger pool, so report lift-over-random.

---

## Session 16 — Multi-seed KGE settles ComplEx as best + diagnostic predictions + S2 key wired (2026-06-16)

**1. Multi-seed KGE result is the new source of truth (`output/kge_multiseed_summary.json`, seeds 1–5, 500 epochs, fixed extractor, 37 papers).** This resolves the Session-15 single-run ambiguity (one run said ComplEx 0.184, another RotatE 0.178 — just variance over ~37 held-out edges). Mean ± std:

| Model   | Hits@1        | Hits@5        | Hits@10       | MRR            |
|---------|---------------|---------------|---------------|----------------|
| TransE  | 0.054 ± 0.000 | 0.065 ± 0.015 | 0.103 ± 0.012 | 0.082 ± 0.002  |
| **ComplEx** | **0.070 ± 0.024** | **0.232 ± 0.065** | **0.330 ± 0.073** | **0.157 ± 0.014** |
| RotatE  | 0.070 ± 0.031 | 0.162 ± 0.033 | 0.287 ± 0.070 | 0.137 ± 0.019  |

→ **ComplEx is best, now with confidence** — wins MRR/Hits@5/Hits@10, beats RotatE on 4 of 5 seeds, tied at Hits@1. TransE consistently weakest. Results are ~16–40× the random baseline (~0.2% Hits@1 over ~500 candidate papers): a solid *proof-of-signal*, not a SOTA retrieval result, and that's the honest framing for any writeup (report the lift-over-random, not the absolute number — 1.0 is not the target for unsupervised citation prediction).

**2. `kg_transe_pipeline.py` `evaluate()` now saves DIAGNOSTIC predictions (per paper).** Old output stored 5 raw S2 hashes, undiagnosable. New per-paper fields: `title`; `top10_predictions` (rank, paper_id, **title**, score, **`is_true_citation`** flag); `gt_ranks` (every real citation neighbor + **where it actually ranked**, or `null` if not a graph node); `total_candidates`. Added a `label()` title lookup (maps ACL ids via paper_meta, raw S2 hashes via aggregated citation-network node titles). **Only affects future runs** — the on-disk `kge_fixed_results_*.json` were written by old code. Purpose: diagnose WHY scores are low — true neighbors at rank 11–30 = near-misses (window too small), `null` = coverage gap (paper not a node), rank 100+ = genuine miss.

**3. Semantic Scholar API key is now wired (clears the 403/429 that blocked parsing).**
- Key stored in **`/home/ubuntu/project_clean_9/.env`** as `SEMANTIC_SCHOLAR_API_KEY=...`. `network.py` auto-loads it via `load_dotenv()` and sends it as the `x-api-key` header.
- **Installed `python-dotenv`** (`python3 -m pip install python-dotenv`) — it was MISSING, so `load_dotenv()` was silently no-opping and `.env` was never read. Without this package the `.env` does nothing. Verified: `import citation.network` prints `Semantic Scholar API key loaded ✔`; live Graph API call returns HTTP 200.
- S2 endpoints the parse uses (all Graph API v1, covered by a free key): `/paper/ACL:{id}`, `/paper/ARXIV:{id}`, `/paper/search`, `/paper/{id}`, `/paper/{id}/citations`, `/paper/{id}/references`. ~5–6 requests/paper, <600 total for the corpus.
- The prior 403 was unauthenticated S2 refusing requests (not the PDFs — those are local in `output/acl/pdfs/`). `citation/apikey.env` is a dead placeholder; only root `.env` is auto-loaded.
- **Hardened `network.py` `load_dotenv()` to load `.env` from the PROJECT ROOT (parent of `citation/`), not the CWD.** A 403 reappeared on a batch run because `load_dotenv()` searched the current dir; now it uses an absolute path off `__file__`, so launching the parse from anywhere still picks up the key. Verified the key loads even when imported from `/tmp`. **Transient 429 backoffs still occur even WITH the key** (per-key pagination limits) — the retry loop absorbs them; D12-1051 parsed fine through them (10 nodes/10 edges). Don't mistake a 429-with-retry for the old 403 failure.

**4b. Parser bug fixed — `html_generator.py` subsubsection counter (K19-1053).** `IDGenerator._subsub` is a flat int counter, but `next_section` seeded it as `{}` (dict) while `next_subsection`/`next_subsubsection` treat it as int → a subsubsection directly under a section did `dict += 1` → `TypeError: unsupported operand type(s) for +=: 'dict' and 'int'`, killing the whole parse. Fixed: seed `self._subsub[sid] = 0` (not `{}`) + `next_subsubsection` now coerces any non-int to 0 before incrementing. K19-1053 re-parsed clean (20 sections). Same class as the Session-13 structural-edge-case fixes. Since `run_acl_batch` spawns a fresh `main.py` per paper, the running batch picks this up automatically for all later papers.

**Open items / the actual next step (parse the 40 remaining papers, then re-run):**
1. **Parse:** `python3 run_acl_batch.py --timeout 1200` — key is wired, auto-skips the 54 done papers, processes the 40 unparsed (7 prior S2 failures + 33 untouched). May still see occasional 429 backoffs (per-key pagination) — the 1200s timeout absorbs them.
2. **Enrich:** `python3 enrich_entity_csv.py --all --source csner` (cached gazetteer, only new papers cost anything).
3. **Fixed extract:** `python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B --skip-existing` (skips the 54 with triples; one LLM extractor at a time on the 20GB vGPU).
4. **Re-run KGE** (now writes the diagnostic per-paper output): `python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge all --epochs 500 --save-results output/kge_fixed_results.json`. Then inspect `gt_ranks` across papers to see if low scores are near-misses vs coverage gaps. For a stable verdict, also re-run the multi-seed sweep.

---

## Session 15 — Corpus expansion + CS-NER entity lists live + new KGE result (2026-06-16)

**Corpus growth:** parsed corpus **34 → 54 papers** (of 93 ACL PDFs) via `run_acl_batch.py`. 55 papers now have fixed triples.
- **`run_acl_batch.py` hardened:** catches `subprocess.TimeoutExpired` per paper (one slow paper no longer aborts the batch), `--timeout` flag (default 900s), and writes `batch_run_log.json` after EVERY paper (resumable).
- **7 papers failed — ALL rate-limit timeouts, NOT parse failures:** `D12-1051, K19-1053, L18-1051, N18-1166, N19-1179, P13-2003, P19-1416`. Cause: Semantic Scholar citation fetch (unauthenticated ~100 req/5min → 429 backoffs exceed the timeout). Clear them with an S2 API key (`export SEMANTIC_SCHOLAR_API_KEY=...`) then `run_acl_batch.py --rerun-failed`, or parse `--no-citations`.

**CS-NER entity lists are now LIVE (the Session-14 migration is done):**
- Ran `enrich_entity_csv.py --all --source csner` (06-16 00:49) → all `Entity_<id>_enriched.csv` rebuilt from the CS-NER gazetteer (BERT→154, conll→102, D17-1028→69 entities — csner-scale, not the old ~16 llm-scale).
- Fixed extraction was re-run AFTER that → **current fixed triples reflect the CS-NER methodology** (verified by mtimes: enrich 00:49 < fixed triples 01:47–09:32 < KGE 11:42).

**New KGE result (06-16 11:42) — 37 papers (up from 22), KG 5,660 entities / 13,779 triples, reflects CS-NER entity lists:**

| Model   | Hits@1 | Hits@5 | Hits@10 | MRR   |
|---------|--------|--------|---------|-------|
| TransE  | 0.054  | 0.054  | 0.108   | 0.081 |
| **ComplEx** | **0.108** | **0.243** | **0.378** | **0.184** |
| RotatE  | 0.027  | 0.189  | 0.378   | 0.112 |

→ **ComplEx now best (MRR 0.184, Hits@1 0.108)** — a shift from the 22-paper baseline where RotatE led (MRR ~0.13). Bigger corpus + CS-NER lists improved MRR ~0.13 → 0.184 and Hits@10 0.318 → 0.378.
> ⚠ This run was printed but NOT saved (no `--save-results`) — recovered from the screen buffer. Re-run to persist: `python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge all --save-results output/kge_fixed_results.json`.

**Open items for next session:**
1. Re-run KGE **with `--save-results`** to persist the table (the 0.184 ComplEx result is only in this doc + a screen buffer).
2. Finish parsing the remaining ~39 PDFs (the 7 failed + ~32 untouched) — needs an **S2 API key** to beat rate-limiting.
3. After more papers are parsed: rebuild their csner entity lists, re-run fixed (+pair), re-run KGE.
4. (Optional) `pair` extractor could also be re-run on the csner lists if a pair eval is ever wanted (currently pair is NOT KGE-evaluated).

---

## Session 14 — CORRECTION: entity list must come from CS-NER per paper, not open extraction (2026-06-15)

**This corrects a wrong assumption baked into Sessions 6-8.** The fixed extractor's subject pool was being built by `enrich_entity_csv.py --source llm`, which harvests entities from our OWN open (`llm`) extraction's `triples.json`. **That is the wrong source** — using one LLM extraction to seed the "constrained" fixed extractor is circular. Confirmed in code: [enrich_entity_csv.py:180](enrich_entity_csv.py#L180) reads `output/<id>/kg/llm/<model>/triples.json`; in practice the open extraction provided the *bulk* of each enriched CSV (e.g. D17-1028 = ~2 title seeds + ~14 open-extraction entities; J13-4001 = 1 seed + 4).

**Intended design (per user):** for a given paper, look up its entity list **from the CS-NER dataset by title/id**, and use THAT as the fixed extractor's subject pool. CS-NER repo: `github.com/jd-coderepos/contributions-ner-cs`.

**Key finding — the repo has richer data than we were using:**
- `acl/` (`train/dev/test.data`) = **title-level only** (~2-3 entities/paper). This is the only ACL-id-keyed slice and all `acl_pipeline.py` currently pulls. Verified: train.data REC 1 = `"PUT at SemEval-2016 Task 4: The ABC of Twitter Sentiment Analysis"` — literally the title of S16-1018, with 3 title entities. So the old "CS-NER = titles only, 2-3 entities" note was correct *but only about the `acl/` folder*.
- `ftd/`, `ncg/`, `pwc/`, `scierc/`, `full dataset/` = **`*-abs.data` = abstract-level annotations** (full abstracts; `full dataset/train-abs.data` has 5,957 records, multi-sentence, many entities each). **This is the richer entity source we want.**

**Coverage investigation → CS-NER abstract data RULED OUT for our papers (2026-06-16):**
- Verified the data flow in code: PDFs come from `aclanthology.org/{id}.pdf` ([acl_pipeline.py:45](acl_pipeline.py#L45)); entities come from `parse_iob()` of the CS-NER **`acl/` title files** matched title→ACL-id → `Entity_<id>.csv` (title-level only).
- **`acl/train-abs.data` → HTTP 404.** There is NO abstract-level annotation for ACL papers in CS-NER. The `acl/` folder is title-only, period.
- The rich `*-abs.data` files live only under `ftd/ncg/pwc/scierc/full dataset/` = **different corpora (NCG/PapersWithCode/SciERC/FTD), not our ACL Anthology papers**, with no id/title to match on. (Common tokens like "BERT"=935 lines just reflect BERT being *cited* widely, not our BERT paper being a record.)
- **Conclusion:** CS-NER cannot provide a richer per-paper entity list for our corpus — title-level (2-3 entities) is all it has for ACL.

**DECISION + IMPLEMENTED (user, 2026-06-16): entity list ← CS-NER global gazetteer, intersected per paper.** (Briefly considered ITER over full text, but the user chose to use the CS-NER annotations directly.) Built as `enrich_entity_csv.py --source csner` (now the DEFAULT source). Non-circular, no GPU.

**How it works (mechanism B):**
- `build_gazetteer()` downloads ALL CS-NER files, aggregates → one global gazetteer (`output/acl/csner_gazetteer.csv`, ~51k entities, 7 CEO types) cached once (`--rebuild-gazetteer` to refresh). Types mapped via `CSNER_TO_CEO`.
- `extract_body_entities_csner()` keeps only gazetteer entities that appear in the paper's `output.html` (n-gram match ≤8 words) → per-paper `Entity_<id>_enriched.csv`.
- Quality gate `_gazetteer_keep`: multi-word entries kept (full); single tokens must be non-function-word (`_FUNCTION_WORDS`) and seen ≥2× — this killed the "and/for/use" noise that otherwise matched every paper. User picked "full" gazetteer (not the ≥2× core) since paper-intersection self-filters most noise.
- `_clean` now also strips surrounding brackets/quotes (fixed `(ZeRO` → `ZeRO`).

**Verified:** BERT→154, P17-1128→114, S16-1018→111, 2020.conll-1.24→102, D17-1028→69 entities, real & properly typed (e.g. BERT paper finds BERT/Pre-training/fine-tuning/GPT/LSTM). `--source llm` (open-extraction harvest) is retired; `--source iter` (ITER/SciERC model) kept as a fallback, not used.

**TODO (next):**
1. `python3 enrich_entity_csv.py --all --source csner` — rebuild ALL entity CSVs from the gazetteer.
2. Re-run **fixed** (and **pair**) extraction off them.
3. Re-run the KGE eval (fixed-only) — triples will have changed.

> ⚠ Existing `Entity_<id>_enriched.csv` from the old `--source llm` are a STOPGAP until overwritten by `--source csner`. `Claude.md` "Entity list source for the fixed extractor" updated to match.

---

## Session 13 — Fixed the 2 "broken" PDF parses + disk cleanup (2026-06-14)

**Disk:** deleted the old cached models `Qwen3-32B` (62G) and `Qwen3-30B-A3B` (57G) from `~/.cache/huggingface/hub/` — freed ~118GB (73G→191G free). Remaining: Qwen3-14B (extractor) + Qwen2.5-7B/14B-Instruct (judges, retired but kept).

**Re-parse of J13-4001 + D17-1028:** both were flagged broken (0 / 83 words) and assumed to have dead text layers. **Both PDFs actually have healthy text layers** (9k / 10.5k chars in first 3 pages) — the failures were two bugs in `parser/structure_builder.py`, now fixed:

1. **`is_heading` missed dotted headers (J13-4001).** Its numeric regex `^\d+(\.\d+)*\s+[A-Z]` accepts `1 Introduction`/`2.1 Model` but not `1. Introduction` (number-dot-space). J13 is an essay with dotted headers (`1. False and True Starts`, `3. Interpretation`) → 0 headings detected → all 95 body blocks dropped via the `if not current_section: continue` guard. **Fix:** added a dotted-numeric branch that also accepts `N. Title`, guarded against numbered *body list items* (`1. All morphemes are created equal.`) by requiring a short remainder (≤8 words) that doesn't end in sentence punctuation.

2. **Table over-detection suppressed body prose (D17-1028).** The borderless detector found 13 "tables" in a 7-page paper; their cell tokens built a 171-entry `table_cell_set`, and `is_table_data_paragraph(threshold=4)` then deleted real body paragraphs containing ≥4 incidental cell words (body 20.6k→8.4k chars). **Fix:** added a match-density gate — long paragraphs (>200 chars) are only suppressed when matches are dense (≤60 chars/match); genuine table-row dumps are short/dense and still caught.

**Verified:** J13-4001 0→5 sections / ~8.8k words; D17-1028 83→3.2k words / 10 sections. Regression-checked on 8 known-good papers (2020.acl-main.130, conll, emnlp, etc.): **0 spurious dotted-headings, section/char counts unchanged** — the new branches only fire on papers that need them. Both regenerated to the canonical `output/<id>/no-llm/output.html` (kg_main's default mode).

**Post-parse extraction results (ran open→enrich→fixed on both):**
- **D17-1028** — open extraction rich; fixed = **7 triples**. Fine: a short single-method paper. (Note: if re-enriching, regenerate the open `llm/triples.json` from the NEW parse first — the enriched CSV is harvested from it, and a stale one built on the old broken parse starves the subject pool.)
- **J13-4001 — EXCLUDE from the fixed/KGE pipeline (content mismatch, NOT a parse bug).** The parse is now correct (open extraction yields **31 real triples**), but the content is Jerry Hobbs' ACL *Lifetime Achievement essay* — a retrospective on computational semantics/abduction/knowledge representation (`Davidson proposed_by Donald Davidson`, `weighted abduction algorithm proposed_by Mark Stickel`, …). It has almost no experimental entities (datasets/models/metrics/tasks), so the enrichment quality filter salvaged only **5 vaguely-typed entities** and the fixed extractor's domain/range validation passed **0**. The CEO ontology simply has nothing to grab onto in a philosophical essay. It also has no valid citations (the S2 network pulled wrong "Bacillus" papers), so it contributes nothing to the KGE eval regardless. **Do not re-investigate** — it's the wrong *kind* of paper, not a pipeline fault.

---

## Session 12 — KG embedding eval is now the PRIMARY triple evaluation (2026-06-13)

**Decision:** the KG-embedding citation-prediction eval (`kg_transe_pipeline.py`, `--kge transe/complex/rotate`) is now the **main way to evaluate triples**. `kg_evaluate.py` (LLM-as-Judge faithfulness) is **retired** — keep its files, but don't rely on it going forward.
- Why: embedding eval measures whether the KG is *useful* (predicts held-out citations across the whole corpus); the judge only measured per-sentence faithfulness.
- These are two different things — don't confuse them. Embedding eval = corpus-level, one results file in `output/` (NOT per-paper). Judge = per-paper, under `output/<id>/kg/fixed/<model>/eval_summary.json`.

**First baseline run** (`--extractor fixed --model Qwen3-14B --kge all --epochs 500`, 22 papers w/ citations, 410 held-out edges, KG 3804 entities / 7931 triples):

| Model   | Hits@1 | Hits@5 | Hits@10 | MRR   |
|---------|--------|--------|---------|-------|
| TransE  | 0.091  | 0.136  | 0.227   | 0.140 |
| ComplEx | 0.000  | 0.182  | 0.364   | 0.085 |
| RotatE  | 0.091  | 0.182  | 0.318   | 0.155 |

→ **RotatE best overall (MRR 0.155); ComplEx best Hits@10 (0.364).** Well above random (~0.2% Hits@1 over 588 paper nodes). Results saved to `output/kge_fixed_results_{transe,complex,rotate}.json`. Run command:
`python3 kg_transe_pipeline.py --extractor fixed --model Qwen3-14B --kge all --epochs 500 --save-results output/kge_fixed_results.json`

> ⚠ **Evaluate ONLY the `fixed` extractor with KGE.** Open (`llm`) and `pair` extractions are NOT evaluated — do not run the embedding eval on them or compare extractors on this metric. (An open-vs-fixed comparison was run once and discarded; the open result files were deleted.)

---

## Session 11 — Bigger judge model + evaluator device_map fix (2026-06-13)

- **kg_evaluate.py (LLM-as-Judge) can use a bigger judge.** It loads the judge in 4-bit NF4, so `--model Qwen/Qwen2.5-14B-Instruct` (~9GB) fits the 20GB vGPU. Recommended over the default `Qwen2.5-7B-Instruct` — stronger, and a *different family* from the Qwen3-14B extractor (less self-judging bias). 32B still won't fit.
- **Fixed a latent CPU-offload bug:** `kg_evaluate.py` loaded the judge with `device_map="auto"` — the exact trap from Session 4 (silent CPU offload → "freeze"). Changed to `device_map={"": 0}` to match both extractors. Mattered more now since a 14B judge is likelier to trigger offload than the 7B.
- Pass `--entity-csv output/acl/<id>/Entity_<id>_enriched.csv` for `fixed` so the judge resolves abbreviations via aliases (more accurate verdicts). Run command:
  `python3 kg_evaluate.py --paper <id> --extractor fixed --model Qwen/Qwen2.5-14B-Instruct --entity-csv output/acl/<id>/Entity_<id>_enriched.csv`

---

## Session 10 — New "pair" extractor: closed entities, open relation (2026-06-12)

**New extraction mode requested:** fix BOTH subject and object to the entity list, let the LLM choose a FREE-FORM relation. Spectrum now:
- `llm`   : subject free,          relation free,        object free
- `fixed` : subject ∈ list,        relation ∈ ontology,  object free
- `pair`  : subject ∈ list,        relation FREE,        object ∈ list   ← NEW

**Built `kg_extraction/pair_extractor.py` → `EntityPairExtractor`:**
- Subclasses `FixedTripleExtractor` (reuses 4-bit model load, paragraph splitting, present-entity detection, `extract_from_sentences` loop). Overrides only the system/user prompt and the parser.
- Parser `_parse_pair_output`: keeps a triple only if BOTH subject and object resolve to a listed entity (`entity_set.match`), they differ, BOTH are named in the source sentence (`_subject_in_sentence`), and the free relation is ≤5 words. Output uses key `predicate` (the chosen free phrase) + `object_type="Entity"`.
- Skips paragraphs with <2 present entities (a pair needs two).
- Added `EXTRACTION_MODE` class attr on `FixedTripleExtractor` (default "fixed"); the pair subclass sets "pair" so triples are tagged correctly.

**Wiring:** exported in `kg_extraction/__init__.py`; `kg_main.py` got `--extractor pair` (and `pair` in `all`), shares the entity-CSV resolution/auto-resolution with `fixed` (generalized `ENTITY_EXTRACTORS = ("fixed","pair")`), and writes to `output/<paper>/kg/pair/<model>/triples.json`.

**Verified (no model load):** imports, CLI choice, and the parser unit test (kept a valid listed-entity pair, rejected a pair whose object wasn't listed). End-to-end LLM run not yet done.

**✅ END-TO-END VALIDATED 2026-06-15** (`kg_main.py --all --extractor pair --model Qwen/Qwen3-14B`):
- **34 papers, 434 pair triples** (avg ~12.8/paper; lower than fixed's 734, as expected since both ends are constrained). All `object_type="Entity"`.
- **3 zero-triple papers** — all explainable, not failures: J13-4001 (award essay, no entity pairs), C16-1036 + E17-3026 (genuinely short ~800–1000-word papers).
- Quality is good: both ends are real listed entities, relations are accurate and pair-appropriate — `BlueBERT outperforms BERT`, `MS-BERT pre-trained on 70,000 MS consult notes`, `classification system consists of {Random Forests, Gradient Boosting Trees, SVMs}`, `TextRunner uses Naïve Bayes classifier`.
- **Note:** the free relations are un-normalized → 210 "distinct" relations inflated by surface variants (`uses`/`use`, `is evaluated on`/`evaluated on`/`evaluates on`, `is based on`/`are based on`). Inherent to open-relation extraction. If pair triples ever feed something keyed on the relation string, add light normalization (lowercase, strip leading `is/are`, lemmatize). Pair is NOT KGE-evaluated (KGE = fixed only), so cosmetic for now.

> ⚠ VRAM: like `fixed`/`llm`, `pair` loads its own Qwen3-14B (~10GB). Do NOT run `--extractor all` on the 20GB vGPU — it would load llm+fixed+pair simultaneously and OOM. Run one LLM extractor at a time.

**Run it:** `python3 kg_main.py --paper C16-1036 --extractor pair --model Qwen/Qwen3-14B` (entity CSV auto-resolves).

---

## Session 9 — Generalize fixed-extractor prompt for multiple papers (2026-06-12)

**Decisions locked in this session:**
- **Triple shape:** subject = from the entity list, predicate = from the CEO ontology, **object = free text** (NOT constrained to the list). An entity→entity experiment was tried and **reverted** per the user — objects stay free.
- **No longer BERT-only.** The `fixed_extractor.py` system prompt was saturated with BERT/NLP examples (BERT, GLUE, SQuAD, MLM, NSP, ELMo, WordPiece, BooksCorpus, "Radford et al."), which biased it toward BERT-style papers.

**Change:** rewrote every illustrative example in `_make_fixed_system_prompt` + the `_CEO_SCHEMA` relation hints to **paper-agnostic placeholders** (`<Model>`, `<Method>`, `<Dataset>`, `<Task>`, `<Metric>`, "a named problem", etc.). The ontology domain/range RULES are unchanged — only the examples were genericized. Verified the built system prompt contains zero paper-specific tokens. Generic metric names (F1/Accuracy/BLEU/RMSE/WER) kept since they're domain-neutral.

> Left as-is (intentionally): BERT mentions in **code comments** and the garbled-section regex (`E[CLS]`, `[unused\d+]`) — those are defensive PDF-artifact detection, not part of the LLM prompt, and harmless for other papers.

**Not yet validated end-to-end on multiple papers** — prompt builds clean, syntax OK. Next: run `kg_main --all --extractor fixed` on a few diverse ACL papers and eyeball that the ontology predicates fire sensibly across domains (speech, NER, MT, etc.), not just BERT.

**✅ VALIDATED 2026-06-14** (via the existing corpus — Session 12's fixed run on 06-12/06-13 already used this generalized prompt, so no re-run needed; just inspected the on-disk triples):
- Corpus-wide: **734 fixed triples across 35 papers, 17 of the 22 CEO predicates firing**, well distributed — `uses` 21.7%, `addresses` 21.4%, `achieves` 19.1%, then `comprises`/`evaluatedon`/`comparesagainst`/`splitfrom`/`trainedon`/`encompasses`/`produces`/… The Session-8 `addresses → Concept` collapse is down to **12.5%** (was 75% on C16-1036 pre-filter).
- Spot-checked a diverse, non-BERT sample (clinical text, Twitter sentiment, word-sense induction, Chinese hypernymy, emotion-distribution meta-learning, type-driven composition, Chinese open relation extraction). The prompt is genuinely domain-adaptive: e.g. clinical paper → `BlueBERT trainedOn PubMed abstracts`, `MS-BERT trainedOn 70,000 MS consult notes`; relation-extraction paper → `TextRunner uses Naïve Bayes classifier`. Subjects are always real in-paper entities; no BERT/GLUE/SQuAD bias leaked.
- Known minor weakness (acceptable, not a regression): **short papers lean on one predicate** (E14-4003 = 8×`uses`/9; P17-1128 heavy on `addresses` with a repeated generic object). Expected when a paper genuinely only describes a method using components — not the Session-8 garbage collapse.

→ **Conclusion: the genericized prompt works across domains. This item is closed.**

---

## Session 8 — Quality filter on entity enrichment (2026-06-12)

**Why:** with the min-count-1 enriched pool, fixed extraction on C16-1036 gave 12 triples but ~quality was poor — **9/12 were `addresses → Concept`** with free-text objects, and several were unfaithful. Root cause: the free-LLM harvest had dumped vague descriptive phrases into the entity list ("human expressiveness", "concept of a continuous system", "ideal expressive TTS system", "new model", "present paper"), all typed `Concept`/`Other`. The extractor obeyed its constraint (all 12 subjects WERE in the list) — the list itself was the problem. Goal per user: not BERT precision, but *proper* use of the entity list + CEO ontology.

**Fix in `enrich_entity_csv.py` (quality filter, default ON):**
- Body entities kept only if their CEO type ∈ `HARD_CEO_TYPES` (Model/Method/Dataset/Tool/Metric/EvaluationMetric/Task/Resource/Language). Drops `Concept`/`Other`/`Generic`.
- `_is_noise` now also drops generic-leading phrases (`_GENERIC_LEADING`: the/a/this/new/present/proposed/concept/… as first word) and over-long phrases via `--max-words` (default 6).
- **Title seeds are exempt from both filters** — always kept.
- Escape hatches: `--keep-all-types`, `--max-words 0`.

**Result:** C16-1036 went 42 noisy → **21 proper ontology-typed entities**; all 8 junk subjects removed. Re-run fixed extraction to get cleaner ontology relations (uses/produces/trainedOn/…) instead of `addresses → Concept`.

---

## Session 7 — Auto-resolve entity CSV for fixed extraction (2026-06-12)

**Goal:** stop having to pass `--entity-csv` for fixed extraction — the CSV is deterministically tied to the paper, so resolve it from the paper id automatically.

**Changes in `kg_main.py`:**
- Added `resolve_entity_csv(paper_name)` — search order: `output/acl/<id>/Entity_<id>_enriched.csv` → `output/acl/<id>/Entity_<id>.csv` → project-root `Entity_<id>.csv` / `Entity-<id>v2.csv` / `Entity - <id>v2.csv`. Returns first hit.
- `--entity-csv` is now **optional**. If omitted (`auto_entity=True`), the fixed extractor is built with an empty placeholder `EntitySet([], {}, {})`, and `process_paper` resolves + loads the CSV **per paper**, swapping `extractors["fixed"].entity_set` before extracting. Works because `FixedTripleExtractor` reads `self.entity_set` at call time and its system prompt is relations-only (entities go in the per-call user prompt).
- If a paper has no resolvable CSV in auto mode, fixed extraction is skipped **for that paper only** (other extractors still run); explicit `--entity-csv` keeps the original single-set-for-all behaviour.

**Verified:** resolver returns enriched CSV for C16-1036/E17-3026, title-only for D17-1028 (no enriched), and the manual `Entity-BERTv2.csv` for BERT. Full extraction not run here (loads the LLM) but wiring is syntax-clean and the per-paper swap is in place.

**Usage now:**
```bash
python3 kg_main.py --paper C16-1036 --extractor fixed --model Qwen/Qwen3-14B   # no --entity-csv needed
python3 kg_main.py --all --extractor fixed --model Qwen/Qwen3-14B              # different CSV per paper
```

---

## Session 6 — Entity-CSV enrichment for fixed extraction (2026-06-12)

**Problem found:** the ACL/CS-NER entity CSVs only have ~2-3 entities each because **CS-NER annotates paper TITLES only** (verified: source files are one ~8.8-token title per record; 98 CSVs avg 2.15 entities). The fixed extractor is hard-gated on its subject list (`fixed_extractor.py` prompt: "Use ONLY these as subjects"), so a 2-entity CSV → ~0 triples. The user's main workflow is now ACL papers, so these CSVs must be enriched into a real subject pool.

**Built `enrich_entity_csv.py`:** keeps the ACL title entities as guaranteed-correct **seeds**, harvests real in-paper entities, dedupes (seeds win), and writes the standard 6-col CSV (`Entity,Abbreviation,Aliases,TP,NER_Type,CEO_Type`, all TP=1).
- Two sources via `--source`:
  - **`llm` (default)** — harvests subjects+objects from the open-extraction `triples.json` we already computed (`output/<id>/kg/llm/<model>/triples.json`). No install, no GPU; entities come pre-typed (Method/Task/Model/...). **This is why we did NOT install ITER** — ITER (`pip install git+https://github.com/fleonce/iter`) is not installed and could repin torch and undo the Session-4 fix.
  - **`iter`** — ITER/SciERC over `output/<id>/no-llm/output.html` (kept as an option; added `ITERExtractor.extract_entities()` to return all entities, not just those in a relation).
- `--in-place` overwrites `Entity_<id>.csv` (default writes `Entity_<id>_enriched.csv`); `--min-count N` trims one-off noise; `--max-entities N` caps.

**Result:** `--all` enriched 31/33 papers, **2-3 → 22-241 entities each**. Verified the output loads via `entity_loader.load_entity_csv(..., tp_only=True)`. (2 papers skipped: their llm triples.json was empty.)

**Use it:**
```bash
python3 enrich_entity_csv.py --all          # min-count 1 (default) — keep it here, see below
python3 kg_main.py --paper <id> --extractor fixed \
    --entity-csv output/acl/<id>/Entity_<id>_enriched.csv --model Qwen/Qwen3-14B
```

> ⚠ **Keep `--min-count` at 1.** A single paper rarely repeats an entity, so `--min-count 2+`
> deletes most of the pool. Session-7 hit this: C16-1036 was enriched with `--min-count 2`,
> dropped 41→9 entities, and fixed extraction produced only **1 triple**. Re-enriched at
> min-count 1 → 42 entities (35 matching text). Frequency filtering only makes sense across a
> corpus, not within one paper.

> Note on TP: the ACL CSVs are all `TP=1` by construction, so the `tp_only` filter in `entity_loader` is a no-op on them — the list length, not TP, was the real constraint.

---

## Session 5 — Added ComplEx + RotatE to KG embedding eval (2026-06-11)

**Goal:** evaluate KG embeddings with TransE, ComplEx, and RotatE (TransE already existed).

**Changes in `kg_transe_pipeline.py`:**
- Added `ComplEx` (complex bilinear, `Re(<h,r,conj(t)>)`) and `RotatE` (relation = unit-modulus rotation, score `-||h∘r - t||`) nn.Module classes alongside the existing `TransE`.
- For both, `entity_emb` stores the full `[real‖imag]` vector (width `2*dim`) so the existing `predict_related_papers`/`evaluate` (cosine sim on `entity_emb.weight`) work **unchanged** — they were already model-agnostic.
- Added `KGE_MODELS` registry + `build_kge_model(kge, ...)`. Renamed `train_transe` → `train_kge(graph, kge=..., ...)` (one training loop: margin-ranking loss + head/tail corruption negative sampling, used by all three).
- New CLI flag `--kge {transe,complex,rotate,all}` (default `transe`). `all` trains+evaluates all three and prints a Hits@1/5/10 + MRR comparison table.
- `--save-model` / `--save-results` get a `_{kge}` suffix when `--kge all`; checkpoints now store the `kge` name so `--load-model` rebuilds the right class.
- `--dim` is the **complex** dim for ComplEx/RotatE (entity storage is `2*dim`).

**To add a 4th model later:** register an nn.Module with `entity_emb` (full entity vector), `relation_emb`, `forward(h,r,t)→score` (higher=better), `loss(...)` in `KGE_MODELS`. Nothing else needs changing.

**Verified:** smoke test `--kge all --epochs 5 --device cpu --extractor llm --model Qwen3-14B` — all three train, evaluate (24 papers), and the comparison table prints. Run real evals with `--epochs 500` (and `--device cuda`).

> Note: `--model` = the LLM subdir (e.g. Qwen3-14B); `--kge` = the embedding model. Two different "models".

---

## Session 4 — "freeze" was a torch-nightly regression, NOT CPU offload (2026-06-10)

**Symptom:** `kg_main --extractor llm` appeared to *freeze right after the model loaded* — long silent gap, `nvidia-smi` showing GPU util 0% and one CPU core pegged at 100%. Looked exactly like the session-3 CPU-offload freeze (#4), but it was a different cause.

**What it was NOT (ruled out with evidence):**
- NOT CPU offload / wrong device — a clean load showed all 443 params on `cuda:0`, generation grew VRAM 9.6→10.8GB, and a profiler showed the work running as the CUDA kernel `kgemm_4bit_inference_naive` (`bitsandbytes::gemv_4bit`). It runs on GPU.
- NOT `llm_extractor.py` / `kg_main.py` — proven by reproducing the slowness in a from-scratch load that bypasses our code.
- NOT bitsandbytes itself — a 4-bit `Linear4bit` microbench was GPU-fast.
- NOT `python` vs `python3` as the perf cause — both resolve to the same `/usr/bin/python3.10` + same `~/.local` site-packages. (NOTE: always invoke commands as `python3` anyway — `python` does not work as expected for running the pipeline on this VM.)
- NOT a clock throttle — SM clock idles at 345MHz but boosts to ~1755MHz under load.

**Root cause:** the installed torch was a **dev nightly** (`2.12.0.dev20260408+cu128`) that had a regression crippling bitsandbytes 4-bit *decode* → ~**2 tok/s** (an H100 should do 30–60). Throughput work (big matmuls) was fine; only latency-bound token-by-token decode was destroyed.

**Fix:** replaced the whole nightly torch stack with **stable** builds:
```bash
pip uninstall -y torch torchvision torchaudio triton
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# → torch 2.6.0+cu124, torchvision 0.21.0, torchaudio 2.6.0, triton 3.2.0, cuDNN 9.1
```
cu124 wheels run fine on the 12.8 driver (CUDA is backward-compatible). After the swap: **~16 tok/s** (8× faster). Do NOT reinstall a torch nightly on this VM.

**CRITICAL gotcha — `nvidia-smi` GPU-util is broken on this H100-20C vGPU.** It reads **0% even when the GPU is fully busy** (verified: a sustained 4096³ matmul loop showed 0% util while the SM clock sat at 1.6GHz). So "GPU 0% + one CPU core at 100%" is **NOT** evidence of CPU execution here, and one core at 100% during decode is normal (kernel-launch dispatch). To tell if a run is alive, check instead:
- log advancing (`Graph: N nodes, M edges` per section)
- VRAM ~10–11GB (`nvidia-smi --query-gpu=memory.used`)
- SM clock ~1755MHz under load (`nvidia-smi --query-gpu=clocks.sm`)
- growing CPU time on the PID (`/proc/<pid>/stat` fields 14+15)

**Also:** at ~16 tok/s, kg_main's default `--max-new-tokens 4096` (and ×2=8192 in the postprocess pass, `llm_extractor.py:488`) means a single generation runs several silent minutes — looks frozen but isn't. Use `--max-new-tokens 512` for snappy, visibly-progressing runs. The first `generate()` after `loaded.` is silent for ~60–75s with no per-token logging — that gap is the thing most easily mistaken for a freeze.

---

## Future plan
- If Qwen3-14B quality is noticeably worse than 32B, options:
  - Try the non-gated `mistralai/Mistral-Small-3.1-22B-Instruct-2503` (~11–12GB, fits 20GB)
  - Request access to gated `google/gemma-3-27b-it` (closest to 32B quality that fits)
- Re-run the BERT precision evaluation (`kg_evaluate.py`) on the new model and compare against the documented 32B baseline.
- Cached models in `~/.cache/huggingface/hub/`: still have the old `Qwen3-32B` and `Qwen3-30B-A3B` (57GB) — can delete to free disk once 14B is confirmed working.
