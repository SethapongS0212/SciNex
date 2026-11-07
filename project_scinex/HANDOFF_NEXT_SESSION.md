# Handoff — next session

**Updated 2026-11-07 15:40 server time (end of Session 38).** Read in this order:
1. **[`OBJECTIVE.md`](OBJECTIVE.md)** — the professor's three steps. Everything serves them.
2. This file — where the work stands and what to do next.
3. `hands_off.md` §38 for detail; `results.md` for tables.

> **Standing rule (user):** update the md docs after every meaningful step, as it happens; commit
> and push to GitHub when the change is important or necessary.

## ⛔ The objective in four lines (full text: `OBJECTIVE.md`)
- Paper → KG; **finetune an LLM so it understands the ontology + vocabulary**; **evaluate the KG
  with simulated Q&A against the real papers** (the MAIN evaluation).
- The finetuned extractor runs **without the ontology in its prompt** (`--prompt-variant noont`):
  entity list + text only. Teacher → student distillation only makes the training data.
- Corpus: 805 ACL papers, fixed entity lists; fold 0 (41 papers) held out.
- Per-triple judging (C1/C3/C4) and judge-free counts are diagnostics, not the headline.

## State (2026-11-07, 15:40 server time = 22:40 user time)
- ✔ **Ontology-free students DONE and checked.** Table 19 (counts): with no ontology in the prompt
  every finetuned student keeps 87–93% of its proposals; the untuned model with the same prompt
  ~10%. Table 19b (blind C1/C3/C4): noont-me50 C4 58 / 61 vs ontology-in-prompt me50 49 / 63 vs
  teacher 54 / 57 (CEO / scinex) — a tie. **Step 2 of the objective is met.**
- ✔ **Q&A gap analysis DONE** (`qa_gap_analysis.py`, hands_off §38.9): 70% of the 1,222 questions
  mention a listed entity; a KG answers ~40% of the questions whose answer it contains and ~0%
  otherwise; the answer is in the graph for only 16–25% of questions (teacher / finetuned KGs).
  **Coverage of paper facts is the bottleneck.**
- **⏳ RUNNING, unattended — check these first:**
  1. `screen -r qa0`, `screen -r qa1` (started 15:11, done ~18:30 server): Q&A answers from the 14
     noont KGs. Log `/home/dev1/scinex_runs/qa_driver.log`.
  2. **`screen -r chain`** = `/home/dev1/scinex_runs/run_after_qa.sh` (copy in
     `ft_runs/acl_scripts/`), log **`/home/dev1/scinex_runs/after_qa_driver.log`**. It waits for
     qa0/qa1, then: grades + reports the Q&A (`qa_eval/acl-f0/report.json`, log `qa_grade.log`) →
     **teacher v2** (`gptoss-20b-v2` = gpt-oss-20b, `--prompt-variant recall`, `--entity-dir
     output/acl_v2`) on all 802 papers × 2 ontologies, the 41 eval papers first, ~30 h, logs
     `teacher_v2full_gpu{0,1}.log` → `ft_compare` v1 vs v2 (`teacher_v2_compare.log`,
     `ft_runs/teacher_v2/compare_v2.json`) → Q&A answers from the v2 teacher KGs + grade + report.
     Expected done ~2026-11-09 00:00 server. Re-running the script resumes every step.
- **⛔ GPU settings are CHANGED for heat while these run** (university property): persistence on,
  400 W, clock locks 1650 / 1900 MHz, root guard `screen gpuguard` (checked: running).
  **Restore when the chain is done and nothing else runs:**
  ```
  sudo screen -S gpuguard -X quit
  sudo nvidia-smi -i 0 -rgc && sudo nvidia-smi -i 1 -rgc
  sudo nvidia-smi -pl 575 && sudo nvidia-smi -pm 0
  ```
  The sudo password is not stored; ask the user.
- Preliminary Q&A (graded 2026-11-07 15:15, before the noont KGs): oracle 86.3, closedbook 4.4, KG
  teacher 5.0 / 3.3, untuned base 12.7 / 12.8, me30 8.5 / 6.3, me50 5.1 / 2.9 (CEO / scinex).
- Models in the HF cache: students Qwen3-4B/8B, Llama-3.1-8B-Instruct, Gemma-3-12B-it, Phi-4;
  teachers gpt-oss-20b and **Qwen3-30B-A3B-Instruct-2507 (second teacher, not run yet)**; Q&A model
  Mistral-Small-24B-Instruct-2501.

## Next steps, in order
1. **When qa0/qa1 are done** (the chain grades automatically): write `results.md` **Table 20**
   from `qa_eval/acl-f0/report.json` — accuracy per KG (teacher, base, ontology-in-prompt me30/me50,
   every noont student, both ontologies) + oracle/closedbook; compare noont vs ontology-in-prompt
   (`qa_eval.py report --run acl-f0 --compare kg__fixed__qwen3-8b-acl-noont-me50-cv
   kg__fixed__qwen3-8b-acl-ft-me50-cv`); run `qa_gap_analysis.py` on the noont KGs. A few
   questions may be unanswered (the report flags them): re-run `run_qa.sh <gpu> kg …` /
   `controls` to fill them, then grade again. Commit + push.
2. **When the teacher v2 lanes finish:** read `teacher_v2_compare.log` (does v2 extract more,
   at what kept rate?) and its Q&A (chain step 5). If v2 covers more:
   - build ontology-free training data from it, e.g.
     `python build_ft_data.py --extractor fixed --teacher gptoss-20b-v2 --entity-dir output/acl_v2
     --prompt-variant noont --papers-file papers_acl.txt --split ft_data/acl/split.json --folds 20
     --eval-fold 0 --max-empty-frac 0.3 --out ft_data/acl_v2_noont_me30` (check the manifest: counts,
     ontology cards 1,300, alignment);
   - train Qwen3-8B on it like `run_noont.sh` (`ft_train_qlora.py … --epochs 1 --save-steps 200`),
     generate with `ft_generate.py --adapter … --extractor fixed --teacher gptoss-20b-v2
     --prompt-variant noont --entity-dir output/acl_v2 --batch-size 8 --ingest`, then Q&A it.
3. **Second teacher** (user request): Qwen3-30B-A3B-Instruct-2507, same v2 setup. ⚠ Not tested yet:
   smoke-test it first (load in 4-bit, a few paragraphs, measure s/paragraph — HF MoE in 4-bit may
   be slow). For a fair comparison, run it on a FIXED subset of training papers and train a
   gpt-oss-20b-v2 student on the same subset, so only the teacher differs; also train a non-Qwen
   student (family confound). Judge by Q&A.
4. Later: more folds; entity-list REVIEW rows (user/professor); guard fixes (`88.5` leak,
   self-loops); `--share-system-prompt` GPU speed test (less relevant now that noont prompts are short).

## Machine
Linux box `kubegpu-g04`, **2× RTX 5090 (32 GB, sm_120)**, 125 GB RAM. venv **`/home/dev1/envsam`**
(Python 3.12.3, torch `2.13.0+cu130`, no `--allow-dev-torch`; `kernels==0.12.3` for gpt-oss;
`peft==0.21.1`). Repo `/home/dev1/SciNex`; `/home/dev1/project_scinex` is a symlink to its
`project_scinex/`. Drivers in `/home/dev1/scinex_runs/` (copies in `ft_runs/acl_scripts/`). Long
runs in `screen`. Push with the deploy key:
`GIT_SSH_COMMAND="ssh -i ~/.ssh/scinex_deploy_ed25519 -o IdentitiesOnly=yes" git push git@github.com:SethapongS0212/SciNex.git main`
(author SethapongS0212 <samaruek2004@gmail.com>, no git config set — pass `-c user.name/-c user.email`).

## Decisions already made — do not re-open
- Objective = `OBJECTIVE.md` (professor, confirmed by the user 2026-11-06).
- Corpus = 805 ACL papers, extractor = fixed. ACL split = `ft_data/acl/split.json` (20 folds,
  seed 13); never regenerate it.
- **Claude is never an extractor, teacher, training-data source or Q&A model.** Its only role is
  the in-session blind triple judge.
- Empty-target cap: 50% imitates the teacher's coverage, 30% gives a bigger graph; both are
  trained for the noont students.
- The GPUs are university property: every setting changed for heat is restored afterwards.

## Traps — do not repeat
- Background watchers die with the chat session; re-arm them.
- `hf download --exclude a b` treats `b` as a file to download; repeat `--exclude` per pattern.
- Gemma-3 loads as `Gemma3ForConditionalGeneration`: training needs `token_type_ids` (zeros) and
  LoRA must skip the vision tower (`re:` target regex). Phi-4 uses fused `qkv_proj`/`gate_up_proj`.
- The ontology cards read the relation schema from the extraction prompt; with `noont` they must
  read the FULL prompt (fixed in `build_ft_data.py`), or the cards silently drop to 0.
- Teacher outputs vary run to run (batching), so n=30 probes are noisy.
- **"No new process can start CUDA" after a GPU fault = the shared `nvidia_uvm` module is in an
  error state** (`/dev/nvidia-uvm` → EIO). Fix it without a reboot: `sudo rmmod nvidia_uvm && sudo
  modprobe nvidia_uvm`. Never touch a card `nvidia-smi` cannot see.
- **GPU heat:** under load GPU0 ran at 89–90 °C, fan 100%, thermally throttled. GPU1 died under the
  same load. Check the cooling before loading both cards for days.
- **A 1-epoch run with `save_strategy="epoch"` has no checkpoint.** Use `ft_train_qlora.py
  --save-steps 200`; a re-run resumes from the latest checkpoint (smoke-tested).
- **A local teacher's `{"triples": []}` is a genuine answer, not a failed API call.**
  `build_ft_data` now asks `is_api_run()`. Before the fix, every empty target would have been dropped.
- **The July ACL HTML keeps PDF hyphenation**, so the strict alignment check rejected correctly
  paired papers. Fixed builds use `alignment_rate(..., loose=True)`.
- **A `--limit` smoke run writes a truncated `prompts.jsonl` that later runs pick up.** Delete smoke
  slug folders.
- gpt-oss needs `kernels==0.12.3` (0.17 breaks transformers 5.5.4) and `--attn-implementation
  flex_attention` (eager attention runs out of memory at batch 8).
- `pkill -f` / `pgrep -f` with a pattern contained in your own command line matches your own shell.
  `nvidia-smi -L` takes no `-i`; use `-i N --query-gpu=name --format=csv,noheader`.

