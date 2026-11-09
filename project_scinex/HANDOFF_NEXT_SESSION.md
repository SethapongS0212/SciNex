# Handoff — next session

**Updated 2026-11-09 13:00 server time (Session 39).** Read in this order:
1. **[`OBJECTIVE.md`](OBJECTIVE.md)** — the professor's three steps. Everything serves them.
2. This file — where the work stands and what to do next.
3. `hands_off.md` §39 (and §38) for detail; `results.md` for tables (Table 20 = the main evaluation).

> **Standing rule (user):** update the md docs after every meaningful step, as it happens; commit
> and push to GitHub when the change is important or necessary.

## ⛔ The objective in four lines (full text: `OBJECTIVE.md`)
- Paper → KG; **finetune an LLM so it understands the ontology + vocabulary**; **evaluate the KG
  with simulated Q&A against the real papers** (the MAIN evaluation).
- The finetuned extractor runs **without the ontology in its prompt** (`--prompt-variant noont`):
  entity list + text only. Teacher → student distillation only makes the training data.
- Corpus: 805 ACL papers, fixed entity lists; fold 0 (41 papers) held out.
- Per-triple judging (C1/C3/C4) and judge-free counts are diagnostics, not the headline.

## State (2026-11-09, 13:00 server time = 20:00 user time)
- ✔ Step 2 of the objective MET (Tables 19 / 19b): ontology-free students keep 87–93% of
  proposals and are as correct as ontology-in-prompt students and the teacher.
- ✔ **Main evaluation (simulated Q&A), `results.md` Table 20**, complete for every KG so far:
  noont = ontology-in-prompt (|diff| ≤ 0.6); noont-me30 beats its teacher (+3.3 / +3.5).
- ✔ **Teacher v2 + student on v2 data DONE** (Tables 20 / 21): teacher v2 6.2 / 4.9 vs v1 5.0 / 3.3;
  **v2-data noont-me30 Qwen3-8B = 10.0 / 8.0, the best finetuned KG** (+3.8 / +3.1 over teacher v2,
  +1.7 / +1.2 over the v1-data student, CIs touch 0); 851 / 730 triples, coverage 53 / 45%.
  Only the noisy untuned full-ontology graph scores higher (12.7 / 12.8).
- **⏳ RUNNING, unattended: `screen -r teacher2`** = `/home/dev1/scinex_runs/run_teacher2.sh` (copy in
  `ft_runs/acl_scripts/`), log `teacher2_driver.log` (hands_off §39.6–39.7). **Second teacher
  Qwen3-30B-A3B-Instruct-2507 W4A16 via vLLM** (`/home/dev1/envvllm`, `ft_generate.py --backend
  vllm`, `VLLM_USE_FLASHINFER_SAMPLER=0`), slug `qwen3-30b-a3b-v2`, all 805 papers on BOTH GPUs
  (started 12:48, logs `teacher2_full_gpu{0,1}.log`) → compare (`teacher2_compare.log`) → its Q&A
  (GPU1) while a noont-me30 Qwen3-8B student trains on its data (GPU0; slug
  `qwen3-8b-acl-q30b-noont-me30-cv`) → Q&A → report (`teacher2_qa.log`,
  `teacher2_students_compare.log`). Re-running the script resumes every step.
- **⛔ GPU settings CHANGED for heat while it runs** (university property): persistence on, 400 W,
  clock locks 1650 / 1900 MHz, root `screen gpuguard`. **Restore is AUTOMATIC:** root `screen
  gpurestore` = `gpu_restore_after.sh 'run_teacher2[.]sh'` restores them 10 min after the driver
  exits and the GPUs go idle (log `gpu_restore.log`, "=== GPU settings RESTORED ==="). If the driver
  dies uncleanly, restore by hand:
  ```
  sudo screen -S gpuguard -X quit; sudo screen -S gpurestore -X quit
  sudo nvidia-smi -i 0 -rgc && sudo nvidia-smi -i 1 -rgc
  sudo nvidia-smi -pl 575 && sudo nvidia-smi -pm 0
  ```
  New GPU runs after a restore start at default settings: re-apply the clock locks first (§38.1).
  The sudo password is not stored; ask the user.
- Models in the HF cache: students Qwen3-8B, Gemma-3-12B-it (also KGRAG's model — never delete),
  Phi-4 (**Qwen3-4B and Llama-3.1-8B-Instruct deleted 2026-11-08 at the user's request**; their
  adapters stay in `ft_runs/`; re-download before reusing them, the drivers run with
  `HF_HUB_OFFLINE=1`; KGRAG's models nllb-200 / bge-m3 / ms-marco / Llama-3.2-1B / Gemma stay);
  teachers gpt-oss-20b and Qwen3-30B-A3B-Instruct-2507 (bf16, unusable 4-bit; to delete) + its
  **W4A16 `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` (vLLM, envvllm)**; Q&A model
  Mistral-Small-24B-Instruct-2501.

## Next steps, in order
1. **When `teacher2` is done:** `results.md` — add teacher q30b and the q30b-data student to Table 20
   (from `qa_eval/acl-f0/report_table20.json` / `teacher2_qa.log`) and Table 21 (from
   `teacher2_compare.log`, `teacher2_students_compare.log`); the key comparison is
   `kg__*__qwen3-8b-acl-q30b-noont-me30-cv` − `kg__*__qwen3-8b-acl-v2-noont-me30-cv` (only the
   teacher differs). Re-run `qa_gap_analysis.py` with the new KGs. Check `gpu_restore.log` says
   RESTORED. Delete the 57 GB bf16 `Qwen/Qwen3-30B-A3B-Instruct-2507` from the HF cache (user
   agreed; the W4A16 copy is the one in use). Commit + push.
2. **Blind C1/C3/C4 round on the v2-data student** (and the q30b one): are the extra triples
   correct? (Q&A rewards coverage; the untuned graph shows coverage alone can mislead.)
3. **Decide with the user**: non-Qwen student on both teachers' data (family confound; Phi-4 /
   Gemma-3-12B in cache), lower empty caps, more folds.
4. Later: more folds; entity-list REVIEW rows (user/professor); guard fixes (`88.5` leak,
   self-loops); `--share-system-prompt` GPU speed test (less relevant now that noont prompts are short).

## Machine
Linux box `kubegpu-g04`, **2× RTX 5090 (32 GB, sm_120)**, 125 GB RAM. Second venv **`/home/dev1/envvllm`**
(vLLM 0.32.0, torch 2.13.0+cu130; only for `ft_generate.py --backend vllm`). venv **`/home/dev1/envsam`**
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

