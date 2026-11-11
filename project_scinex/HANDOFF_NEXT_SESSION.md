# Handoff — next session

**Updated 2026-11-11 08:00 server time (Session 39).** Read in this order:
1. **[`OBJECTIVE.md`](OBJECTIVE.md)** — the professor's three steps. Everything serves them.
2. This file — where the work stands and what to do next.
3. `hands_off.md` §39 (and §38) for detail; `results.md` for tables (Table 20 = the main evaluation; Table 22 = its correctness check).

> **Standing rule (user):** update the md docs after every meaningful step, as it happens; commit
> and push to GitHub when the change is important or necessary.

## ⛔ The objective in four lines (full text: `OBJECTIVE.md`)
- Paper → KG; **finetune an LLM so it understands the ontology + vocabulary**; **evaluate the KG
  with simulated Q&A against the real papers** (the MAIN evaluation).
- The finetuned extractor runs **without the ontology in its prompt** (`--prompt-variant noont`):
  entity list + text only. Teacher → student distillation only makes the training data.
- Corpus: 805 ACL papers, fixed entity lists; fold 0 (41 papers) held out.
- Per-triple judging (C1/C3/C4) and judge-free counts are diagnostics, not the headline.

## State (2026-11-11, 08:00 server time = 15:00 user time)
- **Nothing is running. GPU settings are at defaults** (`gpu_restore.log`: RESTORED 2026-11-10 06:29).
- ✔ Step 2 of the objective MET (Tables 19 / 19b): ontology-free students work.
- ✔ **Main evaluation, `results.md` Table 20** (accuracy CEO / scinex): oracle 86.3, closedbook 4.4;
  teachers gpt-oss v1 5.0 / 3.3, v2 6.2 / 4.9, **Qwen3-30B 16.7 / 17.0**; ontology-free Qwen3-8B me30
  students on v1 data 8.3 / 6.8, v2 data 10.0 / 8.0, **Qwen3-30B data 15.5 / 18.2** (best KG; equal to its
  teacher; above the untuned full-ontology 12.7 / 12.8).
- ⚠ **The blind check qualifies that (Table 22, C4):** gpt-oss v2 75 / 78, v2 student 48 / 59, Qwen3-30B
  26 / 34, its student 27 / 41. The Qwen3-30B line has 1.6–2.4× more correct triples than the v2
  student but 4–5× more wrong ones; Q&A cannot see wrong triples. **Report both numbers.**
- Models in the HF cache: students Qwen3-8B, Gemma-3-12B-it (also KGRAG's model — never delete),
  Phi-4 (**Qwen3-4B and Llama-3.1-8B-Instruct deleted 2026-11-08 at the user's request**; their
  adapters stay in `ft_runs/`; re-download before reusing them, the drivers run with
  `HF_HUB_OFFLINE=1`; KGRAG's models nllb-200 / bge-m3 / ms-marco / Llama-3.2-1B / Gemma stay);
  teachers gpt-oss-20b and Qwen3-30B-A3B-Instruct-2507 as
  **W4A16 `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` (vLLM, envvllm)**; Q&A model
  Mistral-Small-24B-Instruct-2501.

## Next steps, in order
1. **Ask the user which way to go** (asked 2026-11-11; check the chat or ask again):
   - **Tighten the guards** (recommended; CPU only, no retraining). Simulated on the Table 22 sample
     with three rules (subject = object or one inside the other, `employs` on anything, < half of the
     object's words in the sentence): Qwen3-30B student C4 27 → 42 (CEO) / 41 → 51 (scinex), keeping
     60 / 76% of its triples; gpt-oss v2 barely moves (75 → 76, 78 → 79). Rough: n=50, the
     substring self-loop rule also drops some good triples. Then re-ingest, re-run Q&A + a blind
     round. Old numbers stay, labelled.
   - Cleaner training data by teacher consensus is NOT viable: only 3.4% / 2.1% of Qwen3-30B triples
     match gpt-oss's. Retraining on guard-filtered Qwen3-30B data is the natural step AFTER the guards.
   - **Non-Qwen student** on the Qwen3-30B data (family confound; Phi-4 / Gemma-3-12B in cache).
2. ✔ bf16 `Qwen/Qwen3-30B-A3B-Instruct-2507` (57 GB) deleted 2026-11-11 with the user's OK. Still
   unapproved: the 2 AWQ stubs (27 MB) and the pip cache (7.3 GB).
3. New GPU runs start at default settings: re-apply the clock locks first (§38.1); sudo password not
   stored, ask the user. `qa_report_all.sh` re-grades Q&A with all comparisons.
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

