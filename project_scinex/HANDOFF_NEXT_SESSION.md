# Handoff — next session

**Updated 2026-11-07 15:30 server time (Session 38).** Read in this order:
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

## State (2026-11-07, 15:30 server time)
- ✔ **Ontology-free students DONE** (`run_noont.sh`, finished 2026-11-06 22:51; `results.md`
  Table 19, `ft_runs/noont/compare_noont.json`). With NO ontology in the prompt every finetuned
  student still keeps 87–93% of its proposals (CEO); the untuned model with the same prompt keeps
  9.5% → the ontology was learned into the weights. Slugs: `qwen3-8b-acl-noont-me{50,30}-cv`,
  `llama31-8b-…`, `qwen3-4b-…`, `gemma3-12b-…`, `phi4-acl-noont-me50-cv`, control
  `qwen3-8b-acl-base-noont`.
- ✔ **Q&A questions DONE:** run `qa_eval/acl-f0`, 1,222 questions from the 41 papers (generator and
  answerer `mistralai/Mistral-Small-24B-Instruct-2501`, local hf, batched). Answered: oracle,
  closedbook, and the KGs of the teacher, untuned base and ontology-in-prompt me30/me50 (both
  ontologies). **Preliminary (graded 2026-11-07):** oracle 86.3, closedbook 4.4, KG teacher
  5.0 / 3.3 (CEO / scinex), base 12.7 / 12.8, me30 8.5 / 6.3, me50 5.1 / 2.9. The KGs are too
  sparse to answer paper questions: coverage, not triple correctness, is the bottleneck.
- **⏳ IN FLIGHT: `screen -r qa0` / `qa1`** (started 2026-11-07 15:11, ~3–4 h): Q&A answers from the
  14 noont KGs (7 slugs × 2 ontologies), `run_qa.sh <gpu> kg …`, log `qa_driver.log`.
- **⛔ GPU settings are STILL CHANGED for heat** (university property): persistence on, 400 W,
  clock locks, root guard `screen gpuguard`. **Restore as soon as the Q&A is done:**
  ```
  sudo screen -S gpuguard -X quit
  sudo nvidia-smi -i 0 -rgc && sudo nvidia-smi -i 1 -rgc
  sudo nvidia-smi -pl 575 && sudo nvidia-smi -pm 0
  ```
  The sudo password is not stored; ask the user.
- Models in the HF cache: students Qwen3-4B/8B, Llama-3.1-8B-Instruct, Gemma-3-12B-it, Phi-4;
  teacher gpt-oss-20b; Q&A model Mistral-Small-24B-Instruct-2501; second teacher
  Qwen3-30B-A3B-Instruct-2507 (downloaded, not run yet).
- Teacher v2 tooling (`--prompt-variant recall`, `--entity-dir output/acl_v2`) is built and probed
  (hands_off §38.6), not run at scale. Given the Q&A result (coverage is the bottleneck) it is now
  directly relevant.

## Next steps, in order
1. When `qa0`/`qa1` finish: `python qa_eval.py grade --run acl-f0` then `report --run acl-f0
   --compare <noont-me50> <ft-me50> …`. Rerun `run_qa.sh <gpu> controls` / `kg …` once to fill the
   few unanswered questions (the report flags them). Write `results.md` Table 20 (Q&A). Restore
   the GPUs. Commit + push.
2. Analyse WHY the KGs answer so little: how many questions have an answer that is a node of the
   KG (`answer-in-KG`), how many concern a listed entity at all (fixed extraction cannot answer
   questions about unlisted subjects). This decides the next lever.
3. Raise coverage: the teacher-v2 recall fixes + v2 entity lists, a lower empty cap (me30 beats
   me50 on Q&A), and the second teacher (Qwen3-30B-A3B) — all judged by the same Q&A run.
4. Blind C1/C3/C4 diagnostic of the noont students (`blind_judge.py`), more folds, entity-list
   review (REVIEW rows need the user/professor), guard fixes (`88.5` leak, self-loops).

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

