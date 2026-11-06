# Handoff — next session

**Updated 2026-11-06 (Session 38).** Read in this order:
1. **[`OBJECTIVE.md`](OBJECTIVE.md)** — the professor's three steps. Everything serves them.
2. This file — where the work stands and what to do next.
3. `hands_off.md` §38 for detail; `results.md` for tables.

## ⛔ The objective in four lines (full text: `OBJECTIVE.md`)
- Paper → KG; **finetune an LLM so it understands the ontology + vocabulary**; **evaluate the KG
  with simulated Q&A against the real papers** (the MAIN evaluation).
- The finetuned extractor runs **without the ontology in its prompt** (`--prompt-variant noont`):
  entity list + text only. Teacher → student distillation only makes the training data.
- Corpus: 805 ACL papers, fixed entity lists; fold 0 (41 papers) held out.
- Per-triple judging (C1/C3/C4) and judge-free counts are diagnostics, not the headline.

## State (2026-11-06, ~02:00 server time)
- **⏳ IN FLIGHT: `screen -r noont`** — `/home/dev1/scinex_runs/run_noont.sh` (copy in
  `ft_runs/acl_scripts/`), log `/home/dev1/scinex_runs/noont_driver.log`. Ontology-free students,
  data `ft_data/acl_noont_me{50,30}` (identical to `acl_maxempty{50,30}` except the extraction
  system prompt: 218 tokens with no ontology vs 2,744):

  | GPU | runs (train → generate on the 41 papers, both ontologies) |
  |---|---|
  | 0 | `qwen3-8b-acl-noont-me50-cv` → `qwen3-8b-acl-base-noont` (untuned control) → `llama31-8b-acl-noont-me50-cv` → `qwen3-4b-acl-noont-me50-cv` |
  | 1 | `qwen3-8b-acl-noont-me30-cv` → `gemma3-12b-acl-noont-me50-cv` → `phi4-acl-noont-me50-cv` |

  Then `ft_compare` → `ft_runs/noont/compare_noont.json`, log `noont_compare.log`. Every step
  skips if done, so re-running the script resumes. Smoke-tested for all four model families
  (Gemma-3 needed `token_type_ids` and a LoRA regex that skips its vision tower; Phi-4 needs
  `--target-modules qkv_proj o_proj gate_up_proj down_proj`; both are in the script).
- **⛔ GPU settings are CHANGED for heat while it runs** (university property — restore after):
  persistence on, 400 W, clocks GPU0 1650 / GPU1 1900 MHz, root guard `screen gpuguard`.
  Restore when no GPU job is left:
  ```
  sudo screen -S gpuguard -X quit
  sudo nvidia-smi -i 0 -rgc && sudo nvidia-smi -i 1 -rgc
  sudo nvidia-smi -pl 575 && sudo nvidia-smi -pm 0
  ```
  The sudo password is not stored; ask the user.
- **Downloaded models** (HF cache): students Qwen3-4B, Qwen3-8B, Llama-3.1-8B-Instruct,
  Gemma-3-12B-it, Phi-4; teacher gpt-oss-20b; **Q&A model Mistral-Small-24B-Instruct-2501**
  (chosen because it is neither a teacher nor a student family); **second teacher
  Qwen3-30B-A3B-Instruct-2507** (download started 2026-11-06).
- Earlier results (ontology-IN-prompt students, Tables 15–18): the me50 student matches the
  teacher's coverage and is at least as correct (C4 58/71 vs 45/55). They are the comparison
  baseline for the noont students, not the objective.
- Teacher v2 tooling exists (`--prompt-variant recall`, `--entity-dir output/acl_v2` from
  `build_entity_v2.py`): a 30-paragraph probe showed the new entity lists recover some missed
  triples; the prompt alone did little; reasoning "medium" made the teacher MORE cautious. Its
  41-paper test run was stopped by the user (2026-11-06) in favour of the objective; its partial
  output was deleted. Re-run with `/home/dev1/scinex_runs/run_teacher_v2_test.sh low` if wanted.

## Next steps, in order
1. **When `noont run done` appears:** read `noont_compare.log`. Key question: do the noont
   students keep the ontology-in-prompt students' guard-pass rate and coverage? The untuned
   `qwen3-8b-acl-base-noont` is the control (it cannot know the relations).
2. **Simulated Q&A — the MAIN evaluation** (`qa_eval.py`, provider `hf`, model
   `mistralai/Mistral-Small-24B-Instruct-2501`; never Claude, never a teacher as question writer):
   `generate` questions from the 41 eval papers once, then `answer --context kg` for every KG
   (teacher, base, me50/me30 with ontology, every noont student; `fixed/<slug>` and
   `fixed_scinex/<slug>`), plus `oracle` and `closedbook`; `grade` (deterministic overlap);
   `report --compare ...`. One answerer for every condition.
3. Blind C1/C3/C4 diagnostic of the best noont students vs the ontology-in-prompt me50 and the
   teacher (`blind_judge.py`, new run name).
4. **Second teacher (user request, 2026-11-06):** Qwen3-30B-A3B-Instruct-2507 as teacher 2 on a
   fixed subset of TRAINING papers, and a gpt-oss-20b student on the SAME subset, so only the
   teacher differs; then noont students from each, compared by Q&A. Multi-teacher distillation
   is a normal, published technique; here it answers "does the teacher change what the student
   learns?". ⚠ Qwen teacher → Qwen student is a family confound: also train a non-Qwen student.
5. Then: more folds; entity-list cleanup decision (REVIEW rows of
   `entity_cleanup/stoplist_draft.csv` need the user/professor); guard fixes (`88.5` leak,
   self-loops, objects absent from the sentence).

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

