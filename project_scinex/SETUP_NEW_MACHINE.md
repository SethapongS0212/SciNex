# Moving SciNex to a new machine

Written 2026-10-20. Follow it top to bottom on the new laptop or VM. Then read
`HANDOFF_NEXT_SESSION.md` to pick up the work where it stopped.

The project lives in the public GitHub repo **https://github.com/SethapongS0212/SciNex**, inside the
folder **`project_scinex/`**. Run every command below from `project_scinex/`.

---

## 1. What is in GitHub, and what is NOT

The repo is **public**, so three kinds of files are deliberately kept out (see `.gitignore`):

| Not in GitHub | Why | How to get it on the new machine |
|---|---|---|
| API keys (`.env`, `citation/apikey.env`) | secrets | create `.env` from `.env.example` (§4) |
| Paper PDFs (`papers/*.pdf`) | 10 of the 20 are paywalled; publishers' copyright | private bundle, or `fetch_corpus.sh` + manual downloads (§5B) |
| Anything holding **full paper text**: parsed HTML `output/<id>/no-llm/`, `prompts.jsonl`, finetuning data `ft_data/*/train.jsonl` + `parts/`, C2/C5 judging bundles `ontology_eval/*/*/paragraphs/*_batch_*.txt` | copyright (full paragraphs of paywalled papers) | private bundle, or regenerate (§5B) |
| CS-NER gazetteer + per-paper entity lists (`output/acl/`) | large, regenerable | private bundle, or `enrich_entity_csv.py` (§5B) |
| Trained adapters (`ft_runs/*/adapter/`, `checkpoints/`) | 126 MB+, over GitHub's 100 MB file limit | private bundle (teacher-only Qwen3-4B), or retrain |
| Pipeline step markers (`ft_runs/_pipeline_*/*.done`) | machine-local; they would make the pipeline skip steps | nothing; they are recreated by running |
| Hugging Face model weights | huge | download automatically on first use (§6) |

**Everything else IS in GitHub:** all code, all docs, every model's stored extraction replies
(`output/<id>/kg/<extractor>/<model>/responses.jsonl`), the triples (`triples.json`), citation networks,
all judge verdicts (`ontology_eval/*/verdicts.csv`, `gold/`), the fold split `ft_data/split.json`,
data manifests and run manifests.

### The private bundle
`SciNex_private_bundle.zip` (made on the old laptop, 2026-10-20) holds the not-in-GitHub files from that
laptop: the 7 open-access PDFs, their parsed HTML, the gazetteer + entity CSVs, both finetuning
datasets, the trained teacher-only Qwen3-4B adapter, and the C2/C5 bundles. Its `BUNDLE_CONTENTS.txt`
lists every file. **Move it privately (USB drive or a private cloud folder). Never upload it to GitHub
or anywhere public.** The other 13 PDFs were never on that laptop. They are on the lab machine, or
get them by hand (§5B).

---

## 2. Install the basics

- **Git**: https://git-scm.com (Windows: `winget install --id Git.Git -e`).
- **Python 3.12** (the laptop used 3.12.10). Windows: python.org installer, tick "Add to PATH".
- An **NVIDIA driver** recent enough for your GPU (RTX 50-series needs a CUDA 12.8+ capable driver).

```
git clone https://github.com/SethapongS0212/SciNex.git
cd SciNex/project_scinex
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux: source .venv/bin/activate)
```

---

## 3. Install PyTorch FIRST, then everything else

PyTorch must match your GPU. Install it before the rest so nothing else can replace it.

| GPU | torch to install |
|---|---|
| **RTX 50-series (5070, 5090), Blackwell sm_120** | a **CUDA 12.8+ build** (pytorch.org → "CUDA 12.8" or newer). ⛔ cu124 builds have no sm_120 kernels and will not run |
| Older GPUs (e.g. the old H100 vGPU) | stable `torch 2.6.0+cu124` (`--index-url https://download.pytorch.org/whl/cu124`) |

The old laptop ran a nightly, `torch 2.12.0.dev20260408+cu128`. The commented commands at the
bottom of `requirement.txt` install exactly that, if a stable build gives you trouble.

Then install the rest **without letting pip touch torch**:
```
# Windows PowerShell
(Get-Content requirement.txt) -notmatch '^(torch|torchaudio|torchvision)==' | Set-Content req_no_torch.txt
# Linux:  grep -vE '^(torch|torchaudio|torchvision)==' requirement.txt > req_no_torch.txt
pip install -r req_no_torch.txt
pip install -r requirements_kg.txt -r requirements_ft.txt
python -c "import torch, bitsandbytes; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```
The last line must print the torch you installed (unchanged), `True`, and for RTX 50-series `(12, 0)`.
If pip replaced torch anyway, reinstall your torch and add the missing packages with `--no-deps`
(on the old laptop `peft 0.21.1` and `rdflib 7.6.0` were installed that way).

If the torch you end up with is a **dev/nightly** build, pass `--allow-dev-torch` to
`ft_train_qlora.py` / `run_ft_pipeline.py`. The trainer refuses nightlies otherwise, because one
crippled 4-bit speed on the old VM.

---

## 4. API keys

```
copy .env.example .env      # Linux: cp .env.example .env
```
Fill in only what you use: `SEMANTIC_SCHOLAR_API_KEY` (citation networks), `OPENROUTER_API_KEY`
(API extraction/judging). Both `.env` and `citation/apikey.env` are gitignored.

⚠ Use freshly issued keys. Any key that was ever committed or shared counts as leaked: revoke it and
make a new one. Removing a file from the repo does not remove it from the git history.

Claude is **never** called through an API in this project (decision of record; see
`HANDOFF_NEXT_SESSION.md`), so no Anthropic key is needed.

---

## 5. Restore the data

### 5A. With the private bundle (recommended)
Unzip `SciNex_private_bundle.zip` **into `project_scinex/`**. Its paths are relative to that folder,
so files land in `papers/`, `output/`, `ft_data/`, `ft_runs/` and `ontology_eval/`. Nothing in it
overwrites a tracked file.

### 5B. Without the bundle (regenerate)
1. **PDFs.** `bash fetch_corpus.sh` (Git Bash on Windows) downloads the 7 open-access PDFs. The rest
   come by hand, listed with DOIs in `papers/manifest.csv`. **Save each as `papers/<paper_id>.pdf`
   using the manifest's id**: the filename IS the paper id everywhere downstream. Nature PDFs need
   `curl.exe`, not Python's urllib.
2. **Parse.** `python main.py papers/<id>.pdf --no-llm` for each paper → `output/<id>/no-llm/`. It
   reuses the committed `citation_network.json` (no Semantic Scholar calls), which matters: without
   it, paragraphs won't match the teacher's stored replies. A paper with no citation network
   (gamlprop2025) is parsed with `--no-citations`, which writes `output/<id>/no-llm_no-citations/`.
   **Rename that folder to `no-llm`.** Check each parsed `Title:` against `papers/manifest.csv`.
   For title, section and body-word counts per PDF, `parse_check.py` takes the PDF paths:
   Linux `python3 parse_check.py papers/*.pdf`; Windows `python parse_check.py (Get-ChildItem papers\*.pdf).FullName`.
3. **Entity lists.** `python enrich_entity_csv.py --all --source csner` (needs internet the first
   time; builds `output/acl/csner_gazetteer.csv`, ~51k entities).
4. **Finetuning data.**
   ```
   python build_ft_data.py --eval-fold 0
   python build_ft_data.py --eval-fold 0 --consensus-models gemma4-31b qwen3-235b ministral-14b
   ```
   Every training paper × ontology should report `ok`. Never edit or regenerate `ft_data/split.json`.
   It fixes which papers are held out.
5. **C2/C5 bundles** are only needed to re-judge C2/C5. The verdicts are already in GitHub.

---

## 6. Hugging Face models

Downloaded automatically on first use: `Qwen/Qwen3-4B-Instruct-2507` (8 GB laptops) and `Qwen/Qwen3-8B`
(the RTX 5090 VM run). On Windows, if a download fails with `WinError 1314` (symlink privilege):
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', max_workers=1)"
```
Set `HF_TOKEN` only for gated models.

---

## 7. Check the install (no GPU needed, ~1 min)

```
python -c "import ft_common, build_ft_data, ft_compare, blind_judge, qa_eval, run_ft_pipeline, ontology_eval; print('imports ok')"
python ft_compare.py --slugs qwen3-4b-base-hf gptoss-120b --papers strabismus2026 osmotic2026
python build_ft_data.py --eval-fold 0 --dry-run
```
`ft_compare` works straight from the stored replies in GitHub. Expected CEO row: `qwen3-4b-base-hf`
**493 proposed → 172 kept (34.9%)**, `gptoss-120b` **107 → 91 (85.0%)**. Scinex: 464 → 164 (35.3%) and
70 → 57 (81.4%). Different numbers mean a different code version or a broken install. The dry run
reports how many extraction examples it could build: 0 if no parsed HTML is present (§5).

Windows: set `PYTHONIOENCODING=utf-8` (once per terminal: `$env:PYTHONIOENCODING="utf-8"`), or some
scripts crash printing `✔`/`→`. Use `python` on Windows and `python3` on Linux.

---

## 8. Continue the work

1. Read **`HANDOFF_NEXT_SESSION.md`** (current state, decisions, next steps), then `hands_off.md` §34.
2. GPU runs on the **RTX 5090 VM**: exact commands in `hands_off.md` §34.12. On Linux run them
   inside `tmux` or `nohup` so an SSH drop doesn't kill training.
3. On a **Windows** machine, long runs should be launched hidden so closing a window can't kill them.
   `run_ft_pipeline.py` already stops Windows from throttling them. Example (PowerShell, from `project_scinex/`):
   ```powershell
   $repo = (Get-Location).Path
   $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
     CommandLine = "cmd.exe /c `"$repo\ft_runs\<your launcher>.cmd`""; CurrentDirectory = $repo
     ProcessStartupInformation = $si }
   ```
   The launcher `.cmd` files are gitignored (they hold local paths). A launcher is just this, saved
   as e.g. `ft_runs\launch_qwen3-8b-f0.cmd` with your own paths:
   ```bat
   @echo off
   cd /d "C:\path\to\SciNex\project_scinex"
   set "PYTHONIOENCODING=utf-8"
   set "PYTHONUNBUFFERED=1"
   set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
   rem stops numpy/scipy's Fortran runtime aborting the run on a console close event
   set "FOR_DISABLE_CONSOLE_CTRL_HANDLER=1"
   "C:\path\to\SciNex\project_scinex\.venv\Scripts\python.exe" run_ft_pipeline.py --base-model Qwen/Qwen3-8B --prefix qwen3-8b --fold 0 --papers csysguard2024 linkpred2015 osmotic2026 strabismus2026 textaug2023 --steps base_gen train ft_gen compare > ft_runs\pipeline_qwen3-8b-f0.stdout.log 2>&1
   ```
   Progress: `ft_runs\_pipeline_<run>\pipeline.log`. Stop a run by killing its Python processes by
   command line (`Get-CimInstance Win32_Process`). Stopping a parent shell does not kill the Python child.
4. **Keep GitHub current:** `git pull` before you start, and after a session:
   ```
   git add -A
   git status            # make sure no PDF, .env or full-text file appears
   git commit -m "<what changed>"
   git push
   ```
   `.gitignore` already blocks the dangerous files, but always look at `git status` before pushing:
   this repo is public.
5. Docs discipline (standing rule): after any change, update `hands_off.md`, `Claude.md` and
   `results.md` in the same session.
