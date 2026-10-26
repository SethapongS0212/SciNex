"""Session 35 unattended chain (machine-local; RTX 4060 Laptop 8 GB, sm_89).

Written 2026-10-20 because the user left while the two large downloads were still running on a
~400 KB/s link. Launch it DETACHED (hidden, via WMI) so it outlives the assistant's session:
see SETUP_NEW_MACHINE.md section 8.3 and the command in hands_off.md section 35.10.

It takes ownership of everything that was still in flight and refuses to cascade past a failure:

  A  hold a Windows keep-awake request + opt out of EcoQoS (hands_off section 34.5)
  B  make sure torch is importable (retry `pip install torch` only if no pip is still working)
  C  install the minimal training/generation deps (pinned; nothing that would touch torch)
  D  verify torch is unchanged, CUDA is up and the capability is (8, 9)
  E  finish the Qwen3-1.7B snapshot (curl -C - resume, byte-exact size check per file)
  F  SMOKE: 2 paragraphs through ft_generate  -> proves 4-bit load + decode + VRAM headroom
  G  SMOKE: 2 optimizer steps through ft_train_qlora -> proves the training path
  H  full teacher-only pipeline  (base_gen -> train -> ft_gen -> compare)
  I  full consensus pipeline     (train -> ft_gen -> compare, base_gen shared)

Every step appends to ft_runs/chain_session35.log. A step that fails writes "STEP <x> FAILED" and
the chain stops there, leaving the earlier work intact (run_ft_pipeline .done markers make a
re-launch resume rather than redo).
"""

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\Users\cei008\Downloads\SciNex\project_scinex")
MODEL = Path(r"C:\Users\cei008\hf_local\Qwen3-1.7B")
LOG = REPO / "ft_runs" / "chain_session35.log"
PY = sys.executable
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
HF_BASE = "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main"

# Exact Content-Length of every file in the snapshot, read from the HF HEAD responses on
# 2026-10-20. A short file means the download was cut off, not that the repo changed.
EXPECTED = {
    "config.json": 726,
    "generation_config.json": 239,
    "tokenizer_config.json": 9732,
    "vocab.json": 2776833,
    "merges.txt": 1671853,
    "model.safetensors.index.json": 25605,
    "tokenizer.json": 11422654,
    "model-00001-of-00002.safetensors": 3441185608,
    "model-00002-of-00002.safetensors": 622329984,
}

# Pinned to requirement.txt where it pins them; peft/rdflib are the two the old laptop added by
# hand. Deliberately NOT installed: pdfplumber/pymupdf/camelot/opencv/pandas/scipy/sklearn — the
# papers are already parsed, and nothing on the training or generation path imports them.
DEPS = [
    "transformers==5.5.4", "tokenizers==0.22.2", "safetensors==0.7.0",
    "huggingface_hub==1.10.2", "hf-xet==1.4.3", "accelerate==1.13.0",
    "bitsandbytes==0.49.2", "peft==0.21.1", "rdflib==7.6.0",
    "networkx==3.4.2", "beautifulsoup4", "jinja2", "regex", "tqdm", "psutil==7.2.2",
    "numpy==2.4.4",
]

PAPERS = ["csysguard2024", "linkpred2015", "osmotic2026", "strabismus2026", "textaug2023"]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def fail(step, detail=""):
    log(f"STEP {step} FAILED {detail}".rstrip())
    log("chain stopped. Earlier steps are intact; re-launching resumes (.done markers).")
    sys.exit(1)


def child_env():
    env = dict(os.environ)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1", HF_HUB_OFFLINE="1",
               FOR_DISABLE_CONSOLE_CTRL_HANDLER="1")
    # The side directory used while torch was downloading must not shadow the venv's packages.
    env.pop("PYTHONPATH", None)
    return env


def run(cmd, step, timeout=None, check=True):
    log(f"  $ {' '.join(str(c) for c in cmd)}")
    try:
        p = subprocess.run(cmd, cwd=REPO, env=child_env(), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           errors="replace")
    except subprocess.TimeoutExpired:
        fail(step, f"(timed out after {timeout}s)")
        return None
    tail = "\n".join((p.stdout or "").strip().splitlines()[-25:])
    if tail:
        log("  | " + tail.replace("\n", "\n  | "))
    if check and p.returncode != 0:
        fail(step, f"(exit {p.returncode})")
    return p


# ── A. keep the machine awake and off the efficiency cores ──────────────────────────────────
def step_a():
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS|SYSTEM
    sys.path.insert(0, str(REPO))
    try:
        from run_ft_pipeline import no_power_throttling
        no_power_throttling()
        log("A ok: keep-awake held, EcoQoS opt-out applied to this process")
    except Exception as exc:                                    # not fatal, only slower
        log(f"A partial: keep-awake held, EcoQoS opt-out unavailable ({exc})")


# ── B. torch ────────────────────────────────────────────────────────────────────────────────
def torch_ready():
    p = subprocess.run([PY, "-c", "import torch; print(torch.__version__)"],
                       cwd=REPO, env=child_env(), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    return (p.stdout or "").strip() if p.returncode == 0 else None


def pip_busy():
    """True while some other python is still running the torch install, so the chain waits
    instead of starting a second 2.75 GB download on the same link."""
    p = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |"
                        " Where-Object { $_.CommandLine -like '*pip*install*torch*' -and"
                        "   $_.CreationDate -gt (Get-Date).AddMinutes(-90) } |"
                        " Measure-Object).Count"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        return int((p.stdout or "0").strip()) > 0
    except ValueError:
        return False


def step_b(stall_minutes=120, attempts=3):
    """Wait until torch imports. NOTE (2026-10-23): this used to carry an absolute 8 h
    deadline checked BEFORE the readiness probe, so when the laptop slept from 14:39 on
    10-20 to 15:25 on 10-23 the chain failed step B even though torch had finished
    installing at ~14:38. Now: probe first, and the give-up timer counts from the last
    observed progress (a growing wheel or a live pip), so sleeping only pauses the chain."""
    tried = 0
    last_progress = time.time()
    while True:
        ver = torch_ready()                      # readiness FIRST, always
        if ver:
            log(f"B ok: torch {ver} importable")
            return
        if pip_busy():
            log("B waiting: another pip is still installing torch")
            last_progress = time.time()          # someone is working -> not stalled
        elif tried < attempts:
            tried += 1
            log(f"B no torch and no live pip -> installing it myself (attempt {tried}/{attempts})")
            run([PY, "-m", "pip", "install", "torch", "--index-url", TORCH_INDEX],
                "B", timeout=8 * 3600, check=False)
            last_progress = time.time()
        elif time.time() - last_progress > stall_minutes * 60:
            fail("B", f"(no torch and no progress for {stall_minutes} min after {attempts} attempts)")
        time.sleep(120)


# ── C/D. the rest of the deps, then verify the GPU stack ────────────────────────────────────
def step_c():
    for attempt in (1, 2, 3):
        p = run([PY, "-m", "pip", "install", *DEPS], "C", timeout=3 * 3600, check=False)
        if p and p.returncode == 0:
            log("C ok: training/generation deps installed")
            return
        log(f"C attempt {attempt} failed; retrying in 3 min")
        time.sleep(180)
    fail("C", "(pip install kept failing)")


def step_d():
    code = ("import torch, bitsandbytes, peft, transformers as tf;"
            "cap=torch.cuda.get_device_capability();"
            "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),"
            "'| cap', cap, '| bnb', bitsandbytes.__version__, '| peft', peft.__version__,"
            "'| tf', tf.__version__);"
            "assert torch.cuda.is_available(), 'no CUDA';"
            "assert cap == (8, 9), f'expected sm_89, got {cap}';"
            "assert 'dev' not in torch.__version__, 'torch is a nightly; this machine wants stable'")
    run([PY, "-c", code], "D", timeout=900)
    log("D ok: stable torch on sm_89, bitsandbytes + peft import")


# ── E. finish the model snapshot ────────────────────────────────────────────────────────────
# Runs in a background thread from the very start, so the weights download WHILE pip is still
# fetching torch — both are network-bound and the GPU is idle until step F either way. This
# thread is the only writer of the snapshot; nothing else may curl into MODEL concurrently or
# two `curl -C -` would interleave into one file.
_snapshot_done = False


def short_files():
    out = []
    for name, size in EXPECTED.items():
        f = MODEL / name
        have = f.stat().st_size if f.exists() else 0
        if have != size:
            out.append((name, have, size))
    return out


def snapshot_worker(attempts=200):
    global _snapshot_done
    MODEL.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        missing = short_files()
        if not missing:
            _snapshot_done = True
            log(f"E ok: snapshot complete, {sum(EXPECTED.values())/2**30:.2f} GiB, every file byte-exact")
            return
        if attempt == 1 or attempt % 10 == 0:
            log(f"E downloading ({attempt}): "
                + ", ".join(f"{n} {h/2**20:.0f}/{s/2**20:.0f} MB" for n, h, s in missing))
        for name, _have, _size in missing:
            subprocess.run(["curl.exe", "-sL", "-C", "-", "-o", str(MODEL / name),
                            f"{HF_BASE}/{name}"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        time.sleep(15)
    fail("E", "(snapshot still incomplete after many attempts)")


def wait_for_snapshot(stall_minutes=90):
    """Block until every file is byte-exact. Give-up timer counts from the last byte
    downloaded, not from the start, so a laptop sleep pauses rather than fails (see step_b)."""
    last_total, last_progress = -1, time.time()
    while True:
        missing = short_files()
        if not missing:
            log("E ok: snapshot complete, every file byte-exact")
            return
        total = sum(h for _n, h, _s in missing)
        if total != last_total:
            last_total, last_progress = total, time.time()
        elif time.time() - last_progress > stall_minutes * 60:
            fail("E", f"(snapshot stalled for {stall_minutes} min)")
        log("E waiting: " + ", ".join(f"{n} {h/2**20:.0f}/{s/2**20:.0f} MB" for n, h, s in missing))
        time.sleep(120)


# ── F/G. smoke tests — cheap proof before committing to many hours ──────────────────────────
SMOKE_MARK = REPO / "ft_runs" / "_pipeline_qwen3-1.7b-f0" / "chain_smoke.done"


def smoke_passed():
    return SMOKE_MARK.exists()


def mark_smoke():
    SMOKE_MARK.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_MARK.write_text(f"F+G passed {datetime.now():%Y-%m-%d %H:%M:%S}", encoding="utf-8")


def step_f():
    if smoke_passed():
        log("F skip: smokes already passed on this machine (chain_smoke.done)")
        return
    slug = "qwen3-1.7b-smoke"
    out = REPO / "output" / "osmotic2026" / "kg" / "relation" / slug
    run([PY, "ft_generate.py", "--base-model", str(MODEL), "--model-slug", slug,
         "--papers", "osmotic2026", "--ontologies", "ceo", "--limit", "2",
         "--batch-size", "2", "--no-resume"], "F", timeout=3600)
    replies = out / "responses.jsonl"
    n = sum(1 for _ in replies.open(encoding="utf-8")) if replies.exists() else 0
    if n < 2:
        fail("F", f"(expected 2 replies, got {n})")
    log(f"F ok: {n} paragraphs generated through the real extraction path")
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    log("F cleanup: throwaway smoke slug removed")


def step_g():
    if smoke_passed():
        log("G skip: smokes already passed on this machine (chain_smoke.done)")
        return
    run([PY, "ft_train_qlora.py", "--data", "ft_data/fold0", "--base-model", str(MODEL),
         "--run-name", "qwen3-1.7b-smoke", "--max-steps", "2"], "G", timeout=7200)
    adapter = REPO / "ft_runs" / "qwen3-1.7b-smoke" / "adapter"
    if not adapter.exists():
        fail("G", "(no adapter written)")
    log("G ok: 2 optimizer steps ran and an adapter was written")
    import shutil
    shutil.rmtree(REPO / "ft_runs" / "qwen3-1.7b-smoke", ignore_errors=True)
    log("G cleanup: smoke run directory removed")
    mark_smoke()


# ── H/I. the real runs ──────────────────────────────────────────────────────────────────────
def pipeline(tag_args, step, label):
    cmd = [PY, "run_ft_pipeline.py", "--base-model", str(MODEL), "--prefix", "qwen3-1.7b",
           "--fold", "0", "--papers", *PAPERS, "--gen-args=--batch-size=4", *tag_args]
    log(f"{step} starting: {label} (this is the long one; watch ft_runs/_pipeline_*/pipeline.log)")
    run(cmd, step, timeout=48 * 3600)
    log(f"{step} ok: {label} finished")


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 100)
    log("Session 35 chain starting (student Qwen/Qwen3-1.7B, fold 0, RTX 4060 Laptop 8 GB sm_89)")
    step_a()
    threading.Thread(target=snapshot_worker, daemon=True).start()   # E, in parallel with B/C/D
    step_b()
    step_c()
    step_d()
    wait_for_snapshot()
    step_f()
    step_g()
    pipeline(["--steps", "base_gen", "train", "ft_gen", "compare"], "H", "teacher-only variant")
    pipeline(["--tag", "cons", "--data", "ft_data/consensus/fold0",
              "--also-slugs", "qwen3-1.7b-ft-cv",
              "--steps", "train", "ft_gen", "compare"], "I", "consensus variant")
    log("CHAIN COMPLETE — both variants trained, generated and compared.")
    log("Next: blind judge census (blind_judge.py, hands_off section 34.8) with slugs "
        "qwen3-1.7b-base-hf qwen3-1.7b-ft-cv qwen3-1.7b-ft-cons-cv gptoss-120b")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        import traceback
        log(f"CHAIN CRASHED: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise
