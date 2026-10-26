#!/usr/bin/env python3
"""
run_ft_pipeline.py
------------------
Unattended finetune-and-evaluate run for ONE fold — one GPU process at a time, in order:

  base_gen   base model through the extraction path on the held-out papers  (+ ingest)
  train      QLoRA on ft_data/fold<k>                                       (ft_train_qlora.py)
  ft_gen     base + adapter on the same held-out papers                     (+ ingest)
  compare    judge-free base vs finetuned vs teacher table                  (ft_compare.py)
  qa         Q&A evaluation: generate -> answer (kg x3, oracle, closedbook) -> grade -> report

Each finished step drops ft_runs/_pipeline_<run>/<step>.done, so re-running the same command
after a crash or reboot resumes where it stopped. Everything is logged with timestamps to
ft_runs/_pipeline_<run>/pipeline.log.

    python3 run_ft_pipeline.py --base-model Qwen/Qwen3-4B-Instruct-2507 --prefix qwen3-4b \
            --fold 0 --papers strabismus2026 osmotic2026 --allow-dev-torch

A data VARIANT of the same fold (e.g. the consensus-augmented targets, hands_off §34) gets
`--tag`: run ft_runs/<prefix>-f<k>-<tag>, slug <prefix>-ft-<tag>-cv. The base slug is shared, so
a variant reuses the untagged pipeline's base extraction. `--also-slugs` adds other finished
KGs (e.g. the untagged finetune) to compare and qa, so every model faces ONE question set:

    python3 run_ft_pipeline.py ... --tag cons --data ft_data/consensus/fold0 \
            --also-slugs qwen3-4b-ft-cv

The Q&A step defaults to the base model itself as generator/answerer/grader so the run needs
no API key. That makes it a PIPELINE TEST, not a result: the generator shares the student's
family. For numbers worth reporting, rerun `qa` with an independent model (hands_off §32.6).
"""

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PY = sys.executable


def no_power_throttling(pid=None):
    """Opt a process out of Windows 11 EcoQoS. Windows throttles windowless background
    processes onto efficiency cores at low clock; training is bound by ONE CPU thread feeding
    the GPU, so a hidden launch ran at ~40 s/step vs ~9 s/step in a console (hands_off §34.5)."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.GetCurrentProcess() if pid is None else k32.OpenProcess(0x0200 | 0x1000, False, pid)
    if not h:
        return False
    s = State(1, 0x1 | 0x4, 0)     # take control of execution speed + timer resolution; both OFF
    ok = bool(k32.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)))  # ProcessPowerThrottling
    if pid is not None:
        k32.CloseHandle(h)
    return ok


def no_power_throttling_tree(pid, passes=3, settle=6.0):
    """Opt out `pid` AND every descendant, repeatedly, because the PID we spawn is often not
    the process that does the work.

    ⛔ Measured 2026-10-24 (hands_off §36.11): on a venv created by the Windows **Python Install
    Manager**, `.venv\\Scripts\\python.exe` is a trampoline that re-execs the real interpreter
    (`pythoncore-3.12-64\\python.exe`) as a CHILD. `no_power_throttling(proc.pid)` therefore opted
    out a launcher that does no work, while the trainer underneath stayed parked on the efficiency
    cores: **48 s/step vs 6 s/step once the real PID was opted out — an 8x difference** — with the
    GPU at 0% util and 9.75 W, starved by one throttled CPU thread. The 14 h `base_gen` and the
    315.9 min training of the teacher-only run were both throttled this way.

    Children appear a moment after launch, so sweep a few times. Uses psutil (pinned in
    requirement.txt) and falls back to a WMI query."""
    if sys.platform != "win32":
        return
    seen = set()
    for i in range(passes):
        for target in [pid] + _descendants(pid):
            if target not in seen:
                no_power_throttling(target)
                seen.add(target)
        if i < passes - 1:
            time.sleep(settle)
    return sorted(seen)


def _descendants(pid):
    try:
        import psutil
        return [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except Exception:
        pass
    try:                                    # no psutil: ask WMI for the parent/child table
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId |"
             " ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=60).stdout
        kids, found, frontier = {}, [], [pid]
        for line in out.splitlines():
            bits = line.split()
            if len(bits) == 2 and bits[0].isdigit() and bits[1].isdigit():
                kids.setdefault(int(bits[1]), []).append(int(bits[0]))
        while frontier:
            p = frontier.pop()
            for c in kids.get(p, []):
                if c not in found:
                    found.append(c); frontier.append(c)
        return found
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--prefix", required=True, help="slug prefix, e.g. qwen3-4b")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--papers", nargs="+", required=True,
                    help="held-out papers to generate on (must be in the fold's eval_papers.txt)")
    ap.add_argument("--teacher", default="gptoss-120b")
    ap.add_argument("--steps", nargs="+", default=["base_gen", "train", "ft_gen", "compare", "qa"])
    ap.add_argument("--train-args", default="", help="extra args for ft_train_qlora.py, quoted")
    ap.add_argument("--gen-args", default="", help="extra args for ft_generate.py, quoted")
    ap.add_argument("--qa-provider", default="hf")
    ap.add_argument("--qa-model", default=None, help="default: --base-model (pipeline test only)")
    ap.add_argument("--qa-base-url", default=None)
    ap.add_argument("--allow-dev-torch", action="store_true")
    ap.add_argument("--data", default=None, help="data dir (default ft_data/fold<k>)")
    ap.add_argument("--tag", default="", help="data-variant tag; see the module docstring")
    ap.add_argument("--also-slugs", nargs="*", default=[],
                    help="extra finished KG slugs to include in compare and qa")
    args = ap.parse_args()

    run0 = f"{args.prefix}-f{args.fold}"
    run = f"{run0}-{args.tag}" if args.tag else run0
    work = Path("ft_runs") / f"_pipeline_{run}"
    work.mkdir(parents=True, exist_ok=True)
    shared = Path("ft_runs") / f"_pipeline_{run0}"   # base_gen is identical across variants
    log = work / "pipeline.log"
    base_slug = f"{args.prefix}-base-hf"
    ft_slug = f"{args.prefix}-ft-{args.tag}-cv" if args.tag else f"{args.prefix}-ft-cv"
    data = args.data or f"ft_data/fold{args.fold}"
    eval_papers = set(Path(data, "eval_papers.txt").read_text(encoding="utf-8").split())
    leak = [p for p in args.papers if p not in eval_papers]
    if leak:
        sys.exit(f"{leak} are not held out in fold {args.fold} — refusing")

    def say(msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def sh(cmd):
        say("$ " + " ".join(cmd))
        with open(log, "a", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            # the TREE, not just proc.pid: a Python-Install-Manager venv re-execs the real
            # interpreter as a child, and opting out only the launcher left training 8x slow
            # (§36.11). Runs in a thread so the sweeps do not delay the step.
            threading.Thread(target=no_power_throttling_tree, args=(proc.pid,), daemon=True).start()
            rc = proc.wait()
        if rc:
            say(f"FAILED (exit {rc}) — see {log}")
            sys.exit(rc)

    qa_model = args.qa_model or args.base_model
    qa_api = ["--provider", args.qa_provider, "--model", qa_model]
    if args.qa_base_url:
        qa_api += ["--base-url", args.qa_base_url]
    qa_run = f"{run}-smoke" if not args.qa_model else run
    slugs = [base_slug, ft_slug, *args.also_slugs, args.teacher]
    kgs = [f"relation/{s}" for s in slugs]
    qa_compare = [x for other in [base_slug, *args.also_slugs, args.teacher]
                  for x in ("--compare", f"kg__relation__{ft_slug}", f"kg__relation__{other}")]

    steps = {
        "base_gen": [[PY, "ft_generate.py", "--base-model", args.base_model, "--model-slug", base_slug,
                      "--papers", *args.papers, "--ingest", *args.gen_args.split()]],
        "train": [[PY, "ft_train_qlora.py", "--data", data, "--base-model", args.base_model,
                   "--run-name", run, *(["--allow-dev-torch"] if args.allow_dev_torch else []),
                   *args.train_args.split()]],
        "ft_gen": [[PY, "ft_generate.py", "--adapter", f"ft_runs/{run}", "--model-slug", ft_slug,
                    "--papers", *args.papers, "--ingest", *args.gen_args.split()]],
        "compare": [[PY, "ft_compare.py", "--slugs", *slugs,
                     "--papers", *args.papers, "--out", f"ft_runs/{run}/compare.json"]],
        "qa": [
            [PY, "qa_eval.py", "generate", "--run", qa_run, "--papers", *args.papers, *qa_api],
            [PY, "qa_eval.py", "answer", "--run", qa_run, "--context", "kg", "--kg", *kgs, *qa_api],
            [PY, "qa_eval.py", "answer", "--run", qa_run, "--context", "oracle", *qa_api],
            [PY, "qa_eval.py", "answer", "--run", qa_run, "--context", "closedbook", *qa_api],
            [PY, "qa_eval.py", "grade", "--run", qa_run, *qa_api],
            [PY, "qa_eval.py", "report", "--run", qa_run, *qa_compare],
        ],
    }
    if sys.platform == "win32":
        # keep a laptop from sleeping mid-run; released automatically when this process exits
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        no_power_throttling()
    say(f"=== pipeline {run}: steps {args.steps}, papers {args.papers}, QA model {qa_model} ===")
    t0 = time.time()
    for name in args.steps:
        done = (shared if name == "base_gen" else work) / f"{name}.done"
        if done.exists():
            say(f"skip {name} (done)")
            continue
        t = time.time()
        say(f"--- {name} ---")
        for cmd in steps[name]:
            sh(cmd)
        done.write_text(datetime.now().isoformat(), encoding="utf-8")
        say(f"{name} finished in {(time.time() - t) / 60:.1f} min")
    say(f"=== all done in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
