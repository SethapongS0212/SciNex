"""Keep every SciNex run process out of Windows 11 EcoQoS, for as long as the chain runs.

Why this exists (hands_off §36.11): `run_ft_pipeline` opts out the PID it spawns, but on a venv
made by the Windows Python Install Manager that PID is a trampoline — the real interpreter
(`pythoncore-*\\python.exe`) is its child and stayed throttled. Measured 2026-10-24 on the
consensus training: **48 s/step throttled vs ~6 s/step opted out, an 8x difference**, with the GPU
at 0% util / 9.75 W because one parked CPU thread could not feed it.

`run_ft_pipeline.no_power_throttling_tree` now fixes this for FUTURE launches, but a pipeline that
is already running holds the old code in memory and will re-throttle each new step it spawns
(train → ft_gen → compare). This watchdog sweeps by command line every 60 s, so each new child is
opted out within a minute of appearing.

    python ft_runs/ecoqos_watchdog.py          # run detached; exits once the chain is gone
"""

import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from run_ft_pipeline import no_power_throttling          # noqa: E402

LOG = REPO / "ft_runs" / "ecoqos_watchdog.log"
MARKERS = ("chain_session35", "run_ft_pipeline", "ft_generate", "ft_train_qlora", "ft_compare")
IDLE_ROUNDS_BEFORE_EXIT = 5                              # ~5 min of no run processes


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_pids():
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            cl = " ".join(p.info["cmdline"] or ())
            if any(m in cl for m in MARKERS):
                out.append(p.info["pid"])
        except Exception:
            continue
    return out


def main():
    log("watchdog started")
    handled, idle = set(), 0
    while True:
        pids = run_pids()
        if pids:
            idle = 0
            fresh = [p for p in pids if p not in handled]
            for p in pids:                               # re-apply to all; it is idempotent
                no_power_throttling(p)
            handled = set(pids)
            if fresh:
                log(f"opted out {len(fresh)} new process(es): {sorted(fresh)} "
                    f"(tracking {len(pids)} total)")
        else:
            idle += 1
            if idle >= IDLE_ROUNDS_BEFORE_EXIT:
                log("no run processes for ~5 min — chain finished, watchdog exiting")
                return
        time.sleep(60)


if __name__ == "__main__":
    main()
