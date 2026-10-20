#!/usr/bin/env python3
"""
Wait for a running pipeline process to exit, keeping it AND its descendants out of Windows
EcoQoS meanwhile, then run the next command (one GPU process at a time on the 8 GB laptop).

Written for a runner started before run_ft_pipeline.py learnt to un-throttle its children
(hands_off §34.5): its later steps (ft_gen, compare, qa) would otherwise start throttled.

    python ft_runs/wait_then_run.py --pid 16912 --log ft_runs/chain.log -- python run_ft_pipeline.py ...
"""
import argparse
import csv
import ctypes
import io
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_ft_pipeline import no_power_throttling  # noqa: E402

k32 = ctypes.WinDLL("kernel32")
k32.OpenProcess.restype = wintypes.HANDLE
k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
k32.CloseHandle.argtypes = [wintypes.HANDLE]


def alive(pid):
    h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = wintypes.DWORD()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259            # STILL_ACTIVE


def descendants(root):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId "
                          "| ConvertTo-Csv -NoTypeInformation"],
                         capture_output=True, text=True).stdout
    kids = {}
    for r in csv.DictReader(io.StringIO(out)):
        kids.setdefault(int(r["ParentProcessId"]), []).append(int(r["ProcessId"]))
    found, todo = set(), [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in found:
                found.add(c)
                todo.append(c)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd

    def say(msg):
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")

    no_power_throttling()
    say(f"waiting for pid {args.pid}; then: {' '.join(cmd)}")
    done = set()
    while alive(args.pid):
        for pid in ({args.pid} | descendants(args.pid)) - done:
            if no_power_throttling(pid):
                say(f"un-throttled pid {pid}")
            done.add(pid)
        time.sleep(args.poll)
    say(f"pid {args.pid} exited; starting next command")
    with open(args.log, "a", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        no_power_throttling(proc.pid)
        rc = proc.wait()
    say(f"next command exited {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
