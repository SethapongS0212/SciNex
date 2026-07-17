"""
run_acl_batch.py
----------------
Runs main.py --no-llm on every PDF in output/acl/pdfs/ that doesn't
already have both:
  - output/<paper_id>/no-llm/output.html
  - output/<paper_id>/citation_network.json

Captures per-paper success/failure and writes a summary so you can see
exactly what's broken and why.

Usage:
    python run_acl_batch.py                          # all unprocessed PDFs
    python run_acl_batch.py --rerun-failed           # retry previously failed
    python run_acl_batch.py --paper C16-1036         # single paper
    python run_acl_batch.py --pdf-dir output/acl/pdfs --out-dir output
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def load_previous_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {}

    raw = log_path.read_text().strip()
    if not raw:
        print(f"  [warn] {log_path} is empty; starting with a fresh batch log")
        return {}

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [warn] Could not parse {log_path}: {exc}; starting with a fresh batch log")
        return {}

    return {r["paper_id"]: r for r in rows if "paper_id" in r}


def already_complete(paper_id: str, out_dir: Path) -> bool:
    return (
        (out_dir / paper_id / "no-llm" / "output.html").exists()
        and (out_dir / paper_id / "citation_network.json").exists()
    )


def run_paper(pdf_path: Path, out_dir: Path, extra_args: list[str],
              timeout: int = 900) -> dict:
    cmd = [sys.executable, "main.py", str(pdf_path), "--no-llm"] + extra_args
    paper_id = pdf_path.stem
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # A single slow paper (usually stuck in Semantic Scholar 429 backoff)
        # must NOT abort the whole batch. Record it as failed and move on;
        # it can be retried later with --rerun-failed.
        elapsed = time.time() - start
        return {
            "paper_id": paper_id,
            "success":  False,
            "elapsed":  round(elapsed, 1),
            "error":    f"timeout after {timeout}s (likely citation rate-limit)",
            "stdout":   (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
            "stderr":   (e.stderr or "")[-1000:] if isinstance(e.stderr, str) else "",
        }
    elapsed = time.time() - start

    success  = result.returncode == 0 and already_complete(paper_id, out_dir)

    # Extract error line from output
    error_line = ""
    for line in (result.stdout + result.stderr).splitlines():
        if "❌" in line or "Error" in line or "error" in line.lower() or "Traceback" in line:
            error_line = line.strip()
            break

    return {
        "paper_id": paper_id,
        "success":  success,
        "elapsed":  round(elapsed, 1),
        "error":    error_line if not success else "",
        "stdout":   result.stdout[-2000:] if not success else "",
        "stderr":   result.stderr[-1000:] if not success else "",
    }


def main():
    ap = argparse.ArgumentParser(description="Batch parse ACL PDFs")
    ap.add_argument("--pdf-dir",      default="output/acl/pdfs")
    ap.add_argument("--out-dir",      default="output")
    ap.add_argument("--paper",        default=None, help="Run single paper ID only")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="Rerun papers that previously failed")
    ap.add_argument("--log",          default="batch_run_log.json",
                    help="Where to save per-paper results")
    ap.add_argument("--timeout",      type=int, default=900,
                    help="Per-paper timeout in seconds (default 900; "
                         "citation rate-limit backoffs can be slow)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)

    # Load previous log if exists
    log_path = Path(args.log)
    prev_log = load_previous_log(log_path)

    # Collect PDFs to process
    if args.paper:
        pdfs = [pdf_dir / f"{args.paper}.pdf"]
    else:
        pdfs = sorted(pdf_dir.glob("*.pdf"))

    to_run = []
    for pdf in pdfs:
        pid = pdf.stem
        if not pdf.exists():
            print(f"  [skip] {pid} — PDF not found")
            continue
        if already_complete(pid, out_dir) and not args.rerun_failed:
            print(f"  [done] {pid} — already complete")
            continue
        prev = prev_log.get(pid)
        if prev and prev.get("success") and not args.rerun_failed:
            print(f"  [done] {pid} — previously succeeded")
            continue
        to_run.append(pdf)

    if not to_run:
        print("\nNothing to run — all PDFs already have output.html and citation_network.json.")
        return

    print(f"\nRunning {len(to_run)} papers...\n{'─'*50}")

    results = list(prev_log.values())
    ok = fail = 0

    for i, pdf in enumerate(to_run, 1):
        pid = pdf.stem
        print(f"[{i}/{len(to_run)}] {pid} ...", end=" ", flush=True)
        result = run_paper(pdf, out_dir, extra_args=[], timeout=args.timeout)
        if result["success"]:
            print(f"✓ ({result['elapsed']}s)")
            ok += 1
        else:
            print(f"✗ — {result['error'][:80]}")
            fail += 1

        # Update log entry
        existing = next((r for r in results if r["paper_id"] == pid), None)
        if existing:
            results.remove(existing)
        results.append(result)

        # Persist after every paper so an interruption mid-batch (or a long
        # multi-hour citation run that gets killed) doesn't lose progress —
        # --rerun-failed can then pick up exactly where it left off.
        log_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Print summary
    print(f"\n{'─'*50}")
    print(f"Done: {ok} succeeded, {fail} failed")

    if fail > 0:
        print(f"\nFailed papers:")
        for r in results:
            if not r["success"]:
                print(f"  ✗ {r['paper_id']:20s} {r['error'][:70]}")
        print(f"\nFull error logs saved → {log_path}")
        print("Check the 'stdout' field for the full traceback per paper.")


if __name__ == "__main__":
    main()
