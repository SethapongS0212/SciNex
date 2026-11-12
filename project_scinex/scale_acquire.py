#!/usr/bin/env python3
"""
scale_acquire.py
----------------
Session 39 (2026-11-12): acquisition for the scale-up, one resumable stage at a time, for the
papers in scale/candidates.txt (scale_select.py). Never touches the 805 existing papers.

  download  aclanthology.org PDFs -> output/acl/pdfs/<id>.pdf (1 s between requests)
  parse     main.py --no-llm --no-citations, N workers, then no-llm_no-citations/ -> no-llm/
            (citations are not used by the KG pipeline and their API calls are slow)
  enrich    CS-NER gazetteer entity list -> output/acl/<id>/Entity_<id>_enriched.csv
            (enrich_entity_csv.enrich_paper with the CLI defaults the 805 papers used; the
            new papers have no CS-NER title seeds, so their lists are gazetteer-only)
  status    counts per stage

    python scale_acquire.py download --list scale/candidates.txt
    python scale_acquire.py parse --list scale/candidates.txt --workers 8
    python scale_acquire.py enrich --list scale/candidates.txt
then: build_entity_v2.py --papers <ids> --summary scale/entity_v2_summary.json
"""
import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output"
PDF_DIR = OUT / "acl" / "pdfs"


def html(pid):
    return OUT / pid / "no-llm" / "output.html"


def enriched(pid):
    return OUT / "acl" / pid / f"Entity_{pid}_enriched.csv"


def stage_download(ids, delay):
    from acl_pipeline import download_pdf
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i, pid in enumerate(ids, 1):
        if (PDF_DIR / f"{pid}.pdf").exists():
            ok += 1
            continue
        try:
            ok += bool(download_pdf(pid, PDF_DIR, delay=delay))
        except Exception as e:                       # one bad paper must not stop the run
            print(f"  ! {pid}: {e}", flush=True)
            fail += 1
            time.sleep(delay)
        if i % 100 == 0:
            print(f"[{time.strftime('%F %T')}] download {i}/{len(ids)}: have {ok}, failed {fail}", flush=True)
    print(f"download done: {ok} PDFs, {fail} errors", flush=True)


def _parse_one(pid, py, timeout):
    if html(pid).exists():
        return pid, "done"
    pdf = PDF_DIR / f"{pid}.pdf"
    if not pdf.exists():
        return pid, "no_pdf"
    try:
        r = subprocess.run([py, "main.py", str(pdf), "--no-llm", "--no-citations"], cwd=ROOT,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return pid, "timeout"
    src = OUT / pid / "no-llm_no-citations"
    if r.returncode != 0 or not (src / "output.html").exists():
        return pid, f"failed rc={r.returncode}"
    dst = OUT / pid / "no-llm"
    if dst.exists():
        shutil.rmtree(dst)
    src.rename(dst)
    return pid, "ok"


def stage_parse(ids, workers, timeout):
    from collections import Counter
    st = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_parse_one, p, sys.executable, timeout) for p in ids]
        for i, f in enumerate(as_completed(futs), 1):
            pid, s = f.result()
            st[s.split()[0]] += 1
            if s.startswith(("failed", "timeout")):
                print(f"  ! {pid}: {s}", flush=True)
            if i % 200 == 0:
                print(f"[{time.strftime('%F %T')}] parse {i}/{len(ids)}: {dict(st)}", flush=True)
    print(f"parse done: {dict(st)}", flush=True)


def stage_enrich(ids):
    import logging
    import enrich_entity_csv as E
    logging.getLogger("enrich").setLevel(logging.WARNING)
    g = E.load_gazetteer()
    ok = skip = 0
    for i, pid in enumerate(ids, 1):
        if enriched(pid).exists() or not html(pid).exists():
            skip += 1
            continue
        ok += bool(E.enrich_paper(pid, "csner", None, "Qwen3-14B", E.ACL_DIR, 1, None, False,
                                  hard_types_only=True, max_words=6, gazetteer=g))
        if i % 500 == 0:
            print(f"[{time.strftime('%F %T')}] enrich {i}/{len(ids)}: wrote {ok}", flush=True)
    print(f"enrich done: wrote {ok}, skipped {skip}", flush=True)


def stage_status(ids):
    n_pdf = sum((PDF_DIR / f"{p}.pdf").exists() for p in ids)
    n_html = sum(html(p).exists() for p in ids)
    n_ent = sum(enriched(p).exists() for p in ids)
    n_v2 = sum((OUT / "acl_v2" / p / f"Entity_{p}_enriched.csv").exists() for p in ids)
    print(f"{len(ids)} candidates: pdf {n_pdf}, parsed {n_html}, entity list {n_ent}, v2 list {n_v2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "parse", "enrich", "status"])
    ap.add_argument("--list", default="scale/candidates.txt")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    ids = Path(args.list).read_text(encoding="utf-8").split()
    existing = set((ROOT / "papers_acl.txt").read_text(encoding="utf-8").split())
    if existing & set(ids):
        sys.exit("refusing: the list contains papers of the 805-paper corpus")
    {"download": lambda: stage_download(ids, args.delay),
     "parse": lambda: stage_parse(ids, args.workers, args.timeout),
     "enrich": lambda: stage_enrich(ids),
     "status": lambda: stage_status(ids)}[args.stage]()


if __name__ == "__main__":
    main()
