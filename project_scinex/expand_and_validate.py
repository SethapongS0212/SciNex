#!/usr/bin/env python3
"""
expand_and_validate.py
-----------------------
Validation smoke test for the persistent global KG (kg_extraction/global_graph.py):
pull a handful of new papers from ONE seed paper's citation_network.json, run them
through the full parse -> enrich -> fixed-extract chain, and confirm they merge
into the existing global graph correctly (no isolated new papers, citation stubs
resolve, etc.) before trusting the incremental-merge code on the full corpus.

Composes EXISTING tools rather than reimplementing them:
    - citation_expand_pipeline.py  (download new PDFs from the seed's citation network)
    - main.py --no-llm             (parse each new PDF)
    - enrich_entity_csv.py         (CS-NER entity lists for the new papers)
    - kg_main.py --skip-existing   (fixed extraction; auto-merges into the global
                                    graph per the kg_main.py change described in
                                    the "persistent global KG" plan)

Must be run on the GPU VM (step 4 loads Qwen3-14B) — this is handed over as code,
not executed here.

Usage (seed must be a real ACL Anthology paper already in the corpus — e.g. any
already-parsed id under output/, such as 2020.acl-main.185 — NOT the one-off
manually-added "BERT" folder, which has no ACL id for citation-network neighbor
resolution to key off of):
    python3 expand_and_validate.py --seed 2020.acl-main.185 --count 8
    python3 expand_and_validate.py --seed 2020.acl-main.185 --count 8 \\
        --model Qwen/Qwen3-14B --ontology scinex
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from acl_index import ACLIndex
from citation_expand_pipeline import load_citation_nodes, resolve_one, download_pdf
from run_acl_batch import run_paper, already_complete

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _norm_title(t: str) -> str:
    return " ".join(t.lower().split())


# ── Step 1: download N new papers from the seed's citation network only ───────

def fetch_from_seed(seed: str, count: int, pdf_dir: Path, papers_dir: Path,
                     index_path: str, delay: float) -> list[str]:
    nodes = load_citation_nodes(seed, papers_dir)
    if not nodes:
        logger.error(
            f"No citation_network.json / no nodes for seed '{seed}' under "
            f"{papers_dir / seed}. Parse it first (main.py needs citations enabled)."
        )
        return []

    downloaded_ids = {p.stem for p in pdf_dir.glob("*.pdf")}
    visited_titles = {
        _norm_title(json.loads((papers_dir / p / "citation_network.json").read_text()).get("title", ""))
        for p in downloaded_ids
        if (papers_dir / p / "citation_network.json").exists()
    }

    idx = ACLIndex(index_path).load_or_build()
    new_ids: list[str] = []

    logger.info(f"[seed {seed}] {len(nodes)} citation-network papers to consider, target +{count}")
    for _node_id, node in nodes:
        if len(new_ids) >= count:
            break
        title = (node.get("title") or "").strip()
        if not title:
            continue
        norm = _norm_title(title)
        if norm in visited_titles:
            continue
        visited_titles.add(norm)

        pid = resolve_one(title, idx, delay)
        if not pid or pid in downloaded_ids:
            continue
        if (papers_dir / pid / "no-llm" / "output.html").exists():
            continue  # already parsed via some other route

        if download_pdf(pid, pdf_dir, delay=delay):
            downloaded_ids.add(pid)
            new_ids.append(pid)
            logger.info(f"  [+{len(new_ids)}/{count}] {pid}  ({title[:60]})")

    if len(new_ids) < count:
        logger.warning(
            f"Only found {len(new_ids)}/{count} new resolvable papers in "
            f"'{seed}'s citation network — using what's available."
        )
    return new_ids


# ── Step 2: parse each new PDF ─────────────────────────────────────────────────

def parse_new_papers(new_ids: list[str], pdf_dir: Path, out_dir: Path, timeout: int) -> list[str]:
    parsed = []
    for pid in new_ids:
        pdf_path = pdf_dir / f"{pid}.pdf"
        if not pdf_path.exists():
            logger.warning(f"  {pid}: PDF missing at {pdf_path}, skipping parse")
            continue
        logger.info(f"[parse] {pid} ...")
        result = run_paper(pdf_path, out_dir, extra_args=[], timeout=timeout)
        if result["success"]:
            parsed.append(pid)
            logger.info(f"  {pid}: parsed OK ({result['elapsed']}s)")
        else:
            logger.warning(f"  {pid}: parse failed — {result.get('error', 'unknown error')}")
    return parsed


# ── Steps 3-4: enrich + fixed-extract (both are full-corpus --all --skip-existing,
#    so this only does work for the newly-parsed papers) ──────────────────────

def enrich_and_extract(model: str, ontology: str, global_graph_dir: str, max_new_tokens: int) -> None:
    logger.info("[enrich] enrich_entity_csv.py --all --source csner")
    subprocess.run(
        [sys.executable, "enrich_entity_csv.py", "--all", "--source", "csner"],
        check=False,
    )

    logger.info(f"[extract] kg_main.py --all --extractor fixed --ontology {ontology} "
                f"--model {model} --skip-existing (GPU)")
    # NB: kg_main.py's CLI only ever takes --extractor fixed; --ontology scinex is
    # what routes it internally to the "fixed_scinex" extractor name/output dir.
    cmd = [
        sys.executable, "kg_main.py", "--all",
        "--extractor", "fixed",
        "--ontology", ontology,
        "--model", model,
        "--skip-existing",
        "--global-graph-dir", global_graph_dir,
        "--max-new-tokens", str(max_new_tokens),
    ]
    subprocess.run(cmd, check=False)


# ── Step 5: connectivity report ────────────────────────────────────────────────

def report(new_ids: list[str], ontology: str, model: str, global_graph_dir: str) -> None:
    from kg_extraction import global_graph

    extractor_dir_name = "fixed_scinex" if ontology == "scinex" else "fixed"
    model_slug = model.split("/")[-1]
    graph, meta = global_graph.load_global_graph(global_graph_dir, extractor_dir_name, model_slug)

    conn = global_graph.connectivity_report(graph, new_ids)

    print("\n" + "─" * 72)
    print(f"Global graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
          f"{len(meta['merged_papers'])} papers merged")
    print("─" * 72)
    print(f"  {'Paper':<28} {'Present':<8} {'Cites':>6} {'SharedEnt':>10} {'Isolated':>9}")
    print("─" * 72)
    n_isolated = 0
    n_missing = 0
    for pid in new_ids:
        info = conn.get(pid, {"present": False})
        if not info.get("present"):
            print(f"  {pid:<28} {'NO':<8} {'-':>6} {'-':>10} {'-':>9}")
            n_missing += 1
            continue
        isolated = info["isolated"]
        n_isolated += int(isolated)
        print(f"  {pid:<28} {'yes':<8} {info['cites_edges']:>6} "
              f"{info['shared_entity_neighbors']:>10} {('YES' if isolated else 'no'):>9}")
    print("─" * 72)
    print(f"New papers requested: {len(new_ids)} | merged into graph: "
          f"{len(new_ids) - n_missing} | isolated (BUG SIGNAL): {n_isolated}")
    if n_isolated or n_missing:
        print("⚠ Investigate isolated/missing papers before trusting this on the full corpus.")
    else:
        print("✓ All new papers attached to the existing graph via citations and/or shared entities.")
    print("─" * 72 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", required=True, help="Paper id already in the corpus to expand from")
    ap.add_argument("--count", type=int, default=8, help="How many new papers to pull (default: 8)")
    ap.add_argument("--pdf-dir", default="output/acl/pdfs")
    ap.add_argument("--papers-dir", default="output")
    ap.add_argument("--index", default="acl_title_index.json")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--parse-timeout", type=int, default=1200)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--ontology", choices=["ceo", "scinex"], default="ceo",
                     help="CEO (default, → kg/fixed/) or scinex (→ kg/fixed_scinex/)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--global-graph-dir", default="output/global_kg")
    ap.add_argument("--skip-download", action="store_true",
                     help="Skip step 1 (assume the new PDFs are already downloaded)")
    ap.add_argument("--skip-parse", action="store_true", help="Skip step 2")
    ap.add_argument("--skip-extract", action="store_true", help="Skip steps 3-4 (GPU)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    papers_dir = Path(args.papers_dir)
    out_dir = Path(args.papers_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    if not (papers_dir / args.seed / "citation_network.json").exists():
        logger.error(f"Seed '{args.seed}' has no citation_network.json under "
                     f"{papers_dir / args.seed} — parse it (with citations) first.")
        sys.exit(1)

    if args.skip_download:
        # Resume mode: treat whatever isn't parsed yet under pdf_dir as "new"
        new_ids = [p.stem for p in pdf_dir.glob("*.pdf")
                   if not already_complete(p.stem, out_dir)]
        logger.info(f"--skip-download: resuming with {len(new_ids)} unparsed PDF(s) already on disk")
    else:
        new_ids = fetch_from_seed(args.seed, args.count, pdf_dir, papers_dir, args.index, args.delay)

    if not new_ids:
        logger.error("No new papers to validate with. Nothing to do.")
        sys.exit(1)

    if not args.skip_parse:
        parsed = parse_new_papers(new_ids, pdf_dir, out_dir, args.parse_timeout)
        failed = set(new_ids) - set(parsed)
        if failed:
            logger.warning(f"{len(failed)} paper(s) failed to parse and will be missing "
                            f"from the report: {sorted(failed)}")

    if not args.skip_extract:
        enrich_and_extract(args.model, args.ontology,
                            args.global_graph_dir, args.max_new_tokens)

    report(new_ids, args.ontology, args.model, args.global_graph_dir)


if __name__ == "__main__":
    main()
