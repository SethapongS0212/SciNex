"""
citation_expand_pipeline.py
----------------------------
Expands your paper set by walking citation networks instead of pulling
sequentially from the CS-NER IOB list.

Algorithm (BFS / snowball over the citation graph):
  1. Seed queue = papers you've already downloaded (output/acl/pdfs/*.pdf),
     ordered by papers.json if present, else alphabetically.
  2. Pop the next seed. Load output/<seed>/citation_network.json.
  3. Walk its "nodes" (citing + cited papers). For each node:
       - Resolve its title to an ACL Anthology ID (local index -> DBLP -> S2),
         same 3-tier lookup acl_pipeline.py uses.
       - Skip it if not resolvable (i.e. not on ACL Anthology) or already downloaded.
       - Otherwise download the PDF and push the new paper_id onto the back
         of the queue, so ITS citation network gets explored too.
  4. When a seed's citation network is exhausted (nothing new left to pull),
     move on to the next seed in the queue automatically.
  5. Stop as soon as --limit total PDFs is reached.

This never re-downloads anything and only ever pulls papers that are
confirmed to be on ACL Anthology.

Usage:
    python citation_expand_pipeline.py --limit 500
    python citation_expand_pipeline.py --limit 500 --pdf-dir output/acl/pdfs --papers-dir output
    python citation_expand_pipeline.py --limit 500 --seeds P19-1028,D19-1234   # start from specific papers
    python citation_expand_pipeline.py --limit 500 --no-recurse               # only 1 hop from original seeds
"""

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path

from acl_pipeline import (
    ACL_PDF_URL,        # noqa: F401 (kept for reference)
    _dblp_lookup,
    _s2_lookup,
    download_pdf,
    load_papers,
    save_papers,
)
from acl_index import ACLIndex

logger = logging.getLogger(__name__)


def _norm_title(t: str) -> str:
    return " ".join(t.lower().split())


def resolve_one(title: str, idx: ACLIndex, delay: float) -> str | None:
    """Same 3-tier lookup as acl_pipeline.resolve_ids, for a single title."""
    pid = idx.lookup(title)
    if pid:
        return pid
    pid = _dblp_lookup(title)
    if pid:
        time.sleep(delay)
        return pid
    pid = _s2_lookup(title)
    if pid:
        time.sleep(delay)
        return pid
    return None


def load_citation_nodes(paper_id: str, papers_dir: Path) -> list[tuple[str, dict]]:
    cit_path = papers_dir / paper_id / "citation_network.json"
    if not cit_path.exists():
        return []
    try:
        data = json.loads(cit_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  Could not read {cit_path}: {e}")
        return []
    nodes = data.get("nodes", {})
    # citing papers first (papers that build on this one), then cited (references)
    citing = [(k, v) for k, v in nodes.items() if v.get("relation") == "citing"]
    cited  = [(k, v) for k, v in nodes.items() if v.get("relation") != "citing"]
    return citing + cited


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Expand paper set via citation graph walk")
    ap.add_argument("--pdf-dir",     default="output/acl/pdfs",
                     help="Where PDFs already live / get downloaded to")
    ap.add_argument("--out-dir",     default="output/acl",
                     help="ACL pipeline dir containing papers.json")
    ap.add_argument("--papers-dir",  default="output",
                     help="Root dir containing <paper_id>/citation_network.json per paper")
    ap.add_argument("--index",       default="acl_title_index.json")
    ap.add_argument("--limit",       type=int, required=True,
                     help="Stop once total PDFs on disk reaches this count")
    ap.add_argument("--seeds",       default=None,
                     help="Comma-separated paper IDs to start from (default: all PDFs already downloaded)")
    ap.add_argument("--delay",       type=float, default=1.0,
                     help="Seconds between HTTP requests")
    ap.add_argument("--no-recurse",  action="store_true",
                     help="Only expand from the original seeds (don't snowball into newly downloaded papers)")
    args = ap.parse_args()

    pdf_dir     = Path(args.pdf_dir)
    out_dir     = Path(args.out_dir)
    papers_dir  = Path(args.papers_dir)
    papers_json = out_dir / "papers.json"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    downloaded_ids = {p.stem for p in pdf_dir.glob("*.pdf")}
    start_count = len(downloaded_ids)

    if start_count >= args.limit:
        print(f"Already have {start_count} PDFs >= limit {args.limit}. Nothing to do.")
        return

    # ── Build seed order ────────────────────────────────────────────────────
    existing_papers = load_papers(papers_json) if papers_json.exists() else []
    ordered_from_json = [p["paper_id"] for p in existing_papers if p.get("paper_id") in downloaded_ids]
    remaining = sorted(downloaded_ids - set(ordered_from_json))
    default_seed_order = ordered_from_json + remaining

    if args.seeds:
        seed_order = [s.strip() for s in args.seeds.split(",") if s.strip()]
    else:
        seed_order = default_seed_order

    if not seed_order:
        print("No seed papers found — download at least one paper via acl_pipeline.py first.")
        return

    visited_titles = {_norm_title(p["title"]) for p in existing_papers if p.get("title")}
    idx = ACLIndex(args.index).load_or_build()

    queue: deque[str] = deque(seed_order)
    processed_seeds: set[str] = set()
    new_papers: list[dict] = []
    total = start_count

    print(f"Starting with {total} PDFs, {len(seed_order)} seed(s), target {args.limit}")
    print("─" * 60)

    while queue and total < args.limit:
        seed = queue.popleft()
        if seed in processed_seeds:
            continue
        processed_seeds.add(seed)

        nodes = load_citation_nodes(seed, papers_dir)
        if not nodes:
            print(f"[seed {seed}] no citation_network.json / no nodes — moving on")
            continue

        print(f"[seed {seed}] exploring {len(nodes)} citation-network papers...")
        pulled_from_this_seed = 0

        for _node_id, node in nodes:
            if total >= args.limit:
                break

            title = (node.get("title") or "").strip()
            if not title:
                continue
            norm = _norm_title(title)
            if norm in visited_titles:
                continue
            visited_titles.add(norm)

            pid = resolve_one(title, idx, args.delay)
            if not pid:
                continue  # not on ACL Anthology — skip
            if pid in downloaded_ids:
                continue  # already have it

            ok = download_pdf(pid, pdf_dir, delay=args.delay)
            if ok:
                downloaded_ids.add(pid)
                total += 1
                pulled_from_this_seed += 1
                new_papers.append({"paper_id": pid, "title": title, "entities": [],
                                    "source": f"citation_of:{seed}"})
                print(f"  [+{total}/{args.limit}] {pid}  ←  cited/citing via {seed}")
                if not args.no_recurse:
                    queue.append(pid)  # snowball: explore this new paper's network too

        if pulled_from_this_seed == 0:
            print(f"[seed {seed}] exhausted (nothing new) — moving to next")

    # ── Persist ──────────────────────────────────────────────────────────────
    if new_papers:
        merged = existing_papers + new_papers
        save_papers(merged, papers_json)

    print("─" * 60)
    print(f"Done. PDFs: {start_count} → {total}  (+{total - start_count} new)")
    print(f"papers.json updated → {papers_json}")
    if total < args.limit:
        print(f"Note: hit end of reachable citation graph before reaching limit "
              f"({total}/{args.limit}). Add more seeds or run acl_pipeline.py for more source papers.")


if __name__ == "__main__":
    main()
