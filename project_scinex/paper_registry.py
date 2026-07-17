"""
paper_registry.py
-----------------
Builds and saves a registry of all papers from the SciClaimEval dataset.
Fetches paper titles from arXiv API automatically.
Saves to papers.json so you can look up paper IDs by name/domain anytime.

Usage:
    python paper_registry.py --fetch                    # fetch all from SciClaimEval
    python paper_registry.py --list                     # list all papers
    python paper_registry.py --list --domain nlp        # filter by domain
    python paper_registry.py --list --processed-only    # only processed papers
    python paper_registry.py --search bert              # search by name or ID
    python paper_registry.py --sync                     # update processed status
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

REGISTRY_FILE = Path("papers.json")
OUTPUT_DIR    = Path("output")


# ── Registry helpers ──────────────────────────────────────────────────────────

def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"papers": []}


def save_registry(registry: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(registry['papers'])} papers → {REGISTRY_FILE}")


def has_output(paper_id: str) -> bool:
    """Check if paper has been processed."""
    if not OUTPUT_DIR.exists():
        return False
    safe_id = paper_id.replace("/", "_")
    for candidate in OUTPUT_DIR.iterdir():
        if candidate.is_dir() and (safe_id in candidate.name or paper_id in candidate.name):
            return True
    return False


# ── Title fetching ────────────────────────────────────────────────────────────

def _fetch_arxiv_title(paper_id: str) -> str:
    """Fetch paper title from arXiv API. Returns empty string if not found."""
    # Only attempt for arXiv-style IDs
    if not (paper_id[:4].isdigit() or "/" in paper_id):
        return ""
    try:
        import requests
        url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        r   = requests.get(url, timeout=10)
        if r.status_code == 200:
            titles = re.findall(r"<title>(.*?)</title>", r.text, re.DOTALL)
            # First <title> is the feed title, second is the paper title
            if len(titles) > 1:
                return titles[1].strip().replace("\n", " ")
    except Exception:
        pass
    return ""


# ── Fetch from SciClaimEval ───────────────────────────────────────────────────

def fetch_from_sciclaimeval() -> list[dict]:
    """Load all papers from SciClaimEval and fetch titles from arXiv."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run: pip install datasets huggingface_hub")
        sys.exit(1)

    print("Loading SciClaimEval dataset from HuggingFace...")
    print("(If this fails: huggingface-cli login)\n")

    try:
        dataset = load_dataset("alabnii/sciclaimeval-shared-task", split="dev")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    seen   = set()
    papers = []

    for sample in dataset:
        paper_id = sample.get("paper_id", "").strip()
        domain   = sample.get("domain",   "").strip().lower()

        if not paper_id or paper_id in seen:
            continue

        seen.add(paper_id)
        papers.append({
            "id":        paper_id,
            "domain":    domain,
            "title":     "",
            "processed": has_output(paper_id),
        })

    print(f"Found {len(papers)} unique papers.")
    print("Fetching titles from arXiv API (this may take a minute)...")

    for i, p in enumerate(papers):
        title = _fetch_arxiv_title(p["id"])
        p["title"] = title if title else p["id"]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(papers)} done...")
        time.sleep(0.5)   # be polite to arXiv

    found = sum(1 for p in papers if p["title"] != p["id"])
    print(f"Titles fetched: {found}/{len(papers)}")
    return papers


# ── Display ───────────────────────────────────────────────────────────────────

def print_papers(papers: list[dict], fmt: str = "table"):
    if not papers:
        print("No papers found.")
        return

    if fmt == "json":
        print(json.dumps(papers, indent=2, ensure_ascii=False))
        return

    if fmt == "txt":
        for p in papers:
            done = "✓" if p.get("processed") else "○"
            print(f"{done}  {p['id']:<25} [{p['domain']:<6}]  {p.get('title','')[:60]}")
        return

    # Table (default)
    sep = "─" * 95
    print(f"\n  {'Done':<5} {'ID':<25} {'Domain':<8} {'Title'}")
    print(sep)
    for p in papers:
        done  = "✓" if p.get("processed") else "○"
        title = p.get("title", p["id"])[:55]
        print(f"  {done:<5} {p['id']:<25} {p.get('domain',''):<8} {title}")
    print(sep)

    from collections import Counter
    domains    = Counter(p.get("domain", "") for p in papers)
    done_count = sum(1 for p in papers if p.get("processed"))
    print(f"  Total: {len(papers)}  |  " + "  ".join(f"{d}: {c}" for d, c in sorted(domains.items())))
    print(f"  Processed: {done_count}/{len(papers)}\n")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_fetch(args):
    papers   = fetch_from_sciclaimeval()
    registry = {"papers": papers}
    save_registry(registry)
    print_papers(papers)


def cmd_list(args):
    registry = load_registry()
    papers   = registry["papers"]

    if not papers:
        print("Registry empty. Run: python paper_registry.py --fetch")
        return

    if args.domain:
        papers = [p for p in papers if p.get("domain", "").lower() == args.domain.lower()]
    if args.processed_only:
        papers = [p for p in papers if p.get("processed")]

    print_papers(papers, fmt=args.format)


def cmd_search(args):
    registry = load_registry()
    query    = args.search.lower()
    results  = [
        p for p in registry["papers"]
        if query in p.get("id",     "").lower()
        or query in p.get("title",  "").lower()
        or query in p.get("domain", "").lower()
    ]
    if not results:
        print(f"No papers found matching '{args.search}'")
    else:
        print_papers(results)


def cmd_sync(args):
    registry = load_registry()
    updated  = 0
    for p in registry["papers"]:
        new_status = has_output(p["id"])
        if new_status != p.get("processed", False):
            p["processed"] = new_status
            updated += 1
    save_registry(registry)
    done = sum(1 for p in registry["papers"] if p.get("processed"))
    print(f"Synced — {updated} updated. Processed: {done}/{len(registry['papers'])}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper registry — fetch and track SciClaimEval papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_registry.py --fetch
  python paper_registry.py --list
  python paper_registry.py --list --domain nlp
  python paper_registry.py --list --processed-only
  python paper_registry.py --search bert
  python paper_registry.py --sync
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch",  action="store_true", help="Fetch all papers from SciClaimEval + arXiv titles")
    group.add_argument("--list",   action="store_true", help="List papers in registry")
    group.add_argument("--search", metavar="QUERY",     help="Search by name, ID, or domain")
    group.add_argument("--sync",   action="store_true", help="Update processed status from output/")

    parser.add_argument("--domain",         metavar="DOMAIN", help="Filter by domain: nlp, ml, peerj")
    parser.add_argument("--processed-only", action="store_true")
    parser.add_argument("--format", choices=["table", "txt", "json"], default="table")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.fetch:
        cmd_fetch(args)
    elif args.list:
        cmd_list(args)
    elif args.search:
        cmd_search(args)
    elif args.sync:
        cmd_sync(args)


if __name__ == "__main__":
    main()