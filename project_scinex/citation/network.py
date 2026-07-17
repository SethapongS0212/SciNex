"""
citation/network.py
───────────────────
Builds, stores, and queries a citation network for a research paper.

The network is saved as a JSON file on disk so it persists between runs
and doesn't need to re-hit Semantic Scholar every time.

Network JSON format:
{
  "paper_id":   "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
  "title":      "Attention Is All You Need",
  "year":       2017,
  "nodes": {
    "<paperId>": {
      "title": ..., "year": ..., "abstract": ...,
      "relation": "citing" | "cited_by",
      "citation_count": ...,
      "concepts": [...]
    }
  },
  "edges": [
    {"source": "<paperId>", "target": "<paperId>", "type": "cites"}
  ]
}
"""

import json
import re
import os
import time
import urllib.request
import urllib.parse
from collections import Counter

# Load .env file if present (python-dotenv)
# Falls back silently if dotenv is not installed or .env does not exist.
# Load from the PROJECT ROOT (parent of this citation/ package), not the
# current working directory — otherwise launching the parse from another dir
# silently drops the S2 API key and every request 403s.
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────
# SEMANTIC SCHOLAR API
# ─────────────────────────────────────────────────────────────
BASE = "https://api.semanticscholar.org/graph/v1"

# ─────────────────────────────────────────────────────────────
# API KEY (optional but recommended)
#
# Get a free key at: https://www.semanticscholar.org/product/api
# Raises limit from ~100 req/5min to 1 req/sec.
#
# Set via environment variable (recommended — keeps key out of code):
#   export SEMANTIC_SCHOLAR_API_KEY="your_key_here"
#
# Or paste directly (for quick testing only):
#   SEMANTIC_SCHOLAR_API_KEY = "your_key_here"
# ─────────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

if SEMANTIC_SCHOLAR_API_KEY:
    print("[network] Semantic Scholar API key loaded ✔")
else:
    print("[network] No API key found — running unauthenticated "
          "(rate limit: ~100 req/5min). "
          "Set SEMANTIC_SCHOLAR_API_KEY env var for higher limits.")

STOPWORDS = {
    "these", "those", "therefore", "however", "because", "their",
    "about", "which", "with", "that", "this", "based", "using",
    "study", "paper", "method", "model", "models", "results", "show",
    "proposed", "approach", "propose", "achieve", "achieve", "both",
    "between", "performance", "training", "learning", "neural",
    "network", "networks", "language", "tasks", "task"
}


def _api_get(path, params=None, retries=5, delay=2.0):
    url = f"{BASE}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            headers = {"User-Agent": "pdf-parser-research/1.0"}
            if SEMANTIC_SCHOLAR_API_KEY:
                headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — wait much longer before retrying
                wait = delay * (4 ** attempt)   # 2s, 8s, 32s, 128s, 512s
                print(f"[network] Rate limited (429) — waiting {wait:.0f}s before retry {attempt + 1}/{retries}")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"[network] API error ({path}): {e}")
                return {}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"[network] API error ({path}): {e}")
                return {}
    return {}


def search_paper(title):
    data = _api_get("paper/search", {
        "query": title, "limit": 1,
        "fields": "paperId,title,year,citationCount"
    })
    results = data.get("data", [])
    return results[0] if results else None


def lookup_by_external_id(paper_id: str):
    """Look up a paper directly by ACL or arXiv ID — more reliable than
    title search for known paper IDs.
    Returns the Semantic Scholar paper dict, or None if not found.
    """
    import re
    # New-style ACL Anthology: "2024.eacl-short.2" → ACL:2024.eacl-short.2
    if re.match(r"^\d{4}\.[a-z]", paper_id):
        result = _api_get(
            f"paper/ACL:{paper_id}",
            {"fields": "paperId,title,year,citationCount"}
        )
        if result and result.get("paperId"):
            print(f"[network] Found via ACL ID: {result.get('title', '')}"
                  f" (ID: {result['paperId']})")
            return result

    # Old-style ACL Anthology: "C16-1036", "P18-1001", "W13-3105", "S16-1018"
    # Pattern: single uppercase letter + 2-digit year + dash + digits
    if re.match(r"^[A-Z]\d{2}-\d+$", paper_id):
        result = _api_get(
            f"paper/ACL:{paper_id}",
            {"fields": "paperId,title,year,citationCount"}
        )
        if result and result.get("paperId"):
            print(f"[network] Found via ACL ID: {result.get('title', '')}"
                  f" (ID: {result['paperId']})")
            return result
    # arXiv: "1810.04805" → ARXIV:1810.04805
    if re.match(r"^\d{4}\.\d{4,5}$", paper_id) or "/" in paper_id:
        result = _api_get(
            f"paper/ARXIV:{paper_id}",
            {"fields": "paperId,title,year,citationCount"}
        )
        if result and result.get("paperId"):
            print(f"[network] Found via arXiv ID: {result.get('title', '')}"
                  f" (ID: {result['paperId']})")
            return result
    return None


def get_paper_details(paper_id):
    return _api_get(f"paper/{paper_id}", {
        "fields": "paperId,title,year,abstract,citationCount,authors"
    })


def get_citing_papers(paper_id, limit=30):
    data = _api_get(f"paper/{paper_id}/citations", {
        "limit": limit,
        "fields": "citingPaper.paperId,citingPaper.title,"
                  "citingPaper.year,citingPaper.abstract,"
                  "citingPaper.citationCount"
    })
    if not data:
        return []
    return data.get("data", [])


def get_reference_papers(paper_id, limit=30):
    data = _api_get(f"paper/{paper_id}/references", {
        "limit": limit,
        "fields": "citedPaper.paperId,citedPaper.title,"
                  "citedPaper.year,citedPaper.abstract,"
                  "citedPaper.citationCount"
    })
    if not data:
        return []
    return data.get("data", [])


# ─────────────────────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_concepts(text, top_n=15):
    if not text:
        return []
    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(filtered).most_common(top_n)]


# ─────────────────────────────────────────────────────────────
# NETWORK BUILDER
# ─────────────────────────────────────────────────────────────
def build_network(title, citing_limit=30, reference_limit=30, external_id=None):
    """
    Fetch citation network from Semantic Scholar and return as a dict.
    """
    # Try direct external ID lookup first (ACL/arXiv) — bypasses title search
    # and is immune to bad title extraction from PDFs
    paper = None
    if external_id:
        paper = lookup_by_external_id(external_id)
    if not paper:
        print(f"[network] Searching for: {title!r}")
        paper = search_paper(title)
    if not paper:
        print("[network] Paper not found.")
        return None

    paper_id = paper["paperId"]
    print(f"[network] Found: {paper.get('title', '?')} (ID: {paper_id})")

    details = get_paper_details(paper_id)
    root_concepts = extract_concepts(details.get("abstract", ""))

    network = {
        "paper_id":     paper_id,
        "title":        details.get("title", title),
        "year":         details.get("year"),
        "abstract":     details.get("abstract", ""),
        "citation_count": details.get("citationCount", 0),
        "root_concepts": root_concepts,
        "nodes":  {},
        "edges":  []
    }

    # Add citing papers (papers that cite this one)
    print(f"[network] Fetching up to {citing_limit} citing papers...")
    citing = get_citing_papers(paper_id, citing_limit) or []
    for item in citing:
        p = item.get("citingPaper", {})
        pid = p.get("paperId")
        if not pid:
            continue
        network["nodes"][pid] = {
            "title":          p.get("title", ""),
            "year":           p.get("year"),
            "abstract":       p.get("abstract", ""),
            "citation_count": p.get("citationCount", 0),
            "relation":       "citing",
            "concepts":       extract_concepts(p.get("abstract", ""))
        }
        network["edges"].append({
            "source": pid, "target": paper_id, "type": "cites"
        })

    # Add reference papers (papers this one cites)
    print(f"[network] Fetching up to {reference_limit} reference papers...")
    references = get_reference_papers(paper_id, reference_limit) or []
    for item in references:
        p = item.get("citedPaper", {})
        pid = p.get("paperId")
        if not pid:
            continue
        if pid not in network["nodes"]:
            network["nodes"][pid] = {
                "title":          p.get("title", ""),
                "year":           p.get("year"),
                "abstract":       p.get("abstract", ""),
                "citation_count": p.get("citationCount", 0),
                "relation":       "cited_by",
                "concepts":       extract_concepts(p.get("abstract", ""))
            }
        network["edges"].append({
            "source": paper_id, "target": pid, "type": "cites"
        })

    print(
        f"[network] Built network: {len(network['nodes'])} nodes, "
        f"{len(network['edges'])} edges"
    )
    return network


# ─────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────
def save_network(network, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(network, f, indent=2, ensure_ascii=False)
    print(f"[network] Saved → {path}")


def load_network(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_or_build_network(title, cache_path, citing_limit=30, reference_limit=30, external_id=None):
    """
    Load network from cache if available, otherwise build and save.
    """
    if os.path.exists(cache_path):
        print(f"[network] Loading from cache: {cache_path}")
        return load_network(cache_path)

    network = build_network(title, citing_limit, reference_limit, external_id=external_id)
    if network:
        save_network(network, cache_path)
    return network


# ─────────────────────────────────────────────────────────────
# CONCEPT AGGREGATION  (replaces concept_builder.py)
# ─────────────────────────────────────────────────────────────
def build_concept_dict_from_network(network, top_n=50):
    """
    Build a frequency-weighted concept dict from the full citation network.

    Citing papers contribute 2× weight (they use the paper in context).
    Reference papers contribute 1× weight (they provide background).

    Returns: {keyword: weighted_freq, ...}
    """
    if not network:
        return {}

    counter = Counter()
    for node in network["nodes"].values():
        weight = 2 if node.get("relation") == "citing" else 1
        for concept in node.get("concepts", []):
            counter[concept] += weight

    return dict(counter.most_common(top_n))


# ─────────────────────────────────────────────────────────────
# RELATED PAPER SEARCH  (for auto-discovery)
# ─────────────────────────────────────────────────────────────
def find_similar_papers(network, top_n=10):
    """
    Rank nodes by citation count as a proxy for relevance/importance.
    Returns top_n nodes sorted by citation_count descending.
    """
    if not network:
        return []

    nodes = [
        {"paper_id": pid, **info}
        for pid, info in network["nodes"].items()
    ]
    nodes.sort(key=lambda n: n.get("citation_count", 0), reverse=True)
    return nodes[:top_n]


def concept_overlap_score(network, query_text):
    """
    Score each node in the network by concept overlap with query_text.
    Useful for finding papers most related to a specific topic.
    """
    query_concepts = set(extract_concepts(query_text, top_n=30))
    if not query_concepts:
        return []

    scored = []
    for pid, node in network["nodes"].items():
        node_concepts = set(node.get("concepts", []))
        overlap = len(query_concepts & node_concepts)
        if overlap > 0:
            scored.append({
                "paper_id": pid,
                "title":    node.get("title", ""),
                "year":     node.get("year"),
                "overlap":  overlap,
                "relation": node.get("relation", "")
            })

    scored.sort(key=lambda x: -x["overlap"])
    return scored


# ─────────────────────────────────────────────────────────────
# CLI — standalone use
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build citation network for a paper")
    ap.add_argument("title",             help="Paper title (quoted)")
    ap.add_argument("--cache",           default="output/citation_network.json",
                    help="Path to save/load network JSON")
    ap.add_argument("--citing-limit",    type=int, default=30)
    ap.add_argument("--reference-limit", type=int, default=30)
    ap.add_argument("--show-concepts",   action="store_true")
    ap.add_argument("--show-similar",    action="store_true")
    args = ap.parse_args()

    net = get_or_build_network(
        args.title, args.cache,
        args.citing_limit, args.reference_limit
    )
    if not net:
        print("Failed to build network.")
        raise SystemExit(1)

    print(f"\nPaper : {net['title']}")
    print(f"Year  : {net['year']}")
    print(f"Cited : {net['citation_count']:,} times")
    print(f"Nodes : {len(net['nodes'])}")
    print(f"Edges : {len(net['edges'])}")

    if args.show_concepts:
        cd = build_concept_dict_from_network(net, top_n=20)
        print("\nTop concepts:")
        for kw, freq in sorted(cd.items(), key=lambda x: -x[1])[:20]:
            print(f"  {kw:30s} {freq}")

    if args.show_similar:
        similar = find_similar_papers(net, top_n=5)
        print("\nTop related papers:")
        for p in similar:
            print(f"  [{p['relation']:8s}] {p['title'][:60]} ({p['year']})")