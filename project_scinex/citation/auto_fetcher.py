"""
citation/auto_fetcher.py
────────────────────────
Automatic paper discovery, download, and processing pipeline.

Two discovery backends:
  • Semantic Scholar  — rich metadata, citation graph, abstracts
  • arXiv API         — free PDF downloads, structured XML metadata

Storage:
  All discovered papers are stored in <output_dir>/citation_db.json
  (a simple append-only key-value store keyed by paper ID).
  PDFs are saved under <output_dir>/papers/<arxiv_id>.pdf.

Typical usage
─────────────
    from citation.auto_fetcher import CitationFetcher

    fetcher = CitationFetcher(output_dir="output/attention", workers=4)
    network = fetcher.build_network(
        seed_title="Attention Is All You Need",
        depth=2,          # hop depth (1 = direct citations only)
        max_per_hop=20,   # max papers fetched per hop
        download_pdfs=True
    )
    fetcher.save()
    # network is the full citation graph JSON

Network JSON schema
───────────────────
{
  "seed": { "title": ..., "paperId": ... },
  "nodes": {
    "<paperId>": {
      "paperId":  str,
      "title":    str,
      "abstract": str,
      "year":     int | null,
      "authors":  [str, ...],
      "venue":    str | null,
      "arxiv_id": str | null,       # if available
      "pdf_path": str | null,       # local path if downloaded
      "concepts": [str, ...],       # extracted keywords
      "citing_ids":    [str, ...],  # papers this node cites (outgoing)
      "cited_by_ids":  [str, ...],  # papers that cite this node (incoming)
    },
    ...
  },
  "edges": [
    { "from": paperId, "to": paperId, "relation": "cites" },
    ...
  ]
}
"""

import os
import re
import json
import time
import threading
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from collections import Counter, deque

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
S2_BASE      = "https://api.semanticscholar.org/graph/v1"
ARXIV_SEARCH = "https://export.arxiv.org/api/query"
ARXIV_PDF    = "https://arxiv.org/pdf/{arxiv_id}.pdf"

S2_PAPER_FIELDS = (
    "paperId,title,abstract,year,authors,venue,"
    "externalIds,citationCount,referenceCount"
)
S2_CITE_FIELDS  = "citingPaper.paperId,citingPaper.title,citingPaper.abstract,citingPaper.year,citingPaper.authors,citingPaper.venue,citingPaper.externalIds"
S2_REF_FIELDS   = "citedPaper.paperId,citedPaper.title,citedPaper.abstract,citedPaper.year,citedPaper.authors,citedPaper.venue,citedPaper.externalIds"

CONCEPT_STOPWORDS = {
    "these", "those", "therefore", "however", "because",
    "their", "about", "which", "with", "that", "this",
    "based", "using", "study", "paper", "method", "methods",
    "approach", "model", "models", "result", "results",
    "propose", "show", "demonstrate", "present", "network",
    "learning", "training", "performance", "state"
}


# ─────────────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────────────
def _get(url, params=None, timeout=15, retries=3):
    """GET with retry + polite backoff. Works with or without `requests`."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)

    for attempt in range(retries):
        try:
            if HAS_REQUESTS:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 429:        # rate-limited
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "pdf-parser/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())
        except Exception as e:
            print(f"  [HTTP] attempt {attempt+1} failed: {e}")
            time.sleep(1.5 ** attempt)
    return None


def _download_file(url, dest_path, timeout=60):
    """Download a binary file (PDF) to dest_path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pdf-parser/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, \
             open(dest_path, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"  [DOWNLOAD] {url} → {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# CONCEPT EXTRACTION
# ─────────────────────────────────────────────────────────────────────
def extract_concepts(text, top_n=20):
    """Extract meaningful keywords from abstract/title text."""
    if not text:
        return []
    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
    filtered = [w for w in words if w not in CONCEPT_STOPWORDS]
    counter = Counter(filtered)
    return [w for w, _ in counter.most_common(top_n)]


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC SCHOLAR CLIENT
# ─────────────────────────────────────────────────────────────────────
class SemanticScholar:

    @staticmethod
    def search(title, limit=1):
        data = _get(f"{S2_BASE}/paper/search",
                    {"query": title, "limit": limit, "fields": S2_PAPER_FIELDS})
        return (data or {}).get("data", [])

    @staticmethod
    def paper_by_id(paper_id):
        return _get(f"{S2_BASE}/paper/{paper_id}", {"fields": S2_PAPER_FIELDS})

    @staticmethod
    def citing(paper_id, limit=50, offset=0):
        """Papers that CITE paper_id."""
        data = _get(
            f"{S2_BASE}/paper/{paper_id}/citations",
            {"fields": S2_CITE_FIELDS, "limit": limit, "offset": offset}
        )
        raw = (data or {}).get("data", [])
        return [item.get("citingPaper", {}) for item in raw if item.get("citingPaper")]

    @staticmethod
    def references(paper_id, limit=50, offset=0):
        """Papers CITED BY paper_id."""
        data = _get(
            f"{S2_BASE}/paper/{paper_id}/references",
            {"fields": S2_REF_FIELDS, "limit": limit, "offset": offset}
        )
        raw = (data or {}).get("data", [])
        return [item.get("citedPaper", {}) for item in raw if item.get("citedPaper")]


# ─────────────────────────────────────────────────────────────────────
# ARXIV CLIENT
# ─────────────────────────────────────────────────────────────────────
_ARXIV_NS = "http://www.w3.org/2005/Atom"

class ArXiv:

    @staticmethod
    def search(query, max_results=5):
        """Search arXiv and return structured metadata list."""
        try:
            url = (f"{ARXIV_SEARCH}?search_query=ti:{urllib.parse.quote(query)}"
                   f"&max_results={max_results}&sortBy=relevance")
            req = urllib.request.Request(url, headers={"User-Agent": "pdf-parser/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                xml_bytes = r.read()
        except Exception as e:
            print(f"  [arXiv] search failed: {e}")
            return []

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return []

        results = []
        for entry in root.findall(f"{{{_ARXIV_NS}}}entry"):
            def text(tag):
                el = entry.find(f"{{{_ARXIV_NS}}}{tag}")
                return el.text.strip() if el is not None and el.text else ""

            # arXiv ID — strip URL prefix
            raw_id = text("id")
            arxiv_id = re.sub(r".*/abs/", "", raw_id).split("v")[0]

            authors = [
                a.find(f"{{{_ARXIV_NS}}}name").text.strip()
                for a in entry.findall(f"{{{_ARXIV_NS}}}author")
                if a.find(f"{{{_ARXIV_NS}}}name") is not None
            ]

            results.append({
                "arxiv_id":  arxiv_id,
                "title":     text("title").replace("\n", " "),
                "abstract":  text("summary").replace("\n", " "),
                "published": text("published")[:10],
                "authors":   authors,
            })
        return results

    @staticmethod
    def pdf_url(arxiv_id):
        return ARXIV_PDF.format(arxiv_id=arxiv_id)


# ─────────────────────────────────────────────────────────────────────
# NODE BUILDER
# ─────────────────────────────────────────────────────────────────────
def _s2_to_node(paper_dict):
    """Convert a Semantic Scholar paper dict to our node schema."""
    if not paper_dict or not paper_dict.get("paperId"):
        return None

    ext_ids  = paper_dict.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("arxiv")

    authors = []
    for a in (paper_dict.get("authors") or []):
        if isinstance(a, dict):
            authors.append(a.get("name", ""))
        elif isinstance(a, str):
            authors.append(a)

    abstract = paper_dict.get("abstract") or ""

    return {
        "paperId":      paper_dict["paperId"],
        "title":        paper_dict.get("title") or "",
        "abstract":     abstract,
        "year":         paper_dict.get("year"),
        "authors":      authors,
        "venue":        paper_dict.get("venue"),
        "arxiv_id":     arxiv_id,
        "pdf_path":     None,
        "concepts":     extract_concepts(abstract),
        "citing_ids":   [],
        "cited_by_ids": [],
    }


# ─────────────────────────────────────────────────────────────────────
# CITATION FETCHER
# ─────────────────────────────────────────────────────────────────────
class CitationFetcher:
    """
    Main entry point.  Builds a full citation network JSON around a
    seed paper, optionally downloading PDFs for offline processing.

    Parameters
    ──────────
    output_dir    : directory where citation_db.json and PDFs are stored
    workers       : unused (kept for API compatibility; fetch is sequential
                    to respect Semantic Scholar's rate limit)
    s2_api_key    : optional Semantic Scholar API key (raises rate limit)
    """

    def __init__(self, output_dir="output", workers=4, s2_api_key=None):
        self.output_dir  = output_dir
        self.pdf_dir     = os.path.join(output_dir, "papers")
        self.db_path     = os.path.join(output_dir, "citation_db.json")
        self._api_key    = s2_api_key
        os.makedirs(self.pdf_dir, exist_ok=True)

        # Load existing DB if present (enables incremental runs)
        self._db = self._load_db()

    # ── persistence ────────────────────────────────────────────────

    def _load_db(self):
        if os.path.isfile(self.db_path):
            try:
                with open(self.db_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"seed": None, "nodes": {}, "edges": []}

    def save(self):
        """Persist the citation network to disk."""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._db, f, indent=2, ensure_ascii=False)
        print(f"[CitationFetcher] Saved {len(self._db['nodes'])} nodes → {self.db_path}")

    # ── PDF download ────────────────────────────────────────────────

    def _maybe_download_pdf(self, node):
        """Download the PDF for a node if an arXiv ID is available."""
        arxiv_id = node.get("arxiv_id")
        if not arxiv_id:
            return

        dest = os.path.join(self.pdf_dir, f"{arxiv_id}.pdf")
        if os.path.isfile(dest):
            node["pdf_path"] = dest
            return

        url = ArXiv.pdf_url(arxiv_id)
        print(f"  [PDF] Downloading {arxiv_id} …")
        ok = _download_file(url, dest)
        if ok:
            node["pdf_path"] = dest
            print(f"  [PDF] ✓ {dest}")
        else:
            print(f"  [PDF] ✗ failed for {arxiv_id}")
        time.sleep(0.5)   # be polite to arXiv

    # ── graph helpers ───────────────────────────────────────────────

    def _add_node(self, node):
        if node and node.get("paperId"):
            pid = node["paperId"]
            if pid not in self._db["nodes"]:
                self._db["nodes"][pid] = node
            return pid
        return None

    def _add_edge(self, from_id, to_id, relation="cites"):
        edge = {"from": from_id, "to": to_id, "relation": relation}
        if edge not in self._db["edges"]:
            self._db["edges"].append(edge)

    # ── seed resolution ─────────────────────────────────────────────

    def _resolve_seed(self, title):
        """Find and store the seed paper node."""
        print(f"[CitationFetcher] Searching seed: '{title}'")
        results = SemanticScholar.search(title, limit=1)
        if not results:
            print("  → Seed not found on Semantic Scholar")
            return None

        seed_raw  = results[0]
        seed_node = _s2_to_node(seed_raw)
        if not seed_node:
            return None

        seed_id = self._add_node(seed_node)
        self._db["seed"] = {"title": seed_node["title"], "paperId": seed_id}
        print(f"  → Found: {seed_node['title']} ({seed_id})")
        return seed_id

    # ── BFS expansion ───────────────────────────────────────────────

    def build_network(
        self,
        seed_title,
        depth=1,
        max_per_hop=25,
        download_pdfs=False,
        include_references=True,
        include_citations=True,
    ):
        """
        BFS-expand the citation network around seed_title.

        depth=1  → only direct citations / references of the seed
        depth=2  → also their citations / references (can be large)

        Returns the full network dict.
        """
        seed_id = self._resolve_seed(seed_title)
        if not seed_id:
            return self._db

        visited  = set()
        queue    = deque([(seed_id, 0)])

        while queue:
            paper_id, hop = queue.popleft()

            if paper_id in visited or hop > depth:
                continue
            visited.add(paper_id)

            node = self._db["nodes"].get(paper_id)
            if not node:
                continue

            print(f"\n[Hop {hop}] {node.get('title', paper_id)[:70]}")

            # ── Download PDF if requested ─────────────────────────
            if download_pdfs:
                self._maybe_download_pdf(node)

            # ── Fetch papers that CITE this paper ─────────────────
            if include_citations:
                print(f"  Fetching citing papers (max {max_per_hop})…")
                citing = SemanticScholar.citing(paper_id, limit=max_per_hop)
                for cp in citing:
                    cn = _s2_to_node(cp)
                    cid = self._add_node(cn)
                    if cid:
                        self._add_edge(cid, paper_id, "cites")
                        node["cited_by_ids"].append(cid)
                        if hop + 1 <= depth:
                            queue.append((cid, hop + 1))
                print(f"  → {len(citing)} citing papers")
                time.sleep(0.4)

            # ── Fetch papers this paper REFERENCES ────────────────
            if include_references:
                print(f"  Fetching references (max {max_per_hop})…")
                refs = SemanticScholar.references(paper_id, limit=max_per_hop)
                for rp in refs:
                    rn = _s2_to_node(rp)
                    rid = self._add_node(rn)
                    if rid:
                        self._add_edge(paper_id, rid, "cites")
                        node["citing_ids"].append(rid)
                        if hop + 1 <= depth:
                            queue.append((rid, hop + 1))
                print(f"  → {len(refs)} references")
                time.sleep(0.4)

        print(f"\n[CitationFetcher] Network: {len(self._db['nodes'])} nodes, "
              f"{len(self._db['edges'])} edges")
        return self._db

    # ── concept summary ─────────────────────────────────────────────

    def concept_dict(self, top_n=50):
        """
        Aggregate concept keywords from ALL nodes in the network.
        Returns {keyword: frequency} dict, useful for the HTML sidebar.
        """
        counter = Counter()
        for node in self._db["nodes"].values():
            counter.update(node.get("concepts", []))
        return dict(counter.most_common(top_n))

    # ── convenience: get PDF paths for downstream processing ────────

    def downloaded_pdfs(self):
        """Return list of (paper_id, pdf_path) for all downloaded PDFs."""
        return [
            (pid, node["pdf_path"])
            for pid, node in self._db["nodes"].items()
            if node.get("pdf_path") and os.path.isfile(node["pdf_path"])
        ]


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build citation network for a paper")
    parser.add_argument("title",         help="Seed paper title")
    parser.add_argument("--output-dir",  default="output/network", help="Output directory")
    parser.add_argument("--depth",       type=int, default=1)
    parser.add_argument("--max-per-hop", type=int, default=25)
    parser.add_argument("--download",    action="store_true", help="Download PDFs")
    args = parser.parse_args()

    fetcher = CitationFetcher(output_dir=args.output_dir)
    net     = fetcher.build_network(
        seed_title    = args.title,
        depth         = args.depth,
        max_per_hop   = args.max_per_hop,
        download_pdfs = args.download,
    )
    fetcher.save()

    concepts = fetcher.concept_dict()
    print("\nTop 10 concepts across network:")
    for kw, freq in list(concepts.items())[:10]:
        print(f"  {kw:25s}  {freq}")
