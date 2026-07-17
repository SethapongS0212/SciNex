import requests

BASE = "https://api.semanticscholar.org/graph/v1"


# ─────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────
def search_paper(title):
    """Find the Semantic Scholar paper ID for a given title."""
    try:
        r = requests.get(
            f"{BASE}/paper/search",
            params={"query": title, "limit": 1, "fields": "paperId,title"},
            timeout=10
        )
        data = r.json().get("data", [])
        return data[0] if data else None
    except Exception as e:
        print("Search error:", e)
        return None


# ─────────────────────────────────────────────────────────────
# CITING PAPERS  (papers that cite the input paper)
# ─────────────────────────────────────────────────────────────
def get_citing_papers(title, limit=10):
    """
    Returns up to `limit` papers that CITE the paper with the
    given title, each with title + abstract.
    """
    paper = search_paper(title)
    if not paper:
        return []

    paper_id = paper.get("paperId")
    if not paper_id:
        return []

    try:
        r = requests.get(
            f"{BASE}/paper/{paper_id}/citations",
            params={
                "limit": limit,
                "fields": "citingPaper.title,citingPaper.abstract"
            },
            timeout=10
        )
        return r.json().get("data", [])
    except Exception as e:
        print("Citation fetch error:", e)
        return []


# ─────────────────────────────────────────────────────────────
# REFERENCE PAPERS  (papers the input paper cites)
# ─────────────────────────────────────────────────────────────
def get_reference_papers(title, limit=10):
    """
    Returns up to `limit` papers that ARE CITED BY the paper
    with the given title, each with title + abstract.

    These are the upstream sources — useful for pulling concepts
    and terminology the authors themselves relied on.
    """
    paper = search_paper(title)
    if not paper:
        return []

    paper_id = paper.get("paperId")
    if not paper_id:
        return []

    try:
        r = requests.get(
            f"{BASE}/paper/{paper_id}/references",
            params={
                "limit": limit,
                "fields": "citedPaper.title,citedPaper.abstract"
            },
            timeout=10
        )
        raw = r.json().get("data", [])

        # Normalise to the same shape as get_citing_papers so
        # concept_builder can consume both lists identically.
        normalised = []
        for item in raw:
            cited = item.get("citedPaper", {})
            normalised.append({"citingPaper": cited})   # reuse same key name

        return normalised
    except Exception as e:
        print("Reference fetch error:", e)
        return []


# ─────────────────────────────────────────────────────────────
# COMBINED FETCH (convenience wrapper used by main.py)
# ─────────────────────────────────────────────────────────────
def get_related_papers(title, limit=10):
    """
    Fetches both citing papers and reference papers in one call.
    Returns a combined deduplicated list.
    """
    citing    = get_citing_papers(title, limit)
    references = get_reference_papers(title, limit)
    return citing + references
