"""
acl_pipeline.py
---------------
End-to-end pipeline:
  1. Fetch & parse CS-NER IOB files → paper titles + entity annotations
  2. Look up ACL paper IDs using local title index (acl_index.py)
  3. Download PDFs from aclanthology.org
  4. Write Entity CSV per paper (for fixed extractor)

Entity CSVs are written immediately from IOB data — they never depend on
finding a paper ID. Paper ID is only required for PDF download.

Usage:
    # Step 1: Build the title index once (takes a few minutes)
    python acl_pipeline.py --build-index

    # Step 2: Run the full pipeline
    python acl_pipeline.py --all --out-dir output/acl --limit 10

    # Or step by step:
    python acl_pipeline.py --parse-iob --out-dir output/acl        # parse IOB → papers.json + entity CSVs
    python acl_pipeline.py --resolve-ids --out-dir output/acl      # add paper IDs from index
    python acl_pipeline.py --download-pdfs --out-dir output/acl    # download PDFs

    # Then run HTML parser on each PDF:
    python main.py --paper output/acl/pdfs/P18-1234.pdf --paper-id P18-1234 --no-llm
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

CSNER_ACL_BASE  = "https://raw.githubusercontent.com/jd-coderepos/contributions-ner-cs/main/acl"
CSNER_IOB_FILES = ["train.data", "dev.data", "test.data"]
ACL_PDF_URL     = "https://aclanthology.org/{paper_id}.pdf"

CSNER_TO_CEO_TYPE = {
    "solution":         "Model",
    "method":           "Method",
    "dataset":          "Dataset",
    "research_problem": "Task",
    "tool":             "Tool",
    "resource":         "Resource",
    "language":         "Language",
}


# ── Step 1: Download & parse IOB ──────────────────────────────────────────────

def fetch_iob() -> str:
    parts = []
    for fname in CSNER_IOB_FILES:
        url = f"{CSNER_ACL_BASE}/{fname}"
        logger.info(f"Fetching {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "acl_pipeline/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                parts.append(r.read().decode("utf-8", errors="replace"))
        except Exception as e:
            logger.warning(f"  Could not fetch {fname}: {e}")
    return "\n\n".join(parts)


_ENTITY_TYPES = {"solution","method","dataset","research_problem","tool","resource","language"}


def _parse_tag(tag: str):
    """Return (prefix, entity_type) from BIOES tag. Tags are uppercase in the dataset."""
    tag = tag.strip()
    if tag == "O" or not tag:
        return None, None
    if "-" in tag:
        prefix, etype = tag.split("-", 1)
        return prefix.upper(), etype.lower()
    return None, None


def _clean_title(title: str) -> str:
    title = re.sub(r'\s+-\s+', '-', title)
    title = re.sub(r"\s+'\s+", "'", title)
    title = re.sub(r'\s+([,:;])', r'\1', title)
    return re.sub(r'\s{2,}', ' ', title).strip()


def parse_iob(text: str) -> list[dict]:
    """
    Parse BIOES-format IOB text into paper dicts.
    Format in CS-NER ACL dataset: 'token\\tTAG' per line, blank lines between papers.
    Tags: S-SOLUTION, B-METHOD, I-METHOD, E-METHOD, O  (uppercase, BIOES scheme).
    No paper IDs in the files — only titles + annotations.
    Deduplicates by normalized title (train/dev/test may overlap).
    """
    papers = []
    current_tokens   = []
    current_entities = []
    in_entity        = False
    entity_tokens    = []
    entity_type      = None
    token_idx        = 0

    def flush_entity():
        nonlocal in_entity, entity_tokens, entity_type
        if in_entity and entity_tokens:
            current_entities.append({
                "text":  " ".join(entity_tokens),
                "type":  entity_type,
                "start": token_idx - len(entity_tokens),
            })
        in_entity = False; entity_tokens = []; entity_type = None

    def flush_paper():
        nonlocal current_tokens, current_entities, token_idx
        flush_entity()
        if current_tokens:
            papers.append({
                "paper_id": None,
                "title":    _clean_title(" ".join(current_tokens)),
                "entities": list(current_entities),
            })
        current_tokens[:] = []; current_entities[:] = []; token_idx = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paper(); continue
        if line.startswith("#"):
            continue

        parts = line.split("\t") if "\t" in line else line.split()
        if not parts:
            continue
        if len(parts) < 2:
            parts = [parts[0], "O"]

        token, tag = parts[0], parts[1]
        prefix, etype = _parse_tag(tag)
        current_tokens.append(token)

        if prefix == "B":
            flush_entity()
            in_entity = True; entity_tokens = [token]; entity_type = etype
        elif prefix == "I" and in_entity and etype == entity_type:
            entity_tokens.append(token)
        elif prefix == "E" and in_entity and etype == entity_type:
            entity_tokens.append(token); flush_entity()
        elif prefix == "S":
            flush_entity()
            current_entities.append({"text": token, "type": etype, "start": token_idx})
        else:
            flush_entity()

        token_idx += 1

    flush_paper()
    # Deduplicate by normalized title (train/dev/test splits may overlap)
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if len(unique) < len(papers):
        logger.info(f"Parsed {len(papers)} papers, {len(papers)-len(unique)} duplicates removed → {len(unique)} unique")
    else:
        logger.info(f"Parsed {len(papers)} papers from IOB")
    return unique


# ── Step 2: Resolve paper IDs ─────────────────────────────────────────────────

_ACL_ID_FROM_URL_RE = re.compile(
    r'aclanthology\.org/([0-9]{4}\.\S+?|[A-Z]\d{2}-\d{4})'
    r'(?:\.pdf|\.bib|/?["\s<]|$)'
)
_DOI_ACL_RE = re.compile(r'10\.18653/v1/(\S+?)(?:["\s]|$)')


def _dblp_lookup(title: str) -> str | None:
    """Search DBLP by title, extract ACL Anthology ID from the paper URL."""
    params = urllib.parse.urlencode({"q": title, "format": "json", "h": "3"})
    url = f"https://dblp.org/search/publ/api?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "acl_pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        for hit in data.get("result", {}).get("hits", {}).get("hit", []):
            info = hit.get("info", {})
            ee = info.get("ee", "")
            if isinstance(ee, list):
                ee = " ".join(ee)
            m = _ACL_ID_FROM_URL_RE.search(ee)
            if m:
                return m.group(1)
            doi = info.get("doi", "")
            m = _DOI_ACL_RE.search(doi)
            if m:
                return m.group(1)
    except Exception as e:
        logger.debug(f"DBLP failed for '{title[:50]}': {e}")
    return None


def _s2_lookup(title: str) -> str | None:
    """Search Semantic Scholar, return ACL ID from externalIds if present."""
    params = urllib.parse.urlencode({
        "query": title, "limit": "3",
        "fields": "title,externalIds,openAccessPdf",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "acl_pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        for paper in data.get("data", []):
            ext = paper.get("externalIds", {})
            if ext.get("ACL"):
                return ext["ACL"]
            pdf_url = (paper.get("openAccessPdf") or {}).get("url", "")
            m = _ACL_ID_FROM_URL_RE.search(pdf_url)
            if m:
                return m.group(1)
    except Exception as e:
        logger.debug(f"S2 failed for '{title[:50]}': {e}")
    return None


def resolve_ids(papers: list[dict], index_path: str = "acl_title_index.json") -> list[dict]:
    """
    Look up ACL paper IDs for all papers.

    Order:
      1. Local title index (acl_title_index.json — built from anthology.json.gz)
      2. DBLP API  — covers workshop proceedings missing from anthology.json
      3. Semantic Scholar API — final fallback
    """
    from acl_index import ACLIndex
    idx = ACLIndex(index_path).load_or_build()

    found_index = found_dblp = found_s2 = still_missing = 0

    for paper in papers:
        if paper["paper_id"]:
            found_index += 1
            continue

        # 1. Local index
        pid = idx.lookup(paper["title"])
        if pid:
            paper["paper_id"] = pid
            found_index += 1
            continue

        # 2. DBLP
        pid = _dblp_lookup(paper["title"])
        if pid:
            paper["paper_id"] = pid
            found_dblp += 1
            logger.info(f"  [DBLP] '{paper['title'][:60]}' → {pid}")
            time.sleep(0.5)
            continue

        # 3. Semantic Scholar
        pid = _s2_lookup(paper["title"])
        if pid:
            paper["paper_id"] = pid
            found_s2 += 1
            logger.info(f"  [S2]   '{paper['title'][:60]}' → {pid}")
            time.sleep(0.5)
            continue

        still_missing += 1
        logger.debug(f"  [miss] '{paper['title'][:60]}'")

    total = len(papers)
    logger.info(
        f"IDs resolved: {found_index} index + {found_dblp} DBLP + {found_s2} S2 "
        f"= {found_index+found_dblp+found_s2}/{total} total "
        f"({still_missing} unresolved)"
    )
    return papers


# ── Step 3: Download PDFs ─────────────────────────────────────────────────────

def _scrape_pdf_url(paper_id: str) -> str | None:
    """Fetch the anthology page and return the first external PDF link found."""
    page_url = f"https://aclanthology.org/{paper_id}/"
    req = urllib.request.Request(page_url, headers={"User-Agent": "acl_pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        import re as _re
        for m in _re.finditer(r'href="(https?://[^"]+\.pdf)"', html):
            return m.group(1)
    except Exception as e:
        logger.debug(f"  scrape failed for {paper_id}: {e}")
    return None


def download_pdf(paper_id: str, pdf_dir: Path, delay: float = 1.0) -> bool:
    dest = pdf_dir / f"{paper_id}.pdf"
    if dest.exists():
        logger.debug(f"[skip] {paper_id}.pdf already exists")
        return True

    url = ACL_PDF_URL.format(paper_id=paper_id)
    req = urllib.request.Request(url, headers={"User-Agent": "acl_pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 1000:
            logger.warning(f"[skip] {paper_id} — response too small")
            return False
        dest.write_bytes(data)
        logger.info(f"[ok]   {paper_id}.pdf  ({len(data)//1024} KB)")
        time.sleep(delay)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # PDF not hosted on aclanthology.org — scrape the page for an external link
            fallback_url = _scrape_pdf_url(paper_id)
            if fallback_url:
                logger.info(f"  [fallback] {paper_id} → {fallback_url}")
                try:
                    freq = urllib.request.Request(fallback_url, headers={"User-Agent": "acl_pipeline/1.0"})
                    with urllib.request.urlopen(freq, timeout=60) as resp:
                        data = resp.read()
                    if len(data) < 1000:
                        logger.warning(f"[skip] {paper_id} — fallback response too small")
                        return False
                    dest.write_bytes(data)
                    logger.info(f"[ok]   {paper_id}.pdf  ({len(data)//1024} KB, via fallback)")
                    time.sleep(delay)
                    return True
                except Exception as fe:
                    logger.warning(f"[err]  {paper_id}: fallback failed: {fe}")
                    return False
        logger.warning(f"[err]  {paper_id}: HTTP {e.code}")
        return False
    except Exception as e:
        logger.warning(f"[err]  {paper_id}: {e}")
        return False


def download_all_pdfs(papers: list[dict], pdf_dir: Path, delay: float = 1.0) -> dict:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    have_id = [p for p in papers if p.get("paper_id")]
    no_id   = [p for p in papers if not p.get("paper_id")]

    logger.info(f"Downloading {len(have_id)} PDFs ({len(no_id)} skipped — no ID)")
    results = {}
    for p in have_id:
        pid = p["paper_id"]
        results[pid] = "ok" if download_pdf(pid, pdf_dir, delay=delay) else "fail"

    ok   = sum(1 for v in results.values() if v == "ok")
    fail = sum(1 for v in results.values() if v == "fail")
    logger.info(f"PDFs: {ok} downloaded, {fail} failed, {len(no_id)} skipped (no ID)")
    return results


# ── Step 4: Generate Entity CSVs ──────────────────────────────────────────────

_ABBREV_RE = re.compile(r'^(.+?)\s*\(([A-Z][A-Z0-9\-]{1,8})\)\s*$')


def _detect_abbreviation(text: str):
    m = _ABBREV_RE.match(text)
    return (m.group(1).strip(), m.group(2).strip()) if m else (text, "")


def _slugify(text: str) -> str:
    return re.sub(r'[^\w\-]', '_', text[:50]).strip('_')


def write_entity_csv(paper: dict, out_dir: Path):
    entities = paper.get("entities", [])
    if not entities:
        return

    pid  = paper.get("paper_id")
    name = pid or _slugify(paper["title"])

    # Migrate stale slugified dir → paper_id dir when ID is now known
    if pid:
        slug = _slugify(paper["title"])
        if slug != pid:
            slug_dir = out_dir / slug
            id_dir   = out_dir / pid
            if slug_dir.exists() and not id_dir.exists():
                try:
                    slug_dir.rename(id_dir)
                    old_csv = id_dir / f"Entity_{slug}.csv"
                    new_csv = id_dir / f"Entity_{pid}.csv"
                    if old_csv.exists() and not new_csv.exists():
                        old_csv.rename(new_csv)
                    logger.info(f"  Migrated dir {slug} → {pid}")
                except Exception as e:
                    logger.warning(f"  Could not migrate dir {slug}: {e}")

    csv_path = out_dir / name / f"Entity_{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []
    for ent in entities:
        text = ent["text"].strip().rstrip(":,;.!?")
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        name_clean, abbrev = _detect_abbreviation(text)
        rows.append({
            "Entity":       name_clean,
            "Abbreviation": abbrev,
            "Aliases":      "",
            "TP":           1,
            "NER_Type":     ent.get("type", ""),
            "CEO_Type":     CSNER_TO_CEO_TYPE.get(ent.get("type", ""), "Other"),
        })

    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Entity","Abbreviation","Aliases","TP","NER_Type","CEO_Type"])
        writer.writeheader()
        writer.writerows(rows)
    logger.debug(f"  CSV: {csv_path} ({len(rows)} entities)")


def write_all_entity_csvs(papers: list[dict], out_dir: Path):
    written = 0
    for paper in papers:
        write_entity_csv(paper, out_dir)
        written += 1
    logger.info(f"Wrote {written} Entity CSVs → {out_dir}/")


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_existing_id_map(papers_json: Path) -> dict[str, str]:
    """Return title → paper_id map from an existing papers.json so IDs survive re-parse."""
    if not papers_json.exists():
        return {}
    try:
        return {p["title"]: p["paper_id"] for p in load_papers(papers_json) if p.get("paper_id")}
    except Exception:
        return {}


def save_papers(papers: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)


def load_papers(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="ACL pipeline: IOB → Entity CSVs + PDFs")
    p.add_argument("--build-index",   action="store_true",
                   help="Build local title→ID index (run once before anything else)")
    p.add_argument("--all",           action="store_true",
                   help="Run full pipeline: parse IOB → entity CSVs → resolve IDs → download PDFs")
    p.add_argument("--parse-iob",     action="store_true",
                   help="Fetch IOB files + parse → papers.json + entity CSVs")
    p.add_argument("--resolve-ids",   action="store_true",
                   help="Look up paper IDs for all papers in papers.json")
    p.add_argument("--download-pdfs", action="store_true",
                   help="Download PDFs for all papers that have a paper_id")
    p.add_argument("--iob-file",      type=str, default=None,
                   help="Use a local IOB file instead of fetching from GitHub")
    p.add_argument("--out-dir",       type=str, default="output/acl")
    p.add_argument("--index",         type=str, default="acl_title_index.json",
                   help="Path to local title index (built with --build-index)")
    p.add_argument("--limit",         type=int, default=None,
                   help="Limit to first N papers (for testing)")
    p.add_argument("--delay",         type=float, default=1.0,
                   help="Seconds between HTTP requests (default 1.0)")
    args = p.parse_args()

    out_dir     = Path(args.out_dir)
    papers_json = out_dir / "papers.json"
    pdf_dir     = out_dir / "pdfs"

    # ── 0. Build index ────────────────────────────────────────────────────────
    if args.build_index:
        from acl_index import ACLIndex
        ACLIndex(args.index).build()
        return

    # ── 1. Parse IOB ──────────────────────────────────────────────────────────
    if args.all or args.parse_iob:
        # Preserve IDs from any previous run so re-parsing (e.g. with a larger --limit)
        # doesn't force re-resolution of already-looked-up papers.
        existing_ids = _load_existing_id_map(papers_json)

        if args.iob_file:
            iob_text = Path(args.iob_file).read_text(encoding="utf-8", errors="replace")
        else:
            iob_text = fetch_iob()

        papers = parse_iob(iob_text)
        if args.limit:
            papers = papers[:args.limit]
            logger.info(f"Limited to {args.limit} papers")

        preserved = 0
        for p in papers:
            if not p["paper_id"] and p["title"] in existing_ids:
                p["paper_id"] = existing_ids[p["title"]]
                preserved += 1
        if preserved:
            logger.info(f"Preserved {preserved} previously resolved paper IDs")

        save_papers(papers, papers_json)
        # NOTE: entity CSVs written AFTER ID resolution below so dirs use paper_id

    # ── 2. Resolve IDs ────────────────────────────────────────────────────────
    if args.all or args.resolve_ids:
        if not papers_json.exists():
            logger.error("papers.json not found — run --parse-iob first")
            sys.exit(1)
        papers = load_papers(papers_json)
        resolve_ids(papers, index_path=args.index)
        save_papers(papers, papers_json)
        found = sum(1 for p in papers if p["paper_id"])
        logger.info(f"Resolved: {found}/{len(papers)} papers have ACL IDs")

    # ── 3. Write entity CSVs (after ID resolution so dirs use paper_id) ───────
    if args.all or args.parse_iob:
        papers = load_papers(papers_json)
        write_all_entity_csvs(papers, out_dir)

    # ── 4. Download PDFs ──────────────────────────────────────────────────────
    if args.all or args.download_pdfs:
        if not papers_json.exists():
            logger.error("papers.json not found — run --parse-iob first")
            sys.exit(1)
        papers = load_papers(papers_json)
        download_all_pdfs(papers, pdf_dir, delay=args.delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    if papers_json.exists():
        papers = load_papers(papers_json)
        pdfs   = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
        found  = sum(1 for p in papers if p["paper_id"])
        csvs   = len(list(out_dir.rglob("Entity_*.csv")))

        print("\n── Summary ──────────────────────────────")
        print(f"  Papers in index:   {len(papers)}")
        print(f"  With ACL paper ID: {found}")
        print(f"  Entity CSVs:       {csvs}")
        print(f"  PDFs downloaded:   {pdfs}")
        print(f"  Output dir:        {out_dir}/")
        if pdfs > 0:
            print(f"\nNext step — parse PDFs to HTML:")
            for pdf in sorted(pdf_dir.glob("*.pdf"))[:5]:
                print(f"  python main.py {pdf} --no-llm")


if __name__ == "__main__":
    main()