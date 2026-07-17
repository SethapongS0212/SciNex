"""
main.py — Unified PDF parser entry point
=========================================

TWO WAYS TO RUN:

  ── LOCAL PDF ──────────────────────────────────────────────────────
    python main.py paper.pdf              # full pipeline
    python main.py paper.pdf --fast       # skip LLM + citations
    python main.py paper.pdf --no-llm     # skip LLM only
    python main.py paper.pdf --no-citations

  ── SCICLAIMEVAL (auto-download from HuggingFace) ──────────────────
    python main.py --paper 1810.04805              # one paper by arXiv ID
    python main.py --paper 1810.04805 1706.03762   # multiple papers
    python main.py --domain NLP                    # all NLP papers
    python main.py --domain ML --limit 5           # first 5 ML papers
    python main.py --all                           # all 180 papers

  All pipeline flags work in both modes:
    --fast  --no-llm  --no-citations  --citation-limit N

  One-time SciClaimEval setup:
    pip install datasets huggingface_hub
    huggingface-cli login

OUTPUT STRUCTURE:
    output/<paper_stem>/citation_network.json   <- shared across modes
    output/<paper_stem>/default/output.html
    output/<paper_stem>/fast/output.html
    output/<paper_stem>/no-llm/output.html
    output/<paper_stem>/no-citations/output.html
    output/<paper_stem>/no-llm_no-citations/output.html
"""

import os
import sys
import json
import re
import time
import argparse
import requests

from config import OUTPUT_DIR, CITATION_LIMIT

from parser.pdf_loader import load_pdf
from parser.layout import extract_blocks
from parser.table_extractor import (
    extract_tables, dedup_tables, fix_caption_header_tables
)
from parser.structure_builder import (
    build_structure, build_table_cell_set, is_table_data_paragraph, is_table_caption
)
from parser.html_generator import to_html

from citation.network import (
    get_or_build_network,
    build_concept_dict_from_network,
    find_similar_papers
)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
PDF_DIR       = "pdfs"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}"
PEERJ_PDF_URL = "https://peerj.com/articles/{article_id}.pdf"
ACL_PDF_URL   = "https://aclanthology.org/{paper_id}.pdf"
REQUEST_DELAY = 3.0    # seconds between downloads — be polite to ArXiv/PeerJ
ACL_PDF_DIR   = os.path.join(OUTPUT_DIR, "acl", "pdfs")


# ─────────────────────────────────────────────────────────────
# TEXT CLEANER — imported from parser/text_cleaner.py
# ─────────────────────────────────────────────────────────────
from parser.text_cleaner import clean_block as regex_clean_block, clean_document


# ─────────────────────────────────────────────────────────────
# LLM REFINEMENT (optional — needs GPU)
# ─────────────────────────────────────────────────────────────
def refine_document_llm(doc):
    from parser.llm_refiner import refine_blocks_batch
    all_blocks = [b for s in doc["sections"] for b in s["content"]]
    refined    = refine_blocks_batch(all_blocks)
    idx = 0
    for section in doc["sections"]:
        n = len(section["content"])
        section["content"] = refined[idx : idx + n]
        idx += n
    return doc


def refine_document_fast(doc):
    from parser.text_cleaner import clean_document
    return clean_document(doc)


# ─────────────────────────────────────────────────────────────
# SEMANTIC ENRICHMENT
# ─────────────────────────────────────────────────────────────
def enrich_with_concepts(doc, concept_dict):
    if not concept_dict:
        return doc
    for section in doc["sections"]:
        for block in section["content"]:
            if block.get("type") != "paragraph":
                continue
            text_lower = block.get("text", "").lower()
            matched = [kw for kw in concept_dict if kw in text_lower]
            if matched:
                block["concepts"] = matched
    return doc


# ─────────────────────────────────────────────────────────────
# TABLE REBUILD
# ─────────────────────────────────────────────────────────────
def rebuild_tables(doc, all_tables):
    all_tables   = fix_caption_header_tables(all_tables)
    clean_tables = dedup_tables(all_tables)
    cell_set     = build_table_cell_set(clean_tables)

    for section in doc["sections"]:
        section["content"] = [
            item for item in section.get("content", [])
            if item.get("type") != "table"
        ]

    tables_to_place = list(clean_tables)
    for section in doc["sections"]:
        new_content = []
        for item in section.get("content", []):
            itype = item.get("type")
            text  = item.get("text", "")

            if itype == "paragraph" and is_table_data_paragraph(text, cell_set, threshold=4):
                continue

            if itype == "figure" and is_table_caption(text):
                if tables_to_place:
                    tbl = dict(tables_to_place.pop(0))
                    tbl["caption"] = text
                    new_content.append(tbl)
                continue

            new_content.append(item)
        section["content"] = new_content

    if tables_to_place:
        target = None
        for s in doc["sections"]:
            if not s.get("heading", "").lower().startswith("ref"):
                target = s
        if target:
            for tbl in tables_to_place:
                target["content"].append(tbl)

    return doc, clean_tables


# ─────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY HELPER
# ─────────────────────────────────────────────────────────────
def resolve_output_dirs(pdf_path: str, args) -> tuple:
    """
    Returns (paper_dir, out_dir).

    paper_dir — top-level folder shared across all modes for this paper
                e.g. output/attention/
    out_dir   — mode-specific subfolder for this run's outputs
                e.g. output/attention/fast/
    """
    if getattr(args, "output_dir", None):
        return args.output_dir, args.output_dir

    pdf_stem  = os.path.splitext(os.path.basename(pdf_path))[0]
    paper_dir = os.path.join(OUTPUT_DIR, pdf_stem)

    if args.fast:
        mode_suffix = "fast"
    else:
        parts = []
        if args.no_llm:       parts.append("no-llm")
        if args.no_citations: parts.append("no-citations")
        mode_suffix = "_".join(parts) if parts else "default"

    out_dir = os.path.join(paper_dir, mode_suffix)
    return paper_dir, out_dir


# ─────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(pdf_path: str, args) -> bool:
    """
    Run the full parse pipeline on a single PDF.
    Returns True on success, False on error.
    """
    paper_dir, out_dir = resolve_output_dirs(pdf_path, args)
    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)

    if args.fast:
        mode_label = "fast"
    else:
        parts = []
        if args.no_llm:       parts.append("no-llm")
        if args.no_citations: parts.append("no-citations")
        mode_label = "_".join(parts) if parts else "default"

    print(f"\n{'─'*55}")
    print(f"  PDF      : {pdf_path}")
    print(f"  Mode     : {mode_label}")
    print(f"  Output   : {out_dir}/output.html")
    print(f"  LLM      : {'SKIPPED' if args.no_llm else 'Qwen2.5-14B  [slow ~5-15 min]'}")
    print(f"  Citations: {'SKIPPED' if args.no_citations else 'Semantic Scholar  [slow ~1-5 min]'}")
    print(f"{'─'*55}\n")

    try:
        # ── 1. Load ───────────────────────────────────────────────
        print("1. Loading PDF...")
        doc = load_pdf(pdf_path)

        # ── 2. Layout ─────────────────────────────────────────────
        print("2. Extracting layout blocks...")
        blocks = extract_blocks(doc)
        print(f"   {len(blocks)} blocks extracted")

        # ── 3. Tables ─────────────────────────────────────────────
        print("3. Extracting tables (multi-strategy)...")
        tables = extract_tables(pdf_path, blocks=blocks)
        print(f"   {len(tables)} tables found")

        # ── 4. Citations ──────────────────────────────────────────
        concept_dict = {}
        network      = None
        # Shared across all modes for this paper — no re-fetch needed
        network_path = os.path.join(paper_dir, "citation_network.json")

        # Derive paper ID from PDF filename for direct S2 lookup
        pdf_stem  = os.path.splitext(os.path.basename(pdf_path))[0]
        paper_id  = pdf_stem  # e.g. "C16-1036", "P18-1001", "2020.pam-1.10"

        if args.no_citations:
            print("4. [SKIPPED] Citation fetch")
        else:
            print("4. Building citation network via Semantic Scholar...")
            print("   (Makes ~20-60 API calls — may take 1-5 min)")
            try:
                network = get_or_build_network(
                    title=pdf_stem.replace("_", " ").replace("-", " "),
                    cache_path=network_path,
                    citing_limit=args.citation_limit,
                    reference_limit=args.citation_limit,
                    external_id=paper_id,
                )
            except Exception as _e:
                print(f"   ⚠️  Citation fetch error ({_e}) — continuing without")
                network = None
            if network:
                concept_dict = build_concept_dict_from_network(network, top_n=60)
                print(f"   {len(network['nodes'])} network nodes, "
                      f"{len(concept_dict)} concept keywords")
                similar = find_similar_papers(network, top_n=5)
                if similar:
                    print("   Top related papers:")
                    for p_item in similar:
                        print(f"     [{p_item['relation']:8s}] "
                              f"{p_item['title'][:55]} ({p_item['year']})")
            else:
                print("   ⚠️  Paper not found on Semantic Scholar — "
                      "continuing without citation network")
                print("   (Run with --no-citations to suppress this)")

        # ── 5. Structure ──────────────────────────────────────────
        print("5. Building document structure...")
        known_entities = list(concept_dict.keys()) if concept_dict else []
        structured = build_structure(blocks, tables, known_entities=known_entities)
        print(f"   {len(structured['sections'])} sections, "
              f"{len(structured['metrics'])} metrics")

        if (not args.no_citations
                and structured.get("title")
                and not os.path.exists(network_path)
                and not network):
            print("   Re-fetching with real paper title...")
            network = get_or_build_network(
                title=structured["title"],
                cache_path=network_path,
                citing_limit=args.citation_limit,
                reference_limit=args.citation_limit,
                external_id=paper_id,
            )
            if network:
                concept_dict = build_concept_dict_from_network(network, top_n=60)
            else:
                print("   ⚠️  Still not found — proceeding without citation network")

        structured["citation_concepts"] = concept_dict

        # ── 6. Text refinement ────────────────────────────────────
        if args.no_llm:
            print("6. Cleaning text (regex — instant, no GPU)...")
            structured = refine_document_fast(structured)
        else:
            print("6. Refining with Qwen2.5-14B LLM (this will take a while)...")
            structured = refine_document_llm(structured)

        # ── 7. Concept enrichment ─────────────────────────────────
        print("7. Enriching paragraphs with concept keywords...")
        structured = enrich_with_concepts(structured, concept_dict)

        # ── 8. Table rebuild ──────────────────────────────────────
        print("8. Rebuilding and placing tables...")
        structured, clean_tables = rebuild_tables(structured, tables)
        print(f"   {len(clean_tables)} tables placed")

        # ── 9. Save JSON ──────────────────────────────────────────
        json_path = os.path.join(out_dir, "output.json")
        print(f"9. Saving JSON -> {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)

        # ── 10. Generate HTML ─────────────────────────────────────
        html_path = os.path.join(out_dir, "output.html")
        print(f"10. Generating HTML -> {html_path}")
        html = to_html(structured)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # ── Done ──────────────────────────────────────────────────
        print(f"\n{'─'*55}")
        print("DONE")
        print(f"  Tables extracted  : {len(clean_tables)}")
        print(f"  Concept keywords  : {len(concept_dict)}")
        print(f"  OUTPUT            : {html_path}  <- open this")
        print(f"  JSON data         : {json_path}")
        if os.path.exists(network_path):
            print(f"  Citation graph    : {network_path}")
        print(f"{'─'*55}\n")
        return True

    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────
# SCICLAIMEVAL: PDF DOWNLOAD
# ─────────────────────────────────────────────────────────────
def _is_arxiv_id(paper_id: str) -> bool:
    """ArXiv IDs: "2310.12345" or "cs/0601001" — start with digit or contain /"""
    return "/" in paper_id or (len(paper_id) > 4 and paper_id[:4].isdigit())


def _is_acl_id(paper_id: str) -> bool:
    """ACL Anthology IDs: "N15-2004", "P17-1128", "2024.eacl-short.2", etc.
    """
    import re
    return bool(
        re.match(r"^[A-Z]\d{2}-\d{4}$", paper_id)
        or re.match(r"^\d{4}\.[a-z]", paper_id)
    )


def _is_peerj_id(paper_id: str) -> bool:
    """PeerJ IDs: "peerj-1234" or plain numeric."""
    return paper_id.startswith("peerj") or (
        not _is_arxiv_id(paper_id)
        and not _is_acl_id(paper_id)
        and paper_id.replace("-", "").isalnum()
    )


def _build_pdf_url(paper_id: str) -> str:
    if _is_acl_id(paper_id):
        return ACL_PDF_URL.format(paper_id=paper_id)
    if _is_arxiv_id(paper_id):
        return ARXIV_PDF_URL.format(paper_id=paper_id)
    # PeerJ fallback
    article_id = paper_id.replace("peerj-", "").replace("peerj.", "")
    return PEERJ_PDF_URL.format(article_id=article_id)


def download_pdf(paper_id: str, dest_path: str) -> bool:
    url = _build_pdf_url(paper_id)
    print(f"  Downloading: {url}")
    try:
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "pdf-parser-research/1.0 (academic use)"
        })
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and resp.content[:4] != b"%PDF":
            print(f"  ⚠️  Doesn't look like a PDF (Content-Type: {content_type})")
            return False
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        print(f"  ✅ Saved {len(resp.content) // 1024} KB → {dest_path}")
        return True
    except requests.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        return False


def _resolve_acl_paper_query(query: str) -> tuple[str, str] | None:
    """
    Resolve an ACL --paper query to (paper_id, pdf_path).

    Query order:
      1. Existing ACL PDF by exact paper ID
      2. ACL title lookup via acl_title_index.json
      3. Exact ACL paper ID, downloading if the PDF is missing
    """
    query = query.strip()
    if not query:
        return None

    pdf_dir = ACL_PDF_DIR
    os.makedirs(pdf_dir, exist_ok=True)

    if _is_acl_id(query):
        pdf_path = os.path.join(pdf_dir, f"{query}.pdf")
        if os.path.exists(pdf_path):
            return query, pdf_path

    try:
        from acl_index import ACLIndex
        resolved_id = ACLIndex("acl_title_index.json").load_or_build().lookup(query)
    except Exception as e:
        print(f"  ⚠️  ACL title lookup failed for {query!r}: {e}")
        resolved_id = None

    paper_id = resolved_id or (query if _is_acl_id(query) else None)
    if not paper_id:
        return None

    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    if os.path.exists(pdf_path):
        return paper_id, pdf_path

    print(f"  ACL PDF not found locally for {paper_id}; downloading...")
    return (paper_id, pdf_path) if download_pdf(paper_id, pdf_path) else None


def run_acl_papers(args, papers: list[tuple[str, str]]) -> bool:
    results = {"ok": [], "parse_failed": []}

    for i, (paper_id, pdf_path) in enumerate(papers, 1):
        print(f"\n{'─'*55}")
        print(f"[{i}/{len(papers)}] {paper_id}  (ACL)")
        print(f"  PDF: {pdf_path}")

        ok = run_pipeline(pdf_path, args)
        if ok:
            results["ok"].append(paper_id)
        else:
            results["parse_failed"].append(paper_id)

    print(f"\n{'═'*55}")
    print("ACL PAPERS DONE")
    print(f"  ✅ Parsed successfully : {len(results['ok'])}")
    print(f"  ❌ Parser failed       : {len(results['parse_failed'])}")
    if results["parse_failed"]:
        print("\n  Parser failures:")
        for pid in results["parse_failed"]:
            print(f"    {pid}")
    print(f"{'═'*55}\n")
    return not results["parse_failed"]


# ─────────────────────────────────────────────────────────────
# SCICLAIMEVAL: DATASET LOADER
# ─────────────────────────────────────────────────────────────
def load_sciclaimeval_papers(args) -> list:
    print("Loading SciClaimEval dataset from HuggingFace...")
    print("(If this fails, run: huggingface-cli login)\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run: pip install datasets huggingface_hub")
        sys.exit(1)

    try:
        dataset = load_dataset("alabnii/sciclaimeval-shared-task", split="dev")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    seen          = set()
    papers        = []
    requested_ids = set(args.paper) if args.paper else None

    for sample in dataset:
        paper_id = sample.get("paper_id", "").strip()
        domain   = sample.get("domain",   "").strip()

        if not paper_id or paper_id in seen:
            continue
        if requested_ids and paper_id not in requested_ids:
            continue
        if args.domain and domain != args.domain:
            continue

        seen.add(paper_id)
        papers.append((paper_id, domain))

    if requested_ids:
        not_found = requested_ids - seen
        if not_found:
            print(f"⚠️  Not found in SciClaimEval: {not_found}")

    if args.limit:
        papers = papers[:args.limit]

    print(f"Found {len(papers)} unique papers")
    if args.paper:  print(f"  Paper filter : {args.paper}")
    if args.domain: print(f"  Domain filter: {args.domain}")
    print()
    return papers


# ─────────────────────────────────────────────────────────────
# CITATION-ONLY FETCH  (for papers already parsed but missing network)
# ─────────────────────────────────────────────────────────────
def _looks_like_authors(text: str) -> bool:
    """Detect if an extracted "title" is actually an author line.
    Author lines typically contain: digits after words (affiliations),
    special symbols (∗†‡∇⋄♠♢), or multiple comma-separated names.
    """
    import re
    if not text:
        return False
    # Affiliation numbers glued to words: "Baldwin1,3" "Verspoor2,1"
    if re.search(r'[A-Za-z]\d', text):
        return True
    # Special author-line symbols
    if re.search(r'[\u2217\u2020\u2021\u2207\u22c4\u2660\u2662\u2605\u25c6\u0394]', text):
        return True
    return False


def _fetch_missing_citations(paper_id: str, pdf_path: str,
                             paper_dir: str, args) -> None:
    """
    Fetch and save citation_network.json for a paper that was already
    parsed but whose citation network is missing (e.g. due to a prior
    API failure). Does not re-run the full pipeline.
    """
    network_path = os.path.join(paper_dir, "citation_network.json")

    # Try to get the title from the existing output.json first.
    # Check all mode subdirs — use whichever exists.
    title = None
    for mode in ["default", "no-llm", "fast", "no-citations", "no-llm_no-citations"]:
        json_path = os.path.join(paper_dir, mode, "output.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    candidate = json.load(f).get("title", "")
                if candidate and not _looks_like_authors(candidate):
                    title = candidate
                    break
            except Exception:
                pass

    if not title:
        # Fall back to PDF stem
        title = os.path.splitext(os.path.basename(pdf_path))[0]\
                    .replace("_", " ").replace("-", " ")

    print(f"  Fetching citation network for: {title!r}")

    # Rate limit buffer — give Semantic Scholar a moment before each retry
    time.sleep(REQUEST_DELAY)

    try:
        from citation.network import (
            get_or_build_network, build_concept_dict_from_network
        )
        network = get_or_build_network(
            title=title,
            cache_path=network_path,
            citing_limit=args.citation_limit,
            reference_limit=args.citation_limit,
            external_id=paper_id,    # ACL/arXiv direct lookup — bypasses title search
        )
        if network:
            concept_dict = build_concept_dict_from_network(network, top_n=60)
            print(f"  ✅ Citation network saved ({len(network['nodes'])} nodes, "
                  f"{len(concept_dict)} concepts)")
        else:
            print("  ⚠️  Citation network still not found — "
                  "paper may not be indexed on Semantic Scholar")
    except Exception as e:
        print(f"  ❌ Citation fetch failed: {e}")


# ─────────────────────────────────────────────────────────────
# SCICLAIMEVAL BATCH RUNNER
# ─────────────────────────────────────────────────────────────
def run_sciclaimeval(args):
    papers = load_sciclaimeval_papers(args)
    if not papers:
        print("No papers to process.")
        sys.exit(0)

    os.makedirs(PDF_DIR, exist_ok=True)
    results = {"ok": [], "download_failed": [], "parse_failed": [], "skipped": []}

    for i, (paper_id, domain) in enumerate(papers, 1):
        print(f"\n{'─'*55}")
        print(f"[{i}/{len(papers)}] {paper_id}  (domain: {domain})")

        safe_id  = paper_id.replace("/", "_")
        pdf_path = os.path.join(PDF_DIR, f"{safe_id}.pdf")

        paper_dir, out_dir = resolve_output_dirs(pdf_path, args)
        html_path    = os.path.join(out_dir, "output.html")
        network_path = os.path.join(paper_dir, "citation_network.json")

        # Check if output already exists
        html_exists    = os.path.exists(html_path)
        network_exists = os.path.exists(network_path)

        if args.skip_existing and html_exists and network_exists:
            # Fully complete — skip everything
            print(f"  ⏭  Already complete ({out_dir}) — skipping")
            results["skipped"].append(paper_id)
            continue

        if args.skip_existing and html_exists and not network_exists and args.no_citations:
            # HTML exists but citation was skipped intentionally this mode — skip
            print(f"  ⏭  Already processed, no citations in this mode — skipping")
            results["skipped"].append(paper_id)
            continue

        if args.skip_existing and html_exists and not network_exists:
            # HTML exists but citation_network.json is missing — fetch citations only
            print(f"  ⚠️  Output exists but citation_network.json missing — fetching citations...")
            _fetch_missing_citations(paper_id, pdf_path, paper_dir, args)
            results["skipped"].append(paper_id)
            continue

        if not os.path.exists(pdf_path):
            ok = download_pdf(paper_id, pdf_path)
            if not ok:
                results["download_failed"].append(paper_id)
                continue
            time.sleep(REQUEST_DELAY)
        else:
            print(f"  PDF already on disk: {pdf_path}")

        ok = run_pipeline(pdf_path, args)
        if ok:
            results["ok"].append(paper_id)
            json_path    = os.path.join(out_dir, "output.json")
            network_path = os.path.join(paper_dir, "citation_network.json")
            missing = [f for f in [html_path, json_path, network_path]
                       if not os.path.exists(f)]
            if missing:
                print(f"  ⚠️  Missing outputs: {missing}")
        else:
            results["parse_failed"].append(paper_id)

    print(f"\n{'═'*55}")
    print("ALL DONE")
    print(f"  ✅ Parsed successfully : {len(results['ok'])}")
    print(f"  ⏭  Skipped (exists)   : {len(results['skipped'])}")
    print(f"  ❌ Download failed     : {len(results['download_failed'])}")
    print(f"  ❌ Parser failed       : {len(results['parse_failed'])}")
    if results["download_failed"]:
        print("\n  Download failures:")
        for pid in results["download_failed"]: print(f"    {pid}")
    if results["parse_failed"]:
        print("\n  Parser failures:")
        for pid in results["parse_failed"]: print(f"    {pid}")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="PDF -> HTML research paper parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input
    inp = p.add_argument_group("Input (choose one)")
    inp.add_argument("pdf_path", nargs="?", default=None,
                     help="Path to a local PDF file.")
    inp.add_argument("--paper", type=str, nargs="+", default=None, metavar="ID",
                     help="ACL title/ID or SciClaimEval arXiv/PeerJ paper ID(s).")
    inp.add_argument("--domain", type=str, default=None,
                     choices=["ml", "nlp", "peerj"],
                     help="SciClaimEval: all papers in this domain.")
    inp.add_argument("--all", action="store_true",
                     help="SciClaimEval: all 180 papers.")

    # SciClaimEval options
    sci = p.add_argument_group("SciClaimEval options")
    sci.add_argument("--limit", type=int, default=None,
                     help="Cap number of SciClaimEval papers to process.")
    sci.add_argument("--skip-existing", action="store_true", default=True,
                     help="Skip papers already processed in this mode (default: on).")

    # Pipeline flags
    pipe = p.add_argument_group("Pipeline flags")
    pipe.add_argument("--fast", action="store_true",
                      help="Skip LLM + citations. Fastest, no GPU needed.")
    pipe.add_argument("--no-llm", action="store_true",
                      help="Skip Qwen2.5-14B LLM refinement.")
    pipe.add_argument("--no-citations", action="store_true",
                      help="Skip Semantic Scholar citation fetch.")
    pipe.add_argument("--citation-limit", "-c", type=int, default=CITATION_LIMIT,
                      help=f"Max related papers to fetch (default: {CITATION_LIMIT}).")
    pipe.add_argument("--output-dir", "-o", default=None,
                      help="Override output directory (local PDF mode only).")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    sciclaimeval_mode = bool(args.paper or args.domain or args.all)
    local_mode        = bool(args.pdf_path)

    if local_mode and sciclaimeval_mode:
        print("ERROR: Provide either a local pdf_path OR "
              "--paper / --domain / --all, not both.")
        sys.exit(1)

    if not local_mode and not sciclaimeval_mode:
        print(
            "ERROR: No input specified.\n\n"
            "  Local PDF  :  python main.py paper.pdf [--fast]\n"
            "  By paper ID:  python main.py --paper 1810.04805 [--fast]\n"
            "  By domain  :  python main.py --domain NLP [--fast]\n"
            "  All papers :  python main.py --all [--fast]\n"
        )
        sys.exit(1)

    # Expand --fast before anything else
    if args.fast:
        args.no_llm = True
        args.no_citations = True

    if local_mode:
        if not os.path.isfile(args.pdf_path):
            print(f"ERROR: File not found: {args.pdf_path}")
            sys.exit(1)
        success = run_pipeline(args.pdf_path, args)
        sys.exit(0 if success else 1)

    if args.paper and not args.domain and not args.all:
        acl_papers = []
        unresolved = []
        for query in args.paper:
            resolved = _resolve_acl_paper_query(query)
            if resolved:
                acl_papers.append(resolved)
            else:
                unresolved.append(query)

        if acl_papers:
            ok = run_acl_papers(args, acl_papers)
            if not unresolved:
                sys.exit(0 if ok else 1)

        args.paper = unresolved

    run_sciclaimeval(args)


if __name__ == "__main__":
    main()
