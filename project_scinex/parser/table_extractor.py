import camelot
import pdfplumber
import pytesseract
import cv2
import numpy as np
import re


# ─────────────────────────────────────────────────────────────
# TEXT TABLE DETECTION
# ─────────────────────────────────────────────────────────────
def split_row_smart(line):
    cols = re.split(r"\s{3,}", line)
    if len(cols) <= 1:
        cols = re.split(r"\s{2,}", line)
    if len(cols) <= 1:
        cols = re.findall(r"[A-Za-z\-]+|\d+\.\d+|\d+", line)
    return [c.strip() for c in cols if c.strip()]


def normalize_rows(rows):
    if not rows:
        return rows
    max_cols = max(len(r) for r in rows)
    return [r + [""] * (max_cols - len(r)) for r in rows]


def detect_text_tables(blocks):
    """Reconstruct tables from plain text blocks as a last-resort fallback."""
    tables = []
    current_table = []

    for block in blocks:
        text = block.strip() if isinstance(block, str) else block.get("text", "").strip()

        if len(text.split()) >= 5 and re.search(r"\d", text):
            current_table.append(text)
        else:
            if len(current_table) >= 2:
                rows = normalize_rows([split_row_smart(l) for l in current_table])
                if len(rows[0]) >= 2:
                    tables.append({
                        "headers": rows[0],
                        "rows": rows[1:],
                        "page": None,
                        "quality": 0.2
                    })
            current_table = []

    if len(current_table) >= 2:
        rows = normalize_rows([split_row_smart(l) for l in current_table])
        if len(rows[0]) >= 2:
            tables.append({
                "headers": rows[0],
                "rows": rows[1:],
                "page": None,
                "quality": 0.2
            })

    return tables


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def clean_cell(cell):
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def score_df(df):
    if df is None or df.empty:
        return 0
    total = df.size
    if total == 0:
        return 0
    non_empty = df.replace("", np.nan).notna().sum().sum()
    return non_empty / total


def is_valid_table(headers, rows):
    text = " ".join(headers + [cell for r in rows for cell in r])
    if re.search(r"(.)\1{5,}", text):
        return False
    tokens = text.split()
    if not tokens:
        return False
    short_ratio = sum(1 for t in tokens if len(t) <= 2) / len(tokens)
    if short_ratio > 0.6:
        return False
    return True


# ─────────────────────────────────────────────────────────────
# REFERENCE / ACKNOWLEDGEMENT TABLE FILTER  (NEW)
# ─────────────────────────────────────────────────────────────
_CITE_RE = re.compile(r'\[\d[\d,\s]*\]')


def is_reference_table(headers, rows):
    """
    Returns True when the 'table' is a mis-parsed reference list,
    acknowledgements block, or appendix.

    Heuristics (any one triggers rejection):
      1. First header contains "acknowledgement", "reference", "appendix".
      2. Combined text has > 10% [N] citation patterns by token count.
      3. First column is mostly [N] citation numbers.
    """
    header0 = str(headers[0]).lower().strip()
    if any(kw in header0 for kw in
           ["acknowledgement", "acknowledgment", "references", "appendix",
            "thank", "we are grateful"]):
        return True

    all_text = " ".join(str(h) for h in headers)
    all_text += " " + " ".join(str(c) for r in rows for c in r)
    tokens = all_text.split()
    if not tokens:
        return False

    cite_hits = len(_CITE_RE.findall(all_text))
    if cite_hits / max(len(tokens), 1) > 0.10:
        return True

    if rows:
        col0 = [str(r[0]).strip() for r in rows if r]
        cite_col = sum(1 for c in col0 if _CITE_RE.fullmatch(c))
        if cite_col / max(len(col0), 1) > 0.4:
            return True

    return False


# ─────────────────────────────────────────────────────────────
# TABLE DEDUPLICATION  (NEW)
# ─────────────────────────────────────────────────────────────
def _table_word_set(t):
    words = []
    for h in t["headers"]:
        words += re.findall(r'\w+', str(h).lower())
    for row in t["rows"]:
        for cell in row:
            words += re.findall(r'\w+', str(cell).lower())
    return set(words)


def _has_caption_header(t):
    """True when the first header cell is a 'Table N:' caption instead of a column name."""
    return bool(re.match(r'table\s*\d+', str(t["headers"][0]).lower().strip()))


def fix_caption_header_tables(tables):
    """
    When camelot/pdfplumber accidentally captures the 'Table N:' caption
    as the first header cell, the real column headers are actually stored
    in rows[0].  This function promotes them.

    Before:
      headers = ['Table 4: The Transformer...', '', '']
      rows    = [['Parser', 'Training', 'WSJ 23 F1'],   ← real headers
                 ['Vinyals et al.', 'WSJ only', '88.3'],
                 ...]
    After:
      headers = ['Parser', 'Training', 'WSJ 23 F1']
      rows    = [['Vinyals et al.', 'WSJ only', '88.3'], ...]
    """
    fixed = []
    for t in tables:
        if _has_caption_header(t) and t.get("rows"):
            new_t = dict(t)
            new_t["headers"] = list(t["rows"][0])
            new_t["rows"]    = list(t["rows"][1:])
            # Re-score since we now have proper headers
            new_t["quality"] = max(t.get("quality", 0.5), 0.6)
            fixed.append(new_t)
        else:
            fixed.append(t)
    return fixed


def dedup_tables(tables):
    """
    Remove duplicate table extractions.

    Two-stage strategy:
      Stage 1 — Jaccard overlap on all cell words (threshold 0.65).
                 Works well when the same table is extracted twice with
                 identical content.
      Stage 2 — Same-page / same-column-count check for tables whose
                 first header accidentally captured a 'Table N:' caption
                 row. These are always the lower-quality duplicate.

    Within each duplicate group, the table with:
      • higher quality score, AND
      • header[0] that does NOT look like a caption row
    is kept.
    """
    if not tables:
        return tables

    # Safety net: filter reference/acknowledgement tables even if they
    # slipped past the per-extractor checks.
    # Skip ref-filter for caption-anchored tables (already validated by presence of 'Table N:' caption)
    tables = [t for t in tables
              if t.get('source') == 'text_reconstruct'
              or not is_reference_table(t["headers"], t["rows"])]

    if not tables:
        return tables

    # Repair caption-header tables so dedup uses real column words.
    tables = fix_caption_header_tables(tables)

    def sort_key(t):
        caption_penalty = 1 if _has_caption_header(t) else 0
        return (-t["quality"], caption_penalty)

    tables = sorted(tables, key=sort_key)

    unique = []
    for candidate in tables:
        cand_words = _table_word_set(candidate)
        cand_cols  = len(candidate["headers"])
        cand_page  = candidate.get("page") or 0
        cand_is_cap = _has_caption_header(candidate)

        is_dup = False
        for kept in unique:
            kept_cols = len(kept["headers"])
            kept_page = kept.get("page") or 0

            # Stage 1: content Jaccard overlap
            if kept_cols == cand_cols:
                kept_words = _table_word_set(kept)
                union = cand_words | kept_words
                if union:
                    jaccard = len(cand_words & kept_words) / len(union)
                    if jaccard > 0.65:
                        is_dup = True
                        break

            # Stage 2: caption-header table on same page + same col count
            # (e.g. camelot captured the "Table 4: ..." caption row as header)
            if (cand_is_cap
                    and kept_cols == cand_cols
                    and abs(kept_page - cand_page) <= 1):
                is_dup = True
                break

        if not is_dup:
            unique.append(candidate)

    return unique


# ─────────────────────────────────────────────────────────────
# TABLE PAGE DETECTION
# ─────────────────────────────────────────────────────────────
def detect_table_pages(pdf_path):
    pages = set()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if re.search(r"Table\s+\d+", text):
                    pages.add(i + 1)
    except Exception:
        pass
    return sorted(list(pages))


def is_scanned_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_length = sum(
                len(p.extract_text() or "") for p in pdf.pages[:3]
            )
            return text_length < 50
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# CAMELOT EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_camelot(pdf_path, pages):
    tables = []

    try:
        stream = camelot.read_pdf(
            pdf_path, pages=pages, flavor="stream",
            row_tol=10, edge_tol=500
        )
    except Exception:
        stream = []

    try:
        lattice = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
    except Exception:
        lattice = []

    for t in list(stream) + list(lattice):
        try:
            df = t.df.map(clean_cell)

            if df.shape[0] < 2 or df.shape[1] < 2:
                continue

            quality = score_df(df)
            if quality < 0.05:
                continue

            headers = df.iloc[0].tolist()
            rows = df.iloc[1:].values.tolist()

            if not is_valid_table(headers, rows):
                continue

            if is_reference_table(headers, rows):
                continue

            tables.append({
                "headers": headers,
                "rows": rows,
                "page": getattr(t, "page", None),
                "quality": quality
            })
        except Exception:
            continue

    return tables


# ─────────────────────────────────────────────────────────────
# PDFPLUMBER FALLBACK
# ─────────────────────────────────────────────────────────────
def extract_pdfplumber(pdf_path):
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                for table in (page.extract_tables() or []):
                    if not table or len(table) < 2:
                        continue
                    cleaned = [[clean_cell(cell) for cell in row] for row in table]
                    if len(cleaned[0]) < 2:
                        continue
                    headers, rows = cleaned[0], cleaned[1:]

                    if not is_valid_table(headers, rows):
                        continue

                    if is_reference_table(headers, rows):
                        continue

                    tables.append({
                        "headers": headers,
                        "rows": rows,
                        "page": page_idx,
                        "quality": 0.5
                    })
    except Exception:
        pass
    return tables


# ─────────────────────────────────────────────────────────────
# OCR FALLBACK  (scanned PDFs only)
# ─────────────────────────────────────────────────────────────
def extract_ocr_tables(pdf_path):
    tables = []
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page_idx, page in enumerate(doc):
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            text = pytesseract.image_to_string(thresh)
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if len(lines) < 3:
                continue
            rows = normalize_rows([split_row_smart(l) for l in lines])
            if len(rows[0]) < 2:
                continue
            headers, data = rows[0], rows[1:]
            if not is_valid_table(headers, data):
                continue
            if is_reference_table(headers, data):
                continue
            tables.append({
                "headers": headers,
                "rows": data,
                "page": page_idx,
                "quality": 0.3
            })
    except Exception:
        pass
    return tables


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def extract_tables(pdf_path, blocks=None, pages="all"):
    """
    Multi-strategy table extraction pipeline.

    Order:
      1. Camelot  — bordered tables (lattice/stream)
      2. pdfplumber  — semi-bordered tables
      3. Word-alignment detector  — borderless text tables (NEW)
      4. Text reconstruction  — last resort from raw blocks
      5. OCR  — scanned PDFs only

    After all strategies, tables are:
      • Caption-header repaired   (fix_caption_header_tables)
      • Reference/ack filtered    (is_reference_table)
      • Deduplicated              (dedup_tables — Jaccard + page/col check)
    """
    # Avoid circular import (table_detector imports from table_extractor)
    from parser.table_detector import detect_borderless_tables

    table_pages = detect_table_pages(pdf_path)
    pages_str = ",".join(map(str, table_pages)) if table_pages else pages
    print("Table pages:", pages_str)

    # 1. Camelot (best quality for bordered tables)
    camelot_tables = extract_camelot(pdf_path, pages_str)

    # 2. pdfplumber — always run and merge
    plumber_tables = extract_pdfplumber(pdf_path)

    # 3. Word-alignment borderless detection (Tables 1, 2, 3 in many papers)
    print("Running borderless table detector...")
    borderless_tables = detect_borderless_tables(pdf_path)

    # Caption-anchored text reconstructor — most reliable for text PDFs
    from parser.text_table_reconstructor import reconstruct_text_tables
    text_tables = reconstruct_text_tables(pdf_path)

    # Only use borderless on pages not already covered by text_reconstructor
    covered_pages = {t.get('page') for t in text_tables}
    borderless_filtered = [
        t for t in borderless_tables
        if t.get('page') not in covered_pages and t.get('quality', 0) >= 0.78
        and len([h for h in t['headers'] if h.strip()]) >= 2
        and not any(len(h) > 25 for h in t['headers'] if h.strip())
    ]

    tables = camelot_tables + plumber_tables + text_tables + borderless_filtered

    # 4. Text reconstruction fallback
    if not tables and blocks:
        print("Fallback -> text reconstruction")
        tables = detect_text_tables(blocks)

    # 5. OCR (scanned PDFs only)
    if not tables and is_scanned_pdf(pdf_path):
        print("Fallback -> OCR")
        tables = extract_ocr_tables(pdf_path)
    else:
        print("Skipping OCR")

    # Repair caption-header tables
    tables = fix_caption_header_tables(tables)

    # Deduplicate
    tables = dedup_tables(tables)

    print(f"Final table count after dedup: {len(tables)}")
    return tables
