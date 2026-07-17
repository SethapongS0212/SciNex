"""
table_detector.py
─────────────────
Borderless-table detector using word-level column-alignment analysis.

Motivation
──────────
Camelot (lattice) requires visible grid lines. Camelot stream and
pdfplumber both struggle when table columns are separated only by
whitespace, which is how most academic-paper tables are typeset
(Tables 1, 2, 3 in "Attention is All You Need").

Strategy
────────
1. For each page, extract individual word bounding boxes via pdfplumber
   (word_objects = page.extract_words()).
2. Cluster word x-coordinates with a 1-D gap-based algorithm to find
   column bands.
3. Assign every word to a column band and row band (y-coordinate).
4. A block of rows where ≥ 3 columns are consistently filled is a table.
5. Reconstruct headers and rows.
6. Merge with camelot/pdfplumber results (deduplicated upstream).
"""

import re
import numpy as np
import pdfplumber
from parser.table_extractor import (
    clean_cell, is_valid_table, is_reference_table,
    normalize_rows
)


# ─────────────────────────────────────────────────────────────
# 1-D COLUMN BAND DETECTION
# ─────────────────────────────────────────────────────────────
def find_column_bands(x_positions, gap_threshold=12, page_width=612):
    """
    Group x-coordinates into column bands using 1-D gap analysis.

    Returns list of (x_min, x_max) band intervals.
    """
    if not x_positions:
        return []

    xs = sorted(set(round(x, 0) for x in x_positions))
    bands = []
    band_start = xs[0]
    band_end   = xs[0]

    for x in xs[1:]:
        if x - band_end > gap_threshold:
            bands.append((band_start, band_end))
            band_start = x
            band_end   = x
        else:
            band_end = x

    bands.append((band_start, band_end))
    return bands


def assign_band(value, bands):
    """Return the index of the band this value falls into, or -1."""
    for i, (lo, hi) in enumerate(bands):
        if lo - 4 <= value <= hi + 4:
            return i
    return -1


# ─────────────────────────────────────────────────────────────
# ROW BAND DETECTION
# ─────────────────────────────────────────────────────────────
def find_row_bands(y_positions, line_gap=4):
    """
    Cluster y-coordinates into row bands.
    """
    if not y_positions:
        return []

    ys = sorted(set(round(y, 0) for y in y_positions))
    bands = []
    band_start = ys[0]
    band_end   = ys[0]

    for y in ys[1:]:
        if y - band_end > line_gap:
            bands.append((band_start, band_end))
            band_start = y
            band_end   = y
        else:
            band_end = y

    bands.append((band_start, band_end))
    return bands


# ─────────────────────────────────────────────────────────────
# WORD GRID BUILDER
# ─────────────────────────────────────────────────────────────
def build_word_grid(words, col_bands, row_bands):
    """
    Place each word into a (row_idx, col_idx) cell.
    Returns a 2-D list of strings: grid[row][col].
    """
    n_rows = len(row_bands)
    n_cols = len(col_bands)
    grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for w in words:
        x = float(w.get("x0", 0))
        y = float(w.get("top", 0))
        text = w.get("text", "").strip()
        if not text:
            continue
        r = assign_band(y, row_bands)
        c = assign_band(x, col_bands)
        if r >= 0 and c >= 0:
            grid[r][c].append(text)

    # Collapse each cell's word list into a string
    return [
        [" ".join(cell_words) for cell_words in row]
        for row in grid
    ]


# ─────────────────────────────────────────────────────────────
# TABLE REGION SLICER
# ─────────────────────────────────────────────────────────────
def _is_tabular_row(row, max_avg_words=7):
    """
    A row looks tabular if cells are short (not full sentences)
    and the row has at least 2 filled cells.
    """
    filled = [c for c in row if c.strip()]
    if len(filled) < 2:
        return False
    avg_words = sum(len(c.split()) for c in filled) / len(filled)
    return avg_words <= max_avg_words


def _has_numeric(grid):
    """True if any cell in the grid contains a number."""
    for row in grid:
        for cell in row:
            if re.search(r'\d+\.?\d*', cell):
                return True
    return False


def find_table_regions(grid, min_cols=2, min_fill_ratio=0.40):
    """
    Identify contiguous runs of rows where enough columns are filled
    to qualify as a table.

    Extra guards vs. the naive version:
      • Rows must look "tabular" (short cells, ≥2 filled cols).
      • Final region must have ≥ 3 rows AND at least one numeric cell.
      • Max average words per cell ≤ 7 (keeps out paragraph text).

    Returns list of (start_row, end_row) slices into `grid`.
    """
    n_cols = len(grid[0]) if grid else 0
    regions = []
    in_table = False
    start = 0

    for i, row in enumerate(grid):
        filled = sum(1 for cell in row if cell.strip())
        fill_ratio = filled / max(n_cols, 1)
        tabular    = _is_tabular_row(row)

        if fill_ratio >= min_fill_ratio and filled >= min_cols and tabular:
            if not in_table:
                in_table = True
                start = i
        else:
            if in_table:
                candidate = grid[start:i]
                if (i - start >= 3                # need ≥ 3 rows
                        and _has_numeric(candidate)):  # need numbers
                    regions.append((start, i))
                in_table = False

    if in_table:
        candidate = grid[start:]
        if (len(grid) - start >= 3 and _has_numeric(candidate)):
            regions.append((start, len(grid)))

    return regions


# ─────────────────────────────────────────────────────────────
# CAPTION LOOKUP
# ─────────────────────────────────────────────────────────────
def find_nearby_caption(page_text, region_top_y, page_height):
    """
    Search page text for a "Table N:" caption near the top of
    the detected region.  Returns caption string or "".
    """
    for line in page_text.split("\n"):
        m = re.match(r'\s*(Table\s+\d+[:\.])', line, re.IGNORECASE)
        if m:
            return line.strip()
    return ""


# ─────────────────────────────────────────────────────────────
# PER-PAGE TABLE EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_page_tables(page, page_num):
    """
    Extract tables from one pdfplumber page using word-level
    column-alignment analysis.
    """
    try:
        words = page.extract_words(
            x_tolerance=3, y_tolerance=3,
            keep_blank_chars=False
        )
    except Exception:
        return []

    if not words:
        return []

    # Filter header/footer area (y < 50 or y > page.height - 50)
    h = page.height
    words = [w for w in words if 50 < float(w.get("top", 0)) < h - 50]

    if len(words) < 6:
        return []

    # Build bands
    col_bands = find_column_bands([float(w["x0"]) for w in words])
    row_bands  = find_row_bands([float(w["top"]) for w in words])

    if len(col_bands) < 2 or len(row_bands) < 2:
        return []

    grid = build_word_grid(words, col_bands, row_bands)

    if not grid:
        return []

    regions = find_table_regions(
        grid, min_cols=2,
        # For 2-col papers the table may only fill 2 of many columns
        min_fill_ratio=max(0.25, 2 / len(col_bands))
    )

    page_text = page.extract_text() or ""
    tables = []

    for start_r, end_r in regions:
        region_rows = [
            [clean_cell(cell) for cell in row]
            for row in grid[start_r:end_r]
        ]

        if not region_rows:
            continue

        # Remove fully empty rows
        region_rows = [r for r in region_rows if any(c for c in r)]

        if len(region_rows) < 2:
            continue

        # Trim trailing empty columns
        max_filled = max(
            (max((i for i, c in enumerate(r) if c), default=-1)
             for r in region_rows),
            default=-1
        ) + 1
        if max_filled < 2:
            continue
        region_rows = [r[:max_filled] for r in region_rows]

        headers = region_rows[0]
        rows    = region_rows[1:]

        if not is_valid_table(headers, rows):
            continue
        if is_reference_table(headers, rows):
            continue

        # Compute fill quality
        total_cells = sum(len(r) for r in region_rows)
        filled_cells = sum(1 for r in region_rows for c in r if c)
        quality = filled_cells / max(total_cells, 1)

        if quality < 0.25:
            continue

        # Reject if header cells are PDF-extraction artifacts
        # (concatenated words with no spaces, e.g. "AshishVaswani∗")
        concat_headers = sum(
            1 for h in headers
            if h.strip() and len(h) > 10 and " " not in h
        )
        if concat_headers >= len([h for h in headers if h.strip()]) * 0.5:
            continue

        # Reject section-number rows (e.g. '5.1', 'TrainingDataandBatching')
        if headers and re.fullmatch(r'\d+\.\d+', headers[0].strip()):
            continue

        # Reject if header cells look like paragraph sentences
        header_avg_words = (
            sum(len(h.split()) for h in headers if h.strip())
            / max(sum(1 for h in headers if h.strip()), 1)
        )
        if header_avg_words > 5:
            continue

        # Reject if all rows look like running prose (avg cell ≥ 8 words)
        all_cells = [c for r in rows for c in r if c.strip()]
        if all_cells:
            global_avg = sum(len(c.split()) for c in all_cells) / len(all_cells)
            if global_avg > 7:
                continue

        # Need at least one numeric cell somewhere in the table
        numeric_cells = sum(1 for c in all_cells if re.search(r'\d', c))
        if numeric_cells == 0:
            continue

        caption = find_nearby_caption(page_text, start_r, h)

        tables.append({
            "headers": headers,
            "rows":    rows,
            "page":    page_num,
            "quality": quality,
            "caption": caption,
            "source":  "word_align"
        })

    return tables


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def detect_borderless_tables(pdf_path):
    """
    Run word-alignment table detection on every page.
    Returns list of table dicts ready for merge with
    camelot/pdfplumber results.
    """
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = extract_page_tables(page, page_num + 1)
                tables.extend(page_tables)
    except Exception as e:
        print(f"[table_detector] Error: {e}")

    print(f"[table_detector] Found {len(tables)} borderless table(s)")
    return tables
