"""
parser/text_table_reconstructor.py
───────────────────────────────────
Reconstructs tables from raw text extraction when camelot, pdfplumber,
and the word-alignment detector all fail.

Strategy
────────
For each page, search the extracted text for a "Table N:" caption line.
Collect the lines immediately below it that look tabular (short tokens,
numeric content, consistent column spacing). Reconstruct as a dict
compatible with the rest of the pipeline.

This approach is reliable when:
  • The PDF is a digital text PDF (not scanned)
  • Tables have a visible "Table N:" caption above them
  • Rows are separated by newlines (not wrapped inside a single block)

It is NOT reliable for:
  • Tables split across two pages
  • Tables where every row was merged into a single block by PyMuPDF

Usage
─────
    from parser.text_table_reconstructor import reconstruct_text_tables
    extra_tables = reconstruct_text_tables(pdf_path)
    tables.extend(extra_tables)
    tables = dedup_tables(tables)
"""

import re
import pdfplumber
from parser.table_extractor import is_reference_table, normalize_rows, is_valid_table


# ─────────────────────────────────────────────────────────────
# COLUMN SPLITTING
# ─────────────────────────────────────────────────────────────
def _split_cols(line):
    """
    Split a table row line into columns.

    Priority:
      1. Two or more consecutive spaces  (most academic tables)
      2. Tabs
      3. Fall back to space-split if line looks numeric
    """
    line = line.strip()
    if not line:
        return []

    # Try ≥ 2 spaces first
    parts = re.split(r'  +', line)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]

    # Try tabs
    if '\t' in line:
        return [p.strip() for p in line.split('\t') if p.strip()]

    # Fall back: tokenise on whitespace
    # Lower threshold (0.2) handles rows like "Self-Attention O(n2·d) O(1) O(1)"
    # where only 1 of 4 tokens is strictly numeric but the row is valid table data.
    tokens = line.split()
    if tokens and (
        sum(bool(re.search(r'\d', t) or _MATH_NOTATION.search(t)) for t in tokens)
        / len(tokens) > 0.2
    ):
        return tokens

    return [line]


# ─────────────────────────────────────────────────────────────
# TABLE CANDIDATE EXTRACTION
# ─────────────────────────────────────────────────────────────
_CAPTION_RE = re.compile(
    r'^\s*Table\s*(\d+)\s*[:\.]',
    re.IGNORECASE
)

# Tight version for squished captions like "Table2:" - requires word boundary BEFORE
# so it doesn't match "inTable3" mid-sentence
_CAPTION_RE_TIGHT = re.compile(
    r'(?:^|[^a-zA-Z])Table(\d+)\s*[:\.]',
    re.IGNORECASE
)

# Lines that signal the table has ended:
#   "4 WhySelf-Attention"  (numbered section), CamelCase heading, Figure/References
# NOTE: no re.IGNORECASE — [A-Z][a-z]{3,}[A-Z] must stay case-sensitive or it
#       matches any capitalised word like "Recurrent".
_END_SIGNALS = re.compile(
    r'^\s*(?:'
    r'\d+\.?\s+[A-Z]'                                   # "5 Training", "5.1 Training"
    r'|(?:Figure|Algorithm|Appendix|References?)\s*\d'  # "Figure 3", "References"
    r')'
)

# Mathematical complexity notation: O(n), O(n²·d), O(log n), etc.
_MATH_NOTATION = re.compile(r'O\s*\([^)]+\)')


def _is_table_row(line, min_cols=2):
    """
    A line is a table row if it has ≥ min_cols split-able tokens,
    at least one number or math notation, and no tokens that look
    like pdfplumber-merged prose (avg token length ≤ 20 chars).
    """
    cols = _split_cols(line)
    if len(cols) < min_cols:
        return False
    # Reject if cells are very long (merged prose, not table values)
    avg_len = sum(len(c) for c in cols) / len(cols)
    if avg_len > 20:
        return False
    has_number = (
        any(re.search(r'\d', c) for c in cols)
        or bool(_MATH_NOTATION.search(line))
    )
    return has_number


def _extract_table_from_text(page_text, caption_line, table_num):
    """
    Given the full page text and the caption line, extract the table rows
    that follow the caption.

    Returns (caption_str, headers, rows) or None.
    """
    lines = page_text.split('\n')
    # Find where the caption line appears
    cap_idx = None
    cap_prefix = caption_line.strip()[:40]  # first 40 chars — robust to truncation
    for i, line in enumerate(lines):
        if cap_prefix in line:
            cap_idx = i
            break

    if cap_idx is None:
        return None

    # Caption may span multiple lines — scan forward until we hit table rows
    start_idx = cap_idx + 1
    for skip in range(8):
        idx = start_idx + skip
        if idx >= len(lines):
            break
        if _is_table_row(lines[idx]):
            start_idx = idx
            break

    # Collect table rows — stop on prose paragraphs or 2 consecutive non-row lines
    table_lines = []
    consecutive_non_rows = 0

    for line in lines[start_idx:]:
        line_stripped = line.strip()

        # Blank line: allow one inside the table (Table 3 has group separators)
        if not line_stripped:
            if consecutive_non_rows >= 1:
                break    # second blank or blank after a non-row = end of table
            consecutive_non_rows += 1
            table_lines.append('')
            continue

        # Explicit section / figure end signals
        if _END_SIGNALS.match(line_stripped):
            break

        # Long prose = table has ended
        words = line_stripped.split()
        if (len(words) > 12
                and sum(bool(re.search(r'\d', w) or _MATH_NOTATION.search(w))
                        for w in words) / len(words) < 0.15):
            break

        # Single-char lines ("k", "n") are math subscripts — merge into prev cell
        if len(line_stripped) <= 2 and table_lines:
            last = table_lines[-1]
            if last and last != '':
                table_lines[-1] = last + line_stripped
                continue
            # else skip
            continue

        # Check if this is a table row
        if _is_table_row(line_stripped):
            consecutive_non_rows = 0
            table_lines.append(line_stripped)
        else:
            consecutive_non_rows += 1
            # Stop after 2 consecutive non-row lines (avoids pulling in prose)
            if consecutive_non_rows >= 2:
                break
            # One non-row line is tolerated only if it looks like a header
            # (e.g. "BLEU TrainingCost(FLOPs)")
            if len(line_stripped.split()) >= 2:
                table_lines.append(line_stripped)

    # Remove trailing blanks
    while table_lines and table_lines[-1] == '':
        table_lines.pop()

    if len(table_lines) < 2:
        return None

    # Remove blank separators
    non_blank = [l for l in table_lines if l]

    # Build grid
    grid = [_split_cols(line) for line in non_blank]

    # Normalise column count
    max_cols = max(len(r) for r in grid)
    if max_cols < 2:
        return None

    # Pad / truncate to max_cols
    grid = [(r + [''] * max_cols)[:max_cols] for r in grid]

    headers = grid[0]
    rows    = grid[1:]

    if not is_valid_table(headers, rows):
        return None

    # NOTE: we intentionally do NOT call is_reference_table() here.
    # Comparison tables (e.g. Table 2 in "Attention is All You Need") embed
    # inline citations like "ByteNet[18]" in model names, which triggers the
    # citation-ratio heuristic as a false positive. Since we only reach this
    # function via an explicit "Table N:" caption, the content is legitimate.

    return caption_line.strip(), headers, rows


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def reconstruct_text_tables(pdf_path):
    """
    Scan every page for "Table N:" captions and reconstruct the
    following tabular content.

    Returns a list of table dicts compatible with table_extractor.py.
    """
    tables = []
    seen_nums = set()   # avoid duplicate table numbers

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                # Find all "Table N:" caption lines on this page (handles squished captions)
                for line in page_text.split('\n'):
                    m = _CAPTION_RE.match(line) or _CAPTION_RE_TIGHT.search(line)
                    if not m:
                        continue
                    table_num = int(m.group(1))
                    if table_num in seen_nums:
                        continue

                    result = _extract_table_from_text(page_text, line, table_num)
                    if result is None:
                        continue

                    caption_str, headers, rows = result
                    seen_nums.add(table_num)

                    # Quality estimate: ratio of non-empty cells
                    all_cells   = headers + [c for r in rows for c in r]
                    non_empty   = sum(1 for c in all_cells if c.strip())
                    quality     = non_empty / max(len(all_cells), 1)

                    if quality < 0.25:
                        continue

                    tables.append({
                        "headers": headers,
                        "rows":    rows,
                        "page":    page_idx + 1,
                        "quality": round(quality, 3),
                        "caption": caption_str,
                        "source":  "text_reconstruct",
                    })
                    print(f"[text_reconstructor] Table {table_num} recovered "
                          f"(page {page_idx+1}, {len(rows)} rows, q={quality:.2f})")

    except Exception as e:
        print(f"[text_reconstructor] Error: {e}")

    return tables
