import re
import wordfreq


# ─────────────────────────────────────────────────────────────
# WORD VALIDATION
# ─────────────────────────────────────────────────────────────
def is_valid_word(word):
    return wordfreq.zipf_frequency(word.lower(), "en") > 3


def paragraph_quality(text):
    score = 1.0

    if re.search(r'\w-\s+\w', text):       # broken hyphenation
        score -= 0.3

    words = text.split()
    upper = sum(1 for w in words if w.isupper())
    if upper / max(len(words), 1) > 0.3:   # too many ALL-CAPS words → bad OCR
        score -= 0.3

    if "  " in text:                        # weird spacing
        score -= 0.2

    if len(words) > 80:                     # very long → likely merged
        score -= 0.2

    return max(score, 0.0)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
STOP_ENTITIES = {"figure", "table", "section"}


# ─────────────────────────────────────────────────────────────
# BASIC HELPERS
# ─────────────────────────────────────────────────────────────
def normalize_text(text):
    text = text.replace("\u00ad", "")   # soft hyphen
    text = text.replace("\n", " ")
    return text.strip()


def is_references(text):
    return text.lower().startswith("references")


def is_figure_caption(text):
    return text.strip().lower().startswith(("figure", "fig.", "table"))


def is_table_caption(text):
    """True only for 'Table N:' captions, not 'figure'."""
    return bool(re.match(r'table\s*\d+', text.strip(), re.IGNORECASE))


# ─────────────────────────────────────────────────────────────
# GARBAGE FILTER
# ─────────────────────────────────────────────────────────────
def is_garbage(text):
    text = text.strip()
    words = text.split()

    if len(words) < 3:
        if is_heading(text) or is_references(text):
            return False
        return True
    if is_heading(text) or is_references(text):
        return False

    # Standalone diagram/table labels such as "AE AS DS DD" should not
    # be promoted into prose when nearby blocks are merged later.
    if all(re.fullmatch(r"[A-Z]{1,3}", w.strip(".,;:()[]{}")) for w in words):
        return True

    if len(words) <= 7 and not re.search(r"[.!?:]$", text):
        return True

    if text.startswith("Input ") and " AE " in text and " AS " in text:
        return True


    if len(re.findall(r'\d', text)) > len(text) * 0.4:
        return True

    if len(words) > 30:
        valid = sum(is_valid_word(w) for w in words)
        if valid / len(words) < 0.4:
            return True

    return False


# ─────────────────────────────────────────────────────────────
# HEADING DETECTION
# ─────────────────────────────────────────────────────────────

# Roman numerals up to ~20 sections (covers virtually all papers).
# No IGNORECASE — Roman numeral headings in papers are always uppercase,
# and the flag would cause "I think..." to be a false positive.
_ROMAN_RE = re.compile(
    r"^(X{0,2}(?:IX|IV|V?I{0,3}))\.?\s+[A-Z]"
)

# Appendix letter prefix: "A", "B.1", "A.2.3" followed by a capital word
_APPENDIX_RE = re.compile(
    r"^([A-Z](\.[0-9]+)*)\s+[A-Z][a-z]",
)


# Math operator characters that disqualify ALL-CAPS text from being a heading
_MATH_OP_RE = re.compile(r'[=\+\*×∗→←↔≤≥≠±∑∏∫\|{}\\%@]|\d')

def is_heading(text):
    text = text.strip()
    if len(text) < 5 or len(text) > 120:
        return False

    # Numeric heading: "1 Introduction", "2.1 Attention Model"
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", text):
        return True

    # Dotted numeric heading: "1. Introduction", "3. Interpretation"
    # (number followed by a trailing period). Guard against numbered *list
    # items* in body text ("1. All morphemes are created equal.") — require a
    # short, title-like remainder that does not read as a full sentence.
    m = re.match(r"^\d+(\.\d+)*\.\s+([A-Z].*)$", text)
    if m:
        remainder = m.group(2).strip()
        if len(remainder.split()) <= 8 and not remainder.endswith(
            (".", "?", "!", ":", ";", ",")
        ):
            return True

    # Roman numeral heading: "I Introduction", "IV Experiments"
    # Guard: first token must be a plausible Roman numeral (all [IVX], ≤4 chars)
    if _ROMAN_RE.match(text):
        words = text.split()
        if len(words) >= 2 and re.match(r"^[IVXivx]{1,4}\.?$", words[0]):
            return True

    # Appendix / lettered heading: "A Details", "B.1 Hyperparameters"
    if _APPENDIX_RE.match(text):
        words = text.split()
        prefix = words[0].rstrip(".")
        # Single letter prefix ("A Appendix") or dotted ("B.1 Sub")
        if len(prefix) == 1 or re.match(r"^[A-Z](\.[0-9]+)+$", prefix):
            if len(words) <= 12:
                return True

    # ALL-CAPS heading: "INTRODUCTION", "RELATED WORK"
    # Guard: reject math formulas — "A ∗X = B".isupper() is True in Python
    # because .isupper() ignores non-alphabetic chars. Check for math operators.
    if text.isupper() and not _MATH_OP_RE.search(text):
        words = text.split()
        # Reject compact acronym/label rows such as "AE AS DS DD".
        if words and all(len(w) <= 3 for w in words):
            return False
        # All words must be alphabetic (no stray symbols slipping through)
        if 1 < len(words) < 10 and all(w.isalpha() for w in words):
            return True

    return False


def clean_heading(text):
    text = normalize_text(text)

    # Numeric: "2.1 Model" → "2.1 Model"
    match = re.match(r"^(\d+(\.\d+)*)\s+(.*)", text)
    if match:
        return f"{match.group(1)} {match.group(3)}"

    # Roman numeral: "IV. Experiments" → "IV Experiments"
    match = re.match(r"^([IVXivx]{1,4})\.?\s+(.*)", text)
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"

    # Appendix letter: "A.1 Details" stays as-is
    return text


# ─────────────────────────────────────────────────────────────
# PARAGRAPH HELPERS
# ─────────────────────────────────────────────────────────────
def clean_paragraph(text):
    text = normalize_text(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    return text.strip()


def is_sentence_end(text):
    return text.endswith((".", "?", "!", ":"))


def is_bullet(line):
    return line.strip().startswith(("•", "-", "*"))


def merge_lines_into_paragraphs(lines):
    paragraphs = []
    current = ""

    for line in lines:
        line = line.strip()
        if not line:
            if current:
                paragraphs.append(clean_paragraph(current))
                current = ""
            continue

        if is_bullet(line):
            if current:
                paragraphs.append(clean_paragraph(current))
                current = ""
            paragraphs.append(line)
            continue

        # 🔥 FIX: MUCH STRONGER MERGING
        if not current:
            current = line
            continue

        # merge if previous does NOT strongly indicate paragraph end
        if not re.search(r'[.!?]"\s*$', current) and not current.endswith("\n\n"):
            current += " " + line
        else:
            # only split if next line is clearly a new paragraph
            if len(line) > 80:  # long enough to be real paragraph
                paragraphs.append(clean_paragraph(current))
                current = line
            else:
                current += " " + line

    if current:
        paragraphs.append(clean_paragraph(current))

    return paragraphs


# ─────────────────────────────────────────────────────────────
# ENTITY DETECTION
# ─────────────────────────────────────────────────────────────
def detect_entities(text, known_entities=None):
    text_lower = text.lower()
    entities = set()

    for ent in (known_entities or []):
        if ent in text_lower:
            entities.add(ent)

    caps = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
    for c in caps:
        c = c.lower()
        if c not in STOP_ENTITIES:
            entities.add(c)

    return list(entities)


# ─────────────────────────────────────────────────────────────
# METRIC EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_metrics(text):
    metrics = []
    patterns = [
        (r'BLEU[^0-9]{0,10}(\d+(\.\d+)?)',      "BLEU"),
        (r'accuracy[^0-9]{0,10}(\d+(\.\d+)?)',   "accuracy"),
        (r'F1[^0-9]{0,10}(\d+(\.\d+)?)',         "F1"),
        (r'(\d+(\.\d+)?)\s*%',                   "percentage"),
    ]
    for pattern, name in patterns:
        for m in re.findall(pattern, text, re.IGNORECASE):
            metrics.append({
                "metric": name,
                "value": float(m[0]),
                "context": text[:200]
            })
    return metrics


# ─────────────────────────────────────────────────────────────
# TITLE DETECTION
# ─────────────────────────────────────────────────────────────
def extract_title(blocks):
    candidates = []
    for b in blocks[:20]:
        text = normalize_text(b.get("text", ""))
        if not text or len(text) < 5 or len(text) > 150:
            continue
        score = 0
        if len(text.split()) < 12:
            score += 2
        if re.match(r'^\d+\s+', text):
            score -= 2
        if len(text.split()) > 25:
            score -= 2
        candidates.append((score, text))
    if not candidates:
        return ""
    return max(candidates, key=lambda x: x[0])[1]


# ─────────────────────────────────────────────────────────────
# TABLE TEXT FINGERPRINTING  (NEW)
# Used to detect when a paragraph is actually raw table cell data.
# ─────────────────────────────────────────────────────────────
def build_table_cell_set(tables):
    """
    Returns compact, table-like cell text fragments from extracted tables.

    PDF table detectors sometimes capture normal body prose as table cells.
    Long sentence fragments and generic one-word cells are excluded so real
    paragraphs are not suppressed as duplicate table data.
    """
    def _keep_cell(value):
        s = normalize_text(str(value)).lower().strip()
        if len(s) <= 3 or len(s) > 80:
            return None
        words = s.split()
        if len(words) > 6:
            return None
        if len(words) == 1 and not re.search(r'\d|[%=+\-/]', s) and len(s) < 8:
            return None
        return s

    cells = set()
    for t in tables:
        for h in t.get("headers", []):
            s = _keep_cell(h)
            if s:
                cells.add(s)
        for row in t.get("rows", []):
            for cell in row:
                s = _keep_cell(cell)
                if s:
                    cells.add(s)
    return cells


def is_table_data_paragraph(text, table_cell_set, threshold=4):
    """
    Returns True if this paragraph is likely raw table data extracted
    from a table that was successfully captured.

    Strategy: count how many distinct table cell strings appear in the
    paragraph text. If >= threshold hits, it is table data.
    """
    if not table_cell_set or len(text) < 20:
        return False
    text_lower = normalize_text(text).lower()
    matches = sum(1 for cell in table_cell_set if cell in text_lower)
    if matches < threshold:
        return False
    # Genuine table-row dumps are short and token-dense. Real body prose can
    # incidentally contain >= threshold cell words too (e.g. when a borderless
    # detector over-segments prose into spurious tables), but it is long with a
    # SPARSE match density. Only suppress long paragraphs when the cell matches
    # are dense (<= ~60 chars per match); short paragraphs keep the old rule.
    if len(text) <= 200:
        return True
    return (len(text) / matches) <= 60


# ─────────────────────────────────────────────────────────────
# CAPTION-BASED TABLE POSITION ASSIGNMENT  (NEW)
# ─────────────────────────────────────────────────────────────
def assign_table_positions(blocks, table_blocks):
    """
    For each table block, find the nearest 'Table N:' caption block
    on the same (or nearby) page and set the table's (page, y) to
    appear right after that caption.

    This fixes the y=9999 bug that was pushing all tables to the end
    of the document regardless of where they belong.

    Returns table_blocks with updated (page, y, caption) fields.
    """
    # Collect all caption blocks: {table_num: (page, y, caption_text)}
    captions = {}
    for b in blocks:
        text = normalize_text(b.get("text", "")).strip()
        m = re.match(r'[Tt]able\s*(\d+)\s*[:\.]', text)
        if m:
            num = int(m.group(1))
            if num not in captions:
                captions[num] = {
                    "page": b.get("page", 0),
                    "y":    b.get("y", 0),
                    "text": text
                }

    if not captions:
        # No captions found — use a default progression so tables at
        # least appear in order at a midpoint of their page.
        for i, tb in enumerate(table_blocks):
            tb["y"] = 500 + i * 10
        return table_blocks

    # Sort captions by their appearance order
    ordered_caps = sorted(captions.items(), key=lambda kv: (kv[1]["page"], kv[1]["y"]))

    # Assign each table block to a caption in sequence.
    # If the table block already has a page number from camelot/pdfplumber,
    # prefer the caption on the same page.
    remaining_caps = list(ordered_caps)
    assigned = set()

    for tb in table_blocks:
        tb_page = tb.get("page") or 0
        # Find best caption: same page first, then any unassigned
        best = None
        for cap_num, cap_info in remaining_caps:
            if cap_num in assigned:
                continue
            if best is None or cap_info["page"] == tb_page:
                best = (cap_num, cap_info)
            if cap_info["page"] == tb_page:
                break   # exact page match → stop searching

        if best:
            cap_num, cap_info = best
            tb["page"]    = cap_info["page"]
            tb["y"]       = cap_info["y"] + 2   # appear just after caption
            tb["caption"] = cap_info["text"]
            assigned.add(cap_num)
        else:
            # Fallback: put it at a sensible position within its page
            tb["y"] = 500

    return table_blocks


def _page_column_ranks(blocks):
    page_cols = {}
    for b in blocks:
        col = b.get("col")
        if col is None:
            continue
        page_cols.setdefault(b.get("page", 0), {}).setdefault(col, []).append(b.get("x", 0))

    ranks = {}
    for page, cols in page_cols.items():
        ordered = sorted(
            cols.items(),
            key=lambda item: sum(item[1]) / max(len(item[1]), 1)
        )
        ranks[page] = {col: rank for rank, (col, _) in enumerate(ordered)}
    return ranks


def _reading_order_key(block, col_ranks):
    page = block.get("page", 0)
    col = block.get("col")
    rank = col_ranks.get(page, {}).get(col, 0)
    return (page, rank, block.get("y", 0), block.get("x", 0))


# ─────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────
def build_structure(blocks, tables, known_entities=None):
    """
    Build a structured document (title + sections + content) from
    raw layout blocks and extracted tables.

    known_entities: optional list of domain keywords (e.g. from
    citation concept dict) used to seed entity detection.

    Fixes applied vs. original:
      1. Table deduplication / reference filtering — done upstream in
         table_extractor.py; structure_builder trusts the cleaned list.
      2. Table cell fingerprinting — paragraphs whose text matches many
         extracted table cells are suppressed to eliminate the "table
         data as paragraph" duplicate.
      3. Table caption handling — 'Table N:' text is stored as a caption
         on the next table block instead of rendering as a <figure>.
      4. Table positioning — tables are sorted to appear right after
         their caption text rather than at y=9999 (end of page).
    """
    # ── Build table cell fingerprint set for duplicate-suppression ──
    table_cell_set = build_table_cell_set(tables)

    # ── Convert tables to positional blocks with caption assignment ──
    raw_table_blocks = [
        {
            "type":  "table",
            "headers": t["headers"],
            "rows":    t["rows"],
            "page":    t.get("page", 0),
            "x":       0,
            "y":       9999,   # temporary; overwritten below
        }
        for t in tables
    ]

    # FIX: assign realistic positions based on caption blocks
    raw_table_blocks = assign_table_positions(blocks, raw_table_blocks)

    col_ranks = _page_column_ranks(blocks)
    all_blocks = sorted(
        blocks + raw_table_blocks,
        key=lambda b: _reading_order_key(b, col_ranks)
    )

    title = extract_title(all_blocks)

    doc = {
        "title":    title,
        "sections": [],
        "entities": set(),
        "metrics":  []
    }

    current_section     = None
    pending_table_cap   = None   # FIX: stores "Table N:" caption text

    for block in all_blocks:

        # ── TABLE BLOCKS ──────────────────────────────────────────────
        if block.get("type") == "table":
            tbl_item = {
                "type":    "table",
                "headers": block["headers"],
                "rows":    block["rows"],
            }
            # Attach caption that was stored when we saw "Table N:"
            cap = block.get("caption") or pending_table_cap
            if cap:
                tbl_item["caption"] = cap
                pending_table_cap = None

            if current_section:
                current_section["content"].append(tbl_item)
            continue

        # ── TEXT BLOCKS ───────────────────────────────────────────────
        if "text" not in block:
            continue
        text = normalize_text(block["text"])

        if is_garbage(text) or text == title:
            continue

        if is_references(text):
            current_section = {"heading": "References", "content": []}
            doc["sections"].append(current_section)
            continue

        if is_heading(text):
            current_section = {"heading": clean_heading(text), "content": []}
            doc["sections"].append(current_section)
            continue

        if not current_section:
            continue

        # ── FIGURE / TABLE CAPTIONS ───────────────────────────────────
        if is_figure_caption(text):
            match = re.match(r'(figure|fig\.?|table)\s*(\d+)', text, re.IGNORECASE)
            item_type = match.group(1).lower() if match else "figure"

            if item_type == "table":
                # FIX: store as pending caption — do NOT render as <figure>.
                # The caption will be attached to the next table block.
                pending_table_cap = text
            else:
                current_section["content"].append({
                    "type":      "figure",
                    "text":      text,
                    "figure_id": match.group(2) if match else None,
                    "needs_llm": True
                })
            continue

        # ── SUPPRESS TABLE-DATA PARAGRAPHS ────────────────────────────
        # If this paragraph's text closely matches content from a
        # successfully extracted table, skip it to avoid duplication.
        if is_table_data_paragraph(text, table_cell_set, threshold=4):
            continue

        # ── REGULAR PARAGRAPHS ────────────────────────────────────────
        quality = paragraph_quality(text)

        current_section["content"].append({
            "type":     "paragraph",
            "text":     text,
            "quality":  quality,
            "needs_llm": quality < 0.8
        })

        entities = detect_entities(text, known_entities)
        doc["entities"].update(entities)

        for m in extract_metrics(text):
            doc["metrics"].append({**m, "entities": entities})

    # ── PARAGRAPH MERGING PASS ────────────────────────────────────────
    for section in doc["sections"]:
        raw    = section["content"]
        merged = []
        buffer = []

        for item in raw:
            if item["type"] == "paragraph":
                buffer.append(item["text"])
            else:
                if buffer:
                    merged += [
                        {
                            "type":     "paragraph",
                            "text":     p,
                            "quality":  0.5,
                            "needs_llm": True
                        }
                        for p in merge_lines_into_paragraphs(buffer)
                    ]
                    buffer = []
                merged.append(item)

        if buffer:
            merged += [
                {
                    "type":     "paragraph",
                    "text":     p,
                    "quality":  0.5,
                    "needs_llm": True
                }
                for p in merge_lines_into_paragraphs(buffer)
            ]

        section["content"] = merged

    doc["entities"] = list(doc["entities"])
    return doc