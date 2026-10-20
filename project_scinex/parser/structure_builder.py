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

# Unnumbered section names, for venues that don't number their headings.
# ACL/IEEE papers number theirs ("1 Introduction"); Nature/Scientific Reports and
# many journals do not ("Introduction", "Results", "Methods"), which left those
# papers with ZERO detected headings — and since everything before the first
# heading is dropped, the entire body was discarded. Matching a closed vocabulary
# (rather than a general "short title-case line" rule) is what keeps this from
# firing on ordinary body prose.
_SECTION_VOCAB = {
    "abstract", "introduction", "background", "related work", "related works",
    "literature review", "materials", "materials and methods", "method", "methods",
    "methodology", "methods and materials", "experimental", "experimental setup",
    "experimental section", "experiments", "experiment", "results",
    "results and discussion", "results and discussions", "discussion", "discussions",
    "evaluation", "analysis", "implementation", "system overview", "proposed method",
    "proposed approach", "conclusion", "conclusions", "conclusion and future work",
    "conclusions and future work", "concluding remarks", "future work", "limitations",
    "data availability", "code availability", "data availability statement",
    "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
    "author contributions", "competing interests", "conflicts of interest",
    "conflict of interest", "additional information", "supplementary information",
    "supplementary material", "ethics declarations", "declarations", "funding",
    "appendix", "abbreviations", "nomenclature", "keywords", "highlights",
}

_SECTION_NUM_PREFIX = re.compile(r"^\d+(\.\d+)*\.?\s+")

# Running page headers carry a page number and then the journal line, which is
# exactly the shape of a numbered heading: "2268 Sensors and Materials, Vol. 30,
# No. 10 (2018)". Bibliographic markers are what separate the two.
_RUNNING_HEADER_RE = re.compile(
    r"\bVol\.|\bNo\.|\bpp\.|\bISSN\b|\bISBN\b|\bDOI\b|\(\d{4}\)|\b\d{4}\s*\)",
    re.IGNORECASE,
)
# Real section numbers are small; a leading number in the hundreds or thousands
# is a page number, not a section.
_MAX_SECTION_NUMBER = 40


def _is_running_header(text):
    m = re.match(r"^(\d+)(?:\.\d+)*\.?\s+", text)
    if not m:
        return False
    if int(m.group(1)) > _MAX_SECTION_NUMBER:
        return True
    return bool(_RUNNING_HEADER_RE.search(text))


def _section_vocab_key(text):
    """Normalised heading key: 'Materials and Methods:' → 'materials and methods'."""
    t = text.strip().rstrip(":.").strip()
    t = _SECTION_NUM_PREFIX.sub("", t)
    return re.sub(r"\s+", " ", t).lower()


def is_heading(text):
    text = text.strip()
    if len(text) < 5 or len(text) > 120:
        return False

    # Unnumbered vocabulary heading: "Introduction", "Results", "Data availability"
    if _section_vocab_key(text) in _SECTION_VOCAB:
        return True

    # A running page header ("2268 Sensors and Materials, Vol. 30, No. 10
    # (2018)") satisfies the numeric-heading rule on every page, fragmenting the
    # document into one junk section per page.
    if _is_running_header(text):
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


_AFFILIATION_RE = re.compile(
    r"\b(?:Department|Universit|Institute|Faculty|College|School of|Laborator|"
    r"Correspondence|E-?mail|@|ORCID|Received:|Accepted:|Published:|"
    r"©|Copyright|Licensee|Creative Commons|This article is licensed)\b",
    re.IGNORECASE,
)


def looks_like_abstract_prose(text):
    """
    Is this pre-heading block real prose (an abstract) rather than front-matter
    furniture — author lists, affiliations, correspondence, copyright, dates?

    Used to rescue the abstract, which every paper was losing: it sits before the
    first heading, and blocks before the first heading are dropped.
    """
    text = text.strip()
    words = text.split()
    if len(words) < 40:
        return False
    if _AFFILIATION_RE.search(text):
        return False
    if text.count(".") < 2:
        return False
    # Mostly letters/spaces — filters keyword rows, author lists with superscripts,
    # reference-style blocks and numeric tables.
    letters = sum(1 for c in text if c.isalpha() or c.isspace())
    if letters / max(len(text), 1) < 0.80:
        return False
    return True


def extract_abstract_section(pre_heading_texts):
    """
    Build an "Abstract" section from the blocks that preceded the first heading.

    Two layouts are handled: an explicit "Abstract" / "Abstract:" label (the label
    block may or may not also carry the text), and an unlabelled leading paragraph
    (Nature style). Anything that doesn't read as prose is left out, so author
    lists and copyright boilerplate don't leak into the KG.
    """
    items = []
    for raw in pre_heading_texts:
        text = raw.strip()
        low = text.lower()
        if low.startswith("abstract"):
            # Strip the label; keep whatever followed it in the same block.
            stripped = re.sub(r"^abstract\s*[:.\-—]?\s*", "", text, flags=re.IGNORECASE)
            if len(stripped.split()) >= 20:
                items.append(clean_paragraph(stripped))
                continue
        if looks_like_abstract_prose(text):
            items.append(clean_paragraph(text))

    if not items:
        return None
    return {
        "heading": "Abstract",
        "content": [{"type": "paragraph", "text": t} for t in items],
    }


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
# Front-matter lines that are never the paper's title, however prominent they look:
# journal mastheads, DOI/ISSN footers, keyword rows, licence blurbs, section labels.
_NON_TITLE_RE = re.compile(
    r"^(?:abstract|keywords?|highlights?|open access|received|accepted|published|"
    r"doi\b|https?://|www\.|issn|isbn|volume\s*\d|vol\.\s*\d|©|copyright|"
    r"citation:|correspondence|e-?mail|check for updates|a publication of|"
    r"guest editors?|scientific reports\b|nature\b|sensors\b|applied sciences\b|"
    # Publisher furniture that shares the title's font size and so survives the
    # largest-font filter: an Elsevier masthead is the bare journal name, and
    # ScienceDirect brands the header. `measurement` is anchored to end-of-block
    # so a real title starting "Measurement of ..." is not rejected.
    r"sciencedirect\b|measurement\s*$|procedia\b|"

    r"ieee\b|elsevier\b|springer\b|mdpi\b|chemical engineering transactions\b)",
    re.IGNORECASE,
)

# Journal templates print an article-type label in the same style as the title
# ("Article Travel Time Prediction on…"). Strip the label, keep the title.
_TITLE_LABEL_RE = re.compile(
    r"^(?:original\s+)?(?:research\s+)?(?:article|review|communication|letter|"
    r"perspective|brief\s+report|paper)\s+(?=[A-Z0-9])",
    re.IGNORECASE,
)


def _strip_title_label(text):
    return _TITLE_LABEL_RE.sub("", text.strip(), count=1).strip()


def _title_block_ok(text):
    """
    Per-BLOCK filter — deliberately lenient about length, because titles wrap into
    short continuation blocks ("in Thailand", "Properties Prediction", "Language
    Understanding") that a word-count minimum would throw away, truncating the title.
    """
    # Collapse internal runs of whitespace BEFORE the furniture test: journal
    # mastheads are letter-spaced in the PDF ("applied   sciences"), so the
    # _NON_TITLE_RE alternatives would never match the raw text.
    text = re.sub(r"\s+", " ", text.strip())
    if not text or len(text.split()) > 30:
        return False
    if _NON_TITLE_RE.match(text):
        return False
    if "@" in text or re.search(r"\bdoi\.org\b|\bhttps?://|arxiv:", text, re.IGNORECASE):
        return False
    letters = sum(1 for c in text if c.isalpha() or c.isspace())
    if letters / max(len(text), 1) < 0.75:
        return False
    # Author lists: "Name A 1,*, Name B 2 and Name C 3" — a superscript
    # affiliation marker next to a comma/asterisk, or a surname followed by a
    # standalone superscript digit. The digit must be its own token: an earlier
    # version allowed "<digit> and <Name>", which fired on "YOLOv8 and Tracking
    # Algorithms" and rejected a real title.
    if re.search(r"\d\s*,\s*\*|\*\s*,|(?:^|\s)[A-Z][a-z]+\s+\d\s*(?:,|and\s)", text):
        return False
    # Citation-block continuation lines ("M.P.; Chawuthai, R. Parking Time") are
    # set in the same style as their "Citation:" head, which _NON_TITLE_RE only
    # catches on the first line.
    if re.search(r"[A-Z]\.\s*;|[A-Z][a-z]+,\s*[A-Z]\.", text):
        return False
    return True


def _title_final_ok(text):
    words = text.split()
    return 3 <= len(words) <= 30 and _title_block_ok(text)


def extract_title(blocks):
    """
    Title = the largest-font text on page 1, minus the front-matter furniture.

    Font size is the signal that actually separates a title from the journal
    masthead; the previous length-only heuristic scored every short block equally
    and `max()` returned the first one, which is the masthead on essentially every
    journal template ("sensors", "applied sciences", "CHEMICAL ENGINEERING
    TRANSACTIONS"). Falls back to the length heuristic when no size info is
    available (e.g. a PDF the dict extractor can't read).
    """
    page1 = [b for b in blocks if (b.get("page") or 0) == 0] or blocks[:20]

    sized = []
    for b in page1[:40]:
        text = _strip_title_label(normalize_text(b.get("text", "")))
        size = b.get("size")
        if not text or len(text) < 3 or len(text) > 300:
            continue
        if size and _title_block_ok(text):
            sized.append((float(size), float(b.get("y") or 0.0),
                          float(b.get("x") or 0.0), text))

    if sized:
        max_size = max(s for s, _, _, _ in sized)
        # Keep every block within 5% of the largest size — titles wrap across
        # blocks — and join them in READING order. Block order out of
        # extract_blocks has been through column clustering by this point, which
        # can interleave the lines of a wrapped title (it put "Thailand Roads"
        # before "Spatial-Temporal Traffic Speed Prediction on").
        parts = [t for _, _, _, t in
                 sorted((c for c in sized if c[0] >= max_size * 0.95),
                        key=lambda c: (c[1], c[2]))]
        title = re.sub(r"\s+", " ", " ".join(parts).strip())
        if _title_final_ok(title):
            return title

    # ── Fallback: original length heuristic ──────────────────────────────
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
        if _title_final_ok(text):
            score += 3
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
            if tb.get("page") is None:
                tb["page"] = 0
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
            if tb.get("page") is None:
                tb["page"] = 0

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
    # None-safe: table blocks from the detectors can carry page/y/x = None (no
    # caption was matched to position them), and sorting None against an int
    # raises TypeError — which used to abort the whole parse for that paper.
    page = block.get("page") or 0
    col = block.get("col")
    rank = col_ranks.get(page, {}).get(col, 0)
    return (page, rank, block.get("y") or 0, block.get("x") or 0)


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
    pre_heading_texts   = []     # buffered until the first heading (abstract lives here)

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

        # `is_heading_hint` is set by layout.py when a block was split off the
        # front of its own paragraph because it is set in the heading face.
        # Trust that font evidence: these headings ("Related works", "Object
        # detections") are often too short or too far off the section
        # vocabulary for the text rules, and is_garbage would drop them.
        hinted_heading = bool(block.get("is_heading_hint"))

        if not hinted_heading and (is_garbage(text) or text == title):
            continue

        if is_references(text):
            current_section = {"heading": "References", "content": []}
            doc["sections"].append(current_section)
            continue

        if hinted_heading or is_heading(text):
            current_section = {"heading": clean_heading(text), "content": []}
            doc["sections"].append(current_section)
            continue

        if not current_section:
            # Before the first heading. Buffer instead of dropping — this region
            # holds the abstract (and, if heading detection fails entirely, the
            # whole paper).
            pre_heading_texts.append(text)
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

    # ── RESCUE PRE-HEADING CONTENT (abstract) ─────────────────────────
    # Everything before the first heading used to be dropped outright, which cost
    # every paper its abstract — and cost papers whose heading style wasn't
    # recognised their entire body.
    abstract_section = extract_abstract_section(pre_heading_texts)
    if abstract_section:
        for item in abstract_section["content"]:
            item["quality"] = paragraph_quality(item["text"])
            item["needs_llm"] = item["quality"] < 0.8
            entities = detect_entities(item["text"], known_entities)
            doc["entities"].update(entities)
        doc["sections"].insert(0, abstract_section)

    if not doc["sections"]:
        print("[structure] WARNING: no sections detected — the document is empty.")
    elif all(s["heading"].strip().lower().startswith("reference")
             for s in doc["sections"]):
        print("[structure] WARNING: only a References section was detected — "
              "heading detection likely failed for this paper's layout.")

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