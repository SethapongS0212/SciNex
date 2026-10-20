import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re


# -------------------------
# FILTER LAYOUT NOISE
# -------------------------
def filter_layout_noise(blocks):
    filtered = []

    for b in blocks:
        text = b["text"].strip()

        if not text:
            continue

        # page numbers
        if re.fullmatch(r'\d+', text):
            continue

        # too small noise
        if len(text) < 3:
            continue

        # header/footer zones. Page 1 is exempt at the top: running headers are a
        # page-2-onward phenomenon, whereas a journal that sets its title high on
        # page 1 (IEEE conference templates put it at y~24) would otherwise have
        # the first line of its own title deleted before title detection runs.
        top_cut = 0 if (b.get("page") or 0) == 0 else 40
        if b["y"] < top_cut or b["y"] > 780:
            continue

        filtered.append(b)

    return filtered


# -------------------------
# TEXT CLEANING
# -------------------------
def filter_noise(blocks):
    cleaned = []

    for b in blocks:
        text = b["text"].strip()

        if len(text) < 3:
            continue

        if re.fullmatch(r'\d+(\s*/\s*\d+)?', text):
            continue

        if text.lower().startswith(("page", "copyright")):
            continue

        cleaned.append(b)

    return cleaned


# -------------------------
# COLUMN DETECTION (FIXED)
# -------------------------
def detect_columns(blocks):
    """
    Improved:
    - uses x0 positions
    - allows k=1..3
    - avoids unstable clustering
    """

    xs = np.array([[b["x"]] for b in blocks])

    if len(xs) < 10:
        for b in blocks:
            b["col"] = 0
        return blocks

    best_k = 1
    best_score = -1
    best_model = None

    for k in [1, 2, 3]:
        if len(blocks) < k:
            continue

        try:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(xs)

            if k == 1:
                score = 0
            else:
                score = silhouette_score(xs, kmeans.labels_)

            if score > best_score:
                best_k = k
                best_score = score
                best_model = kmeans

        except:
            continue

    if best_model is None:
        for b in blocks:
            b["col"] = 0
        return blocks

    for i, b in enumerate(blocks):
        b["col"] = int(best_model.labels_[i])

    return blocks


# -------------------------
# MERGE BLOCKS (SAFER VERSION)
# -------------------------
def merge_blocks(blocks, y_threshold=10):
    """
    Merge only true line continuations:
    - same page & column
    - very similar x (indent)
    - small positive vertical gap
    """
    if not blocks:
        return []

    merged = []
    blocks = sorted(blocks, key=lambda b: (b["page"], b["col"], b["y"]))

    for b in blocks:
        if not merged:
            merged.append(b)
            continue

        last = merged[-1]

        same_group = (
            b["page"] == last["page"] and
            b["col"] == last["col"]
        )

        dy = b["y"] - last["y"]          # b must be below last
        vertical_close = 0 < dy < y_threshold

        same_indent = abs(b["x"] - last["x"]) < 10  # was 3

        should_merge = same_group and vertical_close and same_indent

        # A heading split off from its paragraph sits directly above that
        # paragraph at the same indent - exactly the shape merge_blocks joins.
        if b.get("is_heading_hint") or last.get("is_heading_hint"):
            should_merge = False

        if should_merge:
            last["text"] += " " + b["text"]
            last["bbox"][2] = max(last["bbox"][2], b["bbox"][2])
            last["bbox"][3] = max(last["bbox"][3], b["bbox"][3])
            if "size" in b:
                last["size"] = max(last.get("size", 0.0), b["size"])
        else:
            merged.append(b)

    return merged



# -------------------------
# COLUMN SORTING (FIXED)
# -------------------------
def sort_columns(blocks):
    columns = {}

    for b in blocks:
        columns.setdefault(b["col"], []).append(b)

    sorted_cols = sorted(
        columns.items(),
        key=lambda item: np.median([b["x"] for b in item[1]])
    )

    return sorted_cols


# -------------------------
# MAIN PIPELINE
# -------------------------
def _page_lines(page):
    """
    Per-LINE style info: [(bbox, text, size, font, styled, uniform)].

    get_text("blocks") gives text+bbox but no font information, so the title
    detector had nothing but text length to go on (and reliably picked the
    journal masthead), and a heading glued to the front of its paragraph was
    invisible. The dict extractor carries per-span size/font/flags; this pass
    annotates and splits the existing blocks rather than replacing the
    extraction pipeline.

    `styled` is True when the line's dominant span is bold or italic
    (PyMuPDF span flags: bit 4 = bold, bit 1 = italic).
    """
    lines = []
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return lines
    for blk in page_dict.get("blocks", []):
        for line in blk.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(sp.get("text", "") for sp in spans)
            inked = [sp for sp in spans if sp.get("text", "").strip()]
            if not text.strip() or not inked or not line.get("bbox"):
                continue
            # Dominant span = the one carrying the most characters, so a
            # superscript citation marker can't define the line's style.
            main = max(inked, key=lambda sp: len(sp.get("text", "")))
            flags = int(main.get("flags", 0))
            uniform = len({sp.get("font") for sp in inked}) == 1
            lines.append((
                tuple(line["bbox"]),
                text,
                max(float(sp.get("size", 0.0)) for sp in inked),
                main.get("font", ""),
                bool(flags & 16 or flags & 2),
                uniform,
            ))
    return lines


def _lines_in(block, lines):
    """The lines whose centre falls inside a block's bbox, in reading order."""
    x0, y0, x1, y1 = block["bbox"]
    inside = []
    for ln in lines:
        lx0, ly0, lx1, ly1 = ln[0]
        cx, cy = (lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            inside.append(ln)
    inside.sort(key=lambda ln: (ln[0][1], ln[0][0]))
    return inside


def _annotate_font_sizes(blocks, lines):
    """Attach the max font size of the lines falling inside each block's bbox."""
    if not lines:
        return blocks
    for b in blocks:
        contained = _lines_in(b, lines)
        if contained:
            b["size"] = max(ln[2] for ln in contained)
    return blocks


# A heading glued to the front of its paragraph (the Nature/Scientific Reports
# layout) is one PDF block whose first line is set in the heading style and
# whose remainder is body prose. These bound what may be split off as a heading.
_GLUED_LEAD_MAX_WORDS = 8
_GLUED_BODY_MIN_CHARS = 100
_GLUED_SIZE_DELTA = 0.5
_GLUED_LEAD_END_RE = re.compile(r"[.?!,;]$")


def _looks_like_paragraph(text):
    """
    A heading only ever runs into PROSE. Front matter has the same shape - the
    first author is set larger than the rest of the author list, so "Jacob
    Devlin" would otherwise be split off BERT's byline as a section heading -
    but it is a name/affiliation/e-mail run, not a paragraph.
    """
    if "@" in text:
        return False
    return bool(_SENTENCE_END_RE.search(text))


_SENTENCE_END_RE = re.compile(r"[.?!](?:\s|$)")


def _dominant_style(lines):
    """(size, font, styled) of the style carrying the most characters."""
    weights = {}
    for _bbox, text, size, font, styled, _uniform in lines:
        key = (round(size, 1), font, styled)
        weights[key] = weights.get(key, 0) + len(text.strip())
    if not weights:
        return None
    return max(weights.items(), key=lambda kv: kv[1])[0]


def _split_glued_headings(blocks, lines):
    """
    Split "Literature reviews The following review includes both related work..."
    into a heading block + a body block.

    Nature-family templates set the heading in a larger and/or bold face but do
    not break the PDF block, so no text rule can find it - `is_heading` never
    fires and every triple extracted from those papers claims to come from the
    section the previous heading named (in practice, "Abstract"). The split-off
    block is marked `is_heading_hint` so `build_structure` trusts the font
    evidence instead of re-deriving the heading from the words.
    """
    if not lines:
        return blocks

    out = []
    for b in blocks:
        contained = _lines_in(b, lines)
        if len(contained) < 2:
            out.append(b)
            continue

        lead, body = contained[0], contained[1:]
        lead_text = lead[1].strip()
        body_text = " ".join(ln[1] for ln in body).strip()

        if (
            not lead[5]                                   # lead line style is mixed
            or len(lead_text.split()) > _GLUED_LEAD_MAX_WORDS
            or len(lead_text) < 3
            or _GLUED_LEAD_END_RE.search(lead_text)       # reads as a sentence
            or len(body_text) < _GLUED_BODY_MIN_CHARS
            or not _looks_like_paragraph(body_text)
        ):
            out.append(b)
            continue

        dom = _dominant_style(body)
        if dom is None:
            out.append(b)
            continue
        body_size, body_font, body_styled = dom

        bigger = lead[2] >= body_size + _GLUED_SIZE_DELTA
        # Same-size sub-headings exist too ("Object detections", in italic); a
        # face change alone is only trusted when the lead is bold/italic and the
        # body is not, which an emphasised opening word cannot satisfy (it would
        # not occupy the whole line uniformly).
        restyled = lead[3] != body_font and lead[4] and not body_styled

        if not (bigger or restyled):
            out.append(b)
            continue

        lx0, ly0, lx1, ly1 = lead[0]
        head_block = dict(b)
        head_block.update({
            "text": lead_text,
            "bbox": [lx0, ly0, lx1, ly1],
            "x": lx0,
            "y": ly0,
            "size": lead[2],
            "is_heading_hint": True,
        })

        bx0 = min(ln[0][0] for ln in body)
        by0 = min(ln[0][1] for ln in body)
        bx1 = max(ln[0][2] for ln in body)
        by1 = max(ln[0][3] for ln in body)
        body_block = dict(b)
        body_block.update({
            "text": body_text,
            "bbox": [bx0, by0, bx1, by1],
            "x": bx0,
            "y": by0,
            "size": max(ln[2] for ln in body),
        })
        body_block.pop("is_heading_hint", None)

        out.append(head_block)
        out.append(body_block)

    return out


def extract_blocks(doc):
    all_blocks = []

    for page_num, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")

        blocks = []
        for b in raw_blocks:
            x0, y0, x1, y1, text = b[:5]

            if not text or not text.strip():
                continue

            blocks.append({
                "page": page_num,
                "text": text.strip(),
                "bbox": [x0, y0, x1, y1],
                "x": x0,
                "y": y0
            })

        # Font/style info is needed on EVERY page now: page 1 for title
        # detection, all pages for splitting headings glued to their paragraph.
        page_lines = _page_lines(page)
        blocks = _annotate_font_sizes(blocks, page_lines)
        blocks = _split_glued_headings(blocks, page_lines)

        # Step 1: noise filtering
        blocks = filter_layout_noise(blocks)

        # Step 2: column detection
        blocks = detect_columns(blocks)

        # Step 3: sort columns properly
        sorted_cols = sort_columns(blocks)

        # Step 4: reading order
        page_blocks = []
        for _, col_blocks in sorted_cols:
            col_sorted = sorted(col_blocks, key=lambda b: b["y"])
            page_blocks.extend(col_sorted)

        # Step 5: merge lines safely
        page_blocks = merge_blocks(page_blocks)

        # Step 6: final cleanup
        page_blocks = filter_noise(page_blocks)

        all_blocks.extend(page_blocks)

    return all_blocks
