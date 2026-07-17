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

        # header/footer zones
        if b["y"] < 40 or b["y"] > 780:
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

        if should_merge:
            last["text"] += " " + b["text"]
            last["bbox"][2] = max(last["bbox"][2], b["bbox"][2])
            last["bbox"][3] = max(last["bbox"][3], b["bbox"][3])
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
