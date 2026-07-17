"""
parser/vision_table_extractor.py
─────────────────────────────────
Vision-based table detector for borderless tables that camelot and
pdfplumber cannot handle (e.g. Tables 1-3 in "Attention Is All You Need").

Approach
────────
Uses Microsoft's Table Transformer (TATR), a DETR-based object-detection
model fine-tuned on PubTables-1M and FinTabNet, to:
  1. Rasterise each PDF page to an image  (PyMuPDF)
  2. Detect table regions                (table-transformer-detection)
  3. Detect table structure              (table-transformer-structure-recognition)
  4. OCR each cell                       (pytesseract)
  5. Return structured table dicts       compatible with table_extractor.py

Requirements
────────────
  pip install transformers timm pillow pytesseract pymupdf

The model is downloaded automatically on first run (~340 MB).

Integration with main pipeline
───────────────────────────────
In extract_tables() in table_extractor.py, add as a final fallback
BEFORE the text-reconstruction step:

    from parser.vision_table_extractor import VisionTableExtractor
    vte = VisionTableExtractor()
    tables += vte.extract(pdf_path)
    tables = dedup_tables(tables)

Design notes
─────────────
• All heavy imports are inside __init__ so the module can be imported
  even when transformers is not installed — it will just log a warning.
• The extractor is stateful (model loaded once) so construct it once
  and call .extract() for each new PDF.
"""

import re
import os
import io

try:
    import fitz          # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import torch
    from transformers import (
        AutoImageProcessor,
        TableTransformerForObjectDetection,
    )
    HAS_TATR = True
except ImportError:
    HAS_TATR = False


# ── label maps ─────────────────────────────────────────────────────────────
DETECTION_MODEL   = "microsoft/table-transformer-detection"
STRUCTURE_MODEL   = "microsoft/table-transformer-structure-recognition"

# TATR structure labels
_STRUC_LABELS = {
    "table row":            "row",
    "table column":         "col",
    "table column header":  "header",
    "table":                "table",
    "table spanning cell":  "span",
    "no object":            None,
}


def _iou(box_a, box_b):
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    aa = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    ab = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / max(aa+ab-inter, 1)


def _crop(pil_img, box):
    """Return a PIL crop from (x1,y1,x2,y2) box with small padding."""
    pad = 4
    x1, y1, x2, y2 = box
    w, h = pil_img.size
    return pil_img.crop((
        max(0, x1-pad), max(0, y1-pad),
        min(w, x2+pad), min(h, y2+pad)
    ))


def _ocr_cell(pil_img, config="--psm 6"):
    """OCR a cell image and return stripped text."""
    if not HAS_TESSERACT:
        return ""
    try:
        return pytesseract.image_to_string(pil_img, config=config).strip()
    except Exception:
        return ""


class VisionTableExtractor:
    """
    Detect and extract borderless tables from a PDF using TATR + Tesseract.

    Parameters
    ──────────
    det_threshold   : confidence threshold for table detection (0-1)
    struc_threshold : confidence threshold for structure recognition (0-1)
    dpi             : rasterisation DPI (higher = better OCR, slower)
    max_pages       : maximum pages to scan (None = all)
    """

    def __init__(
        self,
        det_threshold=0.7,
        struc_threshold=0.5,
        dpi=150,
        max_pages=None,
    ):
        self.det_threshold   = det_threshold
        self.struc_threshold = struc_threshold
        self.dpi             = dpi
        self.max_pages       = max_pages

        self._det_proc   = None
        self._det_model  = None
        self._str_proc   = None
        self._str_model  = None
        self._loaded     = False

        self._try_load()

    # ── model loading ─────────────────────────────────────────────────────

    def _try_load(self):
        if not HAS_TATR:
            print("[VisionTableExtractor] transformers not installed — skipping TATR")
            return
        try:
            print("[VisionTableExtractor] Loading detection model …")
            self._det_proc  = AutoImageProcessor.from_pretrained(DETECTION_MODEL)
            self._det_model = TableTransformerForObjectDetection.from_pretrained(
                DETECTION_MODEL
            )
            self._det_model.eval()

            print("[VisionTableExtractor] Loading structure model …")
            self._str_proc  = AutoImageProcessor.from_pretrained(STRUCTURE_MODEL)
            self._str_model = TableTransformerForObjectDetection.from_pretrained(
                STRUCTURE_MODEL
            )
            self._str_model.eval()
            self._loaded = True
            print("[VisionTableExtractor] Models ready ✓")
        except Exception as e:
            print(f"[VisionTableExtractor] Model load failed: {e}")

    # ── page rasterisation ────────────────────────────────────────────────

    def _rasterise(self, pdf_path):
        """
        Yield (page_index, PIL.Image) for each page in the PDF.
        Uses PyMuPDF for speed and quality.
        """
        if not HAS_FITZ:
            raise RuntimeError("PyMuPDF (fitz) is required for page rasterisation")

        doc   = fitz.open(pdf_path)
        scale = self.dpi / 72.0
        mat   = fitz.Matrix(scale, scale)

        for i, page in enumerate(doc):
            if self.max_pages and i >= self.max_pages:
                break
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i, img

        doc.close()

    # ── table detection ───────────────────────────────────────────────────

    def _detect_tables(self, pil_img):
        """
        Run TATR detection on a page image.
        Returns list of [x1, y1, x2, y2] bounding boxes (pixel coords).
        """
        inputs = self._det_proc(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._det_model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]])  # (H, W)
        results = self._det_proc.post_process_object_detection(
            outputs, threshold=self.det_threshold, target_sizes=target_sizes
        )[0]

        boxes = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_str = self._det_model.config.id2label.get(label.item(), "")
            if "table" in label_str.lower():
                boxes.append([round(v) for v in box.tolist()])
        return boxes

    # ── structure recognition ─────────────────────────────────────────────

    def _recognise_structure(self, table_img):
        """
        Run TATR structure recognition on a cropped table image.
        Returns (rows, col_headers) where:
          rows        : sorted list of [y1, y2] row bands
          col_headers : list of [x1, x2] column bands (header row only)
          all_cols    : sorted list of [x1, x2] column bands
        """
        inputs = self._str_proc(images=table_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._str_model(**inputs)

        target_sizes = torch.tensor([table_img.size[::-1]])
        results = self._str_proc.post_process_object_detection(
            outputs, threshold=self.struc_threshold, target_sizes=target_sizes
        )[0]

        rows, cols, header_boxes = [], [], []

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_str = self._str_model.config.id2label.get(label.item(), "")
            role = _STRUC_LABELS.get(label_str)
            b    = [round(v) for v in box.tolist()]

            if role == "row":
                rows.append(b)
            elif role == "col":
                cols.append(b)
            elif role == "header":
                header_boxes.append(b)

        # Sort rows top→bottom, cols left→right
        rows.sort(key=lambda b: b[1])
        cols.sort(key=lambda b: b[0])

        return rows, cols, header_boxes

    # ── cell assembly ─────────────────────────────────────────────────────

    def _extract_cells(self, table_img, rows, cols):
        """
        OCR the intersection of each (row, col) box.
        Returns a 2-D list of cell strings: grid[row_i][col_i].
        """
        grid = []
        for row_box in rows:
            row_cells = []
            for col_box in cols:
                # Cell = intersection of row band and column band
                cell_box = [
                    max(row_box[0], col_box[0]),
                    max(row_box[1], col_box[1]),
                    min(row_box[2], col_box[2]),
                    min(row_box[3], col_box[3]),
                ]
                if cell_box[2] <= cell_box[0] or cell_box[3] <= cell_box[1]:
                    row_cells.append("")
                    continue
                cell_img = _crop(table_img, cell_box)
                row_cells.append(_ocr_cell(cell_img))
            grid.append(row_cells)
        return grid

    # ── public extract ─────────────────────────────────────────────────────

    def extract(self, pdf_path):
        """
        Main entry point.

        Returns a list of table dicts compatible with table_extractor.py:
          [{"headers": [...], "rows": [[...], ...], "page": int, "quality": float}, ...]
        """
        if not self._loaded:
            print("[VisionTableExtractor] Models not loaded — returning []")
            return []

        if not HAS_FITZ:
            print("[VisionTableExtractor] PyMuPDF not available — returning []")
            return []

        results = []

        for page_idx, page_img in self._rasterise(pdf_path):
            print(f"[VisionTableExtractor] Page {page_idx+1}: detecting tables …")
            table_boxes = self._detect_tables(page_img)
            print(f"  → {len(table_boxes)} table(s) found")

            for box in table_boxes:
                table_crop = _crop(page_img, box)
                rows, cols, header_boxes = self._recognise_structure(table_crop)

                if not rows or not cols:
                    continue

                grid = self._extract_cells(table_crop, rows, cols)

                if not grid or len(grid) < 2:
                    continue

                # Identify header row (first row, or rows that overlap header_boxes)
                header_row_idx = 0
                if header_boxes:
                    # row whose y-band overlaps most header detection boxes
                    best_h = -1; best_r = 0
                    for ri, rb in enumerate(rows):
                        overlap = sum(
                            1 for hb in header_boxes
                            if max(rb[1], hb[1]) < min(rb[3], hb[3])
                        )
                        if overlap > best_h:
                            best_h = overlap; best_r = ri
                    header_row_idx = best_r

                headers = grid[header_row_idx]
                data_rows = [
                    grid[i] for i in range(len(grid)) if i != header_row_idx
                ]

                # Basic quality: non-empty cells / total cells
                all_cells = headers + [c for r in data_rows for c in r]
                non_empty = sum(1 for c in all_cells if c.strip())
                quality   = non_empty / max(len(all_cells), 1)

                if quality < 0.2:
                    continue

                results.append({
                    "headers": headers,
                    "rows":    data_rows,
                    "page":    page_idx,
                    "quality": round(quality, 3),
                    "source":  "vision",
                })

        print(f"[VisionTableExtractor] Total: {len(results)} table(s) extracted")
        return results


# ── standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m parser.vision_table_extractor paper.pdf")
        sys.exit(1)

    vte    = VisionTableExtractor(dpi=200)
    tables = vte.extract(sys.argv[1])

    for i, t in enumerate(tables):
        print(f"\nTable {i+1}  (page {t['page']+1}, quality={t['quality']:.0%})")
        print("  Headers:", t["headers"])
        for row in t["rows"][:3]:
            print("  Row:", row)
        if len(t["rows"]) > 3:
            print(f"  … {len(t['rows'])-3} more rows")
