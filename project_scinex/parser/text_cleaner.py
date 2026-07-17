"""
parser/text_cleaner.py — PDF text artifact cleaner
====================================================

Fast, GPU-free text cleaning for PDF-extracted content.
Covers all common extraction artifacts:

  1. ftfy          — encoding errors, null bytes, garbled unicode
  2. Soft hyphens  — U+00AD invisible join hints → removed
  3. Ligatures     — ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi, ﬄ→ffl etc.
  4. Hyphen breaks — "trans-\\nformer" → "transformer"
  5. Stray newlines / tabs → single space
  6. Zero-width / non-printing chars → removed
  7. Unicode dashes mid-word (U+2010/2011) → standard hyphen
  8. Multiple spaces → single space
  9. Spaced punctuation — " ," → ","  " ." → "."

INSTALL:
    pip install ftfy
"""

import re

try:
    import ftfy as _ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False
    print("[text_cleaner] ftfy not found — install with: pip install ftfy")

# PDF ligature characters not always caught by ftfy
_LIGATURES = {
    "\ufb00": "ff",  "\ufb01": "fi",  "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
}


def clean_text(text: str) -> str:
    """Clean a single string of PDF-extracted text."""
    if not text:
        return text

    # 1. ftfy — encoding artifacts, null bytes, garbled unicode
    if _FTFY_AVAILABLE:
        text = _ftfy.fix_text(text)

    # 2. Soft hyphens (U+00AD) — invisible hints that split words in PDFs
    text = text.replace("\u00ad", "")

    # 3. Ligatures
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)

    # 4. Hyphenated line-breaks: "trans-\nformer" → "transformer"
    text = re.sub(r"-\s*\n\s*", "", text)

    # 5. Stray newlines and tabs → single space
    text = re.sub(r"[\n\t]+", " ", text)

    # 6. Zero-width and non-printing characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\ufffd]", "", text)

    # 7. Unicode dashes mid-word (non-breaking hyphen variants) → hyphen
    text = re.sub(r"[\u2010\u2011]", "-", text)

    # 8. Multiple spaces → single space
    text = re.sub(r" {2,}", " ", text)

    # 9. Spaced punctuation: " ," → ","  " ." → "."
    text = re.sub(r" ([,\.;:!?])", r"\1", text)

    return text.strip()


def clean_block(block: dict) -> dict:
    """
    Clean a single content block dict in-place.
    Only processes blocks where needs_llm=True.
    """
    text = block.get("text", "")
    if not text or not block.get("needs_llm"):
        return block
    block["text"] = clean_text(text)
    return block


def clean_document(doc: dict) -> dict:
    """
    Clean all content blocks across every section of a parsed document.
    Drop-in replacement for refine_document_fast() in main.py.
    """
    for section in doc.get("sections", []):
        section["content"] = [clean_block(b) for b in section.get("content", [])]
    return doc