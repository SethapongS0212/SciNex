"""
html_parser.py
--------------
Extracts clean, section-aware text from the pipeline's output.html files.
Skips: References, figure captions, table cells (noisy for triple extraction).
Returns: list of {section, text} dicts ready for the extractor.
"""

import re
from bs4 import BeautifulSoup


# Sections to skip entirely — they produce garbage triples
SKIP_SECTIONS = {
    "references", "bibliography", "acknowledgements",
    "acknowledgments", "appendix"
}


def _clean_text(text: str) -> str:
    """Remove concept mark tags text, collapse whitespace, fix ligatures."""
    # Remove citation markers like (Vaswani et al., 2017)
    text = re.sub(r'\([A-Z][^)]{0,60}\d{4}[a-z]?\)', '', text)
    # Remove footnote numbers like ¹ ² ³ or superscript digits
    text = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹]|\d+$', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _is_skip_section(heading: str) -> bool:
    heading_lower = heading.lower().strip()
    return any(skip in heading_lower for skip in SKIP_SECTIONS)


def parse_html(html_path: str) -> list[dict]:
    """
    Parse output.html into a list of section dicts:
        [{"section": "Introduction", "text": "Language model pre-training..."}, ...]

    Paragraphs within the same section are joined. Figures and tables are skipped.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        raw_html = f.read()
    raw_html = raw_html.replace('<mark <mark class="concept">class</mark>="concept">', '<mark class="concept">')
    soup = BeautifulSoup(raw_html, "html.parser")

    # Extract paper title
    title_tag = soup.find("h1")
    paper_title = title_tag.get_text(separator=" ").strip() if title_tag else "Unknown"

    sections = []
    current_section = "Abstract"
    current_paragraphs = []
    skip_current = False

    # Walk through article content linearly; ignore sidebar/navigation headings.
    body = soup.find(id="main") or soup.find(class_="ltx_page_content") or soup.find("body") or soup
    heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6")
    for tag in body.find_all([*heading_tags, "p", "figure", "table"]):

        if tag.name in heading_tags:
            # h1 is the paper title, not a section boundary. Keep the default
            # Abstract section active for text that appears before the first h2.
            if tag.name == "h1":
                continue

            # Save previous section if it has content.
            if current_paragraphs and not skip_current:
                sections.append({
                    "section": current_section,
                    "text": " ".join(current_paragraphs)
                })
            heading_text = tag.get_text(separator=" ").strip()
            current_section = _clean_text(heading_text)
            skip_current = _is_skip_section(current_section)
            current_paragraphs = []

        elif tag.name == "p" and not skip_current:
            # Get text, stripping mark tags but keeping their content
            text = tag.get_text(separator=" ").strip()
            text = _clean_text(text)
            # Skip very short paragraphs (likely table labels, noise)
            if len(text.split()) >= 8:
                current_paragraphs.append(text)

        # Skip figure/table tags entirely

    # Don't forget last section
    if current_paragraphs and not skip_current:
        sections.append({
            "section": current_section,
            "text": " ".join(current_paragraphs)
        })

    return paper_title, sections


def split_into_sentences(text: str, max_words: int = 80) -> list[str]:
    """
    Split text into sentence-sized chunks for the REBEL model.
    REBEL works best on single sentences or short passages (<512 tokens).
    Falls back to simple period-splitting if spacy is unavailable.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if len(s.text.split()) >= 5]
    except Exception:
        # Simple fallback: split on ". " or ".\n"
        raw = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw if len(s.split()) >= 5]

    # Merge very short sentences, split very long ones
    result = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= max_words:
            result.append(sent)
        else:
            # Chunk long sentences
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                if chunk:
                    result.append(chunk)

    return result
