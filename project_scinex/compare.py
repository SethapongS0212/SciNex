"""
compare.py — HTML output similarity scorer
==========================================

Compares your parsed output.html against a reference HTML
(e.g. the arXiv HTML version of the same paper) using three metrics:

  1. Cosine Similarity  — scispacy en_core_sci_lg word vectors
                          (semantic, science-aware)
  2. Edit Similarity    — character-level difflib SequenceMatcher
  3. Structure Sim.     — h1/h2/h3/p/table tag count comparison

USAGE:
    python compare.py <your_output.html> <reference.html>

EXAMPLES:
    # default mode output vs arXiv reference
    python compare.py output/BERT/default/output.html reference/BERT.html

    # compare two different pipeline modes against the same reference
    python compare.py output/BERT/fast/output.html reference/BERT.html
    python compare.py output/BERT/no-llm/output.html reference/BERT.html

INSTALL DEPS (one-time):
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
"""

import sys
import argparse
import difflib

import numpy as np
import spacy
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity


# ── Args ──────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Compare parser output HTML against a reference HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("output_html",    help="Your parsed output HTML file.")
    p.add_argument("reference_html", help="Reference HTML (e.g. arXiv HTML).")
    return p.parse_args()


# ── Load scispacy ─────────────────────────────────────────
# en_core_sci_lg: 600k vocab, 300-d vectors trained on scientific text.
# Semantically related terms score high even when exact words differ.
print("Loading scispacy model...")
nlp = spacy.load("en_core_sci_lg")
nlp.max_length = 2_000_000   # full papers can be long


# ── Text extraction ───────────────────────────────────────
def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# ── scispacy document vector ──────────────────────────────
# Split into ~500-word chunks, vectorise each, then average.
# More stable than feeding the entire paper at once.
def chunk_text(text: str, chunk_words: int = 500) -> list:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_words])
        for i in range(0, len(words), chunk_words)
        if words[i : i + chunk_words]
    ]


def document_vector(text: str) -> np.ndarray:
    chunks = chunk_text(text)
    vectors = []
    for chunk in chunks:
        doc = nlp(chunk)
        if doc.has_vector:
            vectors.append(doc.vector)
    if not vectors:
        raise ValueError("No vectors found — is en_core_sci_lg installed correctly?")
    return np.mean(vectors, axis=0)


# ── Structure similarity ──────────────────────────────────
def count_tags(html: str, tag: str) -> int:
    return len(BeautifulSoup(html, "html.parser").find_all(tag))


def structure_similarity(html1: str, html2: str) -> float:
    tags = ["h1", "h2", "h3", "p", "table"]
    scores = []
    for tag in tags:
        c1 = count_tags(html1, tag)
        c2 = count_tags(html2, tag)
        if max(c1, c2) == 0:
            scores.append(1.0)
        else:
            scores.append(1 - abs(c1 - c2) / max(c1, c2))
    return sum(scores) / len(scores)


# ── Main ──────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n  Output HTML   : {args.output_html}")
    print(f"  Reference HTML: {args.reference_html}\n")

    with open(args.output_html,    "r", encoding="utf-8") as f:
        html1 = f.read()
    with open(args.reference_html, "r", encoding="utf-8") as f:
        html2 = f.read()

    text1 = extract_text(html1)
    text2 = extract_text(html2)

    print("Computing scispacy vectors (may take ~10-30 sec for full papers)...")
    vec1 = document_vector(text1).reshape(1, -1)
    vec2 = document_vector(text2).reshape(1, -1)
    cos_sim = cosine_similarity(vec1, vec2)[0][0]

    lev_sim     = difflib.SequenceMatcher(None, text1, text2).ratio()
    struct_sim  = structure_similarity(html1, html2)
    final_score = (0.6 * cos_sim) + (0.2 * lev_sim) + (0.2 * struct_sim)

    print()
    print(f"  Cosine Similarity  (scispacy) : {cos_sim    * 100:.2f}%")
    print(f"  Edit Similarity    (char-lvl) : {lev_sim    * 100:.2f}%")
    print(f"  Structure Similarity (tags)   : {struct_sim * 100:.2f}%")
    print(f"\n  Final Similarity (weighted)   : {final_score * 100:.2f}%")
    print()


if __name__ == "__main__":
    main()