import re
from collections import Counter


STOPWORDS = {
    "these", "those", "therefore", "however", "because",
    "their", "about", "which", "with", "that", "this",
    "based", "using", "study", "paper", "method"
}


def extract_keywords(text):
    if not text:
        return []

    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())

    return [
        w for w in words
        if w not in STOPWORDS
    ]


def build_concept_dict(citations):
    counter = Counter()

    for c in citations:
        paper = c.get("citingPaper", {})
        abstract = paper.get("abstract")

        if not abstract:
            continue

        words = extract_keywords(abstract)
        counter.update(words)

    return dict(counter.most_common(50))