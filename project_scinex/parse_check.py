"""
parse_check.py
--------------
Parse PDFs with the Part-1 parser and print structure stats, for diffing parser
behaviour before/after a change. No GPU, no LLM — pure parsing.

Prints per PDF: title, section count, paragraph/table/figure counts, body words,
reference words, and the full heading list. `body words` = every paragraph OUTSIDE
the References section, which is the number that actually matters: a paper can look
"parsed" while containing nothing but its bibliography (see hands_off.md §9).

Usage (from the project root):
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python parse_check.py BERT.pdf papers/*.pdf --out after.json

  - PYTHONPATH=. so `parser/` resolves as the local package.
  - PYTHONIOENCODING=utf-8 on Windows, or a `✔` printed by citation/network.py
    kills the run with UnicodeEncodeError under cp1252.
  - ALWAYS include BERT.pdf: it is the ACL-style regression check for the frozen
    315-paper corpus. Its section list and body-word count must not degrade.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import fitz  # noqa: E402

from parser.layout import extract_blocks  # noqa: E402
from parser.structure_builder import build_structure  # noqa: E402
from parser.table_extractor import extract_tables  # noqa: E402


def words(s):
    return len(s.split())


def stats(pdf):
    doc = fitz.open(pdf)
    blocks = extract_blocks(doc)
    try:
        tables = extract_tables(pdf, blocks=blocks)
    except Exception as e:
        print(f"  !! table extraction failed: {type(e).__name__}: {e}")
        tables = []
    structured = build_structure(blocks, tables)

    body = refs = 0
    npara = ntab = nfig = 0
    heads = []
    for sec in structured["sections"]:
        h = sec.get("heading", "")
        heads.append(h)
        is_ref = h.strip().lower().startswith("reference")
        for item in sec.get("content", []):
            t = item.get("type")
            if t == "paragraph":
                n = words(item.get("text", ""))
                if is_ref:
                    refs += n
                else:
                    body += n
                npara += 1
            elif t == "table":
                ntab += 1
            elif t == "figure":
                nfig += 1

    print(f"  title    : {structured['title']!r}")
    print(f"  sections : {len(structured['sections'])}")
    print(f"  paras    : {npara}   tables: {ntab}   figures: {nfig}")
    print(f"  body words: {body}   ref words: {refs}")
    print(f"  headings : {' | '.join(heads)}")
    return {
        "title": structured["title"],
        "sections": len(structured["sections"]),
        "paras": npara,
        "tables": ntab,
        "figures": nfig,
        "body_words": body,
        "ref_words": refs,
        "headings": heads,
    }


if __name__ == "__main__":
    out = {}
    args = sys.argv[1:]
    dest = "stats.json"
    if "--out" in args:
        i = args.index("--out")
        dest = args[i + 1]
        args = args[:i] + args[i + 2:]
    for pdf in args:
        print(f"== {pdf}")
        try:
            out[pdf] = stats(pdf)
        except Exception as e:
            import traceback
            print(f"  !! CRASH: {type(e).__name__}: {e}")
            traceback.print_exc()
            out[pdf] = {"crash": f"{type(e).__name__}: {e}"}
        print()
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
