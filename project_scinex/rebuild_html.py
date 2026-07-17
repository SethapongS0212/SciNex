"""
rebuild_html.py — Offline table post-processor
===============================================
Re-generates output.html from an existing output.json WITHOUT
re-running the LLM or PDF extraction. Useful if you want to
tweak table layout or re-run the HTML generator after manual
edits to output.json.

NOTE: In normal usage you do NOT need this script.
      main.py now handles table rebuild automatically as step 8.
      Use this only if you have an old output.json and want to
      re-render it without re-parsing the PDF.

Usage:
    python rebuild_html.py <output.json> <output.html> [<paper.pdf>]

Examples:
    python rebuild_html.py output/paper/output.json output/paper/output.html paper.pdf
    python rebuild_html.py output/paper/output.json output/paper/output.html
"""

import json
import re
import sys
import os

from parser.html_generator import to_html
from parser.table_extractor import (
    is_reference_table, dedup_tables,
    fix_caption_header_tables
)
from parser.structure_builder import (
    build_table_cell_set, is_table_data_paragraph,
    is_table_caption
)


def _all_tables_from_doc(doc):
    tables = []
    for section in doc.get("sections", []):
        for item in section.get("content", []):
            if item.get("type") == "table":
                tables.append(item)
    return tables


def rebuild(json_path, out_html_path, pdf_path=None):
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)

    all_tables = _all_tables_from_doc(doc)
    print(f"Tables in JSON: {len(all_tables)}")

    if pdf_path and os.path.exists(pdf_path):
        print(f"Injecting live tables from: {pdf_path}")
        from parser.text_table_reconstructor import reconstruct_text_tables
        live_tables = reconstruct_text_tables(pdf_path)
        if live_tables:
            print(f"  + {len(live_tables)} tables from text reconstructor")
            for t in all_tables:
                t.setdefault("quality", 0.5)
                t.setdefault("page", 0)
            all_tables = all_tables + live_tables
    else:
        if pdf_path:
            print(f"Warning: PDF not found at {pdf_path}, skipping live extraction")
        for t in all_tables:
            t.setdefault("quality", 0.5)
            t.setdefault("page", 0)

    print(f"Tables before dedup: {len(all_tables)}")
    all_tables = fix_caption_header_tables(all_tables)
    clean_tables = dedup_tables(all_tables)
    print(f"Tables after dedup: {len(clean_tables)}")
    cell_set = build_table_cell_set(clean_tables)

    for section in doc["sections"]:
        section["content"] = [
            item for item in section.get("content", [])
            if item.get("type") != "table"
        ]

    tables_to_place = list(clean_tables)
    for section in doc["sections"]:
        new_content = []
        for item in section.get("content", []):
            itype = item.get("type")
            text  = item.get("text", "")
            if itype == "paragraph" and is_table_data_paragraph(text, cell_set, threshold=4):
                continue
            if itype == "figure" and is_table_caption(text):
                if tables_to_place:
                    tbl = dict(tables_to_place.pop(0))
                    tbl["caption"] = text
                    new_content.append(tbl)
                continue
            new_content.append(item)
        section["content"] = new_content

    if tables_to_place:
        target = None
        for s in doc["sections"]:
            if not s.get("heading", "").lower().startswith("ref"):
                target = s
        if target:
            for tbl in tables_to_place:
                target["content"].append(tbl)

    html = to_html(doc)
    os.makedirs(os.path.dirname(os.path.abspath(out_html_path)), exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ HTML written -> {out_html_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    json_path = sys.argv[1]
    html_path = sys.argv[2]
    pdf_arg   = sys.argv[3] if len(sys.argv) > 3 else None
    rebuild(json_path, html_path, pdf_path=pdf_arg)
