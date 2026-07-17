import re
from html import escape

# ─────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────
def wrap(tag, content, attrs=""):
    if attrs:
        return f"<{tag} {attrs}>{content}</{tag}>"
    return f"<{tag}>{content}</{tag}>"


def clean(text):
    if not text:
        return ""
    return " ".join(str(text).split())


# ─────────────────────────────────────────────────────────────
# HEADING CLASSIFIER
# Returns (depth, ltx_class, id_prefix)
#   depth 0 = abstract
#   depth 1 = top-level  → ltx_section / ltx_appendix
#   depth 2 = subsection → ltx_subsection
#   depth 3 = sub-sub    → ltx_subsubsection
# ─────────────────────────────────────────────────────────────
def _classify_heading(heading: str):
    h = heading.strip()

    if re.match(r"^abstract$", h, re.I):
        return 0, "ltx_abstract", None

    # Numeric: "1 Intro", "1.1 Sub", "1.1.1 SubSub"
    m = re.match(r"^(\d+)(\.(\d+))?(\.(\d+))?\s+", h)
    if m:
        if m.group(5):
            return 3, "ltx_subsubsection", f"{m.group(1)}.{m.group(3)}.{m.group(5)}"
        if m.group(3):
            return 2, "ltx_subsection", f"{m.group(1)}.{m.group(3)}"
        return 1, "ltx_section", m.group(1)

    # Roman numeral: "I Intro", "II Related Work"
    if re.match(r"^[IVX]{1,4}\.?\s+[A-Z]", h):
        return 1, "ltx_section", None

    # Appendix subsection: "A.1 Details", "B.2 More"
    if re.match(r"^[A-Z]\.\d+\s+[A-Z]", h):
        return 2, "ltx_subsection", None

    # Appendix top-level: "A Additional Details", "B Experiments"
    if re.match(r"^[A-Z]\s+[A-Z]", h):
        letter = h[0]
        return 1, "ltx_appendix", letter

    # ALL CAPS heading
    if h.replace(" ", "").isupper() and len(h.split()) > 1:
        return 1, "ltx_section", None

    return 1, "ltx_section", None


# ─────────────────────────────────────────────────────────────
# CONCEPT HIGHLIGHTING
# ─────────────────────────────────────────────────────────────
def highlight_concepts(text, concepts):
    if not concepts:
        return escape(text)
    sorted_concepts = sorted(concepts, key=len, reverse=True)
    escaped = escape(text)
    for kw in sorted_concepts:
        pattern = re.compile(
            r'(?i)(?<!\w)(' + re.escape(escape(kw)) + r')(?!\w)'
        )
        escaped = pattern.sub(r'<mark class="concept">\1</mark>', escaped)
    return escaped


# ─────────────────────────────────────────────────────────────
# ELEMENT RENDERERS
# ─────────────────────────────────────────────────────────────
def render_paragraph(block, elem_id=None):
    text = clean(block.get("text", "") if isinstance(block, dict) else block)
    if not text:
        return ""
    concepts = block.get("concepts", []) if isinstance(block, dict) else []
    inner    = highlight_concepts(text, concepts)
    id_attr  = f' id="{elem_id}"' if elem_id else ""
    return f'<p class="ltx_p"{id_attr}>{inner}</p>'


def render_heading(text, level=2, elem_id=None):
    text  = clean(text)
    if not text:
        return ""
    level = min(max(int(level), 1), 6)
    id_attr = f' id="{elem_id}"' if elem_id else ""
    return f"<h{level}{id_attr}>{escape(text)}</h{level}>"


def render_pre(text, elem_id=None):
    text = clean(text)
    if not text:
        return ""
    id_attr = f' id="{elem_id}"' if elem_id else ""
    return f"<pre{id_attr}>{escape(text)}</pre>"


def render_figure(block, elem_id=None):
    text    = clean(block.get("text", ""))
    caption = clean(block.get("caption", ""))
    id_attr = f' id="{elem_id}"' if elem_id else ""
    content = ""
    if text:
        content += f"<div>{escape(text)}</div>"
    if caption:
        content += f'<figcaption class="ltx_caption">{escape(caption)}</figcaption>'
    return f"<figure{id_attr}>{content}</figure>"


def render_table(table, elem_id=None):
    headers = table.get("headers", [])
    rows    = table.get("rows", [])
    caption = table.get("caption", "")
    # ltx_table on the figure wrapper — her code checks parent figure class
    id_attr = f' id="{elem_id}"' if elem_id else ""
    fig_html = f'<figure class="ltx_table"{id_attr}>'
    tbl_html = "<table>"
    if caption:
        tbl_html += f'<caption class="ltx_caption">{escape(str(caption))}</caption>'
    if headers:
        tbl_html += "<thead><tr>" + "".join(
            f"<th>{escape(str(h))}</th>" for h in headers
        ) + "</tr></thead>"
    if rows:
        tbl_html += "<tbody>"
        for row in rows:
            tbl_html += "<tr>" + "".join(
                f"<td>{escape(str(cell))}</td>" for cell in row
            ) + "</tr>"
        tbl_html += "</tbody>"
    tbl_html += "</table>"
    return fig_html + tbl_html + "</figure>"


def render_concept_sidebar(concept_dict):
    if not concept_dict:
        return ""
    items = "".join(
        f"<li><span class='concept-kw'>{escape(kw)}</span> "
        f"<span class='concept-freq'>({freq})</span></li>"
        for kw, freq in sorted(concept_dict.items(), key=lambda x: -x[1])[:20]
    )
    return (
        f'<aside class="concept-sidebar" id="concept-sidebar">'
        f"<h3>Related Concepts <small>(from citing papers)</small></h3>"
        f"<ul>{items}</ul>"
        f"</aside>"
    )


# ─────────────────────────────────────────────────────────────
# SECTION ID GENERATOR
# Tracks counters to produce arXiv-style IDs:
#   S1, S1.SS1, S1.SS1.SSS1, A1 (appendix), etc.
# ─────────────────────────────────────────────────────────────
class _IdGen:
    def __init__(self):
        self._sec      = 0    # top-level section counter
        self._app      = 0    # appendix counter
        self._sub      = {}   # subsection counters per parent
        self._subsub   = {}   # sub-sub counters per parent

    def next_section(self, ltx_class: str, heading_prefix=None) -> str:
        if ltx_class == "ltx_appendix":
            self._app += 1
            letter = heading_prefix if heading_prefix else chr(ord("A") + self._app - 1)
            sid = f"A{letter}"
        else:
            self._sec += 1
            sid = f"S{self._sec}"
        # Reset sub-counters for this new parent
        self._sub[sid]    = 0
        self._subsub[sid] = 0   # int counter (flat-keyed); was {} → crashed when a
                                # subsubsection hung directly under a section
        return sid

    def next_subsection(self, parent_id: str) -> str:
        self._sub.setdefault(parent_id, 0)
        self._sub[parent_id] += 1
        ssid = f"{parent_id}.SS{self._sub[parent_id]}"
        self._subsub[ssid] = 0
        return ssid

    def next_subsubsection(self, parent_id: str) -> str:
        cur = self._subsub.get(parent_id, 0)
        if not isinstance(cur, int):   # defensive: never let a stray seed type crash +=
            cur = 0
        cur += 1
        self._subsub[parent_id] = cur
        return f"{parent_id}.SSS{cur}"


# ─────────────────────────────────────────────────────────────
# SECTION CONTENT RENDERER
# Renders the content blocks within a section, returning HTML
# and per-type counters (para / fig / tbl / eq)
# ─────────────────────────────────────────────────────────────
def _render_content_blocks(content, sec_id):
    html         = []
    para_counter = 0
    fig_counter  = 0
    tbl_counter  = 0
    eq_counter   = 0

    for item in content:
        t    = item.get("type")
        text = item.get("text", "")

        if t == "paragraph":
            para_counter += 1
            html.append(render_paragraph(item, elem_id=f"{sec_id}.p{para_counter}"))

        elif t == "figure":
            fig_counter += 1
            html.append(render_figure(item, elem_id=f"{sec_id}.F{fig_counter}"))

        elif t == "table":
            tbl_counter += 1
            html.append(render_table(item, elem_id=f"{sec_id}.T{tbl_counter}"))

        elif t in ("equation", "code"):
            eq_counter += 1
            html.append(render_pre(text, elem_id=f"{sec_id}.E{eq_counter}"))

    return "\n".join(html)


# ─────────────────────────────────────────────────────────────
# SECTION NESTING
# Takes the flat list of sections from structure_builder and
# groups subsections inside their parent top-level section.
# Returns a nested list:
#   [
#     { "heading": ..., "ltx_class": ..., "sec_id": ..., "content": [...],
#       "children": [ { subsection dicts }, ... ] },
#     ...
#   ]
# ─────────────────────────────────────────────────────────────
def _build_section_tree(flat_sections):
    id_gen = _IdGen()
    tree   = []
    current_top = None   # current top-level section node
    current_sub = None   # current subsection node

    for section in flat_sections:
        heading = section.get("heading", "")
        content = section.get("content", [])

        depth, ltx_class, prefix = _classify_heading(heading)

        if depth == 0:
            # Abstract — treat as a special top-level node
            node = {
                "heading":   heading,
                "ltx_class": "ltx_abstract",
                "sec_id":    "abstract",
                "content":   content,
                "children":  [],
                "h_level":   2,
            }
            tree.append(node)
            current_top = node
            current_sub = None

        elif depth == 1:
            sec_id = id_gen.next_section(ltx_class, heading_prefix=prefix)
            node = {
                "heading":   heading,
                "ltx_class": ltx_class,
                "sec_id":    sec_id,
                "content":   content,
                "children":  [],
                "h_level":   2,
            }
            tree.append(node)
            current_top = node
            current_sub = None

        elif depth == 2:
            parent = current_top or (tree[-1] if tree else None)
            if parent:
                ssid = id_gen.next_subsection(parent["sec_id"])
            else:
                # Orphaned subsection — promote to top-level
                ssid = id_gen.next_section("ltx_section")
            node = {
                "heading":   heading,
                "ltx_class": "ltx_subsection",
                "sec_id":    ssid,
                "content":   content,
                "children":  [],
                "h_level":   3,
            }
            if parent:
                parent["children"].append(node)
            else:
                tree.append(node)
            current_sub = node

        elif depth == 3:
            parent = current_sub or current_top or (tree[-1] if tree else None)
            if parent:
                sssid = id_gen.next_subsubsection(parent["sec_id"])
            else:
                sssid = id_gen.next_section("ltx_section")
            node = {
                "heading":   heading,
                "ltx_class": "ltx_subsubsection",
                "sec_id":    sssid,
                "content":   content,
                "children":  [],
                "h_level":   4,
            }
            if parent:
                parent["children"].append(node)
            else:
                tree.append(node)

    return tree


# ─────────────────────────────────────────────────────────────
# RECURSIVE SECTION RENDERER
# ─────────────────────────────────────────────────────────────
def _render_section(node) -> str:
    html      = []
    sec_id    = node["sec_id"]
    ltx_class = node["ltx_class"]
    heading   = node["heading"]
    h_level   = node["h_level"]
    content   = node["content"]
    children  = node["children"]

    html.append(f'<section class="{ltx_class}" id="{sec_id}">')

    if heading and ltx_class != "ltx_abstract":
        html.append(render_heading(heading, level=h_level, elem_id=f"{sec_id}.heading"))
    elif ltx_class == "ltx_abstract" and heading:
        html.append(f'<h2 id="{sec_id}.heading">Abstract</h2>')

    # Render direct content blocks
    html.append(_render_content_blocks(content, sec_id))

    # Render nested children (subsections)
    for child in children:
        html.append(_render_section(child))

    html.append("</section>")
    return "\n".join(html)


# ─────────────────────────────────────────────────────────────
# MAIN HTML GENERATOR
# ─────────────────────────────────────────────────────────────
def to_html(doc):
    html = []

    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append('<meta charset="UTF-8">')
    html.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    title_text = escape(clean(doc.get("title", "Research Paper")))
    html.append(f"<title>{title_text}</title>")
    # citation_title meta — her extract_title() checks this first
    html.append(f'<meta name="citation_title" content="{title_text}">')
    html.append("""
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: Georgia, serif;
    line-height: 1.75;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 24px;
    color: #1a1a1a;
    background: #fafafa;
  }
  h1 { font-size: 1.9em; margin-bottom: 6px; color: #111; line-height: 1.3; }
  h2 {
    font-size: 1.25em; margin-top: 2.2em; color: #1a1a2e;
    border-bottom: 2px solid #dee2e6; padding-bottom: 5px;
  }
  h3 { font-size: 1.05em; color: #333; margin-top: 1.4em; }
  h4 { font-size: 0.95em; color: #555; margin-top: 1.2em; }
  p { margin: 0.65em 0; text-align: justify; }
  table {
    margin: 1.4em auto; border-collapse: collapse;
    width: 100%; font-size: 0.88em;
    background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  caption {
    caption-side: top; text-align: left;
    font-style: italic; font-size: 0.9em;
    color: #444; margin-bottom: 6px; padding: 0 4px;
  }
  thead th {
    background: #e8edf3; color: #1a1a2e;
    font-weight: 600; padding: 8px 10px;
    border: 1px solid #c8d0db; text-align: left;
  }
  tbody td {
    padding: 6px 10px; border: 1px solid #dde1e7; vertical-align: top;
  }
  tbody tr:nth-child(even) { background: #f4f6f9; }
  tbody tr:hover           { background: #eaf0fb; }
  figure {
    font-style: italic; margin: 1.2em 0; color: #555;
    padding: 10px 14px; border-left: 3px solid #adb5bd;
    background: #f8f9fa; border-radius: 0 4px 4px 0;
  }
  figcaption { margin-top: 4px; font-size: 0.9em; color: #666; }
  pre {
    background: #f1f3f5; padding: 12px 16px;
    overflow-x: auto; font-size: 0.84em;
    border-radius: 5px; border: 1px solid #dee2e6; line-height: 1.5;
  }
  mark.concept {
    background: #fff3cd; border-bottom: 2px solid #f0ad4e;
    padding: 0 2px; border-radius: 2px; cursor: help;
  }
  .ltx_abstract {
    background: #f0f4f8; border-left: 4px solid #4a90d9;
    padding: 14px 18px; margin: 1.5em 0; border-radius: 0 4px 4px 0;
  }
  aside.concept-sidebar {
    float: right; width: 230px; margin: 0 0 24px 32px;
    padding: 14px 18px; background: #f8f9fa;
    border: 1px solid #dee2e6; border-radius: 6px; font-size: 0.83em;
  }
  aside.concept-sidebar h3 { margin-top: 0; font-size: 0.93em; }
  aside.concept-sidebar ul { padding-left: 16px; margin: 6px 0 0; }
  aside.concept-sidebar li { margin: 3px 0; }
  .concept-freq { color: #999; font-size: 0.82em; }
  @media (max-width: 680px) {
    aside.concept-sidebar { float: none; width: auto; margin: 0 0 20px; }
    table { font-size: 0.8em; }
  }
  @media print {
    body { background: #fff; }
    aside.concept-sidebar { display: none; }
    table { box-shadow: none; }
  }
</style>
""")
    html.append("</head>")
    html.append("<body>")

    # ── Title ─────────────────────────────────────────────────────────
    # class="ltx_title" so extract_title() finds it via h1[class~=title]
    if doc.get("title"):
        html.append(
            f'<h1 class="ltx_title" id="title">{escape(clean(doc["title"]))}</h1>'
        )

    # ── Concept sidebar ───────────────────────────────────────────────
    concept_dict = doc.get("citation_concepts", {})
    html.append(render_concept_sidebar(concept_dict))

    # ── Main content wrapper ──────────────────────────────────────────
    # ltx_page_content — her _find_content_root() looks for this class
    html.append('<div class="ltx_page_content" id="main">')

    # ── Build nested section tree and render ──────────────────────────
    flat_sections = doc.get("sections", [])
    section_tree  = _build_section_tree(flat_sections)

    for node in section_tree:
        html.append(_render_section(node))

    html.append("</div>")   # end ltx_page_content
    html.append("</body></html>")
    return "\n".join(html)