# PDF → HTML Research Paper Parser

Parses academic PDFs into clean, readable HTML with optional citation network enrichment.

---

## Quickstart

```bash
# Fast mode — no GPU, no internet needed (~10-30 sec)
python main.py paper.pdf --fast

# Full mode — GPU + internet (5-20 min)
python main.py paper.pdf
```

**The result is always:** `output/<pdf_name>/output.html`

---

## Why is it slow? (and how to fix it)

| Bottleneck | Time cost | How to skip |
|---|---|---|
| **Qwen2.5-14B LLM** (text cleanup) | 5–15 min (GPU load + inference) | `--no-llm` or `--fast` |
| **Semantic Scholar API** (citation network) | 1–5 min (20–60 API calls) | `--no-citations` or `--fast` |
| **Table extraction** (multi-strategy) | 10–30 sec | always runs, cannot skip |

**`--fast` skips both LLM and citations.** The only difference you'll notice:
- Text may have occasional broken hyphenated words (e.g. "trans- former") — the regex cleaner handles most cases anyway.
- No "Related Concepts" sidebar in the HTML.

---

## Output files

All outputs go to `output/<pdf_name>/`:

```
output/
  my_paper/
    output.html            ← THE RESULT — open in browser
    output.json            ← Structured JSON (sections, tables, concepts)
    citation_network.json  ← Citation graph (only if --no-citations not set)
```

There is only **one** HTML file per run. No more `output_fixed.html` confusion.

---

## Flags

```
python main.py paper.pdf [OPTIONS]

Positional:
  pdf_path              Path to the input PDF

Options:
  --fast                Skip LLM + citations. Fastest, no GPU needed.
  --no-llm              Skip Qwen2.5-14B LLM text refinement only.
  --no-citations        Skip Semantic Scholar citation fetch only.
  --citation-limit N    Max related papers to fetch (default: 10).
  --output-dir PATH     Custom output directory.
```

---

## How citation enrichment works

When citations are enabled (`--no-citations` NOT set):

1. The tool searches **Semantic Scholar** for this paper by title.
2. It fetches papers that **cite this paper** (citing) and papers **this paper cites** (references).
3. It extracts **concept keywords** from all abstracts.
4. Keywords are **highlighted in yellow** in the HTML.
5. A **"Related Concepts" sidebar** appears on the right.
6. The full citation graph is saved as `citation_network.json`.

The network is **cached** — re-running on the same paper reads from cache instantly.

### Using the citation network for deeper research

```python
from citation.auto_fetcher import CitationFetcher

fetcher = CitationFetcher(output_dir="output/my_paper", workers=4)
network = fetcher.build_network(
    seed_title="Attention Is All You Need",
    depth=2,           # hop depth (1 = direct citations only)
    max_per_hop=20,    # max papers fetched per hop
    download_pdfs=True # download PDFs of related papers
)
fetcher.save()
```

This will:
- Discover related papers up to 2 hops away in the citation graph
- Download their PDFs to `output/my_paper/papers/`
- Store all metadata in `output/my_paper/citation_db.json`

You can then parse each downloaded PDF through `main.py` for a full multi-paper HTML library.

---

## Project structure

```
pdf_parser/
├── main.py                    ← Entry point
├── config.py                  ← OUTPUT_DIR, CITATION_LIMIT
├── requirement.txt            ← Dependencies
├── rebuild_html.py            ← Offline table-fix tool (standalone)
│
├── parser/
│   ├── pdf_loader.py          ← Load PDF with PyMuPDF
│   ├── layout.py              ← Extract text blocks + coordinates
│   ├── table_extractor.py     ← Multi-strategy table detection
│   ├── table_detector.py      ← Heuristics for table region detection
│   ├── text_table_reconstructor.py ← Text-based table reconstruction
│   ├── vision_table_extractor.py   ← Vision-based table extraction
│   ├── structure_builder.py   ← Build section/heading structure
│   ├── html_generator.py      ← Render JSON doc → HTML
│   └── llm_refiner.py         ← Qwen2.5-14B text cleanup (optional)
│
└── citation/
    ├── network.py             ← Semantic Scholar citation graph builder
    ├── auto_fetcher.py        ← Auto-download related papers
    ├── semantic.py            ← Semantic Scholar API helpers
    └── concept_builder.py     ← Extract concept keywords from abstracts
```

---

## Dependencies

```bash
pip install -r requirement.txt
```

For LLM refinement (`--no-llm` skips this):
```bash
pip install transformers accelerate bitsandbytes torch
```
Requires a CUDA-capable GPU with ≥16 GB VRAM for Qwen2.5-14B.
