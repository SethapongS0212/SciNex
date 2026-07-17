"""
enrich_entity_csv.py
--------------------
Enrich the (title-only) ACL entity CSVs into a real subject pool for the FIXED
extractor.

Background: the ACL/CS-NER dataset only annotates paper *titles*, so its entity
CSVs have ~2-3 entities each — far too few for the fixed extractor, whose output
is gated on the subject list. This script harvests the real in-paper entities and
merges them into the CSV. The original ACL title entities are kept as
guaranteed-correct seeds.

Two entity sources (--source):
  llm   (default) — harvest subjects/objects from the open-extraction triples we
                    already computed (output/<paper>/kg/llm/<model>/triples.json).
                    No install, no GPU; entities come pre-typed (Method/Task/...).
  iter            — run the ITER/SciERC extractor over output/<paper>/no-llm/
                    output.html. Needs `pip install git+https://github.com/fleonce/iter`.

Output CSV columns match what entity_loader / the fixed extractor expect:
    Entity, Abbreviation, Aliases, TP, NER_Type, CEO_Type   (TP=1 for all rows)

Usage:
    # one paper, harvest from existing open-extraction triples (default source)
    python3 enrich_entity_csv.py --paper D17-1028

    # all papers, overwrite Entity_<id>.csv in place
    python3 enrich_entity_csv.py --all --in-place

    # NOTE: keep --min-count at 1 (default) for per-paper enrichment. A single paper
    # rarely repeats an entity, so --min-count 2+ deletes most of the subject pool
    # and starves the fixed extractor (it produced only 1 triple in testing).

    # use the ITER/SciERC extractor instead (requires the iter package)
    python3 enrich_entity_csv.py --all --source iter

    # then feed it to the fixed extractor:
    python3 kg_main.py --paper D17-1028 --extractor fixed \
        --entity-csv output/acl/D17-1028/Entity_D17-1028_enriched.csv --model Qwen/Qwen3-14B
"""

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("enrich")

OUTPUT_DIR   = Path("output")
ACL_DIR      = OUTPUT_DIR / "acl"
CSV_FIELDS   = ["Entity", "Abbreviation", "Aliases", "TP", "NER_Type", "CEO_Type"]

# SciERC entity type → CEO type (None = drop, e.g. generic "this method" mentions)
SCIERC_TO_CEO = {
    "Method":              "Method",
    "Task":                "Task",
    "Metric":              "EvaluationMetric",
    "Material":            "Resource",
    "OtherScientificTerm": "Other",
    "Generic":             None,
}

# Free-LLM extraction type → CEO type (LLM types are already CEO-ish; keep all)
LLM_TO_CEO = {
    "Model":   "Model",   "Method":  "Method",  "Dataset": "Dataset",
    "Task":    "Task",    "Tool":    "Tool",    "Resource": "Resource",
    "Metric":  "EvaluationMetric",  "Language": "Language",
}

# ── CS-NER gazetteer (source: csner) ──────────────────────────────────────────
# A single global "scientific entity gazetteer" aggregated from ALL CS-NER files
# (github.com/jd-coderepos/contributions-ner-cs). For each paper we keep only the
# gazetteer entities that actually appear in that paper's text (mechanism B:
# per-paper intersection). Non-circular (external human annotation), not from our
# own open extraction. Built once → cached at CSNER_GAZETTEER_CSV.
CSNER_REPO_BASE = "https://raw.githubusercontent.com/jd-coderepos/contributions-ner-cs/main"
CSNER_FILES = [
    "acl/train.data", "acl/dev.data", "acl/test.data",                 # 7 types (title-level)
    "full dataset/train-abs.data", "full dataset/dev-abs.data",        # abstract-level
    "full dataset/test-abs.data",                                      #   (method/research_problem)
    "ncg/train-abs.data", "pwc/train-abs.data",
    "scierc/train-abs.data", "ftd/train-abs.data",
]
# CS-NER BIOES type (uppercase) → CEO type. Mirrors acl_pipeline.CSNER_TO_CEO_TYPE.
CSNER_TO_CEO = {
    "SOLUTION":         "Model",
    "METHOD":           "Method",
    "DATASET":          "Dataset",
    "RESEARCH_PROBLEM": "Task",
    "TOOL":             "Tool",
    "RESOURCE":         "Resource",
    "LANGUAGE":         "Language",
}
CSNER_GAZETTEER_CSV = ACL_DIR / "csner_gazetteer.csv"
CSNER_MAX_NGRAM = 8   # longest gazetteer entity (in words) to match in paper text

# token = alnum run, keeping internal - / + . (so "u-net", "word2vec" stay whole)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/+.]*")

# Function/stop words that occasionally appear as single-token CS-NER annotation
# noise (e.g. "and"/"for"/"with"/"use") and would otherwise match every paper.
# Used ONLY to filter SINGLE-token gazetteer entries; multi-word entries keep all.
_FUNCTION_WORDS = {
    "a", "an", "and", "or", "but", "the", "of", "for", "to", "in", "on", "at", "by",
    "as", "is", "are", "was", "were", "be", "been", "being", "am",
    "this", "that", "these", "those", "it", "its", "we", "our", "ours", "their",
    "they", "them", "you", "your", "he", "she", "his", "her", "i",
    "can", "could", "may", "might", "will", "would", "shall", "should", "must",
    "do", "does", "did", "done", "has", "have", "had",
    "use", "used", "uses", "using", "find", "found", "make", "made", "get", "got",
    "show", "shown", "give", "given", "take", "see", "form", "set", "let",
    "new", "old", "present", "propose", "proposed", "simple", "such", "same",
    "more", "most", "less", "least", "many", "much", "very", "also", "however",
    "thus", "hence", "then", "than", "while", "which", "who", "what", "when",
    "where", "how", "why", "both", "each", "all", "any", "some", "no", "not",
    "only", "other", "others", "between", "within", "without", "via", "per",
    "with", "from", "into", "over", "under", "about", "above", "below", "if",
    "so", "because", "since", "though", "although", "whether", "either", "neither",
}


def _norm_key(text: str) -> str:
    """Whitespace/case-normalised matching key (same tokenisation both sides)."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _gazetteer_keep(key: str, count: int) -> bool:
    """Quality gate for a gazetteer entry. Multi-word entries are kept (full);
    single-token entries must be distinctive — not a function word, seen >=2x,
    >=3 chars — to avoid matching common words in every paper."""
    if len(key) < 3:
        return False
    if " " in key:            # multi-word → keep (the bulk, rarely noise)
        return True
    if key in _FUNCTION_WORDS:  # single common word → drop
        return False
    return count >= 2          # single token must recur (kills annotation noise)


def scierc_to_ceo(stype: str):
    """ITER/SciERC type → CEO type (None drops the entity)."""
    return SCIERC_TO_CEO.get(stype, "Other")


def llm_to_ceo(ltype: str):
    """Free-LLM type → CEO type (falls back to the raw label/Other)."""
    return LLM_TO_CEO.get(ltype, ltype or "Other")


# "Hard" CEO ontology entity types — real, ontology-typed entities the fixed
# extractor can build proper relations from. Anything outside this (Concept,
# Other, Generic, empty) is a vague fragment that pushes the extractor into the
# loose "addresses → Concept" catch-all, so it's dropped from the body harvest.
HARD_CEO_TYPES = {
    "Model", "Method", "Dataset", "Tool", "EvaluationMetric", "Metric",
    "Task", "Resource", "Language",
}

_ABBREV_RE = re.compile(r'^(.*?)\s*\(([A-Za-z0-9\-]{2,10})\)\s*$')
_STOPWORDS = {
    "the", "this", "that", "these", "those", "it", "we", "our", "their", "they",
    "a", "an", "model", "method", "approach", "task", "system", "data", "result",
    "results", "paper", "work", "table", "figure",
}
# Leading words that signal a descriptive phrase, not a named entity
# (e.g. "the task", "new model", "present paper", "proposed approach").
_GENERIC_LEADING = {
    "the", "a", "an", "this", "that", "these", "those", "our", "their", "its",
    "new", "present", "proposed", "current", "existing", "various", "several",
    "many", "most", "some", "each", "any", "other", "such", "same", "given",
    "concept", "idea", "notion", "kind", "number", "amount",
}


def _detect_abbreviation(text: str):
    """Split 'Full Name (ABBR)' → ('Full Name', 'ABBR'); else (text, '')."""
    m = _ABBREV_RE.match(text)
    return (m.group(1).strip(), m.group(2).strip()) if m else (text, "")


def _clean(text: str) -> str:
    return text.strip().strip("()[]{}\"'`:,;.!?").strip()


def _is_noise(name: str, max_words: int | None = None) -> bool:
    low = name.lower()
    words = low.split()
    if len(name) < 3:
        return True
    if low in _STOPWORDS:
        return True
    if re.fullmatch(r'[\d\W]+', name):          # all digits/punctuation
        return True
    if words and words[0] in _GENERIC_LEADING:  # "the task", "new model", ...
        return True
    if max_words and len(words) > max_words:    # over-long descriptive phrase
        return True
    return False


def load_seed_entities(csv_path: Path) -> list[dict]:
    """Load the existing (title) ACL CSV rows as seeds, if present."""
    if not csv_path.exists():
        return []
    seeds = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ent = (row.get("Entity") or "").strip()
            if ent:
                seeds.append({
                    "Entity":       ent,
                    "Abbreviation": (row.get("Abbreviation") or "").strip(),
                    "Aliases":      (row.get("Aliases") or "").strip(),
                    "TP":           1,
                    "NER_Type":     (row.get("NER_Type") or "").strip(),
                    "CEO_Type":     (row.get("CEO_Type") or "").strip(),
                })
    return seeds


def html_for(paper_id: str) -> Path | None:
    p = OUTPUT_DIR / paper_id / "no-llm" / "output.html"
    return p if p.exists() else None


def extract_body_entities_iter(html_path: Path, extractor) -> Counter:
    """Run ITER over the paper body; return Counter of (name, scierc_type) → count."""
    from kg_extraction import parse_html, split_into_sentences
    _title, sections = parse_html(str(html_path))
    sentences = []
    for sec in sections:
        sentences.extend(split_into_sentences(sec["text"]))
    counts: Counter = Counter()
    if not sentences:
        return counts
    for ent in extractor.extract_entities(sentences):
        name = _clean(ent["text"])
        if _is_noise(name):
            continue
        counts[(name, ent["type"])] += 1
    return counts


def extract_body_entities_llm(paper_id: str, llm_model: str) -> Counter:
    """Harvest subjects+objects from open-extraction triples.json → Counter."""
    tpath = OUTPUT_DIR / paper_id / "kg" / "llm" / llm_model / "triples.json"
    counts: Counter = Counter()
    if not tpath.exists():
        return counts
    data = json.loads(tpath.read_text(encoding="utf-8"))
    triples = data["triples"] if isinstance(data, dict) and "triples" in data else data
    for tr in triples:
        for name_key, type_key in (("subject", "subject_type"), ("object", "object_type")):
            name = _clean(str(tr.get(name_key, "")))
            if not name or _is_noise(name):
                continue
            counts[(name, str(tr.get(type_key, "") or ""))] += 1
    return counts


def _parse_csner_entities(text: str):
    """Yield (surface, CSNER_TYPE) from BIOES-tagged 'token\\tTAG' CS-NER text."""
    cur, curtype = [], None
    for line in text.splitlines():
        s = line.rstrip("\n")
        if not s.strip():
            if cur:
                yield " ".join(cur), curtype
            cur, curtype = [], None
            continue
        parts = s.split("\t")
        if len(parts) < 2:
            continue
        tok, tag = parts[0], parts[-1]
        prefix, _, etype = tag.partition("-")
        if prefix == "S":
            if cur:
                yield " ".join(cur), curtype
            cur, curtype = [], None
            yield tok, etype
        elif prefix == "B":
            if cur:
                yield " ".join(cur), curtype
            cur, curtype = [tok], etype
        elif prefix in ("I", "E") and cur:
            cur.append(tok)
            if prefix == "E":
                yield " ".join(cur), curtype
                cur, curtype = [], None
        else:
            if cur:
                yield " ".join(cur), curtype
            cur, curtype = [], None
    if cur:
        yield " ".join(cur), curtype


def build_gazetteer(dest: Path = CSNER_GAZETTEER_CSV) -> int:
    """Download all CS-NER files, aggregate into one global entity gazetteer,
    write to `dest`. Returns the number of unique entities. Built once."""
    import urllib.parse
    import urllib.request
    # key → (canonical surface, Counter of CEO types)
    agg: dict[str, tuple] = {}
    for rel in CSNER_FILES:
        url = f"{CSNER_REPO_BASE}/{urllib.parse.quote(rel)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "enrich/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"  gazetteer: failed to fetch {rel} ({e}) — skipping")
            continue
        n = 0
        for surface, ctype in _parse_csner_entities(text):
            ceo = CSNER_TO_CEO.get((ctype or "").upper())
            if not ceo:
                continue
            surface = _clean(surface)
            key = _norm_key(surface)
            if not key or len(key) < 3:
                continue
            if key not in agg:
                agg[key] = (surface, Counter())
            agg[key][1][ceo] += 1
            n += 1
        logger.info(f"  gazetteer: {rel} → {n} entity mentions")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Entity", "CEO_Type", "Count"])
        kept = 0
        for key, (surface, types) in sorted(agg.items()):
            count = sum(types.values())
            if not _gazetteer_keep(key, count):
                continue
            ceo, _ = types.most_common(1)[0]
            w.writerow([surface, ceo, count])
            kept += 1
    logger.info(f"Gazetteer built: {kept} kept / {len(agg)} raw unique entities → {dest}")
    return kept


def load_gazetteer(path: Path = CSNER_GAZETTEER_CSV) -> dict:
    """Load gazetteer CSV → {normalised_key: (surface, CEO_Type)}."""
    gaz: dict[str, tuple] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            surface = (row.get("Entity") or "").strip()
            key = _norm_key(surface)
            if key:
                gaz[key] = (surface, (row.get("CEO_Type") or "").strip())
    return gaz


def _paper_text(paper_id: str) -> str:
    """Title + all section text of a paper (for gazetteer matching)."""
    html_path = html_for(paper_id)
    if html_path is None:
        return ""
    from kg_extraction import parse_html
    title, sections = parse_html(str(html_path))
    return " ".join([title or ""] + [s.get("text", "") for s in sections])


def extract_body_entities_csner(paper_id: str, gazetteer: dict) -> Counter:
    """Per-paper intersection: keep gazetteer entities that appear in the paper
    text. Returns Counter{(surface, CEO_Type): occurrences}."""
    counts: Counter = Counter()
    tokens = _TOKEN_RE.findall(_paper_text(paper_id).lower())
    n = len(tokens)
    for i in range(n):
        for L in range(1, CSNER_MAX_NGRAM + 1):
            if i + L > n:
                break
            hit = gazetteer.get(" ".join(tokens[i:i + L]))
            if hit:
                counts[hit] += 1   # hit = (surface, CEO_Type)
    return counts


def build_rows(seeds: list[dict], body_counts: Counter, type_to_ceo,
               min_count: int, max_entities: int | None,
               hard_types_only: bool = True, max_words: int | None = 6) -> list[dict]:
    """Merge title seeds + body entities into deduped CSV rows (seeds win).

    type_to_ceo     : callable raw_type -> CEO type (or None to drop the entity).
    hard_types_only : keep only body entities whose CEO type is a real ontology
                      type (HARD_CEO_TYPES); drop Concept/Other/Generic fragments.
    max_words       : drop body entities longer than this many words.
    Seeds (title entities) are exempt from both filters — always kept.
    """
    rows_by_key: dict[str, dict] = {}
    counts_by_key: dict[str, int] = defaultdict(int)

    # Seeds first — always kept, authoritative for type/abbrev
    for s in seeds:
        key = s["Entity"].lower()
        rows_by_key[key] = s
        counts_by_key[key] = 10 ** 6   # force seeds to the top, never filtered

    # Body entities: collapse the same name across SciERC types to its most common
    name_types: dict[str, Counter] = defaultdict(Counter)
    name_total: Counter = Counter()
    for (name, stype), c in body_counts.items():
        name_types[name][stype] += c
        name_total[name] += c

    for name, total in name_total.items():
        key = name.lower()
        if key in rows_by_key:
            counts_by_key[key] = max(counts_by_key[key], total)
            continue   # seed already present; keep its type
        if total < min_count:
            continue
        if _is_noise(name, max_words):        # generic-leading / over-long phrase
            continue
        stype = name_types[name].most_common(1)[0][0]
        ceo = type_to_ceo(stype)
        if ceo is None:                       # e.g. SciERC "Generic" → skip
            continue
        if hard_types_only and ceo not in HARD_CEO_TYPES:  # drop Concept/Other
            continue
        name_clean, abbrev = _detect_abbreviation(name)
        rows_by_key[key] = {
            "Entity":       name_clean,
            "Abbreviation": abbrev,
            "Aliases":      "",
            "TP":           1,
            "NER_Type":     stype,
            "CEO_Type":     ceo,
        }
        counts_by_key[key] = total

    # Sort: seeds + highest-frequency entities first
    ordered = sorted(rows_by_key.values(),
                     key=lambda r: counts_by_key[r["Entity"].lower()], reverse=True)
    if max_entities:
        ordered = ordered[:max_entities]
    return ordered


def enrich_paper(paper_id: str, source: str, extractor, llm_model: str, out_dir: Path,
                 min_count: int, max_entities: int | None, in_place: bool,
                 hard_types_only: bool = True, max_words: int | None = 6,
                 gazetteer: dict | None = None) -> bool:
    if source == "csner":
        if html_for(paper_id) is None:
            logger.warning(f"  {paper_id}: no output.html — skipping")
            return False
        body_counts = extract_body_entities_csner(paper_id, gazetteer or {})
        type_to_ceo = lambda t: t   # gazetteer entries are already CEO-typed
        if not body_counts:
            logger.warning(f"  {paper_id}: no CS-NER gazetteer entities found in text — skipping")
            return False
    elif source == "iter":
        html_path = html_for(paper_id)
        if html_path is None:
            logger.warning(f"  {paper_id}: no output.html — skipping")
            return False
        body_counts = extract_body_entities_iter(html_path, extractor)
        type_to_ceo = scierc_to_ceo
    else:  # llm
        body_counts = extract_body_entities_llm(paper_id, llm_model)
        type_to_ceo = llm_to_ceo
        if not body_counts:
            logger.warning(f"  {paper_id}: no open-extraction triples ({llm_model}) — skipping")
            return False

    seed_csv = out_dir / paper_id / f"Entity_{paper_id}.csv"
    seeds = load_seed_entities(seed_csv)

    rows = build_rows(seeds, body_counts, type_to_ceo, min_count, max_entities,
                      hard_types_only=hard_types_only, max_words=max_words)
    if not rows:
        logger.warning(f"  {paper_id}: no entities after enrichment — skipping")
        return False

    if in_place:
        dest = seed_csv
    else:
        dest = out_dir / paper_id / f"Entity_{paper_id}_enriched.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  {paper_id}: {len(seeds)} title seeds + body → "
                f"{len(rows)} entities → {dest}")
    return True


def discover_paper_ids(out_dir: Path, source: str, llm_model: str) -> list[str]:
    """Papers with an available entity source.

    For html-based sources (csner/iter) this is ANY parsed paper
    (output/<id>/no-llm/output.html) — not just those with a legacy
    output/acl/<id>/ CSV folder — so papers fetched via fetch_more_papers.py
    (PDF-only, no acl/<id>/ dir) are also enriched. The enriched CSV is written
    under output/acl/<id>/ (created on write).
    """
    cand = set()
    # Legacy: papers that already have an ACL CSV folder.
    if out_dir.exists():
        for d in out_dir.iterdir():
            if d.is_dir() and d.name != "pdfs":
                cand.add(d.name)
    # html-based sources: also any parsed paper anywhere under output/.
    if source in ("iter", "csner"):
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and d.name != "acl" and (d / "no-llm" / "output.html").exists():
                cand.add(d.name)

    ids = []
    for pid in sorted(cand):
        if source in ("iter", "csner") and html_for(pid) is None:
            continue
        if source == "llm" and not (OUTPUT_DIR / pid / "kg" / "llm" /
                                    llm_model / "triples.json").exists():
            continue
        ids.append(pid)
    return ids


def main():
    ap = argparse.ArgumentParser(
        description="Enrich title-only ACL entity CSVs with body entities (ITER/SciERC) "
                    "for the fixed extractor.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--paper", metavar="PAPER_ID", help="Single paper id (e.g. D17-1028)")
    g.add_argument("--all", action="store_true", help="All papers with an ACL CSV + HTML")
    ap.add_argument("--source", choices=["csner", "llm", "iter"], default="csner",
                    help="Entity source: 'csner' (default) = intersect the global CS-NER "
                         "gazetteer with the paper text (non-circular, external annotation); "
                         "'llm' = harvest our own open-extraction triples (CIRCULAR, retired); "
                         "'iter' = ITER/SciERC model over HTML")
    ap.add_argument("--rebuild-gazetteer", action="store_true",
                    help="Force re-download + rebuild of the CS-NER gazetteer cache "
                         f"({CSNER_GAZETTEER_CSV})")
    ap.add_argument("--llm-model", default="Qwen3-14B",
                    help="LLM model subdir to read triples from when --source llm "
                         "(default: Qwen3-14B)")
    ap.add_argument("--out-dir", default=str(ACL_DIR),
                    help="Directory holding the ACL entity CSVs (default: output/acl)")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Keep body entities seen at least this many times (default: 1). "
                         "Keep at 1 for per-paper enrichment — a single paper rarely "
                         "repeats entities, so 2+ starves the fixed extractor's subject pool.")
    ap.add_argument("--max-entities", type=int, default=None,
                    help="Cap total entities per paper (seeds + top-frequency body)")
    ap.add_argument("--keep-all-types", action="store_true",
                    help="Keep body entities of ANY type (default: only real CEO ontology "
                         "types — Model/Method/Dataset/Tool/Metric/Task/Resource/Language — "
                         "dropping vague Concept/Other fragments). Title seeds always kept.")
    ap.add_argument("--max-words", type=int, default=6,
                    help="Drop body entities longer than this many words (default: 6; "
                         "0 = no limit). Title seeds exempt.")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite Entity_<id>.csv (default: write Entity_<id>_enriched.csv)")
    ap.add_argument("--no-gpu", action="store_true", help="Force ITER onto CPU (--source iter)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    paper_ids = ([args.paper] if args.paper
                 else discover_paper_ids(out_dir, args.source, args.llm_model))
    if not paper_ids:
        logger.error("No papers to enrich (need an ACL CSV folder + entity source).")
        return

    extractor = None
    if args.source == "iter":
        from kg_extraction import ITERExtractor
        logger.info("Loading ITER/SciERC extractor...")
        extractor = ITERExtractor(device="cpu" if args.no_gpu else None)

    gazetteer = None
    if args.source == "csner":
        if args.rebuild_gazetteer or not CSNER_GAZETTEER_CSV.exists():
            logger.info("Building CS-NER gazetteer (one-time download + aggregate)...")
            build_gazetteer()
        gazetteer = load_gazetteer()
        logger.info(f"Loaded CS-NER gazetteer: {len(gazetteer)} entities "
                    f"({CSNER_GAZETTEER_CSV})")

    max_words = args.max_words if args.max_words and args.max_words > 0 else None
    logger.info(f"Enriching {len(paper_ids)} paper(s) (source: {args.source}, "
                f"hard-types-only: {not args.keep_all_types}, max-words: {max_words})...")
    ok = 0
    for pid in paper_ids:
        if enrich_paper(pid, args.source, extractor, args.llm_model, out_dir,
                        args.min_count, args.max_entities, args.in_place,
                        hard_types_only=not args.keep_all_types, max_words=max_words,
                        gazetteer=gazetteer):
            ok += 1
    logger.info(f"Done — enriched {ok}/{len(paper_ids)} papers.")


if __name__ == "__main__":
    main()
