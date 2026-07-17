"""
acl_index.py
------------
Builds and queries a local title → ACL paper ID index.

Sources tried in order (most to least reliable for a given environment):
  1. anthology.json.gz   from aclanthology.org      (fast, one file, ~10 MB)
  2. XML files           from raw.githubusercontent.com  (guaranteed accessible)

Index is saved to acl_title_index.json and reused on subsequent runs.

Usage:
    # Build the index (run once)
    python acl_index.py --build --index acl_title_index.json

    # Look up a single title (for testing)
    python acl_index.py --lookup "PUT at SemEval-2016 Task 4: The ABC of Twitter Sentiment Analysis"

    # Use from code
    from acl_index import ACLIndex
    idx = ACLIndex()          # loads or builds automatically
    paper_id = idx.lookup("Attention is All You Need")   # → "P17-1037"
"""

import gzip
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("acl_title_index.json")

# ── Title normalisation (for fuzzy matching) ──────────────────────────────────

def _norm(title: str) -> str:
    """Normalise a title for comparison: lowercase, strip punctuation, collapse spaces."""
    title = title.lower()
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()


def _sim(a: str, b: str) -> float:
    """Similarity ratio between two normalised titles."""
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ── Source 1: anthology.json.gz from aclanthology.org ────────────────────────

def _build_from_anthology_json(index: dict) -> int:
    """
    Download anthology.json.gz from aclanthology.org and parse into index.
    Returns number of entries added.
    Entry format: {"id": "P18-1001", "title": "..."}
    """
    url = "https://aclanthology.org/anthology.json.gz"
    logger.info(f"Trying anthology.json.gz from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "acl_pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        logger.info(f"  Downloaded {len(raw)//1024} KB, parsing ...")
        with gzip.open(io.BytesIO(raw)) as gz:
            data = json.loads(gz.read())
        added = 0
        for paper_id, meta in data.items():
            title = meta.get("title", "").strip()
            if title and paper_id:
                index[_norm(title)] = paper_id
                added += 1
        logger.info(f"  Added {added} entries from anthology.json.gz")
        return added
    except urllib.error.HTTPError as e:
        logger.warning(f"  anthology.json.gz: HTTP {e.code} — falling back to XML")
        return 0
    except Exception as e:
        logger.warning(f"  anthology.json.gz failed: {e} — falling back to XML")
        return 0


# ── Source 2: XML files from raw.githubusercontent.com ───────────────────────

RAW_BASE = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml"


def _xml_file_names() -> list[str]:
    """
    Generate the full list of ACL Anthology XML file names.
    Covers old-style (letter+2digit year) and new-style (year.venue) naming.
    """
    names = []

    # ── Old-style: single-letter prefix + 2-digit year ────────────────────────
    # Each letter covers a venue or family of venues:
    #   P=ACL, D=EMNLP, N=NAACL, E=EACL, C=COLING, S=SemEval/*SEM,
    #   K=CoNLL, Q=TACL, J=CL Journal, W=Workshops,
    #   A=Asian NLP, F=Japanese NLP, I=IJCNLP, L=LREC,
    #   M=MUC, O=ROCLING, R=RANLP, T=SIGdial/TIPSTER, U=ALTA, Y=Oriental
    venues_old = list("ABCDEFIJKLMNOPQRSTUWXY")
    for letter in venues_old:
        for year in range(0, 25):          # 00 → 24
            names.append(f"{letter}{year:02d}.xml")

    # ── New-style: year.venue (2020 onwards) ──────────────────────────────────
    # Main conference tracks
    main_venues_new = [
        "acl", "emnlp", "naacl", "eacl", "coling", "conll", "tacl",
        "cl", "lrec", "findings",
        # Workshop series
        "ws", "semeval", "starsem", "sigmorphon", "wmt",
        "blackboxnlp", "clinicalnlp", "computel", "deeplo",
        "eval4nlp", "fever", "gem", "insights", "lantern",
        "mia", "nllp", "privatenlp", "repl4nlp", "scil",
        "socialnlp", "splu-robonlp", "sustainlp", "trac",
        "umios", "winlp",
    ]
    # Track suffixes for main conferences
    tracks = ["long", "short", "srw", "demos", "findings", "industry",
              "tutorials", "system-demonstrations", "main"]

    for year in range(2020, 2026):
        for venue in main_venues_new:
            names.append(f"{year}.{venue}.xml")
        # Main conference tracks
        for conf in ["acl", "emnlp", "naacl", "eacl", "coling"]:
            for track in tracks:
                names.append(f"{year}.{conf}-{track}.xml")
        # SemEval, *SEM
        names.append(f"{year}.semeval-1.xml")
        names.append(f"{year}.starsem-1.xml")

    return names


def _parse_xml_to_index(xml_bytes: bytes, collection_id: str, index: dict) -> int:
    """Parse one ACL Anthology XML file into the index. Returns entries added."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.debug(f"XML parse error in {collection_id}: {e}")
        return 0

    added = 0
    for volume in root.findall("volume"):
        vol_id = volume.attrib.get("id", "1")
        for paper in volume.findall("paper"):
            paper_num = paper.attrib.get("id", "")
            title_el  = paper.find("title")
            if title_el is None:
                continue
            title = "".join(title_el.itertext()).strip()
            if not title or not paper_num:
                continue

            # Form the paper ID
            try:
                num = int(paper_num)
                if "." in collection_id:  # new-style: 2020.pam → 2020.pam-1.10
                    full_id = f"{collection_id}-{vol_id}.{paper_num}"
                else:
                    # Old-style: always 4 digits total after the dash
                    # e.g. vol=1 paper=18 → 1018; vol=31 paper=5 → 3105
                    paper_digits = max(1, 4 - len(str(vol_id)))
                    full_id = f"{collection_id}-{vol_id}{num:0{paper_digits}d}"
            except ValueError:
                continue

            index[_norm(title)] = full_id
            added += 1

    return added


def _build_from_xml(index: dict, delay: float = 0.2) -> int:
    """
    Download ACL Anthology XML files from raw.githubusercontent.com,
    parse them, and populate the index.
    Returns total number of entries added.
    """
    file_names = _xml_file_names()
    total_added = 0
    total_files = 0
    failed = 0

    logger.info(f"Building index from XML ({len(file_names)} candidate files) ...")

    for fname in file_names:
        # Derive collection ID from filename (strip .xml)
        collection_id = fname[:-4]
        url = f"{RAW_BASE}/{fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "acl_pipeline/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                xml_bytes = resp.read()
            added = _parse_xml_to_index(xml_bytes, collection_id, index)
            if added > 0:
                total_added += added
                total_files += 1
                logger.debug(f"  {fname}: {added} papers")
            time.sleep(delay)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                logger.debug(f"  {fname}: HTTP {e.code}")
            failed += 1
        except Exception as e:
            logger.debug(f"  {fname}: {e}")
            failed += 1

        # Progress every 50 files
        n = file_names.index(fname) + 1
        if n % 50 == 0:
            logger.info(f"  Progress: {n}/{len(file_names)} files, "
                        f"{total_files} with data, {total_added} entries so far")

    logger.info(f"XML done: {total_files} files parsed, {total_added} entries added, "
                f"{failed} files missing (404)")
    return total_added


# ── Index class ───────────────────────────────────────────────────────────────

class ACLIndex:
    """
    Local title → ACL paper ID lookup with fuzzy matching.

    On first use, builds the index (takes a few minutes).
    Subsequent uses load from the cached JSON file (instant).
    """

    def __init__(self, index_path: str | Path = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self._index: dict[str, str] = {}   # normalised_title → paper_id
        self._loaded = False

    # ── Loading / building ────────────────────────────────────────────────────

    def load_or_build(self) -> "ACLIndex":
        """Load from cache or build from scratch if not available."""
        if self.index_path.exists():
            self._load()
        else:
            self.build()
        return self

    def _load(self):
        logger.info(f"Loading index from {self.index_path} ...")
        with open(self.index_path, encoding="utf-8") as f:
            self._index = json.load(f)
        self._loaded = True
        logger.info(f"  Loaded {len(self._index)} entries")

    def build(self, save: bool = True):
        """
        Build index from scratch.
        Tries anthology.json.gz first (fast), falls back to XML files.
        """
        logger.info("Building ACL title index ...")
        self._index = {}

        added = _build_from_anthology_json(self._index)
        if added == 0:
            logger.info("Falling back to XML files from GitHub ...")
            added = _build_from_xml(self._index)

        logger.info(f"Index built: {len(self._index)} entries total")
        self._loaded = True

        if save:
            self.save()

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False)
        logger.info(f"Index saved → {self.index_path} ({len(self._index)} entries)")

    # ── Lookup ────────────────────────────────────────────────────────────────

    def lookup(self, title: str, threshold: float = 0.85) -> str | None:
        """
        Look up a paper by title with fuzzy matching.

        Args:
            title:     Paper title (exact or approximate)
            threshold: Minimum similarity ratio (0–1) for a match. 0.85 is strict.

        Returns:
            ACL paper ID string (e.g. "P18-1001") or None if not found.
        """
        if not self._loaded:
            self.load_or_build()

        norm = _norm(title)

        # 1. Exact match
        if norm in self._index:
            return self._index[norm]

        # 2. Fuzzy match — scan all keys and find best similarity
        best_id    = None
        best_score = 0.0
        for key, pid in self._index.items():
            score = _sim(norm, key)
            if score > best_score:
                best_score = score
                best_id    = pid

        if best_score >= threshold:
            logger.debug(f"Fuzzy match ({best_score:.2f}): '{title[:50]}' → {best_id}")
            return best_id

        logger.debug(f"No match ({best_score:.2f}): '{title[:50]}'")
        return None

    def __len__(self):
        return len(self._index)