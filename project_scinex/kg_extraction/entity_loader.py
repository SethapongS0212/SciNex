"""
entity_loader.py
----------------
Loads entity CSVs from your colleague's work and extracts the fixed entity set.

Only loads entities where TP=1 (entity actually appears in the paper).
Builds a complete lookup including abbreviations and aliases for matching.

CSV format expected:
    Entity, Abbreviation, Aliases, LLM Extraction, TP, FN, FP, TN, ...
"""

import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class EntitySet:
    """
    Holds the fixed entity set for one paper.

    Attributes:
        entities       : list of canonical entity names (TP=1 only)
        abbreviations  : dict {canonical → abbreviation}
        aliases        : dict {canonical → [alias1, alias2, ...]}
        lookup         : dict {any form → canonical} for matching
    """

    def __init__(
        self,
        entities: list[str],
        abbreviations: dict[str, str],
        aliases: dict[str, list[str]],
    ):
        self.entities      = entities
        self.abbreviations = abbreviations
        self.aliases       = aliases

        # Build reverse lookup: any form → canonical name
        self.lookup: dict[str, str] = {}
        for canonical in entities:
            # canonical itself
            self.lookup[canonical.lower().strip()] = canonical
            # abbreviation
            abbr = abbreviations.get(canonical, "").strip()
            if abbr:
                self.lookup[abbr.lower()] = canonical
            # aliases
            for alias in aliases.get(canonical, []):
                alias = alias.strip()
                if alias:
                    self.lookup[alias.lower()] = canonical

        # ── Paper-specific surface form expansion ────────────────────────────
        # Some papers use surface forms not captured by the CSV abbreviation/alias
        # columns.  We add them here so both the extractor (entity matching) and
        # the evaluator (alias injection) resolve them correctly.
        #
        # Pattern: for each entity that has an abbreviation, also register common
        # prefixed / suffixed variants that appear in scientific papers:
        #   "OpenAI GPT"  →  "Generative Pre-trained Transformer"
        #   "BERT_BASE", "BERT-Large", "BERT_LARGE" → "Bidirectional Encoder…"
        #   "BERT base", "BERT large"               → "Bidirectional Encoder…"
        _extra: dict[str, str] = {}
        for canonical, abbr in abbreviations.items():
            abbr_lower = abbr.lower()
            # Vendor-prefixed variants: "OpenAI GPT", "Google BERT", etc.
            for prefix in ("openai", "google", "meta", "microsoft", "deepmind",
                           "huggingface", "stanford", "berkeley"):
                _extra[f"{prefix} {abbr_lower}"] = canonical
            # Size-suffixed variants: "BERT_BASE", "BERT-large", "BERT base", etc.
            # No empty separator — avoids noise like "gptxl", "bertlarge"
            for size in ("base", "large", "small", "tiny", "medium", "xl", "xxl"):
                for sep in ("_", "-", " "):
                    _extra[f"{abbr_lower}{sep}{size}"] = canonical
                    _extra[f"{abbr_lower.replace(' ', '')}{sep}{size}"] = canonical
            # All-caps abbr variants
            _extra[abbr.upper().lower()] = canonical

        # Only add if not already in lookup (CSV-defined forms win)
        for form, canonical in _extra.items():
            if form not in self.lookup:
                self.lookup[form] = canonical

    def match(self, text: str) -> str | None:
        """
        Try to match a text string to a canonical entity name.
        Returns canonical name if found, None otherwise.
        """
        return self.lookup.get(text.lower().strip())

    @property
    def all_aliases(self) -> dict[str, str]:
        """
        Return a dict mapping every known surface form → canonical name.
        This is the full lookup table including abbreviations, aliases, and
        paper-specific expansions like "OpenAI GPT".
        Used by kg_evaluate to pass alias context to the LLM judge.
        """
        return dict(self.lookup)

    def match_any_in_text(self, text: str) -> list[str]:
        """
        Find all canonical entities mentioned in a text string.
        Used to check which entities appear in a sentence.
        """
        text_lower = text.lower()
        found = []
        for form, canonical in self.lookup.items():
            # Whole word match to avoid partial matches (e.g. "F1" matching "F1 score")
            pattern = r'\b' + re.escape(form) + r'\b'
            if re.search(pattern, text_lower):
                if canonical not in found:
                    found.append(canonical)
        return found

    def __len__(self):
        return len(self.entities)

    def __repr__(self):
        return f"EntitySet({len(self.entities)} entities)"

    def summary(self) -> str:
        lines = [f"Fixed Entity Set — {len(self.entities)} entities (TP=1)"]
        for e in self.entities[:10]:
            abbr    = self.abbreviations.get(e, "")
            aliases = self.aliases.get(e, [])
            parts   = []
            if abbr:
                parts.append(f"abbr: {abbr}")
            if aliases:
                parts.append(f"aliases: {', '.join(aliases)}")
            suffix = f" ({'; '.join(parts)})" if parts else ""
            lines.append(f"  • {e}{suffix}")
        if len(self.entities) > 10:
            lines.append(f"  ... and {len(self.entities) - 10} more")
        return "\n".join(lines)


def load_entity_csv(csv_path: str | Path, tp_only: bool = True) -> EntitySet:
    """
    Load entity CSV and return an EntitySet.

    Args:
        csv_path : path to the entity CSV file
        tp_only  : if True, only load entities where TP=1 (default: True)

    Returns:
        EntitySet with canonical names, abbreviations, aliases, and lookup table
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Entity CSV not found: {csv_path}")

    entities      = []
    abbreviations = {}
    aliases_map   = {}
    skipped       = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            canonical = row.get("Entity", "").strip()

            # Skip empty entity rows
            if not canonical:
                skipped += 1
                continue

            # Check TP filter
            if tp_only:
                tp_val = row.get("TP", "0").strip()
                try:
                    if int(tp_val) != 1:
                        continue
                except ValueError:
                    continue

            entities.append(canonical)

            # Abbreviation
            abbr = row.get("Abbreviation", "").strip()
            if abbr:
                abbreviations[canonical] = abbr

            # Aliases — comma separated
            alias_str = row.get("Aliases", "").strip()
            if alias_str:
                alias_list = [a.strip() for a in alias_str.split(",") if a.strip()]
                if alias_list:
                    aliases_map[canonical] = alias_list

    logger.info(
        f"Loaded {len(entities)} entities (TP=1) from {csv_path.name} "
        f"[{skipped} rows skipped — empty entity name]"
    )

    if not entities:
        logger.warning(
            f"No entities found in {csv_path.name}. "
            f"Check that TP column exists and has value 1."
        )

    return EntitySet(
        entities=entities,
        abbreviations=abbreviations,
        aliases=aliases_map,
    )


def load_entity_csvs(csv_paths: list[str | Path], tp_only: bool = True) -> EntitySet:
    """
    Load and merge multiple entity CSVs into one EntitySet.
    Useful if a paper has entities split across multiple files.
    """
    all_entities      = []
    all_abbreviations = {}
    all_aliases       = {}

    for path in csv_paths:
        es = load_entity_csv(path, tp_only=tp_only)
        for e in es.entities:
            if e not in all_entities:
                all_entities.append(e)
        all_abbreviations.update(es.abbreviations)
        all_aliases.update(es.aliases)

    logger.info(f"Merged {len(all_entities)} unique entities from {len(csv_paths)} CSV files.")
    return EntitySet(
        entities=all_entities,
        abbreviations=all_abbreviations,
        aliases=all_aliases,
    )