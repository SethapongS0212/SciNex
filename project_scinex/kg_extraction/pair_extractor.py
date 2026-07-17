"""
pair_extractor.py
-----------------
Closed-entity / open-relation triple extraction.

Contrast with the other LLM extractors:
  - free  (llm_extractor)   : subject free,            relation free,      object free
  - fixed (fixed_extractor) : subject ∈ entity list,   relation ∈ ontology, object free
  - pair  (this module)     : subject ∈ entity list,   relation FREE,       object ∈ entity list

This extractor fixes BOTH the subject and the object to the entity list, and lets
the LLM choose a short free-form relation describing how the two listed entities
relate in the text:

    subject ∈ entity list   --(free relation)-->   object ∈ entity list

It reuses FixedTripleExtractor's model loading / paragraph scaffolding and only
overrides the prompt and the output parser.
"""

import logging
from typing import Optional

from .entity_loader import EntitySet
from .fixed_extractor import (
    FixedTripleExtractor,
    _strip_think_and_fences,
    _extract_json_object,
    _subject_in_sentence,
)

logger = logging.getLogger(__name__)

# Relations are free-form but should stay short/verb-like, not full sentences.
MAX_RELATION_WORDS = 5


def _make_pair_system_prompt() -> str:
    return """You are a scientific knowledge extraction expert.

You will be given a list of fixed entities found in a passage of a scientific paper.
Extract relationship triples (subject, relation, object) where:

SUBJECT: must be one of the fixed entities (exact canonical name from the list).
OBJECT:  must ALSO be one of the fixed entities (exact canonical name from the list).
  - BOTH the subject and the object MUST come from the fixed entity list.
  - NEVER use an entity that is not in the list, for either the subject or the object.
RELATION: a short free-form phrase (1-4 words, a lowercase verb phrase) describing how
  the subject relates to the object AS STATED in the text — e.g. "uses", "is based on",
  "outperforms", "is trained on", "is part of", "extends", "is evaluated on".
  - You are NOT restricted to a fixed relation vocabulary — choose the phrase that best
    fits the sentence.

Rules:
1. BOTH subject and object must be from the fixed entity list — no exceptions.
2. The relation must be EXPLICITLY stated in a single source sentence — do not infer.
3. Subject and object must be DIFFERENT entities.
4. BOTH the subject and the object must be NAMED in the source_sentence (by full name or
   a known abbreviation). If only one of the two appears in the sentence, SKIP the triple.
5. Keep the relation short and verb-like; never a full sentence.
6. The source_sentence must be a real verbatim sentence from the text (not a heading/URL).
7. Return ONLY valid JSON, no explanation, no markdown.

If no sentence relates two listed entities, return {"triples": []}.

Return format:
{
  "triples": [
    {
      "subject": "fixed entity name exactly as given in the list",
      "relation": "short free-form relation phrase",
      "object": "another fixed entity name exactly as given in the list",
      "source_sentence": "verbatim sentence from text supporting this triple"
    }
  ]
}

If no valid triples can be extracted, return: {"triples": []}"""


def _make_pair_user_prompt(text: str, section: str,
                           present_entities: list[str],
                           abbreviations: dict[str, str]) -> str:
    lines = []
    for e in present_entities:
        abbr = abbreviations.get(e, "")
        lines.append(f"  - {e}  [also appears in text as: {abbr}]" if abbr else f"  - {e}")
    entity_str = "\n".join(lines)
    return f"""Section: {section}

Fixed entities present in this text — use ONLY these as BOTH subjects AND objects,
written exactly as listed:
{entity_str}

Text:
{text}

Extract (subject, relation, object) triples where BOTH the subject and the object are
from the list above, and the relation is a short free-form phrase you choose.
Only include triples EXPLICITLY stated in a single sentence — not implied or inferred."""


def _parse_pair_output(raw: str, entity_set: EntitySet) -> list[dict]:
    """Parse LLM output; keep only triples whose subject AND object are listed entities."""
    raw = _strip_think_and_fences(raw)
    data = _extract_json_object(raw)
    if not data:
        return []

    out = []
    for t in data.get("triples", []):
        subj = str(t.get("subject", "")).strip()
        obj  = str(t.get("object", "")).strip()
        # accept either "relation" (this extractor) or "predicate" (defensive)
        rel  = str(t.get("relation", "") or t.get("predicate", "")).strip()
        src  = str(t.get("source_sentence", "")).strip()

        if not subj or not obj or not rel:
            continue

        # BOTH ends must resolve to a canonical entity in the list
        canon_subj = entity_set.match(subj)
        canon_obj  = entity_set.match(obj)
        if not canon_subj or not canon_obj:
            continue
        if canon_subj == canon_obj:
            continue

        # BOTH ends must actually be named in the source sentence
        if src and (not _subject_in_sentence(canon_subj, src, entity_set)
                    or not _subject_in_sentence(canon_obj, src, entity_set)):
            continue

        # Normalise the free relation: lowercase, collapse spaces, keep it short
        rel_clean = " ".join(rel.lower().split())
        if not rel_clean or len(rel_clean.split()) > MAX_RELATION_WORDS:
            continue

        out.append({
            "subject":         canon_subj,
            "predicate":       rel_clean,
            "object":          canon_obj,
            "object_type":     "Entity",   # object is always a listed entity here
            "source_sentence": src,
        })
    return out


class EntityPairExtractor(FixedTripleExtractor):
    """
    Subject ∈ entity list, object ∈ entity list, relation = free-form (LLM chooses).

    Reuses FixedTripleExtractor's model loading, paragraph splitting, present-entity
    detection and extract_from_sentences() loop; overrides only the prompt + parser.
    """

    EXTRACTION_MODE = "pair"

    def __init__(
        self,
        entity_set: EntitySet,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 2048,
        paragraph_char_limit: int = 1500,
    ):
        super().__init__(
            entity_set=entity_set,
            relations=None,                 # no fixed relation vocabulary
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            paragraph_char_limit=paragraph_char_limit,
        )
        # Replace the ontology system prompt with the open-relation pair prompt
        self._system_prompt = _make_pair_system_prompt()
        logger.info(
            f"EntityPairExtractor initialized — {len(entity_set)} entities "
            f"(fixed subject+object, free relation), model: {self.model_name}"
        )

    def _extract_paragraph(self, text: str, section: str,
                           present_entities: list[str]) -> list[dict]:
        # A relation needs two listed entities in the paragraph; skip if <2 present
        if len(present_entities) < 2:
            logger.debug("Fewer than 2 fixed entities in paragraph — skipping (pair).")
            return []

        user_content = "/no_think\n\n" + _make_pair_user_prompt(
            text, section, present_entities, self.entity_set.abbreviations
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]
        try:
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            outputs = self._pipeline(prompt)
            raw = outputs[0]["generated_text"].strip()
            logger.warning(f"[RAW LLM OUTPUT] {raw[:500]!r}")
            result = _parse_pair_output(raw, self.entity_set)
            logger.warning(f"[PARSED] {len(result)} pair-triples from this paragraph")
            return result
        except Exception as e:
            logger.warning(f"Pair extraction failed for paragraph: {e}")
            return []
