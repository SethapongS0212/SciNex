"""
relation_extractor.py
---------------------
Relation-only extraction: the RELATION is fixed to the ontology, the SUBJECT is free.

This is the fallback for papers where no usable entity list can be assembled — the
CS-NER gazetteer is annotated over CS/NLP papers, so a chemistry, materials or clinical
paper can intersect it down to almost nothing, and `fixed` extraction then produces
near-zero triples for reasons that have nothing to do with extraction quality.

Constraint ladder across the extractors:
    llm       subject free      relation free        object free
    relation  subject free      relation ∈ ontology  object free   ← this file
    fixed     subject ∈ CSV     relation ∈ ontology  object free
    pair      subject ∈ CSV     relation free        object ∈ CSV

Everything except the subject constraint is shared with FixedTripleExtractor — same
ontology, same prompt rules, same post-parse guards — so `fixed` vs `relation` isolates
exactly one variable: what constraining the subject to a curated list buys you.

Output: output/<paper>/kg/relation/<model>/triples.json  (or relation_scinex/ for
the scinex ontology), mirroring the fixed/fixed_scinex split.
"""

import logging
from typing import Optional

from .fixed_extractor import (
    FixedTripleExtractor,
    _is_garbled_section,
    _make_fixed_system_prompt,
    _parse_fixed_output,
)

logger = logging.getLogger(__name__)


_FREE_SUBJECT_BLOCK = """SUBJECTS: Subjects are NOT fixed — use any specific, named entity from the text.
  - The subject must be a concrete named thing: a model, method, dataset, tool, metric,
    material, task or system (e.g. "Random Forest", "RT-DETR", "oxide scale").
  - NEVER use a generic phrase as the subject: "the model", "the system", "this approach",
    "the proposed method", "we", "the authors", "the study" — skip those triples entirely.
  - The subject must appear LITERALLY in the source sentence, written the same way."""

_FREE_SUBJECT_RULE = ("Only extract triples whose subject is a specific named entity that "
                      "appears literally in the source sentence")

_FREE_SUBJECT_FIELD = "specific named entity, written exactly as it appears in the text"


def _make_relation_user_prompt(text: str, section: str) -> str:
    return f"""Section: {section}

Text:
{text}

Extract triples using the allowed relations. Subjects may be any specific named entity
from the text, but must be written exactly as they appear there.
Only include triples that are EXPLICITLY stated — not implied or inferred."""


class RelationOnlyExtractor(FixedTripleExtractor):
    """Fixed relations, free subjects. Same guards as the fixed extractor."""

    EXTRACTION_MODE = "relation"

    def __init__(
        self,
        relations: Optional[list[str]] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 2048,
        paragraph_char_limit: int = 1500,
        schema: Optional[dict] = None,
    ):
        # entity_set=None is the switch that frees the subject, both here and in
        # _parse_fixed_output's validation path.
        super().__init__(
            entity_set=None,
            relations=relations,
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            paragraph_char_limit=paragraph_char_limit,
            schema=schema,
        )
        self._subject_prompt = {
            "subject_block": _FREE_SUBJECT_BLOCK,
            "subject_rule":  _FREE_SUBJECT_RULE,
            "subject_field": _FREE_SUBJECT_FIELD,
        }
        self._system_prompt = _make_fixed_system_prompt(
            self.relations, self.schema, **self._subject_prompt
        )
        logger.info(
            f"RelationOnlyExtractor initialized — free subjects, "
            f"{len(self.relations)} relations, model: {self.model_name}"
        )

    def extract_from_sentences(
        self,
        sentences: list[str],
        source_meta: Optional[dict] = None,
    ) -> list[dict]:
        """Every paragraph is a candidate — there is no entity list to filter on."""
        self._load()

        section = (source_meta or {}).get("section", "")
        if _is_garbled_section(section):
            logger.debug(f"Skipping garbled section: {section!r}")
            return []

        all_triples = []
        for para in self._sentences_to_paragraphs(sentences):
            for t in self._extract_paragraph(para, section, []):
                if not t.get("source_sentence"):
                    t["source_sentence"] = para[:500]
                t["extraction_mode"] = self.EXTRACTION_MODE
                if source_meta:
                    t.update({k: v for k, v in source_meta.items() if k not in t})
                all_triples.append(t)

        return all_triples

    def _extract_paragraph(
        self,
        text: str,
        section: str,
        present_entities: list[str],
    ) -> list[dict]:
        """One LLM call per paragraph, with no fixed-subject list in the prompt."""
        user_content = "/no_think\n\n" + _make_relation_user_prompt(text, section)
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

            raw = self._pipeline(prompt)[0]["generated_text"].strip()
            logger.warning(f"[RAW LLM OUTPUT] {raw[:500]!r}")

            result = _parse_fixed_output(raw, None, self.relations)
            logger.warning(f"[PARSED] {len(result)} triples from this paragraph")
            return result

        except Exception as e:
            logger.warning(f"Relation-only extraction failed for paragraph: {e}")
            return []
