"""
iter_extractor.py
-----------------
Science-specific triple extraction using ITER (EMNLP 2024) trained on SciERC.

ITER is a modern replacement for DYGIE++ — trained on the same SciERC dataset
but actively maintained, pip-installable, and achieves better results.

Model: fleonce/iter-scierc-deberta-large
  - Typed entities: Method, Task, Metric, Material, Generic, OtherScientificTerm
  - Typed relations: Used-for, Feature-of, Hyponym-of, Part-of, Compare, Conjunction

Install:
    pip install git+https://github.com/fleonce/iter

Paper: https://aclanthology.org/2024.findings-emnlp.655/
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# SciERC relation and entity type reference
SCIERC_ENTITY_TYPES = {
    "Method", "Task", "Metric", "Material", "Generic", "OtherScientificTerm"
}

SCIERC_RELATION_TYPES = {
    "Used-for",      # X is used for Y
    "Feature-of",    # X is a feature of Y
    "Hyponym-of",    # X is a hyponym/subtype of Y
    "Part-of",       # X is part of Y
    "Compare",       # X is compared with Y
    "Conjunction",   # X and Y are mentioned together
}

# Default model — deberta-large trained on SciERC
DEFAULT_MODEL = "fleonce/iter-scierc-deberta-large"


class ITERExtractor:
    """
    Wraps the ITER model (EMNLP 2024) for science-specific triple extraction.

    Unlike REBEL which extracts open-ended relations, ITER extracts typed
    entities and typed relations specifically designed for scientific papers.

    Compatible interface with TripleExtractor and LLMExtractor —
    same extract_from_sentences() method, same output format.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or DEFAULT_MODEL
        self._device_override = device
        self._model = None
        self._tokenizer = None

    @property
    def device(self):
        try:
            import torch
            return self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return self._device_override or "cpu"

    def _load(self):
        """Lazy load — only on first use."""
        if self._model is not None:
            return

        try:
            from iter import ITERModel
        except ImportError:
            raise ImportError(
                "ITER package not installed.\n"
                "Install with: pip install git+https://github.com/fleonce/iter"
            )

        logger.info(f"Loading ITER model ({self.model_name}) on {self.device}...")
        self._model = ITERModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._tokenizer = self._model.tokenizer
        logger.info("ITER model loaded.")

    def extract_from_sentences(
        self,
        sentences: list[str],
        source_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        Run ITER extraction on a list of sentences.
        Returns triples in the same format as TripleExtractor and LLMExtractor.

        ITER works best sentence-by-sentence since SciERC is sentence-level.
        """
        self._load()

        all_triples = []
        for sentence in sentences:
            try:
                triples = self._extract_sentence(sentence, source_meta or {})
                all_triples.extend(triples)
            except Exception as e:
                logger.warning(f"ITER failed on sentence: {e}")
                continue

        return all_triples

    def extract_entities(self, sentences: list[str]) -> list[dict]:
        """
        Run ITER over sentences and return ALL detected entities (not just those
        in a relation). Used to build a subject pool for the fixed extractor.

        Returns a list of {"text": str, "type": str} (SciERC entity type).
        """
        import torch
        self._load()

        out = []
        for sentence in sentences:
            try:
                encodings = self._tokenizer(
                    sentence, return_tensors="pt", truncation=True, max_length=512,
                )
                encodings = {k: v.to(self.device) for k, v in encodings.items()}
                with torch.no_grad():
                    output = self._model.generate(
                        encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                    )
                for ent in (output.entities or []):
                    text = str(ent.text) if hasattr(ent, "text") else str(ent)
                    etype = str(ent.type) if hasattr(ent, "type") else "Unknown"
                    if text.strip():
                        out.append({"text": text.strip(), "type": etype})
            except Exception as e:
                logger.warning(f"ITER entity extraction failed on sentence: {e}")
                continue
        return out

    def _extract_sentence(self, sentence: str, meta: dict) -> list[dict]:
        """Run ITER on a single sentence and convert output to triple dicts."""
        import torch

        encodings = self._tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            output = self._model.generate(
                encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )

        triples = []

        # output.links contains relation pairs between detected entities
        # output.entities contains all detected entities with types
        entities = output.entities   # list of entity objects with .text and .type
        links    = output.links      # list of link objects with .source, .target, .type

        if not links:
            return triples

        # Build entity index for quick lookup
        entity_map = {}
        if entities:
            for i, ent in enumerate(entities):
                entity_map[i] = {
                    "text": str(ent.text) if hasattr(ent, "text") else str(ent),
                    "type": str(ent.type) if hasattr(ent, "type") else "Unknown",
                }

        for link in links:
            try:
                # ITER link object has .source (entity index), .target (entity index), .type (relation)
                src_idx  = link.source if hasattr(link, "source") else None
                tgt_idx  = link.target if hasattr(link, "target") else None
                rel_type = str(link.type) if hasattr(link, "type") else str(link)

                if src_idx is None or tgt_idx is None:
                    continue

                src_ent = entity_map.get(src_idx, {})
                tgt_ent = entity_map.get(tgt_idx, {})

                subject      = src_ent.get("text", "")
                subject_type = src_ent.get("type", "Unknown")
                obj          = tgt_ent.get("text", "")
                obj_type     = tgt_ent.get("type", "Unknown")

                if not subject or not obj or not rel_type:
                    continue

                triple = {
                    "subject":        subject,
                    "subject_type":   subject_type,
                    "predicate":      rel_type,
                    "object":         obj,
                    "object_type":    obj_type,
                    "source_sentence": sentence,
                }
                triple.update(meta)
                triples.append(triple)

            except Exception as e:
                logger.debug(f"Skipping link due to error: {e}")
                continue

        return triples

    def get_all_entities(self, sentences: list[str]) -> list[dict]:
        """
        Extract all named entities from sentences (without requiring relations).
        Useful for inspecting what ITER detects in the text.
        """
        self._load()
        all_entities = []

        for sentence in sentences:
            try:
                import torch
                encodings = self._tokenizer(
                    sentence, return_tensors="pt",
                    truncation=True, max_length=512
                )
                encodings = {k: v.to(self.device) for k, v in encodings.items()}

                with torch.no_grad():
                    output = self._model.generate(
                        encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                    )

                if output.entities:
                    for ent in output.entities:
                        all_entities.append({
                            "text":     str(ent.text) if hasattr(ent, "text") else str(ent),
                            "type":     str(ent.type) if hasattr(ent, "type") else "Unknown",
                            "sentence": sentence,
                        })
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")

        return all_entities

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("ITER model unloaded from memory.")