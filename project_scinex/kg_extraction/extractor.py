"""
extractor.py
------------
Triple extraction using REBEL (Babelscape/rebel-large).
REBEL is a seq2seq model trained end-to-end for relation extraction.
It outputs (subject, predicate, object) triples directly from raw text.

Model card: https://huggingface.co/Babelscape/rebel-large
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_rebel_output(text: str) -> list[dict]:
    """
    Parse REBEL's special output format into (subject, predicate, object) triples.

    REBEL output format:
        <triplet> subject <subj> object <rel> relation <triplet> ...

    Returns list of {"subject": ..., "predicate": ..., "object": ...}
    """
    triples = []
    # Split on <triplet> token
    chunks = re.split(r'<triplet>', text)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Match: subject <subj> object <rel> relation
        match = re.match(
            r'^(.+?)\s*<subj>\s*(.+?)\s*<obj>\s*(.+?)(?:\s*<triplet>|$)',
            chunk,
            re.DOTALL
        )
        if match:
            subject   = match.group(1).strip()
            obj       = match.group(2).strip()
            predicate = match.group(3).strip()

            # Strip leftover special tokens from decoding
            for tok in ['</s>', '<pad>', '<s>']:
                subject   = subject.replace(tok, '').strip()
                predicate = predicate.replace(tok, '').strip()
                obj       = obj.replace(tok, '').strip()
            # Strip any trailing <pad>+ patterns
            import re as _re
            subject   = _re.sub(r'(<pad>)+.*', '', subject).strip()
            predicate = _re.sub(r'(<pad>)+.*', '', predicate).strip()
            obj       = _re.sub(r'(<pad>)+.*', '', obj).strip()

            # Basic quality filters — skip math noise (single chars, pure symbols)
            def _is_valid(s):
                return (s and len(s) >= 2 and len(s) < 100
                        and not _re.fullmatch(r'[^a-zA-Z]+', s))

            if _is_valid(subject) and _is_valid(predicate) and _is_valid(obj):
                triples.append({
                    "subject":   subject,
                    "predicate": predicate,
                    "object":    obj
                })

    return triples


class TripleExtractor:
    """
    Wraps REBEL for triple extraction.
    Loads model once and runs inference sentence-by-sentence (or in batches).
    """

    MODEL_NAME = "Babelscape/rebel-large"

    def __init__(self, device: Optional[str] = None, batch_size: int = 8):
        self.batch_size = batch_size
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
        import torch
        if self._model is not None:
            return

        logger.info(f"Loading REBEL model ({self.MODEL_NAME}) on {self.device}...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
        self._model.to(self.device)
        self._model.eval()
        logger.info("REBEL model loaded.")

    def extract_from_sentences(
        self,
        sentences: list[str],
        source_meta: Optional[dict] = None
    ) -> list[dict]:
        """
        Run triple extraction on a list of sentences.
        Returns list of triple dicts with source metadata attached.

        Args:
            sentences:   List of sentence strings.
            source_meta: Dict to attach to each triple (e.g. {"paper": ..., "section": ...})
        """
        self._load()
        all_triples = []

        # Process in batches
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i: i + self.batch_size]
            try:
                triples = self._run_batch(batch)
                for triple in triples:
                    if source_meta:
                        triple.update(source_meta)
                    all_triples.append(triple)
            except Exception as e:
                logger.warning(f"Batch {i//self.batch_size} failed: {e}")
                continue

        return all_triples

    def _run_batch(self, sentences: list[str]) -> list[dict]:
        import torch
        inputs = self._tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_length=512,
                num_beams=3,
                early_stopping=True
            )

        decoded = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )

        all_triples = []
        for i, text in enumerate(decoded):
            triples = _parse_rebel_output(text)
            # Attach source sentence
            for t in triples:
                t["source_sentence"] = sentences[i]
            all_triples.extend(triples)

        return all_triples

    def unload(self):
        """Free GPU memory after processing."""
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
            logger.info("REBEL model unloaded from memory.")