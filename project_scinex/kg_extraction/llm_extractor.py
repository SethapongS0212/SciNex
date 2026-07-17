"""
llm_extractor.py
----------------
LLM-based triple extraction AND knowledge graph construction using Qwen3-32B.

Unlike REBEL which is a narrow relation extraction model, the LLM:
  - Understands scientific context deeply
  - Returns typed entities (Method, Dataset, Metric, Model, Task, etc.)
  - Returns meaningful scientific relations (outperforms, trained_on, evaluated_on, etc.)
  - Normalizes entity names (BERT == Bidirectional Encoder Representations → same node)
  - Works on full paragraphs, not just sentences (more context = better triples)

Model: Qwen/Qwen3-32B (4-bit quantized via bitsandbytes)
Fits on: Single A100 80GB

Output triple format:
  {
    "subject":        "BERT",
    "subject_type":   "Model",
    "predicate":      "outperforms",
    "object":         "ELMo",
    "object_type":    "Model",
    "paper":          "1810.04805",
    "section":        "Experiments",
    "source_text":    "BERT outperforms ELMo on all GLUE benchmarks..."
  }
"""

import json
import logging
import re
from typing import Optional

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# expandable_segments uses CUDA virtual-memory APIs that vGPUs (e.g. H100-20C)
# do not support -> "CUDA driver error: operation not supported" on any alloc.

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scientific knowledge extraction expert. Your task is to extract structured knowledge triples from scientific paper text.

Extract (subject, predicate, object) triples where:
- Subjects and objects are meaningful scientific entities (models, datasets, methods, metrics, tasks, concepts)
- Predicates are precise scientific relations

Entity types to use:
- Model: neural network architectures, pre-trained models (BERT, GPT, ResNet, etc.)
- Method: algorithms, techniques, approaches (attention mechanism, dropout, fine-tuning, etc.)
- Dataset: training/evaluation datasets (ImageNet, SQuAD, GLUE, etc.)
- Metric: evaluation metrics (accuracy, F1, BLEU, perplexity, etc.)
- Task: NLP/ML tasks (classification, translation, question answering, etc.)
- Framework: software frameworks (PyTorch, TensorFlow, etc.)
- Concept: theoretical concepts (overfitting, gradient descent, etc.)
- Other: anything else important

Relation types to use (prefer these but use others if more accurate):
- outperforms, underperforms, comparable_to (performance comparisons)
- trained_on, fine_tuned_on, evaluated_on (dataset relations)
- uses, extends, replaces, improves_upon (method relations)
- achieves, reports, measures (metric relations)
- part_of, component_of, based_on (structural relations)
- proposed_by, introduced_by (origin relations)
- instance_of, type_of (taxonomy relations)

Rules:
1. Only extract triples that are explicitly stated or strongly implied in the text
2. Normalize entity names: use the most common/canonical form — NO underscores, use spaces
3. Skip trivial triples (e.g. "paper has section", "this paper proposes")
4. NEVER extract math expressions, equations, or formulas as entities (e.g. "xk+1 = xk − η∇F", "dv2 + ||u||2")
5. NEVER extract citation keys as entities (e.g. "[GM21]", "[CFKM20]")
6. NEVER extract theorem/lemma/proposition labels as entities (e.g. "Theorem 4.1", "Lemma C.2")
7. NEVER extract vague phrases as entities (e.g. "this paper", "the authors", "other schemes", "various variants")
8. NEVER extract single letters or short abbreviations as standalone entities (e.g. "n", "t", "GD" alone)
9. Entity names must be human-readable, use spaces not underscores, max 6 words
10. Return ONLY valid JSON, no explanation, no markdown
11. For each triple, include the EXACT sentence from the text that supports it in source_sentence
12. source_sentence must be copied verbatim from the text — do not paraphrase

Return format:
{
  "triples": [
    {
      "subject": "entity name",
      "subject_type": "entity type",
      "predicate": "relation",
      "object": "entity name",
      "object_type": "entity type",
      "source_sentence": "exact sentence from the text that supports this triple"
    }
  ],
  "normalized_entities": {
    "alias": "canonical_name"
  }
}

If no meaningful triples can be extracted, return: {"triples": [], "normalized_entities": {}}"""


def _make_user_prompt(text: str, section: str) -> str:
    return f"""Section: {section}

Text:
{text}

Extract knowledge triples from this text."""


# ── Post-processing prompt ────────────────────────────────────────────────────

POSTPROCESS_SYSTEM_PROMPT = """You are a scientific knowledge graph expert. You will be given a raw list of (subject, predicate, object) triples extracted from a scientific paper.

Your job is to clean and improve this triple list for knowledge graph construction:

1. DEDUPLICATE — remove triples that express the same fact with different wording
2. NORMALIZE ENTITIES — merge entities that refer to the same thing:
   - \"BERT\", \"bert-base\", \"Bidirectional Encoder Representations\" → \"BERT\"
   - Always keep the most common/canonical form
3. NORMALIZE RELATIONS — standardize relation names:
   - \"is used for\", \"used for\", \"used-for\" → \"used_for\"
   - Use snake_case for all relations
4. FIX ENTITY TYPES — correct wrong entity types if obvious
5. REMOVE NOISE — remove triples about math symbols, variables, or non-scientific entities
6. KEEP METADATA — preserve section, paper, source_text fields from original triples

Return ONLY valid JSON, no explanation, no markdown:
{
  "triples": [...cleaned triple list...],
  "entity_aliases": {"alias": "canonical_name"},
  "removed_count": 0,
  "merged_entities": ["list of merge decisions made"]
}"""


def _make_postprocess_prompt(triples: list[dict]) -> str:
    triple_json = json.dumps(triples, indent=2, ensure_ascii=False)
    return f"""Here are {len(triples)} triples extracted from a scientific paper.

{triple_json}

Clean and improve this triple list for knowledge graph construction."""


# ── Parser ────────────────────────────────────────────────────────────────────

def _strip_think_and_fences(raw: str) -> str:
    """Strip Qwen3 think blocks (closed and unclosed) and markdown fences."""
    raw = raw.strip()
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    # Critical: handle unclosed think block when token limit is hit mid-think
    raw = re.sub(r'<think>.*', '', raw, flags=re.DOTALL)
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def _extract_json_object(raw: str) -> dict | None:
    """
    Robustly extract outermost JSON object using brace-counting.
    Avoids the regex r'\\{.*\\}' approach which can fail on very long outputs
    (catastrophic backtracking) or on strings containing bracket characters.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw[start:i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    start = None
    return None


def _parse_llm_output(raw: str) -> tuple[list[dict], dict]:
    """
    Parse the LLM's JSON response into triples and entity normalizations.
    Handles Qwen3 <think> blocks, markdown fences, and prefix/suffix text.
    """
    raw = _strip_think_and_fences(raw)
    if not raw:
        logger.debug("LLM output empty after stripping think/fences")
        return [], {}

    data = _extract_json_object(raw)
    if data is None:
        logger.debug(f"No JSON found in LLM output: {raw[:200]}")
        return [], {}

    triples = data.get("triples", [])
    normalizations = data.get("normalized_entities", {})

    # Validate triple structure
    valid_triples = []
    required_fields = {"subject", "predicate", "object"}
    for t in triples:
        if not isinstance(t, dict):
            continue
        if not required_fields.issubset(t.keys()):
            continue
        # Skip empty or trivial
        subj = str(t["subject"]).strip()
        pred = str(t["predicate"]).strip()
        obj  = str(t["object"]).strip()
        if not subj or not pred or not obj:
            continue
        if len(subj) < 2 or len(obj) < 2:
            continue

        valid_triples.append({
            "subject":         subj,
            "subject_type":    str(t.get("subject_type", "Other")).strip(),
            "predicate":       pred,
            "object":          obj,
            "object_type":     str(t.get("object_type", "Other")).strip(),
            "source_sentence": str(t.get("source_sentence", "")).strip(),
        })

    return valid_triples, normalizations


# ── Extractor class ───────────────────────────────────────────────────────────

class LLMExtractor:
    """
    LLM-based triple extractor using Qwen3-32B (4-bit quantized).

    Processes paragraphs (not sentences) — more context = better extractions.
    Accumulates entity normalizations across the whole paper for consistent naming.
    """

    MODEL_NAME = "Qwen/Qwen3-14B"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
        paragraph_char_limit: int = 1500,
    ):
        self.model_name         = model_name or self.MODEL_NAME
        self._device_override   = device
        self.max_new_tokens     = max_new_tokens
        self.paragraph_char_limit = paragraph_char_limit

        self._model     = None
        self._tokenizer = None
        self._pipeline  = None

        # Global entity normalization map built up across the paper
        self._entity_map: dict[str, str] = {}

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if self._pipeline is not None:
            return

        logger.info(f"Loading {self.model_name} (4-bit quantized)...")

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},         # force entire model onto GPU 0 (no silent CPU offload)
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,            # greedy — deterministic, better for structured output
            temperature=None,
            top_p=None,
            return_full_text=False,     # only return generated part, not the prompt
        )

        self._tokenizer = tokenizer
        logger.info(f"{self.model_name} loaded.")

    # ── Public interface ──────────────────────────────────────────────────────

    def extract_from_sentences(
        self,
        sentences: list[str],
        source_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        Compatible interface with REBEL extractor.
        Internally joins sentences into paragraphs for better LLM context.
        """
        self._load()

        # Join sentences into paragraphs (LLMs work better with more context)
        paragraphs = self._sentences_to_paragraphs(sentences)
        section    = (source_meta or {}).get("section", "")

        all_triples = []
        for para in paragraphs:
            triples, norms = self._extract_paragraph(para, section)

            # Update global entity normalization map
            self._entity_map.update(norms)

            # Apply normalizations already known
            triples = self._apply_normalization(triples)

            # Attach metadata
            for t in triples:
                # Only fall back to paragraph if LLM did not provide a source sentence
                if not t.get("source_sentence"):
                    t["source_sentence"] = para[:500]
                if source_meta:
                    t.update({k: v for k, v in source_meta.items() if k not in t})
                all_triples.append(t)

        return all_triples

    def get_entity_map(self) -> dict:
        """Return the accumulated entity normalization map for this paper."""
        return dict(self._entity_map)

    def reset_entity_map(self):
        """Call between papers to clear accumulated normalizations."""
        self._entity_map = {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sentences_to_paragraphs(self, sentences: list[str]) -> list[str]:
        """
        Group sentences into paragraphs under the char limit.
        Larger chunks = richer context for the LLM.
        """
        paragraphs = []
        current = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current and current_len + sent_len > self.paragraph_char_limit:
                paragraphs.append(" ".join(current))
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len

        if current:
            paragraphs.append(" ".join(current))

        return paragraphs

    def _extract_paragraph(self, text: str, section: str) -> tuple[list[dict], dict]:
        """Run one LLM call on a paragraph, return triples + normalizations."""
        # Qwen3 can spend the whole generation budget inside <think>; disable it
        # at the template level when supported and via /no_think as a fallback.
        user_content = "/no_think\n\n" + _make_user_prompt(text, section)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        try:
            # Apply chat template
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            outputs = self._pipeline(prompt)
            raw = outputs[0]["generated_text"].strip()
            return _parse_llm_output(raw)

        except Exception as e:
            logger.warning(f"LLM extraction failed for paragraph: {e}")
            return [], {}

    def _apply_normalization(self, triples: list[dict]) -> list[dict]:
        """Replace entity aliases with canonical names using accumulated map."""
        normalized = []
        for t in triples:
            t = dict(t)
            t["subject"] = self._entity_map.get(t["subject"].lower(), t["subject"])
            t["object"]  = self._entity_map.get(t["object"].lower(),  t["object"])
            normalized.append(t)
        return normalized

    def postprocess_triples(self, triples: list[dict]) -> list[dict]:
        """
        Step 2 — LLM post-processing of the full triple list.
        One LLM call per paper that deduplicates, normalizes entities,
        standardizes relations, and removes noise across all extracted triples.

        Call this after all sections have been processed via extract_from_sentences().
        """
        if not triples:
            return triples

        self._load()

        # LLM context limit — chunk if too many triples
        # At ~60 tokens per triple, 300 triples = ~18k tokens, safe for most models
        MAX_TRIPLES_PER_CALL = 200

        if len(triples) <= MAX_TRIPLES_PER_CALL:
            return self._postprocess_chunk(triples)

        # For large papers: process in chunks, then do a final merge pass
        logger.info(f"Postprocessing {len(triples)} triples in chunks...")
        chunks = [
            triples[i:i + MAX_TRIPLES_PER_CALL]
            for i in range(0, len(triples), MAX_TRIPLES_PER_CALL)
        ]
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Postprocessing chunk {i+1}/{len(chunks)} ({len(chunk)} triples)...")
            cleaned = self._postprocess_chunk(chunk)
            cleaned_chunks.extend(cleaned)

        # Final pass to deduplicate across chunks
        if len(chunks) > 1:
            logger.info("  Final deduplication pass across chunks...")
            cleaned_chunks = self._postprocess_chunk(cleaned_chunks)

        return cleaned_chunks

    def _postprocess_chunk(self, triples: list[dict]) -> list[dict]:
        """Run one LLM postprocessing call on a chunk of triples."""
        messages = [
            {"role": "system", "content": POSTPROCESS_SYSTEM_PROMPT},
            {"role": "user",   "content": _make_postprocess_prompt(triples)},
        ]

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Postprocessing may need more tokens for large triple lists
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens * 2,
            )
            raw = outputs[0]["generated_text"].strip()
            raw = _strip_think_and_fences(raw)

            data = _extract_json_object(raw)
            if data is None:
                logger.warning("Postprocessing: no JSON found, returning original triples.")
                return triples

            cleaned = data.get("triples", triples)

            # Log what the LLM did
            removed  = data.get("removed_count", len(triples) - len(cleaned))
            merged   = data.get("merged_entities", [])
            aliases  = data.get("entity_aliases", {})

            logger.info(f"  Postprocessing: {len(triples)} → {len(cleaned)} triples")
            if removed:
                logger.info(f"  Removed {removed} duplicate/noisy triples")
            if merged:
                logger.info(f"  Merged entities: {merged[:3]}{'...' if len(merged) > 3 else ''}")
            if aliases:
                self._entity_map.update(aliases)

            # Validate cleaned triples have required fields
            valid = []
            for t in cleaned:
                if isinstance(t, dict) and "subject" in t and "predicate" in t and "object" in t:
                    valid.append(t)
            return valid

        except Exception as e:
            logger.warning(f"Postprocessing failed: {e}. Returning original triples.")
            return triples

    def unload(self):
        """Free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            del self._model
            del self._tokenizer
            self._pipeline  = None
            self._model     = None
            self._tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info(f"{self.model_name} unloaded from memory.")
