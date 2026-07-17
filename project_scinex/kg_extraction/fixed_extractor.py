"""
fixed_extractor.py
------------------
LLM-based triple extraction using FIXED subjects (from entity CSV) and
FIXED relations (from ontology — placeholder until colleague provides his).

For each paragraph:
  1. Detect which fixed entities appear in the text
  2. Ask LLM: given ONLY these subjects and ONLY these relations,
     what objects do they connect to?

This produces the "fixed" version of triples for comparison against
the "free" version (where LLM chooses subjects/relations freely).

Model: Qwen3-32B (4-bit quantized) — change via --model flag
"""

import json
import logging
import re
from typing import Optional

from .entity_loader import EntitySet

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# expandable_segments uses CUDA virtual-memory APIs that vGPUs (e.g. H100-20C)
# do not support -> "CUDA driver error: operation not supported" on any alloc.

logger = logging.getLogger(__name__)


# ── Placeholder relation ontology ─────────────────────────────────────────────
# Will be replaced with your colleague's ontology once received.
# These are sensible scientific relations that cover most paper content.

# ── Core Experiment Ontology (CEO) v0.7 ─────────────────────────────────────
# Source: https://github.com/wpatipon/core-experiment-ontology
CEO_RELATIONS = [
    "cites",            # AcademicPaper → AcademicPaper
    "publishedIn",      # AcademicPaper → PublicationVenue
    "writtenBy",        # AcademicPaper → Author
    "reports",          # AcademicPaper → ResearchProcess
    "affiliatedWith",   # Agent → Organisation
    "employs",          # Organisation → Person
    "locatedIn",        # Organisation → Place
    "addresses",        # Paper/Process → ResearchContext
    "motivates",        # ResearchObjective → ResearchProcess
    "achieves",         # ResearchProcess → ResearchObjective
    "encompasses",      # ResearchDomain → ResearchTask
    "comprises",        # ResearchProcess → sub-ResearchProcess
    "uses",             # ResearchProcess → ResearchArtifact (consumed)
    "produces",         # ResearchProcess → ResearchArtifact (created)
    "trainedOn",        # Model → TrainingDataset
    "evaluatedOn",      # Experiment → Dataset
    "splitFrom",        # Dataset → Dataset (partition)
    "designedFor",      # Model → ResearchTask
    "comparesAgainst",  # Model → Model (benchmarking)
    "configures",       # ExperimentalSpecification → Experiment
    "evaluates",        # EvaluationMetric → ExperimentalResult
    "supports",         # ExperimentalMeasurement → ExperimentalOutcome
]
PLACEHOLDER_RELATIONS = CEO_RELATIONS  # backward-compat alias


# ── Prompts ───────────────────────────────────────────────────────────────────

_CEO_SCHEMA = {
    # Publication relationships
    "cites":           "AcademicPaper → AcademicPaper  |  use when a paper cites another paper by name",
    "publishedIn":     "AcademicPaper → PublicationVenue  |  use for journal/conference names only",
    "writtenBy":       "AcademicPaper → Author  |  use only for explicit author attribution",
    "reports":         "AcademicPaper → ResearchProcess  |  subject MUST be the paper itself (rare in fixed extraction — only use when the sentence is about what the paper as a whole describes). Do NOT use with Model or Method as subject — use achieves instead",

    # Agent/org relationships
    "affiliatedWith":  "Agent → Organisation  |  use for institutional affiliation",
    "employs":         "Organisation → Person  |  use for employment relationships",
    "locatedIn":       "Organisation → Place  |  use for physical location",

    # Research context relationships
    "addresses":       "ResearchProcess/Paper → ResearchContext  |  subject must be a Paper, Model, or Method (never a ResearchTask). Object is a domain/task/objective the subject targets. STRICT: Do NOT use when a task is the subject — tasks ARE ResearchContext, they cannot address themselves",
    "motivates":       "ResearchObjective → ResearchProcess  |  use when a stated problem or gap drives a new method",
    "achieves":        "ResearchProcess → ResearchObjective  |  use when a model/method attains a specific result or goal. USE THIS instead of reports when subject is a Model",
    "encompasses":     "ResearchDomain → ResearchTask  |  subject MUST be a broad domain (NLP, Computer Vision, etc.). Object is a specific task within that domain. NEVER use a ResearchTask as the subject — tasks cannot encompass domains",

    # Process relationships
    "comprises":       "ResearchProcess → ResearchProcess  |  BOTH subject and object must be processes/methods. Object must be an explicitly named sub-process or component — NOT a dataset, representation, or result",
    "uses":            "ResearchProcess → ResearchArtifact  |  use when a method/model explicitly consumes a tool, dataset, or architecture",
    "produces":        "ResearchProcess → ResearchArtifact  |  use when a process explicitly creates a new artifact",

    # Artifact relationships
    "trainedOn":       "Model → TrainingDataset  |  use only for named training corpora/datasets",
    "evaluatedOn":     "Model/Experiment → Dataset  |  use when a model or experiment is explicitly tested on a named benchmark/dataset. Subject can be Model as shorthand for the experiment involving that model",
    "splitFrom":       "Dataset → Dataset  |  use for train/dev/test partition relationships",
    "designedFor":     "Model → ResearchTask  |  use ONLY when the paper explicitly states the model was designed/built/intended for a specific task",
    "comparesAgainst": "Model → Model  |  BOTH subject and object must be named Models. Use when the paper explicitly benchmarks one model against another. Citation keys like 'Smith et al. (2019)' are NOT valid — use the model name",

    # Specification/result relationships
    "configures":      "ExperimentalSpecification → Experiment  |  use for explicit hyperparameter or setup relationships",
    "evaluates":       "EvaluationMetric → ExperimentalResult  |  subject MUST be a named metric (e.g. F1, Accuracy, BLEU, RMSE, WER). Object must be a SPECIFIC numeric result (e.g. '88.5 on the test set') — NOT a full descriptive sentence",
    "supports":        "ExperimentalMeasurement → ExperimentalOutcome  |  use when a specific number backs up a qualitative conclusion",
}


def _make_fixed_system_prompt(relations: list[str], schema: dict | None = None) -> str:
    schema = schema if schema is not None else _CEO_SCHEMA
    lines = []
    for r in relations:
        hint = schema.get(r, "")
        lines.append(f"  - {r}  ({hint})" if hint else f"  - {r}")
    relation_list = "\n".join(lines)
    return f"""You are a scientific knowledge extraction expert.

Your task is to extract knowledge triples from scientific paper text.
You MUST follow these strict constraints:

SUBJECTS: You will be given a list of fixed entities (some with abbreviations in parentheses).
  - Use ONLY these as subjects — no other entities.
  - When an entity has an abbreviation (e.g. "<Full Entity Name> (ABBR)"), use the SHORTER form as the subject if the text uses it (e.g. "ABBR"), but write it as the full canonical name exactly as listed.
  - NEVER invent subject names not in the list. NEVER paraphrase or shorten the listed name.

RELATIONS: You may ONLY use the following relation types (read the usage hints carefully):
{relation_list}

OBJECTS: Objects can be any meaningful entity mentioned in the text — they are NOT fixed.

Rules:
1. Only extract triples where the subject is from the provided fixed entity list
2. Only use relation types from the list above — do not invent new ones
3. Read the domain→range hint AND the usage description before choosing a relation
4. Objects must be explicitly mentioned and fully described in the text
5. NEVER use a URL, section title, citation key, or math expression as an object
6. NEVER extract a triple whose subject and object refer to the same thing
7. Object names must be human-readable, max 6 words, no underscores
8. The source_sentence must be a real verbatim sentence from the text (not a heading or URL)
9. If the paragraph only mentions an entity without a clear relationship, skip it — return {{"triples": []}}
10. Return ONLY valid JSON, no explanation, no markdown

CRITICAL — DO NOT INFER. Only extract what the text explicitly states:
  If the sentence says "X can be used for Y" — OK to extract
  If the sentence says "X is related to Y" — NOT specific enough, skip
  If you have to reason about what the sentence implies — skip it, return {{"triples": []}}
  The source_sentence must make the triple obvious to any reader — if it doesn't, skip it.

CRITICAL — the subject must be NAMED in the source sentence (by full name OR abbreviation):
  If the sentence describes a generic "the model", "a framework", "the system", "the method",
  or "this approach" without naming the specific entity — SKIP IT.
  BAD:  sentence says "a model is adapted for the target task"
        → do NOT extract (<Model>, designedFor, <Task>) — no specific model is named
  BAD:  sentence says "the system uses a tokenizer"
        → do NOT extract (<Model>, uses, <Method>) — which system?
  GOOD: sentence names the entity: "<Model> uses <Method>" or "we use <Method> for <Model>"
        → safe to extract

CRITICAL — reports: subject MUST be the paper itself, NOT a model or method:
  The ontology defines reports as AcademicPaper → ResearchProcess. Since your fixed entities
  are Models and Methods (not the paper itself), reports is almost never the right predicate.
  BAD:  (<Model>, reports, 93.5 score on <Dataset>) — a model is not a paper. Use achieves instead.
  GOOD: Only use reports if the sentence is explicitly about what "this paper" or "this work" describes.
  In almost all cases, replace reports with achieves: (<Model>, achieves, 93.5 score on <Dataset>)

CRITICAL — achieves is the correct predicate when a model/method attains a result:
  The ontology defines achieves as ResearchProcess → ResearchObjective (inverse of motivates).
  Use this whenever a model or method reaches a specific result, score, or goal.
  GOOD: (<Model>, achieves, 93.5 F1 on <Dataset>)
  GOOD: (<Method>, achieves, state-of-the-art on <Task>)

CRITICAL — addresses subject must be a Paper, Model, or Method — NEVER a ResearchTask:
  The ontology defines addresses as ResearchProcess/Paper → ResearchContext.
  ResearchTasks (e.g. classification, translation, recognition, synthesis) ARE ResearchContext —
  they are the OBJECT. They cannot be the subject because tasks do not address other tasks.
  BAD:  (<Task>, addresses, <another task or context>) — a task IS the research context
  BAD:  (<Model>, addresses, <Method the model uses>) — if it USES the thing, use `uses` not addresses
  GOOD: (<Method>, addresses, <a named problem/limitation>) — the method tackles a stated problem
  When in doubt, do NOT use addresses. Leave the triple out.

CRITICAL — encompasses subject MUST be a broad ResearchDomain, never a ResearchTask:
  The ontology defines encompasses as ResearchDomain → ResearchTask.
  A specific ResearchTask CANNOT be the subject — tasks are not domains.
  BAD:  (<specific task>, encompasses, <broader area>) — direction is REVERSED
  GOOD: (<broad field/domain>, encompasses, <specific task within it>)
  Ask: "Is the subject a BROAD FIELD or a specific task?" Only broad fields can be subjects.

CRITICAL — comparesAgainst requires BOTH subject and object to be named Models/systems:
  The ontology defines comparesAgainst as Model → Model.
  Citation keys like "Smith et al. (2019)" are NOT valid model names — use the actual model name.
  BAD:  (<Model>, comparesAgainst, Smith et al. (2019)) — citation key, not a model name
  BAD:  src="including <Model A>, <Model B> and <Model C>" → a list, not a comparison, return {{"triples": []}}
  BAD:  src="<Method> is also referred to as <alias>" → an alias, not a comparison
  GOOD: (<Model A>, comparesAgainst, <Model B>) — both are named models/systems
  Only use when sentence contains: compared to, unlike, outperforms, better than, versus,
  most comparable, in contrast to, differs from.

CRITICAL — comprises: BOTH subject and object must be ResearchProcesses:
  The ontology defines comprises as ResearchProcess → ResearchProcess.
  BOTH sides must be processes/methods/tasks — NOT datasets, representations, or results.
  BAD:  (<Method>, comprises, <a representation or output>) — not a process
  GOOD: (<Method>, comprises, <a named sub-process/component>) — both are processes/methods

CRITICAL — trainedOn: extract one triple per dataset, even if a sentence lists multiple:
  If the sentence says "<Model> is trained on <Dataset A> and <Dataset B>", extract TWO triples.

CRITICAL — designedFor requires explicit design intent stated in the text:
  BAD:  (<Model>, designedFor, <Task>) — "presenting results on <Task>" is NOT design intent
  GOOD: (<Model>, designedFor, <Task>) — the paper explicitly states this design goal
  Only use when text contains: "designed for", "built for", "intended for", "purpose of".

CRITICAL — evaluatedOn subject can be Model as shorthand for its experiment:
  The ontology defines evaluatedOn as Experiment → Dataset, but using Model as subject is accepted
  as shorthand for "the experiment involving this model". Named benchmark/dataset required as object.
  BAD:  (<Model>, evaluatedOn, <Task>) — too vague, not a named benchmark/dataset
  GOOD: (<Model>, evaluatedOn, <named Dataset/benchmark>) — the model is the system under test

CRITICAL — evaluates: subject must be EvaluationMetric, object must be a specific numeric result:
  BAD:  (<Metric>, evaluates, <a descriptive sentence>) — object must be a clean numeric result
  GOOD: (<Metric>, evaluates, 88.5) — metric and clean numeric result

CRITICAL — writtenBy is ONLY for the paper being read, not cited papers:
  BAD: Reference list authors — those are authors of CITED papers, not of the subject entity.
  If source sentence is a bare list of names → return {{"triples": []}}

CRITICAL — reject low-quality source sentences before extracting:
  If the source sentence is ANY of the following, return {{"triples": []}} immediately:
  - A URL or GitHub link
  - A bare author name list
  - A table or figure reference ("The results are presented in Table X")
  - A section heading or phrase under 8 words

CRITICAL — the subject must be NAMED in the source sentence:
  If the sentence says "the model", "a pre-trained model", "this approach" without naming the
  specific entity — SKIP IT. The canonical entity name or its abbreviation must appear literally.

CRITICAL — DO NOT INFER. Only extract what the text explicitly states:
  The source_sentence must make the triple obvious. If you have to reason about what is implied,
  return {{"triples": []}}

Return format:
{{
  "triples": [
    {{
      "subject": "fixed entity name exactly as given in the list",
      "predicate": "relation from the allowed list",
      "object": "entity found in text",
      "object_type": "Model | Method | Dataset | Metric | Task | Concept | Other",
      "source_sentence": "verbatim sentence from text supporting this triple"
    }}
  ]
}}

If no valid triples can be extracted, return: {{"triples": []}}"""


def _make_fixed_user_prompt(
    text: str,
    section: str,
    present_entities: list[str],
    abbreviations: dict[str, str],
) -> str:
    # Build entity display — show full name AND abbreviation clearly.
    # The model must write the full name as the subject, but should recognise
    # the abbreviation in text as referring to that entity.
    entity_lines = []
    for e in present_entities:
        abbr = abbreviations.get(e, "")
        if abbr:
            entity_lines.append(f"  - {e}  [also appears in text as: {abbr}]")
        else:
            entity_lines.append(f"  - {e}")

    entity_str = "\n".join(entity_lines)

    return f"""Section: {section}

Fixed subjects you may use (write the full name exactly as listed):
{entity_str}

Text:
{text}

Extract triples using ONLY the fixed subjects above and the allowed relations.
Only include triples that are EXPLICITLY stated — not implied or inferred."""


import re as _re

# Patterns that indicate a section is a figure caption / diagram label dump
# rather than real prose — these produce garbage triples from token sequences
# like "E[CLS] E1 E2 EN" that appear when pdfplumber captures diagram annotations.
_GARBLED_SECTION_RE = _re.compile(
    r"""
    E\[CLS\]            # BERT diagram token labels
    | E\[SEP\]
    | \bE1\s+E2\b       # sequential token placeholders
    | \bE_\d+\b         # E_1, E_2, ...
    | \[unused\d+\]     # BERT special tokens leaked into section names
    | ^figure\s+\d+$    # bare "Figure 3" section headings
    | ^fig\.\s*\d+$
    | ^table\s+\d+$
    | ^references?$             # bibliography / reference sections
    | ^bibliography$
    | ^acknowledgem             # acknowledgements
    | ^appendix\s+[a-z]?$      # appendix sections (usually tables/figures)
    """,
    _re.IGNORECASE | _re.VERBOSE,
)


def _is_garbled_section(section: str) -> bool:
    """Return True if the section name looks like a figure/diagram label dump."""
    return bool(_GARBLED_SECTION_RE.search(section))

class FixedTripleExtractor:
    """
    LLM-based extractor constrained to fixed subjects and fixed relations.

    Compatible interface with LLMExtractor — same extract_from_sentences() method.
    Accepts an EntitySet and optional relation list.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-14B"
    EXTRACTION_MODE = "fixed"   # tag written onto each triple (subclasses override)

    def __init__(
        self,
        entity_set: EntitySet,
        relations: Optional[list[str]] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 2048,
        paragraph_char_limit: int = 1500,
        schema: Optional[dict] = None,
    ):
        self.entity_set           = entity_set
        self.relations            = relations or PLACEHOLDER_RELATIONS
        # schema = relation→hint dict; defaults to the hardcoded CEO schema. A
        # different ontology (e.g. scinex from an .owl) passes its own here.
        self.schema               = schema if schema is not None else _CEO_SCHEMA
        self.model_name           = model_name or self.DEFAULT_MODEL
        self._device_override     = device
        self.max_new_tokens       = max_new_tokens
        self.paragraph_char_limit = paragraph_char_limit

        self._pipeline  = None
        self._tokenizer = None

        # System prompt is fixed per extractor instance (relations don't change)
        self._system_prompt = _make_fixed_system_prompt(self.relations, self.schema)

        logger.info(
            f"FixedTripleExtractor initialized — "
            f"{len(entity_set)} entities, {len(self.relations)} relations, "
            f"model: {self.model_name}"
        )

    @property
    def device(self):
        try:
            import torch
            return self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return self._device_override or "cpu"

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if self._pipeline is not None:
            return

        logger.info(f"Loading {self.model_name} (4-bit quantized) on {self.device}...")

        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            BitsAndBytesConfig, pipeline
        )

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
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
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
        Extract fixed triples from sentences.
        Joins sentences into paragraphs, detects which fixed entities
        are present, then prompts LLM with only those entities.
        """
        self._load()

        paragraphs = self._sentences_to_paragraphs(sentences)
        section    = (source_meta or {}).get("section", "")
        all_triples = []

        # Skip sections whose names are figure/diagram label dumps
        if _is_garbled_section(section):
            logger.debug(f"Skipping garbled section: {section!r}")
            return []

        for para in paragraphs:
            # Detect which fixed entities appear in this paragraph
            present = self.entity_set.match_any_in_text(para)

            if not present:
                # No fixed entities in this paragraph — skip
                logger.debug(f"No fixed entities in paragraph, skipping.")
                continue

            logger.debug(f"Found {len(present)} fixed entities in paragraph: {present[:3]}...")

            triples = self._extract_paragraph(para, section, present)

            for t in triples:
                if not t.get("source_sentence"):
                    t["source_sentence"] = para[:500]
                t["extraction_mode"] = self.EXTRACTION_MODE
                if source_meta:
                    t.update({k: v for k, v in source_meta.items() if k not in t})
                all_triples.append(t)

        return all_triples

    def get_relation_set(self) -> list[str]:
        """Return the current relation ontology being used."""
        return list(self.relations)

    def update_relations(self, new_relations: list[str]):
        """
        Swap in your colleague's ontology when available.
        Updates system prompt automatically.
        """
        self.relations      = new_relations
        self._system_prompt = _make_fixed_system_prompt(new_relations)
        logger.info(f"Relation ontology updated — {len(new_relations)} relations.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sentences_to_paragraphs(self, sentences: list[str]) -> list[str]:
        paragraphs  = []
        current     = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current and current_len + sent_len > self.paragraph_char_limit:
                paragraphs.append(" ".join(current))
                current     = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len

        if current:
            paragraphs.append(" ".join(current))

        return paragraphs

    def _extract_paragraph(
        self,
        text: str,
        section: str,
        present_entities: list[str],
    ) -> list[dict]:
        """Run one LLM call on a paragraph with the fixed entity constraints."""
        # Prepend /no_think so Qwen3 skips <think> blocks at model level.
        # Works regardless of transformers version (no enable_thinking= needed).
        user_content = "/no_think\n\n" + _make_fixed_user_prompt(
            text, section, present_entities, self.entity_set.abbreviations
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]

        try:
            # Try enable_thinking=False (transformers >= 4.51)
            # Falls back to /no_think prefix above on older versions
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
            raw     = outputs[0]["generated_text"].strip()

            # Always log raw output at WARNING level so you can see it
            logger.warning(f"[RAW LLM OUTPUT] {raw[:500]!r}")

            result = _parse_fixed_output(raw, self.entity_set, self.relations)
            logger.warning(f"[PARSED] {len(result)} triples from this paragraph")
            return result

        except Exception as e:
            logger.warning(f"Fixed extraction failed for paragraph: {e}")
            return []

    def unload(self):
        if self._pipeline is not None:
            del self._pipeline
            del self._tokenizer
            self._pipeline  = None
            self._tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info(f"{self.model_name} unloaded.")


# ── Output parser ─────────────────────────────────────────────────────────────

def _strip_think_and_fences(raw: str) -> str:
    """
    Strip Qwen3 <think> blocks (both closed and unclosed) and markdown fences.

    The critical case: if max_new_tokens is exhausted mid-think, the model
    never emits </think> or any JSON — the old code left the raw <think> text
    in place and the JSON regex found nothing, silently returning 0 triples.
    The second re.sub (open-ended .*) handles that truncation.
    """
    raw = raw.strip()
    # Strip closed think blocks first
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    # Strip unclosed think block — token limit hit mid-generation, no </think>
    raw = re.sub(r'<think>.*', '', raw, flags=re.DOTALL)
    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def _is_low_quality_source(src: str) -> bool:
    """
    Return True if the source sentence is too low-quality to contain a real relation.
    Filters: URLs, bare author name lists, bare table/figure references, too-short sentences.
    """
    if not src or len(src.split()) < 6:
        return True
    # URL-only sentences
    if _re.search(r'https?://|www\.|github\.com|arxiv\.org', src, _re.IGNORECASE):
        return True
    # Bare table/figure reference
    if _re.match(r'(the\s+)?(results?\s+are\s+)?(presented|shown|reported|listed)\s+in\s+(table|figure|fig\.)',
                 src, _re.IGNORECASE):
        return True
    if _re.match(r'^(see\s+)?(table|figure|fig\.)\s+\d+', src, _re.IGNORECASE):
        return True
    # Bare author name list — mostly comma-separated capitalised words, ends in a name
    words = src.split()
    cap_ratio = sum(1 for w in words if w[0].isupper() or w.rstrip('.,').istitle()) / max(len(words), 1)
    comma_ratio = src.count(',') / max(len(words), 1)
    if cap_ratio > 0.7 and comma_ratio > 0.25 and len(words) < 20:
        return True
    return False


def _subject_in_sentence(canonical: str, src: str, entity_set: EntitySet) -> bool:
    """
    Hard check: does the source sentence contain the subject entity or any of its
    known surface forms (abbreviations, aliases, size variants)?
    """
    src_lower = src.lower()
    surface_forms = [
        form for form, canon in entity_set.lookup.items()
        if canon == canonical
    ]
    for form in surface_forms:
        pattern = r'\b' + _re.escape(form) + r'\b'
        if _re.search(pattern, src_lower):
            return True
    return False


def _is_isa_sentence(src: str, subject: str, obj: str) -> bool:
    """
    Return True if the source sentence uses IS-A language rather than containment.
    Catches: "BERT and GPT are fine-tuning approaches" → IS-A, not comprises.
    Patterns: "X is a Y", "X are Y", "X is an example of Y", "X is one of Y".
    """
    src_lower = src.lower()
    # General IS-A patterns
    isa_patterns = [
        r'\bis\s+a\b',
        r'\bis\s+an\b',
        r'\bare\s+a\b',
        r'\bare\s+an\b',
        r'\bare\s+(?:fine-tuning|feature-based|pre-training|classification|regression)',
        r'\bis\s+(?:one\s+of|an\s+example\s+of|a\s+type\s+of|a\s+form\s+of)',
        r'\bare\s+(?:both\s+)?examples?\s+of',
    ]
    for pat in isa_patterns:
        if re.search(pat, src_lower):
            return True
    return False


def _object_in_sentence(obj_str: str, src: str) -> bool:
    """
    Check that the object string (or a meaningful portion of it) appears
    in the source sentence. Uses a relaxed word-overlap approach so that
    minor surface differences don't cause false negatives.
    """
    src_lower = src.lower()
    obj_lower = obj_str.lower().strip()

    # Exact match first
    if obj_lower in src_lower:
        return True

    # Word-boundary match of the full phrase
    pattern = r'\b' + _re.escape(obj_lower) + r'\b'
    if _re.search(pattern, src_lower):
        return True

    # Relaxed: at least half of the object's content words appear in the sentence
    stop = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was"}
    content_words = [w for w in obj_lower.split() if w not in stop and len(w) > 2]
    if not content_words:
        return True  # empty/stopword-only object — can't check
    matches = sum(1 for w in content_words if w in src_lower)
    return matches >= max(1, len(content_words) // 2)


def _extract_json_object(raw: str) -> dict | None:
    """
    Robustly extract the first valid JSON object from raw text.

    Uses brace-counting instead of regex so brackets inside string values
    (e.g. "[Table 2]" in a source_sentence) never cause premature truncation.
    The old regex r'\\{"triples"\\s*:.*?\\]\\s*\\}' used non-greedy .*? which
    would stop at the first ] in any string value, producing invalid JSON.
    """
    # Strategy 1: direct parse — fastest path for clean model output
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: brace-counting scan to isolate the outermost {...}
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
                    # This brace pair was invalid JSON; keep scanning
                    start = None
    return None


def _parse_fixed_output(
    raw: str,
    entity_set: EntitySet,
    valid_relations: list[str],
) -> list[dict]:
    """
    Parse LLM output and validate:
    - Subject must be in fixed entity set
    - Predicate must be in valid relations
    - Object must be non-empty and clean
    """
    raw = _strip_think_and_fences(raw)

    if not raw:
        logger.warning("[PARSE] Output empty after stripping think/fences — likely token-limit truncation mid-think")
        return []

    data = _extract_json_object(raw)
    if data is None:
        logger.warning(f"[PARSE] No valid JSON found. First 200 chars: {raw[:200]!r}")
        return []

    triples       = data.get("triples", [])
    valid_rels_lc = {r.lower() for r in valid_relations}
    result        = []

    for t in triples:
        if not isinstance(t, dict):
            continue

        subj = str(t.get("subject", "")).strip()
        pred = str(t.get("predicate", "")).strip()
        obj  = str(t.get("object",  "")).strip()

        if not subj or not pred or not obj:
            continue

        # Validate subject — must be in fixed entity set (canonical or alias)
        canonical_subj = entity_set.match(subj)
        if canonical_subj is None:
            # Strip parentheticals: "Masked Language Model (MLM)" → "Masked Language Model"
            subj_stripped = re.sub(r'\s*\(.*?\)', '', subj).strip()
            canonical_subj = entity_set.match(subj_stripped)
        if canonical_subj is None:
            # Strip subscript variants: "BERT_BASE", "BERTLARGE", "BERT-Base" → "BERT"
            subj_base = re.sub(r'[-_\s]?(base|large|small|tiny|medium|xl|xxl|\d+[bBmM])$', '',
                               subj, flags=re.IGNORECASE).strip()
            canonical_subj = entity_set.match(subj_base)
        if canonical_subj is None:
            # Try just the part before a colon or dash
            subj_head = re.split(r'[:\-]', subj)[0].strip()
            canonical_subj = entity_set.match(subj_head)
        if canonical_subj is None:
            # Try substring: see if any known entity name is contained in subject
            for known in entity_set.entities:
                if known.lower() in subj.lower() or subj.lower() in known.lower():
                    canonical_subj = known
                    break
        if canonical_subj is None:
            logger.warning(f"Subject '{subj}' not in fixed entity set — skipping")
            continue

        # ── Source sentence quality gate ─────────────────────────────────────
        src = str(t.get("source_sentence", "")).strip()
        if _is_low_quality_source(src):
            logger.debug(f"Skipping triple — low-quality source: {src[:80]!r}")
            continue

        # ── Hard subject-presence check ───────────────────────────────────────
        # The subject entity (or any of its known surface forms) must appear
        # literally in the source sentence.  This catches the common failure where
        # the LLM correctly identifies the relation from context but the SENTENCE
        # being stored says "we" / "the model" / "the framework" rather than BERT.
        if not _subject_in_sentence(canonical_subj, src, entity_set):
            logger.debug(
                f"Skipping triple — subject '{canonical_subj}' not found in source: {src[:100]!r}"
            )
            continue

        # ── Object-presence check for explicit-object relations ───────────────
        _obj_required_preds = {
            "comprises", "uses", "trainedon", "evaluatedon", "comparesagainst",
            "splitfrom", "writtenby", "publishedin",
        }
        pred_norm = pred.lower().replace(" ", "").replace("-", "")
        if any(p in pred_norm for p in _obj_required_preds):
            obj_str = str(t.get("object", "")).strip()
            if obj_str and not _object_in_sentence(obj_str, src):
                logger.debug(
                    f"Skipping triple — object '{obj_str}' not found in source for '{pred}': {src[:100]!r}"
                )
                continue

        # ── addresses subject-type guard ──────────────────────────────────────
        # The ontology defines addresses as ResearchProcess/Paper → ResearchContext.
        # Tasks and Datasets (which ARE ResearchContext) cannot be subjects.
        # Heuristic: if the source sentence defines the subject AS a task/dataset,
        # it's ResearchContext and should be an object, not a subject.
        if pred_norm == "addresses":
            _task_indicators = [
                " is a ", " is an ", " are a ", " are an ",
                " is a binary ", " is a large-scale ", " is a crowdsourced ",
                " is a classification ", " is a regression ",
                " is a task ", " is a dataset ", " is a benchmark ",
            ]
            src_lower_addr = src.lower()
            subj_lower_addr = canonical_subj.lower()
            if any(
                subj_lower_addr in src_lower_addr and ind in src_lower_addr
                for ind in _task_indicators
            ):
                logger.debug(
                    f"Skipping addresses — subject '{canonical_subj}' is defined as a task/dataset in source"
                )
                continue

        # ── uses direction guard ──────────────────────────────────────────────
        # uses: ResearchProcess → ResearchArtifact. The subject must be the AGENT.
        # If source says "X is applied after/before/using Y", then Y uses X, not X uses Y.
        if pred_norm == "uses":
            _passive_patterns = [
                r'\bis applied (?:after|before|during|using|with)\b',
                r'\bis performed (?:after|before|using|with)\b',
                r'\bis done (?:after|before|using|with)\b',
            ]
            src_lower_uses = src.lower()
            subj_lower_uses = canonical_subj.lower()
            # Check if the subject appears to be the passive recipient
            for pat in _passive_patterns:
                if re.search(pat, src_lower_uses):
                    # Subject is likely the tool being used, not the agent
                    # Only skip if subject is short/tool-like (e.g. WordPiece, BPE)
                    if len(canonical_subj.split()) <= 2:
                        logger.debug(
                            f"Skipping uses — '{canonical_subj}' appears to be passive tool: {src[:100]!r}"
                        )
                        continue
        # If the source sentence uses "X is a Y" or "X are Y" language, the
        # relation is IS-A (type membership), NOT containment. Skip these.
        if pred_norm in ("comprises", "encompasses"):
            if _is_isa_sentence(src, str(t.get("subject", "")), str(t.get("object", ""))):
                logger.debug(
                    f"Skipping '{pred}' triple — source uses IS-A language: {src[:100]!r}"
                )
                continue

        # ── designedFor keyword guard ─────────────────────────────────────────
        # designedFor requires explicit design intent language in the sentence.
        if pred_norm == "designedfor":
            _design_keywords = [
                "designed for", "designed to", "built for", "built to",
                "intended for", "intended to", "purpose of", "specifically for",
                "created for", "developed for", "meant for", "goal is to",
                "aims to", "allows bert to model",
            ]
            src_lower_check = src.lower()
            if not any(kw in src_lower_check for kw in _design_keywords):
                logger.debug(
                    f"Skipping designedFor — no design intent language in: {src[:100]!r}"
                )
                continue

        # ── encompasses direction guard ───────────────────────────────────────
        # encompasses subject must be a broad domain, not a specific task.
        # Heuristic: if the subject string itself appears as the narrower concept
        # in the sentence (e.g. "machine translation" alongside "transfer learning"),
        # and the object is the broader concept, the direction is reversed — skip.
        if pred_norm == "encompasses":
            _broad_domains = {
                "natural language processing", "nlp", "machine learning", "ml",
                "deep learning", "artificial intelligence", "ai",
                "computer vision", "transfer learning", "pre-training",
                "representation learning", "supervised learning",
                "semi-supervised learning", "unsupervised learning",
            }
            obj_lower = str(t.get("object", "")).lower().strip()
            subj_lower_check = canonical_subj.lower().strip()
            if obj_lower in _broad_domains and subj_lower_check not in _broad_domains:
                logger.debug(
                    f"Skipping encompasses — subject '{canonical_subj}' is narrower than object '{obj_lower}'"
                )
                continue

        # Validate predicate — must be in allowed relations
        if pred.lower() not in valid_rels_lc:
            # Try closest match
            matched_rel = _closest_relation(pred, valid_relations)
            if matched_rel:
                logger.debug(f"Relation '{pred}' → snapped to '{matched_rel}'")
                pred = matched_rel
            else:
                logger.warning(f"Relation '{pred}' not in ontology and no close match — skipping")
                continue

        # Basic object quality check
        if len(obj) < 2 or len(obj.split()) > 8:
            continue

        result.append({
            "subject":         canonical_subj,
            "predicate":       pred.lower(),
            "object":          obj,
            "object_type":     str(t.get("object_type", "Other")).strip(),
            "source_sentence": str(t.get("source_sentence", "")).strip(),
        })

    return result


def _closest_relation(pred: str, valid_relations: list[str]) -> str | None:
    """
    Snap an invalid predicate to the closest valid relation.
    Handles tense variants (used→uses), plurals, camelCase, underscores.
    """
    import difflib

    pred_lower = pred.lower().replace("_", "").replace("-", "")
    rel_lowers = [(r, r.lower().replace("_", "").replace("-", "")) for r in valid_relations]

    # 1. Direct substring match
    for rel, rl in rel_lowers:
        if rl in pred_lower or pred_lower in rl:
            return rel

    # 2. Stem match — strip common suffixes (ed, ing, s, d)
    def stem(w):
        for suffix in ("ation", "ing", "ed", "es", "on", "d", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) > 3:
                return w[: -len(suffix)]
        return w

    pred_stem = stem(pred_lower)
    for rel, rl in rel_lowers:
        if stem(rl) == pred_stem or pred_stem in stem(rl) or stem(rl) in pred_stem:
            return rel

    # 3. Close edit distance (handles typos)
    rl_list = [rl for _, rl in rel_lowers]
    close = difflib.get_close_matches(pred_lower, rl_list, n=1, cutoff=0.75)
    if close:
        for rel, rl in rel_lowers:
            if rl == close[0]:
                return rel

    return None