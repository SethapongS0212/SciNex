"""
kg_evaluate.py
--------------
LLM-as-Judge evaluation for triple extraction quality.

For each extracted triple, the LLM reads the original source sentence
and judges whether the triple is faithful to it.

Verdicts:
  CORRECT   — triple is explicitly stated in the source
  PARTIAL   — triple is implied but not directly stated, or partially wrong
  INCORRECT — triple is wrong, hallucinated, or not related to the source

Every triple gets a stable `triple_id`, so verdicts can be resumed across runs and
joined against human labels — see gold_eval.py for the human/cross-model side.

Usage:
    python3 kg_evaluate.py --paper 2205.11361 --extractor fixed
    python3 kg_evaluate.py --all --extractor fixed fixed_scinex --resume \
        --summary-out output/eval/judge_corpus.json
    python3 kg_evaluate.py --all --extractor fixed --max-per-paper 40
    python3 kg_evaluate.py --paper BERT --extractor fixed --model Qwen/Qwen2.5-14B-Instruct

Entity aliases are resolved per paper automatically (same CSV the fixed extractor
used); --entity-csv overrides that for every paper.

Output:
    output/<paper>/kg/<extractor>/evaluation.json   — per-triple verdicts
    output/<paper>/kg/<extractor>/eval_summary.json — rates + per-predicate breakdown
    --summary-out                                   — corpus-level report across papers
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
KG_SUBDIR  = "kg"


def triple_id(triple: dict, extractor: str) -> str:
    """Stable id joining a triple across judge runs, human annotation and re-extraction.

    Keyed on content, not position, so re-running the judge or re-exporting a gold
    sample maps back onto the same rows. Includes the extractor family because CEO
    and scinex can produce byte-identical triples that must stay distinguishable.
    """
    parts = [
        extractor.split("/")[0],
        str(triple.get("paper", "")),
        str(triple.get("subject", "")),
        str(triple.get("predicate", "")),
        str(triple.get("object", "")),
        str(triple.get("source_sentence") or triple.get("source_text", ""))[:300],
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]

# ── Prompts ───────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for scientific knowledge extraction.

You will be given:
1. A source sentence from a scientific paper (with entity aliases shown inline as [= full name])
2. A knowledge triple (subject, predicate, object) extracted from that sentence

Your job is to judge whether the triple is correct.

Verdict options:
- CORRECT: The triple is explicitly and accurately stated in the source sentence
- PARTIAL: The triple is partially correct, implied but not directly stated, or has minor errors
- INCORRECT: The triple is wrong, hallucinated, not in the sentence, or completely off

Rules:
- Judge strictly based on the source sentence only
- Ignore your own knowledge — only what the sentence says matters
- A triple is INCORRECT if the source sentence does not support it at all
- A triple is PARTIAL if it captures the general idea but misses details

CRITICAL — entity aliases: The source sentence may use a short form like "BERT", "GPT", "ELMo"
or a variant like "OpenAI GPT", "BERT_BASE". These are shown inline as [= full canonical name].
NEVER mark a triple INCORRECT just because the sentence uses an abbreviation or short form.
"OpenAI GPT [= Generative Pre-trained Transformer]" in the sentence fully supports a triple
whose subject or object is "Generative Pre-trained Transformer".

CRITICAL — multi-entity sentences: If a sentence mentions multiple entities performing the same
action, a triple about ONE of them is still CORRECT if that entity is named.
Example: "GPT is trained on BooksCorpus; BERT is trained on BooksCorpus and Wikipedia."
→ (BERT, trainedOn, Wikipedia) is CORRECT — BERT IS named and Wikipedia IS listed for BERT.
→ (BERT, trainedOn, BooksCorpus) is CORRECT — same logic.
Do NOT mark these INCORRECT because the sentence also mentions GPT.
Do NOT mark trainedOn PARTIAL just because the sentence lists multiple datasets — listing
BooksCorpus as one of BERT's training datasets IS sufficient to mark (BERT, trainedOn, BooksCorpus) CORRECT.

CRITICAL — comprises leniency: if a sentence clearly describes X as a component or task within Y,
mark the triple CORRECT even if the word "comprises" is not used literally.
Example: "we also use a next sentence prediction task that jointly pre-trains text-pair representations"
→ (BERT, comprises, Next sentence prediction) is CORRECT — NSP is clearly a component of BERT's training.

CRITICAL — listing vs comparing: A sentence that merely lists models together does NOT support
a comparesAgainst triple. "including ELMo, GPT and BERT" is a list, not a comparison.
Only mark comparesAgainst CORRECT if the sentence explicitly compares or contrasts two models.
The following phrasings DO count as comparison and should be marked CORRECT:
  - "The most comparable method to BERT is OpenAI GPT" → CORRECT comparesAgainst
  - "Unlike GPT, BERT uses bidirectional attention" → CORRECT
  - "BERT outperforms GPT" → CORRECT
  - "compared to", "in contrast to", "differs from", "versus", "better than" → CORRECT
Do NOT mark INCORRECT just because the sentence says "most comparable" instead of "outperforms".

CRITICAL — ontology conformance: when a "Predicate definition" is given, the triple must
respect it as well as the sentence. The definition gives the predicate's domain → range
(what kind of thing the subject and object must be) and its intended use.
- Sentence supports the fact AND the predicate fits its definition → CORRECT
- Sentence supports the fact but the predicate is a loose/imprecise fit for its definition
  (e.g. the object is not the type the range calls for) → PARTIAL
- The subject and object are swapped relative to the domain → range direction → INCORRECT
Judge the predicate by the definition given, not by the everyday English meaning of its name.

Return ONLY valid JSON, no explanation outside the JSON:
{
  "verdict": "CORRECT" | "PARTIAL" | "INCORRECT",
  "reason": "brief one-sentence explanation"
}"""


def _make_judge_prompt(triple: dict, aliases: dict[str, str] | None = None,
                       relation_hint: str = "") -> str:
    source = triple.get("source_sentence") or triple.get("source_text", "")
    source = source[:500]

    subj = triple.get("subject", "")
    pred = triple.get("predicate", "")
    obj  = triple.get("object", "")

    subj_type = triple.get("subject_type", "")
    obj_type  = triple.get("object_type", "")

    type_info = ""
    if subj_type or obj_type:
        type_info = f"\nEntity types: {subj} = {subj_type}, {obj} = {obj_type}"

    _vendor_prefixes = {"openai", "google", "meta", "microsoft", "deepmind",
                        "huggingface", "stanford", "berkeley"}
    _sizes = {"base", "large", "small", "tiny", "medium", "xl", "xxl"}

    def _clean_surface_forms(entity_name: str) -> list[str]:
        if not aliases:
            return []
        entity_lower = entity_name.lower().strip()
        canonical = aliases.get(entity_lower, entity_name)
        surface_forms = {
            form for form, canon in aliases.items()
            if canon.lower() == canonical.lower() and form != entity_lower
        }
        def _priority(f: str) -> int:
            parts = f.split()
            if len(parts) == 1 and f.isalpha() and len(f) <= 6:
                return 0
            if len(parts) == 2 and parts[0] == "openai":
                return 1
            if len(parts) >= 2 and parts[-1].strip("-_") not in _sizes and parts[0] not in _vendor_prefixes:
                return 2
            return 99
        return sorted(
            (f for f in surface_forms if _priority(f) < 99 and len(f) <= 25),
            key=lambda f: (_priority(f), len(f))
        )[:5]

    # Build a normalised source sentence with aliases substituted inline.
    # This is the most reliable way to make the judge understand that e.g.
    # "OpenAI GPT" in the sentence IS "Generative Pre-trained Transformer" in the triple.
    normalised_source = source
    for entity in [subj, obj]:
        forms = _clean_surface_forms(entity)
        for form in forms:
            # Replace form with "form [= canonical]" so the judge sees the mapping
            import re as _re
            pattern = _re.compile(_re.escape(form), _re.IGNORECASE)
            replacement = f"{form} [= {entity}]"
            if _re.search(pattern, normalised_source) and f"[= {entity}]" not in normalised_source:
                normalised_source = pattern.sub(replacement, normalised_source, count=1)

    # Also build the short alias list note as a backup
    alias_lines = []
    for entity in [subj, obj]:
        forms = _clean_surface_forms(entity)
        if forms:
            alias_lines.append(f"  \"{entity}\" = {', '.join(repr(f) for f in forms)}")

    alias_note = ""
    if alias_lines:
        alias_note = (
            "\nEntity aliases (these are the SAME entity — do NOT mark INCORRECT "
            "just because the sentence uses the short form):\n" + "\n".join(alias_lines)
        )

    hint_note = f"\nPredicate definition: {pred} = {relation_hint}" if relation_hint else ""

    return f"""Source sentence (with entity aliases shown inline):
\"{normalised_source}\"

Extracted triple:
  Subject:   {subj}
  Predicate: {pred}
  Object:    {obj}{type_info}{hint_note}{alias_note}

Is this triple correct based on the source sentence?"""


# ── Judge class ───────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Uses an LLM to evaluate whether extracted triples are correct.
    Loads the model once and evaluates all triples in batches.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 4,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._device_override = device
        self.batch_size = batch_size
        self._pipeline  = None
        self._tokenizer = None

    @property
    def device(self):
        try:
            import torch
            return self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return self._device_override or "cpu"

    def _load(self):
        if self._pipeline is not None:
            return

        logger.info(f"Loading judge model ({self.model_name}) on {self.device}...")

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
            device_map={"": 0},         # force entire model onto GPU 0 (no silent CPU offload — see hands_off.md #4)
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,      # enough for verdict JSON even if thinking leaks
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
        )
        self._tokenizer = tokenizer
        logger.info("Judge model loaded.")

    def evaluate(self, triples: list[dict], aliases: dict[str, str] | None = None,
                 schema: dict[str, str] | None = None) -> list[dict]:
        """
        Evaluate a list of triples. Returns same list with added fields:
            verdict:    CORRECT | PARTIAL | INCORRECT
            reason:     brief explanation
            confidence: 0.0 - 1.0

        aliases: optional dict mapping canonical entity name → abbreviation used in text
                 e.g. {"Bidirectional Encoder Representations from Transformers": "BERT"}
                 Passed to the judge so it does not penalise full-name subjects when
                 the source sentence uses an abbreviation.
        schema:  optional relation → "Domain → Range | usage" map for the ontology the
                 triples were extracted under. Without it the judge scores the predicate
                 by its English name rather than its ontology definition, which makes a
                 CEO-vs-scinex comparison meaningless.
        """
        self._load()

        # Skip triples with no source sentence — can't judge without context
        judgeable = [t for t in triples if t.get("source_sentence") or t.get("source_text")]
        skipped   = [t for t in triples if not t.get("source_sentence") and not t.get("source_text")]

        if skipped:
            logger.warning(f"{len(skipped)} triples have no source sentence — marking as UNVERIFIABLE")
            for t in skipped:
                t["verdict"]    = "UNVERIFIABLE"
                t["reason"]     = "No source sentence available"
                t["confidence"] = 0.0

        logger.info(f"Evaluating {len(judgeable)} triples...")

        for i, triple in enumerate(judgeable):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(judgeable)}")

            verdict = self._judge_triple(triple, aliases=aliases, schema=schema)
            triple.update(verdict)

        return triples

    def _judge_triple(self, triple: dict, aliases: dict[str, str] | None = None,
                      schema: dict[str, str] | None = None) -> dict:
        """Run one LLM judge call for a single triple."""
        hint = ""
        if schema:
            pred = str(triple.get("predicate", "")).strip().lower()
            for rel, rel_hint in schema.items():
                if rel.lower() == pred:
                    hint = rel_hint
                    break

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": _make_judge_prompt(triple, aliases=aliases,
                                                             relation_hint=hint)},
        ]

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # Qwen3: disable <think> blocks
            )

            outputs = self._pipeline(prompt)
            raw = outputs[0]["generated_text"].strip()

            return _parse_verdict(raw)

        except Exception as e:
            logger.warning(f"Judge failed for triple ({triple.get('subject')} → {triple.get('object')}): {e}")
            return {
                "verdict":    "UNVERIFIABLE",
                "reason":     f"Evaluation error: {str(e)[:100]}",
            }

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
            logger.info("Judge model unloaded.")


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_verdict(raw: str) -> dict:
    """Parse LLM judge output into verdict dict."""
    raw = raw.strip()

    # Strip Qwen3 <think> blocks — both closed and unclosed (token limit hit mid-think)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<think>.*', '', raw, flags=re.DOTALL)
    raw = raw.strip()

    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    # Non-greedy: avoid spanning across multiple JSON objects
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if not match:
        raw_upper = raw.upper()
        if "INCORRECT" in raw_upper:
            verdict = "INCORRECT"
        elif "PARTIAL" in raw_upper:
            verdict = "PARTIAL"
        elif "CORRECT" in raw_upper:
            verdict = "CORRECT"
        else:
            verdict = "UNVERIFIABLE"
        return {"verdict": verdict, "reason": raw[:200]}

    try:
        data = json.loads(match.group())
        verdict = str(data.get("verdict", "UNVERIFIABLE")).upper().strip()

        if verdict not in ("CORRECT", "PARTIAL", "INCORRECT"):
            if "INCORRECT" in verdict:
                verdict = "INCORRECT"
            elif "PARTIAL" in verdict:
                verdict = "PARTIAL"
            elif "CORRECT" in verdict:
                verdict = "CORRECT"
            else:
                verdict = "UNVERIFIABLE"

        return {
            "verdict": verdict,
            "reason":  str(data.get("reason", ""))[:300],
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "verdict": "UNVERIFIABLE",
            "reason":  raw[:200],
        }


# ── Summary stats ─────────────────────────────────────────────────────────────

def compute_summary(evaluated_triples: list[dict]) -> dict:
    """Compute evaluation summary statistics."""
    total = len(evaluated_triples)
    if total == 0:
        return {"total": 0}

    counts = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0, "UNVERIFIABLE": 0}

    for t in evaluated_triples:
        verdict = t.get("verdict", "UNVERIFIABLE")
        counts[verdict] = counts.get(verdict, 0) + 1

    judgeable = total - counts["UNVERIFIABLE"]

    summary = {
        "total_triples":     total,
        "judgeable":         judgeable,
        "correct":           counts["CORRECT"],
        "partial":           counts["PARTIAL"],
        "incorrect":         counts["INCORRECT"],
        "unverifiable":      counts["UNVERIFIABLE"],
        "precision":         round(counts["CORRECT"] / judgeable, 3) if judgeable else 0,
        "partial_rate":      round(counts["PARTIAL"] / judgeable, 3) if judgeable else 0,
        "error_rate":        round(counts["INCORRECT"] / judgeable, 3) if judgeable else 0,
        "lenient_precision": round(
            (counts["CORRECT"] + 0.5 * counts["PARTIAL"]) / judgeable, 3
        ) if judgeable else 0,
        "by_predicate":      per_predicate(evaluated_triples),
    }

    return summary


def per_predicate(evaluated_triples: list[dict]) -> dict:
    """Verdict counts + precision per predicate — shows which relations to fix first."""
    buckets: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0, "UNVERIFIABLE": 0}
    )
    for t in evaluated_triples:
        b = buckets[str(t.get("predicate", "")).lower()]
        b["n"] += 1
        b[t.get("verdict", "UNVERIFIABLE")] = b.get(t.get("verdict", "UNVERIFIABLE"), 0) + 1

    out = {}
    for pred, b in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
        judgeable = b["n"] - b["UNVERIFIABLE"]
        out[pred] = {
            **b,
            "precision": round(b["CORRECT"] / judgeable, 3) if judgeable else 0,
        }
    return out


def print_summary(paper_name: str, extractor_name: str, summary: dict, sample_triples: list[dict]):
    """Print a readable evaluation report."""
    sep  = "═" * 65
    sep2 = "─" * 65

    print(f"\n{sep}")
    print(f"  EVALUATION REPORT — {paper_name} [{extractor_name.upper()}]")
    print(f"{sep}")
    print(f"  Total triples evaluated : {summary['total_triples']}")
    print(f"  Judgeable               : {summary['judgeable']}")
    print(sep2)
    print(f"  ✓  CORRECT              : {summary['correct']}  ({summary['precision']*100:.1f}%)")
    print(f"  ~  PARTIAL              : {summary['partial']}  ({summary['partial_rate']*100:.1f}%)")
    print(f"  ✗  INCORRECT            : {summary['incorrect']}  ({summary['error_rate']*100:.1f}%)")
    print(sep2)
    print(f"  Strict precision        : {summary['precision']*100:.1f}%")
    print(f"  Lenient precision       : {summary['lenient_precision']*100:.1f}%")
    print(sep2)

    # Sample triples per verdict
    for verdict in ("CORRECT", "PARTIAL", "INCORRECT"):
        samples = [t for t in sample_triples if t.get("verdict") == verdict][:2]
        if samples:
            print(f"\n  [{verdict} examples]")
            for t in samples:
                print(f"    ({t.get('subject','')[:30]}) --[{t.get('predicate','')[:20]}]--> ({t.get('object','')[:30]})")
                print(f"    ↳ {t.get('reason','')[:80]}")

    print(f"\n{sep}\n")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def find_extractor_dirs(paper_dir: Path, filter_extractors: list[str] | None) -> dict[str, Path]:
    """Find extractor output dirs, same logic as kg_compare.py."""
    kg_dir = paper_dir / KG_SUBDIR
    if not kg_dir.exists():
        return {}

    extractors = {}
    for child in sorted(kg_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "comparison_report.json":
            continue
        if (child / "triples.json").exists():
            extractors[child.name] = child
        else:
            for subchild in sorted(child.iterdir()):
                if subchild.is_dir() and (subchild / "triples.json").exists():
                    extractors[f"{child.name}/{subchild.name}"] = subchild

    if filter_extractors and filter_extractors != ["all"]:
        # Match the extractor family exactly — "fixed" must not also select
        # "fixed_scinex", which a substring test would do.
        wanted = set(filter_extractors)
        extractors = {
            k: v for k, v in extractors.items()
            if k.split("/")[0] in wanted
        }

    return extractors


def _cached_verdicts(ext_dir: Path) -> dict[str, dict]:
    """Verdicts from a previous run of this extractor dir, keyed by triple_id."""
    path = ext_dir / "evaluation.json"
    if not path.exists():
        return {}
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        t["triple_id"]: {"verdict": t.get("verdict"), "reason": t.get("reason", "")}
        for t in prev
        if t.get("triple_id") and t.get("verdict")
    }


_SCHEMA_CACHE: dict[str, dict] = {}


def schema_for(extractor_family: str, ontology_file: str) -> dict[str, str] | None:
    """Relation → definition map matching the ontology the triples were extracted under."""
    if extractor_family in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[extractor_family]

    # fixed / relation are CEO; fixed_scinex / relation_scinex come from the OWL.
    schema = None
    if extractor_family in ("fixed", "relation"):
        from kg_extraction.fixed_extractor import _CEO_SCHEMA
        schema = _CEO_SCHEMA
    elif extractor_family in ("fixed_scinex", "relation_scinex"):
        try:
            from kg_extraction.ontology_loader import load_ontology
            _, schema = load_ontology(ontology_file)
        except Exception as e:
            logger.warning(f"Could not load {ontology_file}: {e}. Judging scinex triples "
                           "without predicate definitions.")
    _SCHEMA_CACHE[extractor_family] = schema
    return schema


def load_aliases_for(paper_name: str, explicit_csv: str | None) -> dict[str, str] | None:
    """Surface-form → canonical map for the judge, resolved per paper.

    A corpus run spans papers with different entity CSVs, so the judge needs the
    right one per paper or it penalises abbreviations it was never told about.
    """
    from kg_extraction.entity_loader import load_entity_csv

    if explicit_csv:
        csv_path = Path(explicit_csv)
    else:
        from kg_main import resolve_entity_csv
        csv_path = resolve_entity_csv(paper_name)

    if not csv_path or not Path(csv_path).exists():
        return None
    try:
        return load_entity_csv(csv_path, tp_only=False).all_aliases
    except Exception as e:
        logger.warning(f"Could not load entity CSV '{csv_path}': {e}")
        return None


def evaluate_paper(
    paper_dir: Path,
    judge: LLMJudge,
    filter_extractors: list[str] | None,
    entity_csv: str | None = None,
    resume: bool = False,
    max_per_paper: int | None = None,
    ontology_file: str = "scinex_refined_14.owl",
) -> dict:
    """Run LLM-as-Judge evaluation for all extractors of one paper."""
    paper_name     = paper_dir.name
    extractor_dirs = find_extractor_dirs(paper_dir, filter_extractors)

    if not extractor_dirs:
        logger.warning(f"No KG outputs found for '{paper_name}'. Run kg_main.py first.")
        return {}

    aliases = load_aliases_for(paper_name, entity_csv)
    if aliases:
        logger.info(f"{paper_name}: {len(aliases)} entity surface forms loaded for the judge")

    paper_results = {}

    for ext_name, ext_dir in extractor_dirs.items():
        triples_path = ext_dir / "triples.json"

        try:
            with open(triples_path, "r", encoding="utf-8") as f:
                triples = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {triples_path}: {e}")
            continue

        if not triples:
            logger.warning(f"No triples found for {ext_name}. Skipping.")
            continue

        for t in triples:
            # Fill `paper` before hashing — gold_eval.py does the same, and the id
            # must come out identical in both or the labels won't rejoin.
            t.setdefault("paper", paper_name)
            t["triple_id"] = triple_id(t, ext_name)
            t.setdefault("extractor", ext_name)

        if max_per_paper is not None and len(triples) > max_per_paper:
            # Deterministic subsample by id so repeated runs judge the same triples.
            triples = sorted(triples, key=lambda t: t["triple_id"])[:max_per_paper]

        cached = _cached_verdicts(ext_dir) if resume else {}
        todo   = [t for t in triples if t["triple_id"] not in cached]
        for t in triples:
            if t["triple_id"] in cached:
                t.update(cached[t["triple_id"]])

        logger.info(f"{'─'*65}")
        logger.info(
            f"Evaluating: {paper_name} / {ext_name} "
            f"({len(todo)} to judge, {len(triples) - len(todo)} cached)"
        )

        # Run evaluation — aliases resolve abbreviations, schema gives the judge the
        # ontology definition of each predicate
        if todo:
            judge.evaluate(todo, aliases=aliases,
                           schema=schema_for(ext_name.split("/")[0], ontology_file))
        evaluated = triples
        summary   = compute_summary(evaluated)

        # Print report
        print_summary(paper_name, ext_name, summary, evaluated)

        # Save evaluation results
        eval_path = ext_dir / "evaluation.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluated, f, indent=2, ensure_ascii=False)

        summary_path = ext_dir / "eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved → {eval_path}")
        logger.info(f"Saved → {summary_path}")

        paper_results[ext_name] = {"summary": summary, "triples": evaluated}

    return paper_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation for triple extraction quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 kg_evaluate.py --paper BERT --extractor fixed
  python3 kg_evaluate.py --all --extractor fixed fixed_scinex --resume \\
      --summary-out output/eval/judge_corpus.json
  python3 kg_evaluate.py --all --extractor fixed --max-per-paper 40
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paper", metavar="PAPER_NAME",
        help="Single paper to evaluate (e.g. 2205.11361)")
    group.add_argument("--all", action="store_true",
        help="Evaluate all papers in output/")

    parser.add_argument("--extractor", nargs="+", default=["all"],
        help="Extractor families to evaluate: rebel, iter, llm, fixed, fixed_scinex, pair, "
             "all (default: all). Matched exactly — 'fixed' does NOT include 'fixed_scinex'")
    parser.add_argument("--entity-csv", default=None, metavar="CSV_PATH",
        help="Entity CSV file (same as used for fixed extraction). When provided, the judge "
             "is told about abbreviations so it does not penalise full-name subjects when "
             "the source sentence uses an abbreviation (e.g. BERT vs full name).")
    parser.add_argument("--model", default=None,
        help="Judge model HuggingFace ID (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--limit", type=int, default=None,
        help="When using --all, process only first N papers")
    parser.add_argument("--max-per-paper", type=int, default=None, metavar="N",
        help="Judge at most N triples per paper+extractor (deterministic subsample by "
             "triple id — the same N triples on every re-run)")
    parser.add_argument("--resume", action="store_true",
        help="Reuse verdicts already in evaluation.json (matched by triple id) and only "
             "judge new triples")
    parser.add_argument("--ontology-file", default="scinex_refined_14.owl", metavar="OWL",
        help="Ontology file used to give the judge predicate definitions for "
             "fixed_scinex triples (CEO definitions are built in)")
    parser.add_argument("--summary-out", default=None, metavar="JSON",
        help="Write a corpus-level summary (all papers, per extractor and per predicate) "
             "to this path")
    parser.add_argument("--no-gpu", action="store_true",
        help="Force CPU inference")

    return parser.parse_args()


def corpus_summary(all_results: dict, per_paper_triples: dict) -> dict:
    """Aggregate per-paper verdicts into one corpus-level report per extractor."""
    out = {}
    for ext_name, triples in per_paper_triples.items():
        summary = compute_summary(triples)
        summary["papers"] = sorted({t.get("paper", "") for t in triples})
        summary["n_papers"] = len(summary["papers"])
        out[ext_name] = summary
    return {"by_extractor": out, "by_paper": all_results}


def main():
    args = parse_args()

    # Resolve papers
    if args.all:
        if not OUTPUT_DIR.exists():
            logger.error(f"Output directory not found.")
            sys.exit(1)
        papers = sorted([
            d for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and (d / KG_SUBDIR).exists()
        ])
        if args.limit:
            papers = papers[:args.limit]
    else:
        paper_dir = OUTPUT_DIR / args.paper
        if not paper_dir.exists():
            logger.error(f"Paper not found: {paper_dir}")
            sys.exit(1)
        papers = [paper_dir]

    if not papers:
        logger.error("No papers with KG outputs found.")
        sys.exit(1)

    filter_ext = None if args.extractor == ["all"] else args.extractor

    # Load judge once
    device = "cpu" if args.no_gpu else None
    judge  = LLMJudge(model_name=args.model, device=device)

    # Evaluate
    all_results: dict[str, dict] = {}
    per_extractor_triples: dict[str, list] = defaultdict(list)
    for paper_dir in papers:
        results = evaluate_paper(
            paper_dir=paper_dir,
            judge=judge,
            filter_extractors=filter_ext,
            entity_csv=args.entity_csv,
            resume=args.resume,
            max_per_paper=args.max_per_paper,
            ontology_file=args.ontology_file,
        )
        if results:
            all_results[paper_dir.name] = {k: v["summary"] for k, v in results.items()}
            for ext, payload in results.items():
                per_extractor_triples[ext].extend(payload["triples"])

    judge.unload()

    # Final summary table across all papers and extractors
    if all_results:
        print(f"\n{'═'*70}")
        print("  FINAL SUMMARY")
        print(f"{'─'*70}")
        print(f"  {'Paper':<25} {'Extractor':<25} {'Precision':>10} {'Lenient':>10}")
        print(f"{'─'*70}")
        for paper, extractors in all_results.items():
            for ext, summary in extractors.items():
                print(
                    f"  {paper:<25} {ext:<25} "
                    f"{summary['precision']*100:>9.1f}% "
                    f"{summary['lenient_precision']*100:>9.1f}%"
                )
        print(f"{'─'*70}")
        print(f"  {'CORPUS':<25} {'Extractor':<25} {'Precision':>10} {'Lenient':>10}")
        corpus = corpus_summary(all_results, per_extractor_triples)
        for ext, summary in corpus["by_extractor"].items():
            print(
                f"  {'(' + str(summary['n_papers']) + ' papers)':<25} {ext:<25} "
                f"{summary['precision']*100:>9.1f}% "
                f"{summary['lenient_precision']*100:>9.1f}%"
            )
        print(f"{'═'*70}\n")

        if args.summary_out:
            out_path = Path(args.summary_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Corpus summary → {out_path}")


if __name__ == "__main__":
    main()