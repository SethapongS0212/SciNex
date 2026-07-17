"""
llm_refiner.py — Qwen2.5-14B whitespace repair for PDF-extracted text.

Speed improvements over the original:
  1. 4-bit quantization (NF4)     — halves VRAM, ~2× inference throughput
  2. Flash Attention 2            — ~30% faster attention (if flash-attn installed)
  3. torch.compile                — ~20-40% extra throughput on repeated calls
  4. Pre-LLM regex bypass         — skips blocks that have no actual repair patterns
  5. Batch processing             — amortises GPU overhead across multiple blocks

To get the full benefit of batching, update refine_document_llm() in main.py:

    OLD (one block at a time):
        from parser.llm_refiner import refine_block
        ...
        for block in section["content"]:
            refined = refine_block(block) if block.get("needs_llm") else block

    NEW (batch across the whole document):
        from parser.llm_refiner import refine_blocks_batch
        ...
        all_blocks = [b for s in doc["sections"] for b in s["content"]]
        refined    = refine_blocks_batch(all_blocks)
        # put them back
        idx = 0
        for section in doc["sections"]:
            n = len(section["content"])
            section["content"] = refined[idx:idx + n]
            idx += n

    refine_block(block) still works unchanged for backward compatibility —
    it just won't batch across sections.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re

# ── CONFIG ──────────────────────────────────────────────────────────
MY_HUGGIEFACE_TOKEN = "hf_xxx"
MODEL_NAME          = "Qwen/Qwen2.5-14B-Instruct"
MAX_INPUT_TOKENS    = 2048
BATCH_SIZE          = 4     # blocks per forward pass
                            # safe default for 16 GB VRAM in 4-bit
                            # bump to 8 if you have ≥24 GB

# ── TOKENIZER ───────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=MY_HUGGIEFACE_TOKEN,
    trust_remote_code=True,
    padding_side="left",   # required for correct batch causal-LM generation
)

# Qwen uses eos as pad by default; make it explicit so generate() doesn't warn
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── 4-BIT QUANTIZATION ──────────────────────────────────────────────
# NF4 + double quant ≈ 3.5 bits/param effective; quality loss is negligible
# for a whitespace-repair task. Halves VRAM vs fp16 and speeds up matmuls.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ── FLASH ATTENTION 2 ───────────────────────────────────────────────
# Install separately if not present:
#   pip install flash-attn --no-build-isolation
_attn_impl = "eager"
try:
    import flash_attn  # noqa: F401
    _attn_impl = "flash_attention_2"
    print("[LLM] Flash Attention 2 enabled")
except ImportError:
    print("[LLM] flash-attn not found — using eager attention "
          "(pip install flash-attn --no-build-isolation to enable)")

# ── LOAD MODEL ──────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    token=MY_HUGGIEFACE_TOKEN,
    trust_remote_code=True,
    quantization_config=bnb_config,    # activates 4-bit
    attn_implementation=_attn_impl,
    low_cpu_mem_usage=True,
)
model.eval()

# ── TORCH.COMPILE ───────────────────────────────────────────────────
# Fuses ops for ~20-40% throughput gain on repeated inference.
# "reduce-overhead" is the sweet spot for inference workloads.
# Skipped gracefully if incompatible (e.g., some bitsandbytes versions).
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("[LLM] torch.compile enabled")
except Exception as exc:
    print(f"[LLM] torch.compile skipped ({exc})")

DEVICE = next(model.parameters()).device


# ── CLEANING HELPERS ────────────────────────────────────────────────
def deduplicate_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    out = []
    for s in sentences:
        if not out or s.strip() != out[-1].strip():
            out.append(s)
    return " ".join(out)


def clean_llm_output(text: str) -> str:
    text = re.sub(r"(?i)cleaned text:\s*", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = deduplicate_sentences(text)
    return text.strip()


# ── SAFETY: CHARACTER-BASED WORD PRESERVATION ───────────────────────
def word_preserved(original: str, cleaned: str) -> bool:
    normalised_original = re.sub(r"-\s+", "", original)
    normalised_original = re.sub(r"\s+", " ", normalised_original).strip()
    normalised_cleaned  = re.sub(r"\s+", " ", cleaned).strip()

    orig_len  = len(normalised_original)
    clean_len = len(normalised_cleaned)

    if orig_len == 0:
        return clean_len == 0

    ratio = clean_len / orig_len
    return 0.90 <= ratio <= 1.10


# ── PRE-LLM BYPASS ──────────────────────────────────────────────────
# The LLM can only help if the text has:
#   • a newline (stray line break or hyphen-split across lines)
#   • multiple consecutive spaces/tabs (weird PDF spacing)
#   • an inline broken hyphen like "trans- former"
# If none of these are present, the output will be identical to the input.
# Skipping saves a full forward pass per block.
_REPAIR_NEEDED_RE = re.compile(r"\n|[ \t]{2,}|\w-[ \t]+\w")


def _repair_needed(text: str) -> bool:
    return bool(_REPAIR_NEEDED_RE.search(text))


# ── PROMPT ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a whitespace repair engine for PDF-extracted text. You operate as a
deterministic formatting tool, not a language model. You have no creative
latitude whatsoever.

━━━ PERMITTED OPERATIONS (the only things you may do) ━━━
  1. Rejoin words split by hyphenation across line breaks
       e.g. "trans-\\nformer"   →  "transformer"
       e.g. "down-\\nstream"    →  "downstream"
  2. Collapse multiple consecutive whitespace characters into a single space
  3. Remove spurious mid-sentence newlines
  4. Strip leading and trailing whitespace from the block

━━━ ABSOLUTE PROHIBITIONS (any violation is a critical failure) ━━━
  - Do NOT add any word that is not present in the input
  - Do NOT remove any word from the input
  - Do NOT reorder words
  - Do NOT paraphrase, rephrase, or rewrite
  - Do NOT correct grammar or spelling
  - Do NOT fix or alter punctuation
  - Do NOT modify numbers, symbols, or citations
  - Do NOT summarize or interpret meaning
  - Do NOT add explanation, commentary, or preamble

━━━ CALIBRATION EXAMPLES ━━━

  <example>
    <input>Trans- former mod-\\nels are widely used in  NLP .</input>
    <output>Transformer models are widely used in NLP.</output>
  </example>

  <example>
    <input>The re- sults demon-\\nstrate that   pre-training on large cor-\\npora
improves down- stream per-\\nformance.</input>
    <output>The results demonstrate that pre-training on large corpora improves downstream performance.</output>
  </example>

  <example>
    <input>Fig. 3 shows a sig-\\nnificant increase ( p < 0.001 )  in the treat-\\nment group.</input>
    <output>Fig. 3 shows a significant increase ( p < 0.001 ) in the treatment group.</output>
  </example>

━━━ OUTPUT RULE ━━━
  Return only the repaired text.
  No quotation marks. No labels. No explanation. Nothing else.

━━━ INPUT ━━━

<input_block>
{raw_text}
</input_block>
"""


def _build_messages(raw_text: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": raw_text},
    ]


# ── SHARED SAFETY FILTER ─────────────────────────────────────────────
def _apply_safety(raw_text: str, decoded: str) -> tuple:
    """
    Validates decoded LLM output against the original text.
    Returns (cleaned_text, ok: bool).
    If ok is False the caller should fall back to raw_text.
    """
    cleaned = clean_llm_output(decoded)

    # Hallucination guard: reject if >3% of output words are novel
    input_vocab  = set(re.findall(r"[a-zA-Z']+", raw_text.lower()))
    output_words = re.findall(r"[a-zA-Z']+", cleaned.lower())
    novel        = [w for w in output_words if w not in input_vocab and len(w) > 3]
    novel_ratio  = len(novel) / max(len(output_words), 1)

    if novel_ratio > 0.03 and len(novel) > 2:
        print(f"[LLM] ⚠️  hallucination guard "
              f"({len(novel)} novel words: {novel[:5]}) → fallback")
        return None, False

    # Word-count / length preservation guard
    if not word_preserved(raw_text, cleaned):
        print("[LLM] ⚠️  word mismatch → fallback")
        return None, False

    raw_char_len   = len(" ".join(raw_text.split()))
    clean_char_len = len(cleaned)
    ratio          = clean_char_len / max(raw_char_len, 1)

    if ratio > 1.5 or ratio < 0.7:
        print(f"[LLM] ⚠️  length anomaly (ratio={ratio:.2f}) → fallback")
        return None, False

    return cleaned, True


# ── BATCH REFINER ────────────────────────────────────────────────────
def refine_blocks_batch(blocks: list, batch_size: int = BATCH_SIZE) -> list:
    """
    Refine a flat list of blocks in batches.

    Blocks that are tables, lack needs_llm, are too short, or have no
    actual repair patterns are passed through untouched. The rest are
    processed in batches of `batch_size` for efficient GPU utilisation.

    This is the recommended entrypoint. See the module docstring for
    the two-line change needed in main.py to use it.
    """
    # ── Identify which blocks actually need the LLM ──────────────────
    to_refine: list[int] = []
    for i, block in enumerate(blocks):
        btype = block.get("type") or block.get("block_type", "")
        if btype == "table":
            continue
        if not block.get("needs_llm"):
            continue
        text = block.get("text", "")
        if len(text.strip()) < 120:
            continue
        if not _repair_needed(text):
            # No repair pattern found — LLM output would be identical to input
            print(f"[LLM] ⏭  bypass (no repair pattern) | len={len(text)}")
            continue
        to_refine.append(i)

    print(f"[LLM] {len(to_refine)} blocks queued "
          f"(skipped {len(blocks) - len(to_refine)}) | batch_size={batch_size}")

    # ── Process in batches ───────────────────────────────────────────
    for batch_start in range(0, len(to_refine), batch_size):
        batch_idx   = to_refine[batch_start : batch_start + batch_size]
        batch_texts = [blocks[i]["text"] for i in batch_idx]

        btype_labels = [
            blocks[i].get("block_type", blocks[i].get("type", "?"))
            for i in batch_idx
        ]
        print(f"[LLM] batch {batch_start // batch_size + 1} | "
              f"{len(batch_idx)} blocks | types={btype_labels}")

        # Build chat-template prompts for the whole batch
        prompts = [
            tokenizer.apply_chat_template(
                _build_messages(t),
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in batch_texts
        ]

        # Tokenise with left-padding so all sequences align on the right
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=True,
        ).to(DEVICE)

        # Scale max_new_tokens to the longest text in the batch
        max_text_chars = max(len(t) for t in batch_texts)
        estimated_toks = max(max_text_chars // 4, 32)
        dynamic_max    = min(int(estimated_toks * 1.2) + 16, 512)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=dynamic_max,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # With left-padding, all sequences share the same input_len offset
        input_len = inputs["input_ids"].shape[1]

        for j, (global_i, raw_text) in enumerate(zip(batch_idx, batch_texts)):
            new_tokens = outputs[j][input_len:]
            decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            cleaned, ok = _apply_safety(raw_text, decoded)
            block = blocks[global_i]

            if ok:
                if cleaned != raw_text:
                    print(f"[LLM] ✅ cleaned | len {len(raw_text)} → {len(cleaned)}")
                block["text"]    = cleaned
                block["refined"] = True
            else:
                block["refined"] = False

    return blocks


# ── SINGLE-BLOCK REFINER (backward-compatible) ───────────────────────
def refine_block(block: dict) -> dict:
    """
    Drop-in replacement for the original refine_block().
    Works without any changes to main.py, but processes one block at a time.
    For better throughput, switch to refine_blocks_batch() — see module docstring.
    """
    return refine_blocks_batch([block], batch_size=1)[0]