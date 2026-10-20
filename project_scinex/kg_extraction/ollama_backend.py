"""
ollama_backend.py
-----------------
Run the fixed / relation extractors against a local **Ollama** server instead of
transformers + bitsandbytes.

Why this exists: on the Windows laptop, `import torch` dies with
`OSError: [WinError 4551] An Application Control policy has blocked this file`.
Windows 11's **Smart App Control** (Home edition, `VerifiedAndReputablePolicyState = 1`)
refuses to load torch's unsigned DLLs. Turning Smart App Control off is a **one-way**
change — Windows will not let you turn it back on without reinstalling — so the
supported way to get a local model here is a signed binary that brings its own runtime.
Ollama is signed, and it also sidesteps the 8GB VRAM ceiling problem by serving GGUF
quantisations that a 14B-class model can't reach through bitsandbytes on this GPU.

What it does NOT change: the prompt, the ontology, and every post-parse guard are the
extractor's own — this file only replaces the generation call. The extractors reach the
model through exactly two attributes:

    self._tokenizer.apply_chat_template(messages, ...) -> prompt
    self._pipeline(prompt) -> [{"generated_text": str}]

so a shim for each is enough, and `_load()` becomes a no-op.

⚠ Results from here are NOT comparable to the project's Qwen3-14B numbers. Different
model, different quantisation. Use this for smoke-testing the pipeline and for a
preliminary quality read — not for the TASK.md deliverable.
"""

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
# Judging with the same model that did the extraction is not an independent
# check. Kept as a separate default so the two can diverge.
DEFAULT_JUDGE_MODEL = "qwen2.5:7b-instruct"


class _ChatTemplateShim:
    """
    Stands in for a HF tokenizer.

    The extractor calls `apply_chat_template(...)` and hands the result straight to
    `self._pipeline`. Ollama's /api/chat wants the message list itself, so the
    "template" here is the identity function — the server applies the model's real
    template. `enable_thinking` is accepted and ignored; thinking is disabled on the
    request instead (see _OllamaPipeline).
    """

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=None):
        if tokenize:
            raise ValueError("ollama_backend does not tokenize locally")
        return messages


class _OllamaPipeline:
    """Mimics a transformers text-generation pipeline over Ollama's /api/chat."""

    def __init__(self, model: str, host: str, max_new_tokens: int,
                 timeout: int = 900, num_ctx: int = 8192):
        self.model = model
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        # Ollama defaults to a 4096-token context REGARDLESS of what the model
        # supports. The relation/fixed system prompt is ~13.6k chars (~3.4k
        # tokens) before the paragraph is even added, so at 4096 the prompt
        # crowds out the answer and generation returns nothing usable.
        # 8192 is the ceiling on an 8GB GPU: measured 6.2GB / 100% GPU here,
        # while 16384 needs 7.8GB and makes Ollama spill 20% to CPU, which
        # roughly quadrupled the time per call.
        self.num_ctx = num_ctx
        self.calls = 0

    def __call__(self, prompt: Any):
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Qwen3 emits a <think> block by default. The extractor strips think
            # blocks anyway, but generating them burns the token budget — which is
            # exactly how a previous run silently truncated 80 papers.
            "think": False,
            "options": {
                "temperature": 0,        # greedy, to match the transformers path
                "top_p": 1,
                "num_predict": self.max_new_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        self.calls += 1
        logger.info(f"[ollama] request #{self.calls} "
                    f"({sum(len(m.get('content', '')) for m in messages)} prompt chars)")
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Is it running? Start it with `ollama serve`."
            ) from e
        if r.status_code != 200:
            # Some builds reject "think" on non-thinking models; retry without it
            # rather than failing the whole paper.
            if "think" in r.text.lower():
                payload.pop("think", None)
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            if r.status_code != 200:
                raise RuntimeError(f"Ollama returned {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
            content = body["message"]["content"]
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Unexpected Ollama response: {r.text[:300]}") from e
        if not content.strip():
            logger.warning(f"[ollama] EMPTY generation "
                           f"(done_reason={body.get('done_reason')!r}, "
                           f"eval_count={body.get('eval_count')}) — if done_reason is "
                           f"'length', raise num_ctx or lower --max-new-tokens")
        else:
            logger.info(f"[ollama] <- {len(content)} chars, "
                        f"eval_count={body.get('eval_count')}")
        return [{"generated_text": content}]


def patch_extractors(model: str = DEFAULT_MODEL,
                     host: str = DEFAULT_HOST,
                     max_new_tokens: int = 1024,
                     num_ctx: int = 8192) -> None:
    """
    Redirect every fixed/relation extractor instance to Ollama, process-wide.

    Patches the class rather than an instance because `kg_main` constructs the
    extractors itself. Three things are replaced:
      _load()   -> installs the shims instead of loading a 4-bit checkpoint
      device    -> the real property does `import torch` inside `except ImportError`,
                   which no longer catches anything now that torch raises OSError
      unload()  -> the real one calls torch.cuda.empty_cache()
    """
    from .fixed_extractor import FixedTripleExtractor

    def _load(self):
        if self._pipeline is not None:
            return
        logger.info(f"[ollama] using {model} at {host} "
                    f"(max_new_tokens={max_new_tokens}, num_ctx={num_ctx})")
        self._tokenizer = _ChatTemplateShim()
        self._pipeline = _OllamaPipeline(
            model=model, host=host,
            max_new_tokens=getattr(self, "max_new_tokens", max_new_tokens),
            num_ctx=num_ctx,
        )

    def _unload(self):
        self._pipeline = None
        self._tokenizer = None

    FixedTripleExtractor._load = _load
    FixedTripleExtractor.unload = _unload
    FixedTripleExtractor.device = property(lambda self: f"ollama:{model}")
    logger.info(f"[ollama] extractors patched -> {model}")


def patch_judge(model: str = DEFAULT_JUDGE_MODEL,
                host: str = DEFAULT_HOST,
                max_new_tokens: int = 512,
                num_ctx: int = 8192) -> None:
    """
    Same treatment for `kg_evaluate.LLMJudge`, which loads its own 4-bit model.

    The judge reaches its model through the identical two attributes as the
    extractors (`_tokenizer.apply_chat_template` then `_pipeline`), so the same
    shims work. A verdict is short — 512 new tokens is plenty, and capping it
    keeps the judge from monologuing.

    Use a DIFFERENT model family from the extractor where possible: a model
    grading its own output is not an independent check.
    """
    import kg_evaluate

    def _load(self):
        if self._pipeline is not None:
            return
        logger.info(f"[ollama] judge = {model} at {host} (num_ctx={num_ctx})")
        self._tokenizer = _ChatTemplateShim()
        self._pipeline = _OllamaPipeline(
            model=model, host=host, max_new_tokens=max_new_tokens, num_ctx=num_ctx,
        )

    def _unload(self):
        self._pipeline = None
        self._tokenizer = None

    kg_evaluate.LLMJudge._load = _load
    kg_evaluate.LLMJudge.unload = _unload
    kg_evaluate.LLMJudge.device = property(lambda self: f"ollama:{model}")
    logger.info(f"[ollama] judge patched -> {model}")


def check_server(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> tuple[bool, str]:
    """(reachable_and_model_present, message) — call before spending time on a batch."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    except requests.exceptions.RequestException as e:
        return False, f"Ollama unreachable at {host}: {type(e).__name__}. Run `ollama serve`."
    if r.status_code != 200:
        return False, f"Ollama at {host} returned {r.status_code}"
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if not any(n == model or n.startswith(model.split(":")[0] + ":") for n in names):
        return False, f"Model {model!r} not pulled. Run `ollama pull {model}`. Available: {names or '(none)'}"
    return True, f"Ollama OK at {host}; {model} available"
