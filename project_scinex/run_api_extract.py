"""
run_api_extract.py
------------------
Drive `claude_extract.py`'s prompt/response protocol with ANY model — Anthropic, OpenAI,
Gemini, Ollama, or any OpenAI-compatible endpoint (OpenRouter, Together, DeepSeek, vLLM,
LM Studio…). This is the missing middle step that lets the same experiment be replicated
across models.

    claude_extract.py prompts   →   run_api_extract.py   →   claude_extract.py ingest
      (renders the real prompts)     (calls the model)        (same guards + KG builder)

Because the prompt, the post-parse guards and the graph builder are all shared code, the
ONLY variable between two runs is the model. That is what makes the comparison meaningful.

Full loop for one model:

    python3 claude_extract.py prompts --paper ugmo2024 --model-slug gpt-5
    python3 run_api_extract.py --paper ugmo2024 --model-slug gpt-5 \
            --provider openai --model gpt-5
    python3 claude_extract.py ingest  --paper ugmo2024 --model-slug gpt-5 \
            --responses output/ugmo2024/kg/relation/gpt-5/responses.jsonl

…or every paper at once with `--all` (it renders the prompts itself if they are missing).
Then compare:  python3 claude_corpus_report.py --model-slug gpt-5

API keys come from the environment, never the command line:
    ANTHROPIC_API_KEY   OPENAI_API_KEY   GEMINI_API_KEY   (Ollama needs none)

Only the standard library is used, so nothing has to be installed to run this.

⚠ Two rules that carry over from the local runs:
  * `--model-slug` must not contain ':' or '/' — it becomes a Windows directory name.
    (qwen3:8b silently discarded every triple at write time until this was caught.)
  * A model must never judge its own extraction. Extract with one family, judge with another.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("output")

# Some providers sit behind Cloudflare, which blocks the default `Python-urllib/3.x`
# user agent outright (HTTP 403, "error code: 1010"). Send a real one everywhere.
USER_AGENT = "scinex-kg-extract/1.0 (+research pipeline)"


def _load_dotenv(path=".env"):
    """Read KEY=VALUE lines from .env into the environment (stdlib only).

    Keeps the several provider keys out of the shell history and lets a Groq key
    and a Gemini key sit side by side in one file.
    """
    f = Path(path)
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _key(*names):
    """First env var that is set, in preference order."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None

# ── provider adapters ────────────────────────────────────────────────────────
# Each returns (url, headers, body). Each parser pulls the assistant text out of
# that provider's response envelope. Add a provider by adding one pair here.


def _anthropic(model, system, user, max_tokens):
    key = _key("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set")
    return (
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01",
         "content-type": "application/json", "User-Agent": USER_AGENT},
        {"model": model, "max_tokens": max_tokens, "temperature": 0,
         "system": system, "messages": [{"role": "user", "content": user}]},
    )


def _anthropic_text(data):
    return "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")


# Which env var holds the key for which OpenAI-compatible host. Resolving by
# ENDPOINT rather than by a fixed preference order matters as soon as more than one
# provider key is present: a plain ordered lookup happily sends the Groq key to
# OpenRouter, which answers 401 "Missing Authentication header".
_HOST_KEYS = (
    ("openrouter.ai", "OPENROUTER_API_KEY"),
    ("groq.com",      "GROQ_API_KEY"),
    ("together",      "TOGETHER_API_KEY"),
    ("deepseek",      "DEEPSEEK_API_KEY"),
    ("mistral",       "MISTRAL_API_KEY"),
)


def _openai(model, system, user, max_tokens, base=None):
    base = base or "https://api.openai.com/v1"
    host_var = next((var for frag, var in _HOST_KEYS if frag in base.lower()), None)
    # the host's own var first, then the generic overrides, then plain OpenAI
    names = ([host_var] if host_var else []) + ["OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY"]
    key = _key(*names) or "sk-noauth"
    return (
        f"{base.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {key}", "content-type": "application/json",
         "User-Agent": USER_AGENT},
        {"model": model, "temperature": 0, "max_tokens": max_tokens,
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]},
    )


def _openai_text(data):
    return data["choices"][0]["message"]["content"] or ""


def _gemini(model, system, user, max_tokens):
    key = _key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        {"content-type": "application/json", "User-Agent": USER_AGENT},
        {"system_instruction": {"parts": [{"text": system}]},
         "contents": [{"role": "user", "parts": [{"text": user}]}],
         "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}},
    )


def _gemini_text(data):
    parts = data["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _ollama(model, system, user, max_tokens, host="http://localhost:11434", num_ctx=8192):
    return (
        f"{host.rstrip('/')}/api/chat",
        {"content-type": "application/json", "User-Agent": USER_AGENT},
        # "think": False matches kg_extraction/ollama_backend.py, and is load-bearing for
        # any reasoning model. Ollama returns reasoning in a SEPARATE `thinking` field; with
        # thinking on, a Qwen3 reply can spend the whole `num_predict` budget reasoning and
        # come back with content == "" — which is indistinguishable from "found nothing".
        # Measured: 93 of 515 calls (18%) returned empty this way before it was set.
        {"model": model, "stream": False, "think": False,
         "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": max_tokens},
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]},
    )


def _ollama_text(data):
    msg = data.get("message", {}) or {}
    content = msg.get("content") or ""
    if content.strip():
        return content
    # Fallback for builds that ignore "think": False and answer inside `thinking`.
    # The extractor's _strip_think_and_fences() handles the tags, so hand it back wrapped
    # rather than dropping a reply that may still carry usable JSON.
    thinking = msg.get("thinking") or ""
    return f"<think>{thinking}</think>" if thinking.strip() else ""


PARSERS = {
    "anthropic": _anthropic_text,
    "openai": _openai_text,
    "openai-compat": _openai_text,
    "gemini": _gemini_text,
    "ollama": _ollama_text,
}


def build_request(args, system, user):
    if args.provider == "anthropic":
        return _anthropic(args.model, system, user, args.max_tokens)
    if args.provider == "openai":
        return _openai(args.model, system, user, args.max_tokens)
    if args.provider == "openai-compat":
        return _openai(args.model, system, user, args.max_tokens, base=args.base_url)
    if args.provider == "gemini":
        return _gemini(args.model, system, user, args.max_tokens)
    if args.provider == "ollama":
        return _ollama(args.model, system, user, args.max_tokens,
                       host=args.base_url or "http://localhost:11434",
                       num_ctx=args.num_ctx)
    sys.exit(f"unknown provider {args.provider}")


def call(args, system, user):
    """One request, with retries. Returns the assistant's raw text."""
    last = None
    for attempt in range(args.retries + 1):
        url, headers, body = build_request(args, system, user)
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return PARSERS[args.provider](data)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {detail}"
            # Reasoning-tier models reject temperature / max_tokens. Retrying without
            # them is cheaper than making the caller guess which family they are on.
            if e.code == 400 and re.search(r"temperature|max_tokens", detail, re.I):
                args.max_tokens = None if "max_tokens" in detail else args.max_tokens
                _strip_unsupported(args, detail)
                continue
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as e:                       # timeouts, connection resets
            last = f"{type(e).__name__}: {e}"
        if attempt < args.retries:
            time.sleep(args.backoff * (2 ** attempt))
    raise RuntimeError(last or "request failed")


_STRIPPED = set()


def _strip_unsupported(args, detail):
    """Remember which params this model rejects so later calls omit them."""
    for p in ("temperature", "max_tokens"):
        if p in detail.lower():
            _STRIPPED.add(p)
    global build_request
    inner = build_request

    def wrapper(a, s, u, _inner=inner):
        url, headers, body = _inner(a, s, u)
        for p in _STRIPPED:
            body.pop(p, None)
        return url, headers, body

    build_request = wrapper


# ── driving one paper ────────────────────────────────────────────────────────

def run_paper(args, paper):
    ex_dir = OUTPUT_DIR / paper / "kg" / args.extractor_dir / args.model_slug
    prompts_f = ex_dir / "prompts.jsonl"
    system_f = ex_dir / "system_prompt.txt"
    if not prompts_f.exists() or not system_f.exists():
        print(f"  {paper}: no prompts — run claude_extract.py prompts first")
        return 0, 0

    system = system_f.read_text(encoding="utf-8")
    prompts = [json.loads(l) for l in open(prompts_f, encoding="utf-8") if l.strip()]

    out_f = ex_dir / "responses.jsonl"
    done = set()
    if out_f.exists() and args.resume:
        for line in open(out_f, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["para_id"])

    todo = [p for p in prompts if p["para_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print(f"  {paper}: nothing to do ({len(done)}/{len(prompts)} already answered)")
        return 0, 0

    ok = fail = 0
    mode = "a" if (done and args.resume) else "w"
    with open(out_f, mode, encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            try:
                raw = call(args, system, p["user_prompt"])
                ok += 1
            except Exception as e:
                print(f"    para {p['para_id']}: FAILED — {e}")
                raw = '{"triples": []}'
                fail += 1
            f.write(json.dumps({"para_id": p["para_id"], "raw": raw},
                               ensure_ascii=False) + "\n")
            f.flush()
            if args.verbose or i % 10 == 0 or i == len(todo):
                print(f"    {paper}: {i}/{len(todo)} paragraphs", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    print(f"  {paper}: {ok} answered, {fail} failed → {out_f}")
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", nargs="+", help="paper id(s); omit with --all")
    ap.add_argument("--all", action="store_true", help="every parsed paper")
    ap.add_argument("--provider", required=True,
                    choices=["anthropic", "openai", "openai-compat", "gemini", "ollama"])
    ap.add_argument("--model", required=True, help="provider's model id, e.g. gpt-5")
    ap.add_argument("--model-slug", required=True,
                    help="output directory name; no ':' or '/' (Windows path)")
    ap.add_argument("--base-url", default=None,
                    help="openai-compat / ollama endpoint override")
    ap.add_argument("--extractor", choices=["relation", "fixed"], default="relation")
    ap.add_argument("--ontology", choices=["ceo", "scinex"], default="ceo")
    ap.add_argument("--ontology-file", default="scinex_refined_14.owl")
    ap.add_argument("--entity-csv", default=None, help="required for --extractor fixed")
    ap.add_argument("--max-tokens", type=int, default=3072,
                    help="1024 truncated 34%% of local calls; a truncated JSON yields ZERO triples")
    ap.add_argument("--num-ctx", type=int, default=8192, help="ollama only")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between calls (rate limits)")
    ap.add_argument("--limit", type=int, default=None, help="only N paragraphs per paper (smoke test)")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--render", action="store_true", default=True,
                    help="render prompts first if missing (default)")
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--ingest", action="store_true",
                    help="run claude_extract.py ingest afterwards")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    _load_dotenv()

    if ":" in args.model_slug or "/" in args.model_slug:
        sys.exit("--model-slug must not contain ':' or '/' — it becomes a directory name")

    args.extractor_dir = args.extractor if args.ontology == "ceo" else f"{args.extractor}_scinex"

    if args.all:
        papers = sorted(p.name for p in OUTPUT_DIR.iterdir()
                        if (p / "no-llm" / "output.html").exists())
    elif args.paper:
        papers = args.paper
    else:
        sys.exit("give --paper <id> ... or --all")

    import subprocess
    common = ["--extractor", args.extractor, "--ontology", args.ontology,
              "--ontology-file", args.ontology_file, "--model-slug", args.model_slug]
    if args.entity_csv:
        common += ["--entity-csv", args.entity_csv]

    print(f"{args.provider}:{args.model} → kg/{args.extractor_dir}/{args.model_slug}/ "
          f"over {len(papers)} paper(s)")
    t0 = time.time()
    tot_ok = tot_fail = 0
    for paper in papers:
        ex_dir = OUTPUT_DIR / paper / "kg" / args.extractor_dir / args.model_slug
        if args.render and not (ex_dir / "prompts.jsonl").exists():
            subprocess.run([sys.executable, "claude_extract.py", "prompts",
                            "--paper", paper] + common,
                           check=False, stdout=subprocess.DEVNULL)
        ok, fail = run_paper(args, paper)
        tot_ok += ok
        tot_fail += fail
        if args.ingest and ok:
            subprocess.run([sys.executable, "claude_extract.py", "ingest",
                            "--paper", paper, "--responses",
                            str(ex_dir / "responses.jsonl")] + common, check=False)

    mins = (time.time() - t0) / 60
    print(f"\n{tot_ok} paragraphs answered, {tot_fail} failed, {mins:.1f} min")
    if not args.ingest:
        print("Next: python3 claude_extract.py ingest --paper <id> "
              f"--model-slug {args.model_slug} --responses output/<id>/kg/"
              f"{args.extractor_dir}/{args.model_slug}/responses.jsonl")
    print(f"Then:  python3 claude_corpus_report.py --model-slug {args.model_slug}")


if __name__ == "__main__":
    main()
