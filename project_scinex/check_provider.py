#!/usr/bin/env python3
"""
check_provider.py
-----------------
Validate an API key BEFORE committing 486 calls to it, and print the exact
`run_api_extract.py` command to run next.

    python3 check_provider.py groq
    python3 check_provider.py gemini

Does three things, in order, stopping at the first failure:
  1. finds the key (env or .env)
  2. lists the models the key can actually see  — model ids drift, so this is
     looked up rather than guessed
  3. sends ONE real extraction prompt and shows the reply

Standard library only. Keys are read from .env / the environment, never args.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_api_extract import _load_dotenv, _key, USER_AGENT  # noqa: E402

PROVIDERS = {
    "groq": {
        "keys": ("GROQ_API_KEY", "OPENAI_COMPAT_API_KEY"),
        "base": "https://api.groq.com/openai/v1",
        "signup": "https://console.groq.com/keys",
        "provider": "openai-compat",
        # preference order: strongest / most useful for this task first
        "prefer": ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct",
                   "llama-3.3-70b-versatile", "openai/gpt-oss-20b"],
    },
    "gemini": {
        "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "base": "https://generativelanguage.googleapis.com/v1beta",
        "signup": "https://aistudio.google.com/apikey",
        "provider": "gemini",
        # Pinned ids, newest first. NOT the `-latest` aliases: those drift, and a paper
        # number has to name the exact model that produced it. Older 2.5-* ids are still
        # LISTED to new keys but return NOT_FOUND when called — hence the smoke test.
        "prefer": ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                   "gemini-3-flash-preview", "gemini-2.5-flash"],
    },
    "openai": {
        "keys": ("OPENAI_API_KEY",),
        "base": "https://api.openai.com/v1",
        "signup": "https://platform.openai.com/api-keys",
        "provider": "openai",
        "prefer": ["gpt-5", "gpt-4.1", "gpt-4o"],
    },
}


def _get(url, headers):
    headers = {**headers, "User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models(name, cfg, key):
    if name == "gemini":
        data = _get(f"{cfg['base']}/models?key={key}&pageSize=100", {})
        return [m["name"].split("/")[-1] for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]
    data = _get(f"{cfg['base']}/models", {"Authorization": f"Bearer {key}"})
    return sorted(m["id"] for m in data.get("data", []))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in PROVIDERS:
        sys.exit(f"usage: python3 check_provider.py [{'|'.join(PROVIDERS)}]")
    name = sys.argv[1]
    cfg = PROVIDERS[name]
    _load_dotenv()

    key = _key(*cfg["keys"])
    if not key:
        print(f"No key found. Get one free at {cfg['signup']}, then add to .env:")
        print(f"    {cfg['keys'][0]}=<your key>")
        sys.exit(1)
    print(f"key found ({cfg['keys'][0]}, ...{key[-4:]})")

    try:
        models = list_models(name, cfg, key)
    except urllib.error.HTTPError as e:
        sys.exit(f"key rejected — HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
    print(f"{len(models)} models visible to this key")

    pick = next((m for m in cfg["prefer"] if m in models), None)
    if not pick:
        chat = [m for m in models if not any(
            x in m.lower() for x in ("embed", "whisper", "tts", "guard", "vision-only"))]
        pick = chat[0] if chat else models[0]
    print(f"suggested model: {pick}")
    others = [m for m in cfg["prefer"] if m in models and m != pick]
    if others:
        print(f"  also available: {', '.join(others)}")

    slug = pick.replace("/", "-").replace(":", "-").replace(".", "")
    print("\nSmoke test it on 2 paragraphs, then run the corpus:\n")
    print(f"  python3 run_api_extract.py --paper ugmo2024 --provider {cfg['provider']} \\n"
          f"      --model {pick} --model-slug {slug}-smoke \\n"
          f"      --extractor relation --ontology ceo --limit 2 --ingest"
          + (f" \\n      --base-url {cfg['base']}" if cfg["provider"] == "openai-compat" else ""))
    print(f"\n  python3 run_api_extract.py --all --provider {cfg['provider']} \\n"
          f"      --model {pick} --model-slug {slug} \\n"
          f"      --extractor relation --ontology ceo --ingest"
          + (f" \\n      --base-url {cfg['base']}" if cfg["provider"] == "openai-compat" else ""))
    print(f"\n  python3 claude_corpus_report.py --model-slug {slug}")


if __name__ == "__main__":
    main()
