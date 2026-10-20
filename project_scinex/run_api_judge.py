#!/usr/bin/env python3
"""
run_api_judge.py
----------------
Drive `judge_paste.py`'s judging bundles with an API model instead of a chat UI.

    gold_eval.py export → judge_paste.py batches → run_api_judge.py → judge_paste.py collect
                                                    (this script)

The chat-UI route needs a human to paste 10 files and save 10 replies. This sends the same
batches over HTTP and writes the replies where `judge_paste.py collect` already looks, so
nothing downstream changes.

One difference from the chat route, and it is an improvement: a chat conversation carries the
JUDGING RULES from the first message, which drifts as the context grows. Here the rubric is
sent as the SYSTEM PROMPT on every call, so every batch is judged under identical instructions.

    python3 run_api_judge.py --slug gemini-judge --provider gemini --model gemini-2.5-flash
    python3 judge_paste.py collect --csv gold/claude_sample_for_judge.csv \
            --slug gemini-judge --out gold/gemini_verdicts.csv

⚠ A model must never judge its own extraction. Judge claude-opus-5 triples with Gemini or
GPT — never with Claude.
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from run_api_extract import _load_dotenv, call  # noqa: E402
from judge_paste import RUBRIC  # noqa: E402
from ontology_eval import (  # noqa: E402
    RUBRIC_TRIPLES, RUBRIC_PARAGRAPHS, RUBRIC_PAPERS,
)

BASE_URLS = {"groq": "https://api.groq.com/openai/v1"}

# Which rubric goes in the system prompt. The old CORRECT/PARTIAL/INCORRECT rubric is one
# text; C1-C6 is THREE, because the criteria do not share a unit of analysis (C1/C3/C4 per
# triple, C2/C5 per paragraph, C6 per paper). Selecting the wrong one silently produces
# verdict keys `collect` cannot read, so the choice is explicit rather than inferred.
RUBRICS = {
    "gold": RUBRIC,                  # legacy: CORRECT / PARTIAL / INCORRECT
    "triples": RUBRIC_TRIPLES,       # C1, C3, C4
    "paragraphs": RUBRIC_PARAGRAPHS,  # C2, C5
    "papers": RUBRIC_PAPERS,         # C6
}


def strip_rubric(text):
    """Batch 01 carries the rubric inline; it is the system prompt here instead."""
    marker = "Judge the following"
    i = text.find(marker)
    return text[i:] if i > 0 else text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True, help="bundle under --dest, e.g. gemini-judge")
    ap.add_argument("--dest", default="judge_upload")
    ap.add_argument("--provider", required=True,
                    choices=["anthropic", "openai", "openai-compat", "gemini", "ollama"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="verdicts + reasons for a whole batch; reasoning models need headroom")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between batches (rate limits)")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rubric", choices=sorted(RUBRICS), default="gold",
                    help="which rubric to send as the system prompt. 'gold' is the legacy "
                         "CORRECT/PARTIAL/INCORRECT set; triples/paragraphs/papers are the "
                         "C1-C6 harnesses and MUST match the ontology_eval bundle being judged")
    args = ap.parse_args()
    _load_dotenv()

    if args.base_url is None:
        args.base_url = BASE_URLS.get(args.provider)

    src = Path(args.dest) / args.slug
    if not src.exists():
        sys.exit(f"no bundle at {src} — run judge_paste.py batches first")
    # only the batch files — the bundle also holds a README and an instructions file
    batches = sorted(p for p in src.glob("*.txt")
                     if p.is_file() and "batch" in p.name.lower())
    if not batches:
        sys.exit(f"no batch .txt files in {src}")
    out_dir = src / "replies"
    out_dir.mkdir(exist_ok=True)

    todo = []
    for b in batches:
        dest = out_dir / f"reply_{b.name}"
        if dest.exists() and dest.stat().st_size > 0 and args.resume:
            continue
        todo.append((b, dest))
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print(f"nothing to do — {len(batches)} replies already present in {out_dir}")
        return

    rubric = RUBRICS[args.rubric]
    print(f"{args.provider}:{args.model} judging {len(todo)} of {len(batches)} batches "
          f"in {src}  [rubric: {args.rubric}]")
    t0 = time.time()
    ok = fail = 0
    for i, (b, dest) in enumerate(todo, 1):
        user = strip_rubric(b.read_text(encoding="utf-8"))
        try:
            raw = call(args, rubric, user)
            dest.write_text(raw, encoding="utf-8")
            ok += 1
            print(f"  [{i}/{len(todo)}] {b.name} → {len(raw)} chars", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(todo)}] {b.name} FAILED — {e}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"\n{ok} judged, {fail} failed, {(time.time()-t0)/60:.1f} min → {out_dir}")
    if args.rubric == "gold":
        print(f"Next: python3 judge_paste.py collect --csv <the gold csv> --slug {args.slug} "
              f"--out gold/{args.slug.replace('-judge','')}_verdicts.csv")
    else:
        print(f"Next: python3 ontology_eval.py collect --model <slug> --kind {args.rubric}")


if __name__ == "__main__":
    main()
