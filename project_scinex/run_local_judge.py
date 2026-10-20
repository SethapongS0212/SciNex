#!/usr/bin/env python3
"""
run_local_judge.py — LLM-as-Judge (kg_evaluate.py) via Ollama on this machine.

Same reason as run_local.py: torch will not load here (Smart App Control), so the
judge's 4-bit transformers path is unavailable. This points it at Ollama instead and
then runs kg_evaluate unchanged — same prompt, same rubric, same ontology definitions,
same output files.

    python run_local_judge.py --check
    python run_local_judge.py --extractor relation --max-per-paper 20
    python run_local_judge.py --extractor relation relation_scinex --resume

⚠ Judge model defaults to qwen2.5:7b-instruct, a DIFFERENT family from the extractor's
qwen3:8b. A model grading its own output is not an independent check. Pull it first:
    ollama pull qwen2.5:7b-instruct

⚠ These verdicts are PRELIMINARY — small local models on both sides. The judge's numbers
are only quotable after Cohen's κ against human labels (gold_eval.py), and the headline
figures for the paper still come from the VM.
"""

import argparse
import logging
import sys

from kg_extraction.ollama_backend import (
    DEFAULT_HOST, DEFAULT_JUDGE_MODEL, check_server, patch_judge,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL,
                    help=f"Ollama judge model (default {DEFAULT_JUDGE_MODEL})")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--extractor", nargs="+", default=["relation"],
                    help="extractor families to judge, e.g. relation relation_scinex")
    # kg_evaluate's --paper takes ONE name and is mutually exclusive with --all.
    ap.add_argument("--paper", help="limit to a single paper (default: all papers)")
    ap.add_argument("--max-per-paper", type=int, default=None,
                    help="judge only N triples per paper (deterministic subsample)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse verdicts already in evaluation.json")
    ap.add_argument("--summary-out", default="output/eval/judge_corpus.json")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--check", action="store_true", help="check the server and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    ok, msg = check_server(args.model, args.host)
    print(("OK  " if ok else "!!  ") + msg)
    if args.check:
        return 0 if ok else 1
    if not ok:
        return 1

    patch_judge(model=args.model, host=args.host, num_ctx=args.num_ctx)

    import kg_evaluate

    # Colon is illegal in Windows paths; see run_local.py.
    model_slug = args.model.replace(":", "-").replace("/", "-")
    argv = ["kg_evaluate.py", "--extractor", *args.extractor,
            "--model", model_slug, "--summary-out", args.summary_out]
    argv += ["--paper", args.paper] if args.paper else ["--all"]
    if args.max_per_paper:
        argv += ["--max-per-paper", str(args.max_per_paper)]
    if args.resume:
        argv += ["--resume"]

    saved, sys.argv = sys.argv, argv
    try:
        return kg_evaluate.main() or 0
    except SystemExit as e:
        return e.code or 0
    finally:
        sys.argv = saved


if __name__ == "__main__":
    sys.exit(main())
