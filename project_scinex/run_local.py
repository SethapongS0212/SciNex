#!/usr/bin/env python3
"""
run_local.py — run the fixed / relation extractors on THIS machine via Ollama.

The Windows laptop cannot load torch (Smart App Control blocks its unsigned DLLs) and
its GPU is 8GB, so the normal `kg_main.py --model Qwen/Qwen3-14B` path is unavailable.
This wrapper points the extractors at a local Ollama server and then runs the ordinary
kg_main pipeline, so the HTML parsing, prompts, ontology, guards, kg_builder and output
layout are all unchanged.

    python run_local.py --check
    python run_local.py --paper strabismus2026 --extractor relation
    python run_local.py --all --extractor relation --ontology scinex
    python run_local.py --all --extractor relation --also-fixed     # both, both ontologies

⚠ Output goes to the SAME place as a real run — output/<paper>/kg/<extractor>/<model>/ —
but under an Ollama model slug, so it can't be confused with Qwen3-14B results.
These numbers are PRELIMINARY: a different, smaller model than the project standard.
"""

import argparse
import logging
import sys

from kg_extraction.ollama_backend import (
    DEFAULT_HOST, DEFAULT_MODEL, check_server, patch_extractors,
)

CORPUS = [
    "strabismus2026",
    "osmotic2026",
    "oxidecrack2025",
    "parkingyolo2023",
    "traveltime2022",
    "roadwaylight2018",
    "gamlprop2025",
    "vehiclemake2025",
    "csysguard2024",
    "reststop2018",
    "pesticide2025",
    "llamacorrupt2025",
    "videoseg2025",
    "ugmo2024",
    "eyelandmark2024",
    "textaug2023",
    "trafficspeed2021",
    "microwave2018",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default {DEFAULT_HOST})")
    ap.add_argument("--paper", nargs="+", help="paper id(s); omit with --all")
    ap.add_argument("--all", action="store_true", help="every paper in the corpus (18 parsed)")
    ap.add_argument("--extractor", default="relation", choices=["relation", "fixed"])
    ap.add_argument("--ontology", default="ceo", choices=["ceo", "scinex"])
    ap.add_argument("--also-fixed", action="store_true",
                    help="after the chosen extractor, also run the other one (for the comparison)")
    ap.add_argument("--both-ontologies", action="store_true", help="run ceo AND scinex")
    ap.add_argument("--max-new-tokens", type=int, default=3072,
                    help="Output cap. MEASURED: each triple echoes its source_sentence "
                         "verbatim, so a triple off an abstract costs ~500 tokens, not the "
                         "~80 a bare triple would. At 1024, 34%% of calls truncated and a "
                         "truncated JSON parses to ZERO triples - the call is wasted "
                         "entirely. 3072 fits ~6 long triples and still leaves headroom "
                         "under num_ctx 8192 alongside the ~3.4k-token prompt.")
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="Ollama context window. Ollama's own default of 4096 is too small "
                         "for the ~3.4k-token system prompt; 16384 overflows an 8GB GPU and "
                         "spills to CPU. 8192 measured 6.2GB / 100%% GPU.")
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

    papers = CORPUS if args.all else (args.paper or [])
    if not papers:
        ap.error("give --paper <id> or --all")

    extractors = [args.extractor]
    if args.also_fixed:
        extractors.append("fixed" if args.extractor == "relation" else "relation")
    ontologies = ["ceo", "scinex"] if args.both_ontologies else [args.ontology]

    patch_extractors(model=args.model, host=args.host,
                     max_new_tokens=args.max_new_tokens, num_ctx=args.num_ctx)

    # Ollama model names contain a colon ("qwen3:8b"), which is ILLEGAL in a
    # Windows path — kg_main builds output/<paper>/kg/<extractor>/<model>/ from
    # this string, so passing the raw name extracts every triple successfully and
    # then dies with NotADirectoryError on write. kg_main only uses --model for
    # the output path and logging; the patched _load ignores it and talks to the
    # real Ollama name, so a slug here is safe.
    model_slug = args.model.replace(":", "-").replace("/", "-")
    if model_slug != args.model:
        print(f"    (output dir uses model slug {model_slug!r})")

    import kg_main

    total = len(papers) * len(extractors) * len(ontologies)
    done = failed = 0
    results = []
    for paper in papers:
        for ext in extractors:
            for onto in ontologies:
                done += 1
                print(f"\n=== [{done}/{total}] {paper} | {ext} | {onto} ===", flush=True)
                argv = ["kg_main.py", "--paper", paper, "--extractor", ext,
                        "--model", model_slug, "--max-new-tokens", str(args.max_new_tokens)]
                if onto == "scinex":
                    argv += ["--ontology", "scinex"]
                saved = sys.argv
                try:
                    sys.argv = argv
                    kg_main.main()
                    results.append((paper, ext, onto, "ok"))
                except SystemExit as e:          # argparse/normal exit inside kg_main
                    results.append((paper, ext, onto, f"exit {e.code}"))
                    failed += bool(e.code)
                except Exception as e:
                    # One paper failing must not abandon the batch.
                    print(f"    !! {type(e).__name__}: {e}", flush=True)
                    results.append((paper, ext, onto, f"FAILED {type(e).__name__}"))
                    failed += 1
                finally:
                    sys.argv = saved

    print("\n=== summary ===")
    for paper, ext, onto, status in results:
        print(f"  {paper:18} {ext:9} {onto:7} {status}")
    print(f"\n{done - failed}/{done} runs ok")
    print("\nTriple counts:  python count_triples.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
