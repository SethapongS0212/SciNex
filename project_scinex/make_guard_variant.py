#!/usr/bin/env python3
"""
make_guard_variant.py
---------------------
Re-ingest STORED model replies under another guard version, into a sibling run folder, so the
old KG stays untouched and both can be compared (Session 39, guards v2 — see
kg_extraction/fixed_extractor.py `guards_v2_reject`). No model is run.

For every paper x ontology where output/<paper>/kg/<exdir>/<slug>/responses.jsonl exists:
  copy prompts.jsonl, responses.jsonl, system_prompt.txt -> <slug><suffix>/
  prompt_meta.json += {"guards": <version>, "guards_source_slug": <slug>}, model_slug renamed
  run `claude_extract.py ingest` in process (same entity list as the source run: its
  prompt_meta "entity_dir", else the default ACL lists) -> triples.json, kg.graphml

Every tool that re-parses stored replies (ft_compare, ft_overlap, build_ft_data, blind_judge)
reads "guards" from prompt_meta.json, so `<slug>-g2` is scored under v2 everywhere.

    python make_guard_variant.py --slugs gptoss-20b qwen3-30b-a3b-v2 \
        --papers-from ft_data/acl/fold0/eval_papers.txt --guards v2 --suffix=-g2
"""
import argparse
import io
import json
import logging
import shutil
import sys
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (ENTITY_DIR_DEFAULT, OUTPUT_DIR, SCINEX_OWL_DEFAULT, entity_csv_for,  # noqa: E402
                       exdir_for, quiet_guard_logs)
from kg_extraction.fixed_extractor import GUARD_VERSIONS  # noqa: E402

COPY = ("prompts.jsonl", "responses.jsonl", "system_prompt.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slugs", nargs="+", required=True)
    ap.add_argument("--papers", nargs="+")
    ap.add_argument("--papers-from")
    ap.add_argument("--ontologies", nargs="+", choices=["ceo", "scinex"], default=["ceo", "scinex"])
    ap.add_argument("--guards", choices=GUARD_VERSIONS, default="v2")
    ap.add_argument("--suffix", default="-g2")
    ap.add_argument("--force", action="store_true", help="rebuild folders that already exist")
    args = ap.parse_args()
    papers = args.papers or Path(args.papers_from).read_text(encoding="utf-8").split()
    if not args.suffix or "/" in args.suffix or ":" in args.suffix:
        sys.exit("--suffix must be a plain, non-empty name part")

    import claude_extract                      # after sys.path; brings the KG builder
    quiet_guard_logs()
    logging.getLogger().setLevel(logging.WARNING)
    done = skipped = missing = 0
    for slug in args.slugs:
        if slug.endswith(args.suffix):
            sys.exit(f"{slug} already carries the suffix {args.suffix}")
        new = slug + args.suffix
        for ont in args.ontologies:
            exdir = exdir_for("fixed", ont)
            for paper in papers:
                src = OUTPUT_DIR / paper / "kg" / exdir / slug
                if not (src / "responses.jsonl").exists():
                    missing += 1
                    continue
                dst = OUTPUT_DIR / paper / "kg" / exdir / new
                if (dst / "triples.json").exists() and not args.force:
                    skipped += 1
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                for name in COPY:
                    if (src / name).exists():
                        shutil.copy2(src / name, dst / name)
                meta = json.loads((src / "prompt_meta.json").read_text(encoding="utf-8"))
                meta.update(model_slug=new, guards=args.guards, guards_source_slug=slug)
                (dst / "prompt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                ent_dir = meta.get("entity_dir") or str(ENTITY_DIR_DEFAULT)
                ns = Namespace(paper=paper, extractor="fixed", ontology=ont,
                               ontology_file=meta.get("ontology_file") or SCINEX_OWL_DEFAULT,
                               entity_csv=entity_csv_for(paper, ent_dir), model_slug=new,
                               responses=dst / "responses.jsonl")
                with redirect_stdout(io.StringIO()):
                    claude_extract.cmd_ingest(ns)
                done += 1
        print(f"{slug} -> {new}: guards {args.guards}", flush=True)
    print(f"ingested {done}, already there {skipped}, no source run {missing}")


if __name__ == "__main__":
    main()
