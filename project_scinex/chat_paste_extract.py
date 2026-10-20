"""
chat_paste_extract.py
---------------------
Run the extraction through a **chat UI** (ChatGPT free, Gemini free, Claude.ai, Copilot…)
instead of an API, by batching the paragraphs into copy-pasteable blocks and reading the
replies back.

    batches   write output/<paper>/kg/<extractor>/<slug>/chat_batches/batch_NN.txt
              — paste one file per message into the chat
    collect   read the replies back (one .txt per batch, or one file with all of them)
              and write responses.jsonl, ready for `claude_extract.py ingest`

    python3 chat_paste_extract.py batches --all --model-slug gpt5-chat --size 12
    #  … paste each batch_NN.txt into the chat, save each reply as reply_NN.txt …
    python3 chat_paste_extract.py collect --all --model-slug gpt5-chat
    python3 claude_extract.py ingest --paper <id> --model-slug gpt5-chat \
            --responses output/<id>/kg/relation/gpt5-chat/responses.jsonl

Why batching. In a chat UI the ~3k-token system prompt is pasted ONCE per conversation and
then applies to every following message, so the per-paragraph cost drops to the paragraph
itself (~330 tokens). That is what makes a free tier viable: the binding limit is the
number of MESSAGES, not tokens.

⚠ METHODOLOGICAL CAVEAT — state this if you report the numbers.
The API/local runs send **one paragraph per call**; batching shows the model several at
once, so it can carry context between them. That is a different experimental condition,
not a reproduction of the qwen3:8b / claude-opus-5 runs. Keep `--size` small (8–12) and
say which mode produced which numbers. For a strict comparison use `--size 1`, or the API
path (`run_api_extract.py`).
"""

import argparse
import json
import re
import sys
from pathlib import Path

OUTPUT_DIR = Path("output")

HEADER = """You are continuing a structured extraction task. Apply the SYSTEM RULES exactly
as given; they were supplied at the start of this conversation and still apply.

Below are {n} numbered paragraphs. Treat EACH paragraph independently — do not carry facts
from one paragraph into another, and do not merge them.

Return ONE JSON object, no markdown fences, no commentary, in exactly this shape:

{{"results": [
  {{"para_id": <the id shown>, "triples": [ ... same triple objects as the SYSTEM RULES ... ]}},
  ...
]}}

Include every para_id below, even those whose triples list is empty.
"""

FIRST_BATCH_NOTE = """########## SYSTEM RULES — paste this whole block first, once per conversation ##########

{system}

########## END SYSTEM RULES ##########

"""


def batch_dir(paper, extractor_dir, slug):
    return OUTPUT_DIR / paper / "kg" / extractor_dir / slug / "chat_batches"


def cmd_batches(args):
    total_files = total_chars = 0
    for paper in args.papers:
        src = OUTPUT_DIR / paper / "kg" / args.extractor_dir / args.model_slug
        prompts_f, system_f = src / "prompts.jsonl", src / "system_prompt.txt"
        if not prompts_f.exists():
            print(f"  {paper}: no prompts.jsonl — run claude_extract.py prompts "
                  f"--paper {paper} --model-slug {args.model_slug}")
            continue
        system = system_f.read_text(encoding="utf-8")
        prompts = [json.loads(l) for l in open(prompts_f, encoding="utf-8") if l.strip()]

        dest = batch_dir(paper, args.extractor_dir, args.model_slug)
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("batch_*.txt"):
            old.unlink()

        chunks = [prompts[i:i + args.size] for i in range(0, len(prompts), args.size)]
        for bi, chunk in enumerate(chunks, 1):
            parts = []
            if bi == 1 or args.repeat_system:
                parts.append(FIRST_BATCH_NOTE.format(system=system))
            parts.append(HEADER.format(n=len(chunk)))
            for p in chunk:
                parts.append(f"\n----- para_id: {p['para_id']} -----\n"
                             f"Section: {p['section']}\n\nText:\n{p['paragraph']}\n")
            text = "".join(parts)
            f = dest / f"batch_{bi:02d}.txt"
            f.write_text(text, encoding="utf-8")
            total_files += 1
            total_chars += len(text)
        print(f"  {paper:18s} {len(prompts):3d} paragraphs → {len(chunks)} batch file(s) in {dest}")

    print(f"\n{total_files} batch files, ~{total_chars//4:,} tokens of pasting in total.")
    print(f"Paste batch_01.txt (it carries the SYSTEM RULES) into a NEW conversation, then the "
          f"rest as follow-up messages.")
    print(f"Start a fresh conversation every ~{args.fresh} batches and paste the SYSTEM RULES "
          f"again (use --repeat-system to put them in every file).")
    print(f"Save each reply as reply_NN.txt beside the batch files, then: "
          f"python3 chat_paste_extract.py collect --model-slug {args.model_slug} --all")


README = """HOW TO RUN THIS  ({slug})
=========================================================

There are {n} numbered files. Do them IN ORDER.

For each file:
  1. Open it, select all, copy.
  2. Paste it as ONE message into the chat.
  3. Copy the whole reply and save it in the "replies" folder next to this README,
     using the SAME name as the batch file but starting with "reply_":
        {ex_batch}   ->   replies/reply_{ex_rest}

Start a NEW CHAT whenever the file name changes to a different paper.
(Every file numbered *_batch01 carries the rules block, so each paper stands alone.)

If the chat says the message is too long, split the file in half and send it as two
messages - the model keeps the rules from the first part.

SHORT ON TIME? Files 01-21 are the six papers that matter most. Stopping there is fine.

When you are done, zip this whole folder (or just the replies folder) and send it back.
"""


def cmd_bundle(args):
    """Flatten every batch file into one folder, in paste order, ready to zip."""
    priority = ["vehiclemake2025", "parkingyolo2023", "csysguard2024",
                "oxidecrack2025", "osmotic2026", "pesticide2025"]
    ordered = [p for p in priority if p in args.papers] + \
              [p for p in args.papers if p not in priority]

    dest = Path(args.dest) / args.model_slug
    (dest / "replies").mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.txt"):
        old.unlink()

    n = 0
    first = None
    for paper in ordered:
        src = batch_dir(paper, args.extractor_dir, args.model_slug)
        for f in sorted(src.glob("batch_*.txt")):
            n += 1
            name = f"{n:02d}_{paper}_{f.stem}.txt"
            (dest / name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
            if first is None:
                first = name
    if not n:
        sys.exit("no batch files found — run `batches` first")

    (dest / "README.txt").write_text(
        README.format(slug=args.model_slug, n=n, ex_batch=first,
                      ex_rest=first), encoding="utf-8")
    chars = sum(len(p.read_text(encoding="utf-8")) for p in dest.glob("*.txt"))
    print(f"{n} files → {dest}  (~{chars//4:,} tokens of pasting)")
    print(f"Zip {dest} and go. Replies go in {dest/'replies'}/ named reply_<same name>.txt")


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def _extract_json(text):
    """Pull the JSON object out of a chat reply (fences, preamble, sign-off all tolerated)."""
    text = re.sub(r"```(?:json)?", "", text)
    m = _JSON_OBJ.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # A chat UI truncation usually breaks the tail; salvage whole result objects.
        out = []
        for obj in re.finditer(r'\{\s*"para_id"\s*:\s*(\d+).*?"triples"\s*:\s*(\[.*?\])\s*\}',
                               blob, re.S):
            try:
                out.append({"para_id": int(obj.group(1)),
                            "triples": json.loads(obj.group(2))})
            except json.JSONDecodeError:
                continue
        return {"results": out} if out else None


def _flat_replies(flat_dir):
    """{paper: [reply file, ...]} from a flat folder named NN_<paper>_batchNN.txt."""
    out = {}
    for f in sorted(Path(flat_dir).glob("*")):
        if f.suffix.lower() not in (".txt", ".json", ".md"):
            continue
        m = re.match(r"(?:reply[_-])?\d+[_-]([A-Za-z0-9.-]+?)[_-]batch[_-]?\d+", f.name, re.I)
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def cmd_collect(args):
    flat = _flat_replies(args.flat_dir) if args.flat_dir else {}
    if flat:
        args.papers = [p for p in args.papers if p in flat] or list(flat)
    grand_ok = grand_missing = 0
    for paper in args.papers:
        dest = batch_dir(paper, args.extractor_dir, args.model_slug)
        if not dest.exists():
            continue
        prompts_f = dest.parent / "prompts.jsonl"
        wanted = [json.loads(l)["para_id"] for l in open(prompts_f, encoding="utf-8") if l.strip()]

        replies = sorted(dest.glob("reply_*.txt")) + sorted(dest.glob("reply_*.json"))
        if flat:
            replies = flat.get(paper, [])
        if args.replies:
            replies = [Path(args.replies)]
        got = {}
        bad = []
        for rf in replies:
            data = _extract_json(rf.read_text(encoding="utf-8", errors="replace"))
            if not data:
                bad.append(rf.name)
                continue
            for r in data.get("results", []):
                try:
                    got[int(r["para_id"])] = json.dumps({"triples": r.get("triples", [])},
                                                        ensure_ascii=False)
                except (KeyError, TypeError, ValueError):
                    continue

        if not replies:
            print(f"  {paper}: no replies found")
            continue
        dest.mkdir(parents=True, exist_ok=True)

        out_f = dest.parent / "responses.jsonl"
        with open(out_f, "w", encoding="utf-8") as f:
            for pid in wanted:
                raw = got.get(pid)
                if raw is None:
                    continue
                f.write(json.dumps({"para_id": pid, "raw": raw}, ensure_ascii=False) + "\n")

        missing = [p for p in wanted if p not in got]
        grand_ok += len(wanted) - len(missing)
        grand_missing += len(missing)
        note = f"  ⚠ unreadable: {', '.join(bad)}" if bad else ""
        print(f"  {paper:18s} {len(wanted)-len(missing):3d}/{len(wanted)} paragraphs collected"
              f"{'  MISSING para_ids: ' + str(missing) if missing else ''}{note}")

    print(f"\n{grand_ok} collected, {grand_missing} missing.")
    if grand_missing:
        print("Re-paste the batches covering the missing ids; collect merges whatever is present.")
    print(f"Next: python3 claude_extract.py ingest --paper <id> --model-slug {args.model_slug} "
          f"--responses output/<id>/kg/{args.extractor_dir}/{args.model_slug}/responses.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("batches", "bundle", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--paper", nargs="+")
        p.add_argument("--all", action="store_true")
        p.add_argument("--model-slug", required=True)
        p.add_argument("--extractor", choices=["relation", "fixed"], default="relation")
        p.add_argument("--ontology", choices=["ceo", "scinex"], default="ceo")
        if name == "batches":
            p.add_argument("--size", type=int, default=12,
                           help="paragraphs per message (default 12; use 1 to match the API run)")
            p.add_argument("--repeat-system", action="store_true",
                           help="put the SYSTEM RULES in every batch file")
            p.add_argument("--fresh", type=int, default=4,
                           help="suggested batches per conversation before starting a new one")
        elif name == "bundle":
            p.add_argument("--dest", default="chat_upload",
                           help="where to write the flat, zip-ready folder")
        else:
            p.add_argument("--replies", default=None,
                           help="one file holding every reply, instead of reply_NN.txt files")
            p.add_argument("--flat-dir", default=None,
                           help="folder of replies named like the bundled batch files")

    args = ap.parse_args()
    if ":" in args.model_slug or "/" in args.model_slug:
        sys.exit("--model-slug must not contain ':' or '/' — it becomes a directory name")
    args.extractor_dir = args.extractor if args.ontology == "ceo" else f"{args.extractor}_scinex"

    if args.all:
        args.papers = sorted(p.name for p in OUTPUT_DIR.iterdir()
                             if (p / "no-llm" / "output.html").exists())
    elif args.paper:
        args.papers = args.paper
    else:
        sys.exit("give --paper <id> ... or --all")

    {"batches": cmd_batches, "bundle": cmd_bundle, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
