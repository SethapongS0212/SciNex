"""
judge_paste.py
--------------
Have ANOTHER model judge a set of triples through a chat UI, and read the verdicts back
into the CSV format `gold_report.py` / `gold_eval.py agree` already understand.

    batches   turn a gold CSV (from `gold_eval.py export`) into copy-pasteable
              judge_batches/batch_NN.txt, and flatten them into a zip-ready folder
    collect   read the replies back and write a filled label CSV

    python3 gold_eval.py export --extractor relation --model claude-opus-5 \
            --n 100 --seed 11 --out gold/claude_sample_for_judge.csv
    python3 judge_paste.py batches --csv gold/claude_sample_for_judge.csv \
            --slug gpt-judge --size 10
    #  … paste each file, save each reply as reply_<same name>.txt …
    python3 judge_paste.py collect --csv gold/claude_sample_for_judge.csv \
            --slug gpt-judge --out gold/gpt_verdicts.csv
    python3 gold_report.py --labels gold/gpt_verdicts.csv --model claude-opus-5

⚠ A model must never judge its own extraction. Judge claude-opus-5 triples with GPT or
Gemini, never with Claude.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

FIELDS = ["triple_id", "paper", "extractor", "model", "subject", "predicate", "object",
          "object_type", "predicate_definition", "section", "source_sentence",
          "verdict", "note"]

RUBRIC = """########## JUDGING RULES — paste this whole block first, once per conversation ##########

You are grading knowledge-graph triples extracted from scientific papers. For each triple
you are given: the triple (subject, predicate, object), the ONTOLOGY DEFINITION of the
predicate (its allowed domain -> range), and the SOURCE SENTENCE the extractor took it from.

Give exactly one verdict per triple:

  CORRECT    The source sentence EXPLICITLY states this fact, AND the predicate fits its
             domain -> range (the subject is the type the domain calls for, the object is
             the type the range calls for).

  PARTIAL    The fact is implied rather than explicitly stated, OR the predicate is a loose
             fit (e.g. the object is not the type the range calls for, but the fact itself
             is supported).

  INCORRECT  Unsupported by the source sentence, hallucinated, or subject and object are
             swapped relative to the predicate's domain -> range direction.

Rules that decide most hard cases:
  * Judge ONLY from the source sentence given. Not from what you know about the paper,
    the field, or the entities involved. If the sentence does not carry it, it is not
    CORRECT no matter how true it is in the world.
  * A triple can be TRUE ABOUT THE PAPER and still INCORRECT here, because the sentence is
    the extractor's evidence.
  * Read the predicate definition before deciding. A plausible-sounding English relation
    with the wrong domain/range is at best PARTIAL.
  * Degenerate triples (subject and object are the same thing, or the object is a section
    heading, table/figure reference, or an ontology class name) are INCORRECT.
  * Do not reward or punish verbosity. Judge the claim, not the wording.

Give a short reason for every verdict that is not CORRECT.

########## END JUDGING RULES ##########

"""

HEADER = """Judge the following {n} triples using the JUDGING RULES from the start of this
conversation.

Return ONE JSON object, no markdown fences, no commentary:

{{"verdicts": [
  {{"triple_id": "<the id shown>", "verdict": "CORRECT|PARTIAL|INCORRECT", "reason": "<short>"}},
  ...
]}}

Include every triple_id below.
"""


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def cmd_batches(args):
    rows = read_csv(args.csv)
    dest = Path(args.dest) / args.slug
    (dest / "replies").mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.txt"):
        old.unlink()

    chunks = [rows[i:i + args.size] for i in range(0, len(rows), args.size)]
    for bi, chunk in enumerate(chunks, 1):
        parts = []
        if bi == 1 or args.repeat_rules:
            parts.append(RUBRIC)
        parts.append(HEADER.format(n=len(chunk)))
        for r in chunk:
            parts.append(
                f"\n----- triple_id: {r['triple_id']} -----\n"
                f"TRIPLE:     ({r['subject']})  --{r['predicate']}-->  ({r['object']})\n"
                f"PREDICATE DEFINITION: {r['predicate_definition']}\n"
                f"SOURCE SENTENCE: {' '.join(r['source_sentence'].split())}\n"
            )
        (dest / f"{bi:02d}_judge_batch_{bi:02d}.txt").write_text("".join(parts), encoding="utf-8")

    readme = (dest / "README.txt")
    readme.write_text(
        f"""HOW TO JUDGE  ({args.slug})
=========================================================

{len(chunks)} numbered files, {len(rows)} triples in total. Do them IN ORDER, one chat.

For each file:
  1. Open it, select all, copy, paste as ONE message.
  2. Save the whole reply in replies/ with the same name prefixed by "reply_", e.g.
     01_judge_batch_01.txt  ->  replies/reply_01_judge_batch_01.txt

File 01 carries the JUDGING RULES. Keep the SAME chat and the SAME model for all files so
every verdict comes from one judge. Turn web search OFF - judge only from the sentence given.

Write the exact model name in MODEL_USED.txt.
When finished, send the replies folder back.
""", encoding="utf-8")
    (dest / "MODEL_USED.txt").write_text(
        "Exact model name from the picker: ...........................\n"
        "Date: ...................  Did it switch model mid-run? yes / no\n", encoding="utf-8")

    chars = sum(len(p.read_text(encoding="utf-8")) for p in dest.glob("*.txt"))
    print(f"{len(chunks)} files, {len(rows)} triples → {dest}  (~{chars//4:,} tokens of pasting)")


_JSON_OBJ = re.compile(r"\{.*\}", re.S)
_VERDICTS = {"CORRECT", "PARTIAL", "INCORRECT"}


def _extract(text):
    text = re.sub(r"```(?:json)?", "", text)
    m = _JSON_OBJ.search(text)
    if not m:
        return []
    blob = m.group(0)
    try:
        return json.loads(blob).get("verdicts", [])
    except json.JSONDecodeError:
        out = []
        for o in re.finditer(
                r'\{\s*"triple_id"\s*:\s*"([^"]+)"\s*,\s*"verdict"\s*:\s*"([^"]+)"'
                r'(?:\s*,\s*"reason"\s*:\s*"([^"]*)")?', blob, re.S):
            out.append({"triple_id": o.group(1), "verdict": o.group(2),
                        "reason": o.group(3) or ""})
        return out


def cmd_collect(args):
    rows = read_csv(args.csv)
    src = Path(args.dest) / args.slug / "replies"
    files = sorted(src.glob("*")) if src.exists() else []
    if args.replies:
        files = [Path(args.replies)]
    if not files:
        sys.exit(f"no reply files in {src}")

    got = {}
    for f in files:
        if f.suffix.lower() not in (".txt", ".json", ".md"):
            continue
        for v in _extract(f.read_text(encoding="utf-8", errors="replace")):
            verdict = str(v.get("verdict", "")).strip().upper()
            if verdict in _VERDICTS:
                got[str(v.get("triple_id", "")).strip()] = (verdict, v.get("reason", ""))

    filled = 0
    for r in rows:
        hit = got.get(r["triple_id"])
        if hit:
            r["verdict"], r["note"] = hit
            filled += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    missing = [r["triple_id"] for r in rows if not r.get("verdict")]
    print(f"{filled}/{len(rows)} verdicts collected → {out}")
    if missing:
        print(f"missing {len(missing)}: {', '.join(missing[:12])}"
              f"{' …' if len(missing) > 12 else ''}")
        print("Re-paste the batches holding those ids; collect merges whatever is present.")
    print(f"Next: python3 gold_report.py --labels {out} --model {rows[0]['model']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("batches", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--csv", required=True, help="gold CSV from gold_eval.py export")
        p.add_argument("--slug", required=True, help="folder name for this judge")
        p.add_argument("--dest", default="judge_upload")
        if name == "batches":
            p.add_argument("--size", type=int, default=10, help="triples per message")
            p.add_argument("--repeat-rules", action="store_true")
        else:
            p.add_argument("--out", required=True, help="filled label CSV to write")
            p.add_argument("--replies", default=None, help="single file holding all replies")
    args = ap.parse_args()
    (cmd_batches if args.cmd == "batches" else cmd_collect)(args)


if __name__ == "__main__":
    main()
