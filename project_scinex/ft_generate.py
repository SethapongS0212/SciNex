#!/usr/bin/env python3
"""
ft_generate.py
--------------
Run a local open-weight model — the base model, or base + a QLoRA adapter — through
the SAME relation-only extraction path the API models used, on held-out papers:

    prompts (current code) -> this script -> responses.jsonl -> claude_extract.py ingest
                                              (same {"para_id","raw"} format, so the guards,
                                               KG builder and ontology_eval/qa_eval all apply
                                               unchanged)

The base and the finetuned model MUST both go through this script: same weights
format (4-bit NF4), same chat template (ft_common.render_chat — the one training used),
same greedy decoding, same token budget. The only variable is the adapter. Comparing a
finetuned HF model against the old Ollama qwen3-8b-v2 run would confound the adapter
with the runtime (GGUF vs NF4, different templating).

    # finetuned — papers default to the adapter's own eval fold; any paper it trained on is refused
    python3 ft_generate.py --adapter ft_runs/qwen3-8b-f0 --model-slug qwen3-8b-ft-cv --ingest
    # baseline — same base model, no adapter, same papers
    python3 ft_generate.py --base-model Qwen/Qwen3-8B --model-slug qwen3-8b-base-hf \
            --papers-from ft_data/fold0/eval_papers.txt --ingest
    # --dry-run: render and count prompts only, no model

Cross-validation: give every fold's adapter the SAME --model-slug (e.g. qwen3-8b-ft-cv).
Each paper is then generated only by the one adapter that held it out, and ontology_eval /
qa_eval can score all papers under one slug with no train/eval leakage.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ft_common import (  # noqa: E402
    OUTPUT_DIR, SCINEX_OWL_DEFAULT, TEACHER_DEFAULT, build_messages, current_system_prompt,
    exdir_for, fast_attention, load_paragraphs, read_jsonl, refuse_excluded, render_chat,
    write_jsonl,
)


def resolve_papers(args, run_manifest):
    if args.papers:
        papers = args.papers
    elif args.papers_from:
        papers = Path(args.papers_from).read_text(encoding="utf-8").split()
    elif run_manifest:
        papers = run_manifest["eval_papers"]
    else:
        sys.exit("give --papers, --papers-from, or --adapter (defaults to its eval papers)")
    if run_manifest:
        trained = set(run_manifest["train_papers"]) | set(run_manifest.get("dev_papers", []))
        leak = [p for p in papers if p in trained]
        if leak:
            sys.exit(f"REFUSING: this adapter was trained on {leak}. Generating on them would "
                     f"score the model on its own training papers.")
    return papers


def load_model(base_model, adapter_dir, trust_remote_code):
    import torch
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok_src = adapter_dir if adapter_dir else base_model
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=trust_remote_code)
    tok.padding_side = "left"                      # batched generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    dtype_kw = ("dtype" if version.parse(transformers.__version__) >= version.parse("4.56")
                else "torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=trust_remote_code, **{dtype_kw: torch.bfloat16})
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    # greedy, deterministic — Qwen3's generation_config defaults to sampling
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    return model, tok


def generate_batch(model, tok, prompts, max_new_tokens):
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad(), fast_attention():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    res = []
    n_in = enc["input_ids"].shape[1]
    for row in out:
        gen = row[n_in:]
        n_tok = int((gen != tok.pad_token_id).sum())
        text = tok.decode(gen, skip_special_tokens=True).strip()
        res.append((text, n_tok, n_tok >= max_new_tokens))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="ft_runs/<run-name> (reads run_manifest.json)")
    ap.add_argument("--base-model", default=None, help="required without --adapter")
    ap.add_argument("--model-slug", required=True, help="output dir name; no ':' or '/'")
    ap.add_argument("--papers", nargs="+")
    ap.add_argument("--papers-from", default=None, help="file of paper ids (e.g. eval_papers.txt)")
    ap.add_argument("--ontologies", nargs="+", choices=["ceo", "scinex"], default=["ceo", "scinex"])
    ap.add_argument("--ontology-file", default=SCINEX_OWL_DEFAULT)
    ap.add_argument("--teacher", default=TEACHER_DEFAULT,
                    help="prefer this run's prompts.jsonl so para_ids match the teacher's")
    ap.add_argument("--max-new-tokens", type=int, default=3072,
                    help="same cap as the API runs; a truncated JSON yields zero triples")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="each sequence's KV cache is ~1 GB for a 4B model at ~7k tokens; "
                         "2 fits an 8 GB GPU, the 20 GB VM can take more")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only the first N paragraphs per paper x ontology")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--ingest", action="store_true", help="run claude_extract.py ingest afterwards")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="render + count prompts, no model")
    args = ap.parse_args()

    if ":" in args.model_slug or "/" in args.model_slug:
        sys.exit("--model-slug must not contain ':' or '/' — it becomes a directory name")
    refuse_excluded(args.model_slug, "model slug")
    run_manifest = None
    adapter_dir = None
    if args.adapter:
        run_dir = Path(args.adapter)
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        adapter_dir = str(run_dir / "adapter")
        if args.base_model and args.base_model != run_manifest["base_model"]:
            sys.exit(f"--base-model {args.base_model} != adapter's base {run_manifest['base_model']}")
        args.base_model = run_manifest["base_model"]
    if not args.base_model:
        sys.exit("--base-model is required without --adapter")
    refuse_excluded(args.base_model, "base model")
    papers = resolve_papers(args, run_manifest)

    # render every prompt first (cheap), so --dry-run and the real run see the same work list
    jobs = []
    for ont in args.ontologies:
        system, relations = current_system_prompt(ont, args.ontology_file)
        exdir = exdir_for("relation", ont)
        for paper in papers:
            rows, source = load_paragraphs(paper, exdir, prefer_slugs=(args.teacher,))
            if rows is None:
                print(f"  ! {paper}/{ont}: no paragraphs (no prompts.jsonl, no parsed HTML) — skipped")
                continue
            if args.limit:
                rows = rows[:args.limit]
            jobs.append((paper, ont, exdir, system, relations, rows, source))
    n_prompts = sum(len(j[5]) for j in jobs)
    print(f"{args.base_model}{' + ' + adapter_dir if adapter_dir else ''} -> "
          f"kg/relation*/{args.model_slug}/  {len(jobs)} paper x ontology runs, {n_prompts} paragraphs")
    if args.dry_run:
        for paper, ont, _e, _s, _r, rows, source in jobs:
            print(f"  {paper:20s} {ont:6s} {len(rows):3d} paragraphs  ({source})")
        return

    model, tok = load_model(args.base_model, adapter_dir, args.trust_remote_code)
    t0 = time.time()
    for paper, ont, exdir, system, relations, rows, source in jobs:
        dest = OUTPUT_DIR / paper / "kg" / exdir / args.model_slug
        dest.mkdir(parents=True, exist_ok=True)
        # the files claude_extract.py ingest reads
        write_jsonl(dest / "prompts.jsonl", rows)
        (dest / "system_prompt.txt").write_text(system, encoding="utf-8")
        title = ""
        tmeta = OUTPUT_DIR / paper / "kg" / exdir / args.teacher / "prompt_meta.json"
        if tmeta.exists():
            title = json.loads(tmeta.read_text(encoding="utf-8")).get("title", "")
        (dest / "prompt_meta.json").write_text(json.dumps({
            "paper": paper, "title": title, "extractor": "relation", "ontology": ont,
            "model_slug": args.model_slug, "n_paragraphs": len(rows),
            "n_relations": len(relations), "relations": relations,
            "base_model": args.base_model, "adapter": adapter_dir, "paragraph_source": source,
        }, indent=2), encoding="utf-8")

        out_f = dest / "responses.jsonl"
        done = set()
        if out_f.exists() and args.resume:
            done = {r["para_id"] for r in read_jsonl(out_f)}
        todo = [r for r in rows if r["para_id"] not in done]
        with open(out_f, "a" if done else "w", encoding="utf-8") as f:
            for i in range(0, len(todo), args.batch_size):
                chunk = todo[i:i + args.batch_size]
                prompts = [render_chat(tok, build_messages(system, r["user_prompt"]), True) for r in chunk]
                for r, (text, n_tok, trunc) in zip(chunk, generate_batch(model, tok, prompts, args.max_new_tokens)):
                    f.write(json.dumps({"para_id": r["para_id"], "raw": text,
                                        "n_tokens": n_tok, "truncated": trunc},
                                       ensure_ascii=False) + "\n")
                f.flush()
                import torch
                peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                print(f"    {paper}/{ont}: {min(i + args.batch_size, len(todo))}/{len(todo)} "
                      f"({(time.time() - t0) / 60:.1f} min, peak VRAM {peak:.1f} GB)", flush=True)
        if args.ingest:
            subprocess.run([sys.executable, "claude_extract.py", "ingest", "--paper", paper,
                            "--extractor", "relation", "--ontology", ont,
                            "--ontology-file", args.ontology_file, "--model-slug", args.model_slug,
                            "--responses", str(out_f)], check=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    if not args.ingest:
        print("next: claude_extract.py ingest per paper (or re-run with --ingest)")


if __name__ == "__main__":
    main()
