#!/usr/bin/env python3
"""
ft_train_qlora.py
-----------------
QLoRA finetuning of one open-weight base model on the dataset built by
build_ft_data.py. One dataset, one run per base model.

    # 1. tokenizer only — length stats, what would be dropped, no model load, no GPU
    python3 ft_train_qlora.py --data ft_data/fold0 --base-model Qwen/Qwen3-8B \
            --run-name qwen3-8b-f0 --dry-run
    # 2. the real run (VM, one GPU process at a time)
    python3 ft_train_qlora.py --data ft_data/fold0 --base-model Qwen/Qwen3-8B --run-name qwen3-8b-f0

Output: ft_runs/<run-name>/adapter/ (LoRA weights + tokenizer) and run_manifest.json,
which records the train/eval papers — ft_generate.py reads it and REFUSES to generate
for any paper this adapter trained on.

Base models (verify the Hugging Face id before use — ids drift):
  Qwen/Qwen3-4B-Instruct-2507   8 GB LAPTOP GPU (RTX 5070 Laptop). Qwen3-8B does NOT fit for
                           training at 8 GB: its 4-bit weights + unquantised embeddings are ~6 GB
                           before activations. The 4B is ~2.7 GB and leaves room for ~5k-token
                           examples. On this laptop pass --allow-dev-torch (Blackwell sm_120 runs
                           on the installed cu128 nightly).
  Qwen/Qwen3-8B           20 GB VM, the default first target there
  Ministral-14B            20 GB VM (tight) / A100 — pass its HF id; the Ministral 3 family
                           may load through a multimodal class, check before a long run
  Gemma-4-31B              A100 only, and only if the first two show gains

How the loss is computed: the prompt (system + user) is rendered with the SAME helper
ft_generate.py uses at inference (ft_common.render_chat, thinking off), and loss is taken
on the assistant tokens only. An example longer than --max-seq-len is DROPPED, never
truncated — a truncated JSON target would teach the model to emit broken JSON.

VM rules (Claude.md): python3; never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
on the H100-20C vGPU (checked below); stable torch 2.6.0+cu124, not a dev nightly (checked
below); nvidia-smi reads 0% util even when busy — watch the log and VRAM instead.
Needs: transformers, peft, bitsandbytes, accelerate (requirements_ft.txt).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ft_common import (  # noqa: E402
    EXCLUDED_MODEL_MARKERS, fast_attention, load_split, read_jsonl, refuse_excluded,
    sha256_file, split_prompt_completion,
)

RUNS_DIR = Path("ft_runs")


def check_environment(allow_dev_torch):
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:true" in alloc.replace(" ", "").lower():
        sys.exit("PYTORCH_CUDA_ALLOC_CONF has expandable_segments:True — unsupported on the "
                 "H100-20C vGPU (every allocation fails). Unset it.")
    import torch
    if "dev" in torch.__version__ and not allow_dev_torch:
        sys.exit(f"torch {torch.__version__} is a dev nightly — it crippled bitsandbytes 4-bit "
                 f"to ~2 tok/s on this VM. Use stable 2.6.0+cu124 (or pass --allow-dev-torch).")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device — QLoRA training needs the GPU VM")


def check_data(data_dir, examples, dev_examples):
    """Enforce the HARD RULE again at train time: no eval-paper example, no Claude data,
    and the data was built from the split that is on disk now."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    held = set(manifest["eval_papers"])
    for e in examples + dev_examples:
        meta = e.get("meta", {})
        if meta.get("paper") in held:
            sys.exit(f"training example from EVAL paper {meta['paper']} — refusing to train")
        for field in ("teacher", "source"):
            if any(m in str(meta.get(field, "")).lower() for m in EXCLUDED_MODEL_MARKERS):
                sys.exit(f"example with {field}={meta[field]} — Claude data is excluded")
    split = load_split(manifest["split_file"])
    if split is not None:
        now = sorted(p for p in manifest["eval_papers"] + manifest["train_papers"]
                     if split["papers"].get(p, {}).get("fold") == manifest["eval_fold"])
        if now != sorted(manifest["eval_papers"]):
            sys.exit("the split on disk no longer matches the one this data was built from — "
                     "rebuild the data with build_ft_data.py")
    got = sha256_file(data_dir / "train.jsonl")
    if manifest.get("files", {}).get("train.jsonl") not in (None, got):
        sys.exit("train.jsonl changed since build_ft_data.py wrote it (sha256 mismatch)")
    return manifest


def tokenize_examples(tokenizer, examples, max_len):
    """-> (features, stats). Each feature: input_ids, labels (prompt masked with -100)."""
    feats, lengths, dropped, modes = [], defaultdict(list), Counter(), Counter()
    for e in examples:
        part = e.get("meta", {}).get("part", "?")
        prompt, completion, mode = split_prompt_completion(tokenizer, e["messages"])
        modes[mode] += 1
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        n = len(p_ids) + len(c_ids)
        lengths[part].append(n)
        if n > max_len:
            dropped[part] += 1
            continue
        feats.append({"input_ids": p_ids + c_ids,
                      "labels": [-100] * len(p_ids) + c_ids,
                      "part": part})
    return feats, {"lengths": lengths, "dropped": dropped, "template_modes": modes}


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else 0


def print_stats(stats, n_feats, args):
    print("token lengths per part (prompt + target):")
    for part, xs in sorted(stats["lengths"].items()):
        print(f"  {part:10s} n={len(xs):5d}  p50={_pct(xs, .5):5d}  p95={_pct(xs, .95):5d}  "
              f"max={max(xs):5d}  dropped(>{args.max_seq_len})={stats['dropped'][part]}")
    print(f"  chat template: {dict(stats['template_modes'])}  "
          f"('fallback' = template could not be split cleanly; check it before training)")
    steps = math.ceil(n_feats / (args.batch_size * args.grad_accum)) * args.epochs
    print(f"  kept {n_feats} examples -> ~{steps} optimizer steps "
          f"(batch {args.batch_size} x accum {args.grad_accum} x {args.epochs} epochs)")


def completion_only_trainer_cls():
    """Trainer whose loss materialises logits ONLY over the target span, normalised per TOKEN.

    The default path computes vocabulary logits (~152k wide for Qwen) at EVERY position:
    for a 5k-token example that is 3-5 GB in fp32 before the backward pass — more than an
    8 GB laptop GPU can spare, and pure waste, since the ~3.4k prompt tokens are masked out
    of the loss anyway. `logits_to_keep=k` makes the model project only the last k hidden
    states, which is exactly the target span because the target always ends the sequence.
    Same loss, a fraction of the memory.
    """
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class CompletionOnlyTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # TOKEN-level normalisation across the whole accumulation window. With this True the
            # Trainer counts every target token in the window (num_items_in_batch) and does NOT
            # divide by the accumulation steps; compute_loss returns sum / that count.
            #
            # ⛔ Measured failure of the alternative (Session 33): a per-EXAMPLE mean gives a
            # 7-token `{"triples": []}` target the same weight as a 300-token extraction, so the
            # abstain decision dominated and the adapter answered `{"triples": []}` to EVERY
            # paragraph — including training paragraphs whose target had triples.
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            L = labels.shape[1]
            has = labels != -100
            first = int(has.float().argmax(dim=1).min())   # earliest target position in the batch
            k = L - first + 1                               # logits at first-1 .. L-1
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        logits_to_keep=k, use_cache=False)
            logits = out.logits[:, :-1, :].float()          # predicts positions first .. L-1
            target = labels[:, first:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                   ignore_index=-100, reduction="sum")
            n = num_items_in_batch if num_items_in_batch is not None else (target != -100).sum()
            loss = loss / n
            return (loss, out) if return_outputs else loss

    return CompletionOnlyTrainer


def progress_callback(total_vram_gb):
    """One flushed line per optimizer step: loss, speed, peak VRAM — tqdm bars get lost in logs.
    Warns when peak VRAM nears the card's limit: on Windows the driver then pages to system
    RAM instead of raising OOM, and training silently slows ~10x."""
    import torch
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
            self.last_loss = None
            self.warned = False

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.last_loss = logs["loss"]

        def on_step_end(self, args, state, control, **kw):
            peak = torch.cuda.max_memory_allocated() / 2**30
            el = time.time() - self.t0
            eta = el / state.global_step * (state.max_steps - state.global_step)
            print(f"step {state.global_step}/{state.max_steps}  loss {self.last_loss}  "
                  f"{el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left  peak VRAM {peak:.2f} GB",
                  flush=True)
            if peak > 0.92 * total_vram_gb and not self.warned:
                self.warned = True
                print(f"  ! peak VRAM {peak:.2f} of {total_vram_gb:.2f} GB — near the limit; on Windows "
                      f"this pages to system RAM and crawls. Lower --max-seq-len or --lora-r.", flush=True)

    return Progress()


class Collator:
    """Right-pad to the longest sequence in the batch; padded labels are -100."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def load_model(args):
    import torch
    import transformers
    from packaging import version
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    dtype_kw = ("dtype" if version.parse(transformers.__version__) >= version.parse("4.56")
                else "torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map={"": 0},   # no silent CPU offload
        trust_remote_code=args.trust_remote_code, **{dtype_kw: torch.bfloat16})
    model.config.use_cache = False
    if args.upcast_fp32:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        # Lean path (default). peft's helper upcasts EVERY non-quantised weight to fp32 —
        # for Qwen3-4B that doubles the tied 152k x 2560 embedding from 0.8 to 1.6 GB, which
        # on an 8 GB laptop GPU pushed training into Windows' silent system-RAM fallback
        # (100% util at 41 W, >10 min for 16 examples). bf16 LoRA training is standard.
        for p in model.parameters():
            p.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=args.target_modules if args.target_modules != ["all-linear"]
                      else "all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="ft_data/fold<k> from build_ft_data.py")
    ap.add_argument("--base-model", required=True, help="Hugging Face id, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--parts", nargs="+", default=["extraction", "ontology", "vocabulary"],
                    help="train on a subset of parts (ablations)")
    ap.add_argument("--max-seq-len", type=int, default=6144,
                    help="the extraction system prompt alone is ~3.4k tokens")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="or: all-linear")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--upcast-fp32", action="store_true",
                    help="peft's prepare_model_for_kbit_training (fp32 non-quantised weights); "
                         "costs ~0.8 GB extra on Qwen3-4B — fine on the VM, not on 8 GB")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--allow-dev-torch", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="tokenizer only: stats, no model, no GPU")
    args = ap.parse_args()

    refuse_excluded(args.base_model, "base model")
    if ":" in args.run_name or "/" in args.run_name:
        sys.exit("--run-name must not contain ':' or '/' — it becomes a directory name")
    data_dir = Path(args.data)
    examples = [e for e in read_jsonl(data_dir / "train.jsonl")
                if e.get("meta", {}).get("part") in args.parts]
    dev_examples = read_jsonl(data_dir / "dev.jsonl") if (data_dir / "dev.jsonl").exists() else []
    manifest = check_data(data_dir, examples, dev_examples)
    print(f"{len(examples)} train examples ({Counter(e['meta']['part'] for e in examples)}), "
          f"{len(dev_examples)} dev; eval papers held out: {manifest['eval_papers']}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    feats, stats = tokenize_examples(tok, examples, args.max_seq_len)
    dev_feats, _ = tokenize_examples(tok, dev_examples, args.max_seq_len)
    print_stats(stats, len(feats), args)
    if args.dry_run:
        print("--dry-run: no model loaded")
        return

    check_environment(args.allow_dev_torch)
    import torch
    from transformers import TrainingArguments, set_seed
    set_seed(args.seed)
    random.Random(args.seed).shuffle(feats)

    run_dir = RUNS_DIR / args.run_name
    if (run_dir / "adapter").exists():
        sys.exit(f"{run_dir}/adapter exists — pick a new --run-name rather than overwrite a run")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args)
    total_steps = args.max_steps if args.max_steps > 0 else \
        math.ceil(math.ceil(len(feats) / (args.batch_size * args.grad_accum)) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(args.warmup_ratio * total_steps)),   # warmup_ratio is deprecated
        optim="paged_adamw_8bit", bf16=True, logging_steps=5,
        save_strategy="epoch", save_total_limit=1,
        eval_strategy="epoch" if dev_feats else "no",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], seed=args.seed, remove_unused_columns=False,
    )
    strip = [{k: v for k, v in f.items() if k != "part"} for f in feats]
    dev_strip = [{k: v for k, v in f.items() if k != "part"} for f in dev_feats]
    trainer = completion_only_trainer_cls()(
        model=model, args=targs, train_dataset=strip,
        eval_dataset=dev_strip or None, data_collator=Collator(tok.pad_token_id),
        callbacks=[progress_callback(torch.cuda.get_device_properties(0).total_memory / 2**30)])
    t0 = time.time()
    with fast_attention(training=True):   # backward + checkpoint recompute run inside this block too
        result = trainer.train()
    model.save_pretrained(run_dir / "adapter")
    tok.save_pretrained(run_dir / "adapter")

    import peft
    import transformers
    run_manifest = {
        "run_name": args.run_name, "base_model": args.base_model,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "minutes": round((time.time() - t0) / 60, 1),
        "data_dir": str(data_dir), "data_manifest": manifest,
        "train_papers": manifest["train_papers"], "eval_papers": manifest["eval_papers"],
        "dev_papers": manifest.get("dev_papers", []),
        "parts": args.parts, "n_train_examples": len(feats), "dropped_too_long": dict(stats["dropped"]),
        "hyperparameters": {k: getattr(args, k) for k in (
            "max_seq_len", "epochs", "max_steps", "lr", "batch_size", "grad_accum", "warmup_ratio",
            "lora_r", "lora_alpha", "lora_dropout", "target_modules", "seed")},
        "train_loss": getattr(result, "training_loss", None),
        "log_history": trainer.state.log_history,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"adapter -> {run_dir / 'adapter'}\nmanifest -> {run_dir / 'run_manifest.json'}")
    print(f"next: python3 ft_generate.py --adapter {run_dir} --model-slug <slug> --ingest")


if __name__ == "__main__":
    main()
