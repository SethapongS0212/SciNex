#!/usr/bin/env python3
"""
CPU equivalence test for ft_train_qlora.py --share-system-prompt (no GPU, no model download).

A tiny randomly initialised Qwen3 (real Qwen3 tokenizer, real training examples) with LoRA on
the same target modules is run on one optimizer step of 8 examples, twice:
  reference  every example alone, full causal attention, full logits (the textbook loss)
  packed     StepPacker + packed_loss_sum (shared prefix, 4D mask, per-segment positions)
and the summed loss and every LoRA gradient must agree to float32 precision. Run for sdpa and
eager attention, with the default cap and with a small cap that forces groups to split.
Then one real Trainer optimizer step (SGD, gradient checkpointing on) is taken in the default
mode and in --share-system-prompt mode from the same weights; the updates must agree.

    python ft_runs/test_share_prompt.py --data ft_data/acl_maxempty50/fold0
"""
import argparse
import copy
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ft_train_qlora as T  # noqa: E402


def pick_step(rows, seed):
    """8 examples like a real step: 3 + 2 extraction from two ontologies, 2 vocabulary, 1 ontology."""
    rng = random.Random(seed)
    ex = [r for r in rows if r["meta"]["part"] == "extraction"]
    onts = sorted({r["meta"]["ontology"] for r in ex})
    pick = rng.sample([r for r in ex if r["meta"]["ontology"] == onts[0]], 3)
    pick += rng.sample([r for r in ex if r["meta"]["ontology"] == onts[-1]], 2)
    pick += rng.sample([r for r in rows if r["meta"]["part"] == "vocabulary"], 2)
    pick += rng.sample([r for r in rows if r["meta"]["part"] == "ontology"], 1)
    rng.shuffle(pick)
    return pick


def tiny_model(vocab, attn, seed):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=8192, tie_word_embeddings=False,
                      attn_implementation=attn)
    m = Qwen3ForCausalLM(cfg).float()
    m.config.use_cache = False
    for p in m.parameters():
        p.requires_grad_(False)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                     task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "gate_proj", "up_proj", "down_proj"]))
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():   # LoRA B starts at 0, which would make every A-gradient 0: perturb all
        for n, p in m.named_parameters():
            if p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return m


def lora_grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad}


def reference(m, feats):
    m.zero_grad(set_to_none=True)
    n = sum(sum(l != -100 for l in f["labels"]) for f in feats)
    total = 0.0
    for f in feats:
        ids, lab = torch.tensor([f["input_ids"]]), torch.tensor([f["labels"]])
        logits = m(input_ids=ids, use_cache=False).logits.float()
        loss = F.cross_entropy(logits[0, :-1], lab[0, 1:], ignore_index=-100, reduction="sum") / n
        loss.backward()
        total += loss.item()
    return total, lora_grads(m), n, sum(len(f["input_ids"]) for f in feats)


def packed(m, feats, cap, attn):
    m.zero_grad(set_to_none=True)
    seqs = T.StepPacker(cap, attn)([{"step": feats}])["seqs"]
    n = sum(int(s["tgt"].numel()) for s in seqs)
    total = 0.0
    for s in seqs:
        loss = T.packed_loss_sum(m, s) / n
        loss.backward()
        total += loss.item()
    return total, lora_grads(m), n, sum(int(s["input_ids"].shape[1]) for s in seqs), len(seqs)


def compare(name, a, b):
    worst = 0.0
    for k in a:
        scale = max(a[k].abs().max().item(), 1e-12)
        worst = max(worst, (a[k] - b[k]).abs().max().item() / scale)
    print(f"  {name}: worst relative gradient difference {worst:.2e}")
    return worst


def trainer_step(m, items, collator, ga, outdir):
    """One real optimizer step through CompletionOnlyTrainer (SGD, lr 1) -> parameter update."""
    from transformers import TrainingArguments
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    args = TrainingArguments(output_dir=outdir, use_cpu=True, per_device_train_batch_size=1,
                             gradient_accumulation_steps=ga, max_steps=1, learning_rate=1.0,
                             optim="sgd", lr_scheduler_type="constant", warmup_steps=0,
                             max_grad_norm=1e9, report_to=[], remove_unused_columns=False,
                             save_strategy="no", logging_steps=1, gradient_checkpointing=True,
                             gradient_checkpointing_kwargs={"use_reentrant": False},
                             dataloader_num_workers=0, seed=13)
    m.enable_input_require_grads()
    tr = T.completion_only_trainer_cls()(model=m, args=args, train_dataset=items,
                                         data_collator=collator)
    out = tr.train()
    upd = {n: (p.detach() - before[n]) for n, p in m.named_parameters() if p.requires_grad}
    return out.training_loss, upd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B", help="tokenizer only")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    rows = [json.loads(l) for l in open(Path(args.data) / "train.jsonl", encoding="utf-8")]
    step = pick_step(rows, args.seed)
    feats, _ = T.tokenize_examples(tok, step, 6144)
    print("step:", Counter(f["part"] for f in feats), " lengths:", [len(f["input_ids"]) for f in feats])
    tol, ok = 1e-4, True

    for attn in ("sdpa", "eager"):
        m = tiny_model(len(tok), attn, args.seed)
        r_loss, r_g, r_n, r_tok = reference(m, feats)
        for cap in (6144, 4000):
            p_loss, p_g, p_n, p_tok, n_seq = packed(m, feats, cap, attn)
            print(f"[{attn}, cap {cap}] loss ref {r_loss:.6f} packed {p_loss:.6f}  answer tokens "
                  f"{r_n}/{p_n}  tokens encoded {r_tok} -> {p_tok} ({p_tok / r_tok:.0%}) in {n_seq} seqs")
            ok &= r_n == p_n and abs(r_loss - p_loss) <= tol * abs(r_loss)
            ok &= compare("LoRA grads", r_g, p_g) <= tol

    print("[trainer] one SGD step, default path vs --share-system-prompt, gradient checkpointing on")
    base = tiny_model(len(tok), "sdpa", args.seed)
    with tempfile.TemporaryDirectory() as d:
        m1 = copy.deepcopy(base)
        l1, u1 = trainer_step(m1, [{k: v for k, v in f.items() if k not in ("part", "sys")}
                                   for f in feats], T.Collator(tok.pad_token_id), 8, d)
        m2 = copy.deepcopy(base)
        l2, u2 = trainer_step(m2, [{"step": feats}], T.StepPacker(6144, "sdpa"), 1, d)
    print(f"  training_loss default {l1:.6f}  shared {l2:.6f}")
    ok &= abs(l1 - l2) <= tol * abs(l1)
    ok &= compare("parameter update", u1, u2) <= tol

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
