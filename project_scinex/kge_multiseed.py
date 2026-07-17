"""
kge_multiseed.py
----------------
Run kg_transe_pipeline.py across several fixed seeds and report mean ± std
(and min/max) per KGE model — turning noisy single runs into a defensible
result.

Why: a single KGE run varies run-to-run (random weight init + negative
sampling). One number is a coin flip; "ComplEx MRR 0.16 ± 0.03 over 5 seeds"
is a statement about the model. Each seed is fixed (via --seed) so the whole
aggregate is itself reproducible — anyone can re-run these seeds and get the
same table.

Usage:
    python3 kge_multiseed.py --extractor fixed --model Qwen3-14B \
        --seeds 1 2 3 4 5 --epochs 500 --device cuda

    # quick smoke test
    python3 kge_multiseed.py --extractor fixed --model Qwen3-14B \
        --seeds 1 2 --epochs 20 --device cpu

Reads each per-seed result JSON written by kg_transe_pipeline.py and prints,
per model: mean ± std, min, max for Hits@1/5/10 and MRR.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).parent / "kg_transe_pipeline.py"
KGE_MODELS = ("transe", "complex", "rotate")
METRICS = ("hits@1", "hits@5", "hits@10", "mrr",
           # temporal-filtered "papers it cited" task (separate, smaller test)
           "filt_hits@1", "filt_hits@5", "filt_hits@10", "filt_mrr")


def run_one_seed(seed: int, args, run_dir: Path) -> dict[str, dict]:
    """Run the pipeline for one seed (--kge all) and load the per-model results.

    Returns {kge: results_dict}. Skips (warns) any model whose file is missing.
    """
    base = run_dir / f"seed{seed}.json"
    cmd = [
        sys.executable, str(PIPELINE),
        "--output-dir", args.output_dir,
        "--extractor", args.extractor,
        "--model", args.model,
        "--kge", args.kge,
        "--epochs", str(args.epochs),
        "--device", args.device,
        "--seed", str(seed),
        "--save-results", str(base),
        "--loss", args.loss,
        "--gamma", str(args.gamma),
        "--adv-temp", str(args.adv_temp),
        "--eval-split", args.eval_split,
        "--val-frac", str(args.val_frac),
        "--split-seed", str(args.split_seed),
    ]
    print(f"\n{'='*60}\n  SEED {seed}  →  {' '.join(cmd[-8:])}\n{'='*60}")
    subprocess.run(cmd, check=True)

    # --kge all suffixes the path with _<kge>; a single kge writes the path as-is
    kges = KGE_MODELS if args.kge == "all" else (args.kge,)
    out = {}
    for kge in kges:
        path = base.with_name(f"{base.stem}_{kge}{base.suffix}") if args.kge == "all" else base
        if path.exists():
            out[kge] = json.loads(path.read_text())
        else:
            print(f"  ⚠ missing result for {kge}: {path}")
    return out


def aggregate(per_seed: dict[int, dict[str, dict]], kges) -> dict:
    """Collect metric values across seeds → mean/std/min/max per model."""
    agg = {}
    for kge in kges:
        agg[kge] = {}
        for metric in METRICS:
            vals = [
                per_seed[s][kge][metric]
                for s in per_seed
                if kge in per_seed[s] and metric in per_seed[s][kge]
            ]
            if not vals:
                continue
            agg[kge][metric] = {
                "values": vals,
                "mean": statistics.mean(vals),
                "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min":  min(vals),
                "max":  max(vals),
                "n":    len(vals),
            }
    return agg


def print_table(agg: dict, kges, seeds) -> None:
    print(f"\n\n{'#'*70}")
    print(f"  MULTI-SEED KGE RESULT  —  seeds {seeds}  ({len(seeds)} runs)")
    print(f"{'#'*70}")
    print(f"\n  {'Model':<9}{'Metric':<9}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    print(f"  {'-'*50}")
    for kge in kges:
        for metric in METRICS:
            s = agg.get(kge, {}).get(metric)
            if not s:
                continue
            print(f"  {kge.upper():<9}{metric:<9}{s['mean']:>8.4f}"
                  f"{s['std']:>8.4f}{s['min']:>8.4f}{s['max']:>8.4f}")
        print(f"  {'-'*50}")

    # Compact "mean ± std" MRR / Hits@10 summary — the stable metrics to report
    print("\n  Headline (report these — stable metrics, not Hits@1):")
    for kge in kges:
        mrr = agg.get(kge, {}).get("mrr")
        h10 = agg.get(kge, {}).get("hits@10")
        if mrr and h10:
            print(f"    {kge.upper():<9} MRR {mrr['mean']:.4f} ± {mrr['std']:.4f}"
                  f"   Hits@10 {h10['mean']:.4f} ± {h10['std']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="Seeds to run (default: 1 2 3 4 5)")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--extractor", default="fixed", choices=["fixed", "fixed_scinex", "llm", "rebel"])
    ap.add_argument("--model", default="Qwen3-14B")
    ap.add_argument("--kge", default="all", choices=["transe", "complex", "rotate", "all"])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loss", default="margin", choices=["margin", "adv"],
                    help="Loss passed to the pipeline (default: margin)")
    ap.add_argument("--gamma", type=float, default=9.0,
                    help="γ for --loss adv (default: 9.0)")
    ap.add_argument("--adv-temp", type=float, default=1.0,
                    help="Self-adversarial temperature α for --loss adv (default: 1.0)")
    ap.add_argument("--eval-split", default="all", choices=["all", "val", "test"],
                    help="Query-paper split to score: all/val/test (default: all)")
    ap.add_argument("--val-frac", type=float, default=0.5,
                    help="Validation fraction of query papers (default: 0.5)")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Seed for the val/test partition, fixed across configs (default: 42)")
    ap.add_argument("--run-dir", default=None,
                    help="Where per-seed JSONs go (default: <output-dir>/kge_seedruns)")
    ap.add_argument("--save-summary", default=None,
                    help="Path to write the aggregate JSON "
                         "(default: <output-dir>/kge_multiseed_summary.json)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir or Path(args.output_dir) / "kge_seedruns")
    run_dir.mkdir(parents=True, exist_ok=True)
    kges = KGE_MODELS if args.kge == "all" else (args.kge,)

    per_seed = {}
    for seed in args.seeds:
        per_seed[seed] = run_one_seed(seed, args, run_dir)

    agg = aggregate(per_seed, kges)
    print_table(agg, kges, args.seeds)

    summary_path = Path(args.save_summary or
                        Path(args.output_dir) / "kge_multiseed_summary.json")
    summary_path.write_text(json.dumps({
        "seeds": args.seeds,
        "extractor": args.extractor,
        "model": args.model,
        "epochs": args.epochs,
        "aggregate": agg,
    }, indent=2))
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
