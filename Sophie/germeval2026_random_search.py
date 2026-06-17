"""
GermEval 2026 – Random Search Config Generator
===============================================
Sampelt N zufaellige Hyperparameter-Kombinationen und schreibt eine
gridsearch_config_random_search.json fuer den germeval2026_run_gridsearch.py Runner.

Hyperparameter-Raeume:
  lr            log-uniform  [1e-5, 5e-5]
  batch_size    choice       [8, 16, 32]
  warmup_steps  choice       [0, 100, 300, 500, 1000]
  dropout       uniform      [0.05, 0.30]

VERWENDUNG:
    python germeval2026_random_search.py
    python germeval2026_random_search.py --n 10 --seed 42
    python germeval2026_random_search.py --n 15 --seed 7 --out gridsearch_config_rs.json
"""

import argparse
import json
import math
import random

parser = argparse.ArgumentParser(description="Random Search Config Generator")
parser.add_argument("--n",    type=int, default=10, help="Anzahl Runs (default: 10)")
parser.add_argument("--seed", type=int, default=42, help="Zufalls-Seed (default: 42)")
parser.add_argument("--out",  default="gridsearch_config_random_search.json",
                    help="Ausgabe-Pfad der generierten Config")
parser.add_argument("--augmentation", default="paraphrase",
                    choices=["none", "paraphrase", "generated", "both"],
                    help="Augmentierung fuer alle Runs (default: paraphrase)")
parser.add_argument("--models", nargs="+", default=["gbert", "deberta"],
                    choices=["gbert", "xlmr", "deberta"],
                    help="Ensemble-Modelle fuer alle Runs (default: gbert deberta)")
args = parser.parse_args()

rng = random.Random(args.seed)

LR_MIN_LOG    = math.log(1e-5)
LR_MAX_LOG    = math.log(5e-5)
BATCH_SIZES   = [8, 16, 32]
WARMUP_OPTS   = [0, 100, 300, 500, 1000]
DROPOUT_MIN   = 0.05
DROPOUT_MAX   = 0.30

runs = []
for i in range(args.n):
    lr           = math.exp(rng.uniform(LR_MIN_LOG, LR_MAX_LOG))
    batch_size   = rng.choice(BATCH_SIZES)
    warmup_steps = rng.choice(WARMUP_OPTS)
    dropout      = rng.uniform(DROPOUT_MIN, DROPOUT_MAX)

    runs.append({
        "name":         f"sprint2_rs_{i:02d}",
        "augmentation": args.augmentation,
        "models":       args.models,
        "lr":           round(lr, 8),
        "batch_size":   batch_size,
        "warmup_steps": warmup_steps,
        "dropout":      round(dropout, 4),
        "notes":        f"random_search seed={args.seed} run={i}",
    })

config = {
    "base_script":   "Sophie/germeval2026_sprint2_train.py",
    "base_out_dir":  "model_dataset_gridsearch",
    "results_csv":   "results.csv",
    "global_args":   {
        "train_file":           "preprocessed/train_minimal.csv",
        "test_file":            "preprocessed/test_minimal.csv",
        "paraphrase_file":      "synthethic_data/paraphrased.csv",
        "generated_file":       "preprocessed/synthetic_data.csv",
        "threshold_agitation":  0.45,
        "threshold_subversive": 0.45,
    },
    "runs":          runs,
}

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\nRandom Search Config geschrieben: {args.out}")
print(f"Seed: {args.seed}  |  N: {args.n}  |  Augmentation: {args.augmentation}")
print(f"Modelle: {args.models}\n")

header = f"{'#':<4} {'lr':>12} {'bs':>4} {'warmup':>8} {'dropout':>8}  Name"
print(header)
print("─" * len(header))
for i, r in enumerate(runs):
    print(f"{i:<4} {r['lr']:>12.2e} {r['batch_size']:>4} {r['warmup_steps']:>8} {r['dropout']:>8.4f}  {r['name']}")
print()
