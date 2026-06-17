"""
C2A Prediction using DBO ensemble (A8_trio: gbert + gelectra + mdeberta)
========================================================================
Runs the trained DBO ensemble on c2a_test_26.csv and maps
the 4-class DBO output to TRUE/FALSE:
    agitation | subversive  ->  TRUE
    nothing   | criticism   ->  FALSE

USAGE:
    python c2a_predict.py --dbo_dir path/to/A8_trio_run

    # Use only specific models:
    python c2a_predict.py --dbo_dir path/to/A8_trio_run --models gbert gelectra

EXPECTED DBO DIR LAYOUT:
    <dbo_dir>/
        model_gbert/best_model_weights.pt
        model_gelectra/best_model_weights.pt
        model_mdeberta/best_model_weights.pt

OUTPUT:
    other_challenges/c2a/c2a_predictions.csv
        id, prediction (TRUE/FALSE), prob_TRUE, dbo_class
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dbo_dir",   required=True,
                    help="Run directory containing model_gbert/, model_gelectra/, model_mdeberta/")
parser.add_argument("--test_file", default="other_challenges/c2a/c2a_test_26.csv")
parser.add_argument("--out_file",  default="other_challenges/c2a/c2a_predictions.csv")
parser.add_argument("--models",    nargs="+",
                    default=["gbert", "gelectra", "mdeberta"],
                    choices=["gbert", "gelectra", "mdeberta"])
parser.add_argument("--batch_size", type=int,   default=32)
parser.add_argument("--max_length", type=int,   default=128)
args = parser.parse_args()

MODEL_REGISTRY = {
    "gbert":    "deepset/gbert-large",
    "gelectra": "deepset/gelectra-large-germanquad",
    "mdeberta": "microsoft/mdeberta-v3-base",
}

# DBO class order produced by sklearn LabelEncoder (alphabetical)
DBO_CLASSES = ["agitation", "criticism", "nothing", "subversive"]
TRUE_CLASSES = {"agitation", "subversive"}

DEVICE  = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")
DBO_DIR = Path(args.dbo_dir)

print(f"\n{'='*60}")
print(f"  C2A Prediction  (DBO ensemble → TRUE/FALSE)")
print(f"{'='*60}")
print(f"  Device:  {DEVICE}")
print(f"  DBO dir: {DBO_DIR}")
print(f"  Models:  {args.models}\n")

# ---------------------------------------------------------------------------
# Model architecture (must match training)
# ---------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, model_id: str, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_id)
        hidden          = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
df = pd.read_csv(args.test_file, sep=";", engine="python", on_bad_lines="skip")
df = df.rename(columns={"description": "text"})
texts = df["text"].fillna("").tolist()
print(f"  Test rows: {len(texts):,}\n")

# ---------------------------------------------------------------------------
# Inference per model
# ---------------------------------------------------------------------------
all_model_probs = []   # each entry: array [N, 4]

for shortname in args.models:
    model_id  = MODEL_REGISTRY[shortname]
    ckpt_path = DBO_DIR / f"model_{shortname}" / "best_model_weights.pt"

    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found, skipping {shortname}: {ckpt_path}")
        continue

    print(f"  Loading {shortname} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = TransformerClassifier(model_id, n_classes=len(DBO_CLASSES)).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # Checkpoints may be raw state_dict or wrapped
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()

    probs_batches = []
    for i in range(0, len(texts), args.batch_size):
        enc = tokenizer(
            texts[i:i+args.batch_size],
            truncation=True, padding="max_length",
            max_length=args.max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_batches.append(probs)

    model_probs = np.vstack(probs_batches)   # [N, 4]
    all_model_probs.append(model_probs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Per-model DBO class distribution
    preds = model_probs.argmax(axis=1)
    print(f"  [{shortname}] DBO distribution:")
    for idx, cls in enumerate(DBO_CLASSES):
        cnt = (preds == idx).sum()
        print(f"      {cls:<12} {cnt:>5,}")

# ---------------------------------------------------------------------------
# Ensemble soft-voting + map to TRUE/FALSE
# ---------------------------------------------------------------------------
ensemble_probs  = np.mean(all_model_probs, axis=0)   # [N, 4]
dbo_pred_idx    = ensemble_probs.argmax(axis=1)
dbo_pred_labels = [DBO_CLASSES[i] for i in dbo_pred_idx]

# Probability of TRUE = sum of agitation + subversive probs
true_indices = [DBO_CLASSES.index(c) for c in TRUE_CLASSES]
prob_true    = ensemble_probs[:, true_indices].sum(axis=1)

c2a_labels = ["TRUE" if lbl in TRUE_CLASSES else "FALSE" for lbl in dbo_pred_labels]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_df = pd.DataFrame({
    "id":         df["id"].values,
    "prediction": c2a_labels,
    "prob_TRUE":  prob_true.round(4),
    "dbo_class":  dbo_pred_labels,
})
Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(args.out_file, index=False, sep=";")

print(f"\n  Ensemble DBO distribution:")
for idx, cls in enumerate(DBO_CLASSES):
    cnt = (dbo_pred_idx == idx).sum()
    print(f"    {cls:<12} {cnt:>5,}")

print(f"\n  C2A prediction distribution:")
for lbl, cnt in pd.Series(c2a_labels).value_counts().items():
    print(f"    {lbl}  {cnt:>5,}  ({cnt/len(c2a_labels)*100:.1f}%)")

print(f"\n  Saved: {Path(args.out_file).resolve()}\n")
