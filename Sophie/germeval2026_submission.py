"""
GermEval 2026 – Inference & Submission Generator
==================================================
Generiert Submission-Dateien aus den final_runs_scratch-Checkpoints.

Task-spezifische Ensemble-Konfiguration:
  c2a  →  gbert (single model)
  def  →  all-5 (gbert, xlmr, deberta, mdeberta, gelectra) aus all5_aug-both
           Run 1 (B1): uniform Soft-Voting
           Run 2 (B2): per-class F1 gewichtetes Soft-Voting
  vio  →  gbert (single model)

Checkpoint-Pfad:
  <base_dir>/<task>/model_<name>/best_model_weights.pt

VERWENDUNG:
    python germeval2026_submission.py --task c2a
    python germeval2026_submission.py --task def --run 1 --strategy uniform
    python germeval2026_submission.py --task def --run 2 --strategy perclass --weights_file val_weights.json
    python germeval2026_submission.py --task vio --base_dir /pfad/zu/final_runs_scratch
    python germeval2026_submission.py --task def --test_file /pfad/zu/def_test.csv

weights_file format (JSON):
  {"gbert": [f1_class0, f1_class1, ...], "xlmr": [...], ...}
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Task-Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "c2a": {
        "classes":    ["FALSE", "TRUE"],
        "label_col":  "c2a",
        "test_file":  "../data/GermEval2026/data/c2a/c2a_test_26.csv",
        "train_file": "../data/GermEval2026/data/c2a/c2a_train_26.csv",
        "run_dir":    "c2a",
        "ensemble": [
            ("gbert", "deepset/gbert-large", None),
        ],
    },
    "def": {
        "classes":    ["FALSE", "TRUE"],
        "label_col":  "def",
        "test_file":  "../data/GermEval2026/data/def/def_test.csv",
        "train_file": "../data/GermEval2026/data/def/def_train.csv",
        "run_dir":    "all5_aug-both",
        # B1/B2: all-5 aus all5_aug-both (weight ignored when strategy=perclass)
        "ensemble": [
            ("gbert",    "deepset/gbert-large",                None),
            ("xlmr",     "FacebookAI/xlm-roberta-large",       None),
            ("deberta",  "microsoft/deberta-v3-base",          None),
            ("mdeberta", "microsoft/mdeberta-v3-base",         None),
            ("gelectra", "deepset/gelectra-large-germanquad",  None),
        ],
    },
    "dbo": {
        "classes":    ["agitation", "criticism", "nothing", "subversive"],
        "label_col":  "dbo",
        "test_file":  "../data/GermEval2026/data/dbo/dbo_test_26.csv",
        "train_file": "../data/GermEval2026/data/dbo/dbo_train_26.csv",
        "run_dir":    "all5_aug-both",
        # B1/B2: all-5 aus all5_aug-both (weight ignored when strategy=perclass)
        "ensemble": [
            ("gbert",    "deepset/gbert-large",                None),
            ("xlmr",     "FacebookAI/xlm-roberta-large",       None),
            ("deberta",  "microsoft/deberta-v3-base",          None),
            ("mdeberta", "microsoft/mdeberta-v3-base",         None),
            ("gelectra", "deepset/gelectra-large-germanquad",  None),
        ],
    },
    "vio": {
        "classes":    ["call2violence", "glorification", "nothing", "other", "propensity", "support"],
        "label_col":  "vio",
        "test_file":  "../data/GermEval2026/data/vio/vio_test_26.csv",
        "train_file": "../data/GermEval2026/data/vio/vio_train_26.csv",
        "run_dir":    "vio",
        "ensemble": [
            ("gbert", "deepset/gbert-large", None),
        ],
    },
}

_SCRIPT_DIR = Path(__file__).parent

LABEL_FIXES = {
    "prospensity": "propensity",
    "True":  "TRUE",
    "False": "FALSE",
}

MAX_LENGTH = 128
BATCH_SIZE = 16
DROPOUT    = 0.1
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 – Submission")
parser.add_argument("--task",       required=True, choices=list(TASK_CONFIG),
                    help="Task: c2a | def | dbo | vio")
parser.add_argument("--team",       default="DetecTUM",
                    help="Teamname (wie auf Codabench registriert)")
parser.add_argument("--run",        default="1",
                    help="Run-Nummer: 1, 2 oder 3")
parser.add_argument("--base_dir",   default="model_dataset_gridsearch",
                    help="Ordner mit den Run-Unterordnern (default: model_dataset_gridsearch)")
parser.add_argument("--train_file", default=None,
                    help="Trainingsdatei fuer LabelEncoder (ueberschreibt Task-Default)")
parser.add_argument("--test_file",  default=None,
                    help="Pfad zur Test-CSV (ueberschreibt Task-Default)")
parser.add_argument("--out_dir",      default=None,
                    help="Ausgabeverzeichnis (default: submissions/<task>)")
parser.add_argument("--strategy",     default="uniform", choices=["uniform", "perclass"],
                    help="Ensemble-Strategie: uniform (B1) | perclass (B2)")
parser.add_argument("--weights_file", default=None,
                    help="JSON mit per-class F1-Gewichten pro Modell (benoetigt fuer --strategy perclass)")
args = parser.parse_args()

TASK   = args.task
CFG    = TASK_CONFIG[TASK]
TEAM   = args.team
RUN_NR = args.run

BASE_DIR = Path(args.base_dir) if args.base_dir != "model_dataset_gridsearch" else _SCRIPT_DIR / "model_dataset_gridsearch"
OUT_DIR  = Path(args.out_dir) if args.out_dir else Path("submissions") / TASK
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_NAME = f"{TEAM}{RUN_NR}_{TASK}.csv"
ZIP_NAME = f"{TEAM}{RUN_NR}_{TASK}.zip"

# ──────────────────────────────────────────────────────────────────────────────
# Architektur (identisch mit Training)
# ──────────────────────────────────────────────────────────────────────────────
class TransformerClassifier(nn.Module):
    def __init__(self, model_id, n_classes, dropout=0.1):
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
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts      = texts
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────
def load_test_data(test_path: str) -> pd.DataFrame:
    """Liest Test-CSV (id;description oder id,description)."""
    for kwargs in [
        {"sep": ";"},
        {"sep": ";", "quoting": 3},
        {"sep": ","},
        {"sep": ";", "engine": "python"},
    ]:
        try:
            df = pd.read_csv(test_path, **kwargs)
            if "id" in df.columns and "description" in df.columns:
                return df[["id", "description"]].dropna(subset=["description"])
        except Exception:
            continue
    raise ValueError(f"Kann Testdaten nicht lesen: {test_path}")


def load_label_classes(train_path: str, label_col: str, fallback: list) -> list:
    """Gibt alle Label-Werte aus der Trainingsdatei zurueck (fuer LabelEncoder)."""
    for kwargs in [{"sep": ";"}, {"sep": ","}, {}]:
        try:
            df = pd.read_csv(train_path, **kwargs)
            if label_col in df.columns:
                vals = df[label_col].dropna().astype(str).tolist()
                # Tippfehler-Fixes anwenden
                vals = [LABEL_FIXES.get(v, v) for v in vals]
                return vals
        except Exception:
            continue
    print(f"  WARNUNG: Trainingsdaten nicht lesbar ({train_path}) – verwende hartcodierte Klassen.")
    return fallback * 10


@torch.no_grad()
def run_inference(model, texts, tokenizer) -> np.ndarray:
    """Gibt Softmax-Wahrscheinlichkeiten zurueck: shape (N, n_classes)."""
    dataset = InferenceDataset(texts, tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    all_probs = []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Hauptprogramm
# ──────────────────────────────────────────────────────────────────────────────
SEP = "-" * 70
print(f"\n{'='*70}")
print(f"  GermEval 2026 – Submission Generator")
print(f"{'='*70}")
print(f"  Task:      {TASK}")
print(f"  Team:      {TEAM}  Run: {RUN_NR}")
print(f"  Device:    {DEVICE}")
print(f"  Ensemble:  {[m for m,_,_ in CFG['ensemble']]}")
print(f"  Strategie: {args.strategy}")
print(f"  Ausgabe:   {OUT_DIR / ZIP_NAME}")
print(f"{'='*70}\n")

# 1. Testdaten laden
test_path = args.test_file or str(_SCRIPT_DIR / CFG["test_file"])
print(f"  Testdaten: {test_path}")
df_test    = load_test_data(test_path)
test_ids   = df_test["id"].astype(str).tolist()
test_texts = df_test["description"].tolist()
print(f"  {len(test_texts)} Test-Beispiele geladen.\n")

# 2. LabelEncoder aufbauen
train_path = args.train_file or str(_SCRIPT_DIR / CFG["train_file"])
print(f"  LabelEncoder aus: {train_path}")
labels_for_le = load_label_classes(train_path, CFG["label_col"], CFG["classes"])

le = LabelEncoder()
le.fit(labels_for_le)
print(f"  Klassen: {list(le.classes_)}\n")
n_classes = len(le.classes_)

# 3. Inferenz fuer jedes Ensemble-Modell
print(f"{SEP}")
print(f"  INFERENZ")
print(f"{SEP}")

probs_list   = []
weight_list  = []

for model_name, model_id, weight in CFG["ensemble"]:
    ckpt_path = BASE_DIR / CFG["run_dir"] / f"model_{model_name}" / "best_model_weights.pt"
    print(f"\n  Lade {model_name:<12} <- {ckpt_path}")

    if not ckpt_path.exists():
        print(f"  FEHLER: Checkpoint nicht gefunden – ueberspringe {model_name}.")
        continue

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = TransformerClassifier(model_id, n_classes, DROPOUT).to(DEVICE)
    state     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in state.items()})

    probs = run_inference(model, test_texts, tokenizer)
    probs_list.append(probs)
    weight_list.append(weight if weight is not None else 1.0)
    print(f"  {model_name}: shape {probs.shape}")

    del model
    torch.cuda.empty_cache()

if not probs_list:
    raise RuntimeError("Keine Checkpoints geladen – bitte Pfade pruefen.")

# 4. Ensemble-Aggregation
print(f"\n{SEP}")

strategy = args.strategy

if len(probs_list) == 1:
    print(f"  Single-Model – kein Voting noetig.")
    ensemble_probs = probs_list[0]
elif strategy == "perclass":
    # B2: per-class F1 gewichtetes Soft-Voting
    if not args.weights_file:
        raise ValueError("--weights_file benoetigt fuer --strategy perclass")
    with open(args.weights_file, encoding="utf-8") as f:
        weights_json = json.load(f)
    print(f"  Gewichte geladen. Verfuegbare Keys: {list(weights_json.keys())}")
    # short name → whatever key the JSON uses
    MODEL_KEY_MAP = {
        "gbert":    next((k for k in weights_json if "gbert"    in k.lower()), None),
        "xlmr":     next((k for k in weights_json if "xlm"      in k.lower()), None),
        "deberta":  next((k for k in weights_json if "deberta"  in k.lower() and "m" not in k.lower().replace("microsoft/","").replace("deberta","")), None),
        "mdeberta": next((k for k in weights_json if "mdeberta" in k.lower()), None),
        "gelectra": next((k for k in weights_json if "gelectra" in k.lower()), None),
    }
    loaded_model_names = [m for m, _, _ in CFG["ensemble"] if len(probs_list) > 0]
    loaded_model_names = loaded_model_names[:len(probs_list)]
    W = np.array([weights_json[MODEL_KEY_MAP[m]] for m in loaded_model_names], dtype=float)  # (M, C)
    P = np.stack(probs_list, axis=0)                                                          # (M, N, C)
    W_bc = W[:, np.newaxis, :]                                                                # (M, 1, C)
    norm = W.sum(axis=0, keepdims=True)                                                       # (1, C)
    norm = np.where(norm == 0, 1.0, norm)
    ensemble_probs = (P * W_bc).sum(axis=0) / norm                                           # (N, C)
    print(f"  Ensemble-Aggregation: per-class F1 gewichtet (B2, {len(probs_list)} Modelle)")
else:
    # B1: uniform Soft-Voting
    ensemble_probs = np.mean(probs_list, axis=0)
    print(f"  Ensemble-Aggregation: uniform (B1, {len(probs_list)} Modelle)")

pred_indices = ensemble_probs.argmax(axis=1)
pred_labels  = le.inverse_transform(pred_indices)

unique, counts = np.unique(pred_labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls:<20}: {cnt:>5} ({cnt/len(pred_labels)*100:.1f}%)")

# 5. Submission-CSV erstellen
print(f"\n{SEP}")
print(f"  SUBMISSION-DATEIEN")
print(f"{SEP}")

csv_path = OUT_DIR / CSV_NAME
df_out   = pd.DataFrame({"id": test_ids, TASK: pred_labels})
df_out.to_csv(csv_path, sep=";", index=False)
print(f"\n  CSV: {csv_path}")
print(df_out.head(5).to_string(index=False))

# 6. ZIP packen
zip_path = OUT_DIR / ZIP_NAME
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(csv_path, arcname=CSV_NAME)

print(f"\n  ZIP: {zip_path}")
print(f"\n{'='*70}")
print(f"  FERTIG – {ZIP_NAME} ist bereit fuer den Codabench-Upload.")
print(f"{'='*70}\n")
