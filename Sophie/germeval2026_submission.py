"""
GermEval 2026 – Inference & Submission Generator
==================================================
Lädt das beste Ensemble (A8: mdeberta + gelectra + gbert, uniform voting,
Val-Macro-F1: 0.80040) und erstellt die Submission-Dateien für Codabench.

Ausgabe (in --out_dir):
  [team][run]_dbo.csv   →  id;dbo  (Predictions auf Testdaten)
  [team][run].zip        →  Submission-ZIP für Codabench-Upload

VERWENDUNG:
    python germeval2026_submission.py
    python germeval2026_submission.py --team TUMination --run 1
    python germeval2026_submission.py --base_dir /cluster/path/model_dataset_gridsearch
    python germeval2026_submission.py --test_file /path/to/dbo_test_26.csv
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
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 – Submission")
parser.add_argument("--team",       default="TUMination",
                    help="Teamname (wie auf Codabench registriert)")
parser.add_argument("--run",        default="1",
                    help="Run-Nummer: 1, 2 oder 3 (max. 3 Submissions erlaubt)")
parser.add_argument("--base_dir",   default="model_dataset_gridsearch",
                    help="Ordner mit den trainierten Run-Unterordnern")
parser.add_argument("--train_file", default=None,
                    help="Trainingsdatei für LabelEncoder (CSV mit 'text'+'label' "
                         "oder 'description'+'dbo' Spalten). "
                         "Default: wird relativ zu base_dir gesucht.")
parser.add_argument("--test_file",  default=None,
                    help="Pfad zur dbo_test_26.csv. "
                         "Default: wird relativ zum Skript gesucht.")
parser.add_argument("--out_dir",    default="submissions",
                    help="Ausgabeverzeichnis für CSV und ZIP")
args = parser.parse_args()

BASE_DIR = Path(args.base_dir)
OUT_DIR  = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEAM     = args.team
RUN      = args.run
CSV_NAME = f"{TEAM}{RUN}_dbo.csv"
ZIP_NAME = f"{TEAM}{RUN}.zip"

# ──────────────────────────────────────────────────────────────────────────────
# Ensemble-Konfiguration A8_trio_mde_gel_gbert_uni  (Val-Macro-F1: 0.80040)
# ──────────────────────────────────────────────────────────────────────────────
ENSEMBLE = [
    ("mdeberta", "microsoft/mdeberta-v3-base",        "all5_aug-paraphrase"),
    ("gelectra", "deepset/gelectra-large-germanquad", "all5_aug-paraphrase"),
    ("gbert",    "deepset/gbert-large",               "all5_aug-paraphrase"),
]

CLASSES    = ["agitation", "criticism", "nothing", "subversive"]  # alphabetisch = LabelEncoder
MAX_LENGTH = 128
BATCH_SIZE = 16
DROPOUT    = 0.1
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def load_test_data(test_file: Path) -> pd.DataFrame:
    """Liest dbo_test_26.csv (id;description) ein."""
    for kwargs in [
        {"sep": ";"},
        {"sep": ";", "quoting": 3},
        {"sep": ";", "engine": "python"},
    ]:
        try:
            df = pd.read_csv(test_file, **kwargs)
            if "id" in df.columns and "description" in df.columns:
                return df[["id", "description"]].dropna(subset=["description"])
        except Exception:
            continue
    raise ValueError(f"Kann Testdaten nicht lesen: {test_file}")


def load_train_labels(train_file: Path) -> list:
    """Liest Trainingsdaten und gibt alle Label-Werte zurück (für LabelEncoder)."""
    for kwargs, label_col in [
        ({"sep": ";"}, "dbo"),
        ({},           "label"),
        ({"sep": ";"}, "label"),
        ({},           "dbo"),
    ]:
        try:
            df = pd.read_csv(train_file, **kwargs)
            if label_col in df.columns:
                return df[label_col].dropna().tolist()
        except Exception:
            continue
    # Fallback: Klassen sind bekannt
    print("  WARNUNG: Trainingsdaten nicht lesbar – verwende hartcodierte Klassen.")
    return CLASSES * 10


def find_test_file() -> Path:
    """Sucht dbo_test_26.csv an bekannten Stellen."""
    candidates = [
        Path(args.test_file) if args.test_file else None,
        Path(__file__).parent / "../data/GermEval2026/data/dbo/dbo_test_26.csv",
        Path(__file__).parent / "../data/preprocessed_data/test_minimal.csv",
        BASE_DIR / "../data/GermEval2026/data/dbo/dbo_test_26.csv",
    ]
    for p in candidates:
        if p and p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "dbo_test_26.csv nicht gefunden. Bitte --test_file angeben.\n"
        f"Gesucht in: {[str(p) for p in candidates if p]}"
    )


def find_train_file() -> Path:
    """Sucht Trainingsdaten für den LabelEncoder."""
    cfg_path = BASE_DIR / "all5_aug-paraphrase" / "ensemble_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Pfad aus Config (relativ zum Skript-Verzeichnis)
        rel = Path(__file__).parent / cfg["train_file"]
        if rel.exists():
            return rel.resolve()

    candidates = [
        Path(args.train_file) if args.train_file else None,
        Path(__file__).parent / "../data/preprocessed_data/train_minimal.csv",
        Path(__file__).parent / "../data/GermEval2026/data/dbo/dbo_train_26.csv",
        BASE_DIR / "../data/GermEval2026/data/dbo/dbo_train_26.csv",
    ]
    for p in candidates:
        if p and p.exists():
            return p.resolve()
    return None


@torch.no_grad()
def run_inference(model, texts, tokenizer) -> np.ndarray:
    """Gibt Softmax-Wahrscheinlichkeiten zurück: shape (N, n_classes)."""
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
SEP = "─" * 70
print(f"\n{'='*70}")
print(f"  GermEval 2026 – Submission Generator")
print(f"{'='*70}")
print(f"  Team:      {TEAM}")
print(f"  Run:       {RUN}")
print(f"  Device:    {DEVICE}")
print(f"  Ensemble:  A8_trio_mde_gel_gbert_uni  (Val-Macro-F1: 0.80040)")
print(f"  Ausgabe:   {OUT_DIR / ZIP_NAME}")
print(f"{'='*70}\n")

# 1. Testdaten laden
test_path = find_test_file()
print(f"  Testdaten: {test_path}")
df_test   = load_test_data(test_path)
test_ids  = df_test["id"].astype(str).tolist()
test_texts = df_test["description"].tolist()
print(f"  {len(test_texts)} Test-Tweets geladen.\n")

# 2. LabelEncoder aufbauen (Klassen-Reihenfolge muss mit Training übereinstimmen)
train_path = find_train_file()
if train_path:
    print(f"  Trainingsdaten für LabelEncoder: {train_path}")
    labels_for_le = load_train_labels(train_path)
else:
    print("  Trainingsdaten nicht gefunden – hartcodierte Klassen.")
    labels_for_le = CLASSES * 10

le = LabelEncoder()
le.fit(labels_for_le)
print(f"  Klassen: {list(le.classes_)}\n")

n_classes = len(le.classes_)

# 3. Inferenz für jedes Ensemble-Modell
print(f"{SEP}")
print(f"  INFERENZ")
print(f"{SEP}")

probs_list = []
for model_name, model_id, run_folder in ENSEMBLE:
    ckpt_path = BASE_DIR / run_folder / f"model_{model_name}" / "best_model_weights.pt"
    print(f"\n  Lade {model_name:<12} ← {ckpt_path}")

    if not ckpt_path.exists():
        print(f"  FEHLER: Checkpoint nicht gefunden: {ckpt_path}")
        print(f"  Überspringe {model_name}.")
        continue

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = TransformerClassifier(model_id, n_classes, DROPOUT).to(DEVICE)
    state     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in state.items()})

    probs = run_inference(model, test_texts, tokenizer)
    probs_list.append(probs)
    print(f"  {model_name}: Predictions shape {probs.shape}  "
          f"(most common: {le.classes_[probs.argmax(axis=1)].tolist().count(le.classes_[probs.argmax(axis=1)][0])} ...)")

    del model
    torch.cuda.empty_cache()

if not probs_list:
    raise RuntimeError("Keine Checkpoints geladen – bitte Pfade prüfen.")

# 4. Uniform Voting
print(f"\n{SEP}")
print(f"  ENSEMBLE-AGGREGATION (uniform voting über {len(probs_list)} Modelle)")
print(f"{SEP}")

ensemble_probs = np.mean(probs_list, axis=0)          # (N, n_classes)
pred_indices   = ensemble_probs.argmax(axis=1)         # (N,)
pred_labels    = le.inverse_transform(pred_indices)    # (N,) → Klassenname

# Verteilung ausgeben
unique, counts = np.unique(pred_labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls:<12}: {cnt:>5} ({cnt/len(pred_labels)*100:.1f}%)")

# 5. Submission-CSV erstellen
print(f"\n{SEP}")
print(f"  SUBMISSION-DATEIEN")
print(f"{SEP}")

csv_path = OUT_DIR / CSV_NAME
df_out   = pd.DataFrame({"id": test_ids, "dbo": pred_labels})
df_out.to_csv(csv_path, sep=";", index=False)
print(f"\n  CSV gespeichert: {csv_path}")
print(f"  Vorschau:")
print(df_out.head(5).to_string(index=False))

# 6. ZIP packen
zip_path = OUT_DIR / ZIP_NAME
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(csv_path, arcname=CSV_NAME)

print(f"\n  ZIP gespeichert: {zip_path}")
print(f"  Enthält: {CSV_NAME}")

print(f"\n{'='*70}")
print(f"  FERTIG – {ZIP_NAME} ist bereit für den Codabench-Upload.")
print(f"{'='*70}\n")
