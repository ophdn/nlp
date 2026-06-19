"""
GermEval 2026 – Ensemble Comparison
=====================================
Lädt die besten bekannten Checkpoints pro Modell (aus Analyse von
final_report_5.txt) und testet 17 Ensemble-Kombinationen:
  - verschiedene Modell-Subsets  (All-5, Top-4, Trios, Duo, Einzelmodelle)
  - vier Gewichtungsstrategien   (uniform, per-class F1, globale Macro-F1, winner-takes-all)
  - Checkpoints aus para / both / gemischt (jedes Modell sein persönliches Bestes)

Bester Checkpoint pro Modell (aus Analyse):
  gbert    → all5_aug-paraphrase  (Macro-F1 0.7277)
  xlmr     → all5_aug-none        (Macro-F1 0.5534) ← Augmentierung schadet
  deberta  → all5_aug-paraphrase  (Macro-F1 0.7347)
  mdeberta → all5_aug-paraphrase  (Macro-F1 0.7885)
  gelectra → all5_aug-paraphrase  (Macro-F1 0.7863)

Referenzwerte (aus final_report_5.txt):
  All-5 uniform paraphrase  → 0.7737
  All-5 uniform both        → 0.7955  ← bisher bestes bekanntes Ergebnis

VERWENDUNG:
    python germeval2026_ensemble_comparison.py
    python germeval2026_ensemble_comparison.py --base_dir model_dataset_gridsearch
    python germeval2026_ensemble_comparison.py --out ensemble_comparison_report.txt
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 – Ensemble Comparison")
parser.add_argument("--base_dir", default="model_dataset_gridsearch",
                    help="Ordner mit den Run-Unterordnern")
parser.add_argument("--out", default=None,
                    help="Ausgabedatei für den Report (default: base_dir/ensemble_comparison_report.txt)")
args = parser.parse_args()

BASE_DIR = Path(args.base_dir)
OUT_PATH = Path(args.out) if args.out else BASE_DIR / "ensemble_comparison_report.txt"

# ──────────────────────────────────────────────────────────────────────────────
# Modell-Registry
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "gbert":    "deepset/gbert-large",
    "xlmr":     "FacebookAI/xlm-roberta-large",
    "deberta":  "microsoft/deberta-v3-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "gelectra": "deepset/gelectra-large-germanquad",
}

# ──────────────────────────────────────────────────────────────────────────────
# Stärken/Schwächen pro Modell (beste Checkpoints, für Report-Narration)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NOTES = {
    "gbert": (
        "para",
        "Guter Allrounder. Stärke: agitation (0.72), solides subversive (0.62). "
        "Schwäche: schwächstes subversive der Top-Modelle."
    ),
    "xlmr": (
        "none",
        "Schwächstes Modell. Stärke: multilinguale Diversität, agitation 0.44 ohne Aug. "
        "Schwäche: bricht bei jeder Augmentierung komplett ein (agit/subv → 0.0 mit para). "
        "Checkpoint aus aug-none nehmen."
    ),
    "deberta": (
        "para",
        "Stark bei seltenen Klassen. Stärke: subversive 0.73, gute Balance. "
        "Schwäche: agitation 0.67 (hinter gbert/mdeberta/gelectra)."
    ),
    "mdeberta": (
        "para",
        "Bestes Einzelmodell. Stärke: subversive 0.889 (gemeinsam bestes), agitation 0.73. "
        "Schwäche: ähnlich wie gelectra → hohe Korrelation, wenig Diversitätsgewinn."
    ),
    "gelectra": (
        "para",
        "Gleich stark wie mdeberta. Stärke: subversive 0.889, agitation 0.73. "
        "Schwäche: Germano-spezifisch (germanquad pre-training) → weniger cross-linguale Diversität."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint-Mapping: (model, run_key) → run-Unterordner
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_MAP = {
    ("gbert",    "para"): "all5_aug-paraphrase",
    ("xlmr",     "para"): "all5_aug-paraphrase",
    ("deberta",  "para"): "all5_aug-paraphrase",
    ("mdeberta", "para"): "all5_aug-paraphrase",
    ("gelectra", "para"): "all5_aug-paraphrase",
    ("gbert",    "both"): "all5_aug-both",
    ("xlmr",     "both"): "all5_aug-both",
    ("deberta",  "both"): "all5_aug-both",
    ("mdeberta", "both"): "all5_aug-both",
    ("gelectra", "both"): "all5_aug-both",
    ("xlmr",     "none"): "all5_aug-none",
    ("gbert",    "3para"):   "3_aug-paraphrase",
    ("gelectra", "3para"):   "3_aug-paraphrase",
    ("mdeberta", "3para"):   "3_aug-paraphrase",
    ("gbert",    "a8final"): "A8_trio_final_submission",
    ("gelectra", "a8final"): "A8_trio_final_submission",
    ("mdeberta", "a8final"): "A8_trio_final_submission",
}

# ──────────────────────────────────────────────────────────────────────────────
# Ensemble-Konfigurationen
# Format: (name, [(model, run_key), ...], weighting_strategy)
# weighting: "uniform" | "per_class" | "macro_f1"
# ──────────────────────────────────────────────────────────────────────────────
_P   = "para"
_B   = "both"
_N   = "none"
_3P  = "3para"
_A8F = "a8final"

_ALL5_P   = [("gbert",_P),("xlmr",_P),("deberta",_P),("mdeberta",_P),("gelectra",_P)]
_TOP4_P   = [("gbert",_P),("deberta",_P),("mdeberta",_P),("gelectra",_P)]
_TRIO1_P  = [("mdeberta",_P),("gelectra",_P),("deberta",_P)]     # beste 3 Modelle (para)
_TRIO2_P  = [("mdeberta",_P),("gelectra",_P),("gbert",_P)]       # mdeberta+gelectra + Diversität
_DUO_P    = [("mdeberta",_P),("gelectra",_P)]                     # die zwei Besten
_ALL5_B   = [("gbert",_B),("xlmr",_B),("deberta",_B),("mdeberta",_B),("gelectra",_B)]
_TOP4_B   = [("gbert",_B),("deberta",_B),("mdeberta",_B),("gelectra",_B)]
_ALL5_MIX  = [("gbert",_P),("xlmr",_N),("deberta",_P),("mdeberta",_P),("gelectra",_P)]
_TRIO_3P   = [("gbert",_3P),("gelectra",_3P),("mdeberta",_3P)]
_TRIO_A8F  = [("gbert",_A8F),("gelectra",_A8F),("mdeberta",_A8F)]

ENSEMBLE_CONFIGS = [
    # ── Einzelmodell-Baselines ─────────────────────────────────────────────────
    ("S1_mdeberta_para",          [("mdeberta",_P)],  "uniform"),
    ("S2_gelectra_para",          [("gelectra",_P)],  "uniform"),
    ("S3_deberta_para",           [("deberta",_P)],   "uniform"),
    # ── Gruppe A: alle Checkpoints aus paraphrase-Run ─────────────────────────
    ("A1_all5_uniform_para",      _ALL5_P,   "uniform"),   # Referenz 0.7737
    ("A2_all5_perclass_para",     _ALL5_P,   "per_class"),
    ("A3_all5_macroF1_para",      _ALL5_P,   "macro_f1"),
    ("A4_top4_noxlmr_uniform",    _TOP4_P,   "uniform"),
    ("A5_top4_noxlmr_perclass",   _TOP4_P,   "per_class"),
    ("A6_trio_mde_gel_deb_uni",   _TRIO1_P,  "uniform"),
    ("A7_trio_mde_gel_deb_pcls",  _TRIO1_P,  "per_class"),
    ("A8_trio_mde_gel_gbert_uni", _TRIO2_P,  "uniform"),
    ("A9_duo_mde_gel",            _DUO_P,    "uniform"),
    # ── Gruppe B: alle Checkpoints aus both-Run ───────────────────────────────
    ("B1_all5_uniform_both",      _ALL5_B,   "uniform"),   # Referenz 0.7955
    ("B2_all5_perclass_both",     _ALL5_B,   "per_class"),
    ("B3_top4_noxlmr_both_uni",   _TOP4_B,   "uniform"),
    ("B4_top4_noxlmr_both_pcls",  _TOP4_B,   "per_class"),
    # ── Gruppe C: Mixed – jedes Modell aus seinem persönlich besten Checkpoint ─
    ("C1_all5_uniform_mixed",     _ALL5_MIX, "uniform"),
    ("C2_all5_perclass_mixed",    _ALL5_MIX, "per_class"),
    # ── Gruppe D: Winner-takes-all – bestes Modell pro Klasse entscheidet absolut ─
    # Erwartetes Routing (para): agit→mdeberta/gelectra, crit→gbert, subv→mdeberta/gelectra
    ("D1_all5_winner_para",       _ALL5_P,   "winner"),
    ("D2_top4_winner_para",       _TOP4_P,   "winner"),
    ("D3_all5_winner_both",       _ALL5_B,   "winner"),
    ("D4_top4_winner_both",       _TOP4_B,   "winner"),
    ("D5_all5_winner_mixed",      _ALL5_MIX, "winner"),
    ("D6_trio_winner_para",       _TRIO1_P,  "winner"),
    # ── Gruppe E: Trio aus 3_aug-paraphrase ───────────────────────────────────
    ("E1_trio_3para_uniform",     _TRIO_3P,  "uniform"),
    ("E2_trio_3para_perclass",    _TRIO_3P,  "per_class"),
    ("E3_trio_3para_macroF1",     _TRIO_3P,  "macro_f1"),
    ("E4_trio_3para_winner",      _TRIO_3P,  "winner"),
    # ── Gruppe F: Trio aus A8_trio_final_submission ───────────────────────────
    ("F1_trio_a8final_uniform",   _TRIO_A8F, "uniform"),
    ("F2_trio_a8final_perclass",  _TRIO_A8F, "per_class"),
    ("F3_trio_a8final_macroF1",   _TRIO_A8F, "macro_f1"),
    ("F4_trio_a8final_winner",    _TRIO_A8F, "winner"),
]

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


class DBODataset(Dataset):
    def __init__(self, texts, labels_encoded, tokenizer, max_length):
        self.texts      = texts
        self.labels     = torch.tensor(labels_encoded, dtype=torch.long)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self._cache     = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx not in self._cache:
            enc = self.tokenizer(
                self.texts[idx], truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt",
            )
            self._cache[idx] = {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
        item = self._cache[idx].copy()
        item["label"] = self.labels[idx]
        return item


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["text", "label"])
    for kwargs in [{}, {"quoting": 3}, {"engine": "python"},
                   {"engine": "python", "on_bad_lines": "skip"}]:
        try:
            df = pd.read_csv(p, **kwargs)
            if "text" in df.columns and "label" in df.columns:
                return df[["text", "label"]].dropna()
        except Exception:
            continue
    return pd.DataFrame(columns=["text", "label"])


def run_inference(model, texts, tokenizer, batch_size, max_length, device, labels_enc):
    dataset = DBODataset(texts, labels_enc, tokenizer, max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs  = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def compute_ensemble(probs_list, weights_uniform, weights_per_class, weights_macro, strategy):
    """
    Berechnet Ensemble-Wahrscheinlichkeiten nach Strategie.

    strategy: "uniform"   → einfaches Mittel der Modell-Probs
              "per_class" → Gewichtung nach per-class F1 jedes Modells
              "macro_f1"  → Gewichtung nach globalem Macro-F1 jedes Modells
              "winner"    → pro Klasse entscheidet nur das Modell mit höchstem F1 dort
                            (bei Gleichstand: gleiche Stimme für alle Erstplatzierten)
    """
    if strategy == "uniform":
        return np.mean(probs_list, axis=0)

    if strategy == "per_class":
        # weights_per_class: list of arrays (n_classes,)
        W = np.array(weights_per_class)      # (M, C)
        P = np.stack(probs_list, axis=0)     # (M, N, C)
        W_bc = W[:, np.newaxis, :]           # (M, 1, C)
        weighted = (P * W_bc).sum(axis=0)    # (N, C)
        norm = W.sum(axis=0)[np.newaxis, :]  # (1, C)
        norm = np.where(norm == 0, 1.0, norm)
        return weighted / norm

    if strategy == "macro_f1":
        # weights_macro: list of scalars
        W = np.array(weights_macro)          # (M,)
        W_bc = W[:, np.newaxis, np.newaxis]  # (M, 1, 1)
        P = np.stack(probs_list, axis=0)     # (M, N, C)
        w_sum = W.sum()
        if w_sum == 0:
            return np.mean(probs_list, axis=0)
        return (P * W_bc).sum(axis=0) / w_sum

    if strategy == "winner":
        # Pro Klasse k: nur das/die Modell(e) mit max per-class F1[k] zählen.
        # Bei Gleichstand (z.B. mdeberta und gelectra bei subversive) teilen sie sich die Stimme.
        W = np.array(weights_per_class)                    # (M, C)
        max_per_class = W.max(axis=0, keepdims=True)       # (1, C)
        mask = (W == max_per_class).astype(float)          # (M, C) — 1 beim Besten, 0 sonst
        # Bei Klassen wo alle Modelle F1=0: uniform fallback
        all_zero = (max_per_class == 0).squeeze(0)         # (C,)
        mask[:, all_zero] = 1.0
        norm = mask.sum(axis=0, keepdims=True)             # (1, C)
        mask = mask / norm                                 # normalisiert → Gleichstand = 0.5/0.5
        P    = np.stack(probs_list, axis=0)                # (M, N, C)
        W_bc = mask[:, np.newaxis, :]                      # (M, 1, C)
        return (P * W_bc).sum(axis=0)                      # (N, C)

    raise ValueError(f"Unbekannte Strategie: {strategy}")


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*70}")
print(f"  GermEval 2026 – Ensemble Comparison")
print(f"{'='*70}")
print(f"  Device:    {DEVICE}")
print(f"  Konfigurationen: {len(ENSEMBLE_CONFIGS)}")
print(f"  Ausgabe:   {OUT_PATH}")
print(f"{'='*70}\n")

# Referenz-Config laden (paraphrase, seed=42 → identisches Val-Set für alle Runs)
ref_run = BASE_DIR / "all5_aug-paraphrase"
with open(ref_run / "ensemble_config.json", encoding="utf-8") as f:
    ref_cfg = json.load(f)

SEED       = ref_cfg["seed"]
MAX_LENGTH = ref_cfg["max_length"]
BATCH_SIZE = ref_cfg["batch_size"]
DROPOUT    = ref_cfg["dropout"]
VAL_SIZE   = ref_cfg["val_size"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Val-Set laden (aus Basis-Trainingsdaten, augmentierungsunabhängig)
ref_train_file = Path(__file__).parent / ref_cfg["train_file"]
df_main = load_csv(ref_train_file)
if len(df_main) == 0:
    # Fallback: relativer Pfad vom base_dir
    df_main = load_csv(BASE_DIR / ref_cfg["train_file"])
if len(df_main) == 0:
    raise FileNotFoundError(
        f"Trainingsdaten nicht gefunden: {ref_cfg['train_file']}\n"
        f"Skript-Verzeichnis: {Path(__file__).parent}"
    )

_, df_val = train_test_split(
    df_main, test_size=VAL_SIZE, stratify=df_main["label"], random_state=SEED
)

le = LabelEncoder()
le.fit(df_main["label"])
CLASSES   = list(le.classes_)
N_CLASSES = len(CLASSES)
val_labels_enc = le.transform(df_val["label"])
val_texts      = df_val["text"].tolist()

print(f"  Val-Set:   {len(df_val)} Samples")
print(f"  Klassen:   {CLASSES}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Schritt 1: Inferenz für alle benötigten (model, run_key)-Kombinationen
# ──────────────────────────────────────────────────────────────────────────────
needed_keys = set()
for _, members, _ in ENSEMBLE_CONFIGS:
    needed_keys.update(members)

print(f"  Benötigte Checkpoints: {len(needed_keys)}")
for k in sorted(needed_keys):
    print(f"    {k[0]:<12} ← {CHECKPOINT_MAP[k]}")

print(f"\n{'─'*70}")
print(f"  INFERENZ-PHASE")
print(f"{'─'*70}")

inference_cache = {}

for model_key in sorted(needed_keys, key=lambda x: (x[1], x[0])):
    model_name, run_key = model_key
    run_dir   = BASE_DIR / CHECKPOINT_MAP[model_key]
    ckpt_path = run_dir / f"model_{model_name}" / "best_model_weights.pt"

    if not ckpt_path.exists():
        print(f"  ERROR: Checkpoint fehlt: {ckpt_path}")
        print(f"  Überspringe {model_key}.")
        continue

    print(f"\n  Lade {model_name:<12} ({run_key:<6})  ← {ckpt_path.parent.parent.name}")
    model_id  = MODEL_REGISTRY[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = TransformerClassifier(model_id, N_CLASSES, DROPOUT).to(DEVICE)
    state     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in state.items()})

    probs, preds, labels = run_inference(
        model, val_texts, tokenizer, BATCH_SIZE, MAX_LENGTH, DEVICE, val_labels_enc
    )

    macro_f1     = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None,
                             zero_division=0, labels=list(range(N_CLASSES)))

    print(f"    Macro-F1: {macro_f1:.5f}  "
          + "  ".join(f"{c}: {f:.3f}" for c, f in zip(CLASSES, per_class_f1)))

    inference_cache[model_key] = {
        "probs":         probs,
        "per_class_f1":  per_class_f1,
        "macro_f1":      macro_f1,
        "val_labels":    labels,
    }

    del model
    torch.cuda.empty_cache()

missing_keys = needed_keys - set(inference_cache.keys())
if missing_keys:
    print(f"\n  WARNUNG: Folgende Checkpoints fehlen und werden übersprungen: {missing_keys}")

# ──────────────────────────────────────────────────────────────────────────────
# Schritt 2: Ensemble-Evaluation
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  ENSEMBLE-EVALUATION")
print(f"{'─'*70}")

results = []

for config_name, members, strategy in ENSEMBLE_CONFIGS:
    # Filter auf verfügbare Checkpoints
    available = [(m, r) for m, r in members if (m, r) in inference_cache]
    if not available:
        print(f"  SKIP {config_name}: keine Checkpoints verfügbar.")
        continue
    if len(available) < len(members):
        missing = set(members) - set(available)
        print(f"  WARNUNG {config_name}: {missing} fehlt, verwende {len(available)} Modelle.")

    probs_list      = [inference_cache[k]["probs"]        for k in available]
    per_class_list  = [inference_cache[k]["per_class_f1"] for k in available]
    macro_list      = [inference_cache[k]["macro_f1"]     for k in available]
    val_labels_arr  = inference_cache[available[0]]["val_labels"]

    ens_probs = compute_ensemble(probs_list, None, per_class_list, macro_list, strategy)
    ens_preds = ens_probs.argmax(axis=1)

    macro_f1     = f1_score(val_labels_arr, ens_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(val_labels_arr, ens_preds, average=None,
                             zero_division=0, labels=list(range(N_CLASSES)))
    cls_report   = classification_report(val_labels_arr, ens_preds,
                                          target_names=CLASSES, zero_division=0)

    print(f"  {config_name:<38} Macro-F1: {macro_f1:.5f}  "
          + "  ".join(f"{c}: {f:.3f}" for c, f in zip(CLASSES, per_class_f1)))

    results.append({
        "name":       config_name,
        "models":     "+".join(m for m, _ in available),
        "n_models":   len(available),
        "strategy":   strategy,
        "macro_f1":   macro_f1,
        "cls_report": cls_report,
        **{f"f1_{c}": float(f) for c, f in zip(CLASSES, per_class_f1)},
    })

# Sortieren nach Macro-F1
results.sort(key=lambda x: x["macro_f1"], reverse=True)

# ──────────────────────────────────────────────────────────────────────────────
# Schritt 3: Report schreiben
# ──────────────────────────────────────────────────────────────────────────────
SEP  = "─" * 70
SEP2 = "=" * 70

# Modell-Stärken/Schwächen-Abschnitt
strengths_section = f"""
{SEP2}
  STÄRKEN & SCHWÄCHEN DER BESTEN CHECKPOINTS
{SEP2}
"""
for m, (run_key, note) in MODEL_NOTES.items():
    run_dir_name = CHECKPOINT_MAP.get((m, run_key), "unbekannt")
    f1_val = inference_cache.get((m, run_key), {}).get("macro_f1", None)
    pcls   = inference_cache.get((m, run_key), {}).get("per_class_f1", None)
    f1_str = f"{f1_val:.5f}" if f1_val is not None else "n/a"
    if pcls is not None:
        pcls_str = "  ".join(f"{c}: {v:.3f}" for c, v in zip(CLASSES, pcls))
    else:
        pcls_str = "n/a"
    strengths_section += (
        f"\n  {m.upper():<12} [{run_dir_name}]\n"
        f"    Macro-F1: {f1_str}\n"
        f"    Per-Klasse: {pcls_str}\n"
        f"    Analyse:  {note}\n"
    )

# Vergleichs-Tabelle
table_header = (
    f"\n{SEP2}\n"
    f"  ENSEMBLE-VERGLEICH (sortiert nach Macro-F1)\n"
    f"{SEP2}\n"
    f"  {'Rang':<5} {'Konfiguration':<38} {'Strategie':<12} {'#':<3} "
    + "  ".join(f"{c[:4]:<6}" for c in CLASSES)
    + f"  {'Macro-F1':>10}\n"
    f"  {SEP}\n"
)

table_rows = ""
for rank, r in enumerate(results, 1):
    cls_vals = "  ".join(f"{r[f'f1_{c}']:.4f}" for c in CLASSES)
    marker = "  ← BEST" if rank == 1 else ("  ← Ref" if r["name"] in ("A1_all5_uniform_para","B1_all5_uniform_both") else "")
    table_rows += (
        f"  {rank:<5} {r['name']:<38} {r['strategy']:<12} {r['n_models']:<3} "
        f"{cls_vals}  {r['macro_f1']:.5f}{marker}\n"
    )

# Per-class F1 Gewichtungsmatrix für gemischte Checkpoints
weight_section = f"\n{SEP}\n  GEWICHTUNGSMATRIZEN (Val-F1 pro Modell × Klasse)\n{SEP}\n"
header_row = f"  {'Modell':<14}" + "".join(f"{c:>12}" for c in CLASSES) + f"{'Macro-F1':>12}\n"

for run_key in ["para", "both", "none", "3para", "a8final"]:
    weight_section += f"\n  [{run_key}]\n" + header_row + f"  {'─'*60}\n"
    for model_name in ["gbert", "xlmr", "deberta", "mdeberta", "gelectra"]:
        mk = (model_name, run_key)
        if mk in inference_cache:
            pcls    = inference_cache[mk]["per_class_f1"]
            macro_v = inference_cache[mk]["macro_f1"]
            row     = f"  {model_name:<14}" + "".join(f"{v:>12.3f}" for v in pcls) + f"{macro_v:>12.5f}\n"
            weight_section += row

# Beste Ensemble-Konfiguration Detail
best = results[0]
detail_section = (
    f"\n{SEP2}\n"
    f"  BESTE KONFIGURATION: {best['name']}\n"
    f"{SEP2}\n"
    f"  Modelle:   {best['models']}\n"
    f"  Strategie: {best['strategy']}\n"
    f"  Macro-F1:  {best['macro_f1']:.5f}\n\n"
    f"{best['cls_report']}"
)

# Empfehlungsabschnitt
ref_b1 = next((r for r in results if r["name"] == "B1_all5_uniform_both"), None)
ref_a1 = next((r for r in results if r["name"] == "A1_all5_uniform_para"), None)

recommend_lines = [
    f"\n{SEP2}",
    f"  EMPFEHLUNG",
    f"{SEP2}",
    f"  Beste Konfiguration:      {best['name']:<38} Macro-F1: {best['macro_f1']:.5f}",
]
if ref_b1:
    delta = best["macro_f1"] - ref_b1["macro_f1"]
    recommend_lines.append(
        f"  vs. Referenz B1 (both):   {ref_b1['name']:<38} Macro-F1: {ref_b1['macro_f1']:.5f}  ({delta:+.5f})"
    )
if ref_a1:
    delta = best["macro_f1"] - ref_a1["macro_f1"]
    recommend_lines.append(
        f"  vs. Referenz A1 (para):   {ref_a1['name']:<38} Macro-F1: {ref_a1['macro_f1']:.5f}  ({delta:+.5f})"
    )

# Frage: Ist All-5 wirklich das Beste?
all5_uniform_both = next((r for r in results if r["name"] == "B1_all5_uniform_both"), None)
top4_both         = next((r for r in results if r["name"] == "B3_top4_noxlmr_both_uni"), None)
if all5_uniform_both and top4_both:
    diff = top4_both["macro_f1"] - all5_uniform_both["macro_f1"]
    if diff > 0:
        recommend_lines.append(
            f"\n  BEFUND: Top-4 ohne xlmr ({top4_both['name']}) schlägt All-5 (both) "
            f"um {diff:+.5f} → xlmr schadet dem Ensemble!"
        )
    else:
        recommend_lines.append(
            f"\n  BEFUND: All-5 (both) ist besser als Top-4 ohne xlmr um {-diff:.5f} "
            f"→ xlmr trägt netto positiv bei."
        )

recommend_lines += [
    "",
    f"  Für Checkpoint-Auswahl beim nächsten Training empfohlen:",
    f"    → {best['name']} als Basis-Ensemble-Strategie",
    f"{SEP2}",
]
recommend_section = "\n".join(recommend_lines)

# Zusammensetzen
report = (
    f"GermEval 2026 – Ensemble Comparison Report\n"
    f"{SEP2}\n"
    f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    f"Konfigurationen getestet: {len(results)}\n"
    f"Val-Set:     {len(df_val)} Samples  |  Klassen: {CLASSES}\n"
    f"{strengths_section}"
    f"{weight_section}"
    f"{table_header}{table_rows}"
    f"{detail_section}"
    f"{recommend_section}\n"
)

OUT_PATH.write_text(report, encoding="utf-8")
print(f"\n{'='*70}")
print(f"  ABGESCHLOSSEN")
print(f"{'='*70}")
print(f"\n  {'Rang':<5} {'Konfiguration':<38} {'Macro-F1':>10}")
print(f"  {'─'*55}")
for rank, r in enumerate(results[:10], 1):
    marker = "  ← BEST" if rank == 1 else ""
    print(f"  {rank:<5} {r['name']:<38} {r['macro_f1']:>10.5f}{marker}")
print(f"\n  Report gespeichert: {OUT_PATH}")
