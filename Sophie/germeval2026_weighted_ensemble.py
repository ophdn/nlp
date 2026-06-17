"""
GermEval 2026 – Weighted Ensemble (per-class Macro-F1 Gewichtung)
==================================================================
Kein Neutraining. Lädt Checkpoints aus abgeschlossenen Runs und berechnet
ein gewichtetes Soft-Voting Ensemble, bei dem jedes Modell pro Klasse
mit seinem Val-F1 für diese Klasse gewichtet wird.

Idee:
  Normales Soft-Voting:   ensemble_prob = mean(probs_model_i)
  Gewichtetes Soft-Voting: ensemble_prob = sum(w_i * probs_model_i) / sum(w_i)
  wobei w_i[k] = F1-Score von Modell i auf Klasse k (Val-Set)

Dadurch zieht z.B. xlmr (F1=0.0 auf subversive) bei subversive kaum mit,
während deberta (F1=0.727) dort dominant ist.

VERWENDUNG:
    # Alle Runs mit verfügbaren Checkpoints:
    python germeval2026_weighted_ensemble.py

    # Nur einen Run:
    python germeval2026_weighted_ensemble.py --run_dirs model_dataset_gridsearch/all5_aug-paraphrase

    # Nur 3 bestehende Modelle verwenden (falls mdeberta/gelectra noch nicht fertig):
    python germeval2026_weighted_ensemble.py --models gbert xlmr deberta

AUSGABE (pro Run-Ordner):
    weighted_ensemble_report.txt   – Vergleich uniform vs. gewichtet
    predictions_test_weighted.csv  – Testvorhersagen mit gewichtetem Ensemble
"""

import argparse
import json
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
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 Weighted Ensemble")
parser.add_argument("--base_dir",  default="model_dataset_gridsearch")
parser.add_argument("--run_dirs",  nargs="+", default=None)
parser.add_argument("--models",    nargs="+", default=None,
                    help="Welche Modelle verwenden (default: alle verfügbaren Checkpoints)")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Run-Ordner ermitteln (paraphrase zuerst)
# ──────────────────────────────────────────────────────────────────────────────
if args.run_dirs:
    run_dirs = [Path(d) for d in args.run_dirs]
else:
    AUG_ORDER = ["paraphrase", "both", "none", "generated"]
    base      = Path(args.base_dir)
    all_dirs  = [p for p in base.iterdir()
                 if p.is_dir() and (p / "ensemble_config.json").exists()]
    def aug_sort_key(p):
        for i, aug in enumerate(AUG_ORDER):
            if aug in p.name:
                return i
        return len(AUG_ORDER)
    run_dirs = sorted(all_dirs, key=aug_sort_key)

if not run_dirs:
    print(f"ERROR: Keine Run-Ordner mit ensemble_config.json in '{args.base_dir}'")
    raise SystemExit(1)

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
    for kwargs in [{}, {"quoting": 3}, {"engine": "python"}, {"engine": "python", "on_bad_lines": "skip"}]:
        try:
            df = pd.read_csv(p, **kwargs)
            if "text" in df.columns and "label" in df.columns:
                return df[["text", "label"]].dropna()
        except Exception:
            continue
    return pd.DataFrame(columns=["text", "label"])


def run_inference(model, texts, tokenizer, batch_size, max_length, device, labels_encoded=None):
    """Val-Inferenz (mit Labels) oder Test-Inferenz (ohne Labels)."""
    dataset = DBODataset(texts, labels_encoded if labels_encoded is not None else [0]*len(texts),
                         tokenizer, max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs  = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            if labels_encoded is not None:
                all_labels.extend(batch["label"].numpy())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels) if labels_encoded is not None else None


def weighted_soft_voting(probs_per_model, per_class_f1_per_model, n_classes):
    """
    Gewichtetes Soft-Voting: Klasse k wird von Modell i mit dessen F1[k] gewichtet.

    probs_per_model:       list of arrays (n_samples, n_classes)
    per_class_f1_per_model: list of arrays (n_classes,) — Val-F1 pro Klasse pro Modell
    """
    # weight_matrix: (n_models, n_classes)
    weight_matrix = np.array(per_class_f1_per_model)           # (M, C)

    # Für jede Klasse k: ensemble_prob[:, k] = sum_i(w_i_k * probs_i[:, k]) / sum_i(w_i_k)
    probs_stack = np.stack(probs_per_model, axis=0)             # (M, N, C)
    weights_bc  = weight_matrix[:, np.newaxis, :]               # (M, 1, C)
    weighted    = (probs_stack * weights_bc).sum(axis=0)        # (N, C)
    norm        = weight_matrix.sum(axis=0)[np.newaxis, :]      # (1, C)
    # Klassen ohne F1-Signal (alle Modelle F1=0) → uniform fallback
    norm        = np.where(norm == 0, 1.0, norm)
    return weighted / norm                                       # (N, C)


# ──────────────────────────────────────────────────────────────────────────────
# Haupt-Loop
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*65}")
print(f"  GermEval 2026 – Weighted Ensemble (kein Neutraining)")
print(f"{'='*65}")
print(f"  Device: {DEVICE}")
print(f"  Runs:   {len(run_dirs)}\n")

summary_rows = []

for run_dir in run_dirs:
    run_name = run_dir.name
    print(f"\n{'='*65}")
    print(f"  RUN: {run_name}")
    print(f"{'='*65}")

    with open(run_dir / "ensemble_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    seed        = cfg["seed"]
    max_length  = cfg["max_length"]
    batch_size  = cfg["batch_size"]
    dropout     = cfg["dropout"]
    augmentation = cfg["augmentation"]

    # Verfügbare Checkpoints bestimmen
    if args.models:
        model_names = args.models
    else:
        model_names = [m for m in MODEL_REGISTRY
                       if (run_dir / f"model_{m}" / "best_model_weights.pt").exists()]

    if not model_names:
        print(f"  SKIP: Keine Checkpoints gefunden.")
        continue

    print(f"  Modelle: {model_names}")

    # Daten laden (gleicher Seed → gleicher Val-Split wie beim Training)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_main = load_csv(cfg["train_file"])
    df_test  = load_csv(cfg["test_file"])
    df_para  = load_csv(cfg["paraphrase_file"]) if augmentation in ("paraphrase", "both") else pd.DataFrame(columns=["text","label"])
    df_gen   = load_csv(cfg["generated_file"])  if augmentation in ("generated",  "both") else pd.DataFrame(columns=["text","label"])

    df_train_base, df_val = train_test_split(
        df_main, test_size=cfg["val_size"], stratify=df_main["label"], random_state=seed
    )

    le = LabelEncoder()
    le.fit(df_main["label"])
    CLASSES   = list(le.classes_)
    N_CLASSES = len(CLASSES)

    val_labels_encoded = le.transform(df_val["label"])

    # ── Inferenz für jedes Modell ──────────────────────────────────────────────
    all_val_probs       = []
    all_per_class_f1    = []
    all_macro_f1        = []
    model_names_ok      = []

    for shortname in model_names:
        model_id     = MODEL_REGISTRY[shortname]
        weights_path = run_dir / f"model_{shortname}" / "best_model_weights.pt"

        if not weights_path.exists():
            print(f"  WARNING: {weights_path} nicht gefunden — übersprungen.")
            continue

        print(f"  Lade {shortname} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model     = TransformerClassifier(model_id, N_CLASSES, dropout).to(DEVICE)
        state     = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.float() for k, v in state.items()})

        val_probs, val_preds, val_lbl = run_inference(
            model, df_val["text"].tolist(), tokenizer,
            batch_size, max_length, DEVICE, val_labels_encoded
        )

        macro_f1     = f1_score(val_lbl, val_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(val_lbl, val_preds, average=None,
                                zero_division=0, labels=list(range(N_CLASSES)))

        print(f"    {shortname:<12} Macro-F1: {macro_f1:.5f}  "
              + "  ".join(f"{cls}: {f:.3f}" for cls, f in zip(CLASSES, per_class_f1)))

        all_val_probs.append(val_probs)
        all_per_class_f1.append(per_class_f1)
        all_macro_f1.append(macro_f1)
        model_names_ok.append(shortname)

        del model
        torch.cuda.empty_cache()

    if not model_names_ok:
        print(f"  SKIP: Keine Checkpoints geladen.")
        continue

    val_labels_arr = val_labels_encoded

    # ── Uniform Soft-Voting (Baseline) ────────────────────────────────────────
    uniform_probs = np.mean(all_val_probs, axis=0)
    uniform_preds = uniform_probs.argmax(axis=1)
    uniform_macro = f1_score(val_labels_arr, uniform_preds, average="macro", zero_division=0)
    uniform_pcls  = f1_score(val_labels_arr, uniform_preds, average=None,
                              zero_division=0, labels=list(range(N_CLASSES)))
    uniform_report = classification_report(val_labels_arr, uniform_preds,
                                            target_names=CLASSES, zero_division=0)

    # ── Gewichtetes Soft-Voting ────────────────────────────────────────────────
    weighted_probs = weighted_soft_voting(all_val_probs, all_per_class_f1, N_CLASSES)
    weighted_preds = weighted_probs.argmax(axis=1)
    weighted_macro = f1_score(val_labels_arr, weighted_preds, average="macro", zero_division=0)
    weighted_pcls  = f1_score(val_labels_arr, weighted_preds, average=None,
                               zero_division=0, labels=list(range(N_CLASSES)))
    weighted_report = classification_report(val_labels_arr, weighted_preds,
                                             target_names=CLASSES, zero_division=0)

    delta = weighted_macro - uniform_macro

    print(f"\n  {'─'*60}")
    print(f"  ERGEBNISSE (Val-Set):")
    print(f"  Uniform  Macro-F1: {uniform_macro:.5f}")
    print(f"  Gewichtet Macro-F1: {weighted_macro:.5f}  ({delta:+.5f})")
    print(f"\n  Klassen-Vergleich:")
    print(f"  {'Klasse':<15} {'Uniform':>8} {'Gewichtet':>10} {'Delta':>8}")
    print(f"  {'─'*45}")
    for cls, u, w in zip(CLASSES, uniform_pcls, weighted_pcls):
        print(f"  {cls:<15} {u:>8.5f} {w:>10.5f} {w-u:>+8.5f}")

    # ── Gewichtungs-Matrix ausgeben ────────────────────────────────────────────
    print(f"\n  Gewichtungs-Matrix (per-class F1 pro Modell):")
    print(f"  {'Modell':<12}" + "".join(f"{cls:>12}" for cls in CLASSES))
    print(f"  {'─'*55}")
    for name, pcls in zip(model_names_ok, all_per_class_f1):
        print(f"  {name:<12}" + "".join(f"{f:>12.3f}" for f in pcls))

    # ── Testset-Vorhersagen ────────────────────────────────────────────────────
    if len(df_test) > 0:
        test_probs_all = []
        for shortname in model_names_ok:
            model_id     = MODEL_REGISTRY[shortname]
            weights_path = run_dir / f"model_{shortname}" / "best_model_weights.pt"
            tokenizer    = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            model        = TransformerClassifier(model_id, N_CLASSES, dropout).to(DEVICE)
            state        = torch.load(weights_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict({k: v.float() for k, v in state.items()})
            test_probs, _, _ = run_inference(
                model, df_test["text"].tolist(), tokenizer,
                batch_size, max_length, DEVICE
            )
            test_probs_all.append(test_probs)
            del model
            torch.cuda.empty_cache()

        test_weighted_probs = weighted_soft_voting(
            test_probs_all, all_per_class_f1, N_CLASSES
        )
        test_preds_labels = le.inverse_transform(test_weighted_probs.argmax(axis=1))
        pred_df = pd.DataFrame({"text": df_test["text"].values,
                                 "prediction": test_preds_labels})
        if "label" in df_test.columns:
            pred_df["true_label"] = df_test["label"].values
        pred_df.to_csv(run_dir / "predictions_test_weighted.csv", index=False)
        print(f"\n  Testvorhersagen gespeichert: predictions_test_weighted.csv")

    # ── Report schreiben ──────────────────────────────────────────────────────
    sep = "─" * 65
    weight_table = (
        f"  {'Modell':<12}" + "".join(f"{cls:>12}" for cls in CLASSES) + "\n" +
        f"  {'─'*55}\n" +
        "\n".join(
            f"  {name:<12}" + "".join(f"{f:>12.3f}" for f in pcls)
            for name, pcls in zip(model_names_ok, all_per_class_f1)
        )
    )

    report_text = f"""GermEval 2026 - Subtask 2: Weighted Ensemble Report
{sep}
Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M')}
Augmentierung:  {augmentation}
Modelle:        {', '.join(model_names_ok)}

Gewichtungs-Matrix (Val-F1 pro Klasse pro Modell):
{weight_table}

Ergebnisse (Val-Set):
  {'Klasse':<15} {'Uniform':>8} {'Gewichtet':>10} {'Delta':>8}
  {'─'*45}
{chr(10).join(f"  {cls:<15} {u:>8.5f} {w:>10.5f} {w-u:>+8.5f}"
              for cls, u, w in zip(CLASSES, uniform_pcls, weighted_pcls))}
  {'─'*45}
  {'Macro-F1':<15} {uniform_macro:>8.5f} {weighted_macro:>10.5f} {delta:>+8.5f}

Uniform Ensemble Classification Report:
{uniform_report}
Gewichtetes Ensemble Classification Report:
{weighted_report}
{sep}
"""
    out_path = run_dir / "weighted_ensemble_report.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"  Report gespeichert: {out_path}")

    summary_rows.append({
        "run":            run_name,
        "augmentation":   augmentation,
        "models":         "+".join(model_names_ok),
        "uniform_macro":  round(uniform_macro, 5),
        "weighted_macro": round(weighted_macro, 5),
        "delta":          round(delta, 5),
        **{f"uniform_{cls}":  round(float(u), 5) for cls, u in zip(CLASSES, uniform_pcls)},
        **{f"weighted_{cls}": round(float(w), 5) for cls, w in zip(CLASSES, weighted_pcls)},
    })

# ──────────────────────────────────────────────────────────────────────────────
# Abschluss
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  ZUSAMMENFASSUNG")
print(f"{'='*65}")
print(f"  {'Run':<40} {'Uniform':>8} {'Gewichtet':>10} {'Delta':>8}")
print(f"  {'─'*60}")
for row in summary_rows:
    print(f"  {row['run']:<40} {row['uniform_macro']:>8.5f} "
          f"{row['weighted_macro']:>10.5f} {row['delta']:>+8.5f}")

if summary_rows:
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = Path(args.base_dir) / "weighted_ensemble_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Zusammenfassung: {summary_path}")
