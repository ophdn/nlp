"""
GermEval 2026 – Subtask 2: DBO Classification
==============================================
Ensemble Training Script: GBERT-large + XLM-RoBERTa-large + DeBERTa-v3-base
                          + optionaler 4. Checkpoint (z.B. Marios fine-getuntes Modell)
Soft-Voting über Klassen-Wahrscheinlichkeiten (nach Köpcke et al., 2025)

VERWENDUNG:
    # Standard 3-Modell Ensemble:
    python germeval2026_ensemble_train.py --augmentation none --out_dir output/ensemble-none

    # Mit Marios Checkpoint als 4. Modell:
    python germeval2026_ensemble_train.py \
        --augmentation both \
        --fourth_model_path /path/to/marios/best_model_weights.pt \
        --fourth_model_base FacebookAI/xlm-roberta-large \
        --out_dir output/ensemble-4model-both

DATEI-LAYOUT (erwartet):
    preprocessed/
        train_minimal.csv          <- Haupttrainingsdaten (text, label)
        test_minimal.csv           <- Testdaten fuer finale Submission
    data/
        aug-llm-paraphrasing.csv   <- LLM-Paraphrasen (text, label)  [optional]
        aug-grok-generated.csv     <- Grok-Generierungen (text, label) [optional]

AUSGABE:
    <out_dir>/
        model_gbert/               <- GBERT-large Gewichte (float16)
        model_xlmr/                <- XLM-RoBERTa-large Gewichte (float16)
        model_deberta/             <- DeBERTa-v3-base Gewichte (float16)
        ensemble_log.csv           <- Val-Metriken pro Evaluierung
        ensemble_config.json       <- alle Parameter dieses Runs
        predictions_test.csv       <- Submission-Datei
        final_report.txt           <- Zusammenfassung
    results.csv                    <- EINE Zeile pro Run, global angehaengt

REFERENZEN:
    Kopcke (2025): nymera@GermEval 2025 - Ensemble GBERT/XLMRoBERTa/DeBERTa, Soft-Voting
    Thelen et al. (2025): Candy Speech - XLM-RoBERTa-Large, MLP-Head, LR 2e-5
"""

import argparse
import json
import os
import random
import time
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ------------------------------------------------------------------------------
# 1. CLI
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GermEval 2026 Ensemble Training")

# Datenpfade
parser.add_argument("--train_file",      default="preprocessed/train_minimal.csv")
parser.add_argument("--test_file",       default="preprocessed/test_minimal.csv")
parser.add_argument("--paraphrase_file", default="data/aug-llm-paraphrasing.csv")
parser.add_argument("--generated_file",  default="preprocessed/synthetic_data.csv")

# Augmentierungs-Variante
parser.add_argument("--augmentation", default="none",
                    choices=["none", "paraphrase", "generated", "both"])

# Modell-Auswahl
parser.add_argument("--models", nargs="+",
                    default=["gbert", "xlmr", "deberta"],
                    choices=["gbert", "xlmr", "deberta"],
                    help="Modelle fuer das Ensemble (default: alle drei)")

# 4. Modell (optional) - externer Checkpoint, wird NICHT neu trainiert
parser.add_argument("--fourth_model_path", default=None,
                    help="Pfad zu einem externen .pt Checkpoint (z.B. Marios Modell). "
                         "Wird nicht trainiert, nur fuer Ensemble-Inferenz genutzt.")
parser.add_argument("--fourth_model_base", default="FacebookAI/xlm-roberta-large",
                    help="Basis-Modell des externen Checkpoints (fuer Tokenizer + Architektur)")
parser.add_argument("--fourth_model_name", default="mario",
                    help="Kurzname fuer den 4. Checkpoint in Logs und results.csv")

# Ausgabe
parser.add_argument("--out_dir",     default="models/ensemble")
parser.add_argument("--results_csv", default="results.csv")

# Training
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--max_length",    type=int,   default=128)
parser.add_argument("--batch_size",    type=int,   default=16)
parser.add_argument("--lr",            type=float, default=2e-5)
parser.add_argument("--weight_decay",  type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--warmup_steps",  type=int,   default=500)
parser.add_argument("--max_epochs",    type=int,   default=10)
parser.add_argument("--eval_every",    type=int,   default=40)
parser.add_argument("--patience",      type=int,   default=50)
parser.add_argument("--val_size",      type=float, default=0.1)
parser.add_argument("--dropout",       type=float, default=0.1)

args, _ = parser.parse_known_args()

# ------------------------------------------------------------------------------
# 2. Modell-Konfigurationen (die 3 trainierten Modelle)
# ------------------------------------------------------------------------------
_ALL_MODEL_CONFIGS = {
    "gbert": {
        "model_id":  "deepset/gbert-large",
        "shortname": "gbert",
    },
    "xlmr": {
        "model_id":  "FacebookAI/xlm-roberta-large",
        "shortname": "xlmr",
    },
    "deberta": {
        "model_id":  "microsoft/deberta-v3-base",
        "shortname": "deberta",
    },
}
MODEL_CONFIGS = {k: v for k, v in _ALL_MODEL_CONFIGS.items() if k in args.models}

# ------------------------------------------------------------------------------
# 3. Setup
# ------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT    = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

USE_FOURTH = args.fourth_model_path is not None and Path(args.fourth_model_path).exists()

print(f"\n{'='*65}")
print(f"  GermEval 2026 - Subtask 2  |  Ensemble Training")
print(f"{'='*65}")
print(f"  Device:        {DEVICE}")
print(f"  Augmentierung: {args.augmentation}")
print(f"  4. Modell:     {args.fourth_model_path if USE_FOURTH else 'keins'}")
print(f"  Ausgabe:       {OUT.resolve()}\n")

if args.fourth_model_path and not Path(args.fourth_model_path).exists():
    print(f"  WARNING: --fourth_model_path nicht gefunden: {args.fourth_model_path}")
    print(f"           Fahre ohne 4. Modell fort.")

# ------------------------------------------------------------------------------
# 4. Daten laden
# ------------------------------------------------------------------------------
def load_csv(path: str, desc: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"  WARNING  Datei nicht gefunden (wird uebersprungen): {p}")
        return pd.DataFrame(columns=["text", "label"])

    # Strategie 1: Standard pandas mit verschiedenen Quoting-Optionen
    for kwargs in [
        {},
        {"quoting": 3},
        {"engine": "python"},
        {"engine": "python", "on_bad_lines": "skip"},
    ]:
        try:
            df = pd.read_csv(p, **kwargs)
            if "text" in df.columns and "label" in df.columns:
                df = df[["text", "label"]].dropna()
                print(f"  OK {desc:<28} {len(df):>6,} Zeilen  ({p.name})")
                return df
        except Exception:
            continue

    # Strategie 2: Semikolon- oder Tab-Separator, ggf. Spalten vertauscht
    for sep in [";", "\t"]:
        try:
            df = pd.read_csv(p, sep=sep, engine="python")
            cols = list(df.columns)
            if "text" in cols and "label" in cols:
                df = df[["text", "label"]].dropna()
                print(f"  OK {desc:<28} {len(df):>6,} Zeilen  ({p.name})  [sep='{sep}']")
                return df
            if len(cols) == 2:
                valid_labels = {"nothing", "criticism", "agitation", "subversive"}
                for label_col, text_col in [(cols[0], cols[1]), (cols[1], cols[0])]:
                    sample = df[label_col].dropna().astype(str).str.lower()
                    if sample.isin(valid_labels).mean() > 0.8:
                        df = df.rename(columns={label_col: "label", text_col: "text"})
                        df = df[["text", "label"]].dropna()
                        print(f"  OK {desc:<28} {len(df):>6,} Zeilen  ({p.name})  [sep='{sep}', getauscht]")
                        return df
        except Exception:
            continue

    # Strategie 3: Manuelles Parsen
    rows = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0 or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                rows.append({
                    "text":  ",".join(parts[:-1]).strip().strip('"'),
                    "label": parts[-1].strip().strip('"'),
                })
    df = pd.DataFrame(rows).dropna()
    print(f"  OK {desc:<28} {len(df):>6,} Zeilen  ({p.name})  [manuell]")
    return df


print("  Lade Daten ...")
df_main = load_csv(args.train_file, "Hauptdaten (train_minimal)")
df_test  = load_csv(args.test_file,  "Testdaten (test_minimal)")

df_para = pd.DataFrame(columns=["text", "label"])
df_gen  = pd.DataFrame(columns=["text", "label"])

if args.augmentation in ("paraphrase", "both"):
    df_para = load_csv(args.paraphrase_file, "LLM-Paraphrasen")
if args.augmentation in ("generated", "both"):
    df_gen  = load_csv(args.generated_file,  "Grok-Generierungen")

aug_files_used = []
if len(df_para) > 0: aug_files_used.append(args.paraphrase_file)
if len(df_gen)  > 0: aug_files_used.append(args.generated_file)

print(f"\n  Trainingsdaten nach Augmentierung:")
df_all = pd.concat([df_main] + ([df_para] if len(df_para)>0 else []) +
                   ([df_gen]  if len(df_gen)>0  else []), ignore_index=True)
for lbl, cnt in df_all["label"].value_counts().items():
    bar = "█" * (cnt // 100)
    pct = cnt / len(df_all) * 100
    print(f"    {lbl:<15} {cnt:>6,}  ({pct:5.1f}%)  {bar}")

# Val-Split aus Hauptdaten, augmentierte Daten NUR ins Training
df_train_base, df_val = train_test_split(
    df_main, test_size=args.val_size, stratify=df_main["label"], random_state=args.seed
)
aug_extra = pd.concat(
    [p for p in [df_para, df_gen] if len(p) > 0], ignore_index=True
) if aug_files_used else pd.DataFrame(columns=["text", "label"])

df_train = pd.concat([df_train_base, aug_extra], ignore_index=True).sample(
    frac=1, random_state=args.seed
).reset_index(drop=True)

print(f"\n  Train: {len(df_train):,}  |  Val: {len(df_val):,}  (Val-Set enthaelt keine augmentierten Daten)")

# Label-Encoder
le = LabelEncoder()
le.fit(df_main["label"])
CLASSES   = list(le.classes_)
N_CLASSES = len(CLASSES)
print(f"\n  Klassen ({N_CLASSES}): {CLASSES}")

# Klassen-Gewichte aus Basis-Trainingsdaten
base_counts = pd.Series(le.transform(df_train_base["label"])).value_counts().sort_index()
counts  = np.array([base_counts.get(i, 1) for i in range(N_CLASSES)], dtype=float)
weights = len(df_train_base) / (N_CLASSES * counts)
weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)

print(f"\n  Class Weights:")
for cls, w in zip(CLASSES, weights):
    print(f"    {cls:<15} {w:.2f}")

# ------------------------------------------------------------------------------
# 5. Dataset
# ------------------------------------------------------------------------------
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


def make_loaders(df_tr, df_vl, tokenizer, batch_size):
    y_train  = le.transform(df_tr["label"])
    y_val    = le.transform(df_vl["label"])
    train_ds = DBODataset(df_tr["text"].tolist(), y_train, tokenizer, args.max_length)
    val_ds   = DBODataset(df_vl["text"].tolist(), y_val,   tokenizer, args.max_length)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )

# ------------------------------------------------------------------------------
# 6. Modell-Architektur
# ------------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """MLP-Head auf [CLS]-Token (nach Thelen et al., 2025)."""
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

# ------------------------------------------------------------------------------
# 7. Training eines einzelnen Modells
# ------------------------------------------------------------------------------
def train_single_model(cfg):
    model_id  = cfg["model_id"]
    shortname = cfg["shortname"]
    model_dir = OUT / f"model_{shortname}"
    model_dir.mkdir(exist_ok=True)
    best_weights_path = model_dir / "best_model_weights.pt"

    print(f"\n{'─'*65}")
    print(f"  Trainiere: {model_id}")
    print(f"{'─'*65}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    train_loader, val_loader = make_loaders(df_train, df_val, tokenizer, args.batch_size)

    model       = TransformerClassifier(model_id, N_CLASSES, args.dropout).to(DEVICE)
    total_steps = len(train_loader) * args.max_epochs
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler   = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    criterion   = nn.CrossEntropyLoss(weight=weights_tensor)

    def evaluate():
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
                probs  = torch.softmax(logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(batch["label"].numpy())
        model.train()
        macro_f1     = f1_score(all_labels, all_preds, average="macro",   zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None,
                                zero_division=0, labels=list(range(N_CLASSES)))
        report       = classification_report(all_labels, all_preds,
                                             target_names=CLASSES, zero_division=0)
        return macro_f1, per_class_f1, report, np.array(all_probs), np.array(all_labels)

    global_step = 0
    best_f1     = 0.0
    no_improve  = 0
    log_rows    = []
    early_stop  = False
    start_time  = time.time()

    print(f"  Training: max {args.max_epochs} Epochs | Eval alle {args.eval_every} Steps | Patience {args.patience}")

    model.train()
    for epoch in range(1, args.max_epochs + 1):
        for batch in train_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.eval_every == 0:
                macro_f1, per_class_f1, report, _, _ = evaluate()
                if macro_f1 > best_f1:
                    best_f1    = macro_f1
                    no_improve = 0
                    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()},
                               best_weights_path)
                    flag = "OK Bestes Modell"
                else:
                    no_improve += 1
                    flag = f"kein Fortschritt {no_improve}/{args.patience}"

                print(f"  [{shortname}] Step {global_step:>5} | Ep {epoch} | Macro-F1 {macro_f1:.4f} | {flag}")
                log_rows.append({
                    "model": shortname, "step": global_step, "epoch": epoch,
                    "val_macro_f1": round(macro_f1, 5),
                    **{f"f1_{cls}": round(float(f), 5) for cls, f in zip(CLASSES, per_class_f1)},
                })
                if no_improve >= args.patience:
                    print(f"  [{shortname}] Early Stopping (beste Macro-F1: {best_f1:.4f})")
                    early_stop = True
                    break
        if early_stop:
            break

    elapsed = time.time() - start_time
    best_state = torch.load(best_weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in best_state.items()})
    final_f1, final_per_class, final_report, val_probs, val_labels = evaluate()

    print(f"\n  [{shortname}] Finaler Report:")
    print(final_report)
    pd.DataFrame(log_rows).to_csv(model_dir / "training_log.csv", index=False)

    return {
        "shortname":           shortname,
        "model_id":            model_id,
        "best_val_macro_f1":   round(final_f1, 5),
        "per_class_f1":        {cls: round(float(f), 5) for cls, f in zip(CLASSES, final_per_class)},
        "weights_path":        str(best_weights_path),
        "val_probs":           val_probs,
        "val_labels":          val_labels,
        "total_steps":         global_step,
        "early_stopped":       early_stop,
        "training_minutes":    round(elapsed / 60, 1),
        "final_report":        final_report,
        "is_external":         False,
    }

# ------------------------------------------------------------------------------
# 8. Externer Checkpoint laden und auf Val-Set evaluieren (kein Training)
# ------------------------------------------------------------------------------
def load_external_checkpoint(ckpt_path, base_model_id, shortname):
    """
    Laedt einen externen Checkpoint (z.B. Marios Modell) und evaluiert ihn
    auf demselben Val-Set wie die trainierten Modelle.
    Gibt dasselbe Dict-Format zurueck wie train_single_model().
    """
    print(f"\n{'─'*65}")
    print(f"  Lade externen Checkpoint: {shortname}")
    print(f"  Basis-Modell: {base_model_id}")
    print(f"  Pfad:         {ckpt_path}")
    print(f"{'─'*65}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
    _, val_loader = make_loaders(df_train, df_val, tokenizer, args.batch_size)

    model = TransformerClassifier(base_model_id, N_CLASSES, args.dropout).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # Unerwartete Keys ignorieren (z.B. lm_head.* aus Pretraining)
    model_keys   = set(model.state_dict().keys())
    filtered     = {k: v for k, v in state.items() if k in model_keys}
    missing      = model_keys - set(filtered.keys())
    if missing:
        print(f"  INFO: {len(missing)} Keys fehlen im Checkpoint (werden ignoriert)")
    model.load_state_dict(filtered, strict=False)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            probs  = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    macro_f1     = f1_score(all_labels, all_preds, average="macro",   zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None,
                            zero_division=0, labels=list(range(N_CLASSES)))
    report       = classification_report(all_labels, all_preds,
                                         target_names=CLASSES, zero_division=0)

    print(f"\n  [{shortname}] Finaler Report (kein Training - nur Inferenz):")
    print(report)

    # Val-Probs fuer Soft-Voting speichern
    val_probs_array = np.array(all_probs)
    torch.save(val_probs_array, OUT / f"model_{shortname}_val_probs.pt")

    return {
        "shortname":         shortname,
        "model_id":          base_model_id,
        "checkpoint_path":   str(ckpt_path),
        "best_val_macro_f1": round(macro_f1, 5),
        "per_class_f1":      {cls: round(float(f), 5) for cls, f in zip(CLASSES, per_class_f1)},
        "weights_path":      str(ckpt_path),
        "val_probs":         val_probs_array,
        "val_labels":        np.array(all_labels),
        "total_steps":       0,
        "early_stopped":     False,
        "training_minutes":  0.0,
        "final_report":      report,
        "is_external":       True,
    }

# ------------------------------------------------------------------------------
# 9. Alle Modelle trainieren / laden
# ------------------------------------------------------------------------------
print(f"\n{'='*65}")
n_models = len(MODEL_CONFIGS) + (1 if USE_FOURTH else 0)
print(f"  STARTE ENSEMBLE-TRAINING ({n_models} Modelle)")
print(f"{'='*65}")

model_results = {}
for cfg in MODEL_CONFIGS.values():
    result = train_single_model(cfg)
    model_results[result["shortname"]] = result

# 4. Modell: nur laden + evaluieren, nicht trainieren
if USE_FOURTH:
    fourth_result = load_external_checkpoint(
        ckpt_path      = args.fourth_model_path,
        base_model_id  = args.fourth_model_base,
        shortname      = args.fourth_model_name,
    )
    model_results[args.fourth_model_name] = fourth_result
    print(f"\n  4. Modell ({args.fourth_model_name}) wurde NICHT neu trainiert.")
    print(f"  Macro-F1 auf Val-Set: {fourth_result['best_val_macro_f1']:.5f}")

# ------------------------------------------------------------------------------
# 10. Ensemble Soft-Voting auf Val-Set
# ------------------------------------------------------------------------------
print(f"\n{'─'*65}")
print(f"  ENSEMBLE SOFT-VOTING (Val-Set)")
print(f"{'─'*65}")

val_labels_arr = list(model_results.values())[0]["val_labels"]
ensemble_probs = np.mean([r["val_probs"] for r in model_results.values()], axis=0)
ensemble_preds = ensemble_probs.argmax(axis=1)

ensemble_macro_f1  = f1_score(val_labels_arr, ensemble_preds, average="macro",   zero_division=0)
ensemble_per_class = f1_score(val_labels_arr, ensemble_preds, average=None,
                               zero_division=0, labels=list(range(N_CLASSES)))
ensemble_report    = classification_report(val_labels_arr, ensemble_preds,
                                           target_names=CLASSES, zero_division=0)

print(f"\n  Ensemble Macro-F1 (Val): {ensemble_macro_f1:.5f}")
print(f"\n  Einzelmodelle zum Vergleich:")
for short, res in model_results.items():
    ext_tag = " [extern, kein Training]" if res.get("is_external") else ""
    print(f"    {short:<12} Macro-F1: {res['best_val_macro_f1']:.5f}{ext_tag}")
print(f"\n  Ensemble Classification Report:")
print(ensemble_report)

# ------------------------------------------------------------------------------
# 11. Testset-Vorhersagen
# ------------------------------------------------------------------------------
print(f"\n{'─'*65}")
print(f"  TESTSET-VORHERSAGEN")
print(f"{'─'*65}")

if len(df_test) > 0:
    test_probs_all = []

    # Trainierte Modelle
    for cfg in MODEL_CONFIGS.values():
        shortname = cfg["shortname"]
        model_id  = cfg["model_id"]
        weights_p = OUT / f"model_{shortname}" / "best_model_weights.pt"

        print(f"  Lade {shortname} fuer Testvorhersage ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model_inf = TransformerClassifier(model_id, N_CLASSES, args.dropout).to(DEVICE)
        state     = torch.load(weights_p, map_location=DEVICE, weights_only=False)
        model_inf.load_state_dict({k: v.float() for k, v in state.items()})
        model_inf.eval()

        test_probs_model = []
        texts = df_test["text"].tolist()
        for i in range(0, len(texts), args.batch_size):
            enc = tokenizer(
                texts[i:i+args.batch_size], truncation=True, padding="max_length",
                max_length=args.max_length, return_tensors="pt",
            )
            with torch.no_grad():
                logits = model_inf(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            test_probs_model.append(probs)

        test_probs_all.append(np.vstack(test_probs_model))
        del model_inf
        torch.cuda.empty_cache()

    # 4. Modell (externer Checkpoint)
    if USE_FOURTH:
        print(f"  Lade {args.fourth_model_name} (externer Checkpoint) fuer Testvorhersage ...")
        tokenizer = AutoTokenizer.from_pretrained(args.fourth_model_base, use_fast=False)
        model_inf = TransformerClassifier(args.fourth_model_base, N_CLASSES, args.dropout).to(DEVICE)
        state     = torch.load(args.fourth_model_path, map_location=DEVICE, weights_only=False)
        model_keys = set(model_inf.state_dict().keys())
        filtered   = {k: v for k, v in state.items() if k in model_keys}
        model_inf.load_state_dict(filtered, strict=False)
        model_inf.eval()

        test_probs_model = []
        texts = df_test["text"].tolist()
        for i in range(0, len(texts), args.batch_size):
            enc = tokenizer(
                texts[i:i+args.batch_size], truncation=True, padding="max_length",
                max_length=args.max_length, return_tensors="pt",
            )
            with torch.no_grad():
                logits = model_inf(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            test_probs_model.append(probs)

        test_probs_all.append(np.vstack(test_probs_model))
        del model_inf
        torch.cuda.empty_cache()

    test_ensemble_probs = np.mean(test_probs_all, axis=0)
    test_preds_idx      = test_ensemble_probs.argmax(axis=1)
    test_preds_labels   = le.inverse_transform(test_preds_idx)

    pred_df = pd.DataFrame({"text": df_test["text"].values, "prediction": test_preds_labels})
    if "label" in df_test.columns:
        pred_df["true_label"] = df_test["label"].values

    pred_path = OUT / "predictions_test.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Testvorhersagen gespeichert: {pred_path}")
    for lbl, cnt in pd.Series(test_preds_labels).value_counts().items():
        print(f"    {lbl:<15} {cnt:>6,}")
else:
    print("  Keine Testdaten gefunden.")

# ------------------------------------------------------------------------------
# 12. Logs + Config speichern
# ------------------------------------------------------------------------------
ensemble_log_rows = []
for short, res in model_results.items():
    if not res.get("is_external"):
        log_path = OUT / f"model_{short}" / "training_log.csv"
        if log_path.exists():
            for row in pd.read_csv(log_path).to_dict("records"):
                ensemble_log_rows.append(row)
pd.DataFrame(ensemble_log_rows).to_csv(OUT / "ensemble_log.csv", index=False)

config = {
    **vars(args),
    "models":                [cfg["model_id"] for cfg in MODEL_CONFIGS.values()],
    "fourth_model_used":     USE_FOURTH,
    "classes":               CLASSES,
    "class_weights":         {cls: round(float(w), 4) for cls, w in zip(CLASSES, weights)},
    "n_train":               len(df_train),
    "n_val":                 len(df_val),
    "aug_files_used":        aug_files_used,
    "ensemble_val_macro_f1": round(ensemble_macro_f1, 5),
    "ensemble_per_class_f1": {cls: round(float(f), 5) for cls, f in zip(CLASSES, ensemble_per_class)},
    "individual_results":    {
        s: {"val_macro_f1": r["best_val_macro_f1"],
            "is_external":  r.get("is_external", False),
            **r["per_class_f1"]}
        for s, r in model_results.items()
    },
    "device":    str(DEVICE),
    "timestamp": datetime.now().isoformat(),
}
with open(OUT / "ensemble_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------------------
# 13. Globale results.csv anhaengen
# ------------------------------------------------------------------------------
results_row = {
    "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M"),
    "script":              "germeval2026_ensemble_train.py",
    "model_type":          f"ensemble_{n_models}model",
    "models":              "+".join(list(cfg["model_id"] for cfg in MODEL_CONFIGS.values()) +
                                   ([args.fourth_model_base] if USE_FOURTH else [])),
    "fourth_model_path":   args.fourth_model_path or "none",
    "fourth_model_name":   args.fourth_model_name if USE_FOURTH else "none",
    "augmentation":        args.augmentation,
    "aug_files":           "|".join(aug_files_used) if aug_files_used else "none",
    "n_train_total":       len(df_train),
    "n_train_base":        len(df_train_base),
    "n_aug_paraphrase":    len(df_para),
    "n_aug_generated":     len(df_gen),
    "n_val":               len(df_val),
    "ensemble_macro_f1":   round(ensemble_macro_f1, 5),
    **{f"ensemble_f1_{cls}": round(float(f), 5) for cls, f in zip(CLASSES, ensemble_per_class)},
    **{f"{short}_macro_f1": res["best_val_macro_f1"] for short, res in model_results.items()},
    **{f"{short}_f1_{cls}": res["per_class_f1"].get(cls, 0.0)
       for short, res in model_results.items() for cls in CLASSES},
    "lr":            args.lr,
    "batch_size":    args.batch_size,
    "max_length":    args.max_length,
    "warmup_steps":  args.warmup_steps,
    "weight_decay":  args.weight_decay,
    "max_grad_norm": args.max_grad_norm,
    "max_epochs":    args.max_epochs,
    "eval_every":    args.eval_every,
    "patience":      args.patience,
    "val_size":      args.val_size,
    "dropout":       args.dropout,
    "seed":          args.seed,
    **{f"{short}_steps":         res["total_steps"]      for short, res in model_results.items()},
    **{f"{short}_early_stopped": res["early_stopped"]    for short, res in model_results.items()},
    **{f"{short}_minutes":       res["training_minutes"] for short, res in model_results.items()},
    "out_dir":       str(OUT),
}

results_path = Path(args.results_csv)
new_row_df   = pd.DataFrame([results_row])
if results_path.exists():
    new_row_df = pd.concat([pd.read_csv(results_path), new_row_df], ignore_index=True)
new_row_df.to_csv(results_path, index=False)
print(f"\n  Ergebnis angehaengt: {results_path.resolve()}")

# ------------------------------------------------------------------------------
# 14. Finaler Report
# ------------------------------------------------------------------------------
total_minutes = sum(r["training_minutes"] for r in model_results.values())
sep = "─" * 65

fourth_info = ""
if USE_FOURTH:
    r4 = model_results[args.fourth_model_name]
    fourth_info = f"""
4. Modell (extern, kein Training):
  Name:         {args.fourth_model_name}
  Basis:        {args.fourth_model_base}
  Pfad:         {args.fourth_model_path}
  Macro-F1:     {r4["best_val_macro_f1"]:.5f}
  agitation:    {r4["per_class_f1"].get("agitation", 0):.3f}
  subversive:   {r4["per_class_f1"].get("subversive", 0):.3f}"""

report_text = f"""GermEval 2026 - Subtask 2: Ensemble Training Report
{sep}
Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M')}
Augmentierung:   {args.augmentation}
Modelle:         {n_models} ({'3 trainiert' + (' + 1 extern' if USE_FOURTH else '')})
Train:           {len(df_train):,} (Basis: {len(df_train_base):,} + Aug: {len(df_train)-len(df_train_base):,})
Val:             {len(df_val):,}
Device:          {DEVICE}
{fourth_info}

Ergebnisse Einzelmodelle (Val-Set):
{chr(10).join(f"  {s:<12} Macro-F1: {r['best_val_macro_f1']:.5f}  "
              f"agitation: {r['per_class_f1'].get('agitation',0):.3f}  "
              f"subversive: {r['per_class_f1'].get('subversive',0):.3f}"
              + (" [extern]" if r.get("is_external") else "")
              for s, r in model_results.items())}

Ensemble Ergebnis (Val-Set):
  Macro-F1: {ensemble_macro_f1:.5f}
{chr(10).join(f"  {cls:<15} {f:.5f}" for cls, f in zip(CLASSES, ensemble_per_class))}

{ensemble_report}
Trainingszeit: {total_minutes:.1f} min
{sep}
"""
(OUT / "final_report.txt").write_text(report_text, encoding="utf-8")

print(f"\n{'='*65}")
print(f"  ENSEMBLE-TRAINING ABGESCHLOSSEN")
print(f"{'='*65}")
print(f"  Ensemble Macro-F1 (Val):  {ensemble_macro_f1:.5f}")
for short, res in model_results.items():
    ext = " [extern]" if res.get("is_external") else ""
    print(f"    {short:<12} {res['best_val_macro_f1']:.5f}{ext}")
print(f"\n  Alle Dateien in: {OUT.resolve()}")
print(f"  predictions_test.csv | ensemble_config.json | {args.results_csv}\n")