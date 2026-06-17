"""
GermEval 2026 – Transfer Learning von Quell-Checkpoints
=======================================================
Preprocesst die Rohdaten (identisch zu preprocessing.py) und fine-tuned
mehrere Modelle vom Quell-Checkpoint auf einen anderen Subtask (c2a, vio, def).

GPU-Zeitersparnis:
  - Encoder bereits auf deutschen politischen Tweets eingestellt
  - Epoch 1: Encoder eingefroren, nur Head trainiert
  - Danach: differenziertes LR (Encoder 5e-6, Head 2e-5)
  - Max 5 Epochs + frühere Patience statt 10 Epochs

VERWENDUNG:
    # Alle 3 Modelle für C2A:
    python germeval2026_transfer.py \\
        --task      c2a \\
        --train_file ../data/GermEval2026/data/c2a/c2a_train_26.csv \\
        --test_file  ../data/GermEval2026/data/c2a/c2a_test_26.csv \\
        --source_run   model_dataset_gridsearch/all5_aug-paraphrase \\
        --models    gbert mdeberta gelectra \\
        --out_dir   transfer_runs/c2a

    # Nur mdeberta für VIO:
    python germeval2026_transfer.py \\
        --task      vio \\
        --train_file ../data/GermEval2026/data/vio/vio_train_26.csv \\
        --source_run   model_dataset_gridsearch/all5_aug-paraphrase \\
        --models    mdeberta \\
        --out_dir   transfer_runs/vio
"""

import argparse
import json
import re
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

# ──────────────────────────────────────────────────────────────────────────────
# Task-Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "c2a": {"classes": ["FALSE", "TRUE"]},
    "vio": {"classes": ["call2violence", "glorification", "nothing", "other", "propensity", "support"]},
    "def": {"classes": ["FALSE", "TRUE"]},
    "dbo": {"classes": ["agitation", "criticism", "nothing", "subversive"]},
}

MODEL_REGISTRY = {
    "gbert":    "deepset/gbert-large",
    "xlmr":     "FacebookAI/xlm-roberta-large",
    "deberta":  "microsoft/deberta-v3-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "gelectra": "deepset/gelectra-large-germanquad",
}

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing (identisch zu preprocessing.py)
# ──────────────────────────────────────────────────────────────────────────────
_URL_RE     = re.compile(r'https?://\S+|www\.\S+')
_MENTION_RE = re.compile(r'@\w+')
_WHITESPACE = re.compile(r'[ \t]+')
_NEWLINE    = re.compile(r'\n+')

def preprocess_minimal(text: str) -> str:
    text = _URL_RE.sub("[URL]", text)
    text = _MENTION_RE.sub("[USER]", text)
    text = _NEWLINE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()

def load_and_preprocess(path: Path, has_label: bool = True) -> pd.DataFrame:
    """Lädt Raw-CSV und preprocesst identisch zu preprocessing.py (minimal-Variante)."""
    df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="warn")

    text_candidates = [c for c in df.columns
                       if c.lower() in ("text", "tweet", "comment", "sentence", "description")]
    text_col = text_candidates[0] if text_candidates else \
        max(df.select_dtypes("object").columns,
            key=lambda c: df[c].astype(str).str.len().mean())

    if has_label:
        label_candidates = [c for c in df.columns
                            if c.lower() in ("label", "class", "dbo", "c2a", "def", "vio",
                                             "subtask2", "subtask3", "category")]
        label_col = label_candidates[0] if label_candidates else \
            [c for c in df.columns if c != text_col][0]
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        df["label"] = df["label"].astype(str).str.strip()
    else:
        id_candidates = [c for c in df.columns if c.lower() == "id"]
        id_col = id_candidates[0] if id_candidates else df.columns[0]
        df = df[[id_col, text_col]].rename(columns={id_col: "id", text_col: "text"})

    df["text"] = df["text"].astype(str).str.strip().apply(preprocess_minimal)
    df = df.dropna(subset=["text"])
    print(f"    {len(df):,} Zeilen geladen und preprocesst ({path.name})")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 – Transfer Learning")
parser.add_argument("--task",         required=True, choices=list(TASK_CONFIG),
                    help="Ziel-Subtask: c2a | vio | def")
parser.add_argument("--train_file",   required=True,
                    help="Rohe Trainings-CSV (z.B. c2a_train_26.csv)")
parser.add_argument("--test_file",    default=None,
                    help="Rohe Test-CSV ohne Labels (optional, für direkte Prediction)")
parser.add_argument("--source_run",   required=True,
                    help="Ordner mit den Quell-Checkpoints (z.B. model_dataset_gridsearch/all5_aug-paraphrase)")
parser.add_argument("--models",       nargs="+", default=["gbert", "mdeberta", "gelectra"],
                    choices=list(MODEL_REGISTRY),
                    help="Welche Modelle trainiert werden sollen")
parser.add_argument("--out_dir",      required=True,
                    help="Ausgabeverzeichnis (z.B. transfer_runs/c2a)")
parser.add_argument("--results_csv",  default="results_transfer.csv")
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--max_length",   type=int,   default=128)
parser.add_argument("--batch_size",   type=int,   default=16)
parser.add_argument("--lr_encoder",   type=float, default=5e-6)
parser.add_argument("--lr_head",      type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_grad_norm",type=float, default=1.0)
parser.add_argument("--warmup_steps", type=int,   default=100)
parser.add_argument("--max_epochs",   type=int,   default=5)
parser.add_argument("--eval_every",   type=int,   default=40)
parser.add_argument("--patience",     type=int,   default=30)
parser.add_argument("--val_size",     type=float, default=0.1)
parser.add_argument("--freeze_epochs",type=int,   default=1,
                    help="Encoder in den ersten N Epochs einfrieren. 0 = kein Freeze.")
parser.add_argument("--dropout",      type=float, default=0.1)
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT        = Path(args.out_dir)
SOURCE_RUN = Path(args.source_run)
OUT.mkdir(parents=True, exist_ok=True)

CLASSES   = TASK_CONFIG[args.task]["classes"]
N_CLASSES = len(CLASSES)

print(f"\n{'='*65}")
print(f"  GermEval 2026 – Transfer Learning")
print(f"{'='*65}")
print(f"  Task:         {args.task.upper()}  ({N_CLASSES} Klassen: {CLASSES})")
print(f"  Modelle:      {args.models}")
print(f"  DBO-Run:      {SOURCE_RUN}")
print(f"  Device:       {DEVICE}")
print(f"  LR Encoder:   {args.lr_encoder}  |  LR Head: {args.lr_head}")
print(f"  Freeze:       Encoder für erste {args.freeze_epochs} Epoch(s)")
print(f"  Max Epochs:   {args.max_epochs}  |  Patience: {args.patience}")
print(f"  Ausgabe:      {OUT.resolve()}")
print(f"{'='*65}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Daten laden & preprocessen
# ──────────────────────────────────────────────────────────────────────────────
print("  Preprocessing ...")
df_all = load_and_preprocess(Path(args.train_file), has_label=True)

df_test_raw = None
if args.test_file:
    df_test_raw = load_and_preprocess(Path(args.test_file), has_label=False)
    print(f"    {len(df_test_raw):,} Test-Tweets geladen.")

df_train, df_val = train_test_split(
    df_all, test_size=args.val_size, stratify=df_all["label"], random_state=args.seed
)
print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}\n")

le = LabelEncoder()
le.fit(CLASSES)

# Class Weights
train_counts   = pd.Series(le.transform(df_train["label"])).value_counts().sort_index()
counts         = np.array([train_counts.get(i, 1) for i in range(N_CLASSES)], dtype=float)
class_weights  = len(df_train) / (N_CLASSES * counts)
print(f"  Class Weights:")
for cls, w in zip(CLASSES, class_weights):
    print(f"    {cls:<22} {w:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, label_encoder=None):
        self.has_labels = "label" in df.columns
        self.encodings  = tokenizer(
            df["text"].tolist(),
            truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        if self.has_labels:
            self.labels = torch.tensor(label_encoder.transform(df["label"]), dtype=torch.long)
        if "id" in df.columns:
            self.ids = df["id"].astype(str).tolist()

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item

# ──────────────────────────────────────────────────────────────────────────────
# Modell-Architektur (identisch mit Training)
# ──────────────────────────────────────────────────────────────────────────────
class TransferClassifier(nn.Module):
    def __init__(self, model_name, n_classes, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
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

# ──────────────────────────────────────────────────────────────────────────────
# Training-Funktion (pro Modell)
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model_key: str) -> dict:
    model_id  = MODEL_REGISTRY[model_key]
    ckpt_path = SOURCE_RUN / f"model_{model_key}" / "best_model_weights.pt"
    model_out = OUT / f"model_{model_key}"
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*65}")
    print(f"  MODELL: {model_key.upper()}  ({model_id})")
    print(f"  Quell-Checkpoint: {ckpt_path}")
    print(f"{'─'*65}")

    if not ckpt_path.exists():
        print(f"  FEHLER: Checkpoint nicht gefunden. Überspringe {model_key}.")
        return {}

    # Tokenizer & Datasets
    print(f"  Lade Tokenizer ...")
    tokenizer     = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    train_dataset = TextDataset(df_train, tokenizer, args.max_length, le)
    val_dataset   = TextDataset(df_val,   tokenizer, args.max_length, le)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Modell initialisieren
    print(f"  Lade Modell-Architektur ...")
    model = TransferClassifier(model_id, N_CLASSES, args.dropout).to(DEVICE)

    # Encoder-Gewichte aus Quell-Checkpoint übernehmen
    print(f"  Übertrage Encoder-Gewichte vom Quell-Checkpoint ...")
    dbo_state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(dbo_state, dict) and "model_state" in dbo_state:
        dbo_state = dbo_state["model_state"]
    encoder_state = {
        k.replace("encoder.", "", 1): v.float()
        for k, v in dbo_state.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"  Encoder geladen. Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    print(f"  Classifier-Head: frisch initialisiert ({N_CLASSES} Klassen)\n")

    # Optimizer, Scheduler, Loss
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": args.lr_encoder},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.max_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    best_model_path = model_out / "best_model_weights.pt"

    def evaluate():
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(batch["label"].numpy())
        model.train()
        macro_f1     = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None,
                                 zero_division=0, labels=list(range(N_CLASSES)))
        report       = classification_report(all_labels, all_preds,
                                              target_names=CLASSES, zero_division=0)
        return macro_f1, per_class_f1, report

    # Training Loop
    global_step = 0
    best_f1     = 0.0
    no_improve  = 0
    log_rows    = []
    early_stop  = False
    start_time  = time.time()

    for epoch in range(1, args.max_epochs + 1):
        frozen = epoch <= args.freeze_epochs
        for param in model.encoder.parameters():
            param.requires_grad = not frozen
        status = "Encoder EINGEFROREN" if frozen else "Encoder + Head"
        print(f"  Epoch {epoch}/{args.max_epochs}  [{status}]")

        epoch_loss = 0.0
        n_batches  = 0
        model.train()

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

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % args.eval_every == 0:
                macro_f1, per_class_f1, _ = evaluate()
                avg_loss = epoch_loss / n_batches

                if macro_f1 > best_f1:
                    best_f1    = macro_f1
                    no_improve = 0
                    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()},
                               best_model_path)
                    flag = "✓ Bestes Modell"
                else:
                    no_improve += 1
                    flag = f"kein Fortschritt {no_improve}/{args.patience}"

                print(f"    Step {global_step:>5} | Loss {avg_loss:.4f} | "
                      f"Macro-F1 {macro_f1:.4f} | {flag}")

                log_rows.append({
                    "step": global_step, "epoch": epoch,
                    "train_loss": round(avg_loss, 5),
                    "val_macro_f1": round(macro_f1, 5),
                    "encoder_frozen": frozen,
                    **{f"f1_{c}": round(float(f), 5) for c, f in zip(CLASSES, per_class_f1)},
                })

                if no_improve >= args.patience:
                    print(f"\n    Early Stopping — Beste Macro-F1: {best_f1:.4f}")
                    early_stop = True
                    break
        if early_stop:
            break

    elapsed = time.time() - start_time

    # Finaler Report
    best_state = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in best_state.items()})
    final_macro_f1, final_per_class_f1, final_report = evaluate()

    print(f"\n  Finaler Report {model_key.upper()} (Val-Set, bestes Modell):")
    print(final_report)

    pd.DataFrame(log_rows).to_csv(model_out / "training_log.csv", index=False)

    # Test-Predictions (falls test_file angegeben)
    if df_test_raw is not None:
        print(f"  Erstelle Test-Predictions für {model_key} ...")
        test_dataset = TextDataset(df_test_raw, tokenizer, args.max_length)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs_arr = np.concatenate(all_probs, axis=0)
        preds     = le.inverse_transform(probs_arr.argmax(axis=1))
        pd.DataFrame({"id": df_test_raw["id"], "prediction": preds, **{
            f"prob_{c}": probs_arr[:, i] for i, c in enumerate(CLASSES)
        }}).to_csv(model_out / "predictions_test.csv", index=False)
        print(f"  Predictions gespeichert: {model_out / 'predictions_test.csv'}")

    result = {
        "model": model_key, "task": args.task,
        "best_val_macro_f1": round(best_f1, 5),
        **{f"f1_{c}": round(float(f), 5) for c, f in zip(CLASSES, final_per_class_f1)},
        "training_minutes": round(elapsed / 60, 1),
        "early_stopped": early_stop,
        "out_dir": str(model_out),
    }

    # Einzelner Modell-Report
    (model_out / "final_report.txt").write_text(
        f"GermEval 2026 – Transfer Learning Report\n"
        f"Task: {args.task.upper()}  |  Modell: {model_key} ({model_id})\n"
        f"Quell-Checkpoint: {ckpt_path}\n"
        f"Trainingszeit: {elapsed/60:.1f} min  |  Early Stopped: {early_stop}\n\n"
        f"Beste Val Macro-F1: {best_f1:.5f}\n\n"
        f"{final_report}",
        encoding="utf-8"
    )

    del model
    torch.cuda.empty_cache()
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Alle Modelle trainieren
# ──────────────────────────────────────────────────────────────────────────────
all_results = []
for model_key in args.models:
    result = train_model(model_key)
    if result:
        all_results.append(result)

# ──────────────────────────────────────────────────────────────────────────────
# Globale Ergebnistabelle
# ──────────────────────────────────────────────────────────────────────────────
if all_results:
    results_path = Path(args.results_csv)
    existing     = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    rows         = [{**r, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "source_run": str(SOURCE_RUN)} for r in all_results]
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(results_path, index=False)

    print(f"\n{'='*65}")
    print(f"  ZUSAMMENFASSUNG – Task {args.task.upper()}")
    print(f"{'='*65}")
    print(f"  {'Modell':<12} {'Macro-F1':>10}  {'Zeit':>8}")
    print(f"  {'─'*35}")
    for r in sorted(all_results, key=lambda x: x["best_val_macro_f1"], reverse=True):
        print(f"  {r['model']:<12} {r['best_val_macro_f1']:>10.5f}  {r['training_minutes']:>6.1f} min")
    print(f"\n  Ergebnisse gespeichert: {results_path.resolve()}")
    print(f"  Modell-Checkpoints in:  {OUT.resolve()}")
    print(f"{'='*65}\n")
