"""
GermEval 2026 – Transfer Learning von DBO-Checkpoint
======================================================
Lädt den Encoder eines trainierten DBO-Modells und fine-tuned ihn
auf einen anderen Subtask (c2a, vio, def).

GPU-Zeitersparnis gegenüber Training von Null:
  - Encoder ist bereits auf deutschen politischen Tweets eingestellt
  - Konvergiert in 2–4 Epochs statt 10
  - Optionales Encoder-Freeze in Epoch 1 (nur Head trainieren)
  - Differenziertes LR: Encoder langsamer als Head

VERWENDUNG:
    # C2A (binär) von DBO-gbert-Checkpoint:
    python germeval2026_transfer.py \\
        --task c2a \\
        --dbo_checkpoint model_dataset_gridsearch/all5_aug-paraphrase/model_gbert/best_model_weights.pt \\
        --model_name deepset/gbert-large \\
        --train_file ../data/GermEval2026/data/c2a/c2a_train_26.csv \\
        --out_dir transfer_runs/gbert_c2a

    # VIO (6 Klassen) von DBO-mdeberta-Checkpoint:
    python germeval2026_transfer.py \\
        --task vio \\
        --dbo_checkpoint model_dataset_gridsearch/all5_aug-paraphrase/model_mdeberta/best_model_weights.pt \\
        --model_name microsoft/mdeberta-v3-base \\
        --train_file ../data/GermEval2026/data/vio/vio_train_26.csv \\
        --out_dir transfer_runs/mdeberta_vio

    # DEF (binär) von DBO-gelectra-Checkpoint:
    python germeval2026_transfer.py \\
        --task def \\
        --dbo_checkpoint model_dataset_gridsearch/all5_aug-paraphrase/model_gelectra/best_model_weights.pt \\
        --model_name deepset/gelectra-large-germanquad \\
        --train_file ../data/GermEval2026/data/def/def_train_26.csv \\
        --out_dir transfer_runs/gelectra_def
"""

import argparse
import json
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
# Task-Konfiguration: Label-Spalte + bekannte Klassen pro Subtask
# ──────────────────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "c2a": {
        "label_col": "c2a",
        "classes":   ["FALSE", "TRUE"],
    },
    "vio": {
        "label_col": "vio",
        "classes":   ["call2violence", "glorification", "nothing", "propensity", "support", "other"],
    },
    "def": {
        "label_col": "def",
        "classes":   ["FALSE", "TRUE"],
    },
    "dbo": {
        "label_col": "dbo",
        "classes":   ["agitation", "criticism", "nothing", "subversive"],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 – Transfer Learning")
parser.add_argument("--task",            required=True, choices=list(TASK_CONFIG),
                    help="Ziel-Subtask: c2a | vio | def")
parser.add_argument("--dbo_checkpoint",  required=True,
                    help="Pfad zur best_model_weights.pt des DBO-Modells (Encoder-Quelle)")
parser.add_argument("--model_name",      required=True,
                    help="HuggingFace Modell-ID (muss zum Checkpoint passen)")
parser.add_argument("--train_file",      required=True,
                    help="Trainingsdaten (CSV mit Semikolon oder Komma, mit Label-Spalte)")
parser.add_argument("--out_dir",         default="transfer_runs/output",
                    help="Ausgabeverzeichnis")
parser.add_argument("--results_csv",     default="results_transfer.csv")
parser.add_argument("--seed",            type=int,   default=42)
parser.add_argument("--max_length",      type=int,   default=128)
parser.add_argument("--batch_size",      type=int,   default=16)
parser.add_argument("--lr_encoder",      type=float, default=5e-6,
                    help="LR für den vortrainierten Encoder (niedrig halten)")
parser.add_argument("--lr_head",         type=float, default=2e-5,
                    help="LR für den neuen Classifier-Head")
parser.add_argument("--weight_decay",    type=float, default=0.01)
parser.add_argument("--max_grad_norm",   type=float, default=1.0)
parser.add_argument("--warmup_steps",    type=int,   default=100,
                    help="Weniger Warmup nötig da Encoder bereits eingestellt")
parser.add_argument("--max_epochs",      type=int,   default=5,
                    help="Weniger Epochs nötig als Training von Null (default 5 statt 10)")
parser.add_argument("--eval_every",      type=int,   default=40)
parser.add_argument("--patience",        type=int,   default=30,
                    help="Frühere Early-Stopping da Konvergenz schneller")
parser.add_argument("--val_size",        type=float, default=0.1)
parser.add_argument("--freeze_epochs",   type=int,   default=1,
                    help="Encoder in den ersten N Epochs einfrieren (nur Head trainieren). "
                         "Spart GPU-Zeit in der Anfangsphase. 0 = kein Freeze.")
parser.add_argument("--dropout",         type=float, default=0.1)
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT    = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

task_cfg  = TASK_CONFIG[args.task]
LABEL_COL = task_cfg["label_col"]

print(f"\n{'='*65}")
print(f"  GermEval 2026 – Transfer Learning")
print(f"{'='*65}")
print(f"  Task:         {args.task.upper()}  ({LABEL_COL})")
print(f"  Modell:       {args.model_name}")
print(f"  DBO-Ckpt:     {args.dbo_checkpoint}")
print(f"  Device:       {DEVICE}")
print(f"  LR Encoder:   {args.lr_encoder}  (niedriger als Head)")
print(f"  LR Head:      {args.lr_head}")
print(f"  Freeze:       Encoder für erste {args.freeze_epochs} Epoch(s) eingefroren")
print(f"  Max Epochs:   {args.max_epochs}")
print(f"  Ausgabe:      {OUT.resolve()}")
print(f"{'='*65}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Daten laden
# ──────────────────────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    """Liest Trainings-CSV flexibel (Semikolon oder Komma, preprocessed oder raw)."""
    for kwargs, text_col, label_col in [
        ({"sep": ";"}, "description", LABEL_COL),
        ({},           "text",        "label"),
        ({"sep": ";"}, "text",        "label"),
        ({},           "description", LABEL_COL),
    ]:
        try:
            df = pd.read_csv(path, **kwargs)
            if text_col in df.columns and label_col in df.columns:
                df = df[[text_col, label_col]].dropna()
                df.columns = ["text", "label"]
                return df
        except Exception:
            continue
    raise ValueError(f"Kann Trainingsdaten nicht lesen: {path}\n"
                     f"Erwartet Spalten: description/{LABEL_COL} oder text/label")

df = load_data(Path(args.train_file))
df_train, df_val = train_test_split(
    df, test_size=args.val_size, stratify=df["label"], random_state=args.seed
)
print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}")

le = LabelEncoder()
le.fit(task_cfg["classes"])  # feste Reihenfolge, unabhängig von Datenlage
CLASSES   = list(le.classes_)
N_CLASSES = len(CLASSES)
print(f"  Klassen ({N_CLASSES}): {CLASSES}\n")

# Class Weights
train_counts = pd.Series(le.transform(df_train["label"])).value_counts().sort_index()
counts  = np.array([train_counts.get(i, 1) for i in range(N_CLASSES)], dtype=float)
weights = len(df_train) / (N_CLASSES * counts)
weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"  Class Weights:")
for cls, w in zip(CLASSES, weights):
    print(f"    {cls:<20} {w:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n  Lade Tokenizer: {args.model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, label_encoder):
        self.labels    = torch.tensor(label_encoder.transform(df["label"]), dtype=torch.long)
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }

print("  Tokenisiere ...")
train_dataset = TextDataset(df_train, tokenizer, args.max_length, le)
val_dataset   = TextDataset(df_val,   tokenizer, args.max_length, le)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────────────────────────────────────
# Modell (identische Architektur wie Training)
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

print(f"\n  Lade Modell-Architektur ...")
model = TransferClassifier(args.model_name, N_CLASSES, args.dropout).to(DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# Encoder-Gewichte aus DBO-Checkpoint laden (Head wird ignoriert)
# ──────────────────────────────────────────────────────────────────────────────
ckpt_path = Path(args.dbo_checkpoint)
if not ckpt_path.exists():
    raise FileNotFoundError(f"DBO-Checkpoint nicht gefunden: {ckpt_path}")

print(f"  Lade Encoder-Gewichte aus DBO-Checkpoint ...")
dbo_state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
# best_model_weights.pt enthält direkt den state_dict (float16)
if isinstance(dbo_state, dict) and "model_state" in dbo_state:
    dbo_state = dbo_state["model_state"]

# Nur Encoder-Gewichte übernehmen, Classifier-Head überspringen
encoder_state = {
    k.replace("encoder.", "", 1): v.float()
    for k, v in dbo_state.items()
    if k.startswith("encoder.")
}
missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
print(f"  Encoder geladen. Missing: {len(missing)}  Unexpected: {len(unexpected)}")
if missing:
    print(f"  WARNUNG – fehlende Keys: {missing[:5]}")

print(f"  Classifier-Head: frisch initialisiert (zufällig, {N_CLASSES} Klassen)\n")

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer mit differenzierten LRs
# ──────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(),    "lr": args.lr_encoder},
    {"params": model.classifier.parameters(), "lr": args.lr_head},
], weight_decay=args.weight_decay)

total_steps = len(train_loader) * args.max_epochs
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps,
)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# ──────────────────────────────────────────────────────────────────────────────
# Speichern & Laden
# ──────────────────────────────────────────────────────────────────────────────
CKPT_PATH       = OUT / "checkpoint_latest.pt"
BEST_MODEL_PATH = OUT / "best_model_weights.pt"

def save_best_model():
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, BEST_MODEL_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
print(f"{'─'*65}")
print(f"  TRAINING  (max {args.max_epochs} Epochs | Eval alle {args.eval_every} Steps)")
print(f"{'─'*65}\n")

global_step = 0
best_f1     = 0.0
no_improve  = 0
log_rows    = []
early_stop  = False
start_time  = time.time()

for epoch in range(1, args.max_epochs + 1):

    # Encoder einfrieren / auftauen
    frozen = epoch <= args.freeze_epochs
    for param in model.encoder.parameters():
        param.requires_grad = not frozen
    if frozen:
        print(f"  Epoch {epoch}: Encoder eingefroren — nur Head wird trainiert")
    else:
        print(f"  Epoch {epoch}: Encoder + Head werden trainiert")

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
            macro_f1, per_class_f1, report = evaluate()
            avg_loss = epoch_loss / n_batches

            improved = macro_f1 > best_f1
            if improved:
                best_f1    = macro_f1
                no_improve = 0
                save_best_model()
                flag = "✓ Bestes Modell"
            else:
                no_improve += 1
                flag = f"kein Fortschritt {no_improve}/{args.patience}"

            print(f"  Step {global_step:>5} | Ep {epoch}/{args.max_epochs} | "
                  f"Loss {avg_loss:.4f} | Macro-F1 {macro_f1:.4f} | {flag}")

            log_rows.append({
                "step": global_step, "epoch": epoch,
                "train_loss": round(avg_loss, 5),
                "val_macro_f1": round(macro_f1, 5),
                "encoder_frozen": frozen,
                **{f"f1_{cls}": round(float(f), 5) for cls, f in zip(CLASSES, per_class_f1)},
            })

            if no_improve >= args.patience:
                print(f"\n  Early Stopping (Step {global_step} | Beste Macro-F1: {best_f1:.4f})")
                early_stop = True
                break

    if early_stop:
        break

elapsed = time.time() - start_time

# ──────────────────────────────────────────────────────────────────────────────
# Finaler Report
# ──────────────────────────────────────────────────────────────────────────────
best_state = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict({k: v.float() for k, v in best_state.items()})
final_macro_f1, final_per_class_f1, final_report = evaluate()

print(f"\n  Finaler Report (bestes Modell auf Val-Set):")
print(final_report)

pd.DataFrame(log_rows).to_csv(OUT / "training_log.csv", index=False)

config = vars(args)
config.update({
    "task":             args.task,
    "classes":          CLASSES,
    "n_classes":        N_CLASSES,
    "class_weights":    {cls: round(float(w), 4) for cls, w in zip(CLASSES, weights)},
    "best_val_macro_f1": round(best_f1, 5),
    "total_steps":      global_step,
    "early_stopped":    early_stop,
    "training_minutes": round(elapsed / 60, 1),
    "device":           str(DEVICE),
    "timestamp":        datetime.now().isoformat(),
})
with open(OUT / "training_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

report_text = f"""GermEval 2026 – Transfer Learning Report
==========================================
Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M')}
Task:             {args.task.upper()}
Modell:           {args.model_name}
DBO-Checkpoint:   {args.dbo_checkpoint}
Device:           {DEVICE}

Transfer-Strategie:
  Encoder LR:     {args.lr_encoder}  (vortrainiert, niedrig)
  Head LR:        {args.lr_head}   (neu initialisiert, normal)
  Freeze Epochs:  {args.freeze_epochs}
  Max Epochs:     {args.max_epochs}
  Patience:       {args.patience}

Class Weights:
{chr(10).join(f'  {cls:<20} {w:.4f}' for cls, w in zip(CLASSES, weights))}

Ergebnis:
  Beste Val Macro-F1:  {best_f1:.5f}
  Early Stopped:       {early_stop}
  Trainingszeit:       {elapsed/60:.1f} Minuten

Classification Report (Val-Set, bestes Modell):
{final_report}
"""
(OUT / "final_report.txt").write_text(report_text, encoding="utf-8")

results_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "task": args.task, "model_name": args.model_name,
    "dbo_checkpoint": args.dbo_checkpoint,
    "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
    "freeze_epochs": args.freeze_epochs,
    "best_val_macro_f1": round(best_f1, 5),
    **{f"f1_{cls}": round(float(f), 5) for cls, f in zip(CLASSES, final_per_class_f1)},
    "training_minutes": round(elapsed / 60, 1),
    "out_dir": str(OUT),
}
results_path = Path(args.results_csv)
existing     = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
pd.concat([existing, pd.DataFrame([results_row])], ignore_index=True).to_csv(results_path, index=False)

print(f"\n  Alle Dateien in: {OUT.resolve()}")
print(f"  → best_model_weights.pt  (bestes Modell, float16)")
print(f"  → training_log.csv       (Trainingsverlauf)")
print(f"  → training_config.json   (Konfiguration)")
print(f"  → final_report.txt       (Zusammenfassung)\n")
