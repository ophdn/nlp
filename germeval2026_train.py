"""
GermEval 2026 – Subtask 2: DBO Classification
==============================================
Baseline Training Script: XLM-RoBERTa-Large

VERWENDUNG:
    python germeval2026_train.py \
        --data_dir  data/processed \
        --variant   minimal \
        --out_dir   models/baseline_xlmr \
        --seed      42

    # Training fortsetzen nach Crash:
    python germeval2026_train.py \
        --data_dir  data/processed \
        --variant   minimal \
        --out_dir   models/baseline_xlmr \
        --resume    (kein weiterer Wert nötig, liest Checkpoint automatisch)

AUSGABE:
    models/baseline_xlmr/
        checkpoint_latest.pt     ← nach jeder Evaluierung überschrieben
        best_model_weights.pt    ← bestes Modell (float16, ~1GB)
        training_log.csv         ← Loss + Macro-F1 pro Evaluation
        training_config.json     ← alle Hyperparameter
        final_report.txt         ← Zusammenfassung
    results.csv                  ← EINE Zeile pro Run, wird immer angehängt
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

# ──────────────────────────────────────────────
# 1. CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",     default="data/processed")
parser.add_argument("--variant",      default="minimal", choices=["raw", "minimal"])
parser.add_argument("--out_dir",      default="models/baseline_xlmr")
parser.add_argument("--results_csv",  default="results.csv",
                    help="Globale Ergebnisdatei (wird angehängt, nicht überschrieben)")
parser.add_argument("--resume",       action="store_true",
                    help="Checkpoint aus --out_dir laden und Training fortsetzen")
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--model_name",   default="FacebookAI/xlm-roberta-large")
parser.add_argument("--max_length",   type=int,   default=128)
parser.add_argument("--batch_size",   type=int,   default=32)
parser.add_argument("--lr",           type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_grad_norm",type=float, default=1.0)
parser.add_argument("--warmup_steps", type=int,   default=500)
parser.add_argument("--max_epochs",   type=int,   default=10)
parser.add_argument("--eval_every",   type=int,   default=40)
parser.add_argument("--patience",     type=int,   default=50)
parser.add_argument("--val_size",     type=float, default=0.1)
args = parser.parse_args()

# ──────────────────────────────────────────────
# 2. Setup
# ──────────────────────────────────────────────
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

print(f"\n{'='*60}")
print(f"  GermEval 2026 – Subtask 2 Baseline Training")
print(f"{'='*60}")
print(f"  Device:   {DEVICE}")
print(f"  Modell:   {args.model_name}")
print(f"  Variante: {args.variant}")
print(f"  Resume:   {args.resume}")
print(f"  Ausgabe:  {OUT.resolve()}\n")

# ──────────────────────────────────────────────
# 3. Daten laden & splitten
# ──────────────────────────────────────────────
train_path = Path(args.data_dir) / f"train_{args.variant}.csv"
assert train_path.exists(), f"Nicht gefunden: {train_path}"

df = pd.read_csv(train_path)
df_train, df_val = train_test_split(
    df, test_size=args.val_size, stratify=df["label"], random_state=args.seed
)
print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}")

le = LabelEncoder()
le.fit(df["label"])
CLASSES   = list(le.classes_)
N_CLASSES = len(CLASSES)
print(f"  Klassen: {CLASSES}\n")

# Class Weights
train_counts = pd.Series(le.transform(df_train["label"])).value_counts().sort_index()
counts  = np.array([train_counts.get(i, 1) for i in range(N_CLASSES)], dtype=float)
weights = len(df_train) / (N_CLASSES * counts)
weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"  Class Weights:")
for cls, w in zip(CLASSES, weights):
    print(f"    {cls:<15} {w:.2f}")

# ──────────────────────────────────────────────
# 4. Dataset
# ──────────────────────────────────────────────
print(f"\n  Lade Tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

class DBODataset(Dataset):
    def __init__(self, df, tokenizer, max_length, label_encoder):
        self.labels    = torch.tensor(label_encoder.transform(df["label"]), dtype=torch.long)
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
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
train_dataset = DBODataset(df_train, tokenizer, args.max_length, le)
val_dataset   = DBODataset(df_val,   tokenizer, args.max_length, le)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────
# 5. Modell
# ──────────────────────────────────────────────
class XLMRoBERTaClassifier(nn.Module):
    def __init__(self, model_name, n_classes, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size  # 1024
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_out)

print(f"\n  Lade Modell ...")
model = XLMRoBERTaClassifier(args.model_name, N_CLASSES).to(DEVICE)

# ──────────────────────────────────────────────
# 6. Optimizer & Scheduler
# ──────────────────────────────────────────────
total_steps = len(train_loader) * args.max_epochs
optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler   = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
)
criterion   = nn.CrossEntropyLoss(weight=weights_tensor)

# ──────────────────────────────────────────────
# 7. Checkpoint: Speichern & Laden
# ──────────────────────────────────────────────
CKPT_PATH      = OUT / "checkpoint_latest.pt"
BEST_MODEL_PATH = OUT / "best_model_weights.pt"

def save_checkpoint(step, epoch, best_f1, no_improve, log_rows):
    """
    Speichert Modellgewichte in float16 (kein Optimizer-State → spart ~2GB).
    Scheduler-State wird gespeichert (winzig, wichtig für korrekten LR-Verlauf).
    """
    ckpt = {
        "step":            step,
        "epoch":           epoch,
        "best_f1":         best_f1,
        "no_improve":      no_improve,
        "log_rows":        log_rows,
        "scheduler_state": scheduler.state_dict(),
        # Float16 halbiert die Dateigröße (~1GB statt ~2GB für Large)
        "model_state":     {k: v.half().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"    💾 Checkpoint gespeichert: Step {step} | Macro-F1: {best_f1:.4f}")

def load_checkpoint():
    """Lädt Checkpoint und gibt Trainingsstate zurück."""
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    # Float16 → float32 für Training
    state = {k: v.float() for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"  ✓ Checkpoint geladen: Step {ckpt['step']} | "
          f"Epoch {ckpt['epoch']} | Beste Macro-F1: {ckpt['best_f1']:.4f}")
    return ckpt["step"], ckpt["epoch"], ckpt["best_f1"], ckpt["no_improve"], ckpt["log_rows"]

def save_best_model():
    """Bestes Modell separat in float16 speichern."""
    torch.save(
        {k: v.half().cpu() for k, v in model.state_dict().items()},
        BEST_MODEL_PATH
    )

# ──────────────────────────────────────────────
# 8. Resume-Logik
# ──────────────────────────────────────────────
global_step = 0
start_epoch = 1
best_f1     = 0.0
no_improve  = 0
log_rows    = []

if args.resume and CKPT_PATH.exists():
    global_step, start_epoch, best_f1, no_improve, log_rows = load_checkpoint()
    print(f"  → Fahre fort ab Step {global_step}, Epoch {start_epoch}\n")
elif args.resume:
    print("  ⚠️  --resume gesetzt, aber kein Checkpoint gefunden. Starte neu.\n")

# ──────────────────────────────────────────────
# 9. Evaluation
# ──────────────────────────────────────────────
def evaluate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    model.train()
    macro_f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None,    zero_division=0, labels=list(range(N_CLASSES)))
    report       = classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)
    return macro_f1, per_class_f1, report

# ──────────────────────────────────────────────
# 10. Training Loop
# ──────────────────────────────────────────────
print(f"{'─'*60}")
print(f"  TRAINING  (max {args.max_epochs} Epochs | Eval alle {args.eval_every} Steps | Patience {args.patience})")
print(f"{'─'*60}\n")

start_time  = time.time()
early_stop  = False
epoch_loss  = 0.0
n_batches   = 0
model.train()

for epoch in range(start_epoch, args.max_epochs + 1):
    for batch in train_loader:
        # Überspringe Steps die bereits im Checkpoint sind
        if global_step < (epoch - 1) * len(train_loader):
            global_step += 1
            continue

        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(ids, mask), lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches  += 1
        global_step += 1

        # ── Evaluierung ──
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
                **{f"f1_{cls}": round(f, 5) for cls, f in zip(CLASSES, per_class_f1)},
            })

            # Checkpoint nach JEDER Evaluierung
            save_checkpoint(global_step, epoch, best_f1, no_improve, log_rows)

            if no_improve >= args.patience:
                print(f"\n  ⏹  Early Stopping (Step {global_step} | Beste Macro-F1: {best_f1:.4f})")
                early_stop = True
                break

    if early_stop:
        break

elapsed = time.time() - start_time

# ──────────────────────────────────────────────
# 11. Finaler Classification Report
# ──────────────────────────────────────────────

# Bestes Modell laden für finalen Report
best_state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict({k: v.float() for k, v in best_state.items()})
final_macro_f1, final_per_class_f1, final_report = evaluate()

print(f"\n  Finaler Classification Report (bestes Modell auf Val-Set):\n")
print(final_report)

# ──────────────────────────────────────────────
# 12. Training Log speichern
# ──────────────────────────────────────────────
log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUT / "training_log.csv", index=False)

# ──────────────────────────────────────────────
# 13. Config speichern
# ──────────────────────────────────────────────
config = vars(args)
config.update({
    "classes":             CLASSES,
    "class_weights":       {cls: round(float(w), 4) for cls, w in zip(CLASSES, weights)},
    "best_val_macro_f1":   round(best_f1, 5),
    "total_steps":         global_step,
    "early_stopped":       early_stop,
    "training_minutes":    round(elapsed / 60, 1),
    "device":              str(DEVICE),
    "timestamp":           datetime.now().isoformat(),
})
with open(OUT / "training_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# ──────────────────────────────────────────────
# 14. Globale Results CSV (anhängen, nicht überschreiben)
# ──────────────────────────────────────────────
results_row = {
    "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
    "model_name":       args.model_name,
    "variant":          args.variant,
    "seed":             args.seed,
    "lr":               args.lr,
    "batch_size":       args.batch_size,
    "max_length":       args.max_length,
    "warmup_steps":     args.warmup_steps,
    "weight_decay":     args.weight_decay,
    "max_grad_norm":    args.max_grad_norm,
    "max_epochs":       args.max_epochs,
    "eval_every":       args.eval_every,
    "patience":         args.patience,
    "val_size":         args.val_size,
    "best_val_macro_f1": round(best_f1, 5),
    **{f"f1_{cls}": round(float(f), 5) for cls, f in zip(CLASSES, final_per_class_f1)},
    "total_steps":      global_step,
    "early_stopped":    early_stop,
    "training_minutes": round(elapsed / 60, 1),
    "out_dir":          str(OUT),
}

results_path = Path(args.results_csv)
results_df   = pd.DataFrame([results_row])

if results_path.exists():
    existing = pd.read_csv(results_path)
    results_df = pd.concat([existing, results_df], ignore_index=True)

results_df.to_csv(results_path, index=False)
print(f"\n  📊 Ergebnis angehängt: {results_path.resolve()}")

# ──────────────────────────────────────────────
# 15. Finaler Report
# ──────────────────────────────────────────────
report_text = f"""GermEval 2026 – Subtask 2: Training Report
===========================================
Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M')}
Modell:          {args.model_name}
Variante:        {args.variant}
Seed:            {args.seed}
Device:          {DEVICE}

Hyperparameter (nach Thelen et al., 2025):
  Learning Rate:    {args.lr}
  Warmup Steps:     {args.warmup_steps}
  Batch Size:       {args.batch_size}
  Weight Decay:     {args.weight_decay}
  Grad Clip (L2):   {args.max_grad_norm}
  Max Epochs:       {args.max_epochs}
  Eval Every:       {args.eval_every} Steps
  Patience:         {args.patience} Evaluierungen
  Max Length:       {args.max_length}

Class Weights:
{chr(10).join(f'  {cls:<15} {w:.4f}' for cls, w in zip(CLASSES, weights))}

Ergebnis:
  Beste Val Macro-F1:  {best_f1:.5f}
  Early Stopped:       {early_stop}
  Gesamt Steps:        {global_step}
  Trainingszeit:       {elapsed/60:.1f} Minuten

Classification Report (Val-Set, bestes Modell):
{final_report}

Gespeicherte Dateien:
  checkpoint_latest.pt   ← letzter Checkpoint (Resume)
  best_model_weights.pt  ← bestes Modell nach Macro-F1 (float16, ~1GB)
  training_log.csv       ← Loss + F1 pro Evaluierung
  training_config.json   ← alle Hyperparameter
  final_report.txt       ← dieser Report
  {args.results_csv:<25} ← globale Ergebnistabelle (alle Runs)
"""
(OUT / "final_report.txt").write_text(report_text, encoding="utf-8")
print(f"\n  ✅ Alle Dateien in: {OUT.resolve()}")
print(f"  → checkpoint_latest.pt   (Resume nach Crash)")
print(f"  → best_model_weights.pt  (bestes Modell, float16)")
print(f"  → training_log.csv       (Trainingsverlauf)")
print(f"  → training_config.json   (Konfiguration)")
print(f"  → final_report.txt       (Zusammenfassung)")
print(f"  → {args.results_csv:<25} (globale Ergebnistabelle)\n")
