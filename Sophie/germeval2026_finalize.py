"""
GermEval 2026 – Finalize: Warm-Start-Training auf train(+trial) für die Submission
===================================================================================
Baut auf den BESTEHENDEN Checkpoints auf (kein Full-Retrain) und produziert das
finale Submission-Modell pro Task (c2a | def | vio).

ZWEI-PHASEN-METHODIK
--------------------
  Phase A  (--mode select):  Warm-Start vom Checkpoint, trainiere auf  train(+aug),
                             validiere ehrlich auf  trial.  → wählt best_epoch +
                             per-class-F1 (für die Ensemble-Gewichte).
  Phase B  (--mode final):   Warm-Start, trainiere auf  train + trial (+aug)  OHNE
                             Holdout für die gewählte Epochenzahl, predicte test_26
                             → predictions_test.csv (+ Submission).
  --mode both  (default):    A dann B in einem Lauf.

WARM-START (kein Neutraining von Null)
--------------------------------------
  1) --warm_start_dir/model_<m>/best_model_weights.pt   (Encoder + Head, task-getuned)
     → bevorzugt; nur wenige Epochen Nachtraining.
  2) Fallback --source_run/model_<m>/best_model_weights.pt
     → nur Encoder, Head frisch (wie germeval2026_transfer.py).

FROM-SCRATCH (komplettes Training, kein Warm-Start)
---------------------------------------------------
  --from_scratch  → nur HF-Pretrain-Gewichte + frischer Head, längeres Training
                    (Default max_epochs=10, lr_encoder=2e-5). Weder --warm_start_dir
                    noch --source_run nötig. Siehe run_finalize.ps1 -Run <task>.

DATEN
-----
  Die 2026-`test`-Files sind UNLABELED (= Blind-Submission). Gelabelt: train + trial.
  Für die Submission wird auf train+trial trainiert und test_26 nur vorhergesagt.
  Optional: --aug_file mit synthetischen Beispielen (gleiche Spalten wie train).

BEISPIELE
---------
  # DEF – Ensemble aus 3 Modellen, mit Augmentierung, beide Phasen:
  python germeval2026_finalize.py \
      --task def \
      --train_file ../GermEval2026/data/def/def_train.csv \
      --trial_file ../GermEval2026/data/def/def_trial.csv \
      --test_file  ../GermEval2026/data/def/def_test.csv \
      --aug_file   synthethic_data/def_aug.csv \
      --warm_start_dir transfer_runs/def \
      --source_run     model_dataset_gridsearch/all5_aug-paraphrase \
      --models gbert gelectra mdeberta \
      --ensemble \
      --out_dir final_runs/def

  # VIO – gbert (+ optional gelectra) mit Augmentierung:
  python germeval2026_finalize.py \
      --task vio \
      --train_file ../GermEval2026/data/vio/vio_train_26.csv \
      --trial_file ../GermEval2026/data/vio/vio_trial.csv \
      --test_file  ../GermEval2026/data/vio/vio_test_26.csv \
      --aug_file   synthethic_data/vio_aug.csv \
      --warm_start_dir transfer_runs/vio \
      --source_run     model_dataset_gridsearch/all5_aug-paraphrase \
      --models gbert \
      --out_dir final_runs/vio

  # C2A – gbert, ohne Augmentierung:
  python germeval2026_finalize.py \
      --task c2a \
      --train_file ../GermEval2026/data/c2a/c2a_train_26.csv \
      --trial_file ../GermEval2026/data/c2a/c2a_trial.csv \
      --test_file  ../GermEval2026/data/c2a/c2a_test_26.csv \
      --warm_start_dir transfer_runs/c2a \
      --source_run     model_dataset_gridsearch/all5_aug-paraphrase \
      --models gbert \
      --out_dir final_runs/c2a
"""

import argparse
import json
import math
import re
import random
import time
import zipfile
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
# Task-Konfiguration  (identisch zu germeval2026_transfer.py)
# ──────────────────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "c2a": {"classes": ["FALSE", "TRUE"]},
    "vio": {"classes": ["call2violence", "glorification", "nothing", "other", "propensity", "support"]},
    "def": {"classes": ["FALSE", "TRUE"]},
    "dbo": {"classes": ["agitation", "criticism", "nothing", "subversive"]},
}

LABEL_FIXES = {
    "prospensity": "propensity",   # VIO: Tippfehler im Datensatz
    "True":        "TRUE",         # C2A / DEF: Python-Casing
    "False":       "FALSE",
}

MODEL_REGISTRY = {
    "gbert":    "deepset/gbert-large",
    "xlmr":     "FacebookAI/xlm-roberta-large",
    "deberta":  "microsoft/deberta-v3-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "gelectra": "deepset/gelectra-large-germanquad",
}

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (identisch zu germeval2026_transfer.py)
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


def load_and_preprocess(path: Path, task: str, has_label: bool = True) -> pd.DataFrame:
    """Lädt Raw-CSV und preprocesst identisch zur transfer.py (minimal-Variante)."""
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
        df["label"] = df["label"].astype(str).str.strip().apply(lambda x: LABEL_FIXES.get(x, x))
        valid_labels = set(TASK_CONFIG[task]["classes"])
        invalid = ~df["label"].isin(valid_labels)
        if invalid.any():
            print(f"    WARNUNG: {invalid.sum()} Zeilen mit unbekannten Labels entfernt: "
                  f"{df.loc[invalid, 'label'].value_counts().to_dict()}")
            df = df[~invalid]
    else:
        id_candidates = [c for c in df.columns if c.lower() == "id"]
        id_col = id_candidates[0] if id_candidates else df.columns[0]
        df = df[[id_col, text_col]].rename(columns={id_col: "id", text_col: "text"})

    df["text"] = df["text"].astype(str).str.strip().apply(preprocess_minimal)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 0]
    print(f"    {len(df):,} Zeilen geladen und preprocesst ({path.name})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 – Finalize (Warm-Start)")
parser.add_argument("--task",         required=True, choices=list(TASK_CONFIG))
parser.add_argument("--train_file",   required=True)
parser.add_argument("--trial_file",   required=True,
                    help="Gelabeltes Dev-Set (trial). Phase A: Validierung. Phase B: ins Training gemischt.")
parser.add_argument("--test_file",    default=None,
                    help="Unlabeled Blind-Submission-Set (test_26). Nur Prediction.")
parser.add_argument("--aug_file",     default=None,
                    help="Optionale synthetische Daten (gleiche Spalten wie train). Nur ins TRAINING.")
parser.add_argument("--warm_start_dir", default=None,
                    help="Ordner mit task-getunten Checkpoints (z.B. transfer_runs/def). Encoder+Head.")
parser.add_argument("--source_run",   default=None,
                    help="Fallback-Quell-Checkpoints (Encoder), z.B. model_dataset_gridsearch/all5_aug-paraphrase.")
parser.add_argument("--from_scratch", action="store_true",
                    help="Training komplett von Grund auf (nur HF-Pretrain-Gewichte, KEIN Warm-Start "
                         "aus euren Checkpoints). Setzt passendere Default-Hyperparameter.")
parser.add_argument("--models",       nargs="+", default=["gbert"], choices=list(MODEL_REGISTRY))
parser.add_argument("--out_dir",      required=True)
parser.add_argument("--mode",         choices=["select", "final", "both"], default="both")
parser.add_argument("--ensemble",     action="store_true",
                    help="Nach Phase B gewichtetes Soft-Voting über alle Modelle.")

# Submission
parser.add_argument("--team",         default="TUMination")
parser.add_argument("--run",          default="1")
parser.add_argument("--submission_col", default=None,
                    help="Spaltenname in der Submission-CSV (Default: Task-Name, z.B. 'def').")

# Hyperparameter
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--max_length",    type=int,   default=128)
parser.add_argument("--batch_size",    type=int,   default=16)
parser.add_argument("--lr_encoder",    type=float, default=5e-6)
parser.add_argument("--lr_head",       type=float, default=2e-5)
parser.add_argument("--weight_decay",  type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--warmup_steps",  type=int,   default=100)
parser.add_argument("--max_epochs",    type=int,   default=5,
                    help="Max Epochen in Phase A (Select).")
parser.add_argument("--final_epochs",  type=int,   default=0,
                    help="Epochen in Phase B. 0 = aus Phase A übernehmen (best_epoch), sonst Default 3.")
parser.add_argument("--eval_every",    type=int,   default=40)
parser.add_argument("--patience",      type=int,   default=30)
parser.add_argument("--freeze_epochs", type=int,   default=1)
parser.add_argument("--dropout",       type=float, default=0.1)
args = parser.parse_args()

if not args.from_scratch and args.warm_start_dir is None and args.source_run is None:
    parser.error("Ohne --from_scratch muss --warm_start_dir ODER --source_run angegeben werden.")

# Bei From-Scratch passendere Defaults setzen (nur wenn vom Nutzer nicht überschrieben).
if args.from_scratch:
    if args.max_epochs   == parser.get_default("max_epochs"):   args.max_epochs   = 10
    if args.lr_encoder   == parser.get_default("lr_encoder"):   args.lr_encoder   = 2e-5
    if args.warmup_steps == parser.get_default("warmup_steps"): args.warmup_steps = 200
    if args.patience     == parser.get_default("patience"):     args.patience     = 40

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
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT       = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
CLASSES   = TASK_CONFIG[args.task]["classes"]
N_CLASSES = len(CLASSES)
SUB_COL   = args.submission_col or args.task

print(f"\n{'='*70}")
print(f"  GermEval 2026 – FINALIZE  (Warm-Start)")
print(f"{'='*70}")
print(f"  Task:          {args.task.upper()}  ({N_CLASSES} Klassen: {CLASSES})")
print(f"  Modelle:       {args.models}")
print(f"  Modus:         {args.mode}   |  Ensemble: {args.ensemble}")
print(f"  From-Scratch:  {args.from_scratch}  (max_epochs={args.max_epochs}, lr_enc={args.lr_encoder})")
print(f"  Warm-Start:    {args.warm_start_dir}")
print(f"  Source-Run:    {args.source_run}")
print(f"  Augmentierung: {args.aug_file}")
print(f"  Device:        {DEVICE}")
print(f"  Ausgabe:       {OUT.resolve()}")
print(f"{'='*70}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Daten laden
# ──────────────────────────────────────────────────────────────────────────────
print("  Preprocessing ...")
df_train_raw = load_and_preprocess(Path(args.train_file), args.task, has_label=True)
df_trial_raw = load_and_preprocess(Path(args.trial_file), args.task, has_label=True)

df_aug = None
if args.aug_file:
    df_aug = load_and_preprocess(Path(args.aug_file), args.task, has_label=True)
    print(f"    Augmentierung: {df_aug['label'].value_counts().to_dict()}")

df_test_raw = None
if args.test_file:
    df_test_raw = load_and_preprocess(Path(args.test_file), args.task, has_label=False)
    print(f"    {len(df_test_raw):,} Test-Tweets (Blind-Submission).")

le = LabelEncoder()
le.fit(CLASSES)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / Architektur  (identisch mit Training)
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


def _load_state_any(path):
    """Lädt ein Checkpoint-Dict (half/full, optional in 'model_state' verschachtelt) als float."""
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    return {k: v.float() for k, v in state.items()}


def build_model(model_key: str):
    """Initialisiert Modell und lädt Warm-Start-Gewichte. Gibt (model, warm_source) zurück."""
    model_id = MODEL_REGISTRY[model_key]
    model    = TransferClassifier(model_id, N_CLASSES, args.dropout).to(DEVICE)

    # 0) From-Scratch: nur HF-Pretrain-Gewichte (AutoModel.from_pretrained), Head frisch.
    if args.from_scratch:
        print(f"  From-Scratch: nur HF-Pretrain ({model_id}), Head frisch, KEIN Warm-Start.")
        return model, "scratch"

    # 1) Bevorzugt: voller task-getunter Checkpoint (Encoder + Head)
    if args.warm_start_dir:
        warm_ckpt = Path(args.warm_start_dir) / f"model_{model_key}" / "best_model_weights.pt"
        if warm_ckpt.exists():
            sd = _load_state_any(warm_ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            head_ok = any(k.startswith("classifier.") for k in sd)
            print(f"  Warm-Start (Encoder+Head) ← {warm_ckpt}")
            print(f"    Missing: {len(missing)}  Unexpected: {len(unexpected)}  Head geladen: {head_ok}")
            return model, "warm"

    # 2) Fallback: nur Encoder aus Quell-Run, Head frisch
    if args.source_run:
        src_ckpt = Path(args.source_run) / f"model_{model_key}" / "best_model_weights.pt"
        if src_ckpt.exists():
            sd = _load_state_any(src_ckpt)
            encoder_state = {k.replace("encoder.", "", 1): v
                             for k, v in sd.items() if k.startswith("encoder.")}
            missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"  Warm-Start (nur Encoder) ← {src_ckpt}")
            print(f"    Missing: {len(missing)}  Unexpected: {len(unexpected)}  Head: frisch")
            return model, "source"

    raise FileNotFoundError(
        f"Kein Checkpoint für '{model_key}' gefunden in "
        f"{args.warm_start_dir or '-'} / {args.source_run or '-'}"
    )


def class_weights_for(df):
    enc    = pd.Series(le.transform(df["label"])).value_counts().sort_index()
    counts = np.array([enc.get(i, 1) for i in range(N_CLASSES)], dtype=float)
    return len(df) / (N_CLASSES * counts)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        labels.extend(batch["label"].numpy())
    model.train()
    macro    = f1_score(labels, preds, average="macro", zero_division=0)
    per_cls  = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(N_CLASSES)))
    report   = classification_report(labels, preds, target_names=CLASSES, zero_division=0)
    return macro, per_cls, report


@torch.no_grad()
def predict_probs(model, df, tokenizer):
    ds     = TextDataset(df, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    probs = []
    for batch in loader:
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_loop(model, train_df, val_df, tokenizer, max_epochs, ckpt_path, with_earlystop):
    """Trainiert. Mit val_df: Early-Stopping + bestes Modell. Ohne: feste Epochen, letztes Modell.
    Gibt (best_f1, best_per_cls, best_report, best_epoch, elapsed_min) zurück."""
    train_loader = DataLoader(TextDataset(train_df, tokenizer, args.max_length, le),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_df is not None:
        val_loader = DataLoader(TextDataset(val_df, tokenizer, args.max_length, le),
                                batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = torch.tensor(class_weights_for(train_df), dtype=torch.float).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": args.lr_encoder},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    total_steps = len(train_loader) * max_epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    criterion   = nn.CrossEntropyLoss(weight=weights)

    best_f1, best_per_cls, best_report, best_epoch = 0.0, None, "", max_epochs
    no_improve, global_step = 0, 0
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        frozen = epoch <= args.freeze_epochs
        for p in model.encoder.parameters():
            p.requires_grad = not frozen
        print(f"  Epoch {epoch}/{max_epochs}  [{'Encoder EINGEFROREN' if frozen else 'Encoder + Head'}]")
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            loss   = criterion(logits, batch["label"].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if val_loader is not None and global_step % args.eval_every == 0:
                macro, per_cls, report = evaluate(model, val_loader)
                if macro > best_f1:
                    best_f1, best_per_cls, best_report, best_epoch = macro, per_cls, report, epoch
                    no_improve = 0
                    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckpt_path)
                    flag = "✓ best"
                else:
                    no_improve += 1
                    flag = f"no-improve {no_improve}/{args.patience}"
                print(f"    Step {global_step:>5} | Macro-F1 {macro:.4f} | {flag}")
                if no_improve >= args.patience:
                    print(f"    Early Stopping — beste Macro-F1: {best_f1:.4f}")
                    elapsed = (time.time() - start) / 60
                    return best_f1, best_per_cls, best_report, best_epoch, elapsed

    elapsed = (time.time() - start) / 60

    if val_loader is None:
        # Phase B ohne Holdout: letztes Modell speichern
        torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckpt_path)
        return None, None, "", max_epochs, elapsed

    # Bestes Modell zurückladen für finalen Report
    model.load_state_dict(_load_state_any(ckpt_path))
    best_f1, best_per_cls, best_report, _ = evaluate(model, val_loader)
    return best_f1, best_per_cls, best_report, best_epoch, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Pro-Modell-Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_model(model_key: str) -> dict:
    model_out = OUT / f"model_{model_key}"
    model_out.mkdir(parents=True, exist_ok=True)
    model_id  = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    result    = {"model": model_key, "task": args.task}

    print(f"\n{'─'*70}\n  MODELL: {model_key.upper()}  ({model_id})\n{'─'*70}")

    select_metrics_path = model_out / "select_metrics.json"
    best_epoch = args.final_epochs if args.final_epochs > 0 else 3
    per_class_f1 = None

    # ── Phase A: Select (train → eval auf trial) ────────────────────────────────
    if args.mode in ("select", "both"):
        print("  ▶ Phase A – Select (Validierung auf trial)")
        model, warm_src = build_model(model_key)
        train_a = pd.concat([df_train_raw] + ([df_aug] if df_aug is not None else []),
                            ignore_index=True)
        f1, per_cls, report, best_epoch, mins = train_loop(
            model, train_a, df_trial_raw, tokenizer,
            args.max_epochs, model_out / "select_best.pt", with_earlystop=True,
        )
        per_class_f1 = [float(x) for x in per_cls]
        print(f"\n  Phase-A Trial-Report ({model_key}):\n{report}")
        select_metrics_path.write_text(json.dumps({
            "best_val_macro_f1": round(float(f1), 5),
            "best_epoch": int(best_epoch),
            "per_class_f1": {c: round(float(x), 5) for c, x in zip(CLASSES, per_cls)},
            "training_minutes": round(mins, 1),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (model_out / "select_report.txt").write_text(
            f"GermEval 2026 – FINALIZE Phase A (Select)\n"
            f"Task: {args.task.upper()}  |  Modell: {model_key} ({model_id})\n"
            f"Warm-Start: {warm_src}  |  Augmentierung: {args.aug_file}\n"
            f"Trainingszeit: {mins:.1f} min  |  best_epoch: {best_epoch}\n\n"
            f"Beste Trial Macro-F1: {f1:.5f}\n\n{report}", encoding="utf-8")
        result["select_macro_f1"] = round(float(f1), 5)
        result["best_epoch"]      = int(best_epoch)
        del model
        torch.cuda.empty_cache()
    elif select_metrics_path.exists():
        sm = json.loads(select_metrics_path.read_text(encoding="utf-8"))
        best_epoch   = sm.get("best_epoch", best_epoch)
        per_class_f1 = [sm["per_class_f1"][c] for c in CLASSES]

    # ── Phase B: Final (train+trial → predict test) ─────────────────────────────
    if args.mode in ("final", "both"):
        n_epochs = args.final_epochs if args.final_epochs > 0 else max(1, int(best_epoch))
        print(f"  ▶ Phase B – Final (train+trial, {n_epochs} Epoche(n), kein Holdout)")
        model, warm_src = build_model(model_key)
        train_b = pd.concat([df_train_raw, df_trial_raw] + ([df_aug] if df_aug is not None else []),
                            ignore_index=True)
        print(f"    Final-Trainingsgröße: {len(train_b):,}  ({train_b['label'].value_counts().to_dict()})")
        _, _, _, _, mins = train_loop(
            model, train_b, None, tokenizer,
            n_epochs, model_out / "best_model_weights.pt", with_earlystop=False,
        )
        result["final_minutes"] = round(mins, 1)

        if df_test_raw is not None:
            probs = predict_probs(model, df_test_raw, tokenizer)
            preds = le.inverse_transform(probs.argmax(axis=1))
            pd.DataFrame({"id": df_test_raw["id"], "prediction": preds,
                          **{f"prob_{c}": probs[:, i] for i, c in enumerate(CLASSES)}}
                         ).to_csv(model_out / "predictions_test.csv", index=False)
            print(f"    Predictions → {model_out / 'predictions_test.csv'}")
            result["_test_probs"] = probs  # für Ensemble (in-memory)
        del model
        torch.cuda.empty_cache()

    result["per_class_f1"] = per_class_f1
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Submission-Helper
# ──────────────────────────────────────────────────────────────────────────────
def write_submission(ids, pred_labels, tag: str):
    sub_dir = OUT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"{args.team}{args.run}_{args.task}_{tag}.csv"
    zip_name = f"{args.team}{args.run}_{args.task}_{tag}.zip"
    csv_path = sub_dir / csv_name
    pd.DataFrame({"id": ids, SUB_COL: pred_labels}).to_csv(csv_path, sep=";", index=False)
    with zipfile.ZipFile(sub_dir / zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=f"{args.team}{args.run}_{args.task}.csv")
    dist = pd.Series(pred_labels).value_counts().to_dict()
    print(f"  Submission [{tag}] → {csv_path}  |  Verteilung: {dist}")
    return csv_path


def weighted_soft_voting(probs_list, per_class_f1_list):
    """ensemble_prob[:,k] = Σ_i w_i[k]·probs_i[:,k] / Σ_i w_i[k],  w_i[k] = Trial-F1."""
    W           = np.array(per_class_f1_list)             # (M, C)
    probs_stack = np.stack(probs_list, axis=0)            # (M, N, C)
    weighted    = (probs_stack * W[:, None, :]).sum(axis=0)
    norm        = W.sum(axis=0)[None, :]
    norm        = np.where(norm == 0, 1.0, norm)
    return weighted / norm


# ──────────────────────────────────────────────────────────────────────────────
# Alle Modelle
# ──────────────────────────────────────────────────────────────────────────────
all_results = []
for mk in args.models:
    all_results.append(run_model(mk))

# Einzel-Submissions (Phase B)
if args.mode in ("final", "both") and df_test_raw is not None:
    ids = df_test_raw["id"].astype(str).tolist()
    for r in all_results:
        if "_test_probs" in r:
            preds = le.inverse_transform(r["_test_probs"].argmax(axis=1))
            write_submission(ids, preds, tag=r["model"])

    # Ensemble-Submission
    usable = [r for r in all_results if "_test_probs" in r and r.get("per_class_f1")]
    if args.ensemble and len(usable) >= 2:
        print(f"\n{'='*70}\n  ENSEMBLE (gewichtetes Soft-Voting, Gewichte = Trial-F1)\n{'='*70}")
        probs_list   = [r["_test_probs"] for r in usable]
        weights_list = [r["per_class_f1"] for r in usable]
        print(f"  Modelle: {[r['model'] for r in usable]}")
        ens_probs  = weighted_soft_voting(probs_list, weights_list)
        ens_preds  = le.inverse_transform(ens_probs.argmax(axis=1))
        pd.DataFrame({"id": ids, "prediction": ens_preds,
                      **{f"prob_{c}": ens_probs[:, i] for i, c in enumerate(CLASSES)}}
                     ).to_csv(OUT / "ensemble_predictions_test.csv", index=False)
        write_submission(ids, ens_preds, tag="ensemble")
    elif args.ensemble:
        print("  Ensemble übersprungen (brauche ≥2 Modelle mit Trial-F1 + Test-Probs).")

# ──────────────────────────────────────────────────────────────────────────────
# Zusammenfassung
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}\n  ZUSAMMENFASSUNG – Task {args.task.upper()}\n{'='*70}")
print(f"  {'Modell':<12} {'Trial-F1':>10} {'best_ep':>8}")
print(f"  {'─'*34}")
for r in all_results:
    print(f"  {r['model']:<12} {r.get('select_macro_f1', float('nan')):>10} "
          f"{r.get('best_epoch', '-'):>8}")

rows = [{k: v for k, v in r.items() if not k.startswith("_") and k != "per_class_f1"}
        for r in all_results]
res_path = OUT / "finalize_results.csv"
pd.DataFrame([{**row, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
             for row in rows]).to_csv(res_path, index=False)
print(f"\n  Ergebnisse: {res_path.resolve()}")
print(f"  Checkpoints/Predictions in: {OUT.resolve()}")
print(f"{'='*70}\n")
