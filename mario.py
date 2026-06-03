"""
Fine-tune chrisrtt/gbert-multi-class-german-hate on our 4-class DBO task.

Original model: 5-class German hate speech (No Hate / Other / Political / Racist / Sexist)
Our classes   : nothing | criticism | agitation | subversive

Usage:
    # Single run with default hyperparameters:
    python mario.py

    # Random hyperparameter search (10 trials):
    python mario.py --search 10

    # Resume best config from a previous search and do a full run:
    python mario.py --config data/best_config.json

Strategy:
    Focal Loss (gamma tunable) to focus on hard minority examples
    WeightedRandomSampler so each batch has minority class representation
    Differential LRs: lower for encoder, higher for fresh classifier head
    Macro-F1 model selection + early stopping
    Random search over lr, focal_gamma, batch_size, weight_decay, warmup_frac
"""

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--search",  type=int,  default=0,
                    help="Number of random search trials (0 = single run with defaults)")
parser.add_argument("--config",  type=str,  default=None,
                    help="Path to JSON config from a previous search run")
parser.add_argument("--seed",    type=int,  default=42)
args = parser.parse_args()

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Fixed config ───────────────────────────────────────────────────────────────
MODEL_NAME = "chrisrtt/gbert-multi-class-german-hate"
DATA_PATH  = "data/processed/dbo_train_26.csv"
SAVE_PATH  = "data/gbert_dbo_finetuned.pt"
MAX_LEN    = 256

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID    = {"nothing": 0, "criticism": 1, "agitation": 2, "subversive": 3}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

# ── Default hyperparameters ────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "lr":           2e-5,
    "focal_gamma":  2.0,
    "batch_size":   32,
    "weight_decay": 0.01,
    "warmup_frac":  0.06,
}

# Search space — (distribution, low, high) or ("choice", [options])
SEARCH_SPACE = {
    "lr":           ("log_uniform", 5e-6, 5e-5),
    "focal_gamma":  ("uniform",     1.0,  3.0),
    "batch_size":   ("choice",      [16, 32, 64]),
    "weight_decay": ("log_uniform", 1e-3, 0.1),
    "warmup_frac":  ("uniform",     0.0,  0.12),
}

# Epochs per trial (short) vs. final full run
TRIAL_EPOCHS = 8
TRIAL_PATIENCE = 3
FULL_EPOCHS  = 30
FULL_PATIENCE = 4


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            label = row["label"].strip().lower()
            if label not in LABEL2ID:
                label = "nothing"
            texts.append(row["text"].strip())
            labels.append(LABEL2ID[label])
    return texts, labels


class DboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.labels = labels
        self.enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Helpers ────────────────────────────────────────────────────────────────────
def sample_config() -> dict:
    cfg = {}
    for key, spec in SEARCH_SPACE.items():
        if spec[0] == "choice":
            cfg[key] = random.choice(spec[1])
        elif spec[0] == "uniform":
            cfg[key] = random.uniform(spec[1], spec[2])
        elif spec[0] == "log_uniform":
            cfg[key] = math.exp(random.uniform(math.log(spec[1]), math.log(spec[2])))
    return cfg


def make_loaders(train_ds, val_ds, test_ds, batch_size, class_weights_cpu):
    sample_weights = [class_weights_cpu[y] for y in train_ds.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=4),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=4),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,    num_workers=4),
    )


def build_model_and_optimizer(cfg):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    no_decay       = {"bias", "LayerNorm.weight"}
    classifier_kws = {"classifier", "pooler"}

    def is_cls(name):
        return any(k in name for k in classifier_kws)

    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not is_cls(n) and not any(nd in n for nd in no_decay)],
         "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if not is_cls(n) and any(nd in n for nd in no_decay)],
         "lr": cfg["lr"], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if is_cls(n)],
         "lr": cfg["lr"] * 10, "weight_decay": cfg["weight_decay"]},
    ]
    optimizer = torch.optim.AdamW(groups)
    return model, optimizer


def train_epoch(model, loader, optimizer, scheduler, criterion, epoch, num_epochs,
                show_progress=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f"  Ep {epoch:02d}/{num_epochs} [train]",
               unit="batch", leave=False, ncols=90, disable=not show_progress)
    for batch in bar:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = out.logits.argmax(dim=1)
        total_loss += loss.item() * len(lbls)
        correct    += (preds == lbls).sum().item()
        total      += len(lbls)
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        out  = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, lbls)
        preds = out.logits.argmax(dim=1)
        total_loss += loss.item() * len(lbls)
        correct    += (preds == lbls).sum().item()
        total      += len(lbls)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1, all_preds, all_labels


# ── Core training loop ─────────────────────────────────────────────────────────
def run(cfg, train_ds, val_ds, test_ds, class_weights, max_epochs, patience,
        verbose=True, label=""):
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, cfg["batch_size"],
        class_weights_cpu=[class_weights[i].item() for i in range(NUM_CLASSES)],
    )
    model, optimizer = build_model_and_optimizer(cfg)
    criterion = FocalLoss(alpha=class_weights, gamma=cfg["focal_gamma"])

    total_steps  = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * cfg["warmup_frac"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    if verbose:
        print(f"\n{'─'*75}")
        print(f"  {'Ep':>3}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
              f"{'Vl Loss':>8}  {'Vl Acc':>7}  {'Macro-F1':>9}  {'Time':>6}")
        print(f"{'─'*75}")

    best_f1, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, epoch, max_epochs,
            show_progress=verbose)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)
        ep_secs = time.time() - t0

        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker     = " ★"
        else:
            no_improve += 1
            marker      = f" ({no_improve}/{patience})"

        if verbose:
            print(f"  {epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}"
                  f"  {vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>9.4f}"
                  f"  {ep_secs:>5.1f}s{marker}")
        else:
            print(f"    ep {epoch}/{max_epochs}  val_f1={vl_f1:.4f}  "
                  f"best={best_f1:.4f}  {ep_secs:.0f}s{marker}", flush=True)

        if no_improve >= patience:
            if verbose:
                print(f"\n  ⏹  Early stop — best macro-F1: {best_f1:.4f}")
            break

    model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, preds, gold = evaluate(model, test_loader, criterion)
    return model, best_state, best_f1, te_f1, preds, gold


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*70}")
    print(f"  Fine-tuning: {MODEL_NAME}")
    print(f"  Device: {DEVICE}  |  max_len: {MAX_LEN}")
    if args.search > 0:
        print(f"  Mode: random search  ({args.search} trials × {TRIAL_EPOCHS} epochs)")
    else:
        print(f"  Mode: single run  (up to {FULL_EPOCHS} epochs)")
    print(f"{'='*70}\n")

    # 1. Load & split data
    print("⏳  Loading data …")
    texts, labels = load_data(DATA_PATH)
    dist = {ID2LABEL[k]: v for k, v in sorted(Counter(labels).items())}
    print(f"    {len(texts):,} rows | {dist}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr)
    print(f"    train: {len(y_tr):,}  |  val: {len(y_vl):,}  |  test: {len(y_te):,}\n")

    # 2. Tokenise once — reused across all trials
    print("⏳  Loading tokenizer & tokenising …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    train_ds = DboDataset(X_tr, y_tr, tokenizer)
    val_ds   = DboDataset(X_vl, y_vl, tokenizer)
    test_ds  = DboDataset(X_te, y_te, tokenizer)
    print(f"    Done in {time.time() - t0:.1f}s\n")

    # 3. Class weights (shared across all trials)
    label_counts  = Counter(y_tr)
    total_tr      = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_tr / (NUM_CLASSES * label_counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float, device=DEVICE,
    )
    cw = {ID2LABEL[i]: round(class_weights[i].item(), 2) for i in range(NUM_CLASSES)}
    print(f"    Class weights: {cw}\n")

    # ── Random search ──────────────────────────────────────────────────────────
    if args.search > 0:
        n_trials = args.search
        print(f"{'═'*70}")
        print(f"  RANDOM SEARCH  ({n_trials} trials, up to {TRIAL_EPOCHS} epochs each)")
        print(f"{'═'*70}")
        print(f"  {'#':>3}  {'lr':>8}  {'gamma':>6}  {'bs':>4}  {'wd':>7}  "
              f"{'wf':>6}  {'Val F1':>8}  {'Time':>6}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*6}  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*6}")

        trial_results = []
        for i in range(1, n_trials + 1):
            cfg = sample_config()
            print(f"\n  Trial {i}/{n_trials}  lr={cfg['lr']:.2e}  gamma={cfg['focal_gamma']:.2f}"
                  f"  bs={cfg['batch_size']}  wd={cfg['weight_decay']:.4f}"
                  f"  wf={cfg['warmup_frac']:.3f}", flush=True)
            t0 = time.time()
            _, _, val_f1, _, _, _ = run(
                cfg, train_ds, val_ds, test_ds, class_weights,
                max_epochs=TRIAL_EPOCHS, patience=TRIAL_PATIENCE,
                verbose=False,
            )
            elapsed_t = time.time() - t0
            trial_results.append((val_f1, cfg))
            print(f"  → val macro-F1: {val_f1:.4f}  ({elapsed_t:.0f}s)")

        trial_results.sort(key=lambda x: x[0], reverse=True)
        best_val_f1, best_cfg = trial_results[0]

        print(f"\n  Best trial val macro-F1: {best_val_f1:.4f}")
        print(f"  Best config: {best_cfg}\n")

        Path("data").mkdir(parents=True, exist_ok=True)
        with open("data/best_config.json", "w") as f:
            json.dump(best_cfg, f, indent=2)
        print("  Saved → data/best_config.json\n")

        print(f"  Running full training with best config (up to {FULL_EPOCHS} epochs) …")
        final_cfg = best_cfg

    elif args.config:
        with open(args.config) as f:
            final_cfg = json.load(f)
        print(f"  Loaded config from {args.config}: {final_cfg}\n")
        final_cfg = {**DEFAULT_CONFIG, **final_cfg}
    else:
        final_cfg = DEFAULT_CONFIG

    # ── Full run with chosen config ────────────────────────────────────────────
    print(f"  Config: {final_cfg}")
    t_start = time.time()
    model, best_state, best_val_f1, te_f1, preds, gold = run(
        final_cfg, train_ds, val_ds, test_ds, class_weights,
        max_epochs=FULL_EPOCHS, patience=FULL_PATIENCE,
        verbose=True,
    )
    elapsed = time.time() - t_start

    target_names = [ID2LABEL[i] for i in range(NUM_CLASSES)]
    print(f"\n{'='*70}")
    print(f"  Test Macro-F1: {te_f1:.4f}  |  Training time: {elapsed:.1f}s")
    print(f"{'='*70}\n")
    print(classification_report(gold, preds, target_names=target_names, digits=3))

    print("Confusion matrix  (rows = true, cols = predicted):")
    cm    = confusion_matrix(gold, preds)
    col_w = 11
    print("            " + "".join(f"{n:>{col_w}}" for n in target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>12}" + "".join(f"{v:>{col_w}}" for v in row))

    torch.save({
        "model_state":   best_state,
        "label2id":      LABEL2ID,
        "id2label":      ID2LABEL,
        "model_name":    MODEL_NAME,
        "max_len":       MAX_LEN,
        "config":        final_cfg,
        "best_val_f1":   best_val_f1,
        "test_macro_f1": te_f1,
    }, SAVE_PATH)
    print(f"\n✅  Model saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
