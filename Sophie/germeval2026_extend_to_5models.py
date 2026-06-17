"""
GermEval 2026 – Extend 3-Model Runs to 5-Model Ensemble
=========================================================
Für jeden bereits abgeschlossenen 3-Modell-Run:
  1. Liest ensemble_config.json (alle exakten Hyperparameter + vorherige Ergebnisse)
  2. Lädt Checkpoints der 3 bestehenden Modelle (gbert, xlmr, deberta) → Val-Inferenz
  3. Trainiert mdeberta + gelectra neu (gleicher Seed → identischer Train/Val-Split)
  4. Berechnet 5-Modell Soft-Voting Ensemble
  5. Schreibt final_report_5.txt in den jeweiligen Run-Ordner

VERWENDUNG:
    # Alle 4 Runs erweitern:
    python germeval2026_extend_to_5models.py

    # Nur bestimmte Runs:
    python germeval2026_extend_to_5models.py --run_dirs model_dataset_gridsearch/all5_aug-none

    # Dry-run (nur auflisten, nicht trainieren):
    python germeval2026_extend_to_5models.py --dry_run

VORAUSSETZUNG:
    In jedem Run-Ordner müssen folgende Checkpoints existieren:
        model_gbert/best_model_weights.pt
        model_xlmr/best_model_weights.pt
        model_deberta/best_model_weights.pt
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
# Modell-Registry: Shortname → HuggingFace Model-ID
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "gbert":    "deepset/gbert-large",
    "xlmr":     "FacebookAI/xlm-roberta-large",
    "deberta":  "microsoft/deberta-v3-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "gelectra": "deepset/gelectra-large-germanquad",
}

EXISTING_MODELS = ["gbert", "xlmr", "deberta"]
NEW_MODELS      = ["mdeberta", "gelectra"]

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Extend 3-model runs to 5-model ensemble")
parser.add_argument("--base_dir",  default="model_dataset_gridsearch",
                    help="Basis-Ordner mit den Run-Unterordnern")
parser.add_argument("--run_dirs",  nargs="+", default=None,
                    help="Konkrete Run-Pfade (default: alle Unterordner in base_dir mit ensemble_config.json)")
parser.add_argument("--dry_run",   action="store_true",
                    help="Nur Runs auflisten, nicht trainieren")
parser.add_argument("--skip_done", action="store_true",
                    help="Runs ueberspringen wenn final_report_5.txt bereits existiert")
parser.add_argument("--reverse",   action="store_true",
                    help="Run-Reihenfolge umkehren (fuer zweite GPU, die von hinten anfaengt)")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Run-Ordner ermitteln
# ──────────────────────────────────────────────────────────────────────────────
if args.run_dirs:
    run_dirs = [Path(d) for d in args.run_dirs]
else:
    base = Path(args.base_dir)
    AUG_ORDER = ["paraphrase", "both", "none", "generated"]
    all_dirs  = [p for p in base.iterdir()
                 if p.is_dir() and (p / "ensemble_config.json").exists()]
    def aug_sort_key(p):
        for i, aug in enumerate(AUG_ORDER):
            if aug in p.name:
                return i
        return len(AUG_ORDER)
    run_dirs = sorted(all_dirs, key=aug_sort_key)
    if args.reverse:
        run_dirs = list(reversed(run_dirs))

if not run_dirs:
    print(f"ERROR: Keine Run-Ordner mit ensemble_config.json gefunden in '{args.base_dir}'")
    raise SystemExit(1)

print(f"\n{'='*65}")
print(f"  GermEval 2026 – Extend to 5-Model Ensemble")
print(f"{'='*65}")
print(f"  Gefundene Runs: {len(run_dirs)}")
for rd in run_dirs:
    done = (rd / "final_report_5.txt").exists()
    ckpts_ok = all((rd / f"model_{m}" / "best_model_weights.pt").exists()
                   for m in EXISTING_MODELS)
    status = "✓ 5-done" if done else ("ckpts OK" if ckpts_ok else "MISSING CKPTS")
    print(f"    {rd.name:<45} [{status}]")
print(f"{'='*65}\n")

if args.dry_run:
    print("  --dry_run: Kein Training gestartet.")
    raise SystemExit(0)

# ──────────────────────────────────────────────────────────────────────────────
# Modell-Architektur (identisch mit germeval2026_ensemble_train.py)
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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    for sep in [";", "\t"]:
        try:
            df = pd.read_csv(p, sep=sep, engine="python")
            if "text" in df.columns and "label" in df.columns:
                return df[["text", "label"]].dropna()
        except Exception:
            continue
    return pd.DataFrame(columns=["text", "label"])


def make_loaders(df_tr, df_vl, tokenizer, batch_size, max_length, le):
    train_ds = DBODataset(df_tr["text"].tolist(), le.transform(df_tr["label"]), tokenizer, max_length)
    val_ds   = DBODataset(df_vl["text"].tolist(), le.transform(df_vl["label"]), tokenizer, max_length)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )


def run_val_inference(model, val_loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs  = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def run_test_inference(model, texts, tokenizer, batch_size, max_length, device):
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i+batch_size], truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


# ──────────────────────────────────────────────────────────────────────────────
# Haupt-Loop über alle Runs
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}\n")

summary = []

for run_dir in run_dirs:
    run_name = run_dir.name
    print(f"\n{'='*65}")
    print(f"  RUN: {run_name}")
    print(f"{'='*65}")

    if args.skip_done and (run_dir / "final_report_5.txt").exists():
        print(f"  SKIP: final_report_5.txt existiert bereits.")
        continue

    # ── Checkpoints prüfen ────────────────────────────────────────────────────
    missing = [m for m in EXISTING_MODELS
               if not (run_dir / f"model_{m}" / "best_model_weights.pt").exists()]
    if missing:
        print(f"  ERROR: Checkpoints fehlen für: {missing}")
        print(f"  Überspringe diesen Run.")
        summary.append({"run": run_name, "status": "MISSING_CKPTS"})
        continue

    # ── Config laden ──────────────────────────────────────────────────────────
    with open(run_dir / "ensemble_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    seed       = cfg["seed"]
    max_length = cfg["max_length"]
    batch_size = cfg["batch_size"]
    lr         = cfg["lr"]
    dropout    = cfg["dropout"]
    val_size   = cfg["val_size"]
    augmentation = cfg["augmentation"]

    set_seed(seed)

    # ── Daten laden (identisch mit Original-Run) ───────────────────────────────
    df_main = load_csv(cfg["train_file"])
    df_test  = load_csv(cfg["test_file"])
    df_para  = load_csv(cfg["paraphrase_file"]) if augmentation in ("paraphrase", "both") else pd.DataFrame(columns=["text","label"])
    df_gen   = load_csv(cfg["generated_file"])  if augmentation in ("generated",  "both") else pd.DataFrame(columns=["text","label"])

    df_train_base, df_val = train_test_split(
        df_main, test_size=val_size, stratify=df_main["label"], random_state=seed
    )
    aug_parts = [p for p in [df_para, df_gen] if len(p) > 0]
    aug_extra = pd.concat(aug_parts, ignore_index=True) if aug_parts else pd.DataFrame(columns=["text","label"])
    df_train  = pd.concat([df_train_base, aug_extra], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    le = LabelEncoder()
    le.fit(df_main["label"])
    CLASSES   = list(le.classes_)
    N_CLASSES = len(CLASSES)

    base_counts     = pd.Series(le.transform(df_train_base["label"])).value_counts().sort_index()
    counts          = np.array([base_counts.get(i, 1) for i in range(N_CLASSES)], dtype=float)
    weights         = len(df_train_base) / (N_CLASSES * counts)
    weights_tensor  = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    print(f"  Augmentierung: {augmentation}")
    print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}")

    all_model_results = {}

    # ── Schritt 1: Bestehende 3 Modelle → Val-Inferenz ────────────────────────
    print(f"\n  [1/2] Lade bestehende Checkpoints und führe Val-Inferenz durch ...")
    for shortname in EXISTING_MODELS:
        model_id     = MODEL_REGISTRY[shortname]
        weights_path = run_dir / f"model_{shortname}" / "best_model_weights.pt"

        print(f"    {shortname}: lade {weights_path.name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        _, val_loader = make_loaders(df_train, df_val, tokenizer, batch_size, max_length, le)

        model = TransformerClassifier(model_id, N_CLASSES, dropout).to(DEVICE)
        state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.float() for k, v in state.items()})

        val_probs, val_preds, val_labels = run_val_inference(model, val_loader, DEVICE)
        macro_f1     = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(val_labels, val_preds, average=None,
                                zero_division=0, labels=list(range(N_CLASSES)))

        print(f"    {shortname}: Macro-F1 = {macro_f1:.5f}  "
              f"(orig: {cfg['individual_results'][shortname]['val_macro_f1']:.5f})")

        all_model_results[shortname] = {
            "shortname":         shortname,
            "model_id":          model_id,
            "best_val_macro_f1": round(macro_f1, 5),
            "per_class_f1":      {cls: round(float(f), 5) for cls, f in zip(CLASSES, per_class_f1)},
            "val_probs":         val_probs,
            "val_labels":        val_labels,
            "is_external":       False,
            "training_minutes":  0.0,
            "total_steps":       0,
            "early_stopped":     False,
        }
        del model
        torch.cuda.empty_cache()

    # ── Schritt 2: Neue 2 Modelle trainieren ──────────────────────────────────
    print(f"\n  [2/2] Trainiere neue Modelle: {NEW_MODELS} ...")

    for shortname in NEW_MODELS:
        model_id  = MODEL_REGISTRY[shortname]
        model_dir = run_dir / f"model_{shortname}"
        model_dir.mkdir(exist_ok=True)
        best_weights_path = model_dir / "best_model_weights.pt"

        print(f"\n  {'─'*60}")
        print(f"  Trainiere: {shortname} ({model_id})")
        print(f"  {'─'*60}")

        set_seed(seed)

        tokenizer    = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        train_loader, val_loader = make_loaders(df_train, df_val, tokenizer, batch_size, max_length, le)

        model       = TransformerClassifier(model_id, N_CLASSES, dropout).to(DEVICE)
        total_steps = len(train_loader) * cfg["max_epochs"]
        optimizer   = torch.optim.AdamW(model.parameters(), lr=lr,
                                         weight_decay=cfg["weight_decay"])
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, cfg["warmup_steps"], total_steps
        )
        criterion   = nn.CrossEntropyLoss(weight=weights_tensor)

        global_step = 0
        best_f1     = 0.0
        no_improve  = 0
        log_rows    = []
        early_stop  = False
        start_time  = time.time()

        model.train()
        for epoch in range(1, cfg["max_epochs"] + 1):
            for batch in train_loader:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["label"].to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(ids, mask), lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step % cfg["eval_every"] == 0:
                    val_p, val_pred, val_lbl = run_val_inference(model, val_loader, DEVICE)
                    macro_f1     = f1_score(val_lbl, val_pred, average="macro", zero_division=0)
                    per_class_f1 = f1_score(val_lbl, val_pred, average=None,
                                            zero_division=0, labels=list(range(N_CLASSES)))
                    if macro_f1 > best_f1:
                        best_f1    = macro_f1
                        no_improve = 0
                        torch.save({k: v.half().cpu() for k, v in model.state_dict().items()},
                                   best_weights_path)
                        flag = "OK Bestes Modell"
                    else:
                        no_improve += 1
                        flag = f"kein Fortschritt {no_improve}/{cfg['patience']}"

                    print(f"  [{shortname}] Step {global_step:>5} | Ep {epoch} | "
                          f"Macro-F1 {macro_f1:.4f} | {flag}")
                    log_rows.append({
                        "model": shortname, "step": global_step, "epoch": epoch,
                        "val_macro_f1": round(macro_f1, 5),
                        **{f"f1_{cls}": round(float(f), 5) for cls, f in zip(CLASSES, per_class_f1)},
                    })
                    if no_improve >= cfg["patience"]:
                        print(f"  [{shortname}] Early Stopping (beste Macro-F1: {best_f1:.4f})")
                        early_stop = True
                        model.train()
                        break
            if early_stop:
                break

        elapsed = time.time() - start_time

        state = torch.load(best_weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.float() for k, v in state.items()})

        val_probs, val_preds, val_labels = run_val_inference(model, val_loader, DEVICE)
        final_f1     = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        final_per_cls = f1_score(val_labels, val_preds, average=None,
                                  zero_division=0, labels=list(range(N_CLASSES)))
        final_report  = classification_report(val_labels, val_preds,
                                               target_names=CLASSES, zero_division=0)

        print(f"\n  [{shortname}] Finaler Report:")
        print(final_report)
        pd.DataFrame(log_rows).to_csv(model_dir / "training_log.csv", index=False)

        all_model_results[shortname] = {
            "shortname":         shortname,
            "model_id":          model_id,
            "best_val_macro_f1": round(final_f1, 5),
            "per_class_f1":      {cls: round(float(f), 5) for cls, f in zip(CLASSES, final_per_cls)},
            "val_probs":         val_probs,
            "val_labels":        val_labels,
            "is_external":       False,
            "training_minutes":  round(elapsed / 60, 1),
            "total_steps":       global_step,
            "early_stopped":     early_stop,
        }
        del model
        torch.cuda.empty_cache()

    # ── Schritt 3: 5-Modell Ensemble ──────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  5-MODELL ENSEMBLE SOFT-VOTING")
    print(f"  {'─'*60}")

    val_labels_arr  = list(all_model_results.values())[0]["val_labels"]
    ensemble_probs  = np.mean([r["val_probs"] for r in all_model_results.values()], axis=0)
    ensemble_preds  = ensemble_probs.argmax(axis=1)

    ensemble_macro_f1  = f1_score(val_labels_arr, ensemble_preds, average="macro", zero_division=0)
    ensemble_per_class = f1_score(val_labels_arr, ensemble_preds, average=None,
                                   zero_division=0, labels=list(range(N_CLASSES)))
    ensemble_report    = classification_report(val_labels_arr, ensemble_preds,
                                               target_names=CLASSES, zero_division=0)

    print(f"\n  Ensemble Macro-F1 (5 Modelle): {ensemble_macro_f1:.5f}")
    print(f"  Zum Vergleich 3-Modell (orig):  {cfg['ensemble_val_macro_f1']:.5f}")
    print(f"\n  Einzelmodelle:")
    for s, res in all_model_results.items():
        new_tag = " [NEU trainiert]" if s in NEW_MODELS else " [Checkpoint]"
        print(f"    {s:<12} Macro-F1: {res['best_val_macro_f1']:.5f}{new_tag}")
    print(f"\n{ensemble_report}")

    # ── Schritt 4: Testset-Vorhersagen ────────────────────────────────────────
    test_probs_all = []
    if len(df_test) > 0:
        print(f"  Testset-Vorhersagen mit allen 5 Modellen ...")
        for shortname, res in all_model_results.items():
            model_id = res["model_id"]
            if shortname in EXISTING_MODELS:
                weights_p = run_dir / f"model_{shortname}" / "best_model_weights.pt"
            else:
                weights_p = run_dir / f"model_{shortname}" / "best_model_weights.pt"

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            model_inf = TransformerClassifier(model_id, N_CLASSES, dropout).to(DEVICE)
            state     = torch.load(weights_p, map_location=DEVICE, weights_only=False)
            model_inf.load_state_dict({k: v.float() for k, v in state.items()})

            probs = run_test_inference(model_inf, df_test["text"].tolist(),
                                       tokenizer, batch_size, max_length, DEVICE)
            test_probs_all.append(probs)
            del model_inf
            torch.cuda.empty_cache()

        test_preds_idx    = np.mean(test_probs_all, axis=0).argmax(axis=1)
        test_preds_labels = le.inverse_transform(test_preds_idx)
        pred_df = pd.DataFrame({"text": df_test["text"].values,
                                 "prediction": test_preds_labels})
        if "label" in df_test.columns:
            pred_df["true_label"] = df_test["label"].values
        pred_df.to_csv(run_dir / "predictions_test_5models.csv", index=False)
        print(f"  Testvorhersagen (5 Modelle): {run_dir / 'predictions_test_5models.csv'}")

    # ── Schritt 5: final_report_5.txt schreiben ───────────────────────────────
    total_new_minutes = sum(
        all_model_results[m]["training_minutes"] for m in NEW_MODELS
    )
    sep = "─" * 65

    orig_3m = cfg["ensemble_val_macro_f1"]
    delta   = ensemble_macro_f1 - orig_3m

    report_text = f"""GermEval 2026 - Subtask 2: 5-Modell Ensemble Report
{sep}
Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M')}
Augmentierung:   {augmentation}
Modelle:         5 (3 bestehende Checkpoints + 2 neu trainiert)
  Bestehend:     {', '.join(EXISTING_MODELS)}
  Neu trainiert: {', '.join(NEW_MODELS)}
Train:           {len(df_train):,} (Basis: {len(df_train_base):,} + Aug: {len(df_train)-len(df_train_base):,})
Val:             {len(df_val):,}
Device:          {DEVICE}

Ergebnisse Einzelmodelle (Val-Set):
{chr(10).join(
    f"  {s:<12} Macro-F1: {r['best_val_macro_f1']:.5f}  "
    f"agitation: {r['per_class_f1'].get('agitation',0):.3f}  "
    f"subversive: {r['per_class_f1'].get('subversive',0):.3f}"
    + (" [NEU]" if s in NEW_MODELS else " [Checkpoint]")
    for s, r in all_model_results.items()
)}

Ensemble Ergebnis 5 Modelle (Val-Set):
  Macro-F1: {ensemble_macro_f1:.5f}
{chr(10).join(f"  {cls:<15} {f:.5f}" for cls, f in zip(CLASSES, ensemble_per_class))}

Vergleich 3-Modell vs. 5-Modell Ensemble:
  3-Modell Macro-F1: {orig_3m:.5f}
  5-Modell Macro-F1: {ensemble_macro_f1:.5f}
  Differenz:         {delta:+.5f}

{ensemble_report}
Trainingszeit (nur neue Modelle): {total_new_minutes:.1f} min
{sep}
"""

    (run_dir / "final_report_5.txt").write_text(report_text, encoding="utf-8")
    print(f"\n  final_report_5.txt gespeichert: {run_dir / 'final_report_5.txt'}")

    summary.append({
        "run":                  run_name,
        "status":               "OK",
        "aug":                  augmentation,
        "ensemble_3model_f1":   orig_3m,
        "ensemble_5model_f1":   round(ensemble_macro_f1, 5),
        "delta":                round(delta, 5),
        "new_model_minutes":    total_new_minutes,
        **{f"{s}_macro_f1": all_model_results[s]["best_val_macro_f1"]
           for s in all_model_results},
    })

# ──────────────────────────────────────────────────────────────────────────────
# Abschluss-Zusammenfassung
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  ABGESCHLOSSEN – Übersicht")
print(f"{'='*65}")
print(f"  {'Run':<40} {'3-Mod':>7} {'5-Mod':>7} {'Delta':>7}")
print(f"  {'─'*60}")
for row in summary:
    if row["status"] == "OK":
        print(f"  {row['run']:<40} {row['ensemble_3model_f1']:>7.5f} "
              f"{row['ensemble_5model_f1']:>7.5f} {row['delta']:>+7.5f}")
    else:
        print(f"  {row['run']:<40} [{row['status']}]")

ok_rows = [r for r in summary if r["status"] == "OK"]
if ok_rows:
    summary_df = pd.DataFrame(ok_rows)
    summary_path = Path(args.base_dir) / "gridsearch_5model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Zusammenfassung: {summary_path}")
