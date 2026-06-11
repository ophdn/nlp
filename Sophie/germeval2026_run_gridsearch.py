"""
GermEval 2026 – Grid-Search Runner
====================================
Führt alle in gridsearch_config.json definierten Runs sequenziell aus.
Jeder Run entspricht einem Aufruf von germeval2026_ensemble_train.py
mit den dort definierten Argumenten.

VERWENDUNG:
    python germeval2026_run_gridsearch.py
    python germeval2026_run_gridsearch.py --config gridsearch_config.json
    python germeval2026_run_gridsearch.py --config gridsearch_config.json --dry_run
    python germeval2026_run_gridsearch.py --config gridsearch_config.json --only_runs 0 2 5

OPTIONEN:
    --config       Pfad zur JSON-Config (default: gridsearch_config.json)
    --dry_run      Nur Runs auflisten, nichts ausführen
    --only_runs    Nur bestimmte Run-Indizes ausführen (0-basiert)
    --skip_done    Runs überspringen, für die final_report.txt bereits existiert
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 Grid-Search Runner")
parser.add_argument("--config",     default="gridsearch_config.json")
parser.add_argument("--dry_run",    action="store_true",
                    help="Nur Runs auflisten, nicht ausfuehren")
parser.add_argument("--only_runs",  nargs="+", type=int, default=None,
                    help="Nur diese Run-Indizes ausfuehren (0-basiert)")
parser.add_argument("--skip_done",  action="store_true",
                    help="Runs ueberspringen wenn final_report.txt existiert")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Config laden
# ──────────────────────────────────────────────────────────────────────────────
config_path = Path(args.config)
if not config_path.exists():
    print(f"ERROR: Config nicht gefunden: {config_path}")
    sys.exit(1)

with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

runs        = config["runs"]
base_script = config.get("base_script", "germeval2026_ensemble_train.py")
base_dir    = config.get("base_out_dir", "model_dataset_gridsearch")
results_csv = config.get("results_csv",  "results.csv")
global_args = config.get("global_args",  {})

print(f"\n{'='*65}")
print(f"  GermEval 2026 – Grid-Search Runner")
print(f"{'='*65}")
print(f"  Config:      {config_path.resolve()}")
print(f"  Script:      {base_script}")
print(f"  Basis-Dir:   {base_dir}")
print(f"  Gesamt Runs: {len(runs)}")
print(f"  Global Args: {global_args}")
print(f"{'='*65}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Runs auflisten
# ──────────────────────────────────────────────────────────────────────────────
for i, run in enumerate(runs):
    run_name  = run["name"]
    out_dir   = f"{base_dir}/{run_name}"
    done_flag = Path(out_dir) / "final_report.txt"
    status    = "✓ done" if done_flag.exists() else "pending"
    models    = " ".join(run.get("models", global_args.get("models", ["gbert","xlmr","deberta","mdeberta","gelectra"])))
    aug       = run.get("augmentation", global_args.get("augmentation", "none"))
    print(f"  [{i:>2}] {run_name:<45} aug={aug:<11} models=[{models}]  {status}")

print()

if args.dry_run:
    print("  --dry_run: Kein Training gestartet.")
    sys.exit(0)

# ──────────────────────────────────────────────────────────────────────────────
# Runs ausführen
# ──────────────────────────────────────────────────────────────────────────────
run_indices = args.only_runs if args.only_runs is not None else list(range(len(runs)))

summary_rows = []
total_start  = time.time()

for i in run_indices:
    if i >= len(runs):
        print(f"  WARNING: Run-Index {i} existiert nicht (max {len(runs)-1}). Uebersprungen.")
        continue

    run      = runs[i]
    run_name = run["name"]
    out_dir  = f"{base_dir}/{run_name}"

    if args.skip_done and (Path(out_dir) / "final_report.txt").exists():
        print(f"\n  [{i:>2}] SKIP (bereits done): {run_name}")
        continue

    # Argumente zusammenbauen: global_args → run-spezifische Args
    merged = {**global_args, **run}
    merged.pop("name", None)           # kein CLI-Argument
    merged.pop("notes", None)          # nur zur Dokumentation
    merged["out_dir"]     = out_dir
    merged["results_csv"] = results_csv

    # models-Liste → space-separated strings
    if "models" in merged and isinstance(merged["models"], list):
        models_str = merged.pop("models")
    else:
        models_str = merged.pop("models", ["gbert", "xlmr", "deberta", "mdeberta", "gelectra"])

    # CLI-Kommando aufbauen
    cmd = [sys.executable, "-u", base_script]
    cmd += ["--models"] + models_str
    for k, v in merged.items():
        if isinstance(v, list):
            cmd += [f"--{k}"] + [str(x) for x in v]
        elif isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]

    print(f"\n{'='*65}")
    print(f"  [{i:>2}/{len(run_indices)-1}]  START: {run_name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "stdout.log"

    run_start = time.time()
    ok        = True

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with open(log_path, "w", encoding="utf-8") as log_f:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)
        proc.wait()
        if proc.returncode != 0:
            print(f"\n  ERROR: Run {run_name} beendet mit Code {proc.returncode}")
            ok = False
    except KeyboardInterrupt:
        print(f"\n  ABBRUCH durch Benutzer bei Run {run_name}")
        proc.kill()
        sys.exit(1)
    except Exception as e:
        print(f"\n  FEHLER bei Run {run_name}: {e}")
        ok = False

    elapsed = time.time() - run_start
    summary_rows.append({
        "run_index":    i,
        "run_name":     run_name,
        "status":       "OK" if ok else "ERROR",
        "minutes":      round(elapsed / 60, 1),
        "out_dir":      out_dir,
        "log":          str(log_path),
        "finished_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    print(f"\n  [{i:>2}] {'OK' if ok else 'ERROR'}: {run_name}  ({elapsed/60:.1f} min)")

# ──────────────────────────────────────────────────────────────────────────────
# Abschluss-Zusammenfassung
# ──────────────────────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start

print(f"\n{'='*65}")
print(f"  GRID-SEARCH ABGESCHLOSSEN")
print(f"{'='*65}")
print(f"  Gesamt: {len(summary_rows)} Runs  |  {total_elapsed/60:.1f} min")
print(f"\n  {'Run':<3} {'Status':<8} {'Minuten':>8}  Name")
print(f"  {'─'*60}")
for row in summary_rows:
    print(f"  [{row['run_index']:>2}] {row['status']:<8} {row['minutes']:>7.1f}  {row['run_name']}")

summary_path = Path(base_dir) / "gridsearch_summary.csv"
import pandas as pd
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print(f"\n  Zusammenfassung gespeichert: {summary_path}")