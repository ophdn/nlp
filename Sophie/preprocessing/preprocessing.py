"""
GermEval 2026 – Subtask 2: DBO Classification
==============================================
Preprocessing Pipeline

Erzeugt zwei Varianten der Daten:
  *_raw.csv      → kein Preprocessing (wie im Baseline-Paper)
  *_minimal.csv  → URLs → [URL], @Mentions → [USER], Whitespace normalisiert

VERWENDUNG:
    python germeval2026_preprocessing.py \\
        --train pfad/zu/train.csv \\
        --test  pfad/zu/test.csv  \\
        --out   data/processed

AUSGABE (im --out Ordner):
    train_raw.csv
    train_minimal.csv
    test_raw.csv        (falls --test angegeben)
    test_minimal.csv    (falls --test angegeben)
    preprocessing_report.txt
"""

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

# ──────────────────────────────────────────────
# 1. CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True, help="Pfad zur Trainingsdatei (CSV/TSV)")
parser.add_argument("--test",  default=None,  help="Pfad zur Testdatei (optional)")
parser.add_argument("--out",   default="data/processed", help="Ausgabeordner")
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 2. Laden (identisch zur Analyse)
# ──────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="warn")

    text_candidates = [c for c in df.columns
                       if c.lower() in ("text", "tweet", "comment", "sentence", "description")]
    text_col = text_candidates[0] if text_candidates else \
        max(df.select_dtypes("object").columns,
            key=lambda c: df[c].astype(str).str.len().mean())

    label_candidates = [c for c in df.columns
                        if c.lower() in ("label", "class", "dbo", "c2a", "def", "vio", "subtask2", "subtask3", "category")]
    label_col = label_candidates[0] if label_candidates else \
        [c for c in df.columns if c != text_col][0]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    print(f"  ✓ {len(df):,} Zeilen  (text='{text_col}', label='{label_col}')")
    return df


# ──────────────────────────────────────────────
# 3. Preprocessing-Varianten
# ──────────────────────────────────────────────

# Kompilierte Regex-Patterns
_URL_RE      = re.compile(r'https?://\S+|www\.\S+')
_MENTION_RE  = re.compile(r'@\w+')
_WHITESPACE  = re.compile(r'[ \t]+')          # mehrfache Leerzeichen/Tabs
_NEWLINE     = re.compile(r'\n+')             # mehrfache Newlines


def preprocess_zero(text: str) -> str:
    """Kein Preprocessing – identisch zum Baseline-Paper."""
    return text


def preprocess_minimal(text: str) -> str:
    """
    Minimal-Preprocessing:
      1. URLs        → [URL]
      2. @Mentions   → [USER]
      3. Whitespace  → normalisieren (kein Strip von Sonderzeichen, Emojis etc.)
    """
    text = _URL_RE.sub("[URL]", text)
    text = _MENTION_RE.sub("[USER]", text)
    text = _NEWLINE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    text = text.strip()
    return text


VARIANTS = {
    "raw":     preprocess_zero,
    "minimal": preprocess_minimal,
}

# ──────────────────────────────────────────────
# 4. Pipeline ausführen & speichern
# ──────────────────────────────────────────────
def process_and_save(df: pd.DataFrame, split: str) -> dict:
    """
    Verarbeitet einen Split (train/test) mit allen Varianten.
    Gibt Statistiken für den Report zurück.
    """
    stats = {}
    for variant_name, fn in VARIANTS.items():
        out_df = df.copy()
        out_df["text"] = out_df["text"].apply(fn)

        # Dateiname: z.B. train_minimal.csv
        out_path = OUT / f"{split}_{variant_name}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8")

        # Statistiken sammeln
        n_url     = df["text"].str.contains(r'https?://', regex=True).sum()
        n_mention = df["text"].str.contains(r'@\w+',     regex=True).sum()
        stats[variant_name] = {
            "path":       str(out_path),
            "rows":       len(out_df),
            "urls_found": int(n_url)     if variant_name == "minimal" else "-",
            "mentions_found": int(n_mention) if variant_name == "minimal" else "-",
            "avg_len_before": df["text"].str.len().mean(),
            "avg_len_after":  out_df["text"].str.len().mean(),
        }
        print(f"  ✓ {out_path.name:<25}  ({len(out_df):,} Zeilen)")

    return stats


print("\n📂 Lade Daten ...")
splits = {"train": load_data(args.train)}
if args.test:
    splits["test"] = load_data(args.test)

all_stats = {}
for split_name, df in splits.items():
    print(f"\n🔧 Verarbeite '{split_name}' ...")
    all_stats[split_name] = process_and_save(df, split_name)

# ──────────────────────────────────────────────
# 5. Report
# ──────────────────────────────────────────────
report_lines = [
    "GermEval 2026 – Subtask 2: Preprocessing Report",
    "=" * 50,
    "",
    "Varianten:",
    "  raw     → kein Preprocessing (Baseline-Paper-Standard)",
    "  minimal → URLs→[URL], @Mentions→[USER], Whitespace normalisiert",
    "",
    "Dateien:",
]

for split_name, split_stats in all_stats.items():
    report_lines.append(f"\n  [{split_name}]")
    for variant, s in split_stats.items():
        report_lines.append(f"    {variant:<10} → {s['path']}")
        if s["urls_found"] != "-":
            delta = s["avg_len_before"] - s["avg_len_after"]
            report_lines.append(
                f"               URLs ersetzt: {s['urls_found']}  |  "
                f"Mentions ersetzt: {s['mentions_found']}  |  "
                f"Ø Längendelta: -{delta:.1f} Zeichen"
            )

report_lines += [
    "",
    "Verwendung in nachfolgenden Scripts:",
    "  from pathlib import Path",
    "  import pandas as pd",
    "",
    f"  DATA = Path('{OUT}')",
    "  train = pd.read_csv(DATA / 'train_minimal.csv')  # empfohlen",
    "  # oder:",
    "  train = pd.read_csv(DATA / 'train_raw.csv')      # zero preprocessing",
]

report_path = OUT / "preprocessing_report.txt"
report_text = "\n".join(report_lines)
report_path.write_text(report_text, encoding="utf-8")

print(f"\n📋 Report gespeichert: {report_path}")
print(f"\n{'─'*50}")
print(report_text)
print(f"{'─'*50}")
print(f"\n✅ Fertig. Alle Dateien in: {OUT.resolve()}\n")