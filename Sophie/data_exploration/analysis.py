"""
GermEval 2026 – Subtask 3: Violence Detection (Fine-grained)
============================================================
Phase 1 Script: Setup-Check & Datenanalyse

VERWENDUNG:
    python germeval2026_subtask3_analysis.py --train path/to/train.csv

Das Script erwartet eine CSV/TSV mit mindestens:
  - einer Text-Spalte  (Auto-Detect: 'text', 'tweet', 'comment')
  - einer Label-Spalte (Auto-Detect: 'label', 'class', 'subtask3')

Separator wird automatisch erkannt (Komma oder Tab).
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────
# 1. CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GermEval 2026 Subtask 3 – Datenanalyse")
parser.add_argument("--train", type=str, required=True, help="Pfad zur Trainingsdatei (CSV/TSV)")
parser.add_argument("--test",  type=str, default=None,  help="Pfad zur Testdatei (optional)")
parser.add_argument("--out",   type=str, default="analysis_output", help="Ausgabeverzeichnis für Plots und Zusammenfassung")
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 2. Daten laden (auto-detect separator + columns)
# ──────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="warn")

    # Auto-detect text column
    text_candidates = [c for c in df.columns if c.lower() in ("text", "tweet", "comment", "sentence")]
    if not text_candidates:
        # Fallback: longest average string column
        text_col = max(df.select_dtypes("object").columns,
                       key=lambda c: df[c].astype(str).str.len().mean())
    else:
        text_col = text_candidates[0]

    # Auto-detect label column
    label_candidates = [c for c in df.columns
                        if c.lower() in ("label", "class", "subtask3", "category", "vio", "violence",
                                         "dbo", "c2a", "def")]
    if not label_candidates:
        label_col = [c for c in df.columns if c != text_col][0]
    else:
        label_col = label_candidates[0]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    print(f"  ✓ Geladen: {len(df):,} Zeilen  |  Spalten erkannt: text='{text_col}', label='{label_col}'")
    return df


print("\n📂 Lade Daten ...")
train = load_data(args.train)
test  = load_data(args.test) if args.test else None


# ──────────────────────────────────────────────
# 3. Basis-Statistiken
# ──────────────────────────────────────────────
def basic_stats(df: pd.DataFrame, name: str):
    print(f"\n{'─'*50}")
    print(f"  {name.upper()}: {len(df):,} Samples")
    print(f"{'─'*50}")

    label_counts = df["label"].value_counts()
    print("\n  Klassenverteilung:")
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {str(label):<20} {count:>5}  ({pct:5.1f}%)  {bar}")

    # Imbalance ratio (majority / minority)
    imbalance = label_counts.max() / label_counts.min()
    print(f"\n  Imbalance-Ratio (max/min):  {imbalance:.1f}x  ", end="")
    if imbalance > 10:
        print("⚠️  KRITISCH – class weighting dringend erforderlich")
    elif imbalance > 3:
        print("⚠️  Moderat – class weighting empfohlen")
    else:
        print("✓  Akzeptabel")

    # Text lengths
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().str.len()
    df["token_approx"] = (df["char_len"] / 4).astype(int)  # rough estimate

    print(f"\n  Textlänge (Zeichen):  min={df['char_len'].min()}  "
          f"median={df['char_len'].median():.0f}  "
          f"max={df['char_len'].max()}  "
          f"mean={df['char_len'].mean():.0f}")
    print(f"  Wortanzahl:           min={df['word_len'].min()}  "
          f"median={df['word_len'].median():.0f}  "
          f"max={df['word_len'].max()}  "
          f"mean={df['word_len'].mean():.0f}")

    # Tweets > 512 tokens (BERT limit)
    long = (df["token_approx"] > 512).sum()
    if long > 0:
        print(f"\n  ⚠️  {long} Tweets überschreiten ~512 Tokens (BERT-Limit)")
    else:
        print(f"\n  ✓ Alle Tweets innerhalb des BERT-Token-Limits (~512)")

    return df


train = basic_stats(train, "Training")
if test is not None:
    test = basic_stats(test, "Test")


# ──────────────────────────────────────────────
# 4. Besondere Merkmale von Social-Media-Text
# ──────────────────────────────────────────────
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF\u2700-\u27BF]")

def social_media_features(df: pd.DataFrame) -> pd.DataFrame:
    df["has_emoji"]    = df["text"].apply(lambda t: bool(EMOJI_RE.search(str(t))))
    df["has_hashtag"]  = df["text"].str.contains(r'#\w+', regex=True)
    df["has_mention"]  = df["text"].str.contains(r'@\w+', regex=True)
    df["has_url"]      = df["text"].str.contains(r'https?://', regex=True)
    df["has_caps"]     = df["text"].str.contains(r'[A-ZÄÖÜ]{3,}', regex=True)
    df["has_punct_rep"] = df["text"].str.contains(r'[!?]{2,}', regex=True)
    return df

train = social_media_features(train)
if test is not None:
    test = social_media_features(test)

features = ["has_emoji", "has_hashtag", "has_mention", "has_url", "has_caps", "has_punct_rep"]
print("\n  Social-Media-Features (% der Tweets pro Klasse):")
feat_df = train.groupby("label")[features].mean().mul(100).round(1)
print(feat_df.to_string())


# ──────────────────────────────────────────────
# 5. Missing / Duplicates
# ──────────────────────────────────────────────
print("\n  Qualitätsprüfung:")
print(f"    Leere Texte:      {train['text'].isna().sum() + (train['text']=='').sum()}")
print(f"    Exakte Duplikate (Text+Label): {train.duplicated(['text','label']).sum()}")
dup_diff_label = train[train.duplicated('text', keep=False)].groupby('text')['label'].nunique()
conf = (dup_diff_label > 1).sum()
if conf > 0:
    print(f"    ⚠️  Gleicher Text, verschiedene Labels: {conf} Fälle")
else:
    print(f"    ✓ Keine widersprüchlichen Annotationen")


# ──────────────────────────────────────────────
# 6. Visualisierungen
# ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("GermEval 2026 – Subtask 3: Trainingsdaten-Analyse", fontsize=15, fontweight="bold")

label_counts = train["label"].value_counts()
colors = sns.color_palette("Set2", len(label_counts))

# Plot 1: Klassenverteilung (Balken)
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.barh(label_counts.index.astype(str), label_counts.values, color=colors)
ax1.set_xlabel("Anzahl Samples")
ax1.set_title("Klassenverteilung")
for bar, val in zip(bars, label_counts.values):
    ax1.text(bar.get_width() + label_counts.max() * 0.01, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=9)

# Plot 2: Pie-Chart
ax2 = fig.add_subplot(gs[0, 2])
ax2.pie(label_counts.values, labels=label_counts.index.astype(str),
        autopct="%1.1f%%", colors=colors, startangle=90, textprops={"fontsize": 8})
ax2.set_title("Anteil (%)")

# Plot 3: Textlänge nach Klasse (Boxplot)
ax3 = fig.add_subplot(gs[1, :2])
class_order = label_counts.index.astype(str).tolist()
sns.boxplot(data=train, x="label", y="char_len", order=class_order,
            palette="Set2", ax=ax3, showfliers=False)
ax3.set_xlabel("Label")
ax3.set_ylabel("Zeichen")
ax3.set_title("Textlänge pro Klasse (Zeichen, ohne Ausreißer)")
ax3.tick_params(axis="x", rotation=20)

# Plot 4: Textlängen-Histogramm gesamt
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(train["char_len"], bins=40, color="#5B9BD5", edgecolor="white")
ax4.axvline(train["char_len"].median(), color="red", linestyle="--", label=f"Median={train['char_len'].median():.0f}")
ax4.set_xlabel("Zeichen")
ax4.set_ylabel("Häufigkeit")
ax4.set_title("Textlängen-Verteilung")
ax4.legend(fontsize=8)

# Plot 5: Social-Media-Features Heatmap
ax5 = fig.add_subplot(gs[2, :2])
feat_df_plot = train.groupby("label")[features].mean().mul(100)
feat_df_plot.columns = [f.replace("has_", "") for f in features]
sns.heatmap(feat_df_plot, annot=True, fmt=".0f", cmap="YlOrRd",
            ax=ax5, cbar_kws={"label": "%"}, linewidths=0.5)
ax5.set_title("Social-Media-Features pro Klasse (%)")
ax5.set_ylabel("")

# Plot 6: Wortanzahl-Verteilung pro Klasse
ax6 = fig.add_subplot(gs[2, 2])
for i, (label, group) in enumerate(train.groupby("label")):
    ax6.hist(group["word_len"], bins=20, alpha=0.5,
             label=str(label), color=colors[i % len(colors)])
ax6.set_xlabel("Wörter")
ax6.set_ylabel("Häufigkeit")
ax6.set_title("Wortanzahl pro Klasse")
ax6.legend(fontsize=7)

out_plot = OUT / "subtask3_data_analysis.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
print(f"\n  📊 Plot gespeichert: {out_plot}")
plt.close()


# ──────────────────────────────────────────────
# 7. Beispiel-Tweets pro Klasse
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  BEISPIEL-TWEETS PRO KLASSE (je 3)")
print("="*60)
for label in train["label"].unique():
    subset = train[train["label"] == label]
    print(f"\n  [{label}]  (n={len(subset):,})")
    for _, row in subset.sample(min(3, len(subset)), random_state=42).iterrows():
        preview = row["text"][:120].replace("\n", " ")
        print(f"    › {preview}")


# ──────────────────────────────────────────────
# 8. Zusammenfassung & Empfehlungen
# ──────────────────────────────────────────────
label_counts = train["label"].value_counts()
imbalance = label_counts.max() / label_counts.min()

print("\n" + "="*60)
print("  ZUSAMMENFASSUNG & EMPFEHLUNGEN FÜR MODELLIERUNG")
print("="*60)

print(f"\n  Anzahl Klassen:          {train['label'].nunique()}")
print(f"  Trainingssamples:        {len(train):,}")
print(f"  Imbalance-Ratio:         {imbalance:.1f}x")
print(f"  Median Textlänge:        {train['char_len'].median():.0f} Zeichen")
print(f"  Emoji-Anteil:            {train['has_emoji'].mean()*100:.1f}%")

print("\n  Modellierungs-Empfehlungen:")
if imbalance > 5:
    print("  ✦ Class-weighted Cross-Entropy verwenden (kritische Imbalance)")
    print("  ✦ Oversampling der Minority-Klassen erwägen")
else:
    print("  ✦ Class-weighted Cross-Entropy trotzdem einsetzen")

if train["has_emoji"].mean() > 0.1:
    print("  ✦ Emoji-fähiger Tokenizer bevorzugen → XLM-RoBERTa oder Qwen3")

avg_len = train["char_len"].mean()
if avg_len < 200:
    print(f"  ✦ Kurze Texte (Ø {avg_len:.0f} Zeichen) → max_length=128 ausreichend, spart Speicher")
else:
    print(f"  ✦ Längere Texte (Ø {avg_len:.0f} Zeichen) → max_length=256 empfohlen")

print("\n  Nächste Schritte (Phase 1 Todos 3–5):")
print("  → Klassenimbalance quantifizieren (erledigt ↑)")
print("  → Baseline-Skript der Organizer laufen lassen")
print("  → Lokales Eval-Framework (Macro-F1, Confusion Matrix) aufsetzen")
print("  → Cross-Validation-Setup implementieren\n")

# Save summary CSV
summary = pd.DataFrame({
    "label": label_counts.index,
    "count": label_counts.values,
    "pct": (label_counts.values / len(train) * 100).round(2),
})
summary.to_csv(OUT / "class_distribution.csv", index=False)
print(f"  💾 Klassenverteilung gespeichert: {OUT / 'class_distribution.csv'}\n")