#!/usr/bin/env bash
# ============================================================================
#  GermEval 2026 - Finalize-Dispatcher (From-Scratch, 1 Task pro Rechner)
#  Aus dem Sophie/-Ordner:   ./run_finalize.sh c2a | def | vio
#  Optional:                 ./run_finalize.sh def TUMination 1
#
#  Training KOMPLETT VON GRUND AUF (--from_scratch, kein Warm-Start).
#  Jeder Run schreibt in ein EIGENES Verzeichnis: final_runs_scratch/<task>/
# ============================================================================
set -euo pipefail

RUN="${1:-}"
TEAM="${2:-TUMination}"
RUNNO="${3:-1}"
D="../GermEval2026/data"
OUTDIR="final_runs_scratch/${RUN}"

case "$RUN" in
  c2a)
    TRAIN="$D/c2a/c2a_train_26.csv"; TRIAL="$D/c2a/c2a_trial.csv"; TEST="$D/c2a/c2a_test_26.csv"
    AUG=""; MODELS="gbert"; ENSEMBLE=0 ;;
  def)
    TRAIN="$D/def/def_train.csv"; TRIAL="$D/def/def_trial.csv"; TEST="$D/def/def_test.csv"
    AUG="synthethic_data/def_aug.csv"; MODELS="gbert gelectra mdeberta"; ENSEMBLE=1 ;;
  vio)
    TRAIN="$D/vio/vio_train_26.csv"; TRIAL="$D/vio/vio_trial.csv"; TEST="$D/vio/vio_test_26.csv"
    AUG="synthethic_data/vio_aug.csv"; MODELS="gbert gelectra"; ENSEMBLE=1 ;;
  *)
    echo "Usage: $0 {c2a|def|vio} [team] [runNo]" >&2; exit 1 ;;
esac

echo "============================================================"
echo "  Run:        $RUN"
echo "  Output-Dir: $OUTDIR"
echo "  Modelle:    $MODELS   Ensemble: $ENSEMBLE   (FROM-SCRATCH)"
echo "============================================================"

ARGS=(germeval2026_finalize.py --task "$RUN"
      --train_file "$TRAIN" --trial_file "$TRIAL" --test_file "$TEST"
      --from_scratch --models $MODELS
      --out_dir "$OUTDIR" --mode both --team "$TEAM" --run "$RUNNO")

[ "$ENSEMBLE" -eq 1 ] && ARGS+=(--ensemble)

if [ -n "$AUG" ]; then
  if [ -f "$AUG" ]; then ARGS+=(--aug_file "$AUG"); echo "  Augmentierung: $AUG";
  else echo "  WARNUNG: $AUG fehlt - Training OHNE Augmentierung."; fi
fi

echo "  > python ${ARGS[*]}"
python "${ARGS[@]}"
echo "FERTIG ($RUN). Submission/Reports in $OUTDIR/"
