#!/usr/bin/env bash
# Reproduce all paper experiments with a single command.
# Produces JSON results + LaTeX tables in results/ directory.
#
# Usage:
#   uv run bash scripts/run_all_experiments.sh
#   uv run bash scripts/run_all_experiments.sh --quick   # n=30 for fast iteration
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results}"
QUICK=0
N_SAMPLES=100
N_EVENTS=60

for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1; N_SAMPLES=30; N_EVENTS=30 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

echo "=== ScholaWrite-Augmented: Full Experiment Suite ==="
echo "  Samples per class: $N_SAMPLES"
echo "  Events per trace:  $N_EVENTS"
echo "  Output:            $RESULTS_DIR/"
echo ""

# 1. Adversarial evaluation (signal × tier AUC matrix)
echo "[1/9] Adversarial evaluation..."
uv run python scripts/run_adversarial_eval.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/adversarial_eval.json" \
  --output-latex "$RESULTS_DIR/adversarial_eval.tex"

# 2. Cross-detector comparison
echo "[2/9] Cross-detector comparison..."
uv run python scripts/run_detector_comparison.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/detector_comparison.json" \
  --output-latex "$RESULTS_DIR/detector_comparison.tex"

# 3. Signal ablation (leave-one-out + standalone)
echo "[3/9] Signal ablation..."
uv run python scripts/run_ablation.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/signal_ablation.json" \
  --output-latex "$RESULTS_DIR/signal_ablation.tex"

# 4. Scaling analysis (trace length vs AUC)
echo "[4/9] Scaling analysis..."
uv run python scripts/run_scaling_analysis.py \
  -n "$N_SAMPLES" \
  -o "$RESULTS_DIR/scaling_analysis.json" \
  --output-latex "$RESULTS_DIR/scaling_analysis.tex"

# 5. Cross-author evaluation
echo "[5/9] Cross-author evaluation..."
uv run python scripts/run_cross_author.py \
  -o "$RESULTS_DIR/cross_author.json"

# 6. Gradient forger (optimization hardness)
echo "[6/9] Gradient forger..."
uv run python scripts/run_gradient_forger.py \
  -o "$RESULTS_DIR/gradient_forger.json"

# 7. Paraphrase robustness
echo "[7/9] Paraphrase robustness..."
uv run python scripts/run_paraphrase_robustness.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/paraphrase_robustness.json" \
  --output-latex "$RESULTS_DIR/paraphrase_robustness.tex"

# 8. Incremental injection attack
echo "[8/9] Incremental injection..."
uv run python scripts/run_incremental_injection.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/incremental_injection.json" \
  --output-latex "$RESULTS_DIR/incremental_injection.tex"

# 9. Information-theoretic analysis (Fano bound + critical length)
echo "[9/9] Information-theoretic analysis..."
uv run python scripts/run_info_theoretic_analysis.py \
  -n "$N_SAMPLES" \
  -o "$RESULTS_DIR/info_theoretic.json" \
  --output-latex "$RESULTS_DIR/info_theoretic.tex"

echo ""
echo "=== All experiments complete ==="
echo "JSON results:  $RESULTS_DIR/*.json"
echo "LaTeX tables:  $RESULTS_DIR/*.tex"
echo ""
echo "To run the full data pipeline first: bash scripts/run_all.sh"
