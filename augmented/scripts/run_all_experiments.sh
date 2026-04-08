#!/usr/bin/env bash
# scripts/run_all_experiments.sh — reproduce all paper experiments
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

echo "=== ScholaWrite-Augmented: Experiment Suite ==="
echo "  Samples per class: $N_SAMPLES"
echo "  Events per trace:  $N_EVENTS"
echo "  Output:            $RESULTS_DIR/"
echo ""

echo "[1/9] Adversarial evaluation..."
uv run python scripts/eval/run_adversarial_eval.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/adversarial_eval.json" \
  --output-latex "$RESULTS_DIR/adversarial_eval.tex"

echo "[2/9] Cross-detector comparison..."
uv run python scripts/eval/run_detector_comparison.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/detector_comparison.json" \
  --output-latex "$RESULTS_DIR/detector_comparison.tex"

echo "[3/9] Signal ablation..."
uv run python scripts/eval/run_ablation.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/signal_ablation.json" \
  --output-latex "$RESULTS_DIR/signal_ablation.tex"

echo "[4/9] Scaling analysis..."
uv run python scripts/eval/run_scaling_analysis.py \
  -n "$N_SAMPLES" \
  -o "$RESULTS_DIR/scaling_analysis.json" \
  --output-latex "$RESULTS_DIR/scaling_analysis.tex"

echo "[5/9] Cross-author evaluation..."
uv run python scripts/eval/run_cross_author.py \
  -o "$RESULTS_DIR/cross_author.json"

echo "[6/9] Gradient forger..."
uv run python scripts/eval/run_gradient_forger.py \
  -o "$RESULTS_DIR/gradient_forger.json"

echo "[7/9] Paraphrase robustness..."
uv run python scripts/eval/run_paraphrase_robustness.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/paraphrase_robustness.json" \
  --output-latex "$RESULTS_DIR/paraphrase_robustness.tex"

echo "[8/9] Incremental injection..."
uv run python scripts/eval/run_incremental_injection.py \
  -n "$N_SAMPLES" --n-events "$N_EVENTS" \
  -o "$RESULTS_DIR/incremental_injection.json" \
  --output-latex "$RESULTS_DIR/incremental_injection.tex"

echo "[9/9] Information-theoretic analysis..."
uv run python scripts/eval/run_info_theoretic_analysis.py \
  -n "$N_SAMPLES" \
  -o "$RESULTS_DIR/info_theoretic.json" \
  --output-latex "$RESULTS_DIR/info_theoretic.tex"

echo ""
echo "=== All experiments complete ==="
echo "JSON:   $RESULTS_DIR/*.json"
echo "LaTeX:  $RESULTS_DIR/*.tex"
echo ""
echo "To rebuild data first: bash scripts/run_all.sh"
