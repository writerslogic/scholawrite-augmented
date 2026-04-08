#!/usr/bin/env bash
# scripts/run_all.sh — full data construction pipeline
set -euo pipefail

SEED_RAW=${1:-"data/seed/raw/hf_scholawrite"}
SPLIT=${2:-"all_sorted"}
OUT_DIR=${3:-"data/augmented/full"}
BUDGET=${4:-"150.0"}
WORKERS=${5:-"20"}

SEED_NORM="data/seed/normalized/${SPLIT}.jsonl"
INJ_RAW="data/injections/raw/${SPLIT}.jsonl"
MODELS_CONFIG="configs/openrouter_models.json"

echo "=== ScholaWrite-Augmented: Data Pipeline ==="
echo "Seed:    $SEED_RAW"
echo "Split:   $SPLIT"
echo "Output:  $OUT_DIR"
echo "Budget:  \$${BUDGET}"

echo "[1/10] Ingesting seed data..."
uv run python scripts/pipeline/ingest_seed.py --input "$SEED_RAW" --output "$SEED_NORM" --split "$SPLIT"

echo "[2/10] Generating injections via OpenRouter..."
uv run python scripts/pipeline/generate_injections.py \
    --input "$SEED_NORM" \
    --output "$INJ_RAW" \
    --provider openrouter \
    --models-file "$MODELS_CONFIG" \
    --level contextual \
    --seed 42 \
    --budget-usd "$BUDGET" \
    --max-total 50

echo "[3/10] Building augmented dataset..."
uv run python scripts/pipeline/build_augmented_dataset.py \
    --seed-docs "$SEED_NORM" \
    --injections "$INJ_RAW" \
    --output-dir "$OUT_DIR" \
    --openrouter \
    --models-file "$MODELS_CONFIG" \
    --workers "$WORKERS"

echo "[4/10] Generating anomalies..."
uv run python scripts/pipeline/generate_anomalies.py --input "$OUT_DIR/documents.jsonl" --output "$OUT_DIR/anomalies.jsonl"

echo "[5/10] Validating annotations..."
uv run python scripts/validate/validate_annotations.py --input "$OUT_DIR/documents.jsonl" --strict

echo "[6/10] Running baselines..."
uv run python scripts/eval/run_baselines.py --input "$OUT_DIR/documents.jsonl" --output-dir "results/baselines"

echo "[7/10] Running harm analysis..."
uv run python scripts/eval/run_harm.py --input "$OUT_DIR/documents.jsonl" --output-dir "results/harm"

echo "[8/10] Linting data and scanning for PII..."
uv run python scripts/validate/lint_data.py --input-dir "$OUT_DIR" --pii-scan

echo "[9/10] Generating trajectory visualization..."
uv run python scripts/viz/visualize_trajectories.py --input "$OUT_DIR/documents.jsonl" --output "results/visualization_full.html"

echo "[10/10] Generating checksums..."
uv run python scripts/utils/hash_artifacts.py --artifacts "$OUT_DIR/documents.jsonl,$OUT_DIR/annotations.jsonl,$OUT_DIR/anomalies.jsonl,$OUT_DIR/generated_text.jsonl,$OUT_DIR/stats.json"

echo "=== Pipeline complete ==="
