#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=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
: "${WANDB_API_KEY:?Please export WANDB_API_KEY before running}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-brainiac-soop-outcome}"

mkdir -p logs
LOG_FILE="logs/soop_raw_image_tabular_$(date +%Y%m%d_%H%M%S).log"

FOLD_DIR="../../../datasets/fold_raw_trace"
OUTPUT_DIR="outputs/soop_smoke_fix/raw_image_tabular"
RUN_NAME="soop-gsrankin-raw-image-tabular"
TARGET_COL="nihss"

echo "[TRAIN][$(date -Iseconds)] start" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=3
python train_lightning_soop_regression.py \
  --config config_soop_regression.yml \
  --fold-dir "$FOLD_DIR" \
  --target-col "$TARGET_COL" \
  --include-tabular \
  --ckpt-path checkpoints/BrainIAC_mock.ckpt \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --batch-size 32 \
  --num-workers 4 \
  --max-epochs 50 \
  --optimizer adamw \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --normalize-features \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --use-wandb \
  --project-name "${WANDB_PROJECT}" \
  --accelerator gpu \
  --devices 1 \
  --precision 32 \
  2>&1 | tee -a "$LOG_FILE"

echo "[TRAIN][$(date -Iseconds)] done" | tee -a "$LOG_FILE"

BEST_CKPT="$(ls -1 "$OUTPUT_DIR"/checkpoints/*.ckpt | head -n 1 || true)"
if [[ -z "$BEST_CKPT" ]]; then
  echo "[ERROR] No checkpoint found under $OUTPUT_DIR/checkpoints" | tee -a "$LOG_FILE"
  exit 1
fi

RESOLVED_CONFIG="$OUTPUT_DIR/resolved_config_soop_regression.yml"
if [[ ! -f "$RESOLVED_CONFIG" ]]; then
  echo "[ERROR] Resolved config not found: $RESOLVED_CONFIG" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[EVAL][$(date -Iseconds)] start" | tee -a "$LOG_FILE"
python eval_soop_regression.py \
  --config "$RESOLVED_CONFIG" \
  --checkpoint "$BEST_CKPT" \
  --split-csv "$FOLD_DIR/test.csv" \
  --output-dir "$OUTPUT_DIR/eval" \
  --target-col "$TARGET_COL" \
  --batch-size 32 \
  --num-workers 4 \
  --include-tabular \
  --use-wandb \
  --project-name "${WANDB_PROJECT}" \
  --run-name "${RUN_NAME}-test" \
  2>&1 | tee -a "$LOG_FILE"

echo "[EVAL][$(date -Iseconds)] done" | tee -a "$LOG_FILE"
echo "[DONE][$(date -Iseconds)] train+eval completed" | tee -a "$LOG_FILE"

echo "Log saved to: $LOG_FILE"
