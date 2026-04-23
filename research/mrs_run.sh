#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"

BRAINIAC_CKPT="${BRAINIAC_CKPT:-/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/BrainIAC/src/checkpoints/BrainIAC.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

PREPROCESSED_FOLD="${PREPROCESSED_FOLD:-/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/datasets/fold_stripped_synthetic_mask/MRS}"
RAW_FOLD="${RAW_FOLD:-/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/datasets/fold_nonstripped_synthetic_mask/MRS}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-50}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

OPTIMIZER="${OPTIMIZER:-adamw}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_API_KEY="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"
WANDB_PROJECT="${WANDB_PROJECT:-brainiac-soop-outcome-2342026-mrs-freeze}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-soop,regression,brainiac}"

if [[ "$USE_WANDB" == "1" && -z "$WANDB_API_KEY" ]]; then
  echo "[ERROR] USE_WANDB=1 nhưng WANDB_API_KEY chưa được set"
  exit 1
fi

if [[ "$USE_WANDB" == "1" ]]; then
  export WANDB_API_KEY
  export WANDB_MODE
fi

if [[ ! -f "$BRAINIAC_CKPT" ]]; then
  echo "[ERROR] BRAINIAC_CKPT not found: $BRAINIAC_CKPT"
  exit 1
fi

echo "[PREFLIGHT] validating checkpoint compatibility: $BRAINIAC_CKPT"
python "$SRC_DIR/train_lightning_soop_regression.py" \
  --config "$SRC_DIR/config_soop_regression.yml" \
  --ckpt-path "$BRAINIAC_CKPT" \
  --validate-checkpoint-only

for split_file in train.csv valid.csv test.csv; do
  [[ -f "$PREPROCESSED_FOLD/$split_file" ]] || { echo "[ERROR] Missing $PREPROCESSED_FOLD/$split_file"; exit 1; }
  [[ -f "$RAW_FOLD/$split_file" ]] || { echo "[ERROR] Missing $RAW_FOLD/$split_file"; exit 1; }
done

run_experiment() {
  local fold_dir="$1"
  local target_col="$2"
  local include_tabular="$3"
  local run_name="$4"
  local output_dir="$OUTPUT_ROOT/$run_name"

  echo "[RUN] $run_name"
  mkdir -p "$output_dir"

  local include_tab_args=()
  if [[ "$include_tabular" == "1" ]]; then
    include_tab_args+=(--include-tabular)
  else
    include_tab_args+=(--no-include-tabular)
  fi

  local wandb_args=()
  if [[ "$USE_WANDB" == "1" ]]; then
    wandb_args+=(--use-wandb --project-name "$WANDB_PROJECT")
  else
    wandb_args+=(--no-use-wandb)
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH=. python "$SRC_DIR/train_lightning_soop_regression.py" \
    --config "$SRC_DIR/config_soop_regression.yml" \
    --fold-dir "$fold_dir" \
    --target-col "$target_col" \
    "${include_tab_args[@]}" \
    --ckpt-path "$BRAINIAC_CKPT" \
    --output-dir "$output_dir" \
    --run-name "$run_name" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --normalize-features \
    --limit-train-batches "$LIMIT_TRAIN_BATCHES" \
    --limit-val-batches "$LIMIT_VAL_BATCHES" \
    --freeze-backbone \
    "${wandb_args[@]}"

  local best_ckpt_file="$output_dir/best_checkpoint_path.txt"
  if [[ ! -f "$best_ckpt_file" ]]; then
    echo "[ERROR] Missing best checkpoint path file: $best_ckpt_file"
    exit 1
  fi

  local ckpt_path
  ckpt_path="$(<"$best_ckpt_file")"
  if [[ -z "$ckpt_path" || ! -f "$ckpt_path" ]]; then
    echo "[ERROR] Best checkpoint path is invalid: $ckpt_path"
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH=. python "$SRC_DIR/eval_soop_regression.py" \
    --config "$output_dir/resolved_config_soop_regression.yml" \
    --checkpoint "$ckpt_path" \
    --split-csv "$fold_dir/test.csv" \
    --output-dir "$output_dir/eval" \
    --target-col "$target_col" \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    "${include_tab_args[@]}"
}


run_experiment "$RAW_FOLD" "gs_rankin_6isdeath" 0 "soop_raw_gsrankin_image_only"&
run_experiment "$RAW_FOLD" "gs_rankin_6isdeath" 1 "soop_raw_gsrankin_image_tabular"&


run_experiment "$PREPROCESSED_FOLD" "gs_rankin_6isdeath" 0 "soop_gsrankin_image_only"&
run_experiment "$PREPROCESSED_FOLD" "gs_rankin_6isdeath" 1 "soop_gsrankin_image_tabular"&





echo "All BrainIAC SOOP outcome experiments completed."
