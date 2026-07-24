#!/usr/bin/env bash
set -euo pipefail


export WANDB_API_KEY="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"
export WANDB_ENTITY="hieupcvp-hust"
mkdir -p /mnt/disk1/$USER/tmp 
export TMPDIR=/mnt/disk1/$USER/tmp
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
KFOLD_ROOT="/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/datasets/fold_raw_trace_fullmodal_mask/NIHSS/kfold"

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

BRAINIAC_CKPT="${BRAINIAC_CKPT:-$SRC_DIR/checkpoints/BrainIAC.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-$SRC_DIR/config_soop_regression.yml}"

# Placeholder: set this to the directory that contains fold subdirectories.
# Expected layout:
#   $KFOLD_ROOT/fold_1/{train.csv,valid.csv,test.csv}
#   $KFOLD_ROOT/fold_2/{train.csv,valid.csv,test.csv}
# KFOLD_ROOT="${KFOLD_ROOT:-__SET_KFOLD_ROOT__}"
FOLD_NAMES="${FOLD_NAMES:-fold_0 fold_1 fold_2 fold_3 fold_4}"
DATASET_TAG="${DATASET_TAG:-soop_kfold}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/raw_nihss_soop}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-50}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

OPTIMIZER="${OPTIMIZER:-adamw}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-16-mixed}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-1}"
NORMALIZE_FEATURES="${NORMALIZE_FEATURES:-1}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-raw-nihss-brainiac-soop-kfold-2752026}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-soop,brainiac,kfold,regression}"

if [[ "$KFOLD_ROOT" == "__SET_KFOLD_ROOT__" || -z "$KFOLD_ROOT" ]]; then
  echo "[ERROR] Set KFOLD_ROOT to the directory containing fold_*/train.csv, valid.csv, test.csv"
  echo "Example:"
  echo "  KFOLD_ROOT=/path/to/kfold bash $0"
  exit 1
fi

if [[ ! -f "$BRAINIAC_CKPT" ]]; then
  echo "[ERROR] BRAINIAC_CKPT not found: $BRAINIAC_CKPT"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] CONFIG_PATH not found: $CONFIG_PATH"
  exit 1
fi

if [[ "$USE_WANDB" == "1" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY or run with USE_WANDB=0}"
  export WANDB_API_KEY
  export WANDB_MODE
  export WANDB_TAGS
fi

echo "[PREFLIGHT] validating checkpoint compatibility: $BRAINIAC_CKPT"
python "$SRC_DIR/train_lightning_soop_regression.py" \
  --config "$CONFIG_PATH" \
  --ckpt-path "$BRAINIAC_CKPT" \
  --validate-checkpoint-only

slug_target() {
  local target_col="$1"
  case "$target_col" in
    gs_rankin_6isdeath|gs_rankin+6isdeath)
      echo "gsrankin6death"
      ;;
    nihss)
      echo "nihss"
      ;;
    *)
      echo "$target_col" | tr '+/' '__'
      ;;
  esac
}

run_experiment() {
  local fold_dir="$1"
  local fold_name="$2"
  local target_col="$3"
  local include_tabular="$4"

  local target_slug
  target_slug="$(slug_target "$target_col")"

  local modality_slug
  local modality_label
  local include_tab_args=()
  if [[ "$include_tabular" == "1" ]]; then
    modality_slug="image_tabular"
    modality_label="image+tabular"
    include_tab_args+=(--include-tabular)
  else
    modality_slug="image_only"
    modality_label="image-only"
    include_tab_args+=(--no-include-tabular)
  fi

  local experiment_name="brainiac_${DATASET_TAG}_${fold_name}_${target_slug}_${modality_slug}"
  local output_dir="$OUTPUT_ROOT/$DATASET_TAG/$fold_name/$target_slug/$modality_slug"

  echo "[RUN] fold=$fold_name target=$target_col modality=$modality_label"
  echo "[RUN] output_dir=$output_dir"
  mkdir -p "$output_dir"

  local train_wandb_args=()
  local eval_wandb_args=()
  if [[ "$USE_WANDB" == "1" ]]; then
    train_wandb_args+=(--use-wandb --project-name "$WANDB_PROJECT")
    eval_wandb_args+=(--use-wandb --project-name "$WANDB_PROJECT" --run-name "${experiment_name}_eval")
  else
    train_wandb_args+=(--no-use-wandb)
    eval_wandb_args+=(--no-use-wandb)
  fi

  local freeze_args=()
  if [[ "$FREEZE_BACKBONE" == "1" ]]; then
    freeze_args+=(--freeze-backbone)
  else
    freeze_args+=(--no-freeze-backbone)
  fi

  local normalize_args=()
  if [[ "$NORMALIZE_FEATURES" == "1" ]]; then
    normalize_args+=(--normalize-features)
  else
    normalize_args+=(--no-normalize-features)
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH="$SRC_DIR" python "$SRC_DIR/train_lightning_soop_regression.py" \
    --config "$CONFIG_PATH" \
    --fold-dir "$fold_dir" \
    --target-col "$target_col" \
    "${include_tab_args[@]}" \
    --ckpt-path "$BRAINIAC_CKPT" \
    --output-dir "$output_dir" \
    --run-name "$experiment_name" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --precision "$PRECISION" \
    --limit-train-batches "$LIMIT_TRAIN_BATCHES" \
    --limit-val-batches "$LIMIT_VAL_BATCHES" \
    "${freeze_args[@]}" \
    "${normalize_args[@]}" \
    "${train_wandb_args[@]}"

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

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH="$SRC_DIR" python "$SRC_DIR/eval_soop_regression.py" \
    --config "$output_dir/resolved_config_soop_regression.yml" \
    --checkpoint "$ckpt_path" \
    --split-csv "$fold_dir/test.csv" \
    --output-dir "$output_dir/eval" \
    --target-col "$target_col" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    "${include_tab_args[@]}" \
    "${eval_wandb_args[@]}"
}

for fold_name in $FOLD_NAMES; do
  fold_dir="$KFOLD_ROOT/$fold_name"
  for split_file in train.csv valid.csv test.csv; do
    if [[ ! -f "$fold_dir/$split_file" ]]; then
      echo "[ERROR] Missing $fold_dir/$split_file"
      exit 1
    fi
  done

  run_experiment "$fold_dir" "$fold_name" "nihss" 0
  run_experiment "$fold_dir" "$fold_name" "nihss" 1
done

echo "All BrainIAC SOOP kfold experiments completed."
