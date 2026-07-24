# BrainIAC SOOP K-Fold 4-Experiment Runner - Implemented 2026-05-26

## Scope

Implemented a new bash runner for the four SOOP regression experiment settings requested:

1. `gs_rankin_6isdeath` + image-only
2. `gs_rankin_6isdeath` + image+tabular
3. `nihss` + image-only
4. `nihss` + image+tabular

The runner uses `BrainIAC.ckpt`, runs over a k-fold directory layout, enables W&B tracking by default, and writes each run to a unique output directory so checkpoints are not overwritten.

## Files Added

- `research/run_soop_kfold_4_experiments_brainiac.sh`
  - Loops over fold directories under `KFOLD_ROOT`.
  - Runs all four target/modality combinations for each fold.
  - Uses `src/checkpoints/BrainIAC.ckpt` by default.
  - Saves artifacts under:
    - `outputs/soop_kfold_4_experiments/<DATASET_TAG>/<fold>/<target>/<modality>/`
  - Writes the best checkpoint path to each run's own:
    - `best_checkpoint_path.txt`
  - Evaluates each run on that fold's `test.csv`.
  - Logs training and evaluation metrics to W&B when `USE_WANDB=1`.

## Expected K-Fold Input Layout

Set `KFOLD_ROOT` to the directory that contains fold subdirectories. Each fold must match the current SOOP input schema:

```text
<KFOLD_ROOT>/
  fold_1/
    train.csv
    valid.csv
    test.csv
  fold_2/
    train.csv
    valid.csv
    test.csv
  fold_3/
    train.csv
    valid.csv
    test.csv
```

Each CSV must include:

- `subject_id`
- `image_path`
- `gs_rankin_6isdeath`
- `nihss`

For image+tabular runs, the current loader auto-selects numeric columns that are not excluded metadata/target columns. Typical tabular columns are:

- `sex`
- `age`
- `race`
- `acuteischaemicstroke`
- `priorstroke`
- `bmi`
- `etiology` or one-hot columns such as `etiology_1` to `etiology_5`

`image_path` must be a path directly readable by MONAI `LoadImaged`; the training code does not join it with another image root.

## Output Layout

For example, with:

```bash
DATASET_TAG=soop_trace_kfold
FOLD_NAMES="fold_1 fold_2"
```

the runner writes:

```text
BrainIAC/outputs/soop_kfold_4_experiments/soop_trace_kfold/fold_1/gsrankin6death/image_only/
BrainIAC/outputs/soop_kfold_4_experiments/soop_trace_kfold/fold_1/gsrankin6death/image_tabular/
BrainIAC/outputs/soop_kfold_4_experiments/soop_trace_kfold/fold_1/nihss/image_only/
BrainIAC/outputs/soop_kfold_4_experiments/soop_trace_kfold/fold_1/nihss/image_tabular/
```

Each leaf directory contains:

- `resolved_config_soop_regression.yml`
- `checkpoints/*.ckpt`
- `best_checkpoint_path.txt`
- `eval/predictions.csv`
- `eval/results_eval_soop_regression.json`

## W&B Tracking

W&B is enabled by default with:

```bash
USE_WANDB=1
WANDB_PROJECT=brainiac-soop-kfold-4exp
WANDB_MODE=online
WANDB_TAGS=soop,brainiac,kfold,regression
```

Set `WANDB_API_KEY` before running. Run names are generated as:

```text
brainiac_<DATASET_TAG>_<fold_name>_<target_slug>_<modality_slug>
```

Example:

```text
brainiac_soop_trace_kfold_fold_1_gsrankin6death_image_tabular
```

Evaluation W&B runs append `_eval`.

## Example Run

From the `BrainIAC` repository root:

```bash
cd /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/BrainIAC

export WANDB_API_KEY="your_wandb_key"

KFOLD_ROOT="/path/to/your/kfold_root" \
FOLD_NAMES="fold_1 fold_2 fold_3 fold_4 fold_5" \
DATASET_TAG="soop_trace_kfold" \
CUDA_DEVICE=0 \
EPOCHS=50 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
bash research/run_soop_kfold_4_experiments_brainiac.sh
```

Smoke-test example:

```bash
cd /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/BrainIAC

export WANDB_API_KEY="your_wandb_key"

KFOLD_ROOT="/path/to/your/kfold_root" \
FOLD_NAMES="fold_1" \
DATASET_TAG="smoke_soop_trace_kfold" \
CUDA_DEVICE=0 \
EPOCHS=1 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
LIMIT_TRAIN_BATCHES=0.02 \
LIMIT_VAL_BATCHES=0.02 \
bash research/run_soop_kfold_4_experiments_brainiac.sh
```

To run without W&B:

```bash
KFOLD_ROOT="/path/to/your/kfold_root" \
USE_WANDB=0 \
bash research/run_soop_kfold_4_experiments_brainiac.sh
```

## Configurable Environment Variables

- `KFOLD_ROOT`: required placeholder path to k-fold root.
- `FOLD_NAMES`: space-separated fold directory names. Default: `fold_1 fold_2 fold_3 fold_4 fold_5`.
- `DATASET_TAG`: included in output paths and W&B run names. Default: `soop_kfold`.
- `BRAINIAC_CKPT`: BrainIAC checkpoint path. Default: `src/checkpoints/BrainIAC.ckpt`.
- `OUTPUT_ROOT`: root output directory. Default: `BrainIAC/outputs/soop_kfold_4_experiments`.
- `CUDA_DEVICE`: CUDA visible device. Default: `0`.
- `EPOCHS`, `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `NUM_WORKERS`.
- `OPTIMIZER`, `LEARNING_RATE`, `WEIGHT_DECAY`, `GRAD_CLIP_NORM`.
- `FREEZE_BACKBONE`: default `1`.
- `NORMALIZE_FEATURES`: default `1`.
- `USE_WANDB`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_MODE`, `WANDB_TAGS`.

## Fairness Note

The four experiments use the same fold split when run from the same `KFOLD_ROOT`, so image-only vs image+tabular and Rankin vs NIHSS comparisons share the same train/valid/test partition.

However, the current `SOOPRegressionDataset` automatically chooses tabular columns from numeric CSV columns. If a CSV includes both outcome columns, then the non-target outcome can become a tabular input. For example, when predicting `gs_rankin_6isdeath`, a numeric `nihss` column can be used as a tabular feature. Keep this in mind when interpreting image+tabular results.

For strict clinical-feature-only experiments, prepare k-fold CSV files where tabular numeric columns contain only the intended covariates, or update the dataset/training code to accept an explicit tabular column list.

## Verification

The runner was syntax-checked with:

```bash
bash -n BrainIAC/research/run_soop_kfold_4_experiments_brainiac.sh
```
