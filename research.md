---
date: 2026-04-22T14:12:23+07:00
researcher: GitHub Copilot
git_commit: cbc3ada
branch: main
repository: BrainIAC
topic: "Current codebase state, run flow, and input organization"
tags: [research, codebase, runtime, input-schema, soop]
status: complete
last_updated: 2026-04-22
last_updated_by: GitHub Copilot
---

# BrainIAC Codebase Research

## Research Scope
This document describes the current state of the BrainIAC codebase, how it is run today, and how input data is organized and switched to another dataset.

Repository root analyzed: `BrainIAC/`

## Current State Snapshot
- Core model and task scripts are under `src/`.
- Main user-facing project description is in `README.md`.
- Task-specific usage docs are in `docs/downstream_tasks/`.
- SOOP regression experiment orchestration is in `research/run_soop_outcome_experiments_brainiac.sh`.
- Existing research logs and implementation history are in `research/` and `research_phase2/`.

### Main code groups in `src/`
- Backbone and model wrappers:
  - `model.py`
  - `load_brainiac.py`
- Dataset and transforms:
  - `dataset.py`
  - `soop_dataset.py`
  - `dataset_segmentation.py`
- Training entrypoints:
  - `train_lightning_brainage.py`
  - `train_lightning_mci.py`
  - `train_lightning_multiclass.py`
  - `train_lightning_idh.py`
  - `train_lightning_os.py`
  - `train_lightning_segmentation.py`
  - `train_lightning_soop_regression.py`
- Evaluation and inference:
  - `eval_soop_regression.py`
  - `test_inference_finetune.py`
  - `test_inference_perturbation.py`
  - `test_segmentation.py`
  - `generate_segmentation.py`
- SOOP helper bash launchers:
  - `bash_image_only.sh`
  - `bash_preprocessed_image_only.sh`
  - `bash_preprocessed_image_tabular.sh`
  - `bash_raw_image_only.sh`
  - `bash_raw_image_tabular.sh`

## How To Run The Codebase

## 1) Setup
Install dependencies from `requirements.txt` in a Python environment (README currently shows conda + pip workflow).

## 2) Run SOOP outcome regression matrix (preprocessed + raw)
Primary runner:
- `research/run_soop_outcome_experiments_brainiac.sh`

This script performs:
1. checkpoint preflight via `train_lightning_soop_regression.py --validate-checkpoint-only`
2. split-file checks for both fold directories (`train.csv`, `valid.csv`, `test.csv`)
3. 6 train+eval runs:
   - preprocessed: image-only `gs_rankin_6isdeath`
   - preprocessed: image+tabular `gs_rankin_6isdeath`
   - preprocessed: image+tabular `nihss`
   - raw: image-only `gs_rankin_6isdeath`
   - raw: image+tabular `gs_rankin_6isdeath`
   - raw: image+tabular `nihss`

Default runtime variables in the runner:
- `BRAINIAC_CKPT`
- `OUTPUT_ROOT`
- `CUDA_DEVICE`
- `PREPROCESSED_FOLD`
- `RAW_FOLD`
- `BATCH_SIZE`
- `NUM_WORKERS`
- `EPOCHS`
- `LIMIT_TRAIN_BATCHES`
- `LIMIT_VAL_BATCHES`
- `OPTIMIZER`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `GRAD_CLIP_NORM`
- `USE_WANDB`
- `WANDB_PROJECT`
- `WANDB_MODE`
- `WANDB_TAGS`

## 3) Run single SOOP scenarios via bash wrappers
Scripts under `src/bash_*.sh` run train then eval for one scenario each. They define local constants such as:
- `FOLD_DIR`
- `TARGET_COL`
- `OUTPUT_DIR`
- `RUN_NAME`

Each wrapper calls:
- `train_lightning_soop_regression.py` (train)
- `eval_soop_regression.py` (test eval on `FOLD_DIR/test.csv`)

## 4) Run non-SOOP downstream tasks
Task docs live in `docs/downstream_tasks/` and map to these scripts:
- Brain age: `train_lightning_brainage.py`
- MCI: `train_lightning_mci.py`
- MR sequence multiclass: `train_lightning_multiclass.py`
- IDH dual-image: `train_lightning_idh.py`
- Overall survival quad-image: `train_lightning_os.py`
- Segmentation: `train_lightning_segmentation.py`

These scripts read input paths from YAML configs (`config_finetune.yml` or `config_finetune_segmentation.yml`).

## 5) Main outputs produced
SOOP runs generate per-run folders under `OUTPUT_ROOT/<run_name>/` containing:
- `resolved_config_soop_regression.yml`
- `checkpoints/*.ckpt`
- `eval/predictions.csv`
- `eval/results_eval_soop_regression.json`

## Input Organization

## A) SOOP input folder organization
SOOP train/eval expects a fold directory with fixed split filenames:

```text
<your_fold_dir>/
  train.csv
  valid.csv
  test.csv
```

`SOOPRegressionDataset` in `src/soop_dataset.py` reads rows directly from each CSV.

Required CSV columns:
- `image_path`
- `subject_id`
- target column (default `gs_rankin_6isdeath`; alias support for `gs_rankin+6isdeath`)

Optional tabular usage:
- If `include_tabular` is enabled, tabular columns are inferred as all non-excluded columns.
- Excluded columns include identifiers/paths/targets such as `subject_id`, `trace_dir`, `image_path`, `mask_path`, `segm_path`, `tabular_path`, `tabular_features`, `nihss`, `gs_rankin_6isdeath`, `gs_rankin+6isdeath`.
- If no direct tabular columns are found but `tabular_features` exists, JSON-like per-row features are parsed.

`image_path` handling:
- The path is consumed as-is from CSV and passed into MONAI load transforms.
- No root prefix is added by SOOP dataset code.

## B) Non-SOOP input conventions in `dataset.py`

Single-image datasets (`BrainAgeDataset`, `MCIStrokeDataset`):
- CSV columns: `pat_id`, `label`
- image path built as: `<root_dir>/<pat_id>.nii.gz`

Dual-image dataset (`DualImageDataset`):
- CSV columns: `pat_id`, `label`
- image paths built as:
  - `<root_dir>/<pat_id>_t2f.nii.gz`
  - `<root_dir>/<pat_id>_t1ce.nii.gz`

Quad-image dataset (`QuadImageDataset`):
- CSV columns: `pat_id`, `survival`
- image paths built as:
  - `<root_dir>/<pat_id>_t1ce.nii.gz`
  - `<root_dir>/<pat_id>_t1n.nii.gz`
  - `<root_dir>/<pat_id>_t2w.nii.gz`
  - `<root_dir>/<pat_id>_t2f.nii.gz`

Sequence multiclass dataset (`SequenceDataset`):
- CSV columns include `PatientID`, `SequenceLabel`, `ScanID`, `Sequence`, `Dataset`
- image path built as:
  - `<root_dir>/<Dataset>/data/<PatientID>-<ScanID>-<Sequence>.nii.gz`

Segmentation paths in `dataset_segmentation.py`:
- CSV columns: `image_path`, `mask_path`
- both paths are consumed directly from CSV.

## How To Change To Another Input Dataset

## A) SOOP pipeline switch points

1. Change fold directory (split location)
- Config file: `src/config_soop_regression.yml` -> `data.fold_dir`
- CLI override: `--fold-dir <new_fold_dir>` in `train_lightning_soop_regression.py`
- Runner env override:
  - `PREPROCESSED_FOLD=<new_preprocessed_fold>`
  - `RAW_FOLD=<new_raw_fold>`

2. Change target variable
- Config file: `data.target_col`
- CLI override: `--target-col <column_name>`

3. Change image-only vs image+tabular
- Config file: `data.include_tabular: false/true`
- CLI flags:
  - image-only: `--no-include-tabular`
  - image+tabular: `--include-tabular`

4. Change checkpoint and output location
- checkpoint:
  - config: `simclrvit.ckpt_path`
  - CLI: `--ckpt-path`
  - runner env: `BRAINIAC_CKPT`
- output:
  - config: `logger.output_dir`
  - CLI: `--output-dir`
  - runner env: `OUTPUT_ROOT`

## B) Non-SOOP pipeline switch points

1. Update YAML config used by the task script:
- `src/config_finetune.yml` for brain age/MCI/multiclass/IDH/OS
- `src/config_finetune_segmentation.yml` for segmentation

2. Adjust dataset pointers in config:
- `data.csv_file`
- `data.val_csv`
- `data.root_dir`
- segmentation: `data.train_csv`, `data.val_csv`

3. Ensure filenames in your input image folder match the naming scheme required by the selected dataset class (`dataset.py`).

## Input Folder Examples In Repository

The repo includes minimal sample inputs under `src/data/`:

```text
src/data/
  csvs/
    sample.csv                # columns: pat_id,label
  sample/
    processed/
      I10307487_0000.nii.gz
      subpixar009_T1w.nii.gz
      00001_t1c.nii.gz
      00001_t1n.nii.gz
      00001_t2f.nii.gz
      00001_t2w.nii.gz
    unprocessed/
      I10307487.nii.gz
```

## Cross-File Runtime Map (Current)
- SOOP train entry: `src/train_lightning_soop_regression.py`
- SOOP eval entry: `src/eval_soop_regression.py`
- SOOP matrix orchestrator: `research/run_soop_outcome_experiments_brainiac.sh`
- SOOP dataset loader: `src/soop_dataset.py`
- Shared transform utilities: `src/dataset.py`

## Notes From Existing Research Docs
- `research/implemented.md` and `research/implemented_fix.md` record SOOP implementation and follow-up fixes.
- `research_phase2/implemented.md` records updates to image-only train+eval wrapper and eval logging flow.

This document is a snapshot of what exists in the codebase at commit `cbc3ada` on branch `main`.
