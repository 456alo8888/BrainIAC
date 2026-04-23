---
date: 2026-04-22T18:13:55+07:00
researcher: namk65hust
git_commit: cbc3adabd20575f7215733aa20b6934f51f71e7f
branch: main
repository: BrainIAC
topic: "Structure and training behavior of train_lightning_soop_regression.py (full training and frozen backbone paths)"
tags: [research, codebase, soop-regression, training, pytorch-lightning]
status: complete
last_updated: 2026-04-22
last_updated_by: namk65hust
---

# Research: Structure and training behavior of `train_lightning_soop_regression.py`

**Date**: 2026-04-22T18:13:55+07:00  
**Researcher**: namk65hust  
**Git Commit**: cbc3adabd20575f7215733aa20b6934f51f71e7f  
**Branch**: main  
**Repository**: BrainIAC

## Research Question
Read the current structure of codebase in `train_lightning_soop_regression.py` to search for training-process bugs and understand/fix training paths (full-train model vs frozen-backbone), then document findings.

## Summary
`src/train_lightning_soop_regression.py` is a Lightning training entrypoint that combines a ViT backbone (`src/model.py`) and a regression head, with optional tabular feature fusion. It validates checkpoint key structure before training, applies CLI-overrides onto YAML config, builds dataloaders from `src/soop_dataset.py`, then creates a `pl.Trainer` and runs `trainer.fit`.

Two training modes are present in current code:
- Full training path: `train.freeze_backbone: "no"` leaves all backbone parameters trainable.
- Frozen backbone path: `train.freeze_backbone: "yes"` sets `requires_grad=False` on backbone parameters before optimizer construction.

The experiment runner is `research/run_soop_outcome_experiments_brainiac.sh`, which performs checkpoint preflight, then executes 8 experiment combinations (preprocessed/raw folds x target x tabular on/off), and finally runs `src/eval_soop_regression.py` for each output.

## Detailed Findings

### 1) Entry script structure and execution flow (`train_lightning_soop_regression.py`)
- Checkpoint preflight is executed first by `inspect_backbone_checkpoint` and logs key counts/schema before training starts (`src/train_lightning_soop_regression.py:19`, `src/train_lightning_soop_regression.py:333`).
- CLI args are parsed and applied to YAML config through `apply_overrides`, including data, model, optimizer, trainer, logging, wandb, and visible-device fields (`src/train_lightning_soop_regression.py:218`, `src/train_lightning_soop_regression.py:278`).
- `--validate-checkpoint-only` exits after preflight (`src/train_lightning_soop_regression.py:321`, `src/train_lightning_soop_regression.py:348`).
- Config is materialized to `resolved_config_soop_regression.yml` in output dir before training (`src/train_lightning_soop_regression.py:366`).
- DataModule is instantiated and prepared with `setup("fit")`, then model is created from config + discovered tabular feature count (`src/train_lightning_soop_regression.py:373`, `src/train_lightning_soop_regression.py:378`).
- Lightning logger/callback wiring is conditional on `train.use_wandb` (`src/train_lightning_soop_regression.py:380`, `src/train_lightning_soop_regression.py:389`).
- Trainer accelerator/devices/precision are resolved from config and runtime CUDA availability before `trainer.fit` (`src/train_lightning_soop_regression.py:400`, `src/train_lightning_soop_regression.py:424`, `src/train_lightning_soop_regression.py:436`).

### 2) Model composition and full-vs-frozen training behavior
- `SOOPRegressionLightningModule` imports `ViTBackboneNet` from `src/model.py` and constructs a regression head with input dim `768` (+ tabular feature width when enabled) (`src/train_lightning_soop_regression.py:96`, `src/train_lightning_soop_regression.py:108`, `src/train_lightning_soop_regression.py:113`).
- Full-training mode:
  - Active when `train.freeze_backbone` is not `"yes"`.
  - All model parameters remain trainable and are included by `params = filter(lambda p: p.requires_grad, self.parameters())` (`src/train_lightning_soop_regression.py:122`, `src/train_lightning_soop_regression.py:192`).
- Frozen-backbone mode:
  - Active when `train.freeze_backbone == "yes"`.
  - Backbone params are set non-trainable before optimizer creation (`src/train_lightning_soop_regression.py:122`).
  - Optimizer still uses the same filtered parameter iterator, now excluding frozen params (`src/train_lightning_soop_regression.py:192`).
- Forward path:
  - Backbone returns image embedding.
  - Optional L2 feature normalization via `model.normalize_features`.
  - Optional concatenation with tabular tensor when `include_tabular=True` (`src/train_lightning_soop_regression.py:127`).

### 3) Backbone checkpoint loading and schema handling
- Preflight checkpoint check:
  - Requires checkpoint file and validates that it contains backbone-prefixed keys (`src/train_lightning_soop_regression.py:21`, `src/train_lightning_soop_regression.py:35`).
  - Reports counts and derived schema (`with_cross_attn` / `without_cross_attn`) (`src/train_lightning_soop_regression.py:41`, `src/train_lightning_soop_regression.py:49`).
- Actual backbone load:
  - `ViTBackboneNet` builds MONAI ViT-B style model and loads only `backbone.*` keys after stripping prefix (`src/model.py:12`, `src/model.py:27`).
  - Weight load uses `strict=False` (`src/model.py:36`).

### 4) Data pipeline used by training
- Data source is CSV-based split directory (`train.csv`, `valid.csv`, `test.csv`) via `SOOPRegressionDataModule` (`src/soop_dataset.py:196`, `src/soop_dataset.py:207`).
- Required columns per split are enforced (`image_path`, `subject_id`, target column) (`src/soop_dataset.py:38`, `src/soop_dataset.py:43`).
- Target name aliasing supports both `gs_rankin_6isdeath` and `gs_rankin+6isdeath` (`src/soop_dataset.py:72`).
- Tabular feature mode:
  - If no explicit tabular columns are provided, features are inferred by exclusion set.
  - Fallback parsing from JSON-like `tabular_features` column exists (`src/soop_dataset.py:82`, `src/soop_dataset.py:106`).
  - Normalization statistics are learned from train split and reused in val/test dataset construction (`src/soop_dataset.py:56`, `src/soop_dataset.py:229`).
- Image transform path uses MONAI Compose:
  - Train transform includes augmentation.
  - Validation transform is deterministic preprocessing (`src/dataset.py:12`, `src/dataset.py:37`).

### 5) Optimizer, scheduler, metrics, and checkpointing behavior
- Loss is MSE in train and validation steps (`src/train_lightning_soop_regression.py:119`, `src/train_lightning_soop_regression.py:147`).
- Validation-end computes MSE/RMSE/MAE/MAPE/R2 from accumulated predictions (`src/train_lightning_soop_regression.py:169`, `src/train_lightning_soop_regression.py:177`).
- Optimizer is configurable (`adamw`/`adam`/`sgd`), and scheduler is `CosineAnnealingWarmRestarts(T_0=50, T_mult=2)` (`src/train_lightning_soop_regression.py:187`, `src/train_lightning_soop_regression.py:201`).
- Best checkpoint is selected by `val_mae` (`src/train_lightning_soop_regression.py:392`).

### 6) Runtime trainer configuration and device selection behavior
- `gpu.visible_device` is applied only when `CUDA_VISIBLE_DEVICES` is not already present in environment (`src/train_lightning_soop_regression.py:351`).
- Accelerator selection starts from config then checks runtime CUDA availability; when CUDA is unavailable with requested GPU accelerator, code switches to CPU and adjusts precision if needed (`src/train_lightning_soop_regression.py:400`, `src/train_lightning_soop_regression.py:409`, `src/train_lightning_soop_regression.py:417`).

### 7) Experiment orchestration and evaluation path
- Runner script performs preflight, then trains/evaluates multiple experiment variants (`research/run_soop_outcome_experiments_brainiac.sh:46`, `research/run_soop_outcome_experiments_brainiac.sh:115`).
- Training calls always pass config + CLI overrides for fold, target, tabular mode, optimizer, LR, limits, and wandb control (`research/run_soop_outcome_experiments_brainiac.sh:81`).
- Evaluation script rebuilds dataset/model and loads trained checkpoint with `strict=True`, writes predictions/results JSON, and optionally logs to wandb (`src/eval_soop_regression.py:99`, `src/eval_soop_regression.py:102`, `src/eval_soop_regression.py:157`, `src/eval_soop_regression.py:182`).

## Code References
- `src/train_lightning_soop_regression.py:19` - Checkpoint preflight inspection entry.
- `src/train_lightning_soop_regression.py:96` - Lightning module definition.
- `src/train_lightning_soop_regression.py:122` - Frozen backbone toggle (`requires_grad=False`).
- `src/train_lightning_soop_regression.py:185` - Optimizer/scheduler configuration.
- `src/train_lightning_soop_regression.py:218` - CLI override application.
- `src/train_lightning_soop_regression.py:278` - Argument parser for train/eval control flags.
- `src/train_lightning_soop_regression.py:326` - Main execution flow.
- `src/model.py:7` - ViT backbone model initialization and checkpoint load.
- `src/soop_dataset.py:17` - SOOP regression dataset implementation.
- `src/soop_dataset.py:180` - SOOP regression datamodule and dataloaders.
- `src/dataset.py:12` - Train transform definition.
- `src/dataset.py:37` - Validation transform definition.
- `src/config_soop_regression.yml:1` - Base configuration consumed by training script.
- `research/run_soop_outcome_experiments_brainiac.sh:1` - Multi-run experiment orchestration.
- `src/eval_soop_regression.py:46` - Evaluation pipeline main entry.

## Architecture Documentation
Current SOOP regression training architecture in codebase:
- Entry + orchestration layer:
  - `train_lightning_soop_regression.py` for model training.
  - `run_soop_outcome_experiments_brainiac.sh` for batch experiment execution.
- Model layer:
  - `ViTBackboneNet` (MONAI ViT) in `model.py`.
  - `SOOPRegressionHead` and Lightning wrapper in `train_lightning_soop_regression.py`.
- Data layer:
  - CSV-driven dataset/datamodule in `soop_dataset.py`.
  - MONAI transforms in `dataset.py`.
- Evaluation layer:
  - `eval_soop_regression.py` for checkpoint-level metrics and artifacts.
- Configuration layer:
  - YAML defaults in `config_soop_regression.yml`.
  - CLI override merge in `apply_overrides`.

## Historical Context (from thoughts/)
No `thoughts/` directory is present in this repository path at research time, so no historical notes were added.

## Related Research
No existing `thoughts/shared/research/` documents were discovered in this repository path.

## Open Questions
- Which exact split CSV schema is currently authoritative for production runs when tabular mode is enabled (direct tabular columns vs `tabular_features` JSON fallback)?
- Which runtime mode is intended as default for current infra: full backbone fine-tuning or frozen-backbone training?
