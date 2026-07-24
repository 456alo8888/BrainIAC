---
date: 2026-05-26T11:53:07+07:00
researcher: hieupc
git_commit: 2c81b24a3d9c5647e57b56978a852c7a4ee67bb5
branch: main
repository: BrainIAC
topic: "BrainIAC src training, inference, model location, modules, and SOOP experiment files"
tags: [research, codebase, brainiac, src, training, inference, soop-regression]
status: complete
last_updated: 2026-05-26
last_updated_by: hieupc
last_updated_note: "Updated model checkpoint compatibility after strict BrainIAC.ckpt load fix"
---

# Research: BrainIAC `src` training, inference, model, modules, and SOOP experiment files

**Date**: 2026-05-26T11:53:07+07:00  
**Researcher**: hieupc  
**Git Commit**: 2c81b24a3d9c5647e57b56978a852c7a4ee67bb5  
**Branch**: main  
**Repository**: BrainIAC

## Research Question
Read and map BrainIAC `src/`: training / inference, where the model is located, what modules exist inside `src`, and which files are involved in SOOP experiments.

## Summary
The central BrainIAC backbone is defined in `src/model.py` as `ViTBackboneNet`, a MONAI ViT-B style 3D backbone that loads checkpoint keys prefixed with `backbone.` and returns the CLS token feature vector with width 768. The active implementation removes unused MONAI 1.5.x cross-attention modules before strict loading so `src/checkpoints/BrainIAC.ckpt` can load exactly without missing backbone parameters.

SOOP regression is implemented as an added pipeline:
- Data: `src/soop_dataset.py`
- Training: `src/train_lightning_soop_regression.py`
- Evaluation: `src/eval_soop_regression.py`
- Config: `src/config_soop_regression.yml`
- Run scripts: `research/nihss_run_soop_outcome_experiments_brainiac.sh`, `research/mrs_run.sh`, plus case-specific scripts in `src/bash_*.sh`
- Manifest / plan: `research/experiment_manifest_brainiac.md`, `research/experiment_plan.md`

The older/general BrainIAC downstream pipeline is also present:
- Backbone/classifier wrappers: `src/model.py`
- Common image transforms and datasets: `src/dataset.py`
- Training scripts: `src/train_lightning_brainage.py`, `src/train_lightning_mci.py`, `src/train_lightning_idh.py`, `src/train_lightning_os.py`, `src/train_lightning_multiclass.py`, `src/train_lightning_segmentation.py`
- Inference scripts: `src/test_inference_finetune.py`, `src/get_brainiac_features.py`, `src/get_brainiac_saliencymap.py`, saliency generation scripts, and segmentation inference scripts.

## Detailed Findings

### Model Location
- `src/model.py:152` defines the active `ViTBackboneNet` implementation.
- `src/model.py:157` constructs a MONAI `ViT` with `in_channels=1`, `img_size=(96,96,96)`, `patch_size=(16,16,16)`, `hidden_size=768`, `num_layers=12`, and `num_heads=12`.
- `src/model.py:168` removes unused `norm_cross_attn` and `cross_attn` modules that MONAI 1.5.x registers on transformer blocks even when cross-attention is disabled.
- `src/model.py:180` loads the checkpoint.
- `src/model.py:184` filters keys starting with `backbone.` and strips that prefix.
- `src/model.py:192` loads those backbone weights with `strict=True`.
- `src/model.py:195` forwards image tensors through the ViT.
- `src/model.py:198` returns the CLS token as a 768-dimensional feature.
- `src/model.py:202` defines a simple `Classifier`.
- `src/model.py:210`, `src/model.py:223`, and `src/model.py:254` define single-, dual-, and quad-scan wrappers.

### SOOP Training Flow
- `src/train_lightning_soop_regression.py:19` checks backbone checkpoint structure before training.
- `src/train_lightning_soop_regression.py:53` defines regression metrics: MSE, RMSE, MAE, MAPE, R2, and loss.
- `src/train_lightning_soop_regression.py:79` defines the SOOP regression head.
- `src/train_lightning_soop_regression.py:107` defines the Lightning module.
- `src/train_lightning_soop_regression.py:119` creates `ViTBackboneNet`.
- `src/train_lightning_soop_regression.py:124` sets regression-head input dimension to `768 + tabular_features` when tabular mode is enabled.
- `src/train_lightning_soop_regression.py:133` freezes the backbone when `train.freeze_backbone` is `"yes"`.
- `src/train_lightning_soop_regression.py:138` runs the forward path: backbone feature, optional L2 normalization, optional tabular concatenation, then regression head.
- `src/train_lightning_soop_regression.py:152` and `src/train_lightning_soop_regression.py:163` implement training and validation steps.
- `src/train_lightning_soop_regression.py:196` configures AdamW/Adam/SGD and cosine warm restarts.
- `src/train_lightning_soop_regression.py:229` applies CLI overrides onto YAML config.
- `src/train_lightning_soop_regression.py:337` is the training entrypoint.
- `src/train_lightning_soop_regression.py:386` instantiates `SOOPRegressionDataModule`.
- `src/train_lightning_soop_regression.py:400` saves best checkpoint by `val_mae`.
- `src/train_lightning_soop_regression.py:447` calls `trainer.fit`.
- `src/train_lightning_soop_regression.py:449` writes `best_checkpoint_path.txt`.

### SOOP Data Flow
- `src/soop_dataset.py:17` defines `SOOPRegressionDataset`.
- `src/soop_dataset.py:31` reads split CSV files.
- `src/soop_dataset.py:38` and `src/soop_dataset.py:40` require `image_path` and `subject_id`.
- `src/soop_dataset.py:72` resolves target aliases between `gs_rankin_6isdeath` and `gs_rankin+6isdeath`.
- `src/soop_dataset.py:82` resolves tabular columns.
- `src/soop_dataset.py:127` supports fallback parsing from a JSON-like `tabular_features` column.
- `src/soop_dataset.py:170` returns each sample with image, label, label mask, and subject ID; tabular tensor is added when enabled.
- `src/soop_dataset.py:201` defines `SOOPRegressionDataModule`.
- `src/soop_dataset.py:217` creates train/valid/test datasets from `fold_dir/{train,valid,test}.csv`.
- `src/soop_dataset.py:241` stores train-set tabular feature count and column names.
- `src/soop_dataset.py:244` and `src/soop_dataset.py:256` reuse train tabular columns and normalization stats for validation/test.

### SOOP Evaluation / Inference
- `src/eval_soop_regression.py:17` defines eval CLI arguments.
- `src/eval_soop_regression.py:46` starts eval.
- `src/eval_soop_regression.py:66` rebuilds the train dataset only to recover tabular columns/stats when tabular mode is enabled.
- `src/eval_soop_regression.py:78` builds the target split dataset.
- `src/eval_soop_regression.py:92` rebuilds `SOOPRegressionLightningModule`.
- `src/eval_soop_regression.py:93` loads the trained checkpoint.
- `src/eval_soop_regression.py:95` loads checkpoint weights with `strict=True`.
- `src/eval_soop_regression.py:110` runs prediction loop.
- `src/eval_soop_regression.py:141` computes metrics.
- `src/eval_soop_regression.py:147` writes `predictions.csv`.
- `src/eval_soop_regression.py:150` writes `results_eval_soop_regression.json`.

### Non-SOOP Training and Inference Modules
- `src/train_lightning_brainage.py:18` defines brain-age regression training using `ViTBackboneNet`, `Classifier`, and `SingleScanModel`.
- `src/train_lightning_mci.py:18` defines single-image binary classification for MCI/stroke-style data.
- `src/train_lightning_idh.py:16` defines dual-image binary classification.
- `src/train_lightning_os.py:16` defines quad-image binary classification.
- `src/train_lightning_multiclass.py:17` defines single-image multiclass sequence classification.
- `src/train_lightning_segmentation.py:15` defines segmentation training.
- `src/segmentation_model.py:5` defines `ViTUNETRSegmentationModel`, which transfers ViT weights into a MONAI UNETR encoder.
- `src/test_inference_finetune.py:102` loads finetuned task checkpoints for multiple task types and image input layouts.
- `src/get_brainiac_features.py:29` extracts 768-dimensional BrainIAC backbone features to CSV.
- `src/get_brainiac_saliencymap.py:19` extracts ViT attention maps for saliency output.

### SOOP Experiment Files
- `src/config_soop_regression.yml:1` is the default SOOP regression config.
- `research/experiment_plan.md` documents the original plan for SOOP regression on preprocessed and raw TRACE splits.
- `research/experiment_manifest_brainiac.md` documents the intended run matrix, expected artifacts, and reproducible command.
- `research/nihss_run_soop_outcome_experiments_brainiac.sh:11` and `:12` point to NIHSS preprocessed/raw fold directories.
- `research/nihss_run_soop_outcome_experiments_brainiac.sh:47` runs checkpoint validation.
- `research/nihss_run_soop_outcome_experiments_brainiac.sh:81` launches SOOP training.
- `research/nihss_run_soop_outcome_experiments_brainiac.sh:115` launches SOOP evaluation.
- `research/nihss_run_soop_outcome_experiments_brainiac.sh:130` and `:131` currently run raw NIHSS image+tabular and raw NIHSS image-only cases.
- `research/mrs_run.sh:11` and `:12` point to MRS preprocessed/raw fold directories.
- `research/mrs_run.sh:127` through `:132` run raw and preprocessed `gs_rankin_6isdeath` image-only/image+tabular cases.
- `src/bash_preprocessed_image_only.sh`, `src/bash_preprocessed_image_tabular.sh`, `src/bash_raw_image_only.sh`, and `src/bash_raw_image_tabular.sh` are case-specific train+eval scripts.

## Code References
- `src/model.py:152` - Active BrainIAC ViT backbone.
- `src/model.py:50` - classifier head.
- `src/dataset.py:12` - single-image train transform.
- `src/dataset.py:37` - single-image validation transform.
- `src/soop_dataset.py:17` - SOOP dataset.
- `src/soop_dataset.py:201` - SOOP data module.
- `src/train_lightning_soop_regression.py:107` - SOOP Lightning module.
- `src/eval_soop_regression.py:46` - SOOP eval entrypoint.
- `src/config_soop_regression.yml:1` - SOOP config.
- `research/nihss_run_soop_outcome_experiments_brainiac.sh:57` - NIHSS experiment function.
- `research/mrs_run.sh:57` - MRS experiment function.

## Historical Context
Relevant existing docs in this repository:
- `2242026/2026-04-22-soop-regression-training-structure.md` - earlier research note for SOOP regression training structure.
- `research/experiment_plan.md` - planned SOOP pipeline and success criteria.
- `research/experiment_manifest_brainiac.md` - run matrix and expected outputs.
- `research/research.md` - broader BrainIAC SOOP notes.
- `research/research_bug.md` and `research/plan_fix_bug.md` - notes around runner/checkpoint/environment failures.

No top-level `thoughts/` directory was present in this workspace. No `hack/spec_metadata.sh` script was present, so metadata was gathered from git and shell commands.

## Follow-up Research 2026-05-26T12:15:00+07:00
`src/checkpoints/BrainIAC.ckpt` contains 137 `backbone.*` keys and no cross-attention weights. In the local `hieupcvp` environment, MONAI 1.5.2 registers `norm_cross_attn` and `cross_attn` modules on each `TransformerBlock`, which created 84 expected parameters that are not present in `BrainIAC.ckpt`.

The active `src/model.py` now deletes those unused cross-attention modules before strict loading. Verification with `ViTBackboneNet('BrainIAC/src/checkpoints/BrainIAC.ckpt')` loaded successfully, produced `backbone_state_keys=137`, and had no `cross_attn`/`norm_cross_attn` parameters in `backbone.state_dict()`.

`src/train_lightning_soop_regression.py` was also updated so the no-cross-attention checkpoint message describes strict loading through this non-cross-attention schema instead of saying partial `strict=False` loading.

## Open Questions
- The manifest still names `research/run_soop_outcome_experiments_brainiac.sh`, but the file present in this checkout is split into `research/nihss_run_soop_outcome_experiments_brainiac.sh` and `research/mrs_run.sh`.
- The active generated feature file `src/inference/features/features.csv` contains 768 feature columns plus `GroundTruthClassLabel`, matching `src/get_brainiac_features.py`.
- Full SOOP training entrypoint verification is blocked in the current `hieupcvp` environment by `ModuleNotFoundError: No module named 'pkg_resources'` during `pytorch_lightning` import.
