# BrainIAC SOOP Regression Experiment Manifest

## Environment
- Conda env: `hieupcvp`
- Repository root: `baseline_encoder/BrainIAC`
- Train script: `src/train_lightning_soop_regression.py`
- Eval script: `src/eval_soop_regression.py`

## Shared Inputs
- Config file: `src/config_soop_regression.yml`
- BrainIAC pretrained checkpoint: `<set BRAINIAC_CKPT>`
- Verified compatible checkpoint (current code): `src/checkpoints/BrainIAC_mock.ckpt`
- Note: `src/checkpoints/BrainIAC.ckpt` is not compatible with current ViT schema in `src/model.py`.
- Preprocessed fold: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold`
- Raw fold: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace`

## Runs

### Preprocessed TRACE (`datasets/fold`)
1. `soop_gsrankin_image_only`
   - target: `gs_rankin_6isdeath`
   - modality: image-only
2. `soop_gsrankin_image_tabular`
   - target: `gs_rankin_6isdeath`
   - modality: image+tabular
3. `soop_nihss_image_tabular`
   - target: `nihss`
   - modality: image+tabular

### Raw TRACE (`datasets/fold_raw_trace`)
4. `soop_raw_gsrankin_image_only`
   - target: `gs_rankin_6isdeath`
   - modality: image-only
5. `soop_raw_gsrankin_image_tabular`
   - target: `gs_rankin_6isdeath`
   - modality: image+tabular
6. `soop_raw_nihss_image_tabular`
   - target: `nihss`
   - modality: image+tabular

## Expected Artifacts Per Run
- `<output_dir>/resolved_config_soop_regression.yml`
- `<output_dir>/checkpoints/*.ckpt`
- `<output_dir>/eval/predictions.csv`
- `<output_dir>/eval/results_eval_soop_regression.json`

## Expected Metrics in JSON
- `mse`
- `rmse`
- `mae`
- `mape`
- `r2`
- `loss`

## Reproducible Command
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/BrainIAC
BRAINIAC_CKPT=/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/BrainIAC/src/checkpoints/BrainIAC_mock.ckpt \
bash research/run_soop_outcome_experiments_brainiac.sh
```

## Optional W&B Logging
Set:
- `USE_WANDB=1`
- `WANDB_API_KEY=<your_key>`
- `WANDB_PROJECT=brainiac-soop-outcome`
- `WANDB_MODE=online`
- `WANDB_TAGS=soop,regression,brainiac`
