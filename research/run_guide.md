# BrainIAC SOOP Run Guide (Post Fix)

## 1) Working directory
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/BrainIAC
```

## 2) Verify environment

### Check versions
```bash
conda run -n hieupcvp python -c "import numpy as np; print(np.__version__)"
conda run -n hieupcvp python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected:
- NumPy < 2 (đã verify: `1.26.4`)
- TensorFlow import thành công

### If NumPy is still 2.x
```bash
conda run -n hieupcvp python -m pip install "numpy<2"
```

## 3) Check trainer CLI startup path
```bash
conda run -n hieupcvp python src/train_lightning_soop_regression.py --help
```

## 4) Checkpoint compatibility preflight

### Compatible checkpoint (should pass)
```bash
conda run -n hieupcvp python src/train_lightning_soop_regression.py \
  --validate-checkpoint-only \
  --ckpt-path src/checkpoints/BrainIAC_mock.ckpt
```

### Incompatible checkpoint (should fail-fast)
```bash
conda run -n hieupcvp python src/train_lightning_soop_regression.py \
  --validate-checkpoint-only \
  --ckpt-path src/checkpoints/BrainIAC.ckpt
```

Expected fail message: missing `cross_attn`/`norm_cross_attn` + suggestion dùng `BrainIAC_mock.ckpt`.

## 5) Smoke run (recommended)

### A. Preprocessed image-only
```bash
conda run -n hieupcvp env PYTHONPATH=. python src/train_lightning_soop_regression.py \
  --config src/config_soop_regression.yml \
  --fold-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --target-col gs_rankin_6isdeath \
  --no-include-tabular \
  --ckpt-path src/checkpoints/BrainIAC_mock.ckpt \
  --output-dir outputs/soop_smoke_fix/preprocessed_image_only \
  --run-name smoke-preprocessed-image-only \
  --batch-size 2 \
  --num-workers 0 \
  --max-epochs 1 \
  --optimizer adamw \
  --learning-rate 8e-4 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --normalize-features \
  --limit-train-batches 0.02 \
  --limit-val-batches 0.02 \
  --no-use-wandb \
  --accelerator cpu \
  --devices 1 \
  --precision 32
```

### B. Raw image+tabular
```bash
conda run -n hieupcvp env PYTHONPATH=. python src/train_lightning_soop_regression.py \
  --config src/config_soop_regression.yml \
  --fold-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace \
  --target-col nihss \
  --include-tabular \
  --ckpt-path src/checkpoints/BrainIAC_mock.ckpt \
  --output-dir outputs/soop_smoke_fix/raw_tabular \
  --run-name smoke-raw-tabular \
  --batch-size 1 \
  --num-workers 0 \
  --max-epochs 1 \
  --optimizer adamw \
  --learning-rate 8e-4 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --normalize-features \
  --limit-train-batches 0.01 \
  --limit-val-batches 0.02 \
  --no-use-wandb \
  --accelerator cpu \
  --devices 1 \
  --precision 32
```

## 6) Full matrix run
```bash
bash research/run_soop_outcome_experiments_brainiac.sh
```

Optional runtime knobs:
```bash
export USE_WANDB=0
export CUDA_DEVICE=0
export EPOCHS=10
export BATCH_SIZE=8
export NUM_WORKERS=4
export LIMIT_TRAIN_BATCHES=1.0
export LIMIT_VAL_BATCHES=1.0
```

## 7) Expected artifacts per run
- `<output_dir>/resolved_config_soop_regression.yml`
- `<output_dir>/checkpoints/*.ckpt`
- `<output_dir>/eval/predictions.csv`
- `<output_dir>/eval/results_eval_soop_regression.json`

Required metric keys:
- `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`

## 8) Quick troubleshooting
- Nếu lỗi ABI NumPy quay lại: kiểm tra lại `numpy.__version__` và reinstall `numpy<2` trong đúng env `hieupcvp`.
- Nếu checkpoint fail-fast: dùng `src/checkpoints/BrainIAC_mock.ckpt` hoặc checkpoint khác có đủ `backbone.*` + `cross_attn/norm_cross_attn` keys.
- Nếu `limit_val_batches` quá nhỏ: tăng lên để đảm bảo có ít nhất 1 val batch.
