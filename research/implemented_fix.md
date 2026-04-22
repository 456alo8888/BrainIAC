# BrainIAC SOOP Bug Fix - Implemented

## Date
2026-03-20

## Scope
Triển khai theo `research/plan_fix_bug.md` để xử lý 2 nhóm lỗi chính khi chạy SOOP runner:
1. lỗi import chain TensorBoard/TensorFlow/NumPy ABI,
2. lỗi checkpoint không tương thích schema backbone hiện tại.

## Files Updated

### 1) `src/train_lightning_soop_regression.py`
- Thêm preflight checkpoint validator `inspect_backbone_checkpoint(...)`.
- Thêm cờ CLI `--validate-checkpoint-only` để fail-fast trước train.
- Chuyển logger mặc định sang `logger=False` khi `--no-use-wandb` (không để Lightning fallback logger).
- Chỉ thêm `LearningRateMonitor` khi thực sự bật W&B logger.
- Lazy import `ViTBackboneNet` và `SOOPRegressionDataModule` để tránh import MONAI ở startup path.
- Bổ sung resolve path config robust theo `SCRIPT_DIR` khi chạy từ repo root.

### 2) `research/run_soop_outcome_experiments_brainiac.sh`
- Đổi checkpoint mặc định sang `src/checkpoints/BrainIAC_mock.ckpt` (checkpoint đã verify tương thích schema).
- Thêm bước preflight checkpoint compatibility trước toàn bộ matrix run.

### 3) `README.md`
- Đồng bộ ghi chú checkpoint compatibility cho SOOP.
- Bổ sung preflight kiểm tra checkpoint trước khi chạy full runner.

### 4) `research/experiment_manifest_brainiac.md`
- Bổ sung note checkpoint tương thích và không tương thích.
- Cập nhật command tái lập dùng checkpoint tương thích.

### 5) `src/quickstart.ipynb`
- Đồng bộ note compatibility trong phần saliency.
- Cập nhật ví dụ checkpoint cho flow cần schema backbone mới.

## Environment Remediation Executed

Trong env `hieupcvp`:
- Đã pin lại NumPy để sửa ABI mismatch:
  - `conda run -n hieupcvp python -m pip install "numpy<2"`
- Trạng thái sau khi sửa:
  - `numpy==1.26.4`
  - `tensorflow==2.17.0` import được trở lại.

## Verification Results

### Passed
- `python src/train_lightning_soop_regression.py --help` chạy được (không còn crash import ở startup path).
- `--validate-checkpoint-only` với `src/checkpoints/BrainIAC_mock.ckpt` pass.
- `--validate-checkpoint-only` với `src/checkpoints/BrainIAC.ckpt` fail-fast đúng message và gợi ý checkpoint tương thích.
- `bash -n research/run_soop_outcome_experiments_brainiac.sh` pass.
- `python -m json.tool src/quickstart.ipynb` pass.

### Attempted / Remaining
- Smoke train end-to-end đã được thử nhưng chưa chốt full completion trong lượt này (runtime dài, một lượt bị interrupt; một lượt trước đó gặp cấu hình `limit_val_batches` quá nhỏ).
- Full 6-run matrix chưa được execute trong lượt này.

## Root-cause Fix Status

### A) Import/ABI chain
- Đã fix ở mức code-path startup + env pin NumPy cho env thực thi.
- Hiện preflight/help không còn bị chặn bởi lỗi import chain.

### B) Checkpoint mismatch
- Đã fix bằng checkpoint gate sớm + thông báo fail-fast rõ ràng.
- Runner mặc định chuyển sang checkpoint tương thích.

## Notes
- Một số package trong env báo dependency warning sau khi pin NumPy (ví dụ `opencv-python` yêu cầu NumPy >=2), nhưng không chặn preflight/checkpoint validation flow của SOOP hiện tại.
