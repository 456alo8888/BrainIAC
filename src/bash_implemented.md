# Bash Scripts Implemented for SOOP Outcome (`gs_rankin_6isdeath`)

## Date
2026-03-20

## Scope
Đã tạo 4 bash scripts theo đúng ma trận thí nghiệm yêu cầu trong `research/`:
- preprocessed + image-only
- preprocessed + image+tabular
- raw + image-only
- raw + image+tabular

Mỗi script đều follow pattern của `src/bash_image_only.sh` hiện tại:
1. train bằng `train_lightning_soop_regression.py`
2. tự lấy checkpoint tốt nhất
3. evaluate test bằng `eval_soop_regression.py`
4. log toàn bộ train/eval ra file `src/logs/*.log`
5. log lên W&B cho cả train/val và test (test qua eval script)

## Files Added

1. `src/bash_preprocessed_image_only.sh`
   - Fold: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold`
   - Mode: `--no-include-tabular`
   - Output: `outputs/soop_smoke_fix/preprocessed_image_only`
   - Run name: `soop-gsrankin-preprocessed-image-only`

2. `src/bash_preprocessed_image_tabular.sh`
   - Fold: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold`
   - Mode: `--include-tabular`
   - Output: `outputs/soop_smoke_fix/preprocessed_image_tabular`
   - Run name: `soop-gsrankin-preprocessed-image-tabular`

3. `src/bash_raw_image_only.sh`
   - Fold: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace`
   - Mode: `--no-include-tabular`
   - Output: `outputs/soop_smoke_fix/raw_image_only`
   - Run name: `soop-gsrankin-raw-image-only`

4. `src/bash_raw_image_tabular.sh`
   - Fold: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace`
   - Mode: `--include-tabular`
   - Output: `outputs/soop_smoke_fix/raw_image_tabular`
   - Run name: `soop-gsrankin-raw-image-tabular`

## Common behavior in all scripts

- Target cố định: `gs_rankin_6isdeath`
- Checkpoint: `checkpoints/BrainIAC_mock.ckpt`
- Sau train có kiểm tra fail-fast:
  - checkpoint tồn tại trong `<output_dir>/checkpoints`
  - `resolved_config_soop_regression.yml` tồn tại
- Eval dùng test split tương ứng:
  - preprocessed: `<fold>/test.csv`
  - raw: `<fold_raw_trace>/test.csv`
- Stage markers trong log:
  - `[TRAIN] start/done`
  - `[EVAL] start/done`
  - `[DONE] train+eval completed`

## Verification performed

Đã chạy syntax check cho cả 4 script:
- `bash -n src/bash_preprocessed_image_only.sh`
- `bash -n src/bash_preprocessed_image_tabular.sh`
- `bash -n src/bash_raw_image_only.sh`
- `bash -n src/bash_raw_image_tabular.sh`

Tất cả pass.

## How to run

Từ thư mục `baseline_encoder/BrainIAC` (đảm bảo đang ở env `hieupcvp`):

```bash
bash src/bash_preprocessed_image_only.sh
bash src/bash_preprocessed_image_tabular.sh
bash src/bash_raw_image_only.sh
bash src/bash_raw_image_tabular.sh
```

## Expected artifacts per run

- `<output_dir>/resolved_config_soop_regression.yml`
- `<output_dir>/checkpoints/*.ckpt`
- `<output_dir>/eval/predictions.csv`
- `<output_dir>/eval/results_eval_soop_regression.json`
- `src/logs/*.log`
