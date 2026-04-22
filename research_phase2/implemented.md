# BrainIAC research_phase2 - Implemented

## Date
2026-03-20

## Scope implemented in this pass
Theo yêu cầu `/implement_plan`, tập trung hoàn thành phần:
1. thêm bước evaluate trên tập test cho luồng image-only,
2. đảm bảo train/val/test có thể theo dõi trên W&B,
3. cập nhật lại `research_phase2/plan.md` theo trạng thái thực thi hiện tại.

Lưu ý: phần terminal live log được giữ theo cập nhật mới nhất của user (đã bỏ `conda run` trong bash script).

## Files updated

### 1) `src/bash_image_only.sh`
Đã nâng cấp từ train-only sang train+eval:
- Thêm biến chuẩn hóa cho `FOLD_DIR`, `OUTPUT_DIR`, `RUN_NAME`, `TARGET_COL`.
- Thêm stage markers vào log:
  - `[TRAIN] start/done`
  - `[EVAL] start/done`
  - `[DONE] train+eval completed`
- Sau train:
  - tự tìm best checkpoint từ `<output_dir>/checkpoints/*.ckpt`
  - fail-fast nếu không tìm thấy checkpoint hoặc thiếu `resolved_config_soop_regression.yml`
- Thêm gọi eval test:
  - `python eval_soop_regression.py`
  - dùng `--split-csv "$FOLD_DIR/test.csv"`
  - output `--output-dir "$OUTPUT_DIR/eval"`
  - truyền `--use-wandb`, `--project-name`, `--run-name "${RUN_NAME}-test"`

Kết quả: chạy `src/bash_image_only.sh` sẽ thực hiện train rồi evaluate test trong cùng pipeline.

### 2) `src/eval_soop_regression.py`
Đã thêm W&B logging cho pha test:
- CLI mới:
  - `--use-wandb` / `--no-use-wandb`
  - `--project-name`
  - `--run-name`
- Giữ cơ chế mặc định theo config nếu không truyền cờ CLI.
- Khi `use_wandb=True`:
  - `wandb.init(...)`
  - log metrics test với prefix `test_` (`test_mse`, `test_rmse`, `test_mae`, `test_mape`, `test_r2`, `test_loss`)
  - log `test_n_samples`
  - ghi summary path cho `predictions.csv` và `results_eval_soop_regression.json`
  - upload artifact evaluation chứa 2 file output
  - `run.finish()`

Kết quả: ngoài train/val đã có từ Lightning + `WandbLogger`, giờ test metrics/artifacts cũng được đẩy lên W&B qua eval script.

### 3) `research_phase2/plan.md`
Đã cập nhật trạng thái kế hoạch:
- thêm mục `Execution Status (2026-03-20)`
- đánh dấu đã implement Phase 1 (orchestration train->eval)
- đánh dấu Phase 2 đã được user xử lý riêng (bỏ `conda run`)
- check `Script syntax pass` trong automated verification
- giữ các mục smoke end-to-end và manual verification ở trạng thái pending

## Verification executed (env: `hieupcvp`)

### Passed
- `bash -n src/bash_image_only.sh`
- `conda run -n hieupcvp python src/eval_soop_regression.py --help`
- `conda run -n hieupcvp python src/train_lightning_soop_regression.py --help | head -n 40`
- `conda run -n hieupcvp python -m py_compile src/eval_soop_regression.py`

### Pending in this pass
- Chưa chạy full smoke `bash src/bash_image_only.sh` để xác nhận end-to-end artifacts thực tế do runtime dài.
- Chưa thực hiện manual verification checklist trong plan.

## Expected artifacts after running updated image-only script
- `outputs/soop_smoke_fix/preprocessed_image_only/resolved_config_soop_regression.yml`
- `outputs/soop_smoke_fix/preprocessed_image_only/checkpoints/*.ckpt`
- `outputs/soop_smoke_fix/preprocessed_image_only/eval/predictions.csv`
- `outputs/soop_smoke_fix/preprocessed_image_only/eval/results_eval_soop_regression.json`

## Notes
- Script hiện vẫn giữ `WANDB_API_KEY` hard-code như trạng thái trước đó, không thay đổi trong lượt implement này.
- Không thay đổi kiến trúc model hoặc metric logic train/val.
