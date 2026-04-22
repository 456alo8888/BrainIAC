# BrainIAC SOOP Regression - Implemented

## Date
2026-03-20

## Scope Implemented
Đã triển khai các phase chính theo `research/experiment_plan.md` cho pipeline SOOP regression trên BrainIAC:
- Phase 1: SOOP data adapter + datamodule
- Phase 2: train script + config cho SOOP regression (image-only và image+tabular)
- Phase 3: eval script + artifacts chuẩn metrics/CSV
- Phase 4: runner script + experiment manifest
- Cập nhật checklist trong `experiment_plan.md` theo trạng thái verify thực tế

## Files Added
- `src/soop_dataset.py`
  - `SOOPRegressionDataset`
  - `SOOPRegressionDataModule`
  - Hỗ trợ target alias `gs_rankin+6isdeath`
  - Hỗ trợ image-only / image+tabular, normalize tabular, drop missing label

- `src/config_soop_regression.yml`
  - Config chuẩn cho SOOP regression

- `src/train_lightning_soop_regression.py`
  - Lightning module cho regression với backbone BrainIAC
  - Hỗ trợ fusion image+tabular
  - Metrics validation: `mse`, `rmse`, `mae`, `mape`, `r2`
  - CLI override phục vụ orchestration trong script runner

- `src/eval_soop_regression.py`
  - Evaluate checkpoint trên split CSV
  - Xuất `predictions.csv` và `results_eval_soop_regression.json`
  - Metrics JSON gồm: `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`

- `research/run_soop_outcome_experiments_brainiac.sh`
  - Runner cho 6 run (preprocessed + raw)
  - Gọi train + eval tuần tự
  - Có biến môi trường cho checkpoint/output/hyperparams/W&B

- `research/experiment_manifest_brainiac.md`
  - Manifest run matrix và expected artifacts

## Files Updated
- `README.md`
  - Thêm section “SOOP Regression Outcome (raw + preprocessed)”
- `research/experiment_plan.md`
  - Đánh dấu các automated checks đã hoàn tất cho Phase 1–3 và một phần Phase 4

## Environment Notes
- Theo yêu cầu, toàn bộ kiểm chứng runtime dùng env `hieupcvp`.
- Đã cài thêm dependency thiếu trong env:
  - `pytorch-lightning==2.3.3`

## Automated Verification Executed

### Phase 1
- Import dataset mới và load sample thành công.
- Instantiate được cho cả:
  - `datasets/fold`
  - `datasets/fold_raw_trace`
- Alias target + tabular path hoạt động (`gs_rankin+6isdeath`, tab_dim=7).

### Phase 2
- `train_lightning_soop_regression.py --help` chạy được.
- Smoke run image-only (preprocessed) hoàn tất, tạo checkpoint:
  - `/tmp/brainiac_soop_smoke_pre/checkpoints/best-model-epoch=00-val_mae=1.5400.ckpt`
- Smoke run image+tabular (raw) hoàn tất, tạo checkpoint:
  - `/tmp/brainiac_soop_smoke_raw_tab/checkpoints/best-model-epoch=00-val_mae=16.3999.ckpt`

### Phase 3
- Eval chạy thành công trên checkpoint smoke.
- Tạo artifacts:
  - `/tmp/brainiac_soop_smoke_raw_tab/eval/predictions.csv`
  - `/tmp/brainiac_soop_smoke_raw_tab/eval/results_eval_soop_regression.json`
- Kiểm tra metrics keys đầy đủ: `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`.
- Số dòng prediction khớp số mẫu test split (111/111 trong run raw smoke).

### Phase 4 (partial automated)
- `bash -n research/run_soop_outcome_experiments_brainiac.sh` pass.
- Đã kiểm chứng smoke cho cả nhánh preprocessed và raw (qua train/eval trực tiếp).

## Known Issues / Constraints
1. **Checkpoint BrainIAC pretrained chưa có sẵn trong workspace**
   - Không tìm thấy file thật ở path mặc định trong config.
   - Để unblock smoke verification, đã tạo checkpoint mock tương thích format tại:
     - `src/checkpoints/BrainIAC_mock.ckpt`
   - Cần thay bằng checkpoint BrainIAC thật trước khi chạy production experiments.

2. **GPU OOM trên GPU 0 ở lần thử đầu**
   - Đã chuyển smoke sang CPU và/hoặc giảm batch/limit batches để hoàn tất kiểm chứng.

3. **Runner full 6 runs chưa được execute end-to-end trong lượt này**
   - Script đã sẵn sàng, syntax pass.
   - Cần chạy full matrix khi có checkpoint pretrained thật và tài nguyên GPU phù hợp.

## Next Execution Command (Suggested)
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/BrainIAC
BRAINIAC_CKPT=/path/to/REAL_BrainIAC.ckpt \
CUDA_DEVICE=0 BATCH_SIZE=4 EPOCHS=10 \
bash research/run_soop_outcome_experiments_brainiac.sh
```

## Manual Verification Pending
Các checklist manual trong `research/experiment_plan.md` vẫn để trống theo đúng quy trình, chờ xác nhận từ người dùng sau khi chạy/đánh giá thực nghiệm thực tế.
