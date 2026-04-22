# BrainIAC SOOP Regression Outcome Experiment Plan

## Overview

Mục tiêu là triển khai pipeline thí nghiệm regression outcome trên BrainIAC cho **2 bộ dữ liệu**:
1. **Preprocessed TRACE split**: `code/datasets/fold`
2. **Raw TRACE split**: `code/datasets/fold_raw_trace`

và chạy bộ thí nghiệm tương tự 3DINO (image-only và image+tabular; target `gs_rankin_6isdeath` và `nihss`) theo pattern trong:
- `baseline_encoder/3DINO/research/experiment_manifest.md`
- `baseline_encoder/3DINO/research/experiment_manifest_raw.md`
- `baseline_encoder/3DINO/research/run_soop_outcome_experiments.sh`
- `baseline_encoder/3DINO/research/run_soop_outcome_experiments_raw.sh`

---

## Current State Analysis

### BrainIAC hiện có
- Regression script hiện tại: `baseline_encoder/BrainIAC/src/train_lightning_brainage.py`
- Dataset loader hiện tại: `baseline_encoder/BrainIAC/src/dataset.py`
  - `BrainAgeDataset` đọc schema `pat_id,label` và tự nối path `root_dir/pat_id.nii.gz`
- Inference script tổng: `baseline_encoder/BrainIAC/src/test_inference_finetune.py`
  - Có regression metrics (MAE/RMSE/R2), nhưng cấu hình dataset/task hard-coded theo task cũ.

### Dataset SOOP cho thí nghiệm
- Split preprocessed: `datasets/fold/{train,valid,test}.csv`
- Split raw: `datasets/fold_raw_trace/{train,valid,test}.csv`
- Thống kê split:
  - preprocessed: 659 subjects (`datasets/fold/split_summary.json`)
  - raw: 738 subjects (`datasets/fold_raw_trace/split_summary.json`)
- Cột target và feature phù hợp regression outcome:
  - target: `gs_rankin_6isdeath`, `nihss` (và alias `gs_rankin+6isdeath`)
  - tabular features: `sex, age, race, acuteischaemicstroke, priorstroke, bmi, etiology`

### Khoảng trống cần lấp
- BrainIAC chưa có dataset class đọc trực tiếp SOOP split schema (`image_path`, tabular columns).
- Chưa có train script chuyên cho SOOP outcome với tùy chọn `image-only` / `image+tabular`.
- Chưa có runner thống nhất cho 6 run (3 preprocessed + 3 raw) giống 3DINO.

---

## Desired End State

Sau khi hoàn thành kế hoạch:
1. BrainIAC có pipeline train/eval regression cho SOOP split schema mà không cần convert CSV thủ công.
2. Chạy được đầy đủ 6 thí nghiệm:
   - preprocessed image-only `gs_rankin_6isdeath`
   - preprocessed image+tabular `gs_rankin_6isdeath`
   - preprocessed image+tabular `nihss`
   - raw image-only `gs_rankin_6isdeath`
   - raw image+tabular `gs_rankin_6isdeath`
   - raw image+tabular `nihss`
3. Mỗi run sinh đủ artifact:
   - checkpoint tốt nhất
   - prediction CSV
   - metrics JSON với: `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`
4. Có manifest + bash runner để tái lập kết quả.

### Key Discoveries
- `datasets/SOOP_dataset.py` đã chuẩn hóa logic đọc NIfTI + tabular + target + label mask, có thể tái dùng trực tiếp hoặc mô phỏng interface.
- `preprocess_MRI/create_dataset/build_trace_splits.py` và `build_raw_trace_splits.py` đảm bảo split đã stratified và có cột target đầy đủ.
- Pattern orchestration ở 3DINO đã ổn định và có thể copy cấu trúc run/manifest.

---

## What We’re NOT Doing

- Không thay đổi pretraining BrainIAC backbone.
- Không chỉnh pipeline segmentation/classification khác SOOP outcome.
- Không chạy hyperparameter sweep lớn trong phase đầu (chỉ fixed profile + 1 baseline profile tùy chọn).
- Không thay đổi dữ liệu nguồn trong `/mnt/disk2/SOOP_TRACE_preprocessed` hoặc `/mnt/disk2/SOOP_dataset/ds004889-download`.

---

## Implementation Approach

- Dùng chiến lược **additive**: thêm module mới cho SOOP thay vì sửa mạnh luồng cũ.
- Tối ưu cho reproducibility:
  - config file riêng cho SOOP,
  - runner script có env vars,
  - metrics schema nhất quán với 3DINO research.
- Mỗi phase có gate kiểm thử rõ ràng (automated + manual).

---

## Phase 1: SOOP Data Adapter cho BrainIAC

### Overview
Thêm lớp dataset đọc trực tiếp split CSV của SOOP cho single-image regression, hỗ trợ kèm tabular.

### Changes Required

#### 1) Tạo dataset module mới
**File**: `baseline_encoder/BrainIAC/src/soop_dataset.py` (new)

**Changes**:
- Đọc CSV split (`train.csv`, `valid.csv`, `test.csv`) với cột:
  - `subject_id`, `image_path`, `gs_rankin_6isdeath`, `nihss`, tabular columns
- Resolve target col (`gs_rankin_6isdeath`, `nihss`, alias `gs_rankin+6isdeath`)
- Trả sample chuẩn cho Lightning:
  - image-only: `{"image", "label", "subject_id"}`
  - image+tabular: `{"image", "tabular", "label", "subject_id"}`
- Có cờ:
  - `include_tabular`
  - `normalize_tabular`
  - `drop_missing_label`

```python
sample = {
    "image": transformed_image,
    "label": torch.tensor(target_value, dtype=torch.float32),
    "subject_id": subject_id,
}
if include_tabular:
    sample["tabular"] = torch.tensor(tabular_vector, dtype=torch.float32)
```

#### 2) Tạo DataModule riêng cho SOOP
**File**: `baseline_encoder/BrainIAC/src/soop_dataset.py` (same file) hoặc `src/soop_datamodule.py` (new)

**Changes**:
- `SOOPRegressionDataModule` nhận:
  - `fold_dir`
  - `target_col`
  - `include_tabular`
  - `batch_size`, `num_workers`, `image_size`
- Tạo train/val/test dataloaders tương ứng `train.csv/valid.csv/test.csv`.

### Success Criteria

#### Automated Verification:
- [x] Import pass:
  - `python -c "from soop_dataset import SOOPRegressionDataset; print('ok')"`
- [x] Có thể instantiate cho cả 2 fold dir:
  - `datasets/fold`
  - `datasets/fold_raw_trace`
- [x] Một batch trả đúng keys và shape (image-only vs image+tabular).

#### Manual Verification:
- [ ] Đối chiếu ngẫu nhiên 5 dòng CSV với sample từ dataset class.
- [ ] Xác nhận target mapping đúng khi dùng alias `gs_rankin+6isdeath`.

**Implementation Note**: Dừng sau phase này để xác nhận data adapter đúng trước khi train.

---

## Phase 2: BrainIAC Regression Trainer cho SOOP

### Overview
Thêm script huấn luyện regression sử dụng backbone BrainIAC, hỗ trợ image-only và image+tabular.

### Changes Required

#### 1) Script train mới
**File**: `baseline_encoder/BrainIAC/src/train_lightning_soop_regression.py` (new)

**Changes**:
- Kế thừa pattern từ `train_lightning_brainage.py`.
- Model options:
  - `image-only`: backbone feature (768) -> regression head
  - `image+tabular`: concat `[img_feat, tabular_feat]` -> MLP regression head
- Loss: `MSELoss`
- Log metrics mỗi epoch:
  - `val_mse`, `val_rmse`, `val_mae`, `val_mape`, `val_r2`
- Checkpoint monitor:
  - default `val_mae` (mode `min`)

```python
if include_tabular:
    fused = torch.cat([img_feat, tabular], dim=1)
    pred = self.reg_head(fused)
else:
    pred = self.reg_head(img_feat)
loss = mse_loss(pred, y)
```

#### 2) Config file riêng cho SOOP
**File**: `baseline_encoder/BrainIAC/src/config_soop_regression.yml` (new)

**Changes**:
- Các nhóm config chính:
  - `data`: `fold_dir`, `target_col`, `include_tabular`, `batch_size`, `num_workers`, `size`
  - `model`: `max_epochs`, `dropout`, `hidden_dim_tabular_head`
  - `simclrvit`: `ckpt_path`
  - `optim`: `lr`, `weight_decay`, `optimizer`
  - `logger`: `project_name`, `run_name`, `save_dir`, `save_name`
  - `train`: `freeze_backbone`, `precision`, `devices`, `accelerator`

#### 3) Bổ sung lựa chọn training profile ổn định
**File**: `baseline_encoder/BrainIAC/src/train_lightning_soop_regression.py`

**Changes**:
- Tùy chọn `optimizer=adamw` mặc định cho regression.
- Thêm `grad_clip_norm` trong trainer.
- Optional feature normalization trước regression head.

### Success Criteria

#### Automated Verification:
- [x] `python src/train_lightning_soop_regression.py --help` chạy được.
- [x] Smoke run (1 epoch) với `datasets/fold` + image-only hoàn tất và sinh checkpoint.
- [x] Smoke run (1 epoch) với `datasets/fold_raw_trace` + image+tabular hoàn tất.

#### Manual Verification:
- [ ] Log train/val không bị divergence bất thường ở 20-50 iter đầu.
- [ ] Model load được BrainIAC ckpt đúng (không random init ngoài ý muốn).

**Implementation Note**: Dừng để xác nhận train stability trước khi mở rộng eval/runner.

---

## Phase 3: Inference + Metrics chuẩn SOOP

### Overview
Tạo script đánh giá riêng cho SOOP regression để xuất prediction và metrics đồng nhất với 3DINO research.

### Changes Required

#### 1) Eval script mới
**File**: `baseline_encoder/BrainIAC/src/eval_soop_regression.py` (new)

**Changes**:
- Input:
  - model checkpoint
  - split CSV (`test.csv`)
  - mode image-only / image+tabular
- Output:
  - `predictions.csv`
  - `results_eval_soop_regression.json`
- Metrics bắt buộc:
  - `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`

#### 2) Chuẩn hóa schema output
**File**: `baseline_encoder/BrainIAC/src/eval_soop_regression.py`

**Changes**:
- JSON có block:
  - `run_info`
  - `data_config`
  - `metrics`
  - `n_samples`
- CSV có cột:
  - `subject_id`, `target`, `prediction`, `abs_error`, `squared_error`

### Success Criteria

#### Automated Verification:
- [x] Eval script chạy thành công trên checkpoint smoke run.
- [x] JSON có đủ 6 metric bắt buộc.
- [x] CSV prediction có số dòng bằng số mẫu test split.

#### Manual Verification:
- [ ] Spot-check 10 sample: `subject_id` đúng với `test.csv`.
- [ ] Metric values hợp lý (không NaN/Inf).

**Implementation Note**: Dừng để xác nhận output schema trước khi chạy full experiment matrix.

---

## Phase 4: Runner + Manifest cho 2 bộ dữ liệu

### Overview
Tạo bộ command tái lập thí nghiệm tương tự 3DINO, gồm raw + preprocessed.

### Changes Required

#### 1) Runner script
**File**: `baseline_encoder/BrainIAC/research/run_soop_outcome_experiments_brainiac.sh` (new)

**Changes**:
- Dùng env vars:
  - `BRAINIAC_CKPT`
  - `CUDA_DEVICE`
  - `USE_WANDB`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_MODE`, `WANDB_TAGS`
  - `BATCH_SIZE`, `EPOCHS`
- Chạy tuần tự 6 run:
  - 3 run trên `datasets/fold`
  - 3 run trên `datasets/fold_raw_trace`
- Mỗi run đặt output dir riêng, ví dụ:
  - `BrainIAC/outputs/soop_gsrankin_image_only`
  - `BrainIAC/outputs/soop_raw_nihss_image_tabular`

#### 2) Manifest thí nghiệm
**File**: `baseline_encoder/BrainIAC/research/experiment_manifest_brainiac.md` (new)

**Changes**:
- Ghi đầy đủ run matrix, output paths, config, seed, expected artifacts.
- Mapping để so sánh trực tiếp với 3DINO manifest.

### Success Criteria

#### Automated Verification:
- [x] `bash -n research/run_soop_outcome_experiments_brainiac.sh` pass.
- [x] Chạy được 1 vòng smoke cho cả preprocessed và raw.
- [ ] Mỗi run tạo đủ checkpoint + prediction CSV + metrics JSON.

#### Manual Verification:
- [ ] Tên run/output rõ ràng, không ghi đè nhau.
- [ ] W&B run tags phân biệt được raw/preprocessed và image-only/tabular.

**Implementation Note**: Sau phase này, dừng để human xác nhận setup before full long-run.

---

## Phase 5: Full Experiment Execution & Result Table

### Overview
Thực thi toàn bộ matrix với cấu hình production và tổng hợp kết quả thành bảng so sánh.

### Changes Required

#### 1) Chạy full 6 run
**Execution**:
- Production epochs (theo resource thực tế GPU).
- Lưu log + artifacts về thư mục output chuẩn.

#### 2) Tổng hợp kết quả
**File**: `baseline_encoder/BrainIAC/research/brainiac_soop_results_summary.md` (new)

**Changes**:
- Bảng metrics theo từng run:
  - MSE, RMSE, MAE, MAPE, R2
- So sánh:
  - preprocessed vs raw
  - image-only vs image+tabular
  - BrainIAC vs 3DINO (nếu có kết quả 3DINO tương ứng)

### Success Criteria

#### Automated Verification:
- [ ] Đủ 6 JSON kết quả hợp lệ và parse được bằng script tổng hợp.
- [ ] Không có run fail/corrupt artifact.

#### Manual Verification:
- [ ] Bảng tổng hợp phản ánh đúng output files.
- [ ] Nhận định ban đầu về lợi ích tabular và ảnh hưởng raw/preprocessed có cơ sở định lượng.

---

## Phase 6: Documentation & Handover

### Overview
Chuẩn hóa tài liệu để người khác có thể tái lập toàn bộ pipeline BrainIAC outcome.

### Changes Required

#### 1) Cập nhật docs BrainIAC
**File**: `baseline_encoder/BrainIAC/README.md`

**Changes**:
- Thêm section “SOOP Regression Outcome (raw + preprocessed)”
- Dẫn command runner + manifest + vị trí outputs.

#### 2) Báo cáo triển khai
**File**: `baseline_encoder/BrainIAC/research/implemented_soop_regression.md` (new)

**Changes**:
- Ghi file nào đã thêm/sửa, lệnh đã chạy, kết quả smoke/full-run, known issues.

### Success Criteria

#### Automated Verification:
- [ ] Command trong README copy-paste chạy được (ít nhất ở smoke mode).
- [ ] Tất cả đường dẫn tài liệu/research hợp lệ.

#### Manual Verification:
- [ ] Một thành viên khác trong team có thể follow docs và chạy được 1 run độc lập.

---

## Testing Strategy

### Unit Tests
- Dataset parsing test:
  - đọc đúng target col,
  - xử lý missing labels,
  - tabular vector đúng chiều.
- Metric helper test:
  - tính đúng `mse/rmse/mae/mape/r2` trên tensor mẫu.

### Integration Tests
- Smoke train/eval cho 4 case tối thiểu:
  1. preprocessed image-only gsrankin
  2. preprocessed image+tabular nihss
  3. raw image-only gsrankin
  4. raw image+tabular nihss

### Manual Testing Steps
1. Kiểm tra ngẫu nhiên file NIfTI từ `image_path` mở được.
2. Verify 1 sample end-to-end từ DataLoader -> model -> prediction row.
3. So khớp số mẫu test giữa CSV split và prediction CSV.
4. So khớp metrics JSON bằng script recompute nhanh.

---

## Performance Considerations

- Ưu tiên mixed precision (`16-mixed`) để giảm VRAM.
- Batch size điều chỉnh theo GPU thực tế; giữ nhất quán giữa raw/preprocessed để so sánh công bằng.
- Tabular fusion head giữ nhỏ (1-2 FC layers) để tránh overfitting khi số mẫu hạn chế.

---

## Risks & Mitigations

1. **Mismatch schema giữa split CSV và BrainIAC loader**
   - Mitigation: adapter riêng `soop_dataset.py`, test parse ngay Phase 1.
2. **Divergence regression ở early iterations**
   - Mitigation: AdamW + grad clipping + feature norm profile.
3. **Data path inaccessible hoặc NIfTI lỗi**
   - Mitigation: preflight check script trước train (verify path tồn tại + load thử 20 sample).
4. **Khó so sánh với 3DINO do metrics format khác**
   - Mitigation: đồng nhất output schema trong `results_eval_soop_regression.json`.

---

## References

- BrainIAC core:
  - `baseline_encoder/BrainIAC/src/train_lightning_brainage.py`
  - `baseline_encoder/BrainIAC/src/dataset.py`
  - `baseline_encoder/BrainIAC/src/test_inference_finetune.py`
  - `baseline_encoder/BrainIAC/src/config_finetune.yml`
- BrainIAC research context:
  - `baseline_encoder/BrainIAC/research/research.md`
- SOOP dataset and split builders:
  - `datasets/SOOP_dataset.py`
  - `preprocess_MRI/create_dataset/build_trace_splits.py`
  - `preprocess_MRI/create_dataset/build_raw_trace_splits.py`
  - `datasets/fold/split_summary.json`
  - `datasets/fold_raw_trace/split_summary.json`
- 3DINO reference experiments:
  - `baseline_encoder/3DINO/research/experiment_manifest.md`
  - `baseline_encoder/3DINO/research/experiment_manifest_raw.md`
  - `baseline_encoder/3DINO/research/run_soop_outcome_experiments.sh`
  - `baseline_encoder/3DINO/research/run_soop_outcome_experiments_raw.sh`
