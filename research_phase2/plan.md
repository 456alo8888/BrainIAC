# BrainIAC Image-only SOOP: Implementation Plan (Test Eval + Live Runtime Logs)

## Overview

Kế hoạch này xử lý đúng 2 vấn đề đã xác định trong `research_phase2/research.md`:
1. Script image-only hiện chỉ train, chưa evaluate trên test.
2. Runtime log trên terminal có thể không hiển thị realtime khi chạy qua `conda run`.

Mục tiêu là cập nhật luồng image-only để sau khi train sẽ tự evaluate trên `test.csv`, đồng thời hiển thị log runtime trực tiếp trên terminal và vẫn lưu log file đầy đủ.

## Current State Analysis

- `src/bash_image_only.sh` chỉ gọi `train_lightning_soop_regression.py` rồi kết thúc (không gọi eval).
- `src/train_lightning_soop_regression.py` dừng ở `trainer.fit(...)`, sau đó chỉ in `Best checkpoint` và `Resolved config`.
- `src/eval_soop_regression.py` là entrypoint riêng để tạo:
  - `predictions.csv`
  - `results_eval_soop_regression.json`
- `research/run_soop_outcome_experiments_brainiac.sh` đã có pattern train -> eval trên `test.csv`.
- Theo `conda run --help`, muốn stream stdout/stderr realtime cần `--no-capture-output` (hoặc `--live-stream`).

## Desired End State

Sau khi hoàn thành plan:
- Chạy `src/bash_image_only.sh` sẽ tạo đầy đủ artifacts train + test eval trong cùng output dir.
- Log runtime hiển thị realtime trên terminal trong suốt quá trình train + eval.
- Vẫn có log file timestamp để truy vết (`src/logs/...log`).
- Luồng chạy giữ nguyên môi trường `hieupcvp`.

### Key Discoveries
- Train-only path hiện tại nằm ở `src/bash_image_only.sh` và dừng sau lệnh train.
- Eval output path đã chuẩn hóa trong `src/eval_soop_regression.py`.
- Runner trong `research/` đã có mẫu tách train/eval có thể tái dùng cho image-only.

## What We’re NOT Doing

- Không thay đổi kiến trúc model hoặc hàm loss/metrics trong `train_lightning_soop_regression.py`.
- Không thay đổi schema dataset/split CSV.
- Không mở rộng sang image+tabular hoặc full 6-run matrix trong kế hoạch này.
- Không refactor toàn bộ hệ logging của project; chỉ chỉnh luồng script image-only.

## Implementation Approach

Áp dụng hướng sửa tối thiểu, tái dùng thành phần đã có:
1. Giữ train script như hiện tại.
2. Nâng cấp `src/bash_image_only.sh` thành script train+eval nối tiếp.
3. Thêm cơ chế live stream cho `conda run` và chuẩn hóa log file train/eval.
4. Đồng bộ hướng dẫn chạy để tránh lệch giữa behavior và kỳ vọng.

## Execution Status (2026-03-20)

- [x] Đã implement Phase 1 phần orchestration train -> test eval trong `src/bash_image_only.sh`.
- [x] Đã bổ sung logging test metrics/artifacts lên W&B trong `src/eval_soop_regression.py`.
- [x] Đã giữ nguyên hướng xử lý terminal log theo cập nhật mới nhất từ user (đã bỏ `conda run` trong script bash).
- [ ] Chưa chạy full smoke train+eval end-to-end trong lượt này.
- [ ] Manual verification còn pending.

---

## Phase 1: Nâng cấp image-only script thành train + eval

### Overview
Thêm bước evaluate test ngay sau train trong `src/bash_image_only.sh`.

### Changes Required:

#### 1) Cập nhật orchestration trong script image-only
**File**: `src/bash_image_only.sh`
**Changes**:
- Giữ train command hiện tại (hyperparameters, output dir, wandb options).
- Sau train, lấy checkpoint tốt nhất từ thư mục `--output-dir/.../checkpoints`.
- Gọi `src/eval_soop_regression.py` với:
  - `--config <output_dir>/resolved_config_soop_regression.yml`
  - `--checkpoint <best_ckpt>`
  - `--split-csv /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold/test.csv`
  - `--output-dir <output_dir>/eval`
  - `--target-col gs_rankin_6isdeath`
  - `--no-include-tabular`
- Log rõ từng stage: `[TRAIN]`, `[EVAL]`, `[DONE]`.

**Status**: [x] Implemented

```bash
# Pseudocode luồng mới
run_train
best_ckpt=$(ls -1 "$OUTPUT_DIR"/checkpoints/*.ckpt | head -n 1)
run_eval --checkpoint "$best_ckpt" --split-csv "$FOLD_DIR/test.csv" --output-dir "$OUTPUT_DIR/eval"
```

### Success Criteria:

#### Automated Verification:
- [x] Script syntax pass: `bash -n src/bash_image_only.sh`
- [ ] Train + eval chạy được trong env `hieupcvp` (smoke):
  - `CUDA_DEVICE=<id> bash src/bash_image_only.sh`
- [ ] Có file sau khi chạy:
  - `<output_dir>/resolved_config_soop_regression.yml`
  - `<output_dir>/checkpoints/*.ckpt`
  - `<output_dir>/eval/predictions.csv`
  - `<output_dir>/eval/results_eval_soop_regression.json`
- [ ] JSON metrics có đủ keys: `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`

#### Manual Verification:
- [ ] Mở `predictions.csv` và kiểm tra có dữ liệu test.
- [ ] Mở JSON và xác nhận số mẫu `n_samples` hợp lý.
- [ ] Xác nhận script dừng fail-fast nếu không tìm thấy checkpoint.

**Implementation Note**: Kết thúc phase này cần xác nhận thủ công rằng artifacts test đã sinh đúng trước khi chuyển phase 2.

---

## Phase 2: Bật live runtime logs trên terminal

### Overview
Đảm bảo log hiển thị realtime trên terminal khi dùng `conda run`.

**Status**: [x] Addressed by user outside this implementation pass (đã bỏ `conda run` trong `src/bash_image_only.sh`).

### Changes Required:

#### 1) Bổ sung live-stream option
**File**: `src/bash_image_only.sh`
**Changes**:
- Chuyển các lệnh `conda run -n hieupcvp ...` sang:
  - `conda run --no-capture-output -n hieupcvp ...`
- Giữ `PYTHONUNBUFFERED=1` + `python -u` cho train path.
- Duy trì `2>&1 | tee -a "$LOG_FILE"` cho cả train và eval để vừa stream terminal vừa lưu file.

#### 2) Chuẩn hóa log file cho cả train/eval
**File**: `src/bash_image_only.sh`
**Changes**:
- Một log file chung theo timestamp cho toàn bộ run (train + eval).
- Ghi marker thời gian bắt đầu/kết thúc từng stage.

```bash
echo "[TRAIN][$(date -Iseconds)] start" | tee -a "$LOG_FILE"
...
echo "[EVAL][$(date -Iseconds)] start" | tee -a "$LOG_FILE"
```

### Success Criteria:

#### Automated Verification:
- [ ] Khi chạy script, terminal hiển thị liên tục progress train/val theo epoch.
- [ ] Log file chứa cả phần train và phần eval trong cùng run.
- [ ] Không mất behavior cũ: vẫn ghi được đường dẫn checkpoint và resolved config.

#### Manual Verification:
- [ ] Quan sát trực tiếp terminal thấy realtime output (không dồn cuối run).
- [ ] Xác nhận log không bị thiếu phần eval.

**Implementation Note**: Sau phase này, cần human xác nhận runtime visibility đã đạt kỳ vọng khi chạy thực tế.

---

## Phase 3: Đồng bộ tài liệu vận hành

### Overview
Cập nhật docs để hành vi script khớp mô tả và tránh chạy nhầm train-only.

### Changes Required:

#### 1) Cập nhật hướng dẫn chạy
**File**: `research/run_guide.md`
**Changes**:
- Thêm mục cho `src/bash_image_only.sh` mới (train+test eval).
- Ghi rõ env bắt buộc: `hieupcvp`.
- Ghi rõ expected artifacts gồm thư mục `eval/`.

#### 2) Cập nhật ghi chú trong research_phase2
**File**: `research_phase2/research.md` (append short update section)
**Changes**:
- Bổ sung trạng thái sau fix (nếu đã chạy smoke).
- Ghi đường dẫn log/artifacts xác thực.

### Success Criteria:

#### Automated Verification:
- [ ] Các command docs copy-paste chạy được với env `hieupcvp`.
- [ ] Đường dẫn artifacts trong docs tồn tại sau smoke run.

#### Manual Verification:
- [ ] Người vận hành mới đọc docs có thể chạy đúng 1 lệnh image-only và thu đủ train + test artifacts.

---

## Testing Strategy

### Unit Tests:
- Không thêm unit test mới trong scope này (thay đổi chủ yếu ở orchestration shell).

### Integration Tests:
- Smoke integration (nhỏ):
  1. Chạy `src/bash_image_only.sh` với `limit_train/val_batches` nhỏ.
  2. Verify checkpoint được sinh.
  3. Verify eval chạy trên `fold/test.csv` và sinh JSON/CSV.

### Manual Testing Steps:
1. Chạy script và quan sát terminal realtime.
2. Mở log file timestamp vừa tạo, xác nhận có cả `[TRAIN]` và `[EVAL]`.
3. Kiểm tra `eval/results_eval_soop_regression.json` có metrics đầy đủ.
4. Kiểm tra `eval/predictions.csv` có cột `subject_id,target,prediction,...`.

## Performance Considerations

- Live stream có thể làm terminal nhiều output hơn nhưng không thay đổi logic train.
- Eval test chạy sau train sẽ tăng tổng runtime mỗi lần gọi script image-only.
- Với smoke/CI nên giữ `limit_*_batches` nhỏ để phản hồi nhanh.

## Migration Notes

- Không có migration dữ liệu.
- Người dùng đang chạy script cũ sẽ thấy behavior mới: có thêm bước evaluate test ngay sau train.

## References

- `research_phase2/research.md`
- `research/implemented_fix.md`
- `research/run_guide.md`
- `src/bash_image_only.sh`
- `src/train_lightning_soop_regression.py`
- `src/eval_soop_regression.py`
- `research/run_soop_outcome_experiments_brainiac.sh`
