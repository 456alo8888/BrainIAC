# BrainIAC SOOP Bug Fix Plan

## Overview
Kế hoạch này nhằm sửa triệt để lỗi khi chạy `research/run_soop_outcome_experiments_brainiac.sh` gồm 2 nhóm: (1) lỗi môi trường NumPy/TensorBoard/TensorFlow và (2) lỗi không tương thích checkpoint backbone với kiến trúc `ViT` hiện tại.

## Current State Analysis
Theo `research/research_bug.md`, lỗi hiện tại là chồng 2 tầng:

1. **Môi trường**: `numpy==2.4.3` xung đột ABI với module build theo NumPy 1.x trong chain `torch.utils.tensorboard -> tensorboard.compat -> tensorflow`.
2. **Checkpoint**: `src/model.py` load backbone với `strict=True`, nhưng `BrainIAC.ckpt` thiếu toàn bộ key `cross_attn/norm_cross_attn`.

### Key discoveries
- `src/model.py` khởi tạo `monai.networks.nets.ViT(..., save_attn=True)` và `load_state_dict(..., strict=True)`.
- `src/train_lightning_soop_regression.py` dùng `pl.Trainer(logger=None)` khi `--no-use-wandb`, nhưng Lightning vẫn fallback TensorBoard logger mặc định.
- `README.md` + `src/quickstart.ipynb` đang tham chiếu cả `BrainIAC_mock.ckpt` (feature extraction) và `BrainIAC.ckpt` (saliency).

### Checkpoint inventory đã kiểm tra trong `src/checkpoints`
- `BrainIAC_mock.ckpt`: `total=221`, `backbone=221`, `cross=84`  -> **khớp schema backbone mới**.
- `BrainIAC.ckpt`: `total=149`, `backbone=137`, `cross=0` -> **không khớp schema backbone mới**.
- `brainage.ckpt`: `total=278`, `backbone=137`, `cross=0` -> **không phù hợp làm backbone trực tiếp cho code hiện tại**.
- `vit_mci.ckpt`: `total=278`, `backbone=137`, `cross=0` -> **không phù hợp làm backbone trực tiếp cho code hiện tại**.
- `segmentation.ckpt`: `total=307`, `backbone=0` -> **không dùng làm `simclrvit.ckpt_path`**.
- `idh.ckpt` và một số file downstream khác hiện chưa giải mã được hoàn chỉnh trong env lỗi (unpickle kéo theo import MONAI/TensorBoard/TensorFlow và đụng lỗi NumPy).

## Desired End State
Sau khi hoàn thành plan:
- Runner `run_soop_outcome_experiments_brainiac.sh` chạy ổn trong env `hieupcvp` không gặp lỗi import NumPy/TensorBoard.
- Pipeline train/eval SOOP chạy được với checkpoint backbone tương thích.
- Có cơ chế chọn checkpoint an toàn (validate trước khi train).
- README/quickstart và script runner nhất quán về checkpoint dùng cho SOOP.

## What We’re NOT Doing
- Không thay đổi kiến trúc backbone cốt lõi của BrainIAC ngoài phạm vi cần để tương thích checkpoint.
- Không retrain foundation model từ đầu.
- Không chỉnh sửa các downstream task ngoài SOOP pipeline (brain age, IDH, segmentation, stroke, ...).

## Implementation Approach
Ưu tiên sửa theo thứ tự khóa chặn:
1. Loại bỏ dependency chain TensorBoard/TensorFlow khi không cần logging.
2. Thiết lập gate kiểm tra checkpoint tương thích trước khi train.
3. Chuẩn hóa checkpoint mặc định dùng cho SOOP theo kết quả compatibility.
4. Cập nhật docs và runner để tránh tái diễn lỗi.

---

## Phase 1: Ổn định môi trường import (NumPy/TensorBoard)

### Overview
Chặn lỗi import ngay khi khởi tạo Trainer và MONAI trong môi trường hiện tại.

### Changes required

#### 1) Ép Trainer không tạo logger mặc định TensorBoard khi `--no-use-wandb`
**File**: `src/train_lightning_soop_regression.py`
**Changes**:
- Khi `use_wandb=False`, truyền `logger=False` thay vì `logger=None` vào `pl.Trainer(...)`.
- Mục tiêu: không đi vào `TensorBoardLogger` fallback mặc định của Lightning.

#### 2) Tách callback phụ thuộc logger
**File**: `src/train_lightning_soop_regression.py`
**Changes**:
- Chỉ thêm `LearningRateMonitor` khi logger bật.
- Giảm khả năng kích hoạt chain tensorboard không cần thiết.

#### 3) Khóa version môi trường cho SOOP
**File**: `requirements.txt` (hoặc tài liệu môi trường BrainIAC nếu đang dùng conda yaml riêng)
**Changes**:
- Chốt `numpy<2` cho env chạy SOOP, hoặc chốt bộ phiên bản tương thích torch/monai/tensorboard/tensorflow hiện có.
- Nếu giữ NumPy 2.x thì cần đảm bảo toàn bộ binary extension liên quan đã rebuild cho NumPy 2.

### Success criteria

#### Automated verification
- [ ] `conda run -n hieupcvp python -c "import numpy; print(numpy.__version__)"` trả về version đúng theo lock.
- [ ] `conda run -n hieupcvp python -c "import pytorch_lightning as pl; t=pl.Trainer(logger=False,enable_checkpointing=False,max_epochs=1,accelerator='cpu',devices=1); print('ok')"` pass.
- [ ] `python src/train_lightning_soop_regression.py --help` pass.

#### Manual verification
- [ ] Chạy 1 smoke train với `--no-use-wandb` không còn xuất hiện traceback `tensorboard.compat`/`numpy.core._multiarray_umath`.

**Implementation note**: Dừng sau phase này để xác nhận bằng tay rằng lỗi import đã biến mất hoàn toàn.

---

## Phase 2: Chuẩn hóa checkpoint tương thích

### Overview
Ngăn crash do load strict bằng cách chọn đúng checkpoint và thêm bước validation sớm.

### Changes required

#### 1) Thêm hàm validate checkpoint trước khi tạo model
**File**: `src/train_lightning_soop_regression.py`
**Changes**:
- Trước `SOOPRegressionLightningModule(...)`, load checkpoint metadata và kiểm tra:
  - có key `backbone.*`
  - có nhóm key `cross_attn/norm_cross_attn` theo kiến trúc hiện tại
- Nếu fail: raise lỗi rõ ràng (in ra số key backbone/cross và gợi ý checkpoint đã test được).

#### 2) Đổi checkpoint mặc định SOOP về checkpoint phù hợp
**File**: `src/config_soop_regression.yml`
**Changes**:
- Đặt `simclrvit.ckpt_path` mặc định trỏ `src/checkpoints/BrainIAC_mock.ckpt` (trong lúc chưa có bản `BrainIAC.ckpt` tương thích schema mới).

#### 3) Cập nhật runner để fail-fast và gợi ý rõ
**File**: `research/run_soop_outcome_experiments_brainiac.sh`
**Changes**:
- Trước khi train, chạy preflight command kiểm tra checkpoint compatibility.
- Nếu không hợp lệ thì dừng ngay với message cụ thể.

### Success criteria

#### Automated verification
- [ ] `python src/train_lightning_soop_regression.py ... --ckpt-path src/checkpoints/BrainIAC_mock.ckpt --limit-train-batches 1 --limit-val-batches 1 --max-epochs 1 --no-use-wandb` pass.
- [ ] `python src/train_lightning_soop_regression.py ... --ckpt-path src/checkpoints/BrainIAC.ckpt` fail-fast với message checkpoint mismatch rõ ràng (không crash sâu trong `load_state_dict`).

#### Manual verification
- [ ] Xác nhận output error message đủ rõ để người vận hành biết nên đổi sang checkpoint nào.

**Implementation note**: Dừng sau phase này để người kiểm thử xác nhận lựa chọn checkpoint cho production run.

---

## Phase 3: Đồng bộ tài liệu và quy trình chạy

### Overview
Đảm bảo hướng dẫn README/quickstart và runner không mâu thuẫn checkpoint.

### Changes required

#### 1) Đồng bộ README phần SOOP
**File**: `README.md`
**Changes**:
- Nêu rõ checkpoint nào được verify cho SOOP pipeline hiện tại.
- Bổ sung bước preflight check trước khi chạy full matrix.

#### 2) Đồng bộ `quickstart.ipynb`
**File**: `src/quickstart.ipynb`
**Changes**:
- Ghi chú rõ:
  - `BrainIAC_mock.ckpt` dùng cho pipeline cần schema backbone mới.
  - Nếu dùng `BrainIAC.ckpt` cũ sẽ mismatch trên code hiện tại.
- Không thay đổi workflow ngoài phạm vi làm rõ checkpoint tương thích.

#### 3) Viết tài liệu ngắn “checkpoint compatibility note”
**File**: `research/implemented.md` (append section) hoặc file research note mới
**Changes**:
- Ghi bảng mapping checkpoint -> mục đích -> trạng thái tương thích với `src/model.py` hiện tại.

### Success criteria

#### Automated verification
- [ ] `bash -n research/run_soop_outcome_experiments_brainiac.sh` pass.
- [ ] Notebook JSON hợp lệ sau chỉnh sửa.

#### Manual verification
- [ ] Người mới trong team đọc README + quickstart có thể chọn đúng checkpoint cho SOOP mà không cần tra cứu thêm.

---

## Phase 4: Chạy smoke + production matrix

### Overview
Xác nhận toàn bộ fix bằng smoke run trước, sau đó mới chạy full matrix.

### Changes required

#### 1) Smoke runs
- 1 run preprocessed image-only.
- 1 run raw image+tabular.

#### 2) Full runner
- Chạy `research/run_soop_outcome_experiments_brainiac.sh` với checkpoint đã verify.

### Success criteria

#### Automated verification
- [ ] Mỗi run tạo đủ:
  - `resolved_config_soop_regression.yml`
  - `checkpoints/*.ckpt`
  - `eval/predictions.csv`
  - `eval/results_eval_soop_regression.json`
- [ ] JSON eval có đủ metric keys (`mse`, `rmse`, `mae`, `mape`, `r2`, `loss`).

#### Manual verification
- [ ] Không còn traceback ABI NumPy/TensorBoard.
- [ ] Không còn lỗi `Missing key(s)` khi load backbone.
- [ ] Metrics sinh ra ổn định trên cả split raw và preprocessed.

---

## Testing strategy

### Unit-level checks
- Validate hàm checkpoint preflight với:
  - ckpt hợp lệ (`BrainIAC_mock.ckpt`)
  - ckpt thiếu `cross_attn` (`BrainIAC.ckpt`)
  - ckpt không có `backbone.*` (`segmentation.ckpt`)

### Integration checks
- Smoke train/eval end-to-end 2 cấu hình đại diện.
- Full 6-run matrix qua runner.

### Manual test steps
1. Chạy lệnh smoke không wandb, xác nhận hết lỗi import chain.
2. Chạy lệnh với ckpt mismatch, xác nhận fail-fast message rõ.
3. Chạy full runner và kiểm tra đủ artifacts.

## Migration notes
- Nếu buộc dùng `BrainIAC.ckpt` gốc (không mock), cần một trong hai hướng:
  - Cố định stack MONAI/ViT theo phiên bản tương ứng với checkpoint cũ; hoặc
  - Chuyển đổi checkpoint sang schema backbone mới trước khi train.
- Không làm migration này trong scope fix nhanh hiện tại; chỉ thực hiện khi có yêu cầu chạy bằng foundation checkpoint gốc.

## References
- `research/research_bug.md`
- `research/run_soop_outcome_experiments_brainiac.sh`
- `src/train_lightning_soop_regression.py`
- `src/model.py`
- `src/quickstart.ipynb`
- `README.md`
