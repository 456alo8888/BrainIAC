---
date: 2026-03-20T16:20:00+07:00
researcher: GitHub Copilot
git_commit: ba60f45bed832ee0e5683c678caeaeefe2072f0d
branch: main
repository: BrainIAC
topic: "Phân tích lỗi khi chạy run_soop_outcome_experiments_brainiac.sh"
status: complete
last_updated: 2026-03-20
last_updated_by: GitHub Copilot
---

# Research bug: run_soop_outcome_experiments_brainiac.sh

## Câu hỏi
Tại sao khi chạy `research/run_soop_outcome_experiments_brainiac.sh` xuất hiện chuỗi lỗi `NumPy 2.x ABI`, `tensorboard.compat.notf`, và `RuntimeError missing key(s)` khi load checkpoint?

## Tóm tắt
Lỗi xảy ra theo 2 lớp độc lập:

1. **Lớp môi trường (import-time):** môi trường `hieupcvp` đang dùng `numpy==2.4.3`, trong khi một số module binary trong stack TensorFlow/TensorBoard được build theo ABI NumPy 1.x. Khi MONAI/PyTorch gọi nhánh TensorBoard (`torch.utils.tensorboard`), TensorBoard cố truy cập `tf` và kéo theo TensorFlow, dẫn tới lỗi ABI NumPy.
2. **Lớp model/checkpoint (runtime):** kiến trúc `ViT` hiện tại (MONAI version đang cài) kỳ vọng có tham số `cross_attn`/`norm_cross_attn`, nhưng checkpoint `BrainIAC.ckpt` không chứa các key này. Vì code load với `strict=True`, quá trình khởi tạo backbone dừng với `Missing key(s)`.

## Bằng chứng đã kiểm tra

### 1) Luồng chạy script
- File runner gọi train bằng:
  - `research/run_soop_outcome_experiments_brainiac.sh`
  - `conda run -n hieupcvp ... python src/train_lightning_soop_regression.py ...`
- CKPT dùng cho train lấy từ biến `BRAINIAC_CKPT` (mặc định trỏ `src/checkpoints/BrainIAC.ckpt`).

### 2) Trạng thái môi trường
- `numpy` trong env `hieupcvp`: **2.4.3**.
- `tensorboard` trong env `hieupcvp`: **2.17.1**.
- `tensorflow` import thất bại với thông báo ABI:
  - `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.3...`
  - `ImportError: numpy.core._multiarray_umath failed to import`

### 3) Vì sao có `tensorboard.compat.notf`
- Trong `tensorboard/compat/__init__.py`, hàm `tf()` thử `from tensorboard.compat import notf` trước.
- Trong thư mục `.../site-packages/tensorboard/compat` hiện có:
  - `__init__.py`, `proto/`, `tensorflow_stub/`
  - **không có `notf.py`**
- Do đó xuất hiện:
  - `ImportError: cannot import name 'notf' from 'tensorboard.compat'`
- Sau nhánh này, TensorBoard thử import TensorFlow thật, và rơi tiếp vào lỗi ABI NumPy ở trên.

### 4) Vì sao dù `--no-use-wandb` vẫn dính TensorBoard
- Trong `train_lightning_soop_regression.py`, khi `use_wandb` tắt thì `logger = None`.
- Nhưng với PyTorch Lightning 2.3.x, `Trainer(logger=None)` vẫn khởi tạo logger mặc định là `TensorBoardLogger`.
- Kiểm tra trực tiếp cho thấy `type(t.logger).__name__ == 'TensorBoardLogger'`.
- Vì vậy luồng TensorBoard vẫn được kích hoạt, dẫn tới chuỗi import lỗi kể trên.

### 5) Mismatch checkpoint vs kiến trúc ViT hiện tại
- `src/model.py` tạo backbone bằng `monai.networks.nets.ViT(...)` và load state dict với `strict=True`.
- Kiểm tra model `ViT` hiện tại cho thấy state dict kỳ vọng:
  - tổng key: **221**
  - key liên quan `cross_attn`/`norm_cross_attn`: **84**
- Kiểm tra `src/checkpoints/BrainIAC.ckpt` cho thấy:
  - tổng key: **149**
  - key prefix `backbone.`: **137**
  - key `cross_attn`/`norm_cross_attn`: **0**
- Vì vậy khi load strict sẽ báo thiếu toàn bộ nhóm key `blocks.*.norm_cross_attn.*` và `blocks.*.cross_attn.*` đúng như traceback bạn gặp.

## Kết luận nguyên nhân
Chuỗi lỗi trong log là **kết quả chồng của hai nguyên nhân độc lập**:

1. **Import chain TensorBoard/TensorFlow không tương thích với NumPy 2.4.3** trong env hiện tại, kèm trạng thái `tensorboard.compat` thiếu `notf`.
2. **Checkpoint `BrainIAC.ckpt` không cùng schema tham số với `ViT` đang được khởi tạo hiện tại** (thiếu hoàn toàn nhóm trọng số cross-attention), nên load strict thất bại.

Do đó kể cả vượt qua lớp lỗi import môi trường, tiến trình vẫn dừng ở bước load backbone vì mismatch checkpoint.

## Tệp liên quan
- `baseline_encoder/BrainIAC/research/run_soop_outcome_experiments_brainiac.sh`
- `baseline_encoder/BrainIAC/src/train_lightning_soop_regression.py`
- `baseline_encoder/BrainIAC/src/model.py`
- `baseline_encoder/BrainIAC/src/config_soop_regression.yml`
- `baseline_encoder/BrainIAC/src/checkpoints/BrainIAC.ckpt`
