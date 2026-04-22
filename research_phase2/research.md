---
date: 2026-03-20T18:46:37+07:00
researcher: GitHub Copilot
git_commit: ba60f45bed832ee0e5683c678caeaeefe2072f0d
branch: main
repository: BrainIAC
topic: "Phân tích vì sao không thấy test result và runtime log khi chạy bash_image_only.sh"
tags: [research, brainiac, soop, logging, evaluation]
status: complete
last_updated: 2026-03-20
last_updated_by: GitHub Copilot
---

# Research: Không thấy kết quả test và runtime log ở BrainIAC image-only

## Câu hỏi
Khi chạy `src/bash_image_only.sh` và xem log `src/logs/soop_image_only_20260320_171621.log`, vì sao không thấy kết quả trên test set và đồng thời không thấy runtime log trên terminal?

## Tóm tắt kết luận
1. **Không có kết quả test** vì script `src/bash_image_only.sh` chỉ chạy train (`train_lightning_soop_regression.py`) và dừng sau `trainer.fit(...)`; không có bước gọi `eval_soop_regression.py`.
2. **Runtime log terminal có thể không live** do `conda run` mặc định capture stdout/stderr; theo help của conda, muốn stream trực tiếp cần `--no-capture-output` (alias `--live-stream`).
3. Log file hiện tại xác nhận tiến trình kết thúc ở `Trainer.fit stopped` và chỉ in đường dẫn checkpoint + resolved config, không có đoạn lưu `predictions.csv`/`results_eval_soop_regression.json`.

## Bằng chứng chi tiết

### 1) `bash_image_only.sh` chỉ gọi train
- `src/bash_image_only.sh:13-36` gọi duy nhất:
  - `conda run -n hieupcvp ... python -u train_lightning_soop_regression.py ...`
  - pipe sang `tee -a "$LOG_FILE"`.
- Không có lệnh thứ hai gọi `eval_soop_regression.py` trong file này.

### 2) Train script kết thúc sau fit, không tự chạy test
- `src/train_lightning_soop_regression.py:406` gọi `trainer.fit(model, datamodule=data_module)`.
- Sau đó chỉ in:
  - `Best checkpoint` tại `:409`
  - `Resolved config` tại `:410`
- Không có `trainer.test(...)` và cũng không spawn eval script trong `main()`.

### 3) Nơi thực sự sinh kết quả test
- Script eval riêng là `src/eval_soop_regression.py`:
  - Bắt buộc có `--split-csv` (`:21`) và `--output-dir` (`:22`)
  - Ghi `predictions.csv` tại `:148`
  - Ghi `results_eval_soop_regression.json` tại `:151`
- Runner có train + eval đầy đủ là `research/run_soop_outcome_experiments_brainiac.sh`:
  - Train ở `:81-99`
  - Eval ngay sau train ở `:104-112` với `--split-csv "$fold_dir/test.csv"` (`:107`)

### 4) Bằng chứng từ log bạn cung cấp
Trong `src/logs/soop_image_only_20260320_171621.log`:
- Có `Best checkpoint ...` ở dòng `232`
- Có `Resolved config ...` ở dòng `233`
- Kết thúc bằng ``Trainer.fit` stopped: `max_epochs=10` reached.` ở dòng `276`
- Không có dòng `Saved predictions:` hoặc `Saved results:` (các dòng chỉ xuất hiện khi chạy `eval_soop_regression.py`).

### 5) Vì sao có thể không thấy runtime log trên terminal
- `src/bash_image_only.sh:13` dùng `conda run -n hieupcvp ...`.
- `conda run --help` mô tả mặc định có capture output; muốn stream live thì dùng:
  - `--no-capture-output` hoặc `--live-stream`.
- Vì vậy có thể gặp hành vi: log không hiển thị realtime như kỳ vọng ở terminal dù vẫn được ghi ra file qua `tee` trong command chain.

## Mapping luồng chạy hiện tại (đúng theo codebase)
- **Image-only script hiện tại**: `src/bash_image_only.sh`
  - Chỉ train + lưu checkpoint/resolved config + log file.
- **Luồng có test/eval đầy đủ**: `research/run_soop_outcome_experiments_brainiac.sh`
  - Train xong sẽ gọi `src/eval_soop_regression.py` để tạo artifacts test (`predictions.csv`, `results_eval_soop_regression.json`).

## Ghi chú bổ sung (fact only)
- `src/bash_image_only.sh` hiện chứa giá trị `WANDB_API_KEY` hard-code tại dòng `5`.
- File đối chiếu 3DINO `dinov2/eval/bash_image_orig.sh` cũng có pattern tương tự (hard-code key + chạy trực tiếp train/eval command).
