---
date: 2026-03-19T00:05:20+07:00
researcher: GitHub Copilot
git_commit: ba60f45bed832ee0e5683c678caeaeefe2072f0d
branch: main
repository: BrainIAC
topic: "Codebase mapping and technical documentation of BrainIAC"
tags: [research, codebase, brainiac, mri, pytorch-lightning, monai]
status: complete
last_updated: 2026-03-19
last_updated_by: GitHub Copilot
---

# Research: BrainIAC Codebase

**Date**: 2026-03-19T00:05:20+07:00  
**Researcher**: GitHub Copilot  
**Git Commit**: ba60f45bed832ee0e5683c678caeaeefe2072f0d  
**Branch**: main  
**Repository**: BrainIAC

## Research Question
Đọc cấu trúc codebase `BrainIAC` để hiểu project là gì, gồm những phần nào, và cách các phần kết nối với nhau.

## Summary
BrainIAC là codebase mô hình nền tảng MRI não 3D dựa trên ViT (MONAI), dùng checkpoint pretrain kiểu SimCLR làm backbone và fine-tune cho nhiều tác vụ downstream: regression, binary classification, multiclass classification, saliency visualization, feature extraction, và tumor segmentation.

Mã nguồn tập trung trong `src/`, gồm:
- Backbone + head wrappers (`model.py`, `load_brainiac.py`)
- Dataset + transform pipelines cho single/dual/quad/segmentation (`dataset.py`, `dataset_segmentation.py`)
- Các entrypoint huấn luyện theo task (`train_lightning_*.py`)
- Các script inference/evaluation (`test_inference_finetune.py`, `test_inference_perturbation.py`, `test_segmentation.py`, `generate_segmentation.py`)
- Các script tạo saliency theo backbone hoặc theo checkpoint downstream (`get_brainiac_saliencymap.py`, `generate_*_vit_saliency.py`)
- Script trích xuất feature backbone (`get_brainiac_features.py`)
- Pipeline preprocessing MRI (`preprocessing/`)

## Top-level Structure

- `README.md`: mô tả mục tiêu project, quickstart notebook, danh sách downstream tasks
- `requirements.txt`: dependencies chính (`pytorch-lightning`, `monai`, `SimpleITK`, `nibabel`, `wandb`, ...)
- `docs/downstream_tasks/`: tài liệu chạy cho từng bài toán
- `pngs/`: hình minh họa
- `research/`: thư mục nghiên cứu nội bộ (hiện dùng để lưu tài liệu này)
- `src/`: toàn bộ code train/inference/preprocessing

## Core Model Components

### 1) Backbone and Heads
**File**: `src/model.py`

- `ViTBackboneNet`: khởi tạo `monai.networks.nets.ViT` (input 1 channel, 96x96x96, patch 16^3, hidden 768, 12 layers/heads).
- Khi load checkpoint, script lấy `state_dict` rồi lọc key bắt đầu bằng `backbone.`; bỏ prefix này trước khi load vào ViT.
- `forward` trả CLS token (`features[0][:, 0]`) làm embedding 768-dim.
- `Classifier`: linear head từ 768 -> `num_classes`.
- `SingleScanModel`: backbone + dropout + classifier cho input đơn.
- `SingleScanModelBP`: input dual (2 scans), chạy shared backbone từng scan, mean-pool features rồi classify.
- `SingleScanModelQuad`: input quad (4 scans), shared backbone 4 lần, mean-pool features rồi classify.

### 2) Lightweight Backbone Loader
**File**: `src/load_brainiac.py`

- Hàm `load_brainiac(checkpoint_path, device)` tạo `ViTBackboneNet` và chuyển sang device.
- Được dùng bởi script feature extraction và saliency backbone.

### 3) Segmentation Model
**File**: `src/segmentation_model.py`

- `ViTUNETRSegmentationModel`:
  1. Tạo ViT encoder cùng cấu hình embedding.
  2. Load SimCLR `backbone.*` vào ViT.
  3. Tạo `UNETR` decoder.
  4. Copy trọng số ViT sang `unetr.vit`.
- `forward` trả output segmentation từ UNETR.

## Data Layer

### 1) General and Multi-input Datasets
**File**: `src/dataset.py`

- Transform builders:
  - `get_default_transform` / `get_validation_transform` cho single-image.
  - `get_default_transform_dual` / `get_validation_transform_dual` cho dual-image.
  - `get_default_transform_quad` / `get_validation_transform_quad` cho quad-image.
  - Segmentation transforms cũng được định nghĩa trong file này (`*_segmentation`).
- Dataset classes:
  - `BrainAgeDataset` (single image + float label)
  - `MCIStrokeDataset` (single image + float label)
  - `SequenceDataset` (single image multiclass; label giảm 1 để về index 0..3)
  - `DualImageDataset` (path `{pat_id}_t2f.nii.gz` + `{pat_id}_t1ce.nii.gz`)
  - `QuadImageDataset` (path `{pat_id}_{t1ce,t1n,t2w,t2f}.nii.gz`, label từ cột `survival`)
  - `SegmentationDataset` (image + mask từ cấu trúc thư mục theo `dataset`)
- Custom collate:
  - `dual_image_collate_fn` trả tensor shape `(B, 2, C, D, H, W)`
  - `quad_image_collate_fn` trả tensor shape `(B, 4, C, D, H, W)`

### 2) Segmentation Cache Dataset Helper
**File**: `src/dataset_segmentation.py`

- `get_segmentation_dataloader(...)` đọc CSV có cột `image_path`, `mask_path`.
- Dùng `CacheDataset` với transform train/val khác nhau.
- Train augment gồm flip/rotate/elastic/bias/noise; val chỉ normalize + tensorize.

## Training Layer (`train_lightning_*.py`)

### Shared Pattern
- Hầu hết scripts dùng:
  - `argparse --config`
  - `yaml.safe_load`
  - module Lightning (`pl.LightningModule`)
  - logger `WandbLogger`
  - callback `ModelCheckpoint`, `LearningRateMonitor`
  - optimizer Adam/AdamW + cosine scheduler
- Cờ freeze backbone dựa vào config (`train.freeze` hoặc `training.freeze`).

### Task-specific Training Scripts

1. `src/train_lightning_brainage.py`
- Regression (MSE), metric chính `val_mae`.
- Model: `SingleScanModel`.
- Dataset: `BrainAgeDataset`.

2. `src/train_lightning_mci.py`
- Binary classification (BCEWithLogits).
- Tính `val_auc` và `val_accuracy` theo epoch.
- Model: `SingleScanModel`.
- Dataset: `MCIStrokeDataset`.

3. `src/train_lightning_multiclass.py`
- Multiclass (CrossEntropy).
- Metric: `val_accuracy`, `val_auc` (OvR) nếu tính được.
- Model: `SingleScanModel`, head 4 classes.
- Dataset: `SequenceDataset`.

4. `src/train_lightning_idh.py`
- Binary classification cho dual input.
- Model: `SingleScanModelBP`.
- Dataset: `DualImageDataset` + `dual_image_collate_fn`.
- Dùng `torchmetrics` (accuracy, precision, recall, f1, auroc).

5. `src/train_lightning_os.py`
- Binary classification cho quad input.
- Model: `SingleScanModelQuad`.
- Dataset: `QuadImageDataset` + `quad_image_collate_fn`.
- Dùng `torchmetrics` tương tự IDH.

6. `src/train_lightning_segmentation.py`
- Model: `ViTUNETRSegmentationModel`.
- Loss: `DiceLoss(sigmoid=True) + BCEWithLogitsLoss`.
- Validation dùng `sliding_window_inference`.
- Metric chính: Dice.

## Inference and Evaluation

### 1) Unified Finetuned Inference
**File**: `src/test_inference_finetune.py`

- Định nghĩa registry `DATASETS` (task configs) và `DATASETS_TO_RUN`.
- Hàm `load_model(...)` reconstruct model theo `image_type` (`single`, `dual`, `quad`) rồi load checkpoint (hỗ trợ Lightning `state_dict` với prefix `model.`).
- `create_test_dataset(...)` chọn class dataset theo task/image type.
- `run_inference(...)` xử lý output theo `task_type`:
  - regression: output thô
  - classification: sigmoid + threshold 0.5
  - multiclass: softmax + argmax
- `save_predictions(...)` ghi CSV output.
- `calculate_metrics(...)` tính metrics theo task.
- `main()` chạy tuần tự các dataset trong `DATASETS_TO_RUN`, ghi `inference/eval_results.json`.

### 2) Perturbation Evaluation
**File**: `src/test_inference_perturbation.py`

- Tái sử dụng hàm từ `test_inference_finetune` để load model/dataset.
- Áp dụng perturbation MONAI (`AdjustContrast`, `RandBiasField`, `RandGibbsNoise`).
- Chạy inference và lưu CSV theo từng mức perturbation.

### 3) Segmentation Evaluation
**File**: `src/test_segmentation.py`

- Load checkpoint Lightning segmentation (`state_dict`) rồi strip prefix `model.`.
- Dùng `sliding_window_inference` cho test set.
- Tính Dice, IoU (Jaccard), precision, recall + per-case Dice.
- Xuất:
  - JSON metrics tổng (`--output_json`, mặc định `./inference/model_outputs/segmentation.json`)
  - CSV per-case (`--csv_output_dir`)

### 4) Single-image Segmentation Generation
**File**: `src/generate_segmentation.py`

- Script nhận input 1 ảnh NIfTI, load checkpoint segmentation.
- Preprocess -> infer -> threshold -> lưu mask NIfTI bằng MONAI saver.

## Saliency and Feature Utilities

### 1) Backbone Feature Extraction
**File**: `src/get_brainiac_features.py`

- Dùng `BrainAgeDataset` + validation transform.
- Load backbone qua `load_brainiac`.
- Trích embedding và ghi CSV với cột `Feature_0...Feature_n` + `GroundTruthClassLabel`.

### 2) Backbone Attention Saliency (Generic)
**File**: `src/get_brainiac_saliencymap.py`

- Wrap attention block để thu attention weights.
- Lấy attention CLS-to-patch tại layer chỉ định.
- Reshape về grid 3D theo patch, upsample về 96^3.
- Chuẩn hóa saliency map và lưu NIfTI:
  - input image
  - attention saliency map

### 3) Downstream Saliency Scripts
**Files**:
- `src/generate_brainage_vit_saliency.py`
- `src/generate_idh_vit_saliency.py`
- `src/generate_mci_stroke_vit_saliency.py`
- `src/generate_multiclass_vit_saliency.py`
- `src/generate_os_vit_saliency.py`

Các script này có pattern chung: load checkpoint downstream, trích attention map từ ViT blocks, nội suy 3D, lưu NIfTI input + saliency.

## Preprocessing Pipeline

### 1) DICOM -> NIfTI
**File**: `src/preprocessing/dicomtonifti_2.py`

- Duyệt các thư mục scan con trong input.
- Mỗi thư mục đọc series DICOM qua `SimpleITK.ImageSeriesReader`.
- Ghi file `.nii.gz` tương ứng ra output.

### 2) Registration + Skull Stripping
**File**: `src/preprocessing/mri_preprocess_3d_simple.py`

- `registration(...)`:
  - Đọc atlas template.
  - N4 bias correction.
  - Registration với mutual information + gradient descent qua SimpleITK.
  - Save ảnh đã đăng ký với hậu tố `_0000.nii.gz`.
- `brain_extraction(...)`:
  - Gọi `hd_bet(...)` từ `HD_BET.hd_bet`.
- `main(...)`:
  - Chạy registration vào thư mục tạm.
  - Chạy brain extraction ra output cuối.
  - Xóa thư mục tạm.

### 3) Preprocessing Assets
**Directory**: `src/preprocessing/`

- `HD_BET/`, `HDBET_Code/`, `hd-bet_params/`: code và params skull stripping.
- `atlases/`: file atlas/reference dùng cho đăng ký ảnh.

## Configuration Files

1. `src/config_finetune.yml`
- Dùng cho hầu hết downstream classification/regression.
- Nhóm keys chính: `model`, `data`, `simclrvit`, `optim`, `logger`, `gpu`, `train`.

2. `src/config_finetune_segmentation.yml`
- Dùng cho segmentation.
- Nhóm keys chính: `data`, `model`, `pretrain`, `training`, `output`, `logger`, `gpu`.

## Documentation Layer (`docs/downstream_tasks`)

Có 7 tài liệu tương ứng các task:
- `brain_age_prediction.md`
- `idh_mutation_classification.md`
- `mild_cognitive_impairment_classification.md`
- `MR_sequence_classification.md`
- `timetostroke_prediction.md`
- `diffuse_glioma_overall_survival.md`
- `tumor_segmentation.md`

Mỗi tài liệu thường mô tả:
- định nghĩa task
- format CSV + cấu trúc thư mục ảnh
- ví dụ cấu hình
- lệnh train/inference/saliency

## End-to-end Flows as Implemented

### A) Downstream classification/regression
1. Chuẩn bị CSV + NIfTI theo dataset class phù hợp.  
2. Chỉnh `config_finetune.yml` (đường dẫn data/checkpoint/gpu/logger).  
3. Chạy `train_lightning_*.py` tương ứng task.  
4. Đánh giá/infer bằng `test_inference_finetune.py` (theo `DATASETS_TO_RUN`).  
5. Tùy chọn: chạy `test_inference_perturbation.py` để đánh giá dưới nhiễu/biến đổi.  
6. Tùy chọn: tạo saliency bằng `generate_*_vit_saliency.py` hoặc saliency backbone chung.

### B) Segmentation
1. Chuẩn bị CSV gồm `image_path,mask_path`.  
2. Chỉnh `config_finetune_segmentation.yml`.  
3. Train với `train_lightning_segmentation.py`.  
4. Evaluate với `test_segmentation.py`.  
5. Inference 1 ảnh với `generate_segmentation.py`.

### C) Pretrained Backbone Utilities
1. Load backbone trực tiếp bằng `load_brainiac.py`.  
2. Trích features bằng `get_brainiac_features.py`.  
3. Tạo attention saliency backbone bằng `get_brainiac_saliencymap.py`.

## Output Artifacts and Paths

- Feature CSV: theo `--output_csv` của `get_brainiac_features.py`
- Saliency NIfTI: theo `--output_dir` của script saliency
- Inference CSV downstream: theo `output_csv_path` trong `DATASETS`
- Inference metrics JSON: `inference/eval_results.json`
- Segmentation metrics JSON: mặc định `./inference/model_outputs/segmentation.json`
- Segmentation per-case CSV: theo `--csv_output_dir`

## Key Internal Dependencies

- DL framework: `pytorch-lightning==2.3.3`
- Medical imaging toolkit: `monai==1.3.2`
- NIfTI IO: `nibabel==5.2.1`
- Medical IO/preprocessing: `SimpleITK==2.4.0`, `pydicom`
- Metrics and ML utilities: `scikit-learn==1.2.2`, `lifelines`
- Logging/tracking: `wandb`

## Notebooks and Example Assets

- `src/quickstart.ipynb`: notebook quickstart cho workflow BrainIAC.
- `src/util.ipynb`: notebook tiện ích.
- `src/data/sample/`: sample data cho chạy thử.
- `src/inference/features/` và `src/inference/saliency_maps/`: ví dụ output artifacts.

## Architecture Documentation (Current State)

- Mô hình hóa theo hướng **shared ViT backbone + task-specific heads/pipelines**.
- Fine-tuning downstream và segmentation tách thành các entrypoint script riêng.
- Dataset handling tách theo số lượng input sequence (single/dual/quad/seg).
- Inference tập trung qua một script tổng hợp (`test_inference_finetune.py`) cho nhiều task (classification/regression/multiclass).
- Saliency được triển khai theo hai nhánh:
  - nhánh backbone generic
  - nhánh per-task checkpoint-specific.
- Tiền xử lý MRI (DICOM conversion, registration, skull stripping) nằm trong module `src/preprocessing/` và có tài nguyên model/phụ trợ đi kèm.

## Notes on Naming/Script Surface

Trong docs có một số ví dụ lệnh script khác tên với file hiện có trong `src/` (ví dụ mô tả lệnh nhưng file hiện hành trong codebase dùng tên khác). Tài liệu này ưu tiên mô tả theo code/file hiện tồn tại trong repository.
