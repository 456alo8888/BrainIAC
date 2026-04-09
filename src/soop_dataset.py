import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import get_default_transform, get_validation_transform

DEFAULT_TARGET = "gs_rankin_6isdeath"
TARGET_ALIAS = "gs_rankin+6isdeath"


class SOOPRegressionDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        target_col: str = DEFAULT_TARGET,
        transform=None,
        include_tabular: bool = False,
        normalize_tabular: bool = False,
        tabular_feature_cols: Optional[Sequence[str]] = None,
        tabular_mean: Optional[np.ndarray] = None,
        tabular_std: Optional[np.ndarray] = None,
        drop_missing_label: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.include_tabular = include_tabular
        self.normalize_tabular = normalize_tabular

        if self.df.empty:
            raise ValueError(f"Split CSV is empty: {self.csv_path}")
        if "image_path" not in self.df.columns:
            raise KeyError(f"Missing required column 'image_path' in {self.csv_path}")
        if "subject_id" not in self.df.columns:
            raise KeyError(f"Missing required column 'subject_id' in {self.csv_path}")

        self.target_col = self._resolve_target_col(target_col)

        if drop_missing_label:
            self.df = self.df.loc[self.df[self.target_col].notna()].reset_index(drop=True)

        self.df[self.target_col] = pd.to_numeric(self.df[self.target_col], errors="coerce")
        if drop_missing_label:
            self.df = self.df.loc[self.df[self.target_col].notna()].reset_index(drop=True)

        if include_tabular:
            self.tabular_feature_cols = self._resolve_tabular_cols(tabular_feature_cols)
            self.tabular_np = self.df[self.tabular_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if normalize_tabular:
                if tabular_mean is None or tabular_std is None:
                    tabular_mean = self.tabular_np.mean(axis=0, keepdims=True)
                    tabular_std = self.tabular_np.std(axis=0, keepdims=True)
                tabular_std = np.where(tabular_std == 0.0, 1.0, tabular_std)
                self.tabular_np = (self.tabular_np - tabular_mean) / tabular_std
                self.tabular_mean = tabular_mean.astype(np.float32)
                self.tabular_std = tabular_std.astype(np.float32)
            else:
                self.tabular_mean = None
                self.tabular_std = None
        else:
            self.tabular_feature_cols = []
            self.tabular_np = None
            self.tabular_mean = None
            self.tabular_std = None

    def _resolve_target_col(self, target_col: str) -> str:
        columns = self.df.columns.tolist()
        if target_col in columns:
            return target_col
        if target_col == TARGET_ALIAS and DEFAULT_TARGET in columns:
            return DEFAULT_TARGET
        if target_col == DEFAULT_TARGET and TARGET_ALIAS in columns:
            return TARGET_ALIAS
        raise KeyError(f"Target column '{target_col}' not found in {self.csv_path}")

    def _resolve_tabular_cols(self, tabular_feature_cols: Optional[Sequence[str]]) -> List[str]:
        if tabular_feature_cols is not None:
            missing = [c for c in tabular_feature_cols if c not in self.df.columns]
            if missing:
                raise KeyError(f"Missing tabular feature columns in {self.csv_path}: {missing}")
            return list(tabular_feature_cols)

        excluded = {
            "subject_id",
            "trace_dir",
            "image_path",
            "mask_path",
            "segm_path",
            "tabular_path",
            "tabular_features",
            "nihss",
            "gs_rankin_6isdeath",
            TARGET_ALIAS,
        }
        cols = [col for col in self.df.columns if col not in excluded]

        if cols:
            return cols

        if "tabular_features" in self.df.columns:
            parsed = self.df["tabular_features"].map(self._parse_json_features).tolist()
            feature_names = sorted({key for row in parsed for key in row.keys()})
            if not feature_names:
                return []
            matrix = np.zeros((len(parsed), len(feature_names)), dtype=np.float32)
            for i, feat_dict in enumerate(parsed):
                for j, feat_name in enumerate(feature_names):
                    matrix[i, j] = np.float32(feat_dict.get(feat_name, 0.0))
            for j, feat_name in enumerate(feature_names):
                self.df[feat_name] = matrix[:, j]
            return feature_names

        return []

    @staticmethod
    def _parse_json_features(value: object) -> Dict[str, float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return {}
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items()}
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}

        out: Dict[str, float] = {}
        for key, val in parsed.items():
            try:
                out[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
        return out

    @property
    def num_tabular_features(self) -> int:
        return len(self.tabular_feature_cols)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = str(row["image_path"])
        subject_id = str(row["subject_id"])

        sample = {"image": image_path}
        if self.transform is not None:
            transformed = self.transform(sample)
            image = transformed["image"] if isinstance(transformed, dict) else transformed
        else:
            image = sample["image"]

        label_val = row[self.target_col]
        label_mask = 0.0 if pd.isna(label_val) else 1.0
        label = 0.0 if pd.isna(label_val) else float(label_val)

        output = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "label_mask": torch.tensor(label_mask, dtype=torch.float32),
            "subject_id": subject_id,
        }

        if self.include_tabular:
            if self.num_tabular_features == 0:
                raise ValueError("include_tabular=True but no tabular features were resolved")
            output["tabular"] = torch.tensor(self.tabular_np[idx], dtype=torch.float32)

        return output


class SOOPRegressionDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.data_cfg = config["data"]
        self._num_tabular_features = 0
        self._tabular_feature_cols: List[str] = []

    @property
    def num_tabular_features(self) -> int:
        return self._num_tabular_features

    @property
    def tabular_feature_cols(self) -> List[str]:
        return self._tabular_feature_cols

    def setup(self, stage: Optional[str] = None):
        fold_dir = Path(self.data_cfg["fold_dir"])
        target_col = self.data_cfg.get("target_col", DEFAULT_TARGET)
        include_tabular = bool(self.data_cfg.get("include_tabular", False))
        normalize_tabular = bool(self.data_cfg.get("normalize_tabular", False))
        drop_missing_label = bool(self.data_cfg.get("drop_missing_label", True))
        image_size = tuple(self.data_cfg.get("size", [96, 96, 96]))

        train_transform = get_default_transform(image_size=image_size)
        eval_transform = get_validation_transform(image_size=image_size)

        train_csv = fold_dir / "train.csv"
        valid_csv = fold_dir / "valid.csv"
        test_csv = fold_dir / "test.csv"

        self.train_dataset = SOOPRegressionDataset(
            csv_path=str(train_csv),
            target_col=target_col,
            transform=train_transform,
            include_tabular=include_tabular,
            normalize_tabular=normalize_tabular,
            drop_missing_label=drop_missing_label,
        )

        self._num_tabular_features = self.train_dataset.num_tabular_features
        self._tabular_feature_cols = list(self.train_dataset.tabular_feature_cols)

        self.val_dataset = SOOPRegressionDataset(
            csv_path=str(valid_csv),
            target_col=target_col,
            transform=eval_transform,
            include_tabular=include_tabular,
            normalize_tabular=normalize_tabular,
            tabular_mean=self.train_dataset.tabular_mean,
            tabular_std=self.train_dataset.tabular_std,
            tabular_feature_cols=self._tabular_feature_cols,
            drop_missing_label=drop_missing_label,
        )

        self.test_dataset = SOOPRegressionDataset(
            csv_path=str(test_csv),
            target_col=target_col,
            transform=eval_transform,
            include_tabular=include_tabular,
            normalize_tabular=normalize_tabular,
            tabular_mean=self.train_dataset.tabular_mean,
            tabular_std=self.train_dataset.tabular_std,
            tabular_feature_cols=self._tabular_feature_cols,
            drop_missing_label=drop_missing_label,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.data_cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(self.data_cfg.get("num_workers", 4)),
            pin_memory=bool(self.data_cfg.get("pin_memory", False)),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.data_cfg.get("val_batch_size", 1)),
            shuffle=False,
            num_workers=int(self.data_cfg.get("num_workers", 4)),
            pin_memory=bool(self.data_cfg.get("pin_memory", False)),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.data_cfg.get("test_batch_size", 1)),
            shuffle=False,
            num_workers=int(self.data_cfg.get("num_workers", 4)),
            pin_memory=bool(self.data_cfg.get("pin_memory", False)),
        )
