import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


SCRIPT_DIR = Path(__file__).resolve().parent


def inspect_backbone_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state_dict).__name__}")

    keys = [key for key in state_dict.keys() if isinstance(key, str)]
    backbone_keys = [key for key in keys if key.startswith("backbone.")]
    cross_attn_keys = [
        key for key in backbone_keys if ("cross_attn" in key or "norm_cross_attn" in key)
    ]

    if len(backbone_keys) == 0:
        raise ValueError(
            "Checkpoint has no 'backbone.*' keys and cannot be used as BrainIAC backbone. "
            f"checkpoint={path}"
        )

    if len(cross_attn_keys) == 0:
        suggestion = path.parent / "BrainIAC_mock.ckpt"
        suggestion_msg = f" Suggested compatible checkpoint: {suggestion}" if suggestion.exists() else ""
        raise ValueError(
            "Checkpoint backbone appears incompatible with current ViT schema: "
            "missing cross-attention keys ('cross_attn'/'norm_cross_attn'). "
            f"checkpoint={path}, total_keys={len(keys)}, backbone_keys={len(backbone_keys)}."
            f"{suggestion_msg}"
        )

    return {
        "path": str(path),
        "total_keys": len(keys),
        "backbone_keys": len(backbone_keys),
        "cross_attn_keys": len(cross_attn_keys),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    diff = y_pred - y_true

    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    eps = 1e-8
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), eps)) * 100.0)

    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - (ss_res / max(ss_tot, eps)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "loss": mse,
    }


class SOOPRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, use_mlp: bool):
        super().__init__()
        if use_mlp:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x)


class SOOPRegressionLightningModule(pl.LightningModule):
    def __init__(self, config: Dict, num_tabular_features: int):
        super().__init__()
        self.save_hyperparameters({"config": config, "num_tabular_features": num_tabular_features})
        self.config = config
        self.num_tabular_features = int(num_tabular_features)

        self.include_tabular = bool(config["data"].get("include_tabular", False))
        self.normalize_features = bool(config["model"].get("normalize_features", False))

        from model import ViTBackboneNet

        self.backbone = ViTBackboneNet(simclr_ckpt_path=config["simclrvit"]["ckpt_path"])

        if self.include_tabular and self.num_tabular_features <= 0:
            raise ValueError("include_tabular=True but num_tabular_features is 0")

        input_dim = 768 + (self.num_tabular_features if self.include_tabular else 0)
        hidden_dim = int(config["model"].get("hidden_dim_tabular_head", 256))
        dropout = float(config["model"].get("dropout", 0.2))
        use_mlp = self.include_tabular

        self.reg_head = SOOPRegressionHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, use_mlp=use_mlp)
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []

        if str(config["train"].get("freeze_backbone", "no")).lower() == "yes":
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen!!")

    def forward(self, image, tabular=None):
        img_feat = self.backbone(image)
        if self.normalize_features:
            img_feat = F.normalize(img_feat, p=2, dim=1)

        if self.include_tabular:
            if tabular is None:
                raise ValueError("Tabular tensor is required when include_tabular=True")
            fused = torch.cat([img_feat, tabular], dim=1)
        else:
            fused = img_feat

        return self.reg_head(fused)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["label"].unsqueeze(1)
        tabular = batch.get("tabular")

        pred = self.forward(image, tabular)
        loss = self.criterion(pred, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["label"].unsqueeze(1)
        tabular = batch.get("tabular")

        pred = self.forward(image, tabular)
        loss = self.criterion(pred, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(
            {
                "target": target.detach().cpu(),
                "pred": pred.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        y_true = torch.cat([o["target"] for o in self.validation_step_outputs], dim=0).numpy().flatten()
        y_pred = torch.cat([o["pred"] for o in self.validation_step_outputs], dim=0).numpy().flatten()
        metrics = compute_regression_metrics(y_true, y_pred)

        self.log("val_mse", metrics["mse"], prog_bar=True)
        self.log("val_rmse", metrics["rmse"], prog_bar=True)
        self.log("val_mae", metrics["mae"], prog_bar=True)
        self.log("val_mape", metrics["mape"], prog_bar=False)
        self.log("val_r2", metrics["r2"], prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optim_cfg = self.config["optim"]
        optimizer_name = str(optim_cfg.get("optimizer", "adamw")).lower()
        lr = float(optim_cfg.get("lr", 8e-4))
        weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        momentum = float(optim_cfg.get("momentum", 0.9))

        params = filter(lambda p: p.requires_grad, self.parameters())

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mae"}


def load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_file():
        candidate = SCRIPT_DIR / path
        if candidate.is_file():
            config_path = candidate
        else:
            raise FileNotFoundError(f"Config not found: {path} (also checked {candidate})")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    if args.fold_dir:
        config["data"]["fold_dir"] = args.fold_dir
    if args.target_col:
        config["data"]["target_col"] = args.target_col
    if args.include_tabular is not None:
        config["data"]["include_tabular"] = args.include_tabular

    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    if args.max_epochs is not None:
        config["model"]["max_epochs"] = args.max_epochs
    if args.normalize_features is not None:
        config["model"]["normalize_features"] = args.normalize_features

    if args.output_dir:
        config["logger"]["output_dir"] = args.output_dir
        config["logger"]["save_dir"] = str(Path(args.output_dir) / "checkpoints")
    if args.run_name:
        config["logger"]["run_name"] = args.run_name
    if args.project_name:
        config["logger"]["project_name"] = args.project_name

    if args.ckpt_path:
        config["simclrvit"]["ckpt_path"] = args.ckpt_path

    if args.optimizer:
        config["optim"]["optimizer"] = args.optimizer
    if args.learning_rate is not None:
        config["optim"]["lr"] = args.learning_rate
    if args.weight_decay is not None:
        config["optim"]["weight_decay"] = args.weight_decay

    if args.freeze_backbone is not None:
        config["train"]["freeze_backbone"] = "yes" if args.freeze_backbone else "no"
    if args.grad_clip_norm is not None:
        config["train"]["grad_clip_norm"] = args.grad_clip_norm
    if args.precision:
        config["train"]["precision"] = args.precision
    if args.devices is not None:
        config["train"]["devices"] = args.devices
    if args.accelerator:
        config["train"]["accelerator"] = args.accelerator
    if args.use_wandb is not None:
        config["train"]["use_wandb"] = args.use_wandb

    if args.limit_train_batches is not None:
        config["train"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config["train"]["limit_val_batches"] = args.limit_val_batches

    if args.visible_device is not None:
        config.setdefault("gpu", {})["visible_device"] = args.visible_device

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BrainIAC SOOP regression model")
    parser.add_argument("--config", type=str, default="config_soop_regression.yml")

    parser.add_argument("--fold-dir", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--include-tabular", action="store_true")
    parser.add_argument("--no-include-tabular", dest="include_tabular", action="store_false")
    parser.set_defaults(include_tabular=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project-name", type=str, default=None)

    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "adam", "sgd"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=None)

    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--no-normalize-features", dest="normalize_features", action="store_false")
    parser.set_defaults(normalize_features=None)

    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-use-wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=None)

    parser.add_argument("--limit-train-batches", type=float, default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--visible-device", type=str, default=None)
    parser.add_argument("--validate-checkpoint-only", action="store_true")

    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision("medium")
    args = parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args)

    checkpoint_stats = inspect_backbone_checkpoint(config["simclrvit"]["ckpt_path"])
    print(
        "Checkpoint validation passed: "
        f"path={checkpoint_stats['path']} "
        f"total_keys={checkpoint_stats['total_keys']} "
        f"backbone_keys={checkpoint_stats['backbone_keys']} "
        f"cross_attn_keys={checkpoint_stats['cross_attn_keys']}"
    )

    if args.validate_checkpoint_only:
        return

    visible_device = str(config.get("gpu", {}).get("visible_device", "")).strip()
    if visible_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

    output_dir = Path(config["logger"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(config["logger"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config_soop_regression.yml"
    with resolved_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    if int(config["train"].get("seed", 42)) >= 0:
        pl.seed_everything(int(config["train"].get("seed", 42)), workers=True)

    from soop_dataset import SOOPRegressionDataModule

    data_module = SOOPRegressionDataModule(config)
    data_module.setup("fit")

    model = SOOPRegressionLightningModule(config=config, num_tabular_features=data_module.num_tabular_features)

    use_wandb = bool(config["train"].get("use_wandb", False))
    logger = False
    if use_wandb:
        logger = WandbLogger(
            project=config["logger"]["project_name"],
            name=config["logger"]["run_name"],
            config=config,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        filename=config["logger"].get("save_name", "best-model-{epoch:02d}-{val_mae:.4f}"),
        monitor="val_mae",
        mode="min",
        save_top_k=1,
    )
    callbacks = [checkpoint_callback]
    if use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = pl.Trainer(
        max_epochs=int(config["model"].get("max_epochs", 10)),
        logger=logger,
        callbacks=callbacks,
        accelerator=config["train"].get("accelerator", "gpu"),
        devices=int(config["train"].get("devices", 1)),
        precision=config["train"].get("precision", "16-mixed"),
        gradient_clip_val=float(config["train"].get("grad_clip_norm", 0.0)),
        limit_train_batches=float(config["train"].get("limit_train_batches", 1.0)),
        limit_val_batches=float(config["train"].get("limit_val_batches", 1.0)),
    )

    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_model_path}")
    print(f"Resolved config: {resolved_config_path}")


if __name__ == "__main__":
    main()
