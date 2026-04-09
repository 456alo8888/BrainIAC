import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from soop_dataset import SOOPRegressionDataset
from train_lightning_soop_regression import SOOPRegressionLightningModule, compute_regression_metrics
from dataset import get_validation_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BrainIAC SOOP regression checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--include-tabular", action="store_true")
    parser.add_argument("--no-include-tabular", dest="include_tabular", action="store_false")
    parser.set_defaults(include_tabular=None)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-use-wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=None)
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.target_col:
        config["data"]["target_col"] = args.target_col
    if args.include_tabular is not None:
        config["data"]["include_tabular"] = args.include_tabular

    image_size = tuple(config["data"].get("size", [96, 96, 96]))
    include_tabular = bool(config["data"].get("include_tabular", False))
    normalize_tabular = bool(config["data"].get("normalize_tabular", False))

    fold_dir = Path(config["data"]["fold_dir"])
    train_csv = fold_dir / "train.csv"

    train_df = pd.read_csv(train_csv)
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
        "gs_rankin+6isdeath",
    }
    tabular_cols = [c for c in train_df.columns if c not in excluded]

    tabular_mean = None
    tabular_std = None
    if include_tabular and tabular_cols and normalize_tabular:
        train_tab = train_df[tabular_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        tabular_mean = train_tab.mean(axis=0, keepdims=True)
        tabular_std = train_tab.std(axis=0, keepdims=True)
        tabular_std = np.where(tabular_std == 0.0, 1.0, tabular_std)

    dataset = SOOPRegressionDataset(
        csv_path=args.split_csv,
        target_col=config["data"].get("target_col", "gs_rankin_6isdeath"),
        transform=get_validation_transform(image_size=image_size),
        include_tabular=include_tabular,
        normalize_tabular=normalize_tabular,
        tabular_feature_cols=tabular_cols if include_tabular else None,
        tabular_mean=tabular_mean,
        tabular_std=tabular_std,
        drop_missing_label=True,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SOOPRegressionLightningModule(config=config, num_tabular_features=dataset.num_tabular_features)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    rows = []
    y_true = []
    y_pred = []
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)
            tabular = batch.get("tabular")
            if tabular is not None:
                tabular = tabular.to(device)

            pred = model.forward(image, tabular).squeeze(1)
            loss = torch.mean((pred - target) ** 2)

            pred_np = pred.detach().cpu().numpy()
            true_np = target.detach().cpu().numpy()
            losses.append(float(loss.detach().cpu().item()))

            subject_ids = batch["subject_id"]
            for subject_id, gt, pdv in zip(subject_ids, true_np, pred_np):
                rows.append(
                    {
                        "subject_id": str(subject_id),
                        "target": float(gt),
                        "prediction": float(pdv),
                        "abs_error": float(abs(pdv - gt)),
                        "squared_error": float((pdv - gt) ** 2),
                    }
                )
            y_true.append(true_np)
            y_pred.append(pred_np)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_csv, index=False)

    result_json = output_dir / "results_eval_soop_regression.json"
    payload = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "device": str(device),
        },
        "data_config": {
            "split_csv": args.split_csv,
            "target_col": config["data"].get("target_col", "gs_rankin_6isdeath"),
            "include_tabular": include_tabular,
            "num_tabular_features": int(dataset.num_tabular_features),
        },
        "n_samples": int(len(rows)),
        "metrics": metrics,
        "artifacts": {
            "predictions_csv": str(pred_csv),
        },
    }

    with result_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    config_use_wandb = bool(config.get("train", {}).get("use_wandb", False))
    use_wandb = config_use_wandb if args.use_wandb is None else bool(args.use_wandb)
    if use_wandb:
        import wandb

        project_name = args.project_name or config.get("logger", {}).get("project_name", "brainiac-soop-outcome")
        default_eval_name = f"{config.get('logger', {}).get('run_name', 'soop-regression')}-test"
        run_name = args.run_name or default_eval_name

        run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                "mode": "evaluation",
                "checkpoint": str(args.checkpoint),
                "split_csv": str(args.split_csv),
                "target_col": config["data"].get("target_col", "gs_rankin_6isdeath"),
                "include_tabular": include_tabular,
                "num_tabular_features": int(dataset.num_tabular_features),
            },
            reinit=True,
        )

        wandb.log({f"test_{k}": float(v) for k, v in metrics.items()})
        wandb.log({"test_n_samples": int(len(rows))})
        wandb.summary["test_predictions_csv"] = str(pred_csv)
        wandb.summary["test_results_json"] = str(result_json)
        artifact = wandb.Artifact(name=f"{run_name}-eval", type="evaluation")
        artifact.add_file(str(pred_csv))
        artifact.add_file(str(result_json))
        run.log_artifact(artifact)
        run.finish()

    print(f"Saved predictions: {pred_csv}")
    print(f"Saved results: {result_json}")


if __name__ == "__main__":
    main()
