from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml  # noqa: E402
from src.utils.paths import project_root, resolve_project_path  # noqa: E402


CONFIGS = [
    Path("configs/ablations/baseline_gap_dropout.yaml"),
    Path("configs/ablations/flatten_dropout.yaml"),
    Path("configs/ablations/gap_no_dropout.yaml"),
]

SUMMARY_COLUMNS = [
    "experiment",
    "pooling_type",
    "dropout",
    "learning_rate",
    "batch_size",
    "epochs",
    "seed",
    "parameter_count",
    "best_epoch",
    "best_val_accuracy",
    "best_val_loss",
    "best_val_macro_f1",
    "best_epoch_train_accuracy",
    "final_epoch_train_accuracy",
    "final_epoch_train_loss",
    "final_epoch_val_accuracy",
    "final_epoch_val_loss",
    "final_epoch_val_macro_f1",
    "training_time_seconds",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation metrics into ../outputs/ablation/ablation_summary.csv.")
    parser.add_argument("--output", type=Path, default=Path("../outputs/ablation/ablation_summary.csv"))
    return parser.parse_args()


def _base_row(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": config.get("experiment", {}).get("name", ""),
        "pooling_type": config.get("model", {}).get("pooling_type", ""),
        "dropout": config.get("model", {}).get("dropout", ""),
        "learning_rate": config.get("training", {}).get("learning_rate", ""),
        "batch_size": config.get("training", {}).get("batch_size", ""),
        "epochs": config.get("training", {}).get("epochs", ""),
        "seed": config.get("experiment", {}).get("seed", ""),
        "parameter_count": "",
        "best_epoch": "",
        "best_val_accuracy": "",
        "best_val_loss": "",
        "best_val_macro_f1": "",
        "best_epoch_train_accuracy": "",
        "final_epoch_train_accuracy": "",
        "final_epoch_train_loss": "",
        "final_epoch_val_accuracy": "",
        "final_epoch_val_loss": "",
        "final_epoch_val_macro_f1": "",
        "training_time_seconds": "",
        "status": "not_completed",
    }


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"metrics.json must contain an object: {path}")
    return data


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _checkpoint_val_macro_f1(checkpoint_path: Path) -> float:
    import torch
    from torch import nn

    from src.models.ablation_cnn import build_ablation_model
    from src.training.checkpoint import load_checkpoint
    from src.training.evaluator import evaluate_model
    from src.training.trainer import class_names_from_mapping, create_manifest_loader, select_device
    from src.utils.config import load_dataset_config

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    train_config = checkpoint["train_config"]
    ablation_model = model_config["ablation"]
    class_to_idx = {str(key): int(value) for key, value in checkpoint["class_to_idx"].items()}
    class_names = class_names_from_mapping(class_to_idx)
    device = select_device(str(train_config.get("training", {}).get("device", "auto")))

    model = build_ablation_model(
        model_config,
        pooling_type=str(ablation_model["pooling_type"]),
        dropout=float(ablation_model["dropout"]),
        num_classes=int(model_config.get("model", {}).get("num_classes", len(class_to_idx))),
        input_size=int(ablation_model["input_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    data_config = train_config["data"]
    dataset_config = load_dataset_config()
    loader = create_manifest_loader(
        data_config["val_manifest"],
        batch_size=int(data_config.get("val_batch_size", 32)),
        shuffle=False,
        return_metadata=False,
        dataset_config=dataset_config,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_config.get("loss", {}).get("label_smoothing", 0.0))).to(device)
    with torch.no_grad():
        val_metrics, _ = evaluate_model(model, loader, criterion, device, class_names)
    return float(val_metrics["macro_f1"])


def _fill_missing_validation_f1(output_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    updated = dict(metrics)
    changed = False
    if "best_val_macro_f1" not in updated:
        best_checkpoint = output_dir / "best_model.pt"
        if best_checkpoint.is_file():
            updated["best_val_macro_f1"] = _checkpoint_val_macro_f1(best_checkpoint)
            changed = True
    if "final_epoch_val_macro_f1" not in updated:
        last_checkpoint = output_dir / "last_model.pt"
        if last_checkpoint.is_file():
            updated["final_epoch_val_macro_f1"] = _checkpoint_val_macro_f1(last_checkpoint)
            changed = True
    if changed:
        _write_metrics(output_dir / "metrics.json", updated)
    return updated


def build_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in CONFIGS:
        config = load_yaml(config_path)
        row = _base_row(config)
        output_dir = resolve_project_path(config["experiment"]["output_dir"])
        metrics = _read_metrics(output_dir / "metrics.json")
        if metrics and metrics.get("status") == "completed":
            metrics = _fill_missing_validation_f1(output_dir, metrics)
            for column in SUMMARY_COLUMNS:
                if column in metrics:
                    row[column] = metrics[column]
            row["status"] = "completed"
        rows.append(row)
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in SUMMARY_COLUMNS} for row in rows)
    return output_path


def main() -> None:
    args = parse_args()
    output = write_summary(args.output, build_summary_rows())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
