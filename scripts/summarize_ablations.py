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
    "best_epoch_train_accuracy",
    "final_epoch_train_accuracy",
    "final_epoch_train_loss",
    "final_epoch_val_accuracy",
    "final_epoch_val_loss",
    "training_time_seconds",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation metrics into output/ablation/ablation_summary.csv.")
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
        "best_epoch_train_accuracy": "",
        "final_epoch_train_accuracy": "",
        "final_epoch_train_loss": "",
        "final_epoch_val_accuracy": "",
        "final_epoch_val_loss": "",
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


def build_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in CONFIGS:
        config = load_yaml(config_path)
        row = _base_row(config)
        output_dir = resolve_project_path(config["experiment"]["output_dir"])
        metrics = _read_metrics(output_dir / "metrics.json")
        if metrics and metrics.get("status") == "completed":
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
