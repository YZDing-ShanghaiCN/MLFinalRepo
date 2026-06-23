from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import read_class_to_idx, read_manifest_csv  # noqa: E402
from src.models.rgb_classifier import build_model_from_config  # noqa: E402
from src.training.trainer import train_model  # noqa: E402
from src.utils.config import load_dataset_config, load_model_config, load_train_config  # noqa: E402
from src.utils.paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline CNN-GAP classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs without editing config.")
    return parser.parse_args()


def _check_manifest(path_value: str, label: str) -> Path:
    path = resolve_project_path(path_value)
    if not path.is_file():
        if label in {"train_sub", "val"}:
            raise FileNotFoundError(f"{label} manifest not found: {path}. Run python scripts/build_validation_split.py first.")
        raise FileNotFoundError(f"{label} manifest not found: {path}")
    return path


def _instances(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["class_name"], row["instance_name"]) for row in rows}


def _paths(rows: list[dict[str, str]]) -> set[str]:
    return {row["image_path"] for row in rows}


def validate_training_inputs(train_config: dict, model_config: dict) -> None:
    """Validate manifests, class mapping, no leakage, and model output dimension."""
    data = train_config["data"]
    train_sub_path = _check_manifest(data["train_manifest"], "train_sub")
    val_path = _check_manifest(data["val_manifest"], "val")
    class_mapping = resolve_project_path(data["class_mapping"])
    if not class_mapping.is_file():
        raise FileNotFoundError(f"class_to_idx not found: {class_mapping}")
    class_to_idx = read_class_to_idx(class_mapping)
    if len(class_to_idx) != 13:
        raise ValueError(f"Expected 13 classes, found {len(class_to_idx)}")
    train_rows = read_manifest_csv(train_sub_path)
    val_rows = read_manifest_csv(val_path)
    if _instances(train_rows) & _instances(val_rows):
        raise RuntimeError("train_sub and val have overlapping instances")
    if _paths(train_rows) & _paths(val_rows):
        raise RuntimeError("train_sub and val have overlapping image paths")
    model = build_model_from_config(model_config)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 128, 128))
    if tuple(logits.shape) != (2, 13):
        raise RuntimeError(f"Model output shape must be (2, 13), got {tuple(logits.shape)}")


def main() -> None:
    args = parse_args()
    train_config = load_train_config(args.config)
    train_config = copy.deepcopy(train_config)
    if args.epochs is not None:
        train_config.setdefault("training", {})["epochs"] = args.epochs
    model_config = load_model_config(args.model_config)
    dataset_config = load_dataset_config(args.dataset_config)
    validate_training_inputs(train_config, model_config)
    model = build_model_from_config(model_config)
    summary = train_model(
        model,
        model_config=model_config,
        train_config=train_config,
        dataset_config=dataset_config,
        config_paths=[args.model_config, args.config, args.dataset_config, "configs/validation_split.yaml"],
    )
    print(f"Best checkpoint: {resolve_project_path(train_config['experiment']['output_dir']) / 'best_model.pt'}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best {summary['best_metric_name']}: {summary['best_metric']:.6f}")


if __name__ == "__main__":
    main()
