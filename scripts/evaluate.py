from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rgb_classifier import build_model_from_config  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.evaluator import (  # noqa: E402
    evaluate_model,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_per_class_metrics_csv,
    save_predictions_csv,
)
from src.training.history import write_json  # noqa: E402
from src.training.trainer import class_names_from_mapping, create_manifest_loader, select_device  # noqa: E402
from src.utils.config import load_dataset_config  # noqa: E402
from src.utils.paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on val or test.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    train_config = checkpoint["train_config"]
    class_to_idx = {str(key): int(value) for key, value in checkpoint["class_to_idx"].items()}
    class_names = class_names_from_mapping(class_to_idx)
    device = select_device(str(train_config.get("training", {}).get("device", "auto")))
    model = build_model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    data_config = train_config["data"]
    manifest_key = "val_manifest" if args.split == "val" else "test_manifest"
    manifest_path = data_config[manifest_key]
    batch_key = "val_batch_size" if args.split == "val" else "test_batch_size"
    dataset_config = load_dataset_config()
    loader = create_manifest_loader(
        manifest_path,
        batch_size=int(data_config.get(batch_key, 64)),
        shuffle=False,
        return_metadata=True,
        dataset_config=dataset_config,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_config.get("loss", {}).get("label_smoothing", 0.0)))
    criterion = criterion.to(device)
    metrics, predictions = evaluate_model(model, loader, criterion, device, class_names)

    output_dir = resolve_project_path(train_config["experiment"]["output_dir"])
    prefix = args.split
    write_json(output_dir / f"{prefix}_metrics.json", metrics)
    save_predictions_csv(output_dir / f"{prefix}_predictions.csv", predictions)
    save_per_class_metrics_csv(output_dir / f"{prefix}_per_class_metrics.csv", metrics["per_class"])
    save_confusion_matrix_csv(output_dir / f"{prefix}_confusion_matrix.csv", metrics["confusion_matrix"], class_names)
    save_confusion_matrix_png(output_dir / f"{prefix}_confusion_matrix.png", metrics["confusion_matrix"], class_names)

    print(f"Split: {args.split}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro precision: {metrics['macro_precision']:.6f}")
    print(f"Macro recall: {metrics['macro_recall']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    print(f"Wrote {output_dir / f'{prefix}_metrics.json'}")


if __name__ == "__main__":
    main()
