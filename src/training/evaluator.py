from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.training.metrics import compute_classification_metrics


def _metadata_item(metadata: dict[str, Any], key: str, index: int) -> Any:
    value = metadata[key]
    if torch.is_tensor(value):
        return value[index].item()
    return value[index]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a model without gradient tracking."""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                metadata = None
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

            probabilities = torch.softmax(logits, dim=1)
            confidences, preds = probabilities.max(dim=1)
            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()
            true_labels.extend(int(value) for value in labels_cpu.tolist())
            pred_labels.extend(int(value) for value in preds_cpu.tolist())

            for index in range(batch_size):
                true_index = int(labels_cpu[index].item())
                pred_index = int(preds_cpu[index].item())
                row = {
                    "true_index": true_index,
                    "true_class": class_names[true_index],
                    "predicted_index": pred_index,
                    "predicted_class": class_names[pred_index],
                    "confidence": float(confidences[index].detach().cpu().item()),
                    "correct": true_index == pred_index,
                }
                if metadata is not None:
                    row["image_path"] = str(_metadata_item(metadata, "path", index))
                    row["instance_name"] = str(_metadata_item(metadata, "instance_name", index))
                predictions.append(row)

    metrics = compute_classification_metrics(
        true_labels,
        pred_labels,
        num_classes=len(class_names),
        class_names=class_names,
    )
    metrics["loss"] = total_loss / max(1, sample_count)
    metrics["num_samples"] = sample_count
    return metrics, predictions


def save_predictions_csv(path: str | Path, predictions: Iterable[dict[str, Any]]) -> None:
    """Save prediction rows to CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(predictions)
    fieldnames = [
        "image_path",
        "instance_name",
        "true_class",
        "true_index",
        "predicted_class",
        "predicted_index",
        "confidence",
        "correct",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_per_class_metrics_csv(path: str | Path, per_class: list[dict[str, Any]]) -> None:
    """Save per-class metrics to CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class_index", "class_name", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(per_class)


def save_confusion_matrix_csv(path: str | Path, matrix: list[list[int]], class_names: Sequence[str]) -> None:
    """Save confusion matrix as CSV with true rows and predicted columns."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def save_confusion_matrix_png(path: str | Path, matrix: list[list[int]], class_names: Sequence[str]) -> None:
    """Render and save a labeled confusion matrix image."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix, dtype=int)
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(array, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    threshold = array.max() / 2 if array.size and array.max() > 0 else 0
    for row_index in range(array.shape[0]):
        for col_index in range(array.shape[1]):
            value = int(array[row_index, col_index])
            axis.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
