from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def compute_topk_accuracy(y_true: Sequence[int], topk_indices: Sequence[Sequence[int]], *, k: int = 3) -> float:
    if not y_true:
        return 0.0
    correct = 0
    for true_index, predicted_indices in zip(y_true, topk_indices):
        if int(true_index) in [int(value) for value in predicted_indices[:k]]:
            correct += 1
    return correct / len(y_true)


def compute_core_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    topk_indices: Sequence[Sequence[int]],
    *,
    test_loss: float,
    num_classes: int,
) -> dict[str, float]:
    labels = list(range(num_classes))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    top_k = min(3, num_classes)
    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_precision": float(macro_precision),
        "test_macro_recall": float(macro_recall),
        "test_macro_f1": float(macro_f1),
        "test_weighted_precision": float(weighted_precision),
        "test_weighted_recall": float(weighted_recall),
        "test_weighted_f1": float(weighted_f1),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_top3_accuracy": float(compute_topk_accuracy(y_true, topk_indices, k=top_k)),
    }


def compute_generalization_metrics(
    *,
    best_epoch_train_accuracy: float,
    best_val_accuracy: float,
    best_val_loss: float,
    best_val_macro_f1: float,
    test_accuracy: float,
    test_loss: float,
    test_macro_f1: float,
) -> dict[str, float]:
    accuracy_gap = float(best_val_accuracy) - float(test_accuracy)
    macro_f1_gap = float(best_val_macro_f1) - float(test_macro_f1)
    relative_accuracy_drop = safe_divide(accuracy_gap, float(best_val_accuracy))
    return {
        "accuracy_generalization_gap": accuracy_gap,
        "accuracy_generalization_gap_percentage_points": accuracy_gap * 100.0,
        "accuracy_retention_rate": safe_divide(float(test_accuracy), float(best_val_accuracy)),
        "accuracy_retention_rate_percent": safe_divide(float(test_accuracy), float(best_val_accuracy)) * 100.0,
        "macro_f1_generalization_gap": macro_f1_gap,
        "macro_f1_retention_rate": safe_divide(float(test_macro_f1), float(best_val_macro_f1)),
        "loss_generalization_change": float(test_loss) - float(best_val_loss),
        "train_test_accuracy_gap": float(best_epoch_train_accuracy) - float(test_accuracy),
        "relative_accuracy_drop": relative_accuracy_drop,
        "relative_accuracy_drop_percent": relative_accuracy_drop * 100.0,
    }


def classification_report_rows(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    labels = list(range(len(class_names)))
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    rows: list[dict[str, Any]] = [
        {
            "class_index": index,
            "class_name": class_names[index],
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1_score": float(per_f1[index]),
            "support": int(per_support[index]),
        }
        for index in labels
    ]
    rows.extend(
        [
            {
                "class_index": "",
                "class_name": "accuracy",
                "precision": "",
                "recall": "",
                "f1_score": float(accuracy_score(y_true, y_pred)),
                "support": len(y_true),
            },
            {
                "class_index": "",
                "class_name": "macro_avg",
                "precision": float(macro_precision),
                "recall": float(macro_recall),
                "f1_score": float(macro_f1),
                "support": len(y_true),
            },
            {
                "class_index": "",
                "class_name": "weighted_avg",
                "precision": float(weighted_precision),
                "recall": float(weighted_recall),
                "f1_score": float(weighted_f1),
                "support": len(y_true),
            },
        ]
    )
    return rows


def confusion_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int], int] = {}
    for true_index, pred_index in zip(y_true, y_pred):
        if int(true_index) == int(pred_index):
            continue
        key = (int(true_index), int(pred_index))
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "true_class_name": class_names[true_index],
            "predicted_class_name": class_names[pred_index],
            "count": count,
        }
        for (true_index, pred_index), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def plot_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    normalized: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = matrix.astype(float)
    value_format = ".2f" if normalized else "d"
    if normalized:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums != 0)

    size = max(8, min(18, len(class_names) * 0.75))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(display, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    threshold = float(display.max()) / 2.0 if display.size and display.max() > 0 else 0.0
    raw_matrix = matrix.astype(int)
    for row_index in range(display.shape[0]):
        for col_index in range(display.shape[1]):
            value = display[row_index, col_index]
            text = format(value, value_format) if normalized else str(int(raw_matrix[row_index, col_index]))
            axis.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix_array(y_true: Sequence[int], y_pred: Sequence[int], *, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(int)
