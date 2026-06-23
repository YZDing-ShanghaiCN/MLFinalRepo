from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro metrics, per-class metrics, and confusion matrix."""
    labels = list(range(num_classes))
    names = list(class_names) if class_names is not None else [str(index) for index in labels]
    true_array = np.asarray(list(y_true), dtype=int)
    pred_array = np.asarray(list(y_pred), dtype=int)
    if true_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set")

    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        true_array,
        pred_array,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_array,
        pred_array,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(true_array, pred_array, labels=labels)
    return {
        "accuracy": float(accuracy_score(true_array, pred_array)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "class_index": index,
                "class_name": names[index],
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }
            for index in labels
        ],
        "confusion_matrix": matrix.astype(int).tolist(),
    }
