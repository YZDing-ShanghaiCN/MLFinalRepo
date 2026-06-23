from __future__ import annotations

import pytest

from src.training.metrics import compute_classification_metrics


def test_metrics_for_constructed_data() -> None:
    y_true = [0, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 2, 3, 0]
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=13)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 13
    assert len(metrics["confusion_matrix"][0]) == 13
    assert len(metrics["per_class"]) == 13


def test_metrics_zero_division_does_not_fail() -> None:
    metrics = compute_classification_metrics([0, 1, 2], [0, 0, 0], num_classes=13)
    assert metrics["macro_precision"] >= 0.0
    assert metrics["macro_recall"] >= 0.0
    assert metrics["macro_f1"] >= 0.0
