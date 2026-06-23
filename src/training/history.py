from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.utils.paths import resolve_project_path


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    """Write epoch history to CSV."""
    if not history:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with UTF-8 encoding."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def plot_history_curves(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    """Save loss, accuracy, macro-F1, and learning-rate curves."""
    if not history:
        return
    output = Path(output_dir)
    epochs = [int(row["epoch"]) for row in history]
    curve_specs = [
        ("loss_curve.png", [("train_loss", "train"), ("val_loss", "val")], "Loss"),
        ("accuracy_curve.png", [("train_accuracy", "train"), ("val_accuracy", "val")], "Accuracy"),
        ("macro_f1_curve.png", [("train_macro_f1", "train"), ("val_macro_f1", "val")], "Macro F1"),
        ("learning_rate_curve.png", [("learning_rate", "lr")], "Learning Rate"),
    ]
    for filename, series, ylabel in curve_specs:
        fig, axis = plt.subplots(figsize=(6, 4))
        for key, label in series:
            axis.plot(epochs, [float(row[key]) for row in history], marker="o", label=label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def copy_config_snapshot(output_dir: str | Path, config_paths: list[str | Path]) -> None:
    """Copy config files into an experiment snapshot directory."""
    snapshot_dir = Path(output_dir) / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for config_path in config_paths:
        source = resolve_project_path(config_path)
        if source.is_file():
            shutil.copy2(source, snapshot_dir / source.name)
