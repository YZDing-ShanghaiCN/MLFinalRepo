from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset import RGBDManifestDataset
from src.data.manifest import read_class_to_idx, read_manifest_csv
from src.data.transforms import build_transform_from_config
from src.training.checkpoint import save_checkpoint
from src.training.evaluator import (
    evaluate_model,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_per_class_metrics_csv,
    save_predictions_csv,
)
from src.training.history import copy_config_snapshot, plot_history_curves, write_history_csv, write_json
from src.training.metrics import compute_classification_metrics
from src.utils.config import dataset_root_from_config, load_dataset_config
from src.utils.paths import project_root, resolve_project_path


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str = "auto") -> torch.device:
    """Select CUDA when requested/available, otherwise CPU."""
    normalized = device_name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(normalized)


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    """Return class names ordered by class index."""
    return [class_name for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def create_manifest_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    return_metadata: bool,
    dataset_config: dict[str, Any] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for any manifest using the phase-one Dataset."""
    config = load_dataset_config() if dataset_config is None else dataset_config
    dataset_root = dataset_root_from_config(config)
    transform = build_transform_from_config(config)
    dataset = RGBDManifestDataset(
        resolve_project_path(manifest_path),
        dataset_root=dataset_root,
        transform=transform,
        return_metadata=return_metadata,
        num_classes=int(config.get("dataset", {}).get("num_classes", 13)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
    )


def compute_class_weights(train_manifest: str | Path, class_to_idx: dict[str, int]) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized to mean 1."""
    rows = read_manifest_csv(resolve_project_path(train_manifest))
    counts = Counter(int(row["class_index"]) for row in rows)
    num_classes = len(class_to_idx)
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for class_index in range(num_classes):
        class_count = counts.get(class_index, 0)
        if class_count <= 0:
            raise ValueError(f"Cannot compute class weight for missing class index {class_index}")
        weights[class_index] = total / (num_classes * class_count)
    return weights / weights.mean()


def build_loss(
    train_config: dict[str, Any],
    train_manifest: str | Path,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[nn.Module, list[float] | None]:
    """Build CrossEntropyLoss with optional label smoothing and class weights."""
    loss_config = train_config.get("loss", {})
    if str(loss_config.get("type", "cross_entropy")) != "cross_entropy":
        raise ValueError(f"Unsupported loss type: {loss_config.get('type')}")
    class_weights: torch.Tensor | None = None
    weight_values: list[float] | None = None
    if bool(loss_config.get("use_class_weights", False)):
        class_weights = compute_class_weights(train_manifest, class_to_idx).to(device)
        weight_values = [float(value) for value in class_weights.detach().cpu().tolist()]
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
    )
    return criterion, weight_values


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_names: list[str],
    *,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    """Run one training epoch with end-to-end backpropagation."""
    model.train()
    total_loss = 0.0
    sample_count = 0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        preds = logits.argmax(dim=1)
        true_labels.extend(int(value) for value in labels.detach().cpu().tolist())
        pred_labels.extend(int(value) for value in preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(
        true_labels,
        pred_labels,
        num_classes=len(class_names),
        class_names=class_names,
    )
    return {
        "loss": total_loss / max(1, sample_count),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def _monitor_improved(value: float, best: float, mode: str) -> bool:
    if mode == "max":
        return value > best
    if mode == "min":
        return value < best
    raise ValueError(f"Unsupported monitor_mode: {mode}")


def _save_val_outputs(output_dir: Path, metrics: dict[str, Any], predictions: list[dict[str, Any]], class_names: list[str]) -> None:
    metric_summary = {
        key: value
        for key, value in metrics.items()
        if key not in {"per_class", "confusion_matrix"}
    }
    write_json(output_dir / "val_metrics.json", metrics)
    write_json(output_dir / "training_val_metric_summary.json", metric_summary)
    save_predictions_csv(output_dir / "val_predictions.csv", predictions)
    save_per_class_metrics_csv(output_dir / "val_per_class_metrics.csv", metrics["per_class"])
    save_confusion_matrix_csv(output_dir / "val_confusion_matrix.csv", metrics["confusion_matrix"], class_names)
    save_confusion_matrix_png(output_dir / "val_confusion_matrix.png", metrics["confusion_matrix"], class_names)


def _log(message: str, log_file: Path) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def train_model(
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    dataset_config: dict[str, Any],
    config_paths: list[str | Path],
) -> dict[str, Any]:
    """Run train/validation training and save experiment artifacts."""
    seed = int(train_config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    output_dir = resolve_project_path(train_config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train_log.txt"
    log_file.write_text("", encoding="utf-8")
    copy_config_snapshot(output_dir, config_paths)

    data_config = train_config.get("data", {})
    train_manifest = data_config["train_manifest"]
    val_manifest = data_config["val_manifest"]
    class_mapping = resolve_project_path(data_config["class_mapping"])
    class_to_idx = read_class_to_idx(class_mapping)
    class_names = class_names_from_mapping(class_to_idx)
    device = select_device(str(train_config.get("training", {}).get("device", "auto")))
    model.to(device)

    train_loader = create_manifest_loader(
        train_manifest,
        batch_size=int(data_config.get("train_batch_size", 32)),
        shuffle=True,
        return_metadata=False,
        dataset_config=dataset_config,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        seed=seed,
    )
    val_loader = create_manifest_loader(
        val_manifest,
        batch_size=int(data_config.get("val_batch_size", 64)),
        shuffle=False,
        return_metadata=True,
        dataset_config=dataset_config,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        seed=seed,
    )

    criterion, class_weights = build_loss(train_config, train_manifest, class_to_idx, device)
    optimizer_config = train_config.get("optimizer", {})
    if str(optimizer_config.get("type", "adamw")).lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_config.get('type')}")
    training_config = train_config.get("training", {})
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
        weight_decay=float(training_config.get("weight_decay", 0.0001)),
    )
    scheduler_config = train_config.get("scheduler", {})
    scheduler = None
    if str(scheduler_config.get("type", "reduce_on_plateau")) == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(training_config.get("monitor_mode", "max")),
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 2)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )

    monitor = str(training_config.get("monitor", "val_macro_f1"))
    monitor_mode = str(training_config.get("monitor_mode", "max"))
    best_metric = float("-inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    early_counter = 0
    history: list[dict[str, Any]] = []
    epochs = int(training_config.get("epochs", 30))
    clip_norm = float(training_config.get("gradient_clip_norm", 0.0))
    patience = int(training_config.get("early_stopping_patience", 7))
    save_best = bool(train_config.get("checkpoint", {}).get("save_best", True))
    save_last = bool(train_config.get("checkpoint", {}).get("save_last", True))

    _log(f"Device: {device}", log_file)
    _log(f"Train samples: {len(train_loader.dataset)}", log_file)
    _log(f"Val samples: {len(val_loader.dataset)}", log_file)
    if class_weights is not None:
        _log(f"Class weights: {class_weights}", log_file)

    last_val_metrics: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            class_names,
            gradient_clip_norm=clip_norm,
        )
        val_metrics, val_predictions = evaluate_model(model, val_loader, criterion, device, class_names)
        last_val_metrics = val_metrics
        current_lr = float(optimizer.param_groups[0]["lr"])
        monitor_value = float(val_metrics[monitor.replace("val_", "")] if monitor.startswith("val_") else val_metrics[monitor])
        if scheduler is not None:
            scheduler.step(monitor_value)

        improved = _monitor_improved(monitor_value, best_metric, monitor_mode)
        if improved:
            best_metric = monitor_value
            best_epoch = epoch
            early_counter = 0
            if save_best:
                save_checkpoint(
                    output_dir / "best_model.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    class_to_idx=class_to_idx,
                    model_config=model_config,
                    train_config=train_config,
                    random_seed=seed,
                )
                _save_val_outputs(output_dir, val_metrics, val_predictions, class_names)
        else:
            early_counter += 1

        if save_last:
            save_checkpoint(
                output_dir / "last_model.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_metric,
                class_to_idx=class_to_idx,
                model_config=model_config,
                train_config=train_config,
                random_seed=seed,
            )

        epoch_seconds = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr,
            "epoch_time_seconds": round(epoch_seconds, 4),
        }
        history.append(row)
        write_history_csv(output_dir / "history.csv", history)
        plot_history_curves(output_dir, history)

        _log(f"Epoch {epoch:02d}/{epochs}", log_file)
        _log(f"Train loss: {train_metrics['loss']:.6f}", log_file)
        _log(f"Train accuracy: {train_metrics['accuracy']:.6f}", log_file)
        _log(f"Train macro F1: {train_metrics['macro_f1']:.6f}", log_file)
        _log(f"Val loss: {val_metrics['loss']:.6f}", log_file)
        _log(f"Val accuracy: {val_metrics['accuracy']:.6f}", log_file)
        _log(f"Val macro F1: {val_metrics['macro_f1']:.6f}", log_file)
        _log(f"Learning rate: {current_lr:.8f}", log_file)
        _log(f"Best val macro F1: {best_metric:.6f}", log_file)
        _log(f"Early stopping counter: {early_counter}", log_file)

        if early_counter >= patience:
            _log("Early stopping triggered.", log_file)
            break

    summary = {
        "best_epoch": best_epoch,
        "best_metric_name": monitor,
        "best_metric": best_metric,
        "final_epoch": history[-1]["epoch"] if history else 0,
        "device": str(device),
        "class_weights": class_weights,
        "last_val_metrics": last_val_metrics,
        "history": history,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary
