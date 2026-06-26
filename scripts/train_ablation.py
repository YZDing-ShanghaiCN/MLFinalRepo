from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import read_class_to_idx, read_manifest_csv  # noqa: E402
from src.models.ablation_cnn import build_ablation_model, count_parameters  # noqa: E402
from src.training.checkpoint import save_checkpoint  # noqa: E402
from src.training.evaluator import evaluate_model  # noqa: E402
from src.training.history import write_json  # noqa: E402
from src.training.trainer import (  # noqa: E402
    build_loss,
    class_names_from_mapping,
    create_manifest_loader,
    select_device,
    set_seed,
    train_one_epoch,
)
from src.utils.config import load_dataset_config, load_model_config, load_train_config, load_yaml  # noqa: E402
from src.utils.paths import project_root, resolve_project_path  # noqa: E402


REQUIRED_EXPERIMENTS = {
    "baseline_gap_dropout": ("gap", 0.3),
    "flatten_dropout": ("flatten", 0.3),
    "gap_no_dropout": ("gap", 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one validation-only ablation experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to an ablation YAML config.")
    return parser.parse_args()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ablation_root() -> Path:
    return project_root() / "output" / "ablation"


def _log(message: str, log_file: Path) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fieldnames = ["epoch", "learning_rate", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row[field] for field in fieldnames})


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def _check_manifest(path_value: str, expected_name: str) -> Path:
    path = resolve_project_path(path_value)
    expected = project_root() / "metadata" / "manifests" / expected_name
    if path.resolve() != expected.resolve():
        raise ValueError(f"Ablations may only use {expected_name}, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    return path


def _instances(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["class_name"], row["instance_name"]) for row in rows}


def _paths(rows: list[dict[str, str]]) -> set[str]:
    return {row["image_path"] for row in rows}


def _validate_inputs(train_config: dict[str, Any]) -> dict[str, int]:
    data_config = train_config.get("data", {})
    if any("test" in key.lower() for key in data_config):
        raise ValueError("Ablation training config must not contain test data keys.")

    train_path = _check_manifest(str(data_config["train_manifest"]), "train_sub.csv")
    val_path = _check_manifest(str(data_config["val_manifest"]), "val.csv")
    class_mapping = resolve_project_path(data_config["class_mapping"])
    if not class_mapping.is_file():
        raise FileNotFoundError(f"class_to_idx not found: {class_mapping}")
    class_to_idx = read_class_to_idx(class_mapping)
    if len(class_to_idx) != 13:
        raise ValueError(f"Expected 13 classes, found {len(class_to_idx)}")

    train_rows = read_manifest_csv(train_path)
    val_rows = read_manifest_csv(val_path)
    if _instances(train_rows) & _instances(val_rows):
        raise RuntimeError("train_sub and val have overlapping instances")
    if _paths(train_rows) & _paths(val_rows):
        raise RuntimeError("train_sub and val have overlapping image paths")
    return class_to_idx


def _resolve_config(ablation_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_configs = ablation_config.get("base_configs", {})
    train_config = copy.deepcopy(load_train_config(base_configs.get("train", "configs/train.yaml")))
    model_config = copy.deepcopy(load_model_config(base_configs.get("model", "configs/model.yaml")))
    dataset_config = copy.deepcopy(load_dataset_config(base_configs.get("dataset", "configs/dataset.yaml")))

    experiment = ablation_config.get("experiment", {})
    model = ablation_config.get("model", {})
    training = ablation_config.get("training", {})
    data = ablation_config.get("data", {})

    name = str(experiment.get("name", "")).strip()
    if name not in REQUIRED_EXPERIMENTS:
        raise ValueError(f"Unsupported ablation experiment: {name}")
    expected_pooling, expected_dropout = REQUIRED_EXPERIMENTS[name]
    if str(model.get("pooling_type", "")).lower() != expected_pooling:
        raise ValueError(f"{name} must use pooling_type={expected_pooling}")
    if float(model.get("dropout", -1.0)) != expected_dropout:
        raise ValueError(f"{name} must use dropout={expected_dropout}")

    output_dir = resolve_project_path(str(experiment.get("output_dir", "")))
    if not _is_relative_to(output_dir, _ablation_root()):
        raise ValueError(f"Ablation output_dir must be inside {_ablation_root()}: {output_dir}")

    epochs = int(training.get("epochs", 30))
    batch_size = int(training.get("batch_size", 32))
    learning_rate = float(training.get("learning_rate", 0.004))
    if epochs != 30:
        raise ValueError("Ablation experiments must run 30 epochs.")
    if batch_size != 32:
        raise ValueError("Ablation batch_size must be 32.")
    if learning_rate != 0.004:
        raise ValueError("Ablation learning_rate must be 0.004.")

    train_config["experiment"] = {
        "name": name,
        "output_dir": str(experiment["output_dir"]),
        "seed": int(experiment.get("seed", train_config.get("experiment", {}).get("seed", 42))),
    }
    train_config["data"] = {
        "train_manifest": str(data.get("train_manifest", "metadata/manifests/train_sub.csv")),
        "val_manifest": str(data.get("val_manifest", "metadata/manifests/val.csv")),
        "class_mapping": str(data.get("class_mapping", "metadata/class_to_idx.json")),
        "train_batch_size": batch_size,
        "val_batch_size": batch_size,
        "num_workers": int(train_config.get("data", {}).get("num_workers", 0)),
        "pin_memory": bool(train_config.get("data", {}).get("pin_memory", False)),
    }
    train_config.setdefault("training", {})
    train_config["training"]["epochs"] = epochs
    train_config["training"]["learning_rate"] = learning_rate
    train_config["training"]["early_stopping_patience"] = epochs
    train_config["training"].setdefault("monitor", "val_macro_f1")
    train_config["training"].setdefault("monitor_mode", "max")
    train_config["ablation"] = {
        "pooling_type": expected_pooling,
        "dropout": expected_dropout,
        "selection_rule": "max val_accuracy, tie-break lower val_loss",
    }

    model_config.setdefault("model", {})
    model_config["model"]["name"] = name
    model_config["model"]["num_classes"] = int(dataset_config.get("dataset", {}).get("num_classes", 13))
    model_config["ablation"] = {
        "pooling_type": expected_pooling,
        "dropout": expected_dropout,
        "input_size": int(dataset_config.get("dataset", {}).get("input_size", 128)),
    }
    model_config.setdefault("classifier", {})
    model_config["classifier"]["dropout"] = expected_dropout
    return train_config, model_config, dataset_config


def _build_scheduler(optimizer: AdamW, train_config: dict[str, Any]) -> ReduceLROnPlateau | None:
    scheduler_config = train_config.get("scheduler", {})
    if str(scheduler_config.get("type", "reduce_on_plateau")) != "reduce_on_plateau":
        return None
    training_config = train_config.get("training", {})
    return ReduceLROnPlateau(
        optimizer,
        mode=str(training_config.get("monitor_mode", "max")),
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 2)),
        min_lr=float(scheduler_config.get("min_lr", 1e-6)),
    )


def _scheduler_value(train_config: dict[str, Any], val_metrics: dict[str, Any]) -> float:
    monitor = str(train_config.get("training", {}).get("monitor", "val_macro_f1"))
    metric_name = monitor.replace("val_", "", 1) if monitor.startswith("val_") else monitor
    if metric_name not in val_metrics:
        raise KeyError(f"Scheduler monitor metric not available: {monitor}")
    return float(val_metrics[metric_name])


def _is_better(val_accuracy: float, val_loss: float, best_accuracy: float, best_loss: float) -> bool:
    return val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss)


def train_ablation(config_path: str | Path) -> dict[str, Any]:
    ablation_config = load_yaml(config_path)
    train_config, model_config, dataset_config = _resolve_config(ablation_config)
    class_to_idx = _validate_inputs(train_config)

    seed = int(train_config["experiment"]["seed"])
    set_seed(seed)
    output_dir = resolve_project_path(train_config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    log_file.write_text("", encoding="utf-8")

    resolved_config = {
        "ablation_config": ablation_config,
        "train_config": train_config,
        "model_config": model_config,
        "dataset_config": dataset_config,
    }
    _write_yaml(output_dir / "resolved_config.yaml", resolved_config)

    class_names = class_names_from_mapping(class_to_idx)
    device = select_device(str(train_config.get("training", {}).get("device", "auto")))
    ablation_model = model_config["ablation"]
    model = build_ablation_model(
        model_config,
        pooling_type=str(ablation_model["pooling_type"]),
        dropout=float(ablation_model["dropout"]),
        num_classes=int(model_config.get("model", {}).get("num_classes", 13)),
        input_size=int(ablation_model["input_size"]),
    ).to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    data_config = train_config["data"]
    train_loader = create_manifest_loader(
        data_config["train_manifest"],
        batch_size=int(data_config["train_batch_size"]),
        shuffle=True,
        return_metadata=False,
        dataset_config=dataset_config,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        seed=seed,
    )
    val_loader = create_manifest_loader(
        data_config["val_manifest"],
        batch_size=int(data_config["val_batch_size"]),
        shuffle=False,
        return_metadata=False,
        dataset_config=dataset_config,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        seed=seed,
    )
    criterion, class_weights = build_loss(train_config, data_config["train_manifest"], class_to_idx, device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["training"]["learning_rate"]),
        weight_decay=float(train_config.get("training", {}).get("weight_decay", 0.0001)),
    )
    scheduler = _build_scheduler(optimizer, train_config)

    epochs = int(train_config["training"]["epochs"])
    clip_norm = float(train_config.get("training", {}).get("gradient_clip_norm", 0.0))
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    best_train_accuracy = 0.0
    best_train_loss = 0.0
    start_time = time.perf_counter()

    _log(f"Experiment: {train_config['experiment']['name']}", log_file)
    _log(f"Device: {device}", log_file)
    _log(f"Train manifest: {data_config['train_manifest']}", log_file)
    _log(f"Val manifest: {data_config['val_manifest']}", log_file)
    _log(f"Train samples: {len(train_loader.dataset)}", log_file)
    _log(f"Val samples: {len(val_loader.dataset)}", log_file)
    _log(f"Total parameters: {total_parameters}", log_file)
    _log(f"Trainable parameters: {trainable_parameters}", log_file)
    if class_weights is not None:
        _log("Class weights enabled.", log_file)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            class_names,
            gradient_clip_norm=clip_norm,
        )
        val_metrics, _ = evaluate_model(model, val_loader, criterion, device, class_names)
        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(_scheduler_value(train_config, val_metrics))

        train_loss = float(train_metrics["loss"])
        train_accuracy = float(train_metrics["accuracy"])
        val_loss = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        _write_history(output_dir / "history.csv", history)

        if _is_better(val_accuracy, val_loss, best_val_accuracy, best_val_loss):
            best_epoch = epoch
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_train_accuracy = train_accuracy
            best_train_loss = train_loss
            save_checkpoint(
                output_dir / "best_model.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_val_accuracy,
                class_to_idx=class_to_idx,
                model_config=model_config,
                train_config=train_config,
                random_seed=seed,
            )

        save_checkpoint(
            output_dir / "last_model.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_val_accuracy,
            class_to_idx=class_to_idx,
            model_config=model_config,
            train_config=train_config,
            random_seed=seed,
        )

        _log(
            (
                f"Epoch {epoch:02d}/{epochs} "
                f"lr={current_lr:.8f} "
                f"train_loss={train_loss:.6f} "
                f"train_acc={train_accuracy:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_acc={val_accuracy:.6f} "
                f"best_val_acc={best_val_accuracy:.6f}"
            ),
            log_file,
        )

    elapsed = time.perf_counter() - start_time
    final_row = history[-1]
    metrics = {
        "experiment": train_config["experiment"]["name"],
        "pooling_type": str(ablation_model["pooling_type"]),
        "dropout": float(ablation_model["dropout"]),
        "learning_rate": float(train_config["training"]["learning_rate"]),
        "batch_size": int(data_config["train_batch_size"]),
        "epochs": epochs,
        "seed": seed,
        "parameter_count": int(trainable_parameters),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "best_epoch_train_accuracy": float(best_train_accuracy),
        "best_epoch_train_loss": float(best_train_loss),
        "final_epoch_train_accuracy": float(final_row["train_accuracy"]),
        "final_epoch_train_loss": float(final_row["train_loss"]),
        "final_epoch_val_accuracy": float(final_row["val_accuracy"]),
        "final_epoch_val_loss": float(final_row["val_loss"]),
        "training_time_seconds": float(round(elapsed, 4)),
        "status": "completed",
    }
    write_json(output_dir / "metrics.json", metrics)
    _log(f"Completed in {elapsed:.2f} seconds.", log_file)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_ablation(args.config)
    print(f"Completed {metrics['experiment']}: best_val_accuracy={metrics['best_val_accuracy']:.6f}")


if __name__ == "__main__":
    main()
