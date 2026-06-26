from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import SequentialSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import read_class_to_idx, read_manifest_csv  # noqa: E402
from src.evaluation.test_metrics import (  # noqa: E402
    classification_report_rows,
    compute_core_metrics,
    compute_generalization_metrics,
    confusion_matrix_array,
    confusion_pairs,
    plot_confusion_matrix,
    write_confusion_matrix_csv,
    write_csv,
    write_json,
)
from src.models.ablation_cnn import build_ablation_model, count_parameters  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.trainer import class_names_from_mapping, create_manifest_loader, select_device  # noqa: E402
from src.utils.config import load_dataset_config, load_yaml  # noqa: E402
from src.utils.paths import project_root, resolve_project_path  # noqa: E402


REQUIRED_OUTPUTS = [
    "test_metrics.json",
    "test_summary.csv",
    "classification_report.csv",
    "predictions.csv",
    "misclassified_samples.csv",
    "most_confused_class_pairs.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "resolved_test_config.yaml",
    "evaluation.log",
]

SUMMARY_COLUMNS = [
    "experiment",
    "checkpoint_type",
    "best_epoch",
    "parameter_count",
    "num_classes",
    "test_samples",
    "best_val_accuracy",
    "test_accuracy",
    "best_val_macro_f1",
    "test_macro_f1",
    "test_loss",
    "test_macro_precision",
    "test_macro_recall",
    "test_weighted_f1",
    "test_balanced_accuracy",
    "test_top3_accuracy",
    "accuracy_generalization_gap_percentage_points",
    "accuracy_retention_rate_percent",
    "macro_f1_generalization_gap",
    "macro_f1_retention_rate",
    "loss_generalization_change",
    "train_test_accuracy_gap",
    "relative_accuracy_drop_percent",
    "evaluation_time_seconds",
    "device",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the baseline GAP-dropout best checkpoint on the held-out test set.")
    parser.add_argument("--config", type=Path, default=Path("configs/test/baseline_test.yaml"))
    parser.add_argument("--force", action="store_true", help="Overwrite only results/test/baseline_gap_dropout.")
    return parser.parse_args()


def results_test_root() -> Path:
    return (project_root().parent / "results" / "test").resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_result_dir(path_value: str | Path) -> Path:
    result_dir = resolve_project_path(path_value)
    if not _is_relative_to(result_dir, results_test_root()):
        raise ValueError(f"Result directory must be inside {results_test_root()}: {result_dir}")
    if "ablation" in [part.lower() for part in result_dir.parts]:
        raise ValueError(f"Result directory must not be inside results/ablation: {result_dir}")
    return result_dir


def _log(message: str, log_file: Path) -> None:
    print(message)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def _complete_result_dir(result_dir: Path) -> bool:
    return result_dir.is_dir() and all((result_dir / filename).is_file() for filename in REQUIRED_OUTPUTS)


def prepare_result_dir(result_dir: Path, *, force: bool) -> None:
    if _complete_result_dir(result_dir) and not force:
        raise FileExistsError(f"Complete test results already exist at {result_dir}. Use --force to regenerate them.")
    if result_dir.exists():
        if not force:
            raise FileExistsError(f"Partial result directory exists at {result_dir}. Use --force to regenerate it.")
        if not _is_relative_to(result_dir, results_test_root()):
            raise ValueError(f"Refusing to delete outside {results_test_root()}: {result_dir}")
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint(config: dict[str, Any]) -> Path:
    checkpoint_config = config.get("checkpoint", {})
    checkpoint_type = str(checkpoint_config.get("type", ""))
    if checkpoint_type != "best_model":
        raise ValueError(f"Only checkpoint.type=best_model is allowed, got {checkpoint_type}")
    configured = resolve_project_path(checkpoint_config.get("path", ""))
    candidates = [
        configured,
        project_root().parent / "results" / "ablation" / "baseline_gap_dropout" / "best_model.pt",
        project_root().parent / "outputs" / "ablation" / "baseline_gap_dropout" / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            checkpoint = candidate.resolve()
            break
    else:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Baseline best_model.pt was not found. Checked: {checked}")

    if checkpoint.name != "best_model.pt" or checkpoint.parent.name != "baseline_gap_dropout":
        raise ValueError(f"Refusing to load non-baseline best checkpoint: {checkpoint}")
    return checkpoint


def validate_test_manifest(path_value: str | Path) -> Path:
    manifest_path = resolve_project_path(path_value)
    expected = project_root() / "metadata" / "manifests" / "test.csv"
    forbidden = {
        (project_root() / "metadata" / "manifests" / "train_sub.csv").resolve(),
        (project_root() / "metadata" / "manifests" / "val.csv").resolve(),
    }
    if manifest_path.resolve() in forbidden:
        raise ValueError(f"Test manifest must not be train_sub.csv or val.csv: {manifest_path}")
    if manifest_path.resolve() != expected.resolve():
        raise ValueError(f"Expected independent test manifest {expected}, got {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")
    rows = read_manifest_csv(manifest_path)
    if not rows:
        raise ValueError(f"Test manifest is empty: {manifest_path}")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError(f"All test manifest rows must have split=test: {manifest_path}")
    return manifest_path


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint['model_state_dict'] must be a state_dict mapping")
        return state_dict, checkpoint
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, None
    raise TypeError("Unsupported checkpoint format: expected rich checkpoint dict or pure state_dict")


def build_model_from_checkpoint(
    checkpoint_data: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    class_to_idx: dict[str, int],
) -> nn.Module:
    dataset_config = load_dataset_config(config.get("data", {}).get("dataset_config", "configs/dataset.yaml"))
    if checkpoint_data is not None and "model_config" in checkpoint_data:
        model_config = copy.deepcopy(checkpoint_data["model_config"])
    else:
        model_config = {
            "model": {"input_channels": 3, "num_classes": len(class_to_idx)},
            "backbone": {
                "stage_channels": [32, 64, 128],
                "convs_per_stage": 2,
                "kernel_size": 3,
                "use_batch_norm": True,
                "activation": "relu",
                "pooling": "maxpool",
            },
            "classifier": {"hidden_dim": 128, "dropout": 0.3, "use_batch_norm": True},
            "ablation": {"pooling_type": "gap", "dropout": 0.3, "input_size": 128},
        }
    ablation_config = model_config.get("ablation", {})
    pooling_type = str(config["model"]["pooling_type"])
    dropout = float(config["model"]["dropout"])
    if str(ablation_config.get("pooling_type", pooling_type)) != pooling_type:
        raise ValueError("Checkpoint model pooling_type does not match test config")
    if float(ablation_config.get("dropout", dropout)) != dropout:
        raise ValueError("Checkpoint model dropout does not match test config")
    if pooling_type != "gap" or dropout != 0.3:
        raise ValueError("This evaluator only supports baseline_gap_dropout (GAP + dropout 0.3)")

    model_config.setdefault("model", {})["num_classes"] = len(class_to_idx)
    model_config.setdefault("ablation", {})
    model_config["ablation"]["pooling_type"] = pooling_type
    model_config["ablation"]["dropout"] = dropout
    model_config["ablation"].setdefault("input_size", int(dataset_config.get("dataset", {}).get("input_size", 128)))
    return build_ablation_model(
        model_config,
        pooling_type=pooling_type,
        dropout=dropout,
        num_classes=len(class_to_idx),
        input_size=int(model_config["ablation"]["input_size"]),
    )


def _metadata_value(metadata: dict[str, Any], key: str, index: int) -> Any:
    value = metadata[key]
    if torch.is_tensor(value):
        return value[index].item()
    return value[index]


def evaluate_model_once(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    topk_indices: list[list[int]] = []
    predictions: list[dict[str, Any]] = []
    total_loss = 0.0
    sample_count = 0
    top_k = min(3, len(class_names))

    with torch.inference_mode():
        for batch in loader:
            images, labels, metadata = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            confidences, preds = probabilities.max(dim=1)
            top_probabilities, top_classes = torch.topk(probabilities, k=top_k, dim=1)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()
            y_true.extend(int(value) for value in labels_cpu.tolist())
            y_pred.extend(int(value) for value in preds_cpu.tolist())
            top_classes_cpu = top_classes.detach().cpu()
            top_probs_cpu = top_probabilities.detach().cpu()
            topk_indices.extend([[int(value) for value in row] for row in top_classes_cpu.tolist()])

            for index in range(batch_size):
                true_index = int(labels_cpu[index].item())
                predicted_index = int(preds_cpu[index].item())
                top_names = [class_names[int(value)] for value in top_classes_cpu[index].tolist()]
                top_probs = [float(value) for value in top_probs_cpu[index].tolist()]
                predictions.append(
                    {
                        "sample_index": len(predictions),
                        "image_path": str(_metadata_value(metadata, "path", index)),
                        "true_class_index": true_index,
                        "true_class_name": class_names[true_index],
                        "predicted_class_index": predicted_index,
                        "predicted_class_name": class_names[predicted_index],
                        "confidence": float(confidences[index].detach().cpu().item()),
                        "correct": true_index == predicted_index,
                        "top3_class_names": "|".join(top_names),
                        "top3_probabilities": "|".join(f"{value:.8f}" for value in top_probs),
                    }
                )

    if sample_count <= 0:
        raise RuntimeError("No test samples were evaluated")
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "topk_indices": topk_indices,
        "predictions": predictions,
        "test_loss": total_loss / sample_count,
        "test_samples": sample_count,
    }


def save_outputs(
    result_dir: Path,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> None:
    write_json(result_dir / "test_metrics.json", metrics)
    write_csv(result_dir / "test_summary.csv", [metrics], SUMMARY_COLUMNS)
    write_csv(
        result_dir / "classification_report.csv",
        classification_report_rows(y_true, y_pred, class_names=class_names),
        ["class_index", "class_name", "precision", "recall", "f1_score", "support"],
    )
    write_csv(
        result_dir / "predictions.csv",
        predictions,
        [
            "sample_index",
            "image_path",
            "true_class_index",
            "true_class_name",
            "predicted_class_index",
            "predicted_class_name",
            "confidence",
            "correct",
            "top3_class_names",
            "top3_probabilities",
        ],
    )
    misclassified = sorted(
        [row for row in predictions if not bool(row["correct"])],
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )
    write_csv(
        result_dir / "misclassified_samples.csv",
        misclassified,
        ["image_path", "true_class_name", "predicted_class_name", "confidence"],
    )
    write_csv(
        result_dir / "most_confused_class_pairs.csv",
        confusion_pairs(y_true, y_pred, class_names=class_names),
        ["true_class_name", "predicted_class_name", "count"],
    )
    matrix = confusion_matrix_array(y_true, y_pred, num_classes=len(class_names))
    write_confusion_matrix_csv(result_dir / "confusion_matrix.csv", matrix, class_names)
    plot_confusion_matrix(result_dir / "confusion_matrix.png", matrix, class_names, normalized=False)
    plot_confusion_matrix(result_dir / "confusion_matrix_normalized.png", matrix, class_names, normalized=True)


def run_evaluation(config_path: str | Path, *, force: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("experiment", {}).get("name") != "baseline_gap_dropout":
        raise ValueError("Only baseline_gap_dropout can be evaluated by this script")
    result_dir = resolve_result_dir(config["experiment"]["result_dir"])
    prepare_result_dir(result_dir, force=force)
    log_file = result_dir / "evaluation.log"
    log_file.write_text("", encoding="utf-8")

    try:
        checkpoint_path = resolve_checkpoint(config)
        test_manifest = validate_test_manifest(config["data"]["test_manifest"])
        class_mapping_path = resolve_project_path(config["data"]["class_mapping"])
        class_to_idx = read_class_to_idx(class_mapping_path)
        class_names = class_names_from_mapping(class_to_idx)
        device = select_device("auto")

        _log(f"Checkpoint: {checkpoint_path}", log_file)
        _log(f"Test manifest: {test_manifest}", log_file)
        _log(f"Result directory: {result_dir}", log_file)
        _log(f"Device: {device}", log_file)

        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        state_dict, checkpoint_data = extract_state_dict(checkpoint)
        if checkpoint_data is not None:
            checkpoint_name = checkpoint_data.get("train_config", {}).get("experiment", {}).get("name")
            if checkpoint_name != "baseline_gap_dropout":
                raise ValueError(f"Checkpoint experiment is not baseline_gap_dropout: {checkpoint_name}")
            if int(checkpoint_data.get("epoch", -1)) != int(config["training_reference"]["best_epoch"]):
                raise ValueError("Checkpoint epoch does not match expected best_epoch")

        model = build_model_from_checkpoint(checkpoint_data, config, class_to_idx=class_to_idx).to(device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(f"State dict mismatch: missing={missing_keys}, unexpected={unexpected_keys}")
        parameter_count = count_parameters(model)[1]
        expected_parameter_count = int(config["model"]["expected_parameter_count"])
        if parameter_count != expected_parameter_count:
            raise RuntimeError(f"Expected {expected_parameter_count} parameters, got {parameter_count}")
        model.eval()

        dataset_config = load_dataset_config(config["data"]["dataset_config"])
        loader = create_manifest_loader(
            test_manifest,
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=bool(config["evaluation"]["shuffle"]),
            return_metadata=True,
            dataset_config=dataset_config,
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=bool(config["evaluation"]["pin_memory"]),
            seed=int(config["training_reference"]["seed"]),
        )
        if not isinstance(loader.sampler, SequentialSampler):
            raise RuntimeError("Test DataLoader must use SequentialSampler (shuffle=False)")
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float((checkpoint_data or {}).get("train_config", {}).get("loss", {}).get("label_smoothing", 0.1))
        ).to(device)

        _write_yaml(
            result_dir / "resolved_test_config.yaml",
            {
                "test_config": config,
                "resolved": {
                    "checkpoint_path": str(checkpoint_path),
                    "test_manifest": str(test_manifest),
                    "class_mapping": str(class_mapping_path),
                    "result_dir": str(result_dir),
                    "dataset_config": str(resolve_project_path(config["data"]["dataset_config"])),
                },
            },
        )

        started = time.perf_counter()
        eval_result = evaluate_model_once(model, loader, criterion, device, class_names)
        elapsed = time.perf_counter() - started
        core_metrics = compute_core_metrics(
            eval_result["y_true"],
            eval_result["y_pred"],
            eval_result["topk_indices"],
            test_loss=float(eval_result["test_loss"]),
            num_classes=len(class_names),
        )
        generalization = compute_generalization_metrics(
            best_epoch_train_accuracy=float(config["training_reference"]["best_train_accuracy"]),
            best_val_accuracy=float(config["training_reference"]["best_val_accuracy"]),
            best_val_loss=float(config["training_reference"]["best_val_loss"]),
            best_val_macro_f1=float(config["training_reference"]["best_val_macro_f1"]),
            test_accuracy=core_metrics["test_accuracy"],
            test_loss=core_metrics["test_loss"],
            test_macro_f1=core_metrics["test_macro_f1"],
        )
        metrics = {
            "experiment": "baseline_gap_dropout",
            "checkpoint_type": "best_model",
            "checkpoint_path": str(checkpoint_path),
            "pooling_type": "gap",
            "dropout": 0.3,
            "learning_rate": float(config["training_reference"]["learning_rate"]),
            "training_batch_size": int(config["training_reference"]["batch_size"]),
            "training_epochs": int(config["training_reference"]["epochs"]),
            "seed": int(config["training_reference"]["seed"]),
            "parameter_count": int(parameter_count),
            "best_epoch": int(config["training_reference"]["best_epoch"]),
            "num_classes": len(class_names),
            "test_samples": int(eval_result["test_samples"]),
            "best_epoch_train_accuracy": float(config["training_reference"]["best_train_accuracy"]),
            "best_val_accuracy": float(config["training_reference"]["best_val_accuracy"]),
            "best_val_loss": float(config["training_reference"]["best_val_loss"]),
            "best_val_macro_f1": float(config["training_reference"]["best_val_macro_f1"]),
            **core_metrics,
            **generalization,
            "evaluation_time_seconds": float(round(elapsed, 4)),
            "device": str(device),
            "status": "completed",
        }
        save_outputs(result_dir, metrics, eval_result["predictions"], eval_result["y_true"], eval_result["y_pred"], class_names)
        _log(f"Test samples: {metrics['test_samples']}", log_file)
        _log(f"Test loss: {metrics['test_loss']:.6f}", log_file)
        _log(f"Test accuracy: {metrics['test_accuracy']:.6f}", log_file)
        _log(f"Test macro F1: {metrics['test_macro_f1']:.6f}", log_file)
        _log("Evaluation completed.", log_file)
        return metrics
    except Exception as exc:
        failure = {"experiment": "baseline_gap_dropout", "status": "failed", "error": str(exc)}
        write_json(result_dir / "test_metrics.json", failure)
        _log(f"FAILED: {exc}", log_file)
        raise


def main() -> None:
    args = parse_args()
    metrics = run_evaluation(args.config, force=args.force)
    print(
        "Completed baseline test: "
        f"accuracy={metrics['test_accuracy']:.6f}, "
        f"macro_f1={metrics['test_macro_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
