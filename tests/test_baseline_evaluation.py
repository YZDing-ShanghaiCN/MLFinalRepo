from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_baseline_test import (  # noqa: E402
    build_model_from_checkpoint,
    extract_state_dict,
    resolve_checkpoint,
    resolve_result_dir,
    results_test_root,
    validate_test_manifest,
)
from src.data.transforms import ResizeLongestSideAndPad, build_transform_from_config  # noqa: E402
from src.evaluation.test_metrics import compute_core_metrics, compute_generalization_metrics  # noqa: E402
from src.models.ablation_cnn import count_parameters  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.trainer import create_manifest_loader  # noqa: E402
from src.utils.config import load_dataset_config, load_yaml  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "test" / "baseline_test.yaml"


def _config() -> dict:
    return load_yaml(CONFIG_PATH)


def test_baseline_test_config_paths_and_checkpoint_resolution() -> None:
    config = _config()
    assert config["experiment"]["name"] == "baseline_gap_dropout"
    result_dir = resolve_result_dir(config["experiment"]["result_dir"])
    assert result_dir == Path(r"C:\Users\dyz18\Desktop\code\mlfinal\results\test\baseline_gap_dropout")
    assert result_dir.is_relative_to(results_test_root())
    assert "ml_repo" not in [part.lower() for part in result_dir.parts]

    checkpoint_path = resolve_checkpoint(config)
    assert checkpoint_path.name == "best_model.pt"
    assert checkpoint_path.parent.name == "baseline_gap_dropout"
    assert checkpoint_path.is_file()


def test_config_uses_baseline_gap_dropout_and_independent_test_manifest() -> None:
    config = _config()
    assert config["checkpoint"]["type"] == "best_model"
    assert config["model"]["pooling_type"] == "gap"
    assert float(config["model"]["dropout"]) == 0.3
    manifest = validate_test_manifest(config["data"]["test_manifest"])
    assert manifest.name == "test.csv"
    assert manifest.name not in {"train_sub.csv", "val.csv"}


def test_checkpoint_strict_load_parameter_count_and_forward_shape() -> None:
    config = _config()
    checkpoint = load_checkpoint(resolve_checkpoint(config), map_location="cpu")
    state_dict, checkpoint_data = extract_state_dict(checkpoint)
    class_to_idx = {str(key): int(value) for key, value in checkpoint["class_to_idx"].items()}
    model = build_model_from_checkpoint(checkpoint_data, config, class_to_idx=class_to_idx)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    assert missing_keys == []
    assert unexpected_keys == []
    assert count_parameters(model)[1] == 305901
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 128, 128))
    assert tuple(logits.shape) == (2, 13)


def test_test_loader_is_sequential_and_transform_is_deterministic() -> None:
    config = _config()
    dataset_config = load_dataset_config(config["data"]["dataset_config"])
    transform = build_transform_from_config(dataset_config)
    assert isinstance(transform, ResizeLongestSideAndPad)
    loader = create_manifest_loader(
        config["data"]["test_manifest"],
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=bool(config["evaluation"]["shuffle"]),
        return_metadata=True,
        dataset_config=dataset_config,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=bool(config["evaluation"]["pin_memory"]),
    )
    assert isinstance(loader.sampler, SequentialSampler)


def test_inference_does_not_modify_model_parameters() -> None:
    config = _config()
    checkpoint = load_checkpoint(resolve_checkpoint(config), map_location="cpu")
    state_dict, checkpoint_data = extract_state_dict(checkpoint)
    class_to_idx = {str(key): int(value) for key, value in checkpoint["class_to_idx"].items()}
    model = build_model_from_checkpoint(checkpoint_data, config, class_to_idx=class_to_idx)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with torch.inference_mode():
        _ = model(torch.randn(2, 3, 128, 128))
    after = {name: parameter.detach() for name, parameter in model.named_parameters()}
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_multiclass_metrics_and_generalization_formulas() -> None:
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    topk = [[0, 1, 2], [2, 1, 0], [2, 1, 0], [1, 2, 0]]
    metrics = compute_core_metrics(y_true, y_pred, topk, test_loss=1.25, num_classes=3)
    assert metrics["test_accuracy"] == 0.5
    assert round(metrics["test_macro_f1"], 6) == round((1.0 + 0.0 + 0.5) / 3.0, 6)
    assert round(metrics["test_balanced_accuracy"], 6) == round((1.0 + 0.0 + 0.5) / 3.0, 6)
    assert metrics["test_top3_accuracy"] == 1.0

    gaps = compute_generalization_metrics(
        best_epoch_train_accuracy=0.9,
        best_val_accuracy=0.8,
        best_val_loss=1.0,
        best_val_macro_f1=0.7,
        test_accuracy=0.5,
        test_loss=1.25,
        test_macro_f1=metrics["test_macro_f1"],
    )
    assert gaps["accuracy_generalization_gap"] == 0.30000000000000004
    assert gaps["accuracy_retention_rate"] == 0.625
    assert round(gaps["macro_f1_generalization_gap"], 6) == round(0.7 - metrics["test_macro_f1"], 6)
    assert round(gaps["macro_f1_retention_rate"], 6) == round(metrics["test_macro_f1"] / 0.7, 6)
    assert gaps["loss_generalization_change"] == 0.25
    assert gaps["train_test_accuracy_gap"] == 0.4


def test_output_directory_guards_do_not_target_ablation_results() -> None:
    config = copy.deepcopy(_config())
    result_dir = resolve_result_dir(config["experiment"]["result_dir"])
    assert "ablation" not in [part.lower() for part in result_dir.parts]
    config["experiment"]["result_dir"] = "../results/ablation/bad"
    try:
        resolve_result_dir(config["experiment"]["result_dir"])
    except ValueError as exc:
        assert "results/ablation" in str(exc) or "results\\ablation" in str(exc)
    else:
        raise AssertionError("results/ablation output path should be rejected")
