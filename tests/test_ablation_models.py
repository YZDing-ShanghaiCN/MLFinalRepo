from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ablation_cnn import AblationCNN  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs" / "ablations"


def _load_config(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    assert isinstance(data, dict)
    return data


def _model(pooling_type: str, dropout: float) -> AblationCNN:
    config = {
        "model": {"input_channels": 3, "num_classes": 13},
        "backbone": {
            "stage_channels": [32, 64, 128],
            "convs_per_stage": 2,
            "kernel_size": 3,
            "use_batch_norm": True,
            "activation": "relu",
            "pooling": "maxpool",
        },
        "classifier": {"hidden_dim": 128, "use_batch_norm": True},
    }
    return AblationCNN(config, pooling_type=pooling_type, dropout=dropout, num_classes=13, input_size=128)


def test_baseline_model_forward_shape() -> None:
    model = _model("gap", 0.3)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 128, 128))
    assert tuple(logits.shape) == (4, 13)


def test_flatten_model_forward_shape_and_auto_feature_dim() -> None:
    model = _model("flatten", 0.3)
    model.eval()
    with torch.no_grad():
        logits, shapes = model.forward_debug(torch.randn(4, 3, 128, 128))
    assert tuple(logits.shape) == (4, 13)
    assert shapes["pool_before"] == shapes["pool_after"]
    assert model.feature_dim == shapes["flatten"][1]
    assert model.feature_dim > 128


def test_no_dropout_model_forward_shape_and_identity_dropout() -> None:
    model = _model("gap", 0.0)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 128, 128))
    assert tuple(logits.shape) == (4, 13)
    assert not any(isinstance(module, nn.Dropout) and module.p > 0 for module in model.modules())
    assert any(isinstance(module, nn.Identity) for module in model.classifier.modules())


def test_ablation_yaml_configs_load_and_match_required_training_values() -> None:
    expected = {
        "baseline_gap_dropout.yaml": ("baseline_gap_dropout", "gap", 0.3),
        "flatten_dropout.yaml": ("flatten_dropout", "flatten", 0.3),
        "gap_no_dropout.yaml": ("gap_no_dropout", "gap", 0.0),
    }
    for filename, (experiment, pooling_type, dropout) in expected.items():
        config = _load_config(filename)
        assert config["experiment"]["name"] == experiment
        assert config["model"]["pooling_type"] == pooling_type
        assert float(config["model"]["dropout"]) == dropout
        assert config["training"]["epochs"] == 30
        assert float(config["training"]["learning_rate"]) == 0.004
        assert int(config["training"]["batch_size"]) == 32
        assert str(config["experiment"]["output_dir"]).startswith("../outputs/ablation/")
        assert config["data"]["train_manifest"].endswith("train_sub.csv")
        assert config["data"]["val_manifest"].endswith("val.csv")
        assert not any("test" in key.lower() for key in config["data"])
