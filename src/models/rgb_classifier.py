from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.models.classifier import ClassificationHead
from src.models.cnn_backbone import CNNBackbone


def initialize_weights(module: nn.Module) -> None:
    """Initialize Conv, Linear, and BatchNorm layers for ReLU CNN training."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class RGBClassifier(nn.Module):
    """End-to-end CNN + GAP + classifier for RGB object classification."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        backbone_config = config.get("backbone", {})
        pool_config = config.get("global_pool", {})
        classifier_config = config.get("classifier", {})

        input_channels = int(model_config.get("input_channels", 3))
        num_classes = int(model_config.get("num_classes", 13))
        stage_channels = [int(value) for value in backbone_config.get("stage_channels", [32, 64, 128])]

        self.backbone = CNNBackbone(
            input_channels=input_channels,
            stage_channels=stage_channels,
            convs_per_stage=int(backbone_config.get("convs_per_stage", 2)),
            kernel_size=int(backbone_config.get("kernel_size", 3)),
            use_batch_norm=bool(backbone_config.get("use_batch_norm", True)),
            activation=str(backbone_config.get("activation", "relu")),
            pooling=str(backbone_config.get("pooling", "maxpool")),
        )
        if str(pool_config.get("type", "adaptive_avg")) != "adaptive_avg":
            raise ValueError(f"Unsupported global_pool.type: {pool_config.get('type')}")
        output_size = int(pool_config.get("output_size", 1))
        self.global_pool = nn.AdaptiveAvgPool2d((output_size, output_size))
        self.flatten = nn.Flatten()
        feature_dim = self.backbone.out_channels * output_size * output_size
        self.classifier = ClassificationHead(
            input_dim=feature_dim,
            hidden_dim=int(classifier_config.get("hidden_dim", 128)),
            num_classes=num_classes,
            dropout=float(classifier_config.get("dropout", 0.3)),
            use_batch_norm=bool(classifier_config.get("use_batch_norm", True)),
        )
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits with shape [B, num_classes]."""
        features = self.backbone(x)
        pooled = self.global_pool(features)
        flat = self.flatten(pooled)
        return self.classifier(flat)

    def forward_debug(self, x: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        """Forward once and return important intermediate shapes."""
        shapes: dict[str, tuple[int, ...]] = {"input": tuple(x.shape)}
        features, backbone_shapes = self.backbone.forward_with_shapes(x)
        shapes.update(backbone_shapes)
        shapes["gap_before"] = tuple(features.shape)
        pooled = self.global_pool(features)
        shapes["gap_after"] = tuple(pooled.shape)
        flat = self.flatten(pooled)
        shapes["flatten"] = tuple(flat.shape)
        logits = self.classifier(flat)
        shapes["classifier_output"] = tuple(logits.shape)
        return logits, shapes


def build_model_from_config(config: dict[str, Any]) -> RGBClassifier:
    """Build the configured RGB classifier."""
    return RGBClassifier(config)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def has_invalid_values(tensor: Tensor) -> bool:
    """Return whether a tensor contains NaN or Inf."""
    return bool(torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item())
