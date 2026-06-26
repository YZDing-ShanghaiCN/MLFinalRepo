from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.models.cnn_backbone import CNNBackbone
from src.models.rgb_classifier import initialize_weights


SUPPORTED_POOLING_TYPES = {"gap", "flatten"}


@dataclass(frozen=True)
class AblationModelInfo:
    pooling_type: str
    dropout: float
    feature_dim: int
    total_parameters: int
    trainable_parameters: int


class AblationClassificationHead(nn.Module):
    """Classification head matching the baseline head with configurable dropout."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        layers: list[nn.Module] = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Identity() if dropout == 0.0 else nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class AblationCNN(nn.Module):
    """CNN ablation model supporting GAP and Flatten pooling variants."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        pooling_type: str = "gap",
        dropout: float = 0.3,
        num_classes: int = 13,
        input_size: int = 128,
    ) -> None:
        super().__init__()
        config = {} if config is None else config
        model_config = config.get("model", {})
        backbone_config = config.get("backbone", {})
        classifier_config = config.get("classifier", {})

        normalized_pooling = pooling_type.lower()
        if normalized_pooling not in SUPPORTED_POOLING_TYPES:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
        if input_size <= 0:
            raise ValueError("input_size must be positive")

        input_channels = int(model_config.get("input_channels", 3))
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
        self.pooling_type = normalized_pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) if normalized_pooling == "gap" else nn.Identity()
        self.flatten = nn.Flatten()
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.dropout = float(dropout)
        self.feature_dim = self._infer_feature_dim(input_channels)
        self.classifier = AblationClassificationHead(
            input_dim=self.feature_dim,
            hidden_dim=int(classifier_config.get("hidden_dim", 128)),
            num_classes=self.num_classes,
            dropout=self.dropout,
            use_batch_norm=bool(classifier_config.get("use_batch_norm", True)),
        )
        self.apply(initialize_weights)

    def _infer_feature_dim(self, input_channels: int) -> int:
        device = next(self.backbone.parameters()).device
        dummy = torch.zeros(1, input_channels, self.input_size, self.input_size, device=device)
        with torch.no_grad():
            features = self.backbone(dummy)
            pooled = self.global_pool(features)
            flat = self.flatten(pooled)
        feature_dim = int(flat.shape[1])
        if feature_dim <= 0:
            raise RuntimeError(f"Invalid inferred feature dimension: {feature_dim}")
        return feature_dim

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        pooled = self.global_pool(features)
        flat = self.flatten(pooled)
        logits = self.classifier(flat)
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise RuntimeError(f"Expected logits shape [batch, {self.num_classes}], got {tuple(logits.shape)}")
        return logits

    def forward_debug(self, x: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        shapes: dict[str, tuple[int, ...]] = {"input": tuple(x.shape)}
        features, backbone_shapes = self.backbone.forward_with_shapes(x)
        shapes.update(backbone_shapes)
        shapes["pool_before"] = tuple(features.shape)
        pooled = self.global_pool(features)
        shapes["pool_after"] = tuple(pooled.shape)
        flat = self.flatten(pooled)
        shapes["flatten"] = tuple(flat.shape)
        logits = self.classifier(flat)
        shapes["classifier_output"] = tuple(logits.shape)
        return logits, shapes

    def model_info(self) -> AblationModelInfo:
        total, trainable = count_parameters(self)
        return AblationModelInfo(
            pooling_type=self.pooling_type,
            dropout=self.dropout,
            feature_dim=self.feature_dim,
            total_parameters=total,
            trainable_parameters=trainable,
        )


def build_ablation_model(
    config: dict[str, Any],
    *,
    pooling_type: str,
    dropout: float,
    num_classes: int,
    input_size: int,
) -> AblationCNN:
    return AblationCNN(
        config,
        pooling_type=pooling_type,
        dropout=dropout,
        num_classes=num_classes,
        input_size=input_size,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)
