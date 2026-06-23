from __future__ import annotations

from collections import OrderedDict

from torch import nn


def build_activation(name: str) -> nn.Module:
    """Create an activation module from config."""
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """Repeated Conv-BN-ReLU layers followed by 2x2 max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        convs_per_stage: int,
        kernel_size: int,
        use_batch_norm: bool,
        activation: str,
        pooling: str,
        stage_index: int,
    ) -> None:
        super().__init__()
        if convs_per_stage <= 0:
            raise ValueError("convs_per_stage must be positive")
        if pooling.lower() != "maxpool":
            raise ValueError(f"Unsupported pooling: {pooling}")
        padding = kernel_size // 2
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        current_channels = in_channels
        for conv_index in range(convs_per_stage):
            layers[f"conv{conv_index + 1}"] = nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            )
            if use_batch_norm:
                layers[f"bn{conv_index + 1}"] = nn.BatchNorm2d(out_channels)
            layers[f"act{conv_index + 1}"] = build_activation(activation)
            current_channels = out_channels
        layers["pool"] = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage_index = stage_index
        self.block = nn.Sequential(layers)

    def forward(self, x):  # type: ignore[no-untyped-def]
        """Apply the convolutional block."""
        return self.block(x)
