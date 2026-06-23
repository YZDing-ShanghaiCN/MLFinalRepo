from __future__ import annotations

from torch import Tensor, nn

from src.models.blocks import ConvBlock


class CNNBackbone(nn.Module):
    """Shallow CNN backbone with three configurable convolution stages."""

    def __init__(
        self,
        *,
        input_channels: int,
        stage_channels: list[int],
        convs_per_stage: int,
        kernel_size: int,
        use_batch_norm: bool,
        activation: str,
        pooling: str,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_channels = input_channels
        for index, out_channels in enumerate(stage_channels, start=1):
            stages.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    convs_per_stage=convs_per_stage,
                    kernel_size=kernel_size,
                    use_batch_norm=use_batch_norm,
                    activation=activation,
                    pooling=pooling,
                    stage_index=index,
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.out_channels = stage_channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Return CNN feature maps."""
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_with_shapes(self, x: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        """Forward through each stage and collect output shapes."""
        shapes: dict[str, tuple[int, ...]] = {}
        for index, stage in enumerate(self.stages, start=1):
            x = stage(x)
            shapes[f"conv_block_{index}"] = tuple(x.shape)
        return x, shapes
