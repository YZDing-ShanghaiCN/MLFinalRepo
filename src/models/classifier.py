from __future__ import annotations

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """MLP classification head that consumes GAP features."""

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
        layers: list[nn.Module] = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Return unnormalized class logits."""
        return self.layers(x)
