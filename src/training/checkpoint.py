from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: Any,
    best_metric: float,
    class_to_idx: dict[str, int],
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    random_seed: int,
) -> None:
    """Save a rich checkpoint that can be loaded on CPU."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "class_to_idx": class_to_idx,
        "model_config": model_config,
        "train_config": train_config,
        "random_seed": random_seed,
        "pytorch_version": torch.__version__,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint with CPU-safe map_location."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
