from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rgb_classifier import build_model_from_config, count_parameters, has_invalid_values  # noqa: E402
from src.training.trainer import select_device  # noqa: E402
from src.utils.config import load_model_config  # noqa: E402


def main() -> None:
    config = load_model_config()
    device = select_device("auto")
    model = build_model_from_config(config).to(device)
    model.eval()
    x = torch.randn(2, 3, 128, 128, device=device)
    with torch.no_grad():
        logits, shapes = model.forward_debug(x)
    total, trainable = count_parameters(model)
    parameter_memory_mb = total * 4 / (1024**2)
    print(model)
    print(f"Device: {device}")
    print(f"Input shape: {shapes['input']}")
    for key in ("conv_block_1", "conv_block_2", "conv_block_3", "gap_before", "gap_after", "flatten", "classifier_output"):
        print(f"{key}: {shapes[key]}")
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")
    print(f"Parameter memory (float32): {parameter_memory_mb:.2f} MB")
    print(f"Final output shape: {tuple(logits.shape)}")
    print(f"Contains NaN/Inf: {has_invalid_values(logits)}")
    if tuple(logits.shape) != (2, 13):
        raise RuntimeError(f"Expected output shape (2, 13), got {tuple(logits.shape)}")
    if has_invalid_values(logits):
        raise RuntimeError("Model output contains NaN or Inf")


if __name__ == "__main__":
    main()
