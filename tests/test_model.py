from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW

from src.models.rgb_classifier import build_model_from_config
from src.utils.config import load_model_config


def test_model_forward_shapes_and_values() -> None:
    model = build_model_from_config(load_model_config())
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        logits, shapes = model.forward_debug(x)
    assert tuple(logits.shape) == (2, 13)
    assert logits.dtype == torch.float32
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert shapes["gap_after"] == (2, 128, 1, 1)
    assert shapes["flatten"] == (2, 128)


def test_model_backward_updates_backbone_and_classifier() -> None:
    model = build_model_from_config(load_model_config())
    model.train()
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(2, 3, 128, 128)
    y = torch.tensor([0, 1], dtype=torch.long)
    backbone_before = next(model.backbone.parameters()).detach().clone()
    classifier_before = next(model.classifier.parameters()).detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    optimizer.step()

    backbone_after = next(model.backbone.parameters()).detach()
    classifier_after = next(model.classifier.parameters()).detach()
    assert not torch.equal(backbone_before, backbone_after)
    assert not torch.equal(classifier_before, classifier_after)
