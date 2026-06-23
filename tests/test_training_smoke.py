from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.manifest import MANIFEST_COLUMNS
from src.data.dataset import RGBDManifestDataset
from src.data.transforms import ResizeLongestSideAndPad
from src.models.rgb_classifier import build_model_from_config
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.trainer import train_one_epoch
from src.utils.config import load_model_config


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _temp_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "data"
    rows: list[dict[str, object]] = []
    for index in range(8):
        class_index = index % 2
        class_name = "apple" if class_index == 0 else "binder"
        rel_path = f"{class_name}/{class_name}_1/{class_name}_{index}_crop.png"
        image_path = dataset_root / rel_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64 + index, 60 + index), (20 * index, 100, 200)).save(image_path)
        rows.append(
            {
                "image_path": rel_path,
                "class_name": class_name,
                "class_index": class_index,
                "instance_name": f"{class_name}_1",
                "split": "train",
                "width": 64 + index,
                "height": 60 + index,
            }
        )
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)
    return dataset_root, manifest


def test_training_smoke_forward_backward_checkpoint(tmp_path: Path) -> None:
    dataset_root, manifest = _temp_dataset(tmp_path)
    dataset = RGBDManifestDataset(
        manifest,
        dataset_root=dataset_root,
        transform=ResizeLongestSideAndPad(size=128),
        num_classes=13,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    model_config = load_model_config()
    model = build_model_from_config(model_config)
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    images, labels = next(iter(loader))
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"), [str(i) for i in range(13)])
    assert metrics["loss"] >= 0.0

    checkpoint_path = tmp_path / "smoke.pt"
    save_checkpoint(
        checkpoint_path,
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        best_metric=0.0,
        class_to_idx={str(i): i for i in range(13)},
        model_config=model_config,
        train_config={"experiment": {"seed": 42}},
        random_seed=42,
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    assert checkpoint["epoch"] == 1
    reloaded = build_model_from_config(checkpoint["model_config"])
    reloaded.load_state_dict(checkpoint["model_state_dict"])
