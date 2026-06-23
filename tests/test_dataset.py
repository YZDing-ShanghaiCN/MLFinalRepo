from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataloader import create_dataloaders
from src.data.dataset import RGBDManifestDataset
from src.data.manifest import MANIFEST_COLUMNS, build_class_to_idx, read_manifest_csv
from src.data.transforms import ResizeLongestSideAndPad
from src.utils.config import dataset_root_from_config, load_dataset_config
from src.utils.paths import project_root


CLASS_NAMES = [
    "apple",
    "binder",
    "coffee_mug",
    "dry_battery",
    "greens",
    "kleenex",
    "lightbulb",
    "lime",
    "mushroom",
    "notebook",
    "pitcher",
    "sponge",
    "water_bottle",
]


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _make_temp_dataset(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    dataset_root = tmp_path / "data"
    rows: list[dict[str, object]] = []
    samples = [
        ("apple", 0, "apple_1", "apple/apple_1/apple_1_1_crop.png", (83, 81)),
        ("binder", 1, "binder_1", "binder/binder_1/binder_1_1_crop.png", (328, 282)),
        ("water_bottle", 12, "water_bottle_1", "water_bottle/water_bottle_1/water_bottle_1_1_crop.png", (90, 195)),
    ]
    for class_name, class_index, instance_name, rel_path, size in samples:
        image_path = dataset_root / rel_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (120, 20 + class_index, 200)).save(image_path)
        rows.append(
            {
                "image_path": rel_path,
                "class_name": class_name,
                "class_index": class_index,
                "instance_name": instance_name,
                "split": "train",
                "width": size[0],
                "height": size[1],
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, rows)
    return dataset_root, manifest_path, rows


def test_class_to_idx_contains_13_classes() -> None:
    splits = {"train": {class_name: [] for class_name in CLASS_NAMES}, "test": {}}
    class_to_idx = build_class_to_idx(splits)
    assert len(class_to_idx) == 13
    assert sorted(class_to_idx.values()) == list(range(13))


def test_dataset_length_shape_and_label_dtype(tmp_path: Path) -> None:
    dataset_root, manifest_path, rows = _make_temp_dataset(tmp_path)
    dataset = RGBDManifestDataset(
        manifest_path,
        dataset_root=dataset_root,
        transform=ResizeLongestSideAndPad(size=128),
        return_metadata=True,
        num_classes=13,
    )
    image, label, metadata = dataset[0]
    assert len(dataset) == len(rows)
    assert tuple(image.shape) == (3, 128, 128)
    assert label.dtype == torch.long
    assert 0 <= int(label.item()) <= 12
    assert metadata["class_name"] == "apple"


def test_dataloader_can_build_batch(tmp_path: Path) -> None:
    dataset_root, manifest_path, _ = _make_temp_dataset(tmp_path)
    dataset = RGBDManifestDataset(
        manifest_path,
        dataset_root=dataset_root,
        transform=ResizeLongestSideAndPad(size=128),
        num_classes=13,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    assert tuple(images.shape) == (2, 3, 128, 128)
    assert labels.dtype == torch.long


def test_config_relative_path_resolves_from_project_root() -> None:
    config = load_dataset_config()
    resolved_root = dataset_root_from_config(config)
    assert resolved_root == (project_root().parent / "rgbd-dataset_eval").resolve()


def test_real_manifests_have_no_train_test_overlap() -> None:
    root = project_root()
    train_manifest = root / "metadata" / "manifests" / "train.csv"
    test_manifest = root / "metadata" / "manifests" / "test.csv"
    if not train_manifest.is_file() or not test_manifest.is_file():
        pytest.skip("Real manifests are missing; run python scripts/build_manifests.py first.")

    train_rows = read_manifest_csv(train_manifest)
    test_rows = read_manifest_csv(test_manifest)
    train_instances = {(row["class_name"], row["instance_name"]) for row in train_rows}
    test_instances = {(row["class_name"], row["instance_name"]) for row in test_rows}
    train_paths = {row["image_path"] for row in train_rows}
    test_paths = {row["image_path"] for row in test_rows}
    assert train_instances.isdisjoint(test_instances)
    assert train_paths.isdisjoint(test_paths)


def test_real_create_dataloaders_batch_shape() -> None:
    root = project_root()
    if not (root / "metadata" / "manifests" / "train.csv").is_file():
        pytest.skip("Real manifests are missing; run python scripts/build_manifests.py first.")
    train_loader, test_loader, class_to_idx = create_dataloaders(return_metadata=False)
    assert len(class_to_idx) == 13
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    assert train_images.ndim == 4
    assert test_images.ndim == 4
    assert tuple(train_images.shape[1:]) == (3, 128, 128)
    assert tuple(test_images.shape[1:]) == (3, 128, 128)
    assert train_labels.dtype == torch.long
    assert test_labels.dtype == torch.long
