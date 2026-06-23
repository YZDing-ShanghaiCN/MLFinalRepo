from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import RGBDManifestDataset
from src.data.manifest import read_class_to_idx
from src.data.transforms import build_transform_from_config
from src.utils.config import dataset_root_from_config, load_dataset_config
from src.utils.paths import project_root, resolve_project_path


def create_dataloaders(
    config_path: str | Path | None = None,
    *,
    return_metadata: bool = False,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Create train/test DataLoaders and return the shared class mapping."""
    config = load_dataset_config(config_path)
    root = project_root()
    manifest_dir = root / "metadata" / "manifests"
    train_manifest = manifest_dir / "train.csv"
    test_manifest = manifest_dir / "test.csv"
    class_to_idx_path = root / "metadata" / "class_to_idx.json"

    missing = [path for path in (train_manifest, test_manifest, class_to_idx_path) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Manifest files are missing ({missing_text}). Run python scripts/build_manifests.py first.")

    class_to_idx = read_class_to_idx(class_to_idx_path)
    transform = build_transform_from_config(config)
    dataset_root = dataset_root_from_config(config)
    num_classes = int(config.get("dataset", {}).get("num_classes", len(class_to_idx)))

    train_dataset = RGBDManifestDataset(
        train_manifest,
        dataset_root=dataset_root,
        transform=transform,
        return_metadata=return_metadata,
        num_classes=num_classes,
    )
    test_dataset = RGBDManifestDataset(
        test_manifest,
        dataset_root=dataset_root,
        transform=transform,
        return_metadata=return_metadata,
        num_classes=num_classes,
    )

    loader_config = config.get("dataloader", {})
    generator = torch.Generator()
    generator.manual_seed(int(loader_config.get("seed", 2026)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(loader_config.get("train_batch_size", 32)),
        shuffle=bool(loader_config.get("shuffle_train", True)),
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(loader_config.get("test_batch_size", 64)),
        shuffle=bool(loader_config.get("shuffle_test", False)),
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
    )
    return train_loader, test_loader, class_to_idx


def manifest_paths() -> tuple[Path, Path]:
    """Return the configured train and test manifest paths."""
    root = project_root()
    return (
        resolve_project_path(root / "metadata" / "manifests" / "train.csv"),
        resolve_project_path(root / "metadata" / "manifests" / "test.csv"),
    )
