from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.manifest import read_manifest_csv
from src.data.transforms import ResizeLongestSideAndPad
from src.utils.config import dataset_root_from_config, load_dataset_config
from src.utils.paths import project_root, resolve_project_path


class RGBDManifestDataset(Dataset):
    """PyTorch Dataset backed by a prebuilt manifest CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        return_metadata: bool = False,
        num_classes: int = 13,
    ) -> None:
        self.manifest_path = resolve_project_path(manifest_path)
        self.rows = read_manifest_csv(self.manifest_path)
        if dataset_root is None:
            config = load_dataset_config()
            self.dataset_root = dataset_root_from_config(config)
        else:
            self.dataset_root = resolve_project_path(dataset_root, project_root())
        self.transform = transform or ResizeLongestSideAndPad(size=128)
        self.return_metadata = return_metadata
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, row: dict[str, str]) -> Path:
        path = Path(row["image_path"])
        return path if path.is_absolute() else self.dataset_root / path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[index]
        image_path = self._image_path(row)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found for manifest row {index}: {image_path}")

        try:
            label_value = int(row["class_index"])
        except Exception as exc:  # noqa: BLE001 - include row context
            raise ValueError(f"Invalid class_index in {self.manifest_path} row {index}: {row}") from exc
        if label_value < 0 or label_value >= self.num_classes:
            raise ValueError(f"Label out of range for image {image_path}: {label_value}")

        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image)
        except Exception as exc:  # noqa: BLE001 - include image path in error
            raise RuntimeError(f"Failed to read or transform image {image_path}: {exc}") from exc

        expected_shape = (3, 128, 128)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Unexpected tensor shape for {image_path}: {tuple(tensor.shape)}, expected {expected_shape}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"Unexpected tensor dtype for {image_path}: {tensor.dtype}, expected torch.float32")

        label = torch.tensor(label_value, dtype=torch.long)
        if not self.return_metadata:
            return tensor, label

        metadata: dict[str, Any] = {
            "path": row["image_path"],
            "class_name": row["class_name"],
            "class_index": label_value,
            "instance_name": row["instance_name"],
            "split": row["split"],
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        return tensor, label, metadata
