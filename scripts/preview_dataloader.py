from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloaders  # noqa: E402
from src.data.manifest import read_manifest_csv  # noqa: E402
from src.data.transforms import build_transform_from_config  # noqa: E402
from src.utils.config import dataset_root_from_config, load_dataset_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview DataLoader batches and resize/pad examples.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/dataset.yaml.")
    parser.add_argument("--max-images", type=int, default=16, help="Images per batch preview.")
    return parser.parse_args()


def _require_manifests() -> None:
    missing = [
        path
        for path in (
            PROJECT_ROOT / "metadata" / "manifests" / "train.csv",
            PROJECT_ROOT / "metadata" / "manifests" / "test.csv",
            PROJECT_ROOT / "metadata" / "class_to_idx.json",
        )
        if not path.is_file()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing manifest files: {missing_text}. Please run python scripts/build_manifests.py first.")


def _metadata_value(metadata: dict[str, Any], key: str, index: int) -> Any:
    value = metadata[key]
    if torch.is_tensor(value):
        return value[index].item()
    return value[index]


def _batch_summary(name: str, images: torch.Tensor, labels: torch.Tensor, idx_to_class: dict[int, str]) -> None:
    label_values = [int(value) for value in labels.tolist()]
    class_names = [idx_to_class[value] for value in label_values]
    print(f"{name} batch image shape: {tuple(images.shape)}")
    print(f"{name} batch label shape: {tuple(labels.shape)}")
    print(f"{name} dtype: {images.dtype}")
    print(f"{name} min: {float(images.min()):.6f}")
    print(f"{name} max: {float(images.max()):.6f}")
    print(f"{name} class indices: {label_values[:16]}")
    print(f"{name} class names: {class_names[:16]}")


def _save_batch_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict[str, Any],
    idx_to_class: dict[int, str],
    output_path: Path,
    *,
    max_images: int,
) -> None:
    count = min(max_images, images.shape[0])
    cols = 4
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.25), squeeze=False)
    axes_list = [axis for row in axes for axis in row]
    for axis in axes_list:
        axis.axis("off")

    for index in range(count):
        axis = axes_list[index]
        image = images[index].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        label = int(labels[index].item())
        class_name = str(_metadata_value(metadata, "class_name", index)) or idx_to_class[label]
        instance_name = str(_metadata_value(metadata, "instance_name", index))
        axis.imshow(image)
        axis.set_title(f"{class_name}\nidx={label}  {instance_name}", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _select_examples(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    random.seed(2026)
    shuffled = rows[:]
    random.shuffle(shuffled)
    selected: list[tuple[str, dict[str, str]]] = []
    used_paths: set[str] = set()

    criteria: list[tuple[str, Callable[[dict[str, str]], bool]]] = [
        ("small", lambda row: max(int(row["width"]), int(row["height"])) < 128),
        ("large", lambda row: max(int(row["width"]), int(row["height"])) > 128),
        ("wide", lambda row: int(row["width"]) / max(1, int(row["height"])) >= 1.5),
        ("tall", lambda row: int(row["height"]) / max(1, int(row["width"])) >= 1.5),
        ("near-square", lambda row: abs(int(row["width"]) - int(row["height"])) / max(int(row["width"]), int(row["height"])) <= 0.1),
    ]

    for label, predicate in criteria:
        match = next((row for row in shuffled if predicate(row) and row["image_path"] not in used_paths), None)
        if match is None:
            match = next((row for row in shuffled if predicate(row)), None)
        if match is not None:
            selected.append((label, match))
            used_paths.add(match["image_path"])
    return selected


def _save_resize_examples(config: dict[str, Any], output_path: Path) -> None:
    train_rows = read_manifest_csv(PROJECT_ROOT / "metadata" / "manifests" / "train.csv")
    test_rows = read_manifest_csv(PROJECT_ROOT / "metadata" / "manifests" / "test.csv")
    examples = _select_examples(train_rows + test_rows)
    if not examples:
        raise RuntimeError("No images available for resize/pad examples.")

    dataset_root = dataset_root_from_config(config)
    transform = build_transform_from_config(config)
    fig, axes = plt.subplots(len(examples), 2, figsize=(7.5, len(examples) * 2.5), squeeze=False)

    for row_index, (label, row) in enumerate(examples):
        image_path = dataset_root / row["image_path"]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            result = transform.apply(rgb_image)

        axes[row_index][0].imshow(rgb_image)
        axes[row_index][0].axis("off")
        axes[row_index][0].set_title(f"{label} original\n{result.original_size[0]}x{result.original_size[1]}", fontsize=8)

        axes[row_index][1].imshow(result.image)
        axes[row_index][1].axis("off")
        axes[row_index][1].set_title(
            f"resized {result.resized_size[0]}x{result.resized_size[1]}\nfinal 128x128",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _require_manifests()
    config = load_dataset_config(args.config)
    train_loader, test_loader, class_to_idx = create_dataloaders(args.config, return_metadata=True)
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    train_images, train_labels, train_metadata = next(iter(train_loader))
    test_images, test_labels, test_metadata = next(iter(test_loader))

    _batch_summary("Train", train_images, train_labels, idx_to_class)
    _batch_summary("Test", test_images, test_labels, idx_to_class)

    preview_dir = PROJECT_ROOT / "outputs" / "previews"
    _save_batch_grid(
        train_images,
        train_labels,
        train_metadata,
        idx_to_class,
        preview_dir / "train_batch.png",
        max_images=args.max_images,
    )
    _save_batch_grid(
        test_images,
        test_labels,
        test_metadata,
        idx_to_class,
        preview_dir / "test_batch.png",
        max_images=args.max_images,
    )
    _save_resize_examples(config, preview_dir / "resize_pad_examples.png")
    print("Wrote outputs/previews/train_batch.png")
    print("Wrote outputs/previews/test_batch.png")
    print("Wrote outputs/previews/resize_pad_examples.png")


if __name__ == "__main__":
    main()
