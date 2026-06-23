from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.utils.config import dataset_root_from_config, load_dataset_config, load_splits_config
from src.utils.paths import project_root


EXPECTED_COUNTS = {"train": 6894, "test": 2284}
MANIFEST_COLUMNS = [
    "image_path",
    "class_name",
    "class_index",
    "instance_name",
    "split",
    "width",
    "height",
]


@dataclass
class ImageFilter:
    """Configurable image file filter."""

    extensions: set[str]
    required_keywords: list[str]
    excluded_keywords: list[str]


@dataclass
class ValidationReport:
    """Validation findings collected while scanning the dataset."""

    missing_class_dirs: list[str] = field(default_factory=list)
    missing_instance_dirs: list[str] = field(default_factory=list)
    broken_images: list[dict[str, str]] = field(default_factory=list)
    duplicate_image_paths: list[str] = field(default_factory=list)
    overlapping_instances: list[str] = field(default_factory=list)
    overlapping_image_paths: list[str] = field(default_factory=list)


@dataclass
class ManifestBuildResult:
    """Rows, mappings, and stats produced by manifest generation."""

    rows_by_split: dict[str, list[dict[str, Any]]]
    class_to_idx: dict[str, int]
    stats: dict[str, Any]
    report: ValidationReport


def natural_key(value: str | Path) -> list[object]:
    """Sort paths with embedded integers in human order."""
    name = value.name if isinstance(value, Path) else str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def image_filter_from_config(config: dict[str, Any]) -> ImageFilter:
    """Create an image filter from dataset config."""
    image_config = config.get("image", {})
    return ImageFilter(
        extensions={str(ext).lower() for ext in image_config.get("extensions", [])},
        required_keywords=[str(word).lower() for word in image_config.get("required_keywords", [])],
        excluded_keywords=[str(word).lower() for word in image_config.get("excluded_keywords", [])],
    )


def is_rgb_crop_image(path: Path, image_filter: ImageFilter) -> bool:
    """Return True if a file should be treated as an RGB crop input."""
    name = path.name.lower()
    if path.suffix.lower() not in image_filter.extensions:
        return False
    if any(word not in name for word in image_filter.required_keywords):
        return False
    if any(word in name for word in image_filter.excluded_keywords):
        return False
    return True


def build_class_to_idx(splits: dict[str, Any]) -> dict[str, int]:
    """Build a deterministic class-to-index mapping from split config."""
    classes: set[str] = set()
    for split_name in ("train", "test"):
        split_config = splits.get(split_name, {})
        if not isinstance(split_config, dict):
            raise ValueError(f"splits.{split_name} must be a mapping")
        classes.update(str(class_name) for class_name in split_config.keys())
    return {class_name: index for index, class_name in enumerate(sorted(classes))}


def _relative_to_dataset(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    return width, height


def _flatten_instances(splits: dict[str, Any], split_name: str) -> set[tuple[str, str]]:
    split_config = splits.get(split_name, {})
    instances: set[tuple[str, str]] = set()
    for class_name, instance_names in split_config.items():
        for instance_name in instance_names or []:
            instances.add((str(class_name), str(instance_name)))
    return instances


def find_overlapping_instances(splits: dict[str, Any]) -> list[str]:
    """Return class/instance names present in both train and test."""
    train_instances = _flatten_instances(splits, "train")
    test_instances = _flatten_instances(splits, "test")
    return [f"{class_name}/{instance_name}" for class_name, instance_name in sorted(train_instances & test_instances)]


def scan_dataset(
    config: dict[str, Any],
    splits: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], ValidationReport]:
    """Scan the configured dataset and return manifest rows."""
    root = project_root()
    dataset_root = dataset_root_from_config(config)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    class_to_idx = build_class_to_idx(splits)
    expected_classes = int(config.get("dataset", {}).get("num_classes", len(class_to_idx)))
    if len(class_to_idx) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(class_to_idx)} in splits.yaml")

    image_filter = image_filter_from_config(config)
    report = ValidationReport(overlapping_instances=find_overlapping_instances(splits))
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    seen_paths: set[str] = set()

    for split_name in ("train", "test"):
        split_config = splits.get(split_name, {})
        if not isinstance(split_config, dict):
            raise ValueError(f"splits.{split_name} must be a mapping")
        for class_name in sorted(split_config.keys()):
            class_name = str(class_name)
            class_dir = dataset_root / class_name
            if not class_dir.is_dir():
                report.missing_class_dirs.append(class_name)
                continue
            class_index = class_to_idx[class_name]
            if class_index < 0 or class_index >= expected_classes:
                raise ValueError(f"class_index out of range for {class_name}: {class_index}")

            instance_names = split_config.get(class_name) or []
            for instance_name in instance_names:
                instance_name = str(instance_name)
                instance_dir = class_dir / instance_name
                if not instance_dir.is_dir():
                    report.missing_instance_dirs.append(f"{class_name}/{instance_name}")
                    continue

                image_paths = sorted(
                    (
                        path
                        for path in instance_dir.iterdir()
                        if path.is_file() and is_rgb_crop_image(path, image_filter)
                    ),
                    key=natural_key,
                )
                for image_path in image_paths:
                    rel_path = _relative_to_dataset(image_path, dataset_root)
                    if rel_path in seen_paths:
                        report.duplicate_image_paths.append(rel_path)
                        continue
                    seen_paths.add(rel_path)

                    try:
                        width, height = _read_image_size(image_path)
                    except Exception as exc:  # noqa: BLE001 - report path and continue scanning
                        report.broken_images.append({"path": rel_path, "error": str(exc)})
                        continue

                    rows_by_split[split_name].append(
                        {
                            "image_path": rel_path,
                            "class_name": class_name,
                            "class_index": class_index,
                            "instance_name": instance_name,
                            "split": split_name,
                            "width": width,
                            "height": height,
                        }
                    )

    for split_name in ("train", "test"):
        rows_by_split[split_name].sort(key=lambda row: natural_key(str(row["image_path"])))

    train_paths = {str(row["image_path"]) for row in rows_by_split["train"]}
    test_paths = {str(row["image_path"]) for row in rows_by_split["test"]}
    report.overlapping_image_paths = sorted(train_paths & test_paths)
    return rows_by_split, class_to_idx, report


def _dimension_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [int(row[key]) for row in rows]
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
    }


def _count_by_class(rows: Iterable[dict[str, Any]], classes: Iterable[str]) -> dict[str, int]:
    counter = Counter(str(row["class_name"]) for row in rows)
    return {class_name: int(counter.get(class_name, 0)) for class_name in classes}


def compute_stats(
    rows_by_split: dict[str, list[dict[str, Any]]],
    class_to_idx: dict[str, int],
    report: ValidationReport,
    expected_counts: dict[str, int] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute dataset statistics from scanned manifest rows."""
    expected = EXPECTED_COUNTS if expected_counts is None else expected_counts
    image_config = (config or {}).get("image", {})
    classes = sorted(class_to_idx)
    all_rows = rows_by_split["train"] + rows_by_split["test"]

    per_instance_counts: dict[str, dict[str, dict[str, int]]] = {"train": {}, "test": {}}
    for split_name, rows in rows_by_split.items():
        split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            split_counts[str(row["class_name"])][str(row["instance_name"])] += 1
        per_instance_counts[split_name] = {
            class_name: {instance: int(count) for instance, count in sorted(counter.items(), key=lambda item: natural_key(item[0]))}
            for class_name, counter in sorted(split_counts.items())
        }

    class_max_dimensions: dict[str, dict[str, int | None]] = {
        class_name: {"max_width": None, "max_height": None} for class_name in classes
    }
    for row in all_rows:
        class_name = str(row["class_name"])
        width = int(row["width"])
        height = int(row["height"])
        dims = class_max_dimensions[class_name]
        dims["max_width"] = width if dims["max_width"] is None else max(int(dims["max_width"]), width)
        dims["max_height"] = height if dims["max_height"] is None else max(int(dims["max_height"]), height)

    actual_counts = {split_name: len(rows) for split_name, rows in rows_by_split.items()}
    count_differences = {
        split_name: actual_counts.get(split_name, 0) - expected.get(split_name, 0)
        for split_name in ("train", "test")
    }

    return {
        "total_images": actual_counts,
        "expected_images": expected,
        "count_differences": count_differences,
        "class_count": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "per_class_counts": {
            split_name: _count_by_class(rows, classes) for split_name, rows in rows_by_split.items()
        },
        "per_instance_counts": per_instance_counts,
        "split_width": {
            split_name: _dimension_summary(rows, "width") for split_name, rows in rows_by_split.items()
        },
        "split_height": {
            split_name: _dimension_summary(rows, "height") for split_name, rows in rows_by_split.items()
        },
        "class_max_dimensions": class_max_dimensions,
        "broken_images": report.broken_images,
        "missing_class_dirs": sorted(set(report.missing_class_dirs)),
        "missing_instance_dirs": sorted(set(report.missing_instance_dirs), key=natural_key),
        "duplicate_image_paths": sorted(set(report.duplicate_image_paths), key=natural_key),
        "overlapping_instances": report.overlapping_instances,
        "overlapping_image_paths": report.overlapping_image_paths,
        "image_path_is_relative_to": "dataset.root",
        "image_filter": {
            "required_keywords": image_config.get("required_keywords", []),
            "excluded_keywords": image_config.get("excluded_keywords", []),
            "extensions": image_config.get("extensions", []),
        },
    }


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write manifest rows to UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    """Write UTF-8 formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    """Read a manifest CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_class_to_idx(path: Path | None = None) -> dict[str, int]:
    """Read metadata/class_to_idx.json."""
    class_path = project_root() / "metadata" / "class_to_idx.json" if path is None else path
    if not class_path.is_file():
        raise FileNotFoundError(f"class_to_idx not found: {class_path}")
    with class_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(key): int(value) for key, value in data.items()}


def assert_no_leakage(report: ValidationReport) -> None:
    """Raise if train/test data leakage is detected."""
    messages: list[str] = []
    if report.overlapping_instances:
        messages.append(f"Overlapping train/test instances: {report.overlapping_instances}")
    if report.overlapping_image_paths:
        messages.append(f"Overlapping train/test image paths: {report.overlapping_image_paths}")
    if messages:
        raise RuntimeError("; ".join(messages))


def build_manifests(
    config_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    *,
    write_outputs: bool = True,
) -> ManifestBuildResult:
    """Build manifest CSVs, class mapping, and stats from configured dataset."""
    config = load_dataset_config(config_path)
    splits = load_splits_config(splits_path)
    rows_by_split, class_to_idx, report = scan_dataset(config, splits)
    stats = compute_stats(rows_by_split, class_to_idx, report, config=config)
    assert_no_leakage(report)

    if write_outputs:
        root = project_root()
        write_manifest_csv(root / "metadata" / "manifests" / "train.csv", rows_by_split["train"])
        write_manifest_csv(root / "metadata" / "manifests" / "test.csv", rows_by_split["test"])
        write_json(root / "metadata" / "class_to_idx.json", class_to_idx)
        write_json(root / "metadata" / "dataset_stats.json", stats)

    return ManifestBuildResult(
        rows_by_split=rows_by_split,
        class_to_idx=class_to_idx,
        stats=stats,
        report=report,
    )


def print_summary(stats: dict[str, Any]) -> None:
    """Print the concise dataset summary required by the phase-one task."""
    train_count = stats["total_images"].get("train", 0)
    test_count = stats["total_images"].get("test", 0)
    train_expected = stats["expected_images"].get("train", 0)
    test_expected = stats["expected_images"].get("test", 0)
    print(f"Train samples: {train_count} (expected {train_expected}, diff {train_count - train_expected})")
    print(f"Test samples: {test_count} (expected {test_expected}, diff {test_count - test_expected})")
    print(f"Classes: {stats['class_count']}")
    print(f"Broken images: {len(stats['broken_images'])}")
    print(f"Missing folders: {len(stats['missing_class_dirs']) + len(stats['missing_instance_dirs'])}")
    print(f"Overlapping instances: {len(stats['overlapping_instances'])}")
    print(f"Overlapping image paths: {len(stats['overlapping_image_paths'])}")
