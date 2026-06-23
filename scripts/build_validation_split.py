from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import MANIFEST_COLUMNS, read_class_to_idx, read_manifest_csv, write_manifest_csv  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instance-safe train_sub/validation manifests.")
    parser.add_argument("--config", type=Path, default=Path("configs/validation_split.yaml"))
    return parser.parse_args()


def _rows_by_class(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["class_name"] for row in rows)
    return {class_name: int(counts[class_name]) for class_name in sorted(counts)}


def _instances(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["class_name"], row["instance_name"]) for row in rows}


def _paths(rows: list[dict[str, str]]) -> set[str]:
    return {row["image_path"] for row in rows}


def _assert_columns(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    missing = [column for column in MANIFEST_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {missing}")


def build_validation_split(config_path: str | Path = "configs/validation_split.yaml") -> dict[str, Any]:
    """Build train_sub.csv and val.csv from train.csv using validation instances."""
    config = load_yaml(config_path)
    val_config = config.get("val_instances", {})
    if not isinstance(val_config, dict):
        raise ValueError("validation_split.yaml must contain val_instances mapping")

    manifest_dir = PROJECT_ROOT / "metadata" / "manifests"
    train_path = manifest_dir / "train.csv"
    test_path = manifest_dir / "test.csv"
    train_rows = read_manifest_csv(train_path)
    test_rows = read_manifest_csv(test_path)
    _assert_columns(train_rows, train_path)
    _assert_columns(test_rows, test_path)

    val_instances = {
        (str(class_name), str(instance_name))
        for class_name, instance_names in val_config.items()
        for instance_name in instance_names or []
    }
    if len(val_instances) != 13:
        raise ValueError(f"Expected 13 validation instances, found {len(val_instances)}")

    train_sub_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in train_rows:
        copied = dict(row)
        key = (copied["class_name"], copied["instance_name"])
        if key in val_instances:
            copied["split"] = "val"
            val_rows.append(copied)
        else:
            copied["split"] = "train_sub"
            train_sub_rows.append(copied)

    train_sub_instances = _instances(train_sub_rows)
    val_row_instances = _instances(val_rows)
    test_instances = _instances(test_rows)
    train_sub_paths = _paths(train_sub_rows)
    val_paths = _paths(val_rows)
    test_paths = _paths(test_rows)

    missing_val_instances = sorted(f"{cls}/{inst}" for cls, inst in val_instances - val_row_instances)
    overlapping_train_val_instances = sorted(f"{cls}/{inst}" for cls, inst in train_sub_instances & val_row_instances)
    overlapping_train_val_paths = sorted(train_sub_paths & val_paths)
    overlapping_val_test_instances = sorted(f"{cls}/{inst}" for cls, inst in val_row_instances & test_instances)
    overlapping_val_test_paths = sorted(val_paths & test_paths)

    class_to_idx = read_class_to_idx(PROJECT_ROOT / "metadata" / "class_to_idx.json")
    classes = sorted(class_to_idx)
    train_sub_classes = {row["class_name"] for row in train_sub_rows}
    val_classes = {row["class_name"] for row in val_rows}
    test_classes = {row["class_name"] for row in test_rows}
    all_have_13_classes = all({*classes} == class_set for class_set in (train_sub_classes, val_classes, test_classes))

    bad_index_rows = [
        row["image_path"]
        for row in train_sub_rows + val_rows + test_rows
        if class_to_idx.get(row["class_name"]) != int(row["class_index"])
    ]
    validation_instance_counts = Counter((row["class_name"], row["instance_name"]) for row in val_rows)
    stats: dict[str, Any] = {
        "original_train_total": len(train_rows),
        "train_sub_total": len(train_sub_rows),
        "val_total": len(val_rows),
        "expected_original_train_total": 6894,
        "train_sub_plus_val_equals_original_train": len(train_sub_rows) + len(val_rows) == len(train_rows),
        "train_sub_per_class": _rows_by_class(train_sub_rows),
        "val_per_class": _rows_by_class(val_rows),
        "validation_instance_counts": {
            f"{class_name}/{instance_name}": int(count)
            for (class_name, instance_name), count in sorted(validation_instance_counts.items())
        },
        "missing_validation_instances": missing_val_instances,
        "train_val_overlapping_instances": overlapping_train_val_instances,
        "train_val_overlapping_image_paths": overlapping_train_val_paths,
        "val_test_overlapping_instances": overlapping_val_test_instances,
        "val_test_overlapping_image_paths": overlapping_val_test_paths,
        "all_splits_cover_13_classes": all_have_13_classes,
        "class_index_consistent": len(bad_index_rows) == 0,
        "bad_class_index_rows": bad_index_rows,
    }

    errors: list[str] = []
    if missing_val_instances:
        errors.append(f"Validation instances not found in train.csv: {missing_val_instances}")
    if overlapping_train_val_instances:
        errors.append(f"train_sub/val instance overlap: {overlapping_train_val_instances}")
    if overlapping_train_val_paths:
        errors.append(f"train_sub/val image path overlap: {overlapping_train_val_paths[:5]}")
    if overlapping_val_test_instances:
        errors.append(f"val/test instance overlap: {overlapping_val_test_instances}")
    if overlapping_val_test_paths:
        errors.append(f"val/test image path overlap: {overlapping_val_test_paths[:5]}")
    if not stats["train_sub_plus_val_equals_original_train"]:
        errors.append("train_sub + val does not equal original train row count")
    if not all_have_13_classes:
        errors.append("train_sub, val, and test must each cover all 13 classes")
    if bad_index_rows:
        errors.append(f"class_index mismatch in {len(bad_index_rows)} rows")
    if errors:
        raise RuntimeError("; ".join(errors))

    write_manifest_csv(manifest_dir / "train_sub.csv", train_sub_rows)
    write_manifest_csv(manifest_dir / "val.csv", val_rows)
    stats_path = PROJECT_ROOT / "metadata" / "validation_split_stats.json"
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return stats


def main() -> None:
    args = parse_args()
    stats = build_validation_split(resolve_project_path(args.config))
    print(f"Original train samples: {stats['original_train_total']}")
    print(f"Train_sub samples: {stats['train_sub_total']}")
    print(f"Validation samples: {stats['val_total']}")
    print(f"train_sub + val == train: {stats['train_sub_plus_val_equals_original_train']}")
    print(f"All splits cover 13 classes: {stats['all_splits_cover_13_classes']}")
    print(f"train_sub/val overlapping instances: {len(stats['train_val_overlapping_instances'])}")
    print(f"train_sub/val overlapping image paths: {len(stats['train_val_overlapping_image_paths'])}")
    print(f"val/test overlapping instances: {len(stats['val_test_overlapping_instances'])}")
    print(f"val/test overlapping image paths: {len(stats['val_test_overlapping_image_paths'])}")
    print("Wrote metadata/manifests/train_sub.csv")
    print("Wrote metadata/manifests/val.csv")
    print("Wrote metadata/validation_split_stats.json")


if __name__ == "__main__":
    main()
