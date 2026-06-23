from __future__ import annotations

import pytest

from scripts.build_validation_split import build_validation_split
from src.data.manifest import read_class_to_idx, read_manifest_csv
from src.utils.paths import project_root


def _instances(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["class_name"], row["instance_name"]) for row in rows}


def _paths(rows: list[dict[str, str]]) -> set[str]:
    return {row["image_path"] for row in rows}


def test_validation_split_has_no_leakage_and_keeps_class_mapping() -> None:
    root = project_root()
    if not (root / "metadata" / "manifests" / "train.csv").is_file():
        pytest.skip("train.csv is missing; run python scripts/build_manifests.py first.")
    stats = build_validation_split()
    train_rows = read_manifest_csv(root / "metadata" / "manifests" / "train.csv")
    train_sub_rows = read_manifest_csv(root / "metadata" / "manifests" / "train_sub.csv")
    val_rows = read_manifest_csv(root / "metadata" / "manifests" / "val.csv")
    test_rows = read_manifest_csv(root / "metadata" / "manifests" / "test.csv")
    class_to_idx = read_class_to_idx(root / "metadata" / "class_to_idx.json")

    assert len(train_sub_rows) + len(val_rows) == len(train_rows) == 6894
    assert _instances(train_sub_rows).isdisjoint(_instances(val_rows))
    assert _paths(train_sub_rows).isdisjoint(_paths(val_rows))
    assert _instances(val_rows).isdisjoint(_instances(test_rows))
    assert _paths(val_rows).isdisjoint(_paths(test_rows))
    assert {row["class_name"] for row in train_sub_rows} == set(class_to_idx)
    assert {row["class_name"] for row in val_rows} == set(class_to_idx)
    assert {row["class_name"] for row in test_rows} == set(class_to_idx)
    assert all(class_to_idx[row["class_name"]] == int(row["class_index"]) for row in train_sub_rows + val_rows + test_rows)
    assert stats["all_splits_cover_13_classes"] is True
