from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import project_root, resolve_project_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    yaml_path = resolve_project_path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {yaml_path}")
    return data


def load_dataset_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/dataset.yaml by default."""
    path = Path("configs/dataset.yaml") if config_path is None else Path(config_path)
    return load_yaml(path)


def load_splits_config(splits_path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/splits.yaml by default."""
    path = Path("configs/splits.yaml") if splits_path is None else Path(splits_path)
    return load_yaml(path)


def load_model_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/model.yaml by default."""
    path = Path("configs/model.yaml") if config_path is None else Path(config_path)
    return load_yaml(path)


def load_train_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/train.yaml by default."""
    path = Path("configs/train.yaml") if config_path is None else Path(config_path)
    return load_yaml(path)


def dataset_root_from_config(config: dict[str, Any]) -> Path:
    """Resolve dataset.root relative to the ml_repo project root."""
    try:
        root_value = config["dataset"]["root"]
    except KeyError as exc:
        raise KeyError("Missing dataset.root in dataset config") from exc
    return resolve_project_path(root_value, project_root())
