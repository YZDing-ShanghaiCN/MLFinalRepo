from __future__ import annotations

from pathlib import Path

from src.utils.paths import project_root, resolve_project_path


def test_resolve_project_path_accepts_workspace_root_relative_paths(monkeypatch) -> None:
    root = project_root()
    monkeypatch.chdir(root.parent)

    resolved = resolve_project_path(Path(root.name) / "configs" / "train.yaml")

    assert resolved == (root / "configs" / "train.yaml").resolve()
