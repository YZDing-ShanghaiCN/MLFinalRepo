from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the ml_repo project root without relying on cwd."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    base = project_root() if root is None else root
    return (base / path).resolve()


def relative_to_project(path: Path, root: Path | None = None) -> str:
    """Return a POSIX-style path relative to the project root when possible."""
    base = project_root() if root is None else root
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
