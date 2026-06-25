from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the ml_repo project root without relying on cwd."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless it is absolute.

    Existing paths are also accepted relative to the current working directory
    so scripts can be launched from the repository parent with paths such as
    ``ml_repo/configs/train.yaml``.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    base = project_root() if root is None else root
    project_candidate = (base / path).resolve()
    if project_candidate.exists() or root is not None:
        return project_candidate
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if path.parts and path.parts[0] == base.name:
        return (base.parent / path).resolve()
    return project_candidate


def relative_to_project(path: Path, root: Path | None = None) -> str:
    """Return a POSIX-style path relative to the project root when possible."""
    base = project_root() if root is None else root
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
