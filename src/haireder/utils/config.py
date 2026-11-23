"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root from this utility module location."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a user-provided path relative to the project root by default."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    base = Path(base_dir).expanduser() if base_dir is not None else project_root()
    return (base / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {config_path}")
    return data


def resolve_scoped_path(
    path_value: str | Path,
    allowed_roots: tuple[str, ...],
    base_dir: str | Path | None = None,
) -> Path:
    resolved = resolve_path(path_value, base_dir)
    repo_root = project_root()
    allowed = [repo_root / root for root in allowed_roots]
    if any(resolved.is_relative_to(root.resolve()) for root in allowed):
        return resolved
    raise ValueError(
        f"Resolved path {resolved} is outside allowed roots: {', '.join(str(root) for root in allowed)}"
    )
