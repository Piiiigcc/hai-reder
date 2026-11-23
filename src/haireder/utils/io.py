"""File IO helpers for JSON, JSONL, and Markdown outputs."""

from __future__ import annotations

import json
from contextlib import contextmanager
import tempfile
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the resolved path."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


@contextmanager
def atomic_text(path: Path):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: str | Path, data: Any) -> Path:
    """Write indented UTF-8 JSON."""
    output_path = ensure_parent(path)
    with atomic_text(output_path) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write UTF-8 JSON Lines."""
    output_path = ensure_parent(path)
    with atomic_text(output_path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSON Lines."""
    input_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: str | Path, text: str) -> Path:
    """Write UTF-8 text."""
    output_path = ensure_parent(path)
    with atomic_text(output_path) as handle:
        handle.write(text)
    return output_path
