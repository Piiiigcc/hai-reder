"""Small logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_file_logger(name: str, log_path: str | Path) -> logging.Logger:
    """Create a logger that writes plain text logs to a file."""
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
