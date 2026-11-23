"""Environment inspection helpers."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


CORE_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "sklearn",
    "pandas",
    "numpy",
    "sentence_transformers",
    "yaml",
    "pytest",
    "pymongo",
]


def module_available(name: str) -> bool:
    """Return whether a Python module can be imported in the current interpreter."""
    return importlib.util.find_spec(name) is not None


def detect_torch_backend() -> dict[str, Any]:
    """Detect torch and common acceleration backends."""
    if not module_available("torch"):
        return {
            "torch_available": False,
            "mps_available": False,
            "cuda_available": False,
            "selected_backend": "unavailable",
        }

    import torch

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    selected = "mps" if mps_available else "cuda" if cuda_available else "cpu"
    return {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "selected_backend": selected,
    }


def local_port_open(host: str, port: int, timeout_seconds: float = 0.5) -> bool:
    """Check whether a local TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def detect_mongodb() -> dict[str, Any]:
    """Detect MongoDB command-line tools, Python driver, and localhost service."""
    cli_tools = {name: shutil.which(name) for name in ["mongosh", "mongo", "mongod", "mongorestore"]}
    result: dict[str, Any] = {
        "cli_tools": cli_tools,
        "pymongo_available": module_available("pymongo"),
        "localhost_27017_open": local_port_open("localhost", 27017),
    }
    if result["pymongo_available"] and result["localhost_27017_open"]:
        try:
            from pymongo import MongoClient

            client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=800)
            result["server_info_ok"] = bool(client.server_info())
            result["databases"] = client.list_database_names()
        except Exception as exc:  # pragma: no cover - diagnostic only
            result["server_info_ok"] = False
            result["error"] = repr(exc)
    else:
        result["server_info_ok"] = False
    return result


def detect_conda() -> dict[str, Any]:
    """Return available conda environments when conda is installed."""
    conda_path = shutil.which("conda")
    if not conda_path:
        return {"conda_available": False, "conda_path": None, "envs": []}
    try:
        completed = subprocess.run(
            [conda_path, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            envs = payload.get("envs", [])
        else:
            envs = []
    except Exception:
        envs = []
    return {"conda_available": True, "conda_path": conda_path, "envs": envs}


def inspect_environment() -> dict[str, Any]:
    """Collect reproducibility-relevant environment details."""
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "modules": {name: module_available(name) for name in CORE_MODULES},
        "torch_backend": detect_torch_backend(),
        "mongodb": detect_mongodb(),
        "conda": detect_conda(),
    }


def environment_markdown(report: dict[str, Any]) -> str:
    """Render environment inspection results as Markdown."""
    lines = [
        "# Environment Check",
        "",
        "## Python",
        "",
        f"- Executable: `{report['python']['executable']}`",
        f"- Version: `{report['python']['version'].splitlines()[0]}`",
        "",
        "## Platform",
        "",
        f"- System: `{report['platform']['system']}`",
        f"- Machine: `{report['platform']['machine']}`",
        f"- Platform: `{report['platform']['platform']}`",
        "",
        "## Python Modules",
        "",
        "| Module | Available |",
        "|---|---|",
    ]
    for name, available in report["modules"].items():
        lines.append(f"| `{name}` | {available} |")
    torch_backend = report["torch_backend"]
    lines.extend(
        [
            "",
            "## Torch Backend",
            "",
            f"- Torch available: `{torch_backend.get('torch_available')}`",
            f"- Torch version: `{torch_backend.get('torch_version')}`",
            f"- MPS available: `{torch_backend.get('mps_available')}`",
            f"- CUDA available: `{torch_backend.get('cuda_available')}`",
            f"- Selected backend: `{torch_backend.get('selected_backend')}`",
            "",
            "## MongoDB",
            "",
            f"- pymongo available: `{report['mongodb']['pymongo_available']}`",
            f"- localhost:27017 open: `{report['mongodb']['localhost_27017_open']}`",
            f"- server info ok: `{report['mongodb']['server_info_ok']}`",
            "",
            "| CLI Tool | Path |",
            "|---|---|",
        ]
    )
    for name, path in report["mongodb"]["cli_tools"].items():
        lines.append(f"| `{name}` | `{path}` |")
    lines.extend(["", "## Conda", ""])
    lines.append(f"- Conda available: `{report['conda']['conda_available']}`")
    lines.append(f"- Conda path: `{report['conda']['conda_path']}`")
    for env_path in report["conda"]["envs"]:
        lines.append(f"- Env: `{env_path}`")
    lines.append("")
    return "\n".join(lines)
