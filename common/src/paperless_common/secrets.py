"""Shared secret/env helpers."""

from __future__ import annotations

import os
from pathlib import Path


def read_secret(env_var: str) -> str | None:
    """Read a secret from FOO_FILE if present, otherwise from FOO."""
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        path = Path(file_path)
        try:
            if path.is_file():
                return path.read_text().strip()
        except (OSError, ValueError):
            pass
    return os.environ.get(env_var)
