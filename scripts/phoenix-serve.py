#!/usr/bin/env python3
"""Start Phoenix after loading its PostgreSQL password from a Docker secret."""

from pathlib import Path
import os
import sys


password = (
    Path("/run/secrets/phoenix_postgres_password").read_text(encoding="utf-8").strip()
)
if not password:
    raise SystemExit("Phoenix PostgreSQL password secret is empty")

os.environ["PHOENIX_POSTGRES_PASSWORD"] = password
os.execv(sys.executable, [sys.executable, "-m", "phoenix.server.main", "serve"])
