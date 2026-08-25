"""Small file helpers: read/write JSON, ensure results dirs exist, etc."""

from __future__ import annotations

import json
from pathlib import Path


def read_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
