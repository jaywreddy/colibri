from __future__ import annotations

import hashlib
import json
from pathlib import Path


def key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def cache_path(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / name
