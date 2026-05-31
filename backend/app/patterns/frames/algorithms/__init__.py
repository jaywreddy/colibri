from __future__ import annotations

from . import colonize

ALGORITHMS = {
    "colonize": colonize.generate,
}

__all__ = ["ALGORITHMS", "colonize"]
