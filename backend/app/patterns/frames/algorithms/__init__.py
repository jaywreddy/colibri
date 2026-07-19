from __future__ import annotations

from . import colonize, wreath

ALGORITHMS = {
    "colonize": colonize.generate,
    "wreath": wreath.generate,
}

__all__ = ["ALGORITHMS", "colonize", "wreath"]
