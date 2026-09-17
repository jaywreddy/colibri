"""The one simulation route left: the renderer's diffraction colour table.

Everything else that lived under ``/sim`` was a holography or parallax LAB —
``/fft``, ``/propagate`` (the angular-spectrum and Fraunhofer kernels),
``/parallax2d`` and its curve, ``/collage``, ``/readability``. They scored the
two-ply constructions the box abandoned on 2026-09-15, were reachable only from
views that went with them, and are in git history at 22d1634.

What is left is not a simulation of the box at all: it is a BAKED TABLE the
shader samples. The physics (grating equation, square-wave order series, CIE
integration) runs in ``app.diffraction`` where it is unit-tested against closed
forms; the renderer does one texture fetch per pixel. See ``/diffraction/lut``.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/sim", tags=["sim"])
_log = logging.getLogger("optics.sim")


@router.get("/diffraction/lut")
def diffraction_lut(
    duty: float = 0.5,
    u_max_um: float = 10.0,
    size: int = 1024,
    orders: int = 16,
) -> dict:
    """Baked diffraction colour table for the renderer's spectral accent.

    Indexed by the optical path term ``u = period * (V.g + L.g)`` in um, so ONE
    table serves every pitch on the plate — the region_art period map, the photo
    colour zones and the garland's leaf families all read it. Values are LINEAR
    sRGB. See ``app.diffraction`` for the derivation.
    """
    from ..diffraction import lut_payload

    if not (0.0 < duty < 1.0):
        raise HTTPException(400, f"duty must be in (0, 1) (got {duty})")
    if not (2 <= size <= 4096):
        raise HTTPException(400, f"size must be 2..4096 (got {size})")
    if not (1 <= orders <= 64):
        raise HTTPException(400, f"orders must be 1..64 (got {orders})")
    if not (0.1 <= u_max_um <= 100.0):
        raise HTTPException(400, f"u_max_um must be 0.1..100 (got {u_max_um})")
    return lut_payload(duty=duty, u_max_um=u_max_um, size=size, orders=orders)
