"""Per-ply cut rectangles and bench marks for the box faces.

What survives of the old bonded-pair panelizer. The box is a SINGLE-PLY build
now (one written plate per face, diced out of the 5" witness blank), so the
blank solver, the packer, the vernier combs and the panel GDS writer are gone;
what the witness plate still needs from that work is the pure geometry:

  * :func:`pair_rects` — the per-face cut list, ``face:F`` (outer ply, full cut
    dims) and ``face:B`` (inner ply, inset one ply per edge), derived from
    ``assembly.bonded_cut_list``. The witness plate writes the F ply of each
    face; the B entry is kept because the cut list is stated per pair and the
    inner ply is still real glass in the finished box (bare, unwritten).
  * :func:`id_tick_rects` — the tick-code plate ID that makes a diced plate
    identifiable under a loupe, placed in the interior foil-fold band where the
    tape hides it.
  * :func:`dice_tick_rects` — the L-shaped scribe ticks just outside a plate's
    corners, inside the dicing street.
  * :func:`mirror_rects` / :func:`_transform_rects` / :func:`_transform_verts` —
    the chrome-down write transforms (x → −x, optional +90°).

All lengths are micrometers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- ID tick code -------------------------------------------------------------
# Face index (1..6 bars) + a B-layer underline, near the bottom edge of the
# plate, inside the interior foil-fold band.
ID_TICK_W_UM = 150.0
ID_TICK_LEN_UM = 400.0
ID_TICK_PITCH_UM = 350.0

# The PRODUCTION plate's pinned mark-centre offset lives with the rest of the
# box constants; re-exported here because the tick geometry is this module's.
from .production import ID_TICK_OFFSET_UM  # noqa: E402

# --- dicing ticks -------------------------------------------------------------
# Just outside each plate corner, inside the street.
DICE_TICK_LEN_UM = 400.0
DICE_TICK_W_UM = 60.0
DICE_TICK_GAP_UM = 150.0

SUBPLATE_SUFFIXES = ("F", "B")  # F = outer ply (front layer), B = inner ply


@dataclass
class PlateRect:
    """One ply's cut rectangle, W x H in um, tagged by sub-plate id."""

    face: str
    width_um: float
    height_um: float

    def area_um2(self) -> float:
        return self.width_um * self.height_um


@dataclass
class Placement:
    """A placed plate: lower-left corner (x0, y0) and the (possibly rotated)
    footprint, in blank-centered um coords (origin = blank center, +x right,
    +y up)."""

    face: str
    x0: float
    y0: float
    width_um: float   # footprint width AFTER any rotation
    height_um: float  # footprint height AFTER any rotation
    rotated: bool

    @property
    def cx(self) -> float:
        return self.x0 + self.width_um / 2.0

    @property
    def cy(self) -> float:
        return self.y0 + self.height_um / 2.0

    def corners(self) -> list[tuple[float, float]]:
        return [
            (self.x0, self.y0),
            (self.x0 + self.width_um, self.y0),
            (self.x0 + self.width_um, self.y0 + self.height_um),
            (self.x0, self.y0 + self.height_um),
        ]


def subplate_id(face: str, suffix: str) -> str:
    return f"{face}:{suffix}"


def pair_rects(
    width_um: float,
    depth_um: float,
    height_um: float,
    ply_um: float,
    spare_faces: tuple[str, ...] = (),
) -> list[PlateRect]:
    """12 sub-plate rects from the nested-shell cut list, plus one full spare
    PAIR per entry in ``spare_faces``.

    ``face:F`` is the OUTER ply (full cut dims), ``face:B`` the INNER ply (inset
    one ply per edge — genuinely smaller). Spares are hand-cleave insurance: a
    face may appear more than once in ``spare_faces`` for multiple spare pairs;
    each spare's id gains a ``:spareN`` suffix but its GEOMETRY is identical to
    the original (same masks, same ID ticks — interchangeable at the bench).
    Sorted longest-edge first, which is the order a shelf packer wants.
    """
    from .assembly import bonded_cut_list

    entries = bonded_cut_list(width_um, depth_um, height_um, ply_um)
    by_id: dict[str, tuple[float, float]] = {}
    rects = []
    for e in entries:
        sid = subplate_id(e["face"], "F" if e["ply"] == "outer" else "B")
        by_id[sid] = (e["width_um"], e["height_um"])
        rects.append(PlateRect(sid, e["width_um"], e["height_um"]))
    for i, fid in enumerate(spare_faces):
        for suf in SUBPLATE_SUFFIXES:
            sid = subplate_id(fid, suf)
            if sid not in by_id:
                raise ValueError(f"spare face {fid!r} is not a box face")
            w, h = by_id[sid]
            rects.append(PlateRect(f"{sid}:spare{i + 1}", w, h))
    rects.sort(key=lambda r: (max(r.width_um, r.height_um), r.area_um2()), reverse=True)
    return rects


def fold_band_offset_um(ply_um: float, fold_um: float) -> float:
    """Mark-center distance from the STACK (outer ply) edge.

    Centered in the interior foil-fold band [ply, ply + fold] — the ring where
    BOTH plies have glass and the interior fold hides the marks from inside.
    """
    return ply_um + fold_um / 2.0


def id_tick_rects(
    face_index: int,
    is_back: bool,
    stack_w_um: float,
    stack_h_um: float,
    ply_um: float,
    fold_um: float,
    *,
    band_offset_um: float | None = None,
) -> np.ndarray:
    """Tick-code plate ID near the bottom edge, inside the foil-fold band:
    ``face_index + 1`` bars, plus a long underline bar for the B (inner)
    sub-plate. Diced plates all look alike under a loupe — this keeps the bench
    build sane. Stack-centered coords, identical on both plies.

    ``band_offset_um`` overrides the derived ``fold_band_offset_um(ply, fold)``
    — the production plate passes its PINNED offset (see
    ``production.ID_TICK_OFFSET_UM``)."""
    n = face_index + 1
    off = (fold_band_offset_um(ply_um, fold_um)
           if band_offset_um is None else float(band_offset_um))
    cy = -(stack_h_um / 2.0 - off)
    total = (n - 1) * ID_TICK_PITCH_UM
    out = []
    for i in range(n):
        cx = -total / 2.0 + i * ID_TICK_PITCH_UM
        out.append(
            (cx - ID_TICK_W_UM / 2.0, cx + ID_TICK_W_UM / 2.0,
             cy - ID_TICK_LEN_UM / 2.0, cy + ID_TICK_LEN_UM / 2.0)
        )
    if is_back:
        half = (total + ID_TICK_PITCH_UM) / 2.0
        y0 = cy + ID_TICK_LEN_UM / 2.0 + ID_TICK_W_UM
        out.append((-half, half, y0, y0 + ID_TICK_W_UM))
    return np.asarray(out, dtype=float)


def dice_tick_rects(p: Placement) -> np.ndarray:
    """L-shaped scribe ticks just OUTSIDE each corner of a placed sub-plate,
    inside the dicing street — blank-frame coords. Doubles as a post-dice
    orientation reference (the L opens toward the plate)."""
    out = []
    for sx, cx in ((-1.0, p.x0), (1.0, p.x0 + p.width_um)):
        for sy, cy in ((-1.0, p.y0), (1.0, p.y0 + p.height_um)):
            gx = cx + sx * DICE_TICK_GAP_UM
            gy = cy + sy * DICE_TICK_GAP_UM
            out.append((
                min(gx, gx + sx * DICE_TICK_W_UM), max(gx, gx + sx * DICE_TICK_W_UM),
                min(gy, gy + sy * DICE_TICK_LEN_UM), max(gy, gy + sy * DICE_TICK_LEN_UM),
            ))
            out.append((
                min(gx, gx + sx * DICE_TICK_LEN_UM), max(gx, gx + sx * DICE_TICK_LEN_UM),
                min(gy, gy + sy * DICE_TICK_W_UM), max(gy, gy + sy * DICE_TICK_W_UM),
            ))
    return np.asarray(out, dtype=float)


# --- geometry transforms ------------------------------------------------------

def mirror_rects(rects: np.ndarray) -> np.ndarray:
    """(N,4) [x0,x1,y0,y1] mirrored about the plate's vertical axis (x → −x)."""
    if rects.size == 0:
        return rects
    out = rects.copy()
    out[:, 0] = -rects[:, 1]
    out[:, 1] = -rects[:, 0]
    return out


def _transform_rects(rects: np.ndarray, *, mirror: bool, rotated: bool) -> np.ndarray:
    if mirror:
        rects = mirror_rects(rects)
    if rotated and rects.size:
        # +90° CCW: (x, y) → (−y, x); rect [x0,x1,y0,y1] → [−y1,−y0,x0,x1].
        rects = np.stack(
            [-rects[:, 3], -rects[:, 2], rects[:, 0], rects[:, 1]], axis=1
        )
    return rects


def _transform_verts(verts: np.ndarray, *, mirror: bool, rotated: bool) -> np.ndarray:
    v = verts
    if mirror:
        v = np.stack([-v[:, 0], v[:, 1]], axis=1)
    if rotated:
        v = np.stack([-v[:, 1], v[:, 0]], axis=1)
    return v
