"""Metrology cells for the witness plate: polarity, the step wedge, the ladders.

Split from ``witness_cells`` (halftone, gratings, the production dies) when this
module was the plate's moiré programme — a third of its area in beat, rotation,
vector, harmonic and contrast cells. Those went with the two-ply design
(2026-09-16); their physics is not lost, it is pinned as FORMULAS in
``tests/test_optics_math.py`` against ``witness_geom`` and
``patterns.effects.moire``, which is where a number that cannot be measured on
this plate belongs.

What is left is the bench the single-ply box actually reads: the polarity
witness, the halftone step wedge, and the CD / duty / period / acuity ladders —
each of them ONE cell carrying every rung, which is the shape correction that
bought back a third of the plate.

These cells are SMALL in shape count (a few hundred to a few thousand
rectangles, against 10^5-10^6 for a halftone), which is why the plate may take
their clear-field complement with a klayout boolean per cell rather than
analytically: at this scale it is exact, trivial, and immune to the
per-structure reasoning errors an analytic inverse invites. The analytic path
stays reserved for the cells that would otherwise be a whole-plate GEOS
operation.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .witness_geom import (CellArt, _cat, _rect, grating_array,
                           grating_array_inverse)

# A cell whose metal exceeds this many shapes must invert analytically; the
# boolean is only safe while it is small. Sized well under the halftone cells
# (which are 10^5-10^6 and carry their own inverse) and well over every cell in
# this module (10^2-10^4).
MAX_BOOLEAN_SHAPES = 60_000


# --- metrology --------------------------------------------------------------
def build_polarity_witness(cx: float, cy: float, w: float, h: float) -> CellArt:
    """M-POL — is the write the polarity you think it is?

    A solid square with a square hole in it, beside a bar-and-gap pair of known
    asymmetry. Under a correct clear-field write the SQUARE reads dark (chrome)
    with a clear centre; inverted, it is a clear square with a dark centre. Ten
    seconds, and it gates the interpretation of everything else on the plate.

    Emitted as METAL like every other builder; the writer inverts it with the
    rest, which is the point — a witness that bypassed the polarity path would
    witness nothing.
    """
    s = min(w, h)
    outer = s * 0.42
    inner = s * 0.16
    ox = cx - s * 0.24
    ring = _cat(
        _rect(ox - outer / 2, cy - outer / 2, ox + outer / 2, cy - inner / 2),
        _rect(ox - outer / 2, cy + inner / 2, ox + outer / 2, cy + outer / 2),
        _rect(ox - outer / 2, cy - inner / 2, ox - inner / 2, cy + inner / 2),
        _rect(ox + inner / 2, cy - inner / 2, ox + outer / 2, cy + inner / 2),
    )
    # A 1:3 bar/gap pair: unmistakably asymmetric, so an inversion cannot be
    # mistaken for a shift.
    bx = cx + s * 0.24
    bars = []
    for i, frac in enumerate((0.25, 0.75)):
        y0 = cy - outer / 2 + i * outer * 0.55
        bars.append(_rect(bx - outer / 2, y0, bx - outer / 2 + outer * frac,
                          y0 + outer * 0.35))
    return CellArt(
        front=_cat(ring, *bars),
        stats={"outer_um": round(outer, 1), "inner_um": round(inner, 1),
               "n_rects": int(len(ring) + len(bars))},
    )


def build_step_wedge(cx: float, cy: float, w: float, h: float, *,
                     line_period_um: float = 44.0, steps: int = 16,
                     tone_steps: int = 22, polarity: str = "metal") -> CellArt:
    """H-WEDGE — the dot-gain instrument.

    ``steps`` patches of halftone at known, evenly spaced duty. Measure the
    realized coverage of each and the curve inverts straight into the prep's
    ``gain``. This is what replaced eight subjective portrait sweeps: a
    photograph reads many coupled variables at once, a wedge reads one.

    In clear polarity each band becomes the two gaps either side of it, exactly
    as ``screenrects.screen_bands(emit="clear")`` does for a photograph.
    """
    patch_w = w / steps
    n_lines = max(1, int(h / line_period_um))
    parts: list[np.ndarray] = []
    duties: list[float] = []
    levels: list[int] = []
    top = cy + h / 2.0
    for i in range(steps):
        # Duty is quantised to the SCREEN's own ladder, not to a round number,
        # so the wedge measures tones the plate can actually make.
        t = (i + 0.5) / steps
        lvl = max(1, min(tone_steps - 1, int(round(t * tone_steps))))
        band = (lvl / tone_steps) * line_period_um
        duties.append(round(lvl / tone_steps, 4))
        levels.append(lvl)
        x0 = cx - w / 2.0 + i * patch_w
        ys = top - (np.arange(n_lines) + 0.5) * line_period_um
        if polarity == "metal":
            r = np.empty((n_lines, 4), dtype=np.float64)
            r[:, 0], r[:, 1] = x0, x0 + patch_w
            r[:, 2], r[:, 3] = ys - band / 2.0, ys + band / 2.0
        else:
            hp = line_period_um / 2.0
            r = np.empty((2 * n_lines, 4), dtype=np.float64)
            r[:, 0], r[:, 1] = x0, x0 + patch_w
            r[:n_lines, 2], r[:n_lines, 3] = ys + band / 2.0, ys + hp
            r[n_lines:, 2], r[n_lines:, 3] = ys - hp, ys - band / 2.0
        parts.append(r)
    if polarity != "metal":
        rem_top = top - n_lines * line_period_um
        if rem_top > cy - h / 2.0 + 1e-9:
            parts.append(_rect(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, rem_top))
    return CellArt(
        front=_cat(*parts),
        stats={"polarity": polarity, "line_period_um": line_period_um,
               "steps": steps, "duties": duties, "levels": levels,
               "tone_steps": tone_steps,
               "finest_band_um": round(line_period_um / tone_steps, 3),
               "n_rects": int(sum(len(p_) for p_ in parts))},
    )
def build_ladder_strip(
    cx: float, cy: float, w: float, h: float, *,
    rungs: Sequence[tuple[str, float, float]],
    gap_frac: float = 0.12, polarity: str = "metal",
) -> CellArt:
    """A whole ladder as a single cell: N sub-patches side by side.

    This is the shape correction that bought back a third of the plate. A ladder
    drawn as N separate cells pays the per-cell overhead N times — a label band
    and a row gutter, 2.4 mm against a 4 mm rung, 60% overhead — and scatters
    rungs across rows so the eye cannot compare them. As one cell the rungs abut,
    the comparison is direct, and the overhead is paid once.

    ``rungs`` is ``(label, period_um, duty)``. A clear separator between rungs
    keeps a boundary from reading as a defect. Each rung is ONE array reference
    in either polarity — the clear complement of a grating is a grating — which
    keeps a 36 mm ladder of 2 um lines to a dozen records.
    """
    n = max(1, len(rungs))
    pitch = w / n
    patch = pitch * (1.0 - gap_frac)
    arrays: list[dict[str, Any]] = []
    seps: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    for i, (lab, period, duty) in enumerate(rungs):
        x = cx - w / 2.0 + (i + 0.5) * pitch
        f = grating_array if polarity == "metal" else grating_array_inverse
        entry, edges = f(x, cy, patch, h, period, duty)
        arrays.append(entry)
        seps.append(edges)
        if polarity != "metal":
            # Separators are bare glass in metal polarity, so CLEAR here, and
            # have to be written.
            seps.append(_rect(cx - w / 2.0 + i * pitch, y0, x - patch / 2.0, y1))
            seps.append(_rect(x + patch / 2.0, y0, cx - w / 2.0 + (i + 1) * pitch, y1))
        line = period * duty
        meta.append({
            "label": lab, "period_um": period, "duty": duty,
            "line_um": round(line, 3), "gap_um": round(period - line, 3),
            "clears_floor": bool(line >= 2.0 and period - line >= 2.0),
        })
    art = CellArt(front=_cat(*seps), arrays=arrays)
    art.stats = {"polarity": polarity, "n_rungs": n, "patch_um": round(patch, 1),
                 "rungs": meta,
                 "n_below_floor": sum(1 for m in meta if not m["clears_floor"]),
                 "n_rects": int(len(art.front)), "n_arrays": len(arrays)}
    return art


def cd_rungs(periods: Sequence[float], dense: bool = True
             ) -> list[tuple[str, float, float]]:
    """M-CD rungs. Isolated lines are the same WIDTH at ten times the pitch, so
    the two strips differ only in their environment — which is the whole point
    of an iso-dense pair."""
    if dense:
        return [(f"{p:g}", p, 0.5) for p in periods]
    return [(f"{p:g}", p * 10.0, 0.05) for p in periods]


def duty_rungs(period_um: float, duties: Sequence[float]
               ) -> list[tuple[str, float, float]]:
    return [(f"{c:.2f}", period_um, c) for c in duties]


def period_rungs(periods: Sequence[float], duty: float = 0.5
                 ) -> list[tuple[str, float, float]]:
    return [(f"{p:g}", p, duty) for p in periods]