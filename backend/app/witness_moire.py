"""Moiré, metrology and edge-of-envelope cells for the witness plate.

Split from ``witness_cells`` (halftone, gratings, parallax) because this is the
largest block on the plate and it exists for one reason: moiré is the mechanism
with the most unmeasured parameters and the least margin.

Two facts from ``docs/witness-physics-plan.md`` shape everything here.

FIRST, a single-layer superposition is optically identical to a two-layer stack
at zero gap, because ``1 - (A1 | A2) == (1 - A1)(1 - A2)``. So the whole STATIC
moiré programme — beat geometry, rotation, the vector formula, harmonic beats,
contrast — costs no bond at all. Only motion, sampling and registration need two
plies. That is what let the plate give a third of its area to moiré without
gambling it all on a bonding step that may never happen.

SECOND, these cells are SMALL in shape count. A 10 mm cell of 63.5 µm gratings
is a few hundred rectangles, where a halftone portrait is half a million. That
is why the clear-field complement here is taken with a klayout boolean per cell
rather than analytically: at this scale it is exact, trivial, and immune to the
per-structure reasoning errors that an analytic inverse invites. The analytic
path stays reserved for the cells that would otherwise be a whole-plate GEOS
operation.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .witness_geom import (CellArt, _cat, _grating_rects, _rect,
                           column_complement, grating_array,
                           grating_array_inverse, invert_grating,
                           outside_boxes)

# A cell whose metal exceeds this many shapes must invert analytically; the
# boolean is only safe while it is small. Sized well under the halftone cells
# (which are 10^5-10^6 and carry their own inverse) and well over every cell in
# this module (10^2-10^4).
MAX_BOOLEAN_SHAPES = 60_000


# --- rotated geometry -------------------------------------------------------


def rotate_rects(rects: np.ndarray, angle_deg: float,
                 about: tuple[float, float]) -> np.ndarray:
    """``(N, 4)`` rects -> ``(N, 4, 2)`` polygons, rotated about a point.

    Rotational moiré needs real rotated geometry: staircasing a 1° rotation onto
    an axis-aligned grid would introduce its own periodic error at exactly the
    scale the cell is trying to measure.
    """
    r = np.asarray(rects, dtype=np.float64)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    ox, oy = about
    v = np.empty((len(r), 4, 2), dtype=np.float64)
    v[:, 0, 0], v[:, 0, 1] = r[:, 0], r[:, 2]
    v[:, 1, 0], v[:, 1, 1] = r[:, 1], r[:, 2]
    v[:, 2, 0], v[:, 2, 1] = r[:, 1], r[:, 3]
    v[:, 3, 0], v[:, 3, 1] = r[:, 0], r[:, 3]
    x, y = v[..., 0] - ox, v[..., 1] - oy
    out = np.empty_like(v)
    out[..., 0] = x * ca - y * sa + ox
    out[..., 1] = x * sa + y * ca + oy
    return out


def _clip_convex(poly: np.ndarray, x0: float, y0: float,
                 x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon to an axis-aligned box.

    A rotated rectangle clipped to a box is convex with at most eight vertices,
    so the classic algorithm is exact here and needs no library.
    """
    # (axis, bound, keep-side): keep points on the inner side of each edge.
    edges = ((0, x0, +1), (0, x1, -1), (1, y0, +1), (1, y1, -1))
    pts = np.asarray(poly, dtype=np.float64)
    for axis, bound, side in edges:
        if not len(pts):
            return np.empty((0, 2))
        inside = (pts[:, axis] - bound) * side >= 0.0
        if inside.all():
            continue
        out: list[np.ndarray] = []
        n = len(pts)
        for i in range(n):
            a_, b_ = pts[i], pts[(i + 1) % n]
            ia, ib = inside[i], inside[(i + 1) % n]
            if ia:
                out.append(a_)
            if ia != ib:
                d = b_ - a_
                if abs(d[axis]) > 1e-12:
                    out.append(a_ + ((bound - a_[axis]) / d[axis]) * d)
        pts = np.asarray(out, dtype=np.float64)
    return pts


def _clip_polys_to_box(polys: np.ndarray, cx: float, cy: float,
                       w: float, h: float) -> list[np.ndarray]:
    """Clip rotated polygons to the cell, not merely reject them.

    A bbox reject was not enough: a rotated grating's end lines overhang the
    cell by up to half its diagonal, so in METAL polarity they would spill into
    the neighbouring cell, and the clear-field inverse (a box minus the metal)
    would then not be the metal's complement. The area check catches exactly
    this, which is why it is worth running.
    """
    if not len(polys):
        return []
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    out: list[np.ndarray] = []
    for pv in polys:
        if (pv[:, 0].max() <= x0 or pv[:, 0].min() >= x1
                or pv[:, 1].max() <= y0 or pv[:, 1].min() >= y1):
            continue
        if (pv[:, 0].min() >= x0 and pv[:, 0].max() <= x1
                and pv[:, 1].min() >= y0 and pv[:, 1].max() <= y1):
            out.append(pv)
            continue
        c = _clip_convex(pv, x0, y0, x1, y1)
        if len(c) >= 3:
            out.append(c)
    return out


def _rotated_grating(cx: float, cy: float, w: float, h: float,
                     period_um: float, duty: float, angle_deg: float,
                     phase_um: float = 0.0) -> list[np.ndarray]:
    """A grating at an arbitrary angle, as clipped polygons.

    Built oversized (sqrt(2) on the diagonal) then rotated and bbox-rejected, so
    the cell is fully covered whatever the angle.
    """
    span = math.hypot(w, h) * 1.05
    r = _grating_rects(cx, cy, span, span, period_um, duty,
                       phase_um=phase_um, vertical=True)
    return _clip_polys_to_box(rotate_rects(r, angle_deg, (cx, cy)), cx, cy, w, h)


# --- moiré ------------------------------------------------------------------


def beat_delta_for(period_um: float, beat_um: float) -> float:
    """Pitch difference that produces a given beat: ``delta = p^2/(beat - p)``.

    Solved from the beat rather than the other way round. Pinning delta instead
    is what made an earlier cell read 0.075 where it should have read 0.47.
    """
    if beat_um <= period_um:
        raise ValueError(f"beat {beat_um} must exceed the pitch {period_um}")
    return period_um * period_um / (beat_um - period_um)


def build_beat(cx: float, cy: float, w: float, h: float, *,
               period_um: float = 63.5, beat_um: float = 1635.0,
               duty: float = 0.5) -> CellArt:
    """B-BEAT — two pitches superposed on ONE plane. Static fringes.

    The union of the two gratings, which by the complement identity is exactly
    what two plies in contact would transmit. Coverage swings from ``duty``
    where the two align to ``2*duty - duty^2`` where they do not, so the fringe
    contrast is high and the cell reads without any bond.

    The cell also doubles as a pitch-accuracy meter with ``p/delta`` gain — 24.7x
    at the shipping numbers — because the fringe count you measure inverts
    straight back to the delta you actually got.
    """
    delta = beat_delta_for(period_um, beat_um)
    a = _grating_rects(cx, cy, w, h, period_um, duty, vertical=True)
    b = _grating_rects(cx, cy, w, h, period_um + delta, duty, vertical=True)
    return CellArt(
        front=_cat(a, b),
        stats={
            "period_um": period_um,
            "period2_um": round(period_um + delta, 4),
            "delta_um": round(delta, 4),
            "beat_um": beat_um,
            "amplification": round(period_um / delta, 1),
            "fringes_across": round(w / beat_um, 2),
            "duty": duty,
            "single_layer": True,
            "n_rects": int(len(a) + len(b)),
        },
    )


def build_rotation_beat(cx: float, cy: float, w: float, h: float, *,
                        period_um: float = 63.5, angle_deg: float = 2.0,
                        duty: float = 0.5) -> CellArt:
    """B-ROT — equal pitches at a small relative angle.

    ``p_beat = p / (2 sin(alpha/2))``. The fringes run roughly ALONG the grating
    lines, not across them, because ``k1 - k2`` is perpendicular to the bisector
    of the two grating vectors — which is the detail that makes rotational moiré
    look nothing like pitch moiré even though the algebra is the same.
    """
    beat = period_um / (2.0 * math.sin(math.radians(angle_deg) / 2.0))
    a = _grating_rects(cx, cy, w, h, period_um, duty, vertical=True)
    b = _rotated_grating(cx, cy, w, h, period_um, duty, angle_deg)
    art = CellArt(front=a, polys=b)
    art.stats = {
        "period_um": period_um,
        "angle_deg": angle_deg,
        "beat_um": round(beat, 1),
        "beat_arcmin": round(beat / 87.0, 2),
        "fringes_across": round(w / beat, 1),
        "single_layer": True,
        "n_rects": int(len(a)),
        "n_polys": int(len(b)),
    }
    return art


def combined_beat_um(p1: float, p2: float, angle_deg: float) -> float:
    """The general vector beat: ``|k1 - k2|`` for two gratings.

    ``p_beat = p1 p2 / sqrt(p1^2 + p2^2 - 2 p1 p2 cos alpha)``. Reduces to the
    pitch beat at alpha = 0 and to ``p/(2 sin(alpha/2))`` at p1 = p2.
    """
    a = math.radians(angle_deg)
    den = math.sqrt(p1 * p1 + p2 * p2 - 2.0 * p1 * p2 * math.cos(a))
    return float("inf") if den < 1e-12 else p1 * p2 / den


def build_vector_beat(cx: float, cy: float, w: float, h: float, *,
                      period_a_um: float = 63.5, period_b_um: float = 66.07,
                      angle_deg: float = 2.0, duty: float = 0.5) -> CellArt:
    """B-VEC — pitch AND angle together, to check the general formula.

    The repo's ``beat_period_combined_um`` already uses this; it is cheap to
    verify on glass and expensive to have wrong, because every perimeter frame
    on the box is a rotated pair rather than a parallel one.
    """
    beat = combined_beat_um(period_a_um, period_b_um, angle_deg)
    a = _grating_rects(cx, cy, w, h, period_a_um, duty, vertical=True)
    b = _rotated_grating(cx, cy, w, h, period_b_um, duty, angle_deg)
    return CellArt(
        front=a, polys=b,
        stats={
            "period_a_um": period_a_um, "period_b_um": period_b_um,
            "angle_deg": angle_deg, "beat_um": round(beat, 1),
            "fringes_across": round(w / beat, 2) if beat < 1e9 else 0.0,
            "single_layer": True,
            "n_rects": int(len(a)), "n_polys": int(len(b)),
        },
    )


def harmonic_beats(p1: float, p2: float, duty: float,
                   max_order: int = 3) -> list[dict[str, Any]]:
    """Every ``|m/p1 - n/p2|`` beat and its amplitude at a given duty.

    The amplitude of harmonic ``k`` of a duty-``c`` square wave is
    ``c*sinc(kc)``, which VANISHES for even ``k`` at exactly c = 0.5. That is
    why this table changes character the moment the process drifts.
    """
    def amp(k: int, c: float) -> float:
        x = math.pi * k * c
        return abs(c * math.sin(x) / x) if x else 0.0

    out = []
    for m in range(1, max_order + 1):
        for n in range(1, max_order + 1):
            f = abs(m / p1 - n / p2)
            if f < 1e-12:
                continue
            beat = 1.0 / f
            if beat > 60000.0:
                continue
            out.append({
                "m": m, "n": n,
                "beat_um": round(beat, 1),
                "arcmin": round(beat / 87.0, 2),
                "amplitude": round(amp(m, duty) * amp(n, duty), 5),
            })
    return sorted(out, key=lambda r: -r["beat_um"])


def build_harmonic(cx: float, cy: float, w: float, h: float, *,
                   period_a_um: float = 44.0, period_b_um: float = 63.5,
                   duty: float = 0.5) -> CellArt:
    """B-HARM — the moiré that only exists when the process is off.

    The halftone screen (44 µm) over the carrier (63.5 µm). At nominal 50% duty
    the only visible beat is the (1,1) at 143 µm = 1.65 arcmin. The (2,3) beat
    at 559 µm = 6.4 arcmin is PLAINLY visible and has amplitude exactly zero at
    50% — because it needs an even harmonic — so it appears only under duty
    bias. Sweeping duty across this cell makes a process error visible as a
    half-millimetre banding that the nominal design does not have.
    """
    a = _grating_rects(cx, cy, w, h, period_a_um, duty, vertical=True)
    b = _grating_rects(cx, cy, w, h, period_b_um, duty, vertical=True)
    tab = harmonic_beats(period_a_um, period_b_um, duty)
    visible = [t for t in tab if t["arcmin"] > 1.5 and t["amplitude"] > 2e-3]
    return CellArt(
        front=_cat(a, b),
        stats={
            "period_a_um": period_a_um, "period_b_um": period_b_um,
            "duty": duty,
            "visible_beats": visible,
            "n_visible": len(visible),
            "single_layer": True,
            "n_rects": int(len(a) + len(b)),
        },
    )


def build_screen_over_carrier(cx: float, cy: float, w: float, h: float, *,
                              screen_um: float = 44.0, carrier_um: float = 63.5,
                              screen_angle_deg: float = 0.0,
                              duty: float = 0.5) -> CellArt:
    """B-SCREEN — the halftone screen's own beat with the back carrier, by angle.

    The hazard note in the design says running the screen perpendicular to the
    carrier drives the beat from 1.65 arcmin to 0.41. This cell is that claim,
    at three angles, on glass.
    """
    beat = combined_beat_um(screen_um, carrier_um, screen_angle_deg)
    a = _grating_rects(cx, cy, w, h, carrier_um, duty, vertical=True)
    b = (_grating_rects(cx, cy, w, h, screen_um, duty, vertical=True)
         if abs(screen_angle_deg) < 1e-9
         else np.empty((0, 4)))
    polys = ([] if abs(screen_angle_deg) < 1e-9
             else _rotated_grating(cx, cy, w, h, screen_um, duty, screen_angle_deg))
    return CellArt(
        front=_cat(a, b), polys=polys,
        stats={
            "screen_um": screen_um, "carrier_um": carrier_um,
            "screen_angle_deg": screen_angle_deg,
            "beat_um": round(beat, 1), "beat_arcmin": round(beat / 87.0, 2),
            "single_layer": True,
            "n_rects": int(len(a) + len(b)), "n_polys": int(len(polys)),
        },
    )


def build_beat_contrast(cx: float, cy: float, w: float, h: float, *,
                        period_um: float = 63.5, beat_um: float = 1635.0,
                        duty: float = 0.5) -> CellArt:
    """B-CONT — the same beat at three duties.

    Union coverage runs from ``c`` (aligned) to ``2c - c^2`` (anti-aligned), so
    a low duty is BRIGHTER at similar fringe contrast. Whether that reads better
    on a lid is not a calculation.
    """
    art = build_beat(cx, cy, w, h, period_um=period_um, beat_um=beat_um, duty=duty)
    art.stats["coverage_aligned"] = round(duty, 3)
    art.stats["coverage_anti"] = round(2 * duty - duty * duty, 3)
    art.stats["contrast"] = round(
        (2 * duty - duty * duty - duty) / (2 * duty - duty * duty + duty), 3)
    return art


# --- metrology --------------------------------------------------------------


def build_cd_ladder(cx: float, cy: float, w: float, h: float, *,
                    period_um: float = 4.0, duty: float = 0.5,
                    dense: bool = True) -> CellArt:
    """M-CD — line/space pairs, in a dense array or as isolated lines.

    Both environments, because they do not print the same. Iso-dense bias is a
    real litho effect, and the colour ladder's whole premise is that period
    RATIOS survive the process — ratios are set by the writer grid and are safe,
    but the duty is not, and duty is what sets efficiency.
    """
    if dense:
        r = _grating_rects(cx, cy, w, h, period_um, duty, vertical=True)
    else:
        # Isolated: the same line width, spaced ten periods apart.
        r = _grating_rects(cx, cy, w, h, period_um * 10.0, duty / 10.0,
                           vertical=True)
    return CellArt(
        front=r,
        stats={"period_um": period_um, "line_um": round(period_um * duty, 3),
               "dense": dense, "n_rects": int(len(r))},
    )


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


def build_swatch(cx: float, cy: float, w: float, h: float, *,
                 base_period_um: float = 5.0, spread: float = 1.45,
                 n_rungs: int = 12, duty: float = 0.5,
                 polarity: str = "metal") -> CellArt:
    """D-SWATCH — a whole hue ladder side by side, as stripes.

    The instrument that replaced the colour portrait sweeps. One swatch shows
    every rung of one ladder at once, so a base period and a spread can be
    judged against each other in a 6 mm cell instead of eight 10 mm photographs.
    One array reference per rung, either polarity.
    """
    from .patterns.bitmap.colourzone import hue_ladder

    ladder = hue_ladder(n_rungs, spread)
    sw = w / n_rungs
    f = grating_array if polarity == "metal" else grating_array_inverse
    built = [f(cx - w / 2.0 + (i + 0.5) * sw, cy, sw, h, base_period_um * sc, duty)
             for i, sc in enumerate(ladder)]
    arrays = [e for e, _ in built]
    periods = [round(base_period_um * s_, 3) for s_ in ladder]
    lines = [round(p_ * duty, 3) for p_ in periods]
    art = CellArt(front=_cat(*[r for _, r in built]), arrays=arrays)
    art.stats = {"polarity": polarity, "base_period_um": base_period_um,
                 "spread": spread, "n_rungs": n_rungs, "periods_um": periods,
                 "finest_line_um": min(lines),
                 "all_printable": bool(min(lines) >= 2.0),
                 "n_rects": 0, "n_arrays": len(arrays)}
    return art


# --- two-layer, bonded ------------------------------------------------------


def build_near_field(cx: float, cy: float, w: float, h: float, *,
                     period_um: float = 44.0, beat_um: float = 2000.0,
                     duty: float = 0.5, gap_um: float = 2290.0,
                     polarity: str = "metal") -> CellArt:
    """E-NF — where does the two-layer shadow die?

    The same beat pair as B-BEAT but split across the two plies, at a pitch
    swept through the near-field boundary. A grating does not cast a sharp
    shadow across millimetres: a slit of width p/2 spreads by roughly
    ``lambda z / (n p)`` over a gap z inside glass of index n, and once that
    spread reaches p/2 the shadow is gone. ``p_min = sqrt(2 lambda z / n)`` is
    33 um at the box's 1.5 mm and 42 um at this plate's 2.29 mm quartz pair.
    This cell measures the boundary instead of assuming it.
    """
    # Incoherent white light: the eye, not the source, is the collimator, so the
    # question is how far a p/2 slit's shadow spreads by DIFFRACTION across the
    # gap inside the glass. Fresnel number N = p^2 n / (4 lambda z); N >= 1 the
    # shadow is intact, N < 0.25 it is gone. (Coherent Talbot self-imaging is
    # NOT the mechanism, and z_T/4 is where a 50% grating's shadow VANISHES,
    # which an earlier version of this cell had backwards.)
    lam = 0.55
    n_idx = 1.4585
    fresnel = period_um * period_um * n_idx / (4.0 * lam * gap_um)
    p_min = math.sqrt(2.0 * lam * gap_um / n_idx)
    delta = beat_delta_for(period_um, beat_um)
    f = grating_array if polarity == "metal" else grating_array_inverse
    fe, fr = f(cx, cy, w, h, period_um + delta, duty)
    be, br = f(cx, cy, w, h, period_um, duty)
    art = CellArt(front=fr, back=br, arrays=[fe], back_arrays=[be])
    art.stats = {
        "polarity": polarity, "period_um": period_um, "beat_um": beat_um,
        "gap_um": gap_um, "fresnel_number": round(fresnel, 3),
        "p_min_um": round(p_min, 1),
        "predicted": ("intact" if fresnel >= 1.0
                      else "degraded" if fresnel >= 0.25 else "washed out"),
        "single_layer": False, "n_rects": int(len(fr) + len(br)), "n_arrays": 2,
    }
    return art


def build_parallax_ruler(cx: float, cy: float, w: float, h: float, *,
                         comb_um: float = 60.0, n: int = 60,
                         gap_um: float = 2290.0, n_index: float = 1.4585,
                         polarity: str = "metal") -> CellArt:
    """P-RULE — read the bond gap directly, by tilting.

    A fine comb on one ply and a single index line on the other. Tilt until the
    index sits over tooth k and the shift is ``k * comb``; against
    ``shift = t tan(asin(sin(theta)/n))`` that is a direct measurement of
    ``t`` and ``n`` together — the one number every parallax cell on the plate
    depends on and which nothing else measures.

    The comb is 60 um, not the 200 that first suggested itself: at 27.4 um/deg a
    200 um tooth needs 7.3 deg of tilt, so a hand-held read would cover barely
    two teeth. 60 um gives 2.2 deg per tooth and nine teeth inside +-10 deg,
    and still sits clear of the near-field boundary (z_T = 13 mm, gap/z_T = 0.17).
    """
    cw = min(w, comb_um * n)
    comb_cy, comb_h = cy - h * 0.15, h * 0.5
    ix0, ix1 = cx - comb_um * 0.12, cx + comb_um * 0.12
    iy0, iy1 = cy + h * 0.12, cy + h * 0.45
    per_deg = gap_um * math.tan(math.asin(math.sin(math.radians(1.0)) / n_index))
    if polarity == "metal":
        be, br = grating_array(cx, comb_cy, cw, comb_h, comb_um, 0.5)
        art = CellArt(front=_rect(ix0, iy0, ix1, iy1), back=br, back_arrays=[be])
        n_r = 1 + len(br)
    else:
        front = outside_boxes(cx, cy, w, h, [(ix0, ix1, iy0, iy1)])
        back = outside_boxes(cx, cy, w, h, [(cx - cw / 2, cx + cw / 2,
                                             comb_cy - comb_h / 2, comb_cy + comb_h / 2)])
        be, br = grating_array_inverse(cx, comb_cy, cw, comb_h, comb_um, 0.5)
        art = CellArt(front=front, back=_cat(back, br), back_arrays=[be])
        n_r = len(front) + len(back) + len(br)
    art.stats = {"polarity": polarity, "comb_um": comb_um, "n_teeth": n,
                 "parallax_um_per_deg": round(per_deg, 2),
                 "deg_per_tooth": round(comb_um / per_deg, 3),
                 "single_layer": False, "n_rects": int(n_r), "n_arrays": 1}
    return art


# --- ladders as ONE cell ----------------------------------------------------


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


def build_crossed(cx: float, cy: float, w: float, h: float, *,
                  period_x_um: float = 5.0, period_y_um: float = 5.0,
                  duty: float = 0.5, polarity: str = "metal") -> CellArt:
    """D-CROSS — two orthogonal gratings on ONE layer: a 2-D diffraction lattice.

    Orders appear on a grid rather than a line, so the colour fans in two axes
    at once. It is also the cheapest possible check of the union identity that
    the whole single-layer moire programme rests on: the crossed pattern's
    transmission is the PRODUCT of the two 1-D transmissions, and if that did
    not hold on glass, every B- cell would be measuring something else.

    This cell must invert ANALYTICALLY. Its clear region is the intersection of
    two sets of gaps — a two-dimensional grid of holes — and at 5 um over 6 mm
    that is 1.44 MILLION of them. Handing it to the per-cell boolean took the
    plate build from 10 s to 64 s and would have put a million polygons in the
    file. Written as one horizontal gap band per row, each carrying the vertical
    grating's complement as an array reference, it is 1200 arrays instead.
    """
    if polarity == "metal":
        a = _grating_rects(cx, cy, w, h, period_x_um, duty, vertical=True)
        b = _grating_rects(cx, cy, w, h, period_y_um, duty, vertical=False)
        art = CellArt(front=_cat(a, b))
        n = int(len(a) + len(b))
    else:
        # One band per horizontal GAP; the vertical grating's complement then
        # rides inside it as a 1-D array, which is the machinery that already
        # exists for the colour sub-gratings.
        bands = _grating_rects(cx, cy, w, h, period_y_um, 1.0 - duty,
                               phase_um=period_y_um * duty, vertical=False)
        art = CellArt()
        if len(bands):
            art.arrays.append({
                "rects": bands,
                "period_um": np.full(len(bands), period_x_um),
                "line_um": np.full(len(bands), period_x_um * (1.0 - duty)),
                "phase_um": np.full(len(bands), period_x_um * duty),
            })
        n = int(len(bands))
    art.stats = {
        "polarity": polarity,
        "period_x_um": period_x_um, "period_y_um": period_y_um,
        "duty": duty, "open_fraction": round((1 - duty) ** 2, 3),
        "single_layer": True, "n_rects": n,
    }
    return art
