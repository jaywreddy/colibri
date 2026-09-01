"""Geometry for every witness-plate cell, in rect space.

One builder per effect. They all return :class:`~app.witness_geom.CellArt` and
they all work analytically — a lamellar grating is arithmetic on line indices,
not a rasterized lattice — so a 2 um grating across a 12 mm patch costs 6000
rectangles and no raster at all. That is what lets a 127 mm plate be built on a
13.7 GB host.

The halftone builder is the exception that proves the rule: it has to look at a
photograph, so it samples one at the resolution the cell actually needs
(:func:`asset_px_for`) rather than at the plate's own cell pitch.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .patterns.bitmap import colourplan as cp
from .patterns.bitmap import imageprep as ip
from .patterns.bitmap import screenrects as sr
from .patterns.bitmap.colourzone import MIN_FEATURE_UM
from .witness_geom import (
    PORTRAIT_CROP,
    REF_SCREEN_UM,
    REF_TONE_STEPS,
    SOURCE_PHOTO,
    CellArt,
    _cat,
    _grating_rects,
    _rect,
)

_SRC_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def photo_path() -> Path:
    """The source photograph, at the repo root."""
    return Path(__file__).resolve().parents[2] / SOURCE_PHOTO


def portrait_source(size: int = 1400) -> tuple[np.ndarray, np.ndarray]:
    """``(gray, rgb)`` for the reference crop, both at ``size`` square.

    Loaded through ``imageprep.load_gray`` and ``colourplan.load_rgb`` with the
    SAME crop, which is what guarantees the darkness map and the period field
    are in register — a half-pixel disagreement would colour the wrong petal.
    """
    if size not in _SRC_CACHE:
        p = photo_path()
        if not p.is_file():
            raise FileNotFoundError(f"source photo not found: {p}")
        _SRC_CACHE[size] = (
            ip.load_gray(p, crop=PORTRAIT_CROP, size=size),
            cp.load_rgb(p, crop=PORTRAIT_CROP, size=size),
        )
    return _SRC_CACHE[size]


def asset_px_for(
    extent_um: float, line_period_um: float, oversample: float = 1.25
) -> int:
    """Source resolution to prep for a cell of this size.

    Rectangle count is set by the ASSET, not by the plate (see ``screenrects``),
    so this is the real cost dial — and 1.25 px per halftone line is already
    past what anyone can see.

    The eye's integration cell is 87 um, which at the reference 44 um screen is
    exactly TWO line periods. A 30 mm cell therefore resolves 345 elements
    across however finely it is written, while an oversample of 2.0 preps 1364
    source pixels and spends a rectangle on each — four times the detail the
    viewer can resolve, at four times the file. 1.25 keeps a margin over acuity
    for the sharpening to work with and roughly halves the plate's GDS.
    """
    n_lines = max(1.0, extent_um / line_period_um)
    return int(max(160, min(2200, round(n_lines * oversample))))


# --- the headline cell ------------------------------------------------------


def build_halftone(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    plan: cp.ColourPlan,
    prep: ip.PrepSpec | None = None,
    line_period_um: float = REF_SCREEN_UM,
    tone_steps: int = REF_TONE_STEPS,
    defer_arrays: bool = True,
) -> CellArt:
    """One halftone portrait cell, plain or colour-shaded.

    The whole three-variant question reduces to which ``plan`` is passed: a
    ``plain`` plan yields an empty period field and no sub-grating at all, so
    the control cell runs the same code path rather than a different one. Any
    difference on the finished plate is the colour and nothing else.
    """
    steps = max(2, min(int(tone_steps), int(line_period_um / MIN_FEATURE_UM)))
    px = asset_px_for(min(w, h), line_period_um)
    gray, rgb = portrait_source(px)
    spec = prep or ip.PrepSpec(tone_steps=steps)
    if spec.tone_steps != steps:
        spec = ip.PrepSpec(**{**spec.__dict__, "tone_steps": steps})
    dark = ip.prep_darkness(gray, spec)

    ids, periods, field_report = cp.build_period_field(rgb, plan)
    bands, pid, band_report = sr.screen_bands(
        dark,
        extent_um=min(w, h),
        line_period_um=line_period_um,
        tone_steps=steps,
        period_id=None if plan.mode == "plain" else ids,
        origin=(cx, cy),
    )
    plain, coloured, per = sr.split_by_colour(bands, pid, periods)

    art = CellArt(front=plain)
    n_stripes = 0
    if len(coloured):
        sp = sr.stripe_plan(coloured, per, plan.duty)
        n_stripes = sp["total"]
        if defer_arrays:
            art.arrays.append(
                {
                    "rects": coloured,
                    "period_um": per,
                    "line_um": per * plan.duty,
                    "phase_um": 0.0,
                }
            )
        else:
            art.front = _cat(art.front, sr.stripe_rects(coloured, per, plan.duty))

    art.stats = {
        "mode": plan.mode,
        "extent_um": round(min(w, h), 1),
        "asset_px": px,
        "line_period_um": line_period_um,
        "tone_steps": steps,
        "finest_band_um": round(line_period_um / steps, 3),
        "n_band_rects": int(len(bands)),
        "n_plain_rects": int(len(plain)),
        "n_coloured_bands": int(len(coloured)),
        "n_stripes_if_flat": int(n_stripes),
        "frac_coloured": field_report["frac_coloured"],
        "rungs_used": field_report["rungs_used"],
        "base_period_um": plan.base_period_um,
        "hue_equalize": field_report.get("hue_equalize", False),
        "all_used_rungs_printable": field_report["all_used_rungs_printable"],
        "rects_per_line": band_report["rects_per_line"],
        # Eye cells across the picture at 300 mm; under ~150 it is a thumbnail.
        "eye_cells_across": int(min(w, h) / 87.0),
    }
    return art


# --- single-layer test structures -------------------------------------------


def build_grating_patch(
    cx: float, cy: float, w: float, h: float, *,
    period_um: float, duty: float = 0.5, angle_deg: float = 0.0,
) -> CellArt:
    """A bare lamellar patch — B1, D5, and every rung of C2 and C3."""
    vertical = abs(angle_deg) < 1e-9
    if vertical or abs(angle_deg - 90.0) < 1e-9:
        r = _grating_rects(cx, cy, w, h, period_um, duty, vertical=vertical)
    else:
        # Off-axis patches are cut on the projected axis and clipped, so the
        # writer never has to handle a rotated polygon for a test structure.
        r = _grating_rects(cx, cy, w * 1.6, h, period_um, duty, vertical=True)
        r = r[(r[:, 1] > cx - w / 2) & (r[:, 0] < cx + w / 2)]
        r[:, 0] = np.clip(r[:, 0], cx - w / 2, cx + w / 2)
        r[:, 1] = np.clip(r[:, 1], cx - w / 2, cx + w / 2)
    line = period_um * duty
    return CellArt(
        front=r,
        stats={
            "period_um": period_um,
            "duty": duty,
            "line_um": round(line, 3),
            "gap_um": round(period_um - line, 3),
            "clears_litho_floor": bool(
                line >= MIN_FEATURE_UM - 1e-9
                and period_um - line >= MIN_FEATURE_UM - 1e-9
            ),
            "n_rects": int(len(r)),
        },
    )


def build_chirp(
    cx: float, cy: float, w: float, h: float, *,
    period_start_um: float = 22.0, period_end_um: float = 3.0, duty: float = 0.5,
) -> CellArt:
    """B2 — period swept along the patch, so the fan is graded, not flat.

    Doubles as a continuous resolution check: the sweep crosses the litho floor
    somewhere, and where it stops diffracting is where the process gave out.

    The fine end is 3.0 um, not the 4.0 the accent zone uses. At 4.0 the sweep
    STOPS exactly at the floor (a 50%-duty 4 um period is a 2 um line) and never
    crosses it, so the cell could only ever confirm the assumed limit and never
    find the real one — which is the single thing it is for.
    """
    x0 = cx - w / 2.0
    x1 = cx + w / 2.0
    xs: list[float] = []
    ws: list[float] = []
    x = x0
    while x < x1:
        t = (x - x0) / w
        d = period_start_um + (period_end_um - period_start_um) * t
        xs.append(x)
        ws.append(d * duty)
        x += d
    a = np.asarray(xs, dtype=np.float64)
    b = a + np.asarray(ws, dtype=np.float64)
    keep = b <= x1
    a, b = a[keep], b[keep]
    r = np.empty((a.size, 4), dtype=np.float64)
    r[:, 0], r[:, 1] = a, b
    r[:, 2], r[:, 3] = cy - h / 2.0, cy + h / 2.0
    return CellArt(
        front=r,
        stats={
            "period_start_um": period_start_um,
            "period_end_um": period_end_um,
            "crosses_floor_at_um": round(MIN_FEATURE_UM / duty, 2),
            "n_rects": int(len(r)),
        },
    )


def build_colour_band(
    cx: float, cy: float, w: float, h: float, *,
    tone: float = 0.5, line_period_um: float = REF_SCREEN_UM,
    tone_steps: int = REF_TONE_STEPS, period_um: float = 5.0, duty: float = 0.5,
    hold_tone: bool = True,
) -> CellArt:
    """D1–D3 — a FLAT-tone halftone whose bands carry a sub-grating.

    The clean version of the colour question, with the photograph taken out of
    it: one tone, one period, so the only thing the cell can tell you is
    whether a gratinged band still holds its tone and still diffracts. Beside
    the portrait cells it separates "the colour works" from "the picture works".
    """
    steps = max(2, min(int(tone_steps), int(line_period_um / MIN_FEATURE_UM)))
    eff = min(tone / duty, 1.0) if hold_tone else tone
    band_h = (math.floor(eff * steps) / steps) * line_period_um
    n = max(1, int(h / line_period_um))
    cyc = cy + h / 2.0 - (np.arange(n) + 0.5) * line_period_um
    bands = np.empty((n, 4), dtype=np.float64)
    bands[:, 0], bands[:, 1] = cx - w / 2.0, cx + w / 2.0
    bands[:, 2], bands[:, 3] = cyc - band_h / 2.0, cyc + band_h / 2.0
    art = CellArt(front=np.empty((0, 4)))
    art.arrays.append(
        {
            "rects": bands,
            "period_um": np.full(n, period_um),
            "line_um": np.full(n, period_um * duty),
            "phase_um": 0.0,
        }
    )
    art.stats = {
        "tone": tone,
        "hold_tone": hold_tone,
        "band_um": round(band_h, 3),
        "period_um": period_um,
        "line_um": round(period_um * duty, 3),
        "periods_per_band": round(band_h / period_um, 2),
        "clears_litho_floor": bool(period_um * duty >= MIN_FEATURE_UM - 1e-9),
        "enough_periods_for_a_spectrum": bool(band_h >= 2.0 * period_um),
        "n_bands": n,
    }
    return art


def build_vernier(
    cx: float, cy: float, w: float, h: float, *,
    front_pitch_um: float = 80.0, back_pitch_um: float = 88.0, n: int = 24,
) -> CellArt:
    """C1 — two combs whose beat amplifies a real offset by p/(pb − p).

    Read this cell before anything else on the plate: it is the number that
    decides whether Group A is viable at all.
    """
    bar_h = h * 0.40
    f = _grating_rects(cx, cy + h * 0.25, min(w, front_pitch_um * n), bar_h,
                       front_pitch_um, 0.5, vertical=True)
    b = _grating_rects(cx, cy - h * 0.25, min(w, back_pitch_um * n), bar_h,
                       back_pitch_um, 0.5, vertical=True)
    return CellArt(
        front=f, back=b,
        stats={
            "front_pitch_um": front_pitch_um,
            "back_pitch_um": back_pitch_um,
            "amplification": round(front_pitch_um / abs(back_pitch_um - front_pitch_um), 2),
            "n_lines": n,
            "n_rects": int(len(f) + len(b)),
        },
    )


# --- two-layer cells (need a bonded pair) -----------------------------------


def build_barrier_switch(
    cx: float, cy: float, w: float, h: float, *, comb_um: float = 173.0
) -> CellArt:
    """A1 test panel — two interlaced lane classes under a slit comb.

    Back carries A/B bars in alternating half-comb lanes; front is the comb,
    registered a QUARTER period over so a clean image shows head-on. That phase
    is the free improvement the analysis turned up: as originally built the
    slit straddled a lane boundary at rest, so head-on showed a blend and you
    had to tilt 2.5 deg to see anything at all.
    """
    lane = comb_um / 2.0
    n = max(2, int(w / lane))
    k = np.arange(n)
    x0 = cx - w / 2.0 + k * lane
    back = np.empty((n, 4), dtype=np.float64)
    back[:, 0], back[:, 1] = x0, x0 + lane
    tall = (k % 2) == 0
    back[:, 2] = cy - h / 2.0
    back[:, 3] = np.where(tall, cy + h * 0.45, cy - h * 0.10)
    front = _grating_rects(cx, cy, w, h, comb_um, 0.5,
                           phase_um=comb_um * 0.25, vertical=True)
    return CellArt(
        front=front, back=back,
        stats={
            "comb_um": comb_um, "lane_um": lane, "peak_shift_um": comb_um / 4.0,
            "quarter_period_registered": True,
            "n_rects": int(len(front) + len(back)),
        },
    )


def build_scanimation(
    cx: float, cy: float, w: float, h: float, *, comb_um: float = 173.0, phases: int = 4
) -> CellArt:
    """A2 test panel — N-phase kinegram, N frames in 1/N-pitch lanes.

    The bare-line version, where the direction of travel is unambiguous; on the
    capybara's water it plausibly moves either way.
    """
    lane = comb_um / phases
    n = max(phases, int(w / lane))
    k = np.arange(n)
    x0 = cx - w / 2.0 + k * lane
    ph = k % phases
    back = np.empty((n, 4), dtype=np.float64)
    back[:, 0], back[:, 1] = x0, x0 + lane
    y = cy - h / 2.0 + (ph / float(phases)) * h * 0.78
    back[:, 2], back[:, 3] = y, y + h * 0.18
    front = _grating_rects(cx, cy, w, h, comb_um, 1.0 / phases, vertical=True)
    return CellArt(
        front=front, back=back,
        stats={"comb_um": comb_um, "phases": phases, "slot_um": round(lane, 2),
               "n_rects": int(len(front) + len(back))},
    )


def build_shading_moire(
    cx: float, cy: float, w: float, h: float, *,
    back_period_um: float = 63.5, beat_um: float = 1635.0, duty: float = 0.5,
) -> CellArt:
    """A4 / C6 — two near-equal pitches whose drift paints bands.

    ``beat = p(p+delta)/delta``, so the front pitch is SOLVED from the beat you
    want rather than the other way round. Pinning delta instead is what made an
    earlier jamón cell read 0.075 where it should have read 0.47.
    """
    from .patterns.effects.gratings import beat_delta_um

    delta = beat_delta_um(back_period_um, beat_um)
    front_period = back_period_um + delta
    return CellArt(
        front=_grating_rects(cx, cy, w, h, front_period, duty, vertical=True),
        back=_grating_rects(cx, cy, w, h, back_period_um, duty, vertical=True),
        stats={
            "back_period_um": back_period_um,
            "front_period_um": round(front_period, 4),
            "delta_um": round(delta, 4),
            "beat_um": beat_um,
            "bands_across": round(w / beat_um, 2),
        },
    )


def build_moire_magnifier(
    cx: float, cy: float, w: float, h: float, *,
    sampler_um: float = 60.0, motif_um: float = 62.0,
) -> CellArt:
    """A6 — a pinhole array over a slightly different motif pitch.

    Transmission is ``(1 − f)(1 − b)``, so the sampler must be GOLD WITH HOLES
    rather than sparse dots; the inverted version reads as a grey field and
    nothing floats.
    """
    mag = sampler_um / (sampler_um - motif_um)
    hole = max(MIN_FEATURE_UM * 2.0, sampler_um * 0.18)
    nx, ny = max(1, int(w / sampler_um)), max(1, int(h / sampler_um))
    gx = cx - w / 2.0 + (np.arange(nx) + 0.5) * sampler_um
    gy = cy - h / 2.0 + (np.arange(ny) + 0.5) * sampler_um
    parts: list[np.ndarray] = []
    for yy in gy:
        parts.append(_rect(cx - w / 2, yy - sampler_um / 2, cx + w / 2, yy - hole / 2))
        xs = np.concatenate(([cx - w / 2], gx + hole / 2))
        xe = np.concatenate((gx - hole / 2, [cx + w / 2]))
        seg = np.empty((xs.size, 4), dtype=np.float64)
        seg[:, 0], seg[:, 1] = xs, xe
        seg[:, 2], seg[:, 3] = yy - hole / 2, yy + hole / 2
        parts.append(seg[seg[:, 1] - seg[:, 0] > 1e-9])
        parts.append(_rect(cx - w / 2, yy + hole / 2, cx + w / 2, yy + sampler_um / 2))
    dot = max(MIN_FEATURE_UM * 3.0, motif_um * 0.35)
    mx, my = max(1, int(w / motif_um)), max(1, int(h / motif_um))
    px = cx - w / 2.0 + (np.arange(mx) + 0.5) * motif_um
    py = cy - h / 2.0 + (np.arange(my) + 0.5) * motif_um
    PX, PY = np.meshgrid(px, py)
    back = np.empty((PX.size, 4), dtype=np.float64)
    back[:, 0], back[:, 1] = PX.ravel() - dot / 2, PX.ravel() + dot / 2
    back[:, 2], back[:, 3] = PY.ravel() - dot / 2, PY.ravel() + dot / 2
    return CellArt(
        front=_cat(*parts), back=back,
        stats={"sampler_um": sampler_um, "motif_um": motif_um,
               "magnification": round(mag, 1), "hole_um": round(hole, 2),
               "n_rects": int(sum(len(p) for p in parts) + len(back))},
    )
