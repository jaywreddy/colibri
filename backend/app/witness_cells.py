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
    BOX_CARRIER_UM,
    BOX_COMB_UM,
    CLEAR,
    METAL,
    PORTRAIT_CROP,
    REF_SCREEN_UM,
    REF_TONE_STEPS,
    SOURCE_PHOTO,
    CellArt,
    _cat,
    _grating_rects,
    _rect,
    column_complement,
    grating_array,
    grating_array_inverse,
    invert_grating,
    outside_boxes,
)

_SRC_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def photo_path() -> Path:
    """The source photograph: ``photos/<SOURCE_PHOTO>`` at the repo root, or the
    root itself where it lived before the photos folder existed."""
    root = Path(__file__).resolve().parents[2]
    for cand in (root / "photos" / SOURCE_PHOTO, root / SOURCE_PHOTO):
        if cand.is_file():
            return cand
    return root / "photos" / SOURCE_PHOTO


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


def build_halftone_bands(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    coverage: np.ndarray,
    period_id: np.ndarray | None,
    periods: dict[int, float],
    duty: float = 0.5,
    line_period_um: float = REF_SCREEN_UM,
    tone_steps: int = REF_TONE_STEPS,
    defer_arrays: bool = True,
    polarity: str = METAL,
) -> tuple[CellArt, dict[str, Any]]:
    """The line screen itself: coverage + a period field -> band rectangles.

    Split out of :func:`build_halftone` so a caller with its OWN prepared image
    can reuse the band logic instead of copying it. The witness plate's portrait
    cells go through ``build_halftone`` (which loads the fixed ``SOURCE_PHOTO``
    and preps it here); the box's ``photo-halftone`` faces go through
    ``plates.photo_band_rects``, which prepares an arbitrary asset — with its own
    edge fade, which is why it hands in a finished COVERAGE map rather than a
    photograph. Both then land in this one function, so the plate and the box
    screen a picture identically.

    ``period_id`` is ``None`` for a plain screen (solid gold bands, no
    sub-grating). Returns ``(art, report)``; the report carries the band counts
    the callers fold into their own stats.
    """
    steps = max(2, min(int(tone_steps), int(line_period_um / MIN_FEATURE_UM)))
    bands, pid, band_report = sr.screen_bands(
        coverage,
        extent_um=min(w, h),
        line_period_um=line_period_um,
        tone_steps=steps,
        period_id=period_id,
        origin=(cx, cy),
        emit=polarity,
    )
    plain, coloured, per = sr.split_by_colour(bands, pid, periods)

    # In CLEAR polarity the sub-grating inverts too: the clear stripes are the
    # gaps of the metal ones, which is duty 1-c at phase c*d. The phase is
    # per-rectangle because d differs from rung to rung.
    sub_duty = duty if polarity == METAL else 1.0 - duty
    sub_phase = 0.0 if polarity == METAL else per * duty

    art = CellArt(front=plain)
    n_stripes = 0
    if len(coloured):
        sp = sr.stripe_plan(coloured, per, sub_duty, phase_um=sub_phase)
        n_stripes = sp["total"]
        if defer_arrays:
            art.arrays.append(
                {
                    "rects": coloured,
                    "period_um": per,
                    "line_um": per * sub_duty,
                    "phase_um": sub_phase,
                }
            )
        else:
            # Real rectangles: band ends held to the 2 um floor, and the CLEAR
            # side built as the exact complement of the METAL one (so the two
            # tile every band; see stripe_rects_floored).
            art.front = _cat(art.front, sr.stripe_rects_floored(
                coloured, per, duty, metal=(polarity == METAL)))

    report = {
        "tone_steps": steps,
        "finest_band_um": round(line_period_um / steps, 3),
        "n_band_rects": int(len(bands)),
        "n_plain_rects": int(len(plain)),
        "n_coloured_bands": int(len(coloured)),
        "n_stripes_if_flat": int(n_stripes),
        "rects_per_line": band_report["rects_per_line"],
    }
    return art, report


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
    polarity: str = METAL,
) -> CellArt:
    """One halftone portrait cell of the reference photo, plain or colour-shaded.

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
    art, report = build_halftone_bands(
        cx,
        cy,
        w,
        h,
        coverage=dark,
        period_id=None if plan.mode == "plain" else ids,
        periods=periods,
        duty=plan.duty,
        line_period_um=line_period_um,
        tone_steps=steps,
        defer_arrays=defer_arrays,
        polarity=polarity,
    )

    art.stats = {
        "polarity": polarity,
        "mode": plan.mode,
        "extent_um": round(min(w, h), 1),
        "asset_px": px,
        "line_period_um": line_period_um,
        "tone_steps": report["tone_steps"],
        "finest_band_um": report["finest_band_um"],
        "n_band_rects": report["n_band_rects"],
        "n_plain_rects": report["n_plain_rects"],
        "n_coloured_bands": report["n_coloured_bands"],
        "n_stripes_if_flat": report["n_stripes_if_flat"],
        "frac_coloured": field_report["frac_coloured"],
        "rungs_used": field_report["rungs_used"],
        "base_period_um": plan.base_period_um,
        "hue_equalize": field_report.get("hue_equalize", False),
        "all_used_rungs_printable": field_report["all_used_rungs_printable"],
        "rects_per_line": report["rects_per_line"],
        # Eye cells across the picture at 300 mm; under ~150 it is a thumbnail.
        "eye_cells_across": int(min(w, h) / 87.0),
    }
    return art


# --- single-layer test structures -------------------------------------------


def build_grating_patch(
    cx: float, cy: float, w: float, h: float, *,
    period_um: float, duty: float = 0.5, angle_deg: float = 0.0,
    polarity: str = METAL,
) -> CellArt:
    """A bare lamellar patch — the rungs of every period and duty ladder."""
    vertical = abs(angle_deg) < 1e-9
    emit_duty = duty if polarity == METAL else 1.0 - duty
    phase = 0.0 if polarity == METAL else period_um * duty
    if vertical or abs(angle_deg - 90.0) < 1e-9:
        r = _grating_rects(cx, cy, w, h, period_um, emit_duty,
                           phase_um=phase, vertical=vertical)
    else:
        # Off-axis patches are cut on the projected axis and clipped, so the
        # writer never has to handle a rotated polygon for a test structure.
        r = _grating_rects(cx, cy, w * 1.6, h, period_um, emit_duty,
                           phase_um=phase, vertical=True)
        r = r[(r[:, 1] > cx - w / 2) & (r[:, 0] < cx + w / 2)]
        r[:, 0] = np.clip(r[:, 0], cx - w / 2, cx + w / 2)
        r[:, 1] = np.clip(r[:, 1], cx - w / 2, cx + w / 2)
    line = period_um * duty
    return CellArt(
        front=r,
        stats={
            "polarity": polarity,
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
    polarity: str = METAL,
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
    if polarity == METAL:
        r = np.empty((a.size, 4), dtype=np.float64)
        r[:, 0], r[:, 1] = a, b
    else:
        # the gaps: after each line up to the next, plus the two ends
        g0 = np.concatenate(([x0], b))
        g1 = np.concatenate((a, [x1]))
        keep = g1 - g0 > 1e-9
        r = np.empty((int(keep.sum()), 4), dtype=np.float64)
        r[:, 0], r[:, 1] = g0[keep], g1[keep]
    r[:, 2], r[:, 3] = cy - h / 2.0, cy + h / 2.0
    return CellArt(
        front=r,
        stats={
            "polarity": polarity,
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
    polarity: str = METAL,
) -> CellArt:
    """C1 — two combs whose beat amplifies a real offset by p/(pb − p).

    Read this cell before anything else on the plate: it is the number that
    decides whether Group A is viable at all.
    """
    bar_h = h * 0.40
    fw, bw = min(w, front_pitch_um * n), min(w, back_pitch_um * n)
    fcy, bcy = cy + h * 0.25, cy - h * 0.25
    f = grating_array if polarity == METAL else grating_array_inverse
    fe, fr = f(cx, fcy, fw, bar_h, front_pitch_um, 0.5)
    be, br = f(cx, bcy, bw, bar_h, back_pitch_um, 0.5)
    art = CellArt(front=fr, back=br, arrays=[fe], back_arrays=[be])
    if polarity != METAL:
        art.front = _cat(art.front, outside_boxes(
            cx, cy, w, h, [(cx - fw / 2, cx + fw / 2, fcy - bar_h / 2, fcy + bar_h / 2)]))
        art.back = _cat(art.back, outside_boxes(
            cx, cy, w, h, [(cx - bw / 2, cx + bw / 2, bcy - bar_h / 2, bcy + bar_h / 2)]))
    art.stats = {
        "polarity": polarity,
        "front_pitch_um": front_pitch_um, "back_pitch_um": back_pitch_um,
        "amplification": round(front_pitch_um / abs(back_pitch_um - front_pitch_um), 2),
        "n_lines": n, "n_rects": int(len(art.front) + len(art.back)), "n_arrays": 2,
    }
    return art


# --- two-layer cells (need a bonded pair) -----------------------------------


def build_barrier_switch(
    cx: float, cy: float, w: float, h: float, *, comb_um: float = BOX_COMB_UM,
    polarity: str = METAL,
) -> CellArt:
    """P-SWAP test panel — two interlaced lane classes under a slit comb.

    Back carries A/B bars in alternating half-comb lanes; front is the comb.
    The comb is anchored to the CELL's left edge, where the lanes start — not to
    the plate origin. Anchored to the origin, head-on registration was
    ``x0 mod p``, which happened to be zero for three combs and 65 um for the
    shipping 173 um one, so that cell alone came out 25/75 head-on and swapped
    at 0.79 and 2.37 deg instead of a symmetric pair. Found by a reviewer
    recomputing the cell, not by any test; the ``head_on_A_fraction`` stat now
    exists so the page prints it.

    Registration is the box's straddle convention (CLAUDE.md): the slit sits on
    a lane boundary head-on, so head-on is a 50/50 blend and the clean images
    are at a back shift of +-p/4 — one image per tilt sign.
    """
    lane = comb_um / 2.0
    n = max(2, int(w / lane))
    k = np.arange(n)
    x_left = cx - w / 2.0
    x0 = x_left + k * lane
    back = np.empty((n, 4), dtype=np.float64)
    back[:, 0], back[:, 1] = x0, x0 + lane
    tall = (k % 2) == 0
    back[:, 2] = cy - h / 2.0
    back[:, 3] = np.where(tall, cy + h * 0.45, cy - h * 0.10)
    # metal lines of the comb start p/4 past a lane boundary, so the slit
    # (their complement) is centred ON the boundary: a 50/50 blend head-on
    phase = (x_left % comb_um) + comb_um * 0.25
    # fraction of the head-on slit that looks at lane class A: the slit is
    # [u - p/4, u + p/4] with u its centre inside the A|B period, A = [0, p/2)
    u = (phase + 0.75 * comb_um - x_left) % comb_um
    lo, hi = u - comb_um / 4.0, u + comb_um / 4.0
    over_a = (max(0.0, min(hi, lane) - max(lo, 0.0))
              + max(0.0, min(hi, comb_um + lane) - max(lo, comb_um))   # next period's A
              + max(0.0, min(hi, 0.0) - max(lo, -lane)) * 0.0)          # previous period is B
    head_on_a = over_a / lane
    f = grating_array if polarity == METAL else grating_array_inverse
    fe, fr = f(cx, cy, w, h, comb_um, 0.5, phase_um=phase)
    art = CellArt(front=fr, arrays=[fe])
    if polarity == METAL:
        art.back = back
    else:
        rem = _rect(x0[-1] + lane, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0) \
            if x0[-1] + lane < cx + w / 2.0 - 1e-9 else np.empty((0, 4))
        art.back = _cat(column_complement(back, cy - h / 2.0, cy + h / 2.0), rem)
    art.stats = {
        "polarity": polarity,
        "comb_um": comb_um, "lane_um": lane, "peak_shift_um": comb_um / 4.0,
        "head_on_A_fraction": round(float(head_on_a), 3),
        "registration": "straddle: blend head-on, clean images at +-p/4",
        "n_rects": int(len(art.back)), "n_arrays": 1,
    }
    return art


def build_scanimation(
    cx: float, cy: float, w: float, h: float, *, comb_um: float = BOX_COMB_UM,
    phases: int = 4, polarity: str = METAL,
) -> CellArt:
    """P-SCAN test panel — N-phase kinegram, N frames in 1/N-pitch lanes.

    The bare-line version, where the direction of travel is unambiguous. The
    comb is anchored to the cell's left edge like the lanes (see
    :func:`build_barrier_switch` for what origin-anchoring did).
    """
    lane = comb_um / phases
    n = max(phases, int(w / lane))
    k = np.arange(n)
    x_left = cx - w / 2.0
    x0 = x_left + k * lane
    ph = k % phases
    back = np.empty((n, 4), dtype=np.float64)
    back[:, 0], back[:, 1] = x0, x0 + lane
    y = cy - h / 2.0 + (ph / float(phases)) * h * 0.78
    back[:, 2], back[:, 3] = y, y + h * 0.18
    f = grating_array if polarity == METAL else grating_array_inverse
    fe, fr = f(cx, cy, w, h, comb_um, 1.0 / phases, phase_um=(x_left % comb_um))
    art = CellArt(front=fr, arrays=[fe])
    if polarity == METAL:
        art.back = back
    else:
        rem = _rect(x0[-1] + lane, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0) \
            if x0[-1] + lane < cx + w / 2.0 - 1e-9 else np.empty((0, 4))
        art.back = _cat(column_complement(back, cy - h / 2.0, cy + h / 2.0), rem)
    art.stats = {"polarity": polarity, "comb_um": comb_um, "phases": phases,
                 "slot_um": round(lane, 2), "n_rects": int(len(art.back)),
                 "n_arrays": 1}
    return art


def build_shading_moire(
    cx: float, cy: float, w: float, h: float, *,
    back_period_um: float = BOX_CARRIER_UM, beat_um: float = 1635.0, duty: float = 0.5,
    polarity: str = METAL,
) -> CellArt:
    """A4 / B-MOVE — two near-equal pitches on TWO plies whose drift paints bands.

    ``beat = p(p+delta)/delta``, so the front pitch is SOLVED from the beat you
    want rather than the other way round. Pinning delta instead is what made an
    earlier jamón cell read 0.075 where it should have read 0.47.
    """
    from .patterns.effects.gratings import beat_delta_um

    delta = beat_delta_um(back_period_um, beat_um)
    front_period = back_period_um + delta
    f = grating_array if polarity == METAL else grating_array_inverse
    fe, fr = f(cx, cy, w, h, front_period, duty)
    be, br = f(cx, cy, w, h, back_period_um, duty)
    art = CellArt(front=fr, back=br, arrays=[fe], back_arrays=[be])
    art.stats = {
        "polarity": polarity,
        "back_period_um": back_period_um,
        "front_period_um": round(front_period, 4),
        "delta_um": round(delta, 4), "beat_um": beat_um,
        "bands_across": round(w / beat_um, 2), "n_rects": 0, "n_arrays": 2,
    }
    return art


def build_moire_magnifier(
    cx: float, cy: float, w: float, h: float, *,
    sampler_um: float = 60.0, motif_um: float = 62.0, polarity: str = METAL,
) -> CellArt:
    """A6 — a pinhole array over a slightly different motif pitch.

    Transmission is ``(1 − f)(1 − b)``, so the sampler must be CHROME WITH HOLES
    rather than sparse dots; the inverted version reads as a grey field and
    nothing floats. ``M = p_s/(p_s − p_m)`` is negative here (motif coarser than
    sampler), so the magnified image is inverted.

    Both plies are 2-D lattices, so both are written as one array per row: the
    sampler's clear complement is its pinhole grid, and the dot field's clear
    complement is the gaps between dots plus the strips between rows. As
    booleans these two cells alone were 55,000 polygons.
    """
    mag = sampler_um / (sampler_um - motif_um)     # signed: negative = inverted
    hole = max(MIN_FEATURE_UM * 2.0, sampler_um * 0.18)
    dot = max(MIN_FEATURE_UM * 3.0, motif_um * 0.35)
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    ny = max(1, int(h / sampler_um))
    gy = y1 - (np.arange(ny) + 0.5) * sampler_um
    my = max(1, int(h / motif_um))
    py = y1 - (np.arange(my) + 0.5) * motif_um

    def rows(ys, size, pitch, line, phase):
        """One array entry per row band of height ``size``."""
        r = np.empty((len(ys), 4), dtype=np.float64)
        r[:, 0], r[:, 1] = x0, x1
        r[:, 2], r[:, 3] = ys - size / 2.0, ys + size / 2.0
        return {"rects": r, "period_um": np.full(len(ys), pitch),
                "line_um": np.full(len(ys), line), "phase_um": np.full(len(ys), phase)}

    def between(ys, size):
        """Full-width strips between row bands, and above/below the first/last."""
        edges_top = np.concatenate(([y1], ys - size / 2.0))
        edges_bot = np.concatenate((ys + size / 2.0, [y0]))
        r = np.empty((len(ys) + 1, 4), dtype=np.float64)
        r[:, 0], r[:, 1] = x0, x1
        r[:, 2], r[:, 3] = edges_bot, edges_top
        return r[r[:, 3] - r[:, 2] > 1e-9]

    s_phase = x0 + 0.5 * sampler_um - hole / 2.0     # first hole's left edge
    m_phase = x0 + 0.5 * motif_um - dot / 2.0         # first dot's left edge
    if polarity == METAL:
        # front: opaque field with holes = strips between rows + per-row segments
        # between holes (an array of the GAPS between holes, i.e. line = pitch - hole)
        front_arr = rows(gy, hole, sampler_um, sampler_um - hole, s_phase + hole)
        art = CellArt(front=between(gy, hole), arrays=[front_arr],
                      back_arrays=[rows(py, dot, motif_um, dot, m_phase)])
    else:
        # clear: the holes themselves; and the dot field's complement
        art = CellArt(arrays=[rows(gy, hole, sampler_um, hole, s_phase)],
                      back=between(py, dot),
                      back_arrays=[rows(py, dot, motif_um, motif_um - dot, m_phase + dot)])
    art.stats = {"polarity": polarity, "sampler_um": sampler_um, "motif_um": motif_um,
                 "magnification": round(mag, 1), "hole_um": round(hole, 2),
                 "n_rects": int(len(art.front) + len(art.back)), "n_arrays": 2}
    return art


