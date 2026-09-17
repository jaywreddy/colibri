"""The photo-halftone centrepiece: screen inputs, band stamping, colour bands.

The three photograph faces are single-ply colour-by-region plates — tone is the
HEIGHT of a band and colour is the PERIOD of the grating inside it — so this
module is where an image becomes rectangles. ``photo_band_rects`` and
``photo_colour_band_periods`` are the shared entry points: the preview
compositor, the fab bake (``export_fine``) and the witness dies all screen a
picture through exactly these, so a face cannot be screened two ways.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import MultiPolygon

from ..patterns.base import registry
from .recipe import CENTERPIECE_FILL
from .spec import PHOTO_SLUG, FaceKind, PlateSpec, _aperture

# --- photo-halftone centerpiece ---------------------------------------------
# A photo face's centerpiece is a LINE SCREEN, so it has no silhouette and no
# back layer: the picture is written entirely in the height of the front bands.
# Three writers need it — the composed preview PNG, the fab SVG and the fine GDS
# — and all three go through this pair, so the picture on screen and the picture
# in the mask are the same screen at the same tone model.


def _photo_screen_inputs(spec: PlateSpec) -> dict[str, Any] | None:
    """Resolved photo params + the art-box side, or None if the face has none."""
    from ..patterns.bitmap import photo as ph

    side_um = CENTERPIECE_FILL * _aperture(spec)
    if side_um <= 0:
        return None
    p = {**registry[PHOTO_SLUG].defaults(), **spec.pattern_params}
    period = float(p["line_period_um"])
    steps = ph.resolve_steps(period, int(p["tone_steps"]))
    return {
        "side_um": side_um,
        "image": str(p["image"]),
        "colour_mode": str(p["colour_mode"]),
        "fade_start": float(p["fade_start"]),
        "fade_gate": float(p["fade_gate"]),
        "line_period_um": period,
        "tone_steps": steps,
        # Sized off the ART BOX, not the plate, and identically in all three
        # writers — so the preview stamp and the two fab bakes prep the SAME
        # source array and cannot disagree about a pixel of coverage.
        "asset_px": ph.asset_px_for(side_um, period),
    }


def _photo_band_stamp(
    spec: PlateSpec, side_px: int
) -> tuple["np.ndarray", "np.ndarray"] | None:
    """``(gold, coloured)`` bool grids of ``side_px``², y-DOWN, over the art box.

    This is ``screenrects.screen_bands``' geometry evaluated ON THE PLATE RASTER
    instead of emitted as rectangles: line ``j`` occupies ``[j·P, (j+1)·P)``
    measured down from the art box's top edge, the band is CENTRED in its line,
    and its height is ``floor(coverage·steps)/steps · P`` — the screen's real
    tone ladder, not a rounding artefact. The coverage and colour fields are
    sampled through ``screenrects._sample_rows`` at the same ``(n_lines,
    n_cols)`` grid the rectangle path uses, so the preview is a rasterization of
    the fab geometry rather than a second, similar-looking screen.

    On a 29 mm face the plate raster is ~19 um, i.e. only ~2.3 px per 44 um
    line, so the preview's bands are coarse. That is accepted (the preview is
    what it is); the exact bands are ``photo_band_rects``.
    """
    import numpy as np

    from ..patterns.bitmap import screenrects as sr
    from ..patterns.bitmap import photo as ph

    inp = _photo_screen_inputs(spec)
    if inp is None or side_px <= 0:
        return None
    side_um = inp["side_um"]
    period = inp["line_period_um"]
    steps = inp["tone_steps"]

    cov, ids, periods = ph.photo_coverage(
        inp["image"],
        inp["fade_start"],
        inp["fade_gate"],
        steps,
        inp["asset_px"],
        colour_mode=inp["colour_mode"],
    )

    # The screen's own sampling grid (screen_bands): one line per period, one
    # column per tone cell, capped against the asset so we never sample finer
    # than the source has.
    n_lines = max(1, int(round(side_um / period)))
    want_cols = int(round(side_um / (period / steps)))
    n_cols = max(8, min(want_cols, cov.shape[1] * 4))
    tone = sr._sample_rows(cov, n_lines, n_cols)
    level = np.floor(np.clip(tone, 0.0, 1.0) * steps).astype(np.int32)
    if inp["colour_mode"] == "plain":
        pid = np.zeros(level.shape, dtype=np.int32)
    else:
        pid = sr._sample_rows(ids.astype(np.float32), n_lines, n_cols).astype(np.int32)

    # Art-box pixel centres, in um from the top-left of the box.
    cell = side_um / side_px
    ys = (np.arange(side_px) + 0.5) * cell
    xs = (np.arange(side_px) + 0.5) * cell
    li = np.minimum((ys / period).astype(np.int64), n_lines - 1)
    col_um = side_um / n_cols
    ci = np.minimum((xs / col_um).astype(np.int64), n_cols - 1)

    lvl = level[li[:, None], ci[None, :]]
    band_h = (lvl.astype(np.float64) / steps) * period
    line_centre = (li[:, None] + 0.5) * period
    gold = (np.abs(ys[:, None] - line_centre) <= band_h / 2.0) & (lvl > 0)

    # A band is "coloured" when its period id maps to a real sub-grating period;
    # id 0 (and any id absent from the table) stays plain gold — the same rule
    # ``screenrects.split_by_colour`` applies to the rectangles.
    lut = np.zeros(int(pid.max()) + 1 if pid.size else 1, dtype=np.float64)
    for k, v in periods.items():
        if 0 <= int(k) < lut.size:
            lut[int(k)] = float(v)
    coloured = gold & (lut[pid[li[:, None], ci[None, :]]] > 0.0)
    return gold, coloured


def _photo_halftone_art(spec: PlateSpec, *, defer_arrays: bool):
    """The face's screened halftone (``witness_cells.CellArt``), or None.

    One preparation of the picture — coverage, colour plan, band ladder —
    behind both photo consumers, so the exact rectangles the fab writes and the
    period map the renderer samples cannot be screened from two different
    coverage maps. ``defer_arrays`` picks which form the coloured bands come
    back in (see ``build_halftone_bands``): False concatenates every whole
    sub-grating stripe into ``art.front``; True leaves them as one
    ``art.arrays`` band-plus-period reference.
    """
    from .. import witness_cells as wc
    from ..patterns.bitmap import photo as ph

    inp = _photo_screen_inputs(spec)
    if inp is None:
        return None
    cov, ids, periods = ph.photo_coverage(
        inp["image"],
        inp["fade_start"],
        inp["fade_gate"],
        inp["tone_steps"],
        inp["asset_px"],
        colour_mode=inp["colour_mode"],
    )
    plan = ph.colour_plan(inp["colour_mode"], inp["image"])
    side = inp["side_um"]
    art, _report = wc.build_halftone_bands(
        0.0,
        0.0,
        side,
        side,
        coverage=cov,
        period_id=None if plan.mode == "plain" else ids,
        periods=periods,
        duty=plan.duty,
        line_period_um=inp["line_period_um"],
        tone_steps=inp["tone_steps"],
        defer_arrays=defer_arrays,
        polarity="metal",
    )
    return art


def photo_band_rects(spec: PlateSpec) -> "np.ndarray":
    """EXACT front-layer geometry of a photo face: ``(N, 4)`` ``[x0,x1,y0,y1]`` um.

    The line-screen bands plus, inside every coloured band, the vertical
    diffraction sub-grating (``screenrects.stripe_plan``'s whole stripes). Plate
    coords, origin at the plate centre, y UP — the same frame
    ``_helpers.raster_to_polygons`` and ``export_fine``'s rect sets use, so this
    drops into the fab SVG and the fine GDS without a transform.

    Front layer ONLY. A halftone's tone IS its band height; putting anything on
    the inner ply under it would show through the gaps and lift the shadows.
    """
    import numpy as np

    # defer_arrays=False: the SVG/GDS writers here want real rectangles. The
    # array-reference path (`stripe_plan`) is the witness plate's, where a
    # periodic sub-grating becomes one GDS array instead of 100k boxes.
    art = _photo_halftone_art(spec, defer_arrays=False)
    if art is None:
        return np.empty((0, 4), dtype=float)
    return np.asarray(art.front, dtype=float)


def photo_colour_band_periods(spec: PlateSpec) -> tuple["np.ndarray", "np.ndarray"]:
    """The COLOURED halftone bands and their sub-grating periods.

    ``((N,4) [x0,x1,y0,y1] µm bands, (N,) period µm)`` in plate coords, y UP —
    the same band rectangles ``photo_band_rects`` fills with whole stripes, but
    kept whole so the period behind each one is still a number rather than
    geometry. This is what ``period_front.png`` is painted from: at 2048 px the
    stripes themselves are far below a pixel, so the raster carries their
    COVERAGE while this carries the period that sets their diffracted colour.

    Empty on a plain (uncoloured) screen and on any non-photo face.
    """
    import numpy as np

    empty = (np.empty((0, 4), dtype=float), np.empty((0,), dtype=float))
    if spec.kind is not FaceKind.PHOTO:
        return empty
    # defer_arrays=True: we want the band + period reference, NOT the expanded
    # stripes (which is also why this is cheaper than photo_band_rects).
    art = _photo_halftone_art(spec, defer_arrays=True)
    if art is None or not art.arrays:
        return empty
    plan = art.arrays[0]
    rects = np.asarray(plan["rects"], dtype=float)
    periods = np.asarray(plan["period_um"], dtype=float)
    if rects.shape[0] == 0 or periods.shape[0] != rects.shape[0]:
        return empty
    return rects, periods



def _photo_multipolygon(spec: PlateSpec) -> MultiPolygon:
    """A photo face's exact front geometry as polygons (bands + sub-gratings)."""
    from ..patterns.bitmap.photo import rects_to_multipolygon

    return rects_to_multipolygon(photo_band_rects(spec))
