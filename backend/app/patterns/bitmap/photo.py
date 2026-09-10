"""A prepared PHOTOGRAPH as a gold line screen, faded into the carrier field.

``halftone`` screens an arbitrary bitmap; this module screens one of the SIX
prepared side-plate photographs in ``app/assets/photos`` and does the two extra
things a picture on a box side needs:

EDGE FADE. A photograph with a hard rectangular border reads as a sticker. On a
single-ply side the garland's carrier fills the window outside the art box at
50 % coverage, so the picture should dissolve INTO that field rather than stop
at an edge. The fade is therefore done in COVERAGE space, after the tone prep:
the prep (``imageprep.prep_darkness``) is image-RELATIVE — it normalises to the
image's own percentiles — so a "target grey" does not exist at source level and
fading toward one there would land somewhere different for every photo.

Two stages, because detail and level must die at different rates (a picture
whose contrast is faded uniformly reads as a blur, not as a dissolve):

  stage 1  DETAIL — cross-fade coverage to its own 8 %-of-width blur over the
           edge distance d in [s, s + 0.30].
  stage 2  LEVEL  — cross-fade to the 0.5 carrier field over d in
           [s + 0.25, gate]; hard 0.5 past the gate.

``d`` is a rounded-square edge metric (0.7·Chebyshev + 0.3·Euclidean) so the
fade follows the art box's shape rather than a circle inscribed in it, and the
per-pixel start ``s`` is pushed OUTWARD where the people are (the ``.subject``
rembg matte), so faces survive further into the margin than the ground does.
One photo (``night-group``) was ground-built in the fade study with its own
authored field and ships a ``.fade.png``; where that exists it replaces both
stages verbatim.

POLARITY. The pipeline's convention is the one thing to get right here:
``prep_darkness`` returns LIGHT-source → HIGH value, and the screen writes gold
proportional to that value. So a bright sky is dense gold and a shadow is bare
glass, which is what a gold-on-glass positive wants.

COLOUR is a period field (see ``colourplan``): ``plain`` writes solid gold
bands, ``hue`` and ``faces`` take a rung per pixel hue (``faces`` coarsens
harder and demands more saturation/value, so on a group photo only the skin and
clothing take colour and the sky and sea stay gold). Colour DIES where the fade
weight drops below a half — a coloured band drifting out into the carrier field
would advertise the dissolve instead of hiding it.

The tone model is :func:`photo_coverage`, and it is shared: the plate
compositor's preview stamp, the fab SVG bake and the fine GDS bake all call it,
so preview and mask cannot disagree about a single pixel of coverage.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon
from PIL import Image

from .._helpers import empty_layer
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from . import colourplan as cp
from . import imageprep as ip
from . import screenrects as sr
from .colourzone import MIN_FEATURE_UM

# backend/app/patterns/bitmap/photo.py -> backend/app/assets/photos
PHOTOS_DIR = Path(__file__).resolve().parents[2] / "assets" / "photos"

SUBJECT_SUFFIX = ".subject.png"
FADE_SUFFIX = ".fade.png"

# The coverage a single-ply garland's carrier holds outside the art box, and
# therefore the level the picture's edge must dissolve INTO. 0.5 is the carrier
# duty (plates.GRATING_DUTY); a fade toward anything else would leave a visible
# step where the art box ends.
CARRIER_COV = 0.5

# The reference screen (witness_geom.REF_SCREEN_UM / REF_TONE_STEPS): 44 um
# carries 22 grey levels at a 2.0 um finest band — right on the litho floor —
# and still subtends only 0.50 arcmin at 300 mm, so the lines stay invisible.
DEFAULT_LINE_PERIOD_UM = 44.0
DEFAULT_TONE_STEPS = 22
DEFAULT_PHOTO = "beach"


def available_photos() -> list[str]:
    """Sorted stems of the prepared photographs (call-time lookup).

    ``<name>.subject.png`` and ``<name>.fade.png`` are SIDECARS of a photo, not
    photos, so they are filtered out — otherwise the param would offer
    "beach.subject" as a picture and screen a matte.
    """
    if not PHOTOS_DIR.is_dir():
        return []
    out: list[str] = []
    for p in sorted(PHOTOS_DIR.glob("*.png")):
        stem = p.name[: -len(".png")]
        if stem.endswith(".subject") or stem.endswith(".fade"):
            continue
        out.append(stem)
    return out


_PHOTO_CHOICES = available_photos()
_DEFAULT_IMAGE = (
    DEFAULT_PHOTO
    if DEFAULT_PHOTO in _PHOTO_CHOICES
    else (_PHOTO_CHOICES[0] if _PHOTO_CHOICES else DEFAULT_PHOTO)
)

COLOUR_MODES = ["plain", "zones", "hue", "faces"]


def photo_path(image: str) -> Path:
    p = PHOTOS_DIR / f"{image}.png"
    if not p.is_file():
        raise ValueError(
            f"photo {image!r} not found under {PHOTOS_DIR} "
            f"(available: {', '.join(available_photos()) or 'none'})"
        )
    return p


def resolve_steps(line_period_um: float, tone_steps: int) -> int:
    """Grey levels this screen can hold, clamped to the litho floor.

    The finest band a line can carry is ``period / steps`` and it must clear the
    2 um minimum, so tone depth is a property of how coarse a screen you are
    willing to use — same clamp ``halftone._resolve_steps`` and
    ``witness_cells.build_halftone`` make, for the same reason.
    """
    ceiling = max(2, int(line_period_um / MIN_FEATURE_UM))
    return max(2, min(int(tone_steps), ceiling))


def asset_px_for(
    extent_um: float, line_period_um: float, oversample: float = 1.25
) -> int:
    """Source resolution to prep for a picture of this size.

    Rectangle count is set by the ASSET, not by the plate (see the
    ``screenrects`` module docstring), so this is the real cost dial. 1.25 px
    per halftone line is already past what the eye can resolve — the 87 um
    integration cell spans two 44 um lines — and it keeps a margin for the
    prep's sharpening to work with. Same rule as ``witness_cells.asset_px_for``;
    the preview stamp and both fab bakes call it with the same extent so they
    prep the SAME array and cannot drift.
    """
    n_lines = max(1.0, extent_um / max(1e-6, line_period_um))
    return int(max(160, min(1400, round(n_lines * oversample))))


# --- fade primitives (ported from tools/dev/render_side_plate.py) -----------
# Ported, not imported: tools/dev is a scratch area outside the package and the
# fab path may not depend on it. Keep the two in step if either moves.


def _box1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """One box-blur pass along ``axis``, reflect-padded, via a running sum."""
    if r < 1:
        return a
    a = np.moveaxis(a, axis, 0)
    pad = np.concatenate([a[r:0:-1], a, a[-2 : -r - 2 : -1]], axis=0)
    c = np.cumsum(pad, axis=0, dtype=np.float32)
    c = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    out = (c[2 * r + 1 :] - c[: -2 * r - 1]) / float(2 * r + 1)
    return np.moveaxis(out[: a.shape[0]], 0, axis)


def blur(a: np.ndarray, r: float, passes: int = 3) -> np.ndarray:
    """Three box passes ≈ a Gaussian, at a fraction of the cost."""
    out = np.asarray(a, np.float32)
    rb = max(1, int(round(r * 0.85)))
    for _ in range(passes):
        out = _box1d(_box1d(out, rb, 0), rb, 1)
    return out


def smootherstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Ken Perlin's quintic — C2 at both ends, so the fade has no visible seam."""
    t = np.clip((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def box_edge(n: int) -> np.ndarray:
    """0 at the centre .. 1 at the art-box border, ROUNDED-SQUARE metric.

    Pure Chebyshev (``max(|u|,|v|)``) fades along square contours and puts a
    visible crease down each diagonal; pure Euclidean fades along a circle and
    reaches the border only at the edge midpoints, leaving the corners hard.
    The 0.7/0.3 blend is a square with rounded corners — the shape the art box
    actually has to the eye.
    """
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    u = np.abs((x + 0.5) / n - 0.5) * 2
    v = np.abs((y + 0.5) / n - 0.5) * 2
    return 0.7 * np.maximum(u, v) + 0.3 * np.hypot(u, v) / math.sqrt(2)


def context_fade(
    cov: np.ndarray, subj: np.ndarray, start: float, gate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage dissolve of a COVERAGE map into the carrier field.

    Returns ``(coverage, weight)`` where ``weight`` is 1 where the picture is
    intact and 0 where it has become carrier — the colour field is killed off
    it, so a coloured band never survives into the margin.
    """
    n = cov.shape[0]
    d = box_edge(n)
    # The matte is a soft 0..1; 0.75 keeps the push short of the gate even where
    # it saturates, so a subject standing at the border cannot defeat the fade.
    hold = np.clip(0.75 * subj, 0, 1)
    s = start + (gate - start - 0.06) * hold
    detail = smootherstep(0.0, 1.0, np.clip((d - s) / 0.30, 0, 1))
    level = smootherstep(
        0.0, 1.0, np.clip((d - (s + 0.25)) / np.maximum(1e-3, gate - (s + 0.25)), 0, 1)
    )
    cov_blur = blur(cov, 0.08 * n)
    cov1 = cov * (1 - detail) + cov_blur * detail
    out = cov1 * (1 - level) + CARRIER_COV * level
    out = np.where(d > gate, CARRIER_COV, out)
    return out.astype(np.float32), (1 - level).astype(np.float32)


def colour_plan(mode: str) -> cp.ColourPlan:
    """The ``ColourPlan`` for a ``colour_mode`` choice.

    ``zones`` is deliberately the same plan as ``faces`` for now: authored
    per-photo rule lists (the sunset shirt, the garden dress — see
    ``tools/dev/render_side_plate.py::RECIPES``) are a per-photo authoring job,
    and shipping the mode with a hue fallback keeps the param honest until they
    land rather than silently rendering plain gold.
    """
    if mode == "plain":
        return cp.ColourPlan(name="photo-plain", mode="plain")
    if mode == "hue":
        return cp.ColourPlan(name="photo-hue", mode="hue", coarsen_px=7)
    # "faces" (and, for now, "zones"): coarsen harder and refuse to read hue off
    # low-saturation / low-value pixels, so a group photo colours the people and
    # leaves sky, sea and pavement as plain gold.
    return cp.ColourPlan(
        name=f"photo-{mode}",
        mode="hue",
        coarsen_px=14,
        hue_min_sat=0.28,
        hue_min_value=0.22,
    )


def _load_field(path: Path, n: int) -> np.ndarray:
    im = Image.open(path).convert("L").resize((n, n), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


# The tone model is called once per face by the preview compositor, once by the
# SVG bake and once by the fine bake, all with the same arguments — and it costs
# a percentile pass, three box blurs and a 12-rung mode filter. Memo it.
_COVERAGE_CACHE: dict[tuple, tuple] = {}
_COVERAGE_CACHE_MAX = 8


def photo_coverage(
    image: str,
    fade_start: float = 0.50,
    fade_gate: float = 0.94,
    tone_steps: int = DEFAULT_TONE_STEPS,
    n_px: int = 700,
    *,
    colour_mode: str = "plain",
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """THE tone model — ``(coverage, period_ids, periods)`` for one photo.

    ``coverage`` is float32 in [0, 1] at ``n_px`` square: the gold AREA fraction
    each pixel asks the line screen for, already faded into the carrier field at
    the edges. ``period_ids`` is the int32 colour field (0 = plain gold) at the
    same resolution and in register with it, and ``periods`` maps an id to its
    sub-grating period in um.

    Every consumer — the composed preview stamp, the fab SVG bake and the fine
    GDS bake — reads exactly this, so the picture on screen and the picture in
    the mask are the same numbers.
    """
    steps = max(2, int(tone_steps))
    n = max(64, int(n_px))
    key = (
        str(image),
        round(float(fade_start), 6),
        round(float(fade_gate), 6),
        steps,
        n,
        str(colour_mode),
    )
    hit = _COVERAGE_CACHE.get(key)
    if hit is not None:
        return hit

    src = photo_path(image)
    # Same loader pair (and therefore the same crop/resize) that
    # witness_cells.portrait_source uses — a half-pixel disagreement between the
    # darkness map and the colour field would colour the wrong petal.
    gray = ip.load_gray(src, size=n)
    rgb = cp.load_rgb(src, size=n)

    # LIGHT source -> HIGH coverage -> gold. prep_darkness linearises, fits the
    # printable window and dithers; it does NOT invert, so bright stays dense.
    cov = ip.prep_darkness(gray, ip.PrepSpec(tone_steps=steps))

    fade_png = PHOTOS_DIR / f"{image}{FADE_SUFFIX}"
    if fade_png.is_file():
        # An authored ground (night-group): its fade field IS the dissolve, and
        # re-deriving one from the edge metric would fight the matting the
        # ground was built with.
        fade = _load_field(fade_png, n)
        cov = cov + (1.0 - fade) * (CARRIER_COV - cov)
        weight = fade
    else:
        subj_png = PHOTOS_DIR / f"{image}{SUBJECT_SUFFIX}"
        subj = (
            _load_field(subj_png, n)
            if subj_png.is_file()
            else np.zeros((n, n), dtype=np.float32)
        )
        cov, weight = context_fade(cov, subj, float(fade_start), float(fade_gate))

    plan = colour_plan(str(colour_mode))
    ids, periods, _report = cp.build_period_field(rgb, plan)
    # Colour dies with the picture: past the halfway point of the dissolve the
    # bands are most of the way to the carrier field, and a sub-grating there
    # would put a coloured fringe in what is meant to read as plain gold.
    ids = np.where(weight < 0.5, 0, ids).astype(np.int32)

    result = (cov.astype(np.float32), ids, periods)
    if len(_COVERAGE_CACHE) >= _COVERAGE_CACHE_MAX:
        _COVERAGE_CACHE.pop(next(iter(_COVERAGE_CACHE)))
    _COVERAGE_CACHE[key] = result
    return result


def rects_to_multipolygon(rects: np.ndarray) -> MultiPolygon:
    """``(N, 4)`` ``[x0, x1, y0, y1]`` um rectangles -> a MultiPolygon.

    Concatenation, not union — same contract as ``_helpers.raster_to_polygons``:
    the consumers are fill-only (rasterize / to_svg) and a GEOS union over tens
    of thousands of bands is exactly what this pipeline exists to avoid.
    """
    r = np.asarray(rects, dtype=np.float64)
    if r.size == 0:
        return MultiPolygon()
    verts = np.empty((r.shape[0], 4, 2), dtype=np.float64)
    verts[:, 0, 0], verts[:, 0, 1] = r[:, 0], r[:, 2]
    verts[:, 1, 0], verts[:, 1, 1] = r[:, 1], r[:, 2]
    verts[:, 2, 0], verts[:, 2, 1] = r[:, 1], r[:, 3]
    verts[:, 3, 0], verts[:, 3, 1] = r[:, 0], r[:, 3]
    return MultiPolygon(list(shapely.polygons(verts)))


@register
class PhotoHalftone(Pattern):
    slug = "photo-halftone"
    name = "Photograph (line-screen halftone)"
    description = (
        "One of the couple's photographs written as a gold line screen: each "
        "44 um line carries a band whose height tracks the local brightness, "
        "so the picture reads as a continuous-tone gold image. The edges "
        "dissolve into the surrounding carrier field in two stages — detail "
        "first, then level — so the picture has no border, and an optional "
        "colour field gives chosen regions a diffraction sub-grating so they "
        "flash their own hue as the box turns."
    )
    tags = ["photo", "halftone", "portrait", "colour", "side"]
    tier = 1
    theme = "Global Travel"
    # A composed box face; the plate manifest forces foliage_moire anyway
    # (CLAUDE.md renderer honesty), and declaring it here keeps the standalone
    # Pattern Lab view on the same recipe the box uses.
    render_recipe = "foliage_moire"
    params = [
        ParamSpec(
            "image",
            "Photograph",
            "choice",
            _DEFAULT_IMAGE,
            choices=_PHOTO_CHOICES,
        ),
        ParamSpec(
            "colour_mode",
            "Colour",
            "choice",
            "plain",
            choices=COLOUR_MODES,
        ),
        # Where the dissolve BEGINS, as a fraction of the rounded-square edge
        # distance. Lower = the picture is smaller and floats further inside the
        # art box; higher = it runs closer to the garland.
        ParamSpec("fade_start", "Fade start", "float", 0.50, 0.30, 0.90, 0.01),
        # Where it is FINISHED — past the gate the art box is pure carrier.
        ParamSpec("fade_gate", "Fade gate", "float", 0.94, 0.80, 0.98, 0.01),
        ParamSpec("line_period_um", "Line period", "float", 44.0, 20.0, 60.0, 1.0, "μm"),
        # Grey levels. Clamped to line_period / 2 um so the finest band never
        # drops under the litho floor.
        ParamSpec("tone_steps", "Tone steps", "int", 22, 2, 22, 1),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    # --- cheap metadata (no geometry) — see Pattern.metadata ----------------
    # Everything the plate compositor needs here is pure arithmetic on the
    # params, so none of these has to run generate(). ``extra`` deliberately
    # carries no MEASURED field (no coverage mean, no realized duty) precisely
    # so the two paths cannot disagree — the measurements live in the fab
    # bakes' own stats.

    @classmethod
    def pixel_pitch_um(
        cls,
        image: str = _DEFAULT_IMAGE,
        colour_mode: str = "plain",
        fade_start: float = 0.50,
        fade_gate: float = 0.94,
        line_period_um: float = 44.0,
        tone_steps: int = 22,
        extent_um: float = 2000.0,
    ) -> float:
        return line_period_um / resolve_steps(line_period_um, tone_steps)

    @classmethod
    def min_feature_um(
        cls,
        image: str = _DEFAULT_IMAGE,
        colour_mode: str = "plain",
        fade_start: float = 0.50,
        fade_gate: float = 0.94,
        line_period_um: float = 44.0,
        tone_steps: int = 22,
        extent_um: float = 2000.0,
    ) -> float:
        """The finest gold band = one tone step of a line period.

        The prep clamps coverage into ``[1/steps, 1 - 1/steps]``, so the thinnest
        band AND the thinnest gap are both exactly one step — which is why
        ``resolve_steps`` caps steps at ``period / 2 um``.
        """
        return line_period_um / resolve_steps(line_period_um, tone_steps)

    @classmethod
    def extra_metadata(
        cls,
        image: str = _DEFAULT_IMAGE,
        colour_mode: str = "plain",
        fade_start: float = 0.50,
        fade_gate: float = 0.94,
        line_period_um: float = 44.0,
        tone_steps: int = 22,
        extent_um: float = 2000.0,
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
        steps = resolve_steps(line_period_um, tone_steps)
        extra = {
            "image": image,
            "colour_mode": colour_mode,
            "line_period_um": float(line_period_um),
            "tone_steps": int(steps),
            "finest_band_um": round(line_period_um / steps, 3),
            "asset_px": asset_px_for(extent_um, line_period_um),
            "fade_start": float(fade_start),
            "fade_gate": float(fade_gate),
            # The eye's integration cell is 87 um at 300 mm; under ~150 cells
            # across the picture is a thumbnail, however finely it is written.
            "eye_cells_across": int(extent_um / 87.0),
        }
        return extra, {"art_solid": True}, ()

    @classmethod
    def generate(
        cls,
        image: str = _DEFAULT_IMAGE,
        colour_mode: str = "plain",
        fade_start: float = 0.50,
        fade_gate: float = 0.94,
        line_period_um: float = 44.0,
        tone_steps: int = 22,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        """The standalone Pattern Lab view: the line screen, front layer only.

        Deliberately just the bands — no colour sub-grating. The sub-grating is
        periodic and is emitted as array references by the fab writers
        (``screenrects.stripe_plan``); materialising it here would be millions
        of rectangles for a catalog thumbnail. The composed box face and the two
        fab bakes carry the full construction.
        """
        steps = resolve_steps(line_period_um, tone_steps)
        n_px = asset_px_for(extent_um, line_period_um)
        cov, ids, _periods = photo_coverage(
            image, fade_start, fade_gate, steps, n_px, colour_mode=colour_mode
        )
        plan = colour_plan(colour_mode)
        bands, _pid, report = sr.screen_bands(
            cov,
            extent_um=extent_um,
            line_period_um=line_period_um,
            tone_steps=steps,
            period_id=None if plan.mode == "plain" else ids,
            emit="metal",
        )
        extra, recipe_data, _names = cls.extra_metadata(
            image=image,
            colour_mode=colour_mode,
            fade_start=fade_start,
            fade_gate=fade_gate,
            line_period_um=line_period_um,
            tone_steps=tone_steps,
            extent_um=extent_um,
        )
        extra = {
            **extra,
            "n_band_rects": int(report["n_band_rects"]),
            "n_lines": int(report["n_lines"]),
            "coverage": round(float(report["coverage"]), 4),
        }
        return GeneratedPattern(
            front=rects_to_multipolygon(bands),
            back=empty_layer(),
            extent_um=(extent_um, extent_um),
            pixel_pitch_um=cls.pixel_pitch_um(
                line_period_um=line_period_um, tone_steps=tone_steps
            ),
            min_feature_um=cls.min_feature_um(
                line_period_um=line_period_um, tone_steps=tone_steps
            ),
            extra=extra,
            recipe_data=recipe_data,
        )
