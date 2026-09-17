"""Composing a plate: the masks, the centrepiece paste, the cached materialize.

The runtime path is ``materialize_plate`` -> ``_raster_compose_plate``: the
cached central-pattern PNGs and a RasterPen-rendered frame composite straight
into PIL images (no Shapely on the hot path). Frame polygons are CONCATENATED
onto the front gold layer where a polygon form is needed (never
``unary_union`` — see ``_concat_polygons``).

``PLATE_COMPOSE_VERSION`` lives here because this module is what it covers.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon

from .. import region_art as _RA
from ..patterns.base import registry
from ..patterns.frames import (
    RectFrame,
    Scene,
    generate_frame,
    render_scene_to_image,
)
from ..patterns.frames.api import FrameParams
from ..rasterize import (
    DEFAULT_THUMBNAIL_METAL,
    THUMBNAIL_PALETTES,
    make_thumbnail,
)
from ..service import (
    cache_lock,
    extra_layer_url,
    heavy_compute_gate,
    read_json_cache,
    save_png_atomic,
    validate_params,
    variant_key,
    write_json_atomic,
)
from .literal import _write_literal_rasters
from .photo import _photo_band_stamp
from .recipe import (
    ART_LEVEL,
    CENTERPIECE_FILL,
    FRAME_LEVEL,
    RAINBOW_LEVEL,
    _carrier_recipe_data,
    _frame_level_for,
)
from .spec import FaceKind, PlateSpec, _aperture, plate_hash
# PLATES_ROOT is read through the module, never bound by value: the test
# fixtures repoint it at a tmp dir (see tests/conftest.py::isolated_data) and a
# by-value import in each submodule would keep writing to backend/data.
from . import spec as _spec

_log = logging.getLogger("optics.plates")

def _concat_polygons(*sources: MultiPolygon) -> MultiPolygon:
    """Combine MultiPolygons by concatenation, not union.

    For the lithography raster we don't care that overlapping front-layer
    polygons appear twice — ``ImageDraw.polygon`` paints them with the same
    gold color and the result is visually identical. Skipping ``unary_union``
    on tens of thousands of small polygons (a dense frame's stroke buffer
    output) is what keeps the polygon path inside GEOS memory budgets.
    """
    geoms: list[Polygon] = []
    for src in sources:
        if src is None or src.is_empty:
            continue
        if isinstance(src, MultiPolygon):
            geoms.extend(g for g in src.geoms if not g.is_empty)
        elif isinstance(src, Polygon):
            geoms.append(src)
    return MultiPolygon(geoms)


def _centerpiece_masks(
    spec: PlateSpec, n_px: int
) -> tuple["np.ndarray", "np.ndarray"] | None:
    """(front_art, back_art) bool silhouettes for the tilt-switch centerpiece.

    Rendered directly from the motif silhouettes (clean masks, NO baked stripe
    carrier — the shader draws the glimmer procedurally) at ``n_px`` resolution.
    Returns ``None`` for faces that don't define a paired centerpiece, so the
    plate is frame-only. Which motif to render is driven by the pattern slug so
    the composition stays data-directed rather than hard-wiring one pattern into
    the plate compositor; the face's ``pattern_params`` feed the slugs whose
    silhouette depends on them (neither of the two that reach the body today
    does).

    Only a ``FaceKind.TWO_PLY`` face has a silhouette pair at all, and saying so
    HERE is what keeps every consumer consistent: the composed preview, the fab
    SVG's ``_center_masks`` and ``export_fine._build_zone_masks`` all read this
    one function, so a blank face gets no art on either layer and a photo face
    gets no ``front_art``/``back_art`` for the switch carrier to fill. (A
    photo's geometry is a LINE SCREEN, not a silhouette — it comes from
    ``_photo_band_stamp`` / ``photo_band_rects``; a region face's is a map of
    gratings — ``region_art.centerpiece_regions``.)

    TWO SLUGS REACH THE BODY (2026-09-16), both hidden two-ply exemplars:
    ``globe-duo-phase`` (the barrier switch) and ``monogram-jp`` composed with
    ``single_ply=False`` (the shading moiré).
    """
    if spec.kind is not FaceKind.TWO_PLY:
        return None
    slug = spec.pattern_slug
    if slug == "globe-duo-phase":
        # Duo-globe A/B tilt switch: front = orthographic globe centered on
        # CALIFORNIA (phase 0), back = orthographic globe centered on COLOMBIA
        # (phase π). Same disc size/position so the switch reads as one globe
        # ROTATING between the couple's homes. Clean silhouettes (no baked stripe
        # carrier) — the shader's centerpiece phase-switch supplies the glimmer,
        # exactly like the colibrí/globe branch. Centers/star markers mirror
        # patterns.artistic.globe_duo_phase so preview == the standalone pattern.
        from ..patterns.artistic.globe_duo_phase import (
            CAL_LAT0,
            CAL_LON0,
            CAL_STAR_LONLAT,
            COL_LAT0,
            COL_LON0,
            COL_STAR_LONLAT,
        )
        from ..patterns.geo.render import render_globe

        n = max(64, int(n_px))
        front = render_globe(
            n,
            lat0=CAL_LAT0,
            lon0=CAL_LON0,
            highlight_country=None,
            highlight_star=True,
            star_lonlat=CAL_STAR_LONLAT,
        )
        back = render_globe(
            n,
            lat0=COL_LAT0,
            lon0=COL_LON0,
            highlight_country="Colombia",
            highlight_star=True,
            star_lonlat=COL_STAR_LONLAT,
        )
        return (front, back)
    if slug == "monogram-jp":
        # Interlocked cursive J+P monogram on the FRONT face; the BACK face gets
        # no centerpiece art (all-False), so the back reads as plain carrier and
        # the monogram *shimmers* on tilt — the shading-moiré exemplar
        # (SHIMMER_MOIRE_SLUGS). The PRODUCTION lid is the same slug on ONE ply,
        # where the monogram is a region_art colour mapping instead and never
        # reaches here.
        import numpy as np

        from ..patterns.motifs import monogram

        n = max(64, int(n_px))
        front = monogram.monogram_silhouette((1.0, 1.0), n_grid=n)
        back = np.zeros_like(front)
        return (front, back)
    return None



def _paste_centerpiece(
    spec: PlateSpec,
    front: Image.Image,
    back: Image.Image,
    pitch_um: float,
    plate_w: int,
    plate_h: int,
) -> None:
    """Paint the centerpiece into the aperture.

    The centerpiece is a square of side ``CENTERPIECE_FILL × aperture`` centered
    on the plate. Three constructions, in the order every writer tests them:
    a ``region_art`` colour map (the production lid and front), a photo LINE
    SCREEN (the three photo sides), or — on the two hidden two-ply exemplars —
    a pair of silhouettes from ``_centerpiece_masks`` at ART_LEVEL, front and
    back. All are clamped to their layer's keep-out later by the rim zero.
    """
    import numpy as np

    aperture = _aperture(spec)
    if aperture <= 0:
        return
    side_um = CENTERPIECE_FILL * aperture
    side_px = max(8, int(round(side_um / pitch_um)))

    cx = plate_w // 2
    cy = plate_h // 2
    x0 = cx - side_px // 2
    y0 = cy - side_px // 2

    if spec.kind is FaceKind.REGION:
        # SINGLE-LAYER DIFFRACTION centrepiece: solid regions at ART_LEVEL
        # (``art_solid`` → plain gold), diffractive regions at RAINBOW_LEVEL
        # (the shader's spectral sheen; the period itself rides period_front).
        ra = _RA.centerpiece_regions(spec.pattern_slug, side_px, spec.pattern_params)
        if ra is None:
            return
        metal, coloured = ra.metal(), ra.coloured()
        lvl = np.where(coloured, RAINBOW_LEVEL, np.where(metal, ART_LEVEL, 0)).astype(np.uint8)
        front.paste(
            Image.fromarray(lvl, "L"),
            box=(x0, y0),
            mask=Image.fromarray((metal.astype(np.uint8) * 255), "L"),
        )
        return

    if spec.kind is FaceKind.PHOTO:
        # LINE SCREEN, front only. Bands go in at ART_LEVEL — with
        # ``recipe_data['art_solid']`` the shader fills that level with solid
        # gold instead of the procedural switch carrier, because the band height
        # IS the tone and a second grating over it would halve every duty. The
        # bands that carry a colour period go in at RAINBOW_LEVEL instead, where
        # the shader already adds the travelling spectral sheen that stands in
        # for their sub-grating (which the fab bakes for real).
        stamp = _photo_band_stamp(spec, side_px)
        if stamp is None:
            return
        gold, coloured = stamp
        lvl = np.where(coloured, RAINBOW_LEVEL, np.where(gold, ART_LEVEL, 0)).astype(
            np.uint8
        )
        # Binary paste mask, not the graylevel — an 'L' mask alpha-blends and
        # would drag RAINBOW_LEVEL 200 down to ~157 (same trap as _paste below).
        front.paste(
            Image.fromarray(lvl, "L"),
            box=(x0, y0),
            mask=Image.fromarray((gold.astype(np.uint8) * 255), "L"),
        )
        return

    masks = _centerpiece_masks(spec, side_px)
    if masks is None:
        return
    front_art, back_art = masks

    def _resize_to_side(art: "np.ndarray") -> "np.ndarray":
        if art.shape[0] != side_px or art.shape[1] != side_px:
            m = Image.fromarray((art.astype(np.uint8) * 255), "L").resize(
                (side_px, side_px), Image.NEAREST
            )
            return np.asarray(m) > 127
        return art

    def _paste(dst: Image.Image, art: "np.ndarray") -> None:
        # Resize the motif to exactly side_px if the motif grid differs.
        a = _resize_to_side(art)
        # Paint ART_LEVEL where the silhouette covers. A per-pixel graylevel
        # stamp (not a flat paste) so the level survives; where the stamp is 0
        # the frame/window underneath stays intact (paste mask = the stamp).
        lvl = np.where(a, ART_LEVEL, 0).astype(np.uint8)
        stamp = Image.fromarray(lvl, "L")
        # Binary paste mask (NOT the graylevel itself — an "L" mask alpha-blends,
        # which would drag RAINBOW_LEVEL 200 down to ~157). We want a hard stamp
        # of the graylevel wherever the art covers, frame/window intact elsewhere.
        pmask = Image.fromarray((a.astype(np.uint8) * 255), "L")
        dst.paste(stamp, box=(x0, y0), mask=pmask)

    _paste(front, front_art)
    _paste(back, back_art)


def _raster_compose_plate(spec: PlateSpec, out_dir: Path) -> dict[str, Any]:
    """Raster-space compose: foliage FRAME at the edges + tilt-switch CENTERPIECE.

    The single-channel ``L`` masks carry TWO regions each, separated by
    intensity so the shader's ``foliage_moire`` recipe can run both effects at
    once (see plate.frag runFoliageMoire):

    FRONT  = a perimeter FOLIAGE FRAME (colonize band, value ``FRAME_LEVEL``,
             carrying the per-motif angle-bucket palette) wrapping the four
             edges, PLUS the CENTERPIECE (``ART_LEVEL`` for solid gold,
             ``RAINBOW_LEVEL`` where it carries a sub-grating) filling the
             aperture inside the band. Zone codes only — the fine gold grating
             that flashes inside them is written in the fab bake and sampled by
             the renderer off the literal rasters, so the PNG carries no
             grating (no raster aliasing rings).
    BACK   = empty on a SINGLE-PLY face (the production box is all six). On a
             two-ply exemplar it is a uniform CARRIER window (``FRAME_LEVEL``)
             spanning the whole EXPOSED face (``back_dims`` — foil overlap
             only), plus that construction's back silhouette at ``ART_LEVEL``.

    Returns the *partial* manifest dict (files + extras + recipe data); the
    caller stamps the spec/labels and writes manifest.json.
    """
    central_cls = registry[spec.pattern_slug]
    merged_params = {**central_cls.defaults(), **spec.pattern_params}
    # The ParamSpec bounds check used to ride in for free on the materialize call
    # this replaced; keep it explicit so an out-of-range face param still raises
    # (the /plates and /boxes routes surface it as a 400/422) instead of sizing a
    # lattice off it.
    validate_params(spec.pattern_slug, merged_params)

    # METADATA ONLY. The centerpiece is drawn from the motif silhouettes
    # (_paste_centerpiece / _centerpiece_masks) and the frame from generate_frame,
    # so the central pattern's POLYGONS are never used on this path — the compose
    # consumes exactly four things: the pitch below, the min feature, and the
    # ``extra`` / ``recipe_data`` blocks the plate manifest carries through.
    # Materializing the whole variant for those cost 1-4 s per face (8-20 s per
    # cold six-face box). Each generator overrides the metadata accessors with the
    # same arithmetic its generate() uses; the two that MEASURE a field off their
    # own emitted raster (capybara-scanimation, bitmap-halftone) still fall back
    # to generate() there. See Pattern.metadata.
    central_meta = central_cls.metadata(**merged_params)
    central_pitch = central_meta.pixel_pitch_um

    # Plate canvas — pitch chosen so the longest plate side caps at ~1500 px.
    # We deliberately stay below the rasterize cap (16M px) by a wide margin
    # so the *six* face textures uploaded together to the WebGL preview don't
    # exhaust VRAM (3 mm cube was fine at 4000 px; 30 mm cube needs the cap).
    # The fab SVG export goes through the polygon path at full precision, so
    # this only bounds the live raster, not the actual mask.
    plate_pitch = max(central_pitch, max(spec.width_um, spec.height_um) / 1500.0)
    plate_w = max(1, int(round(spec.width_um / plate_pitch)))
    plate_h = max(1, int(round(spec.height_um / plate_pitch)))

    front = Image.new("L", (plate_w, plate_h), 0)
    back = Image.new("L", (plate_w, plate_h), 0)

    # BARE GLASS: no frame, no carrier, no centerpiece, on either layer. The
    # plate still exists — the box needs its cut dims, glass and assembly entry
    # — it just describes a rectangle of quartz. See BLANK_SLUG.
    is_blank = spec.kind is FaceKind.BLANK
    # SOLID GOLD: no frame and no centerpiece either — the front mask is
    # ART_LEVEL over the whole plate (painted below) and the rim is NOT zeroed.
    is_solid = spec.kind is FaceKind.SOLID
    # SINGLE PLY: there is no inner ply to write on, and (2026-09-10) the one
    # ply carries NO carrier: the photograph and the leaf frame on bare glass.
    # The back layer stays empty. Reference: export_fine's single-ply note.
    single_ply = bool(getattr(spec, "single_ply", False)) and not is_blank
    back_margin = spec.weld_margin_um if spec.back_margin_um is None else spec.back_margin_um

    # --- FRONT: perimeter foliage frame band ---------------------------------
    # The colonize band hugs the four edges (fill_interior=False), leaving the
    # center open for the art centerpiece. Painted at FRAME_LEVEL so the shader
    # tells frame foliage apart from the ART_LEVEL centerpiece.
    active_w, active_h = spec.active_dims()
    if is_blank or is_solid:
        scene = None
    elif active_w > 0 and active_h > 0:
        rect = RectFrame(width_um=active_w, height_um=active_h)
        frame_params = spec.frame.to_frame_params()
        frame_params.fill_interior = False  # perimeter band, center stays open
        scene = generate_frame(rect, frame_params)
        # The frame PNG now carries a PER-MOTIF graylevel (angle bucket) rather
        # than a flat silhouette; paste the graylevels straight through so the
        # shader can decode each motif's fringe direction. The mask (frame_png
        # itself, nonzero) restricts the paste to the foliage pixels so the
        # open center + weld rim stay untouched.
        frame_png = render_scene_to_image(
            scene, rect, frame_params, plate_pitch, level_fn=_frame_level_for
        )
        active_w_px = max(1, int(round(active_w / plate_pitch)))
        active_h_px = max(1, int(round(active_h / plate_pitch)))
        weld_x_px = (plate_w - active_w_px) // 2
        weld_y_px = (plate_h - active_h_px) // 2
        # Binarize the paste mask. Using frame_png itself as an 'L' mask
        # alpha-blends the graylevels onto the black plate (result = L*L/255),
        # squaring every angle-bucket level so orchid/coffee fall below the
        # shader's FRAME_MIN and the six fringe directions collapse to two.
        # A 0/255 mask keeps each foliage pixel's graylevel intact — same trap
        # documented for the centerpiece _paste above.
        bin_mask = frame_png.point(lambda v: 255 if v else 0)
        front.paste(frame_png, box=(weld_x_px, weld_y_px), mask=bin_mask)
    else:
        scene = None  # weld swallowed the active area — degenerate but legal

    # --- BACK: solid uniform carrier window across the whole exposed face ----
    # Skipped on a single-ply face (the carrier is already on the front, above)
    # and on a blank one (no gold anywhere).
    back_w, back_h = spec.back_dims()
    if not is_blank and not single_ply and back_w > 0 and back_h > 0:
        back_w_px = max(1, int(round(back_w / plate_pitch)))
        back_h_px = max(1, int(round(back_h / plate_pitch)))
        bx0 = (plate_w - back_w_px) // 2
        by0 = (plate_h - back_h_px) // 2
        ImageDraw.Draw(back).rectangle(
            (bx0, by0, bx0 + back_w_px - 1, by0 + back_h_px - 1), fill=FRAME_LEVEL
        )

    # --- CENTERPIECE: colibrí (front) + globe (back) tilt-switch art ---------
    # Fills the aperture inside the frame band, generously, with a breathing gap
    # to the frame. The two silhouettes are painted at ART_LEVEL so the shader
    # runs the phase-shift switch inside the centerpiece while the frame band
    # around it runs the foliage-moiré shimmer.
    if is_solid:
        ImageDraw.Draw(front).rectangle((0, 0, plate_w, plate_h), fill=ART_LEVEL)
    elif not is_blank:
        _paste_centerpiece(spec, front, back, plate_pitch, plate_w, plate_h)

    # Belt-and-suspenders: clear each layer's OWN keep-out rim. The front art
    # must stay inside the weld margin (foil overlap + safety); the back
    # carrier only has to clear the foil overlap (its wider window), so each
    # gets its own border zeroed. Rounding overshoot can't leak under the foil.
    def _zero_rim(img: Image.Image, margin_um: float) -> None:
        if margin_um <= 0:
            return
        m = max(1, int(round(margin_um / plate_pitch)))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, plate_w, m), fill=0)
        d.rectangle((0, plate_h - m, plate_w, plate_h), fill=0)
        d.rectangle((0, 0, m, plate_h), fill=0)
        d.rectangle((plate_w - m, 0, plate_w, plate_h), fill=0)

    # On a SINGLE-PLY face the carrier is the back layer moved onto the outer
    # ply, so its keep-out is the BACK window (foil overlap only), not the front
    # weld margin — the same split the single-ply block in
    # ``export_fine.build_plate_fine`` makes. The frame band is bounded by the
    # active rect and the centerpiece by the art box, so nothing else can
    # reach into the ring between the two rims; taking
    # the smaller of them keeps the frame's own keep-out honest whichever way
    # round the two margins happen to fall for a given foil/glass pair.
    if not is_solid:
        _zero_rim(
            front,
            min(spec.weld_margin_um, back_margin) if single_ply else spec.weld_margin_um,
        )
    _zero_rim(back, back_margin)

    save_png_atomic(front, out_dir / "front.png")
    save_png_atomic(back, out_dir / "back.png")
    # One chip per litho metal. Written at compose time (three tiny pastes)
    # rather than keyed into the plate hash, so switching metal in the UI is an
    # instant swap instead of a full six-plate recompose — the METAL never
    # changes the mask geometry, only how the chip is tinted.
    thumb = make_thumbnail(front, back, size=320)
    save_png_atomic(thumb, out_dir / "thumbnail.png")
    for _metal in THUMBNAIL_PALETTES:
        if _metal == DEFAULT_THUMBNAIL_METAL:
            continue
        save_png_atomic(
            make_thumbnail(front, back, size=320, metal=_metal),
            out_dir / f"thumbnail_{_metal}.png",
        )

    # The frame scene (potentially MBs of segment dicts) goes to a sidecar,
    # NOT into manifest.json: embedding it made every plate manifest 0.4-2.3
    # MB and dominated both the warm box regen (6 × json decode) and the cold
    # compose's manifest encode. The frontend never reads it and export strips
    # it from the fab bundle, but it is NOT debug-only: the fab SVG bake and
    # export_fine's zone masks load it back (``frame_scene_for_plate``) instead
    # of regrowing the band, which is what keeps all three renders of a face
    # pinned to one scene. Keyed to PLATE_COMPOSE_VERSION by that reader.
    frame_scene = scene.to_dict() if scene is not None else {"segments": [], "flowers": [], "leaves": [], "max_t": 0.0}
    write_json_atomic(out_dir / "scene.json", frame_scene, indent=None)

    # The standalone variant's recipe_data, reproduced without materializing it:
    # ``_materialize_locked`` stamps one ``<name>_png`` URL per extra layer after
    # the generator's own keys, and the plate manifest carries the merged block
    # through. Same variant id and same URL builder as that path, so the strings
    # are identical — the layer PNGs themselves only exist once someone actually
    # materializes the standalone variant (nothing on the plate path reads them;
    # the composed plate binds foliage_moire against its own front/back masks).
    central_variant = variant_key(merged_params)
    central_recipe_data = {
        **central_meta.recipe_data,
        **{
            f"{name}_png": extra_layer_url(spec.pattern_slug, central_variant, name)
            for name in central_meta.extra_layer_names
        },
    }

    return {
        "pixel_pitch_um": plate_pitch,
        "min_feature_um": central_meta.min_feature_um,
        "central_extra": central_meta.extra,
        "central_recipe_data": central_recipe_data,
    }



# Bump when the composed-plate output changes under an unchanged spec hash:
# ``_raster_compose_plate``, ``_paste_centerpiece``, ``_centerpiece_masks``, the
# mask level palette (FRAME_LEVEL / ART_LEVEL / FRAME_BUCKET* / RAINBOW_LEVEL),
# or ``_carrier_recipe_data`` (including the render_recipe the manifest forces
# and every shader knob it emits). ``plate_hash`` covers spec fields only, so a
# cached manifest+PNG pair is otherwise served forever after a code or constant
# change — the same trap PLATE_SVG_VERSION already closes on the fab SVGs. See
# CLAUDE.md; a mismatch on the hit path is treated as a miss.
#
# The per-version history (v2 .. v25) lives in docs/plates-changelog.md.
# v26: the two-ply optics come off the plate (2026-09-16). ``_carrier_recipe_data``
#     drops the water-scanimation keys (``water_scan_n``,
#     ``water_ripple_wavelength_um``, ``water_body_carrier_*``,
#     ``water_waterline_y``) and the diffraction-accent keys
#     (``rainbow_period_um``/``_angle_deg``/``_duty``/``_zero_order``,
#     ``accent_interleave_pitch_um``, ``accent_moire_*``); ``_paste_centerpiece``
#     no longer stamps RAINBOW_LEVEL accent patches or the full-width water band,
#     and ``_centerpiece_masks`` keeps only the two hidden exemplars. The mask
#     PNGs of every PRODUCTION face are unchanged (none of them entered any of
#     those branches) but recipe_data is, and it is cached.
# v27: FaceKind (2026-09-16). The blank/solid/photo/region/two-ply decision moves
#     to ``PlateSpec.kind`` and ``recipe_data`` gains ``face_kind``. No writer's
#     BRANCH changes — the enum classifies on exactly the inputs the nine slug
#     tests read — so every face's PNGs and SVG are byte-identical; the marker
#     moves because the manifest gained a key.
PLATE_COMPOSE_VERSION = 27


# Plate ids whose ``scene.json`` was written by the compose CURRENTLY running on
# this thread. The literal-raster bake runs INSIDE ``_materialize_plate_locked``,
# before the manifest is published — so ``get_plate`` still reads the old
# manifest (or none at all) and the compose_version gate below would reject a
# sidecar this very compose just wrote, regrowing the band a second time in the
# same call. Thread-local rather than module-global: the gate exists to stop a
# sidecar from an OLDER compose leaking in, and only the thread inside the
# compose knows the file is current. Set/cleared in _materialize_plate_locked.
_fresh_scene = threading.local()


def _scene_sidecar_is_fresh(plate_id: str) -> bool:
    return plate_id in getattr(_fresh_scene, "ids", ())


def frame_scene_for_plate(
    plate_dir: Path,
    manifest: dict[str, Any] | None,
    rect: RectFrame,
    frame_params: FrameParams,
) -> Scene:
    """The composed plate's frame scene: ``scene.json`` sidecar first, regrow on miss.

    ``_raster_compose_plate`` writes the sidecar from exactly this
    ``(rect, frame_params)`` pair, and the grower is seed-deterministic, so
    reusing it is not merely equivalent — it is the SAME scene the cached
    preview PNG was painted from, which is what the "fab SVG = preview PNG"
    contract wants. Regrowing instead costs 1-3 s per face and would silently
    diverge from a cached PNG whenever the grower code changed without a
    version bump.

    Gated on the manifest's ``compose_version`` so a sidecar left by an older
    compose cannot leak into a plate that the current compose would rebuild —
    or, mid-compose, on ``_scene_sidecar_is_fresh``, which is the same
    guarantee before there is a manifest to read it from.
    """
    fresh = _scene_sidecar_is_fresh(plate_dir.name)
    if fresh or (
        manifest is not None and manifest.get("compose_version") == PLATE_COMPOSE_VERSION
    ):
        data = read_json_cache(plate_dir / "scene.json")
        if data is not None and "segments" in data:
            return Scene.from_dict(data)
    return generate_frame(rect, frame_params)


def materialize_plate(spec: PlateSpec, force: bool = False) -> dict[str, Any]:
    """Compose the plate, cache PNG/SVG/manifest under ``data/plates/<hash>/``.

    Serialized per plate hash (see ``service.cache_lock``): a box regen fans six
    faces out sequentially, and two overlapping regens of the same face would
    otherwise both compose into the same directory. A manifest that is
    unreadable or carries a stale ``compose_version`` counts as a miss.
    """
    _spec.PLATES_ROOT.mkdir(parents=True, exist_ok=True)
    pid = plate_hash(spec)
    with cache_lock(f"plate:{pid}"):
        return _materialize_plate_locked(spec, pid, force)


def _materialize_plate_locked(spec: PlateSpec, pid: str, force: bool) -> dict[str, Any]:
    out = _spec.PLATES_ROOT / pid
    manifest_path = out / "manifest.json"
    if not force:
        cached = read_json_cache(manifest_path)
        if cached is not None and cached.get("compose_version") == PLATE_COMPOSE_VERSION:
            _log.info("materialize_plate cache_hit id=%s slug=%s", pid, spec.pattern_slug)
            return cached
        if manifest_path.exists():
            _log.info(
                "materialize_plate regenerate id=%s slug=%s reason=%s",
                pid,
                spec.pattern_slug,
                "unreadable_manifest"
                if cached is None
                else f"compose_version {cached.get('compose_version')!r} != {PLATE_COMPOSE_VERSION}",
            )

    # Everything below is the heavy path, so it runs under the process-wide
    # compute gate (CLAUDE.md machine constraint; see service.heavy_compute_gate).
    # Taken HERE rather than around the whole function so the cache-hit return
    # above — six of them on a warm box regen — never queues behind an unrelated
    # cold compose.
    with heavy_compute_gate:
        t0 = time.perf_counter()
        out.mkdir(parents=True, exist_ok=True)
        raster_result = _raster_compose_plate(spec, out)
        pitch = raster_result["pixel_pitch_um"]

        # LITERAL rasters — the fabricated chrome the renderer actually samples
        # (see LITERAL_RASTER_PX). Built here, next to the shader masks, so a
        # plate slot is never half a contract: the manifest published at the
        # bottom of this block is the only thing that makes any of it visible,
        # and it advertises both. The fresh-scene window lets the fine bake read
        # the scene.json _raster_compose_plate just wrote instead of regrowing
        # the foliage band a second time in the same call.
        ids = getattr(_fresh_scene, "ids", None)
        if ids is None:
            ids = _fresh_scene.ids = set()
        ids.add(pid)
        try:
            literal_files = _write_literal_rasters(spec, out, pid)
        finally:
            ids.discard(pid)

        # SVG is built on demand by the /export endpoint (see ensure_plate_svg)
        # — the polygon path is ~40× slower than raster and the interactive UI
        # never needs it. Manifest carries empty svg paths until requested.
        # Any pair already in this slot was baked from the geometry we just
        # replaced (force, or a PLATE_COMPOSE_VERSION bump), and it still carries
        # the current PLATE_SVG_VERSION marker — so ensure_plate_svg would happily
        # serve it and break the fab-SVG = preview-PNG contract. Drop it.
        for stale_svg in (out / "front.svg", out / "back.svg"):
            stale_svg.unlink(missing_ok=True)

        central_cls = registry[spec.pattern_slug]
        manifest = {
            "kind": "plate",
            "id": pid,
            # Compose-code marker; a mismatch on the hit path is a miss (plate_hash
            # covers spec fields only). See PLATE_COMPOSE_VERSION.
            "compose_version": PLATE_COMPOSE_VERSION,
            "spec": spec.to_dict(),
            "name": spec.label or central_cls.name,
            "description": central_cls.description,
            "tags": central_cls.tags,
            "substrate": {
                "thickness_um": spec.glass.thickness_um,
                "material": spec.glass.material,
                "n": spec.glass.n,
            },
            "extent_um": [spec.width_um, spec.height_um],
            "pixel_pitch_um": pitch,
            "min_feature_um": raster_result["min_feature_um"],
            "extra": {
                **raster_result["central_extra"],
                "central_pattern": spec.pattern_slug,
                "aperture_um": _aperture(spec),
            },
            # The plate is a box-first moiré carrier: front foliage grating vs
            # back uniform grating, beat procedurally in the shader — NOT the
            # central pattern's own recipe. Force the foliage_moire recipe and
            # carry the grating knobs regardless of which central pattern seeded
            # the metadata.
            "render_recipe": "foliage_moire",
            # frame_scene deliberately NOT embedded — see _raster_compose_plate
            # (scene.json sidecar keeps the manifest ~20 KB instead of ~MBs).
            "recipe_data": {
                **raster_result["central_recipe_data"],
                **_carrier_recipe_data(spec),
            },
            "files": {
                "front_png": f"/data/plates/{pid}/front.png",
                "back_png": f"/data/plates/{pid}/back.png",
                # The literal chrome rasters the renderer samples on the two
                # pattern planes: literal_front / literal_back always (an empty
                # layer is an all-zero raster, not a missing key), period_front
                # only where a halftone carries colour sub-gratings. Same URL
                # form as front_png — the files are already on disk above.
                **literal_files,
                "front_svg": "",
                "back_svg": "",
                "thumbnail": f"/data/plates/{pid}/thumbnail.png",
                # Per-metal chips; the renderer picks by spec.metal and falls
                # back to `thumbnail` for manifests written before this existed.
                "thumbnails": {
                    _m: (
                        f"/data/plates/{pid}/thumbnail.png"
                        if _m == DEFAULT_THUMBNAIL_METAL
                        else f"/data/plates/{pid}/thumbnail_{_m}.png"
                    )
                    for _m in THUMBNAIL_PALETTES
                },
            },
        }
        # Published last and by rename — the PNGs above are already in place, so a
        # readable manifest implies a complete slot.
        write_json_atomic(manifest_path, manifest)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        _log.info(
            "materialize_plate done id=%s slug=%s %dms pitch=%.3f",
            pid,
            spec.pattern_slug,
            dt_ms,
            pitch,
        )
        return manifest


def _mask_rim(grid: "np.ndarray", margin_um: float, pitch_um: float) -> None:
    """Zero the keep-out rim of a boolean grid in place (fab clip)."""
    if margin_um <= 0:
        return
    m = max(1, int(round(margin_um / pitch_um)))
    grid[:m, :] = False
    grid[-m:, :] = False
    grid[:, :m] = False
    grid[:, -m:] = False



def list_plates() -> list[dict[str, Any]]:
    if not _spec.PLATES_ROOT.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(_spec.PLATES_ROOT.iterdir()):
        m = d / "manifest.json"
        if m.exists():
            cached = read_json_cache(m)  # None = corrupt slot, skip it
            if cached is not None:
                out.append(cached)
    return out


def get_plate(plate_id: str) -> dict[str, Any] | None:
    """The cached plate manifest, or None if absent OR unreadable.

    A truncated manifest reads as 'missing' (a 404 the caller can recover from
    by regenerating) rather than raising a 500 out of the route.
    """
    return read_json_cache(_spec.PLATES_ROOT / plate_id / "manifest.json")
