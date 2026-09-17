"""Plate composition — central optical pattern + decorative gold frame.

A ``PlateSpec`` is the user-level recipe for one glass plate:
  * a central pattern slug + params (any existing Pattern subclass), and
  * a frame spec (theme, algorithm, dials, seed) wrapping it.

The runtime path is ``materialize_plate`` -> ``_raster_compose_plate``: the
cached central-pattern PNGs and a RasterPen-rendered frame composite straight
into PIL images (no Shapely on the hot path). Frame polygons are CONCATENATED
onto the front gold layer where a polygon form is needed (never
``unary_union`` — see ``compose._concat_polygons``).

SVGs are NOT built at materialize time: ``ensure_plate_svg`` builds the fab
pair lazily on first export request (the polygon/SVG path is ~40x slower
than raster and the interactive UI never needs it).

Plate layout: the ``weld_margin_um`` rim around the plate edge stays blank
glass (reserved for copper foil + solder); the frame band sits inside that
on the *active rect*; the square central aperture sits inside the band at
``min(active) - 2*band``. The central pattern always generates square so
existing patterns (all of which take a single ``extent_um`` scalar) drop in
without changes.

THE PACKAGE (2026-09-17). This was one 2,800-line module; it is now six, in
dependency order, and this file is a pure facade over them:

    spec     what a plate IS — FrameSpec / GlassSpec / PlateSpec, FaceKind and
             the slugs it classifies on, plate_hash, the band and aperture
             geometry, PLATES_ROOT
    recipe   how a face is DRAWN — the mask level palette, the grating and
             preview constants, and _carrier_recipe_data, the shader contract
    photo    the photo-halftone centrepiece: screen inputs, band stamping,
             photo_band_rects / photo_colour_band_periods
    compose  the masks, the centrepiece paste, _raster_compose_plate and the
             cached materialize_plate; PLATE_COMPOSE_FINGERPRINT
    literal  the LITERAL rasters — fabricated chrome sampled as coverage, and
             the period map the renderer binds
    svg      the FAB SVG: ensure_plate_svg / _bake_plate_svg;
             PLATE_SVG_FINGERPRINT

The split was a pure move — every body line came across verbatim, and the
plate rasters, the fab SVGs and the witness mask are byte-identical across it.

Everything below is re-exported for the importers that predate the package
(`from . import plates as P`, `from app.plates import X`), including the
underscore names the fab writers, the witness dies and the tests reach for.
Import from the submodule directly in new code.
"""
from __future__ import annotations

from typing import Any

from . import compose, literal, photo, recipe, spec, svg

from .spec import (  # noqa: F401
    FrameSpec,
    GlassSpec,
    FaceKind,
    BLANK_SLUG,
    SOLID_SLUG,
    PHOTO_SLUG,
    face_kind,
    PlateSpec,
    plate_hash,
    _band_um,
    _aperture,
)
from .recipe import (  # noqa: F401
    FRAME_LEVEL,
    ART_LEVEL,
    RAINBOW_LEVEL,
    FRAME_BUCKET0,
    FRAME_BUCKET_STEP,
    N_FRAME_BUCKETS,
    FRAME_ANGLE_SPAN_DEG,
    frame_level,
    _frame_level_for,
    MOTIF_ANGLE_BUCKET,
    CENTERPIECE_FILL,
    BACK_CARRIER_PERIOD_UM,
    FRONT_GRATING_RATIO,
    CARRIER_ANGLE_OFFSET_DEG,
    GRATING_DUTY,
    SINGLE_PLY_LEAF_PERIOD_UM,
    SINGLE_PLY_LEAF_FILL,
    SINGLE_PLY_LEAF_HUE_PERIODS_UM,
    SINGLE_PLY_LEAF_DOT_COVERAGE,
    single_ply_leaf_period_um,
    PREVIEW_BACK_PERIOD_UM,
    PREVIEW_PITCH_MAGNIFY,
    CENTER_CARRIER_PERIOD_UM,
    CENTER_SWITCH_AXIS_DEG,
    CENTERPIECE_BEAT_UM,
    SHIMMER_MOIRE_SLUGS,
    FAB_CENTER_PERIOD_UM,
    SWITCH_INTERLACE_SLUGS,
    BASE_PARALLAX_GAP_UM,
    parallax_gap_um,
    parallax_period_scale,
    _snap_half_um,
    fab_center_period_um,
    _carrier_recipe_data,
)
from .photo import (  # noqa: F401
    _photo_screen_inputs,
    _photo_band_stamp,
    _photo_halftone_art,
    photo_band_rects,
    photo_colour_band_periods,
    _photo_multipolygon,
)
from .compose import (  # noqa: F401
    _concat_polygons,
    _centerpiece_masks,
    _paste_centerpiece,
    _raster_compose_plate,
    PLATE_COMPOSE_FINGERPRINT,
    _fresh_scene,
    _scene_sidecar_is_fresh,
    frame_scene_for_plate,
    materialize_plate,
    _materialize_plate_locked,
    _mask_rim,
    list_plates,
    get_plate,
)
from .literal import (  # noqa: F401
    LITERAL_RASTER_PX,
    LITERAL_PERIOD_SCALE,
    _literal_raster_dims,
    _literal_layer_raster,
    _literal_period_raster,
    _single_ply_leaf_period_raster,
    _write_literal_rasters,
)
from .svg import (  # noqa: F401
    _back_window_grid,
    _strip_svg_body,
    _grating_grid,
    BARRIER_COLS_QUANTUM,
    BARRIER_SNAP_TOL,
    _barrier_plate_lattice,
    _barrier_masks,
    _check_front_comb_pure,
    ensure_plate_svg,
    _bake_plate_svg,
    _publish_svg_pair,
    _SVG_BAKE_KEYS,
    PLATE_SVG_FINGERPRINT,
    _svg_is_current,
    _wrap_svg,
    _group,
    _inner_svg_paths,
)

def __getattr__(name: str) -> Any:
    """``PLATES_ROOT`` (and the ``DATA_ROOT`` it derives from) are served LIVE
    off ``spec`` rather than bound into this namespace.

    Both are repointed at a tmp directory by the test fixtures
    (``tests/conftest.py::isolated_data``). A ``from .spec import PLATES_ROOT``
    here would hand every reader a snapshot taken at import time, and a patched
    root would silently stop being where the package writes. The submodules
    read ``spec.PLATES_ROOT`` through the module object for the same reason, so
    there is exactly one live value and this forwards to it.
    """
    if name in ("PLATES_ROOT", "DATA_ROOT"):
        return getattr(spec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
