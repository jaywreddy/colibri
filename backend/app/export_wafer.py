"""4-inch wafer GDS layout — pack all 6 box plates onto one 100 mm Si wafer.

The stained-glass ring box is six fused-silica plates (see ``assembly.py``);
for a lithography run we want every plate's gold mask on ONE wafer so a single
exposure/etch pass produces the whole box. This module:

  1. Packs the 6 plate rectangles into the wafer's usable region (100 mm circle
     minus a 3 mm edge-exclusion ring) with >= 300 um dicing streets between
     plates — :func:`pack_plates` (a deterministic shelf packer, both plate
     orientations tried).
  2. Finds the largest box that fits — :func:`solve_max_scale` binary-searches
     the box size (W = D, H/W ratio held, glass + foil kept physical) because
     the default 50x50x40 mm box provably does NOT fit: its 6 plates total
     ~11,700 mm2 of glass against a ~7,850 mm2 wafer (~6,940 mm2 usable).
  3. Writes the wafer GDS — :func:`build_wafer_gds`, which DELEGATES to
     ``export_fine.build_wafer_fine_gds`` (true optical periods, per-face
     centerpiece, carrier/barrier gratings, DRC healing, BSA fiducials).
  4. Emits a matching layout-preview SVG (rects + circle only, no pattern
     polygons) for fast vision checks — :func:`write_layout_svg`.

SUPERSEDED — the original writer here composed each plate with
``plates.compose_plate``, which concatenates the STANDALONE central pattern (at
the pattern class's own few-mm ``extent_um``, never scaled into the plate
aperture) with a SOLID frame silhouette: no carrier grating, no barrier comb, no
tilt-switch centerpiece, no back carrier window. Layers (10,0)/(20,0) of such a
wafer bear no resemblance to the preview PNG or to front/back.svg, and a box
fabbed from it has no moiré and no tilt switch — the whole optical function is
missing. That path survives only as :func:`build_wafer_gds_coarse` behind the
explicit ``--legacy-coarse`` flag, for layout inspection; it is NOT fabbable.
What this module remains the single source of truth for is the packer, the
max-scale solver and the layout SVG — ``export_fine`` imports all three.

All lengths are micrometers unless a ``_mm`` suffix says otherwise.

CLI (do NOT run full generation casually — it materializes 6 plates):

    uv run python -m app.export_wafer --out data/wafer/wafer.gds

Always the largest fitting box from :func:`solve_max_scale` (the default
50x50x40 box provably does not fit); ``--mini`` is accepted as a no-op for
compatibility. ``--pack-only`` runs just the packer + layout SVG.
``--force-default`` / ``--detail`` apply to ``--legacy-coarse`` only.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# --- wafer + dicing constants (um) -----------------------------------------

WAFER_DIAMETER_UM = 100_000.0          # 4-inch wafer nominal
WAFER_RADIUS_UM = WAFER_DIAMETER_UM / 2.0
EDGE_EXCLUSION_UM = 3_000.0            # unpatternable rim near the wafer edge
USABLE_RADIUS_UM = WAFER_RADIUS_UM - EDGE_EXCLUSION_UM  # 47 mm
DICING_STREET_UM = 300.0              # minimum saw-street gap between plates

# GDS layer map (layer, datatype).
LAYER_WAFER = (99, 0)      # wafer usable-region outline circle
LAYER_OUTLINE = (1, 0)     # plate outlines + dicing-street annotation
LAYER_FRONT = (10, 0)      # front-face gold mask (viewer side, incl. frame)
LAYER_BACK = (20, 0)       # back-face gold mask (far side)
LAYER_LABEL = (3, 0)       # text: plate/face name

# Faces packed onto the wafer, in a size-descending-friendly default order.
_FACE_ORDER = ("bottom", "top", "front", "back", "left", "right")


# --- box spec bridge --------------------------------------------------------
# We keep the packer/scale-solver dependency-light: they operate on a tiny
# BoxDims value object built from assembly.cut_list, so the pure-math packer
# smoke checks never import the heavy plate/pattern stack.


@dataclass(frozen=True)
class PlateRect:
    """One plate's cut rectangle, W x H in um, tagged by face id."""

    face: str
    width_um: float
    height_um: float

    def area_um2(self) -> float:
        return self.width_um * self.height_um


@dataclass
class Placement:
    """A packed plate: lower-left corner (x0, y0) and the (possibly rotated)
    footprint on the wafer, in wafer-centered um coords (origin = wafer
    center, +x right, +y up)."""

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


def _cut_rects(
    width_um: float,
    depth_um: float,
    height_um: float,
    glass_thickness_um: float,
) -> list[PlateRect]:
    """The 6 plate rects for a box, largest-area first (better shelf packing).

    Delegates the actual cut math to ``assembly.cut_list`` so there is exactly
    one source of truth for plate dimensions.
    """
    from .assembly import cut_list

    rects = [
        PlateRect(e["face"], e["width_um"], e["height_um"])
        for e in cut_list(width_um, depth_um, height_um, glass_thickness_um)
    ]
    # Sort by longer side then area, descending: tall/wide plates placed first
    # pack more predictably in a shelf packer.
    rects.sort(key=lambda r: (max(r.width_um, r.height_um), r.area_um2()), reverse=True)
    return rects


# --- circle containment -----------------------------------------------------

def _rect_fits_circle(x0: float, y0: float, w: float, h: float, radius: float) -> bool:
    """True iff all 4 corners of the axis-aligned rect are within ``radius``
    of the wafer center. For an axis-aligned rectangle the farthest points
    from the center are always the corners, so testing corners is exact."""
    r2 = radius * radius
    # Worst corner is the one farthest from origin on each axis.
    fx = max(abs(x0), abs(x0 + w))
    fy = max(abs(y0), abs(y0 + h))
    return fx * fx + fy * fy <= r2 + 1e-6


# --- shelf packer -----------------------------------------------------------

def pack_plates(
    rects: list[PlateRect],
    *,
    usable_radius_um: float = USABLE_RADIUS_UM,
    street_um: float = DICING_STREET_UM,
) -> list[Placement] | None:
    """Deterministic shelf/row packer for rectangles inside a circle.

    Every plate is inflated by ``street_um`` on its right and top edges (a
    half-street would double up between neighbors; we instead reserve a full
    street to the right/above each plate and let the wafer edge absorb the
    outer margin), so any two packed plates are separated by at least one full
    dicing street. Plates are placed left-to-right into horizontal shelves; a
    new shelf starts above the tallest plate of the previous one.

    For each plate BOTH orientations (as-is and 90-deg rotated) are tried and
    the one that fits at the current cursor — preferring the shorter footprint
    height so shelves stay tight — is chosen. The whole layout is centered on
    the wafer by packing into a bounding box first, then translating so the
    bbox center sits at the wafer origin, and finally verifying every plate's
    true (un-inflated) corners lie within ``usable_radius_um``.

    Returns the list of :class:`Placement` (un-inflated footprints, wafer-
    centered coords) or ``None`` if the plates do not fit.
    """
    if not rects:
        return []

    # A shelf packer's density depends heavily on the strip width it packs
    # into: too wide and it makes one long row the circle's narrow top/bottom
    # can't hold; too narrow and it makes a tall stack that overflows the
    # sides. Since a circle's usable bbox is squarish, we SWEEP candidate strip
    # widths and keep the packing with the smallest bounding *diagonal* (the
    # quantity the circle actually constrains). Each plate reserves
    # (w+street) x (h+street) so any two neighbors are >= one dicing street
    # apart; the trailing street on the outermost plates is absorbed by the
    # wafer edge margin.
    def _try_pack(order: list[PlateRect], strip_width: float) -> list[Placement] | None:
        placements: list[Placement] = []
        cursor_x = 0.0
        shelf_y = 0.0
        shelf_h = 0.0
        for r in order:
            # Candidate orientations: (w, h, rotated). Prefer the orientation
            # whose reserved height is smaller (keeps shelves short); tie-break
            # to un-rotated for determinism.
            cands = [(r.width_um, r.height_um, False)]
            if abs(r.width_um - r.height_um) > 1e-6:
                cands.append((r.height_um, r.width_um, True))
            cands.sort(key=lambda c: (c[1], c[2]))

            placed = False
            for w, h, rot in cands:
                rw = w + street_um
                if cursor_x + rw <= strip_width + 1e-6:
                    placements.append(Placement(r.face, cursor_x, shelf_y, w, h, rot))
                    cursor_x += rw
                    shelf_h = max(shelf_h, h + street_um)
                    placed = True
                    break
            if placed:
                continue
            # Start a new shelf for the FIRST fitting candidate.
            new_shelf_y = shelf_y + shelf_h
            for w, h, rot in cands:
                rw = w + street_um
                if rw <= strip_width + 1e-6:
                    placements.append(Placement(r.face, 0.0, new_shelf_y, w, h, rot))
                    cursor_x = rw
                    shelf_y = new_shelf_y
                    shelf_h = h + street_um
                    placed = True
                    break
            if not placed:
                return None  # plate too wide even for the whole strip
        return placements

    def _center_and_check(placements: list[Placement]) -> list[Placement] | None:
        min_x = min(p.x0 for p in placements)
        min_y = min(p.y0 for p in placements)
        max_x = max(p.x0 + p.width_um for p in placements)
        max_y = max(p.y0 + p.height_um for p in placements)
        dx = -(min_x + max_x) / 2.0
        dy = -(min_y + max_y) / 2.0
        centered = [replace(p, x0=p.x0 + dx, y0=p.y0 + dy) for p in placements]
        for p in centered:
            if not _rect_fits_circle(p.x0, p.y0, p.width_um, p.height_um, usable_radius_um):
                return None
        return centered

    # Candidate strip widths: enough to hold N plates side by side, N=1..6,
    # using the widest reserved plate width as the unit, plus the exact usable
    # diameter. Deduplicated and bounded to the usable diameter.
    widest = max(max(r.width_um, r.height_um) + street_um for r in rects)
    diameter = 2.0 * usable_radius_um
    raw_widths = [widest * k for k in range(1, len(rects) + 1)] + [diameter]
    candidate_widths = sorted({w for w in raw_widths if w <= diameter + 1e-6})

    best: list[Placement] | None = None
    best_diag = float("inf")
    for sw in candidate_widths:
        packed = _try_pack(list(rects), sw)
        if packed is None:
            continue
        centered = _center_and_check(packed)
        if centered is None:
            continue
        bw = max(p.x0 + p.width_um for p in centered) - min(p.x0 for p in centered)
        bh = max(p.y0 + p.height_um for p in centered) - min(p.y0 for p in centered)
        diag = bw * bw + bh * bh
        if diag < best_diag:
            best_diag = diag
            best = centered
    return best


# --- max-scale solver -------------------------------------------------------

@dataclass
class MiniBoxResult:
    """The largest box that packs, plus its packing."""

    width_um: float
    depth_um: float
    height_um: float
    glass_thickness_um: float
    tape_width_um: float
    placements: list[Placement]

    def dims_mm(self) -> dict[str, float]:
        return {
            "width_mm": round(self.width_um / 1000.0, 3),
            "depth_mm": round(self.depth_um / 1000.0, 3),
            "height_mm": round(self.height_um / 1000.0, 3),
            "glass_mm": round(self.glass_thickness_um / 1000.0, 3),
            "tape_mm": round(self.tape_width_um / 1000.0, 3),
        }


def solve_max_scale(
    *,
    aspect_h_over_w: float = 40.0 / 50.0,   # default box H/W ratio (0.8)
    glass_thickness_um: float = 500.0,      # physical fused-silica plate
    tape_width_um: float = 4763.0,          # smallest foil preset (3/16")
    usable_radius_um: float = USABLE_RADIUS_UM,
    street_um: float = DICING_STREET_UM,
    lo_um: float = 5_000.0,
    hi_um: float = 60_000.0,
    iters: int = 40,
) -> MiniBoxResult | None:
    """Binary-search the largest cubic-footprint box (W = D) whose 6 plates
    pack onto the wafer.

    Height tracks width via ``aspect_h_over_w`` so the box stays proportionate;
    glass thickness and foil tape width are held at their smallest physical
    values (they do not change the cut-list footprint materially but must stay
    real for the downstream assembly to be fabbable). Returns the largest
    fitting box or ``None`` if even ``lo_um`` will not fit.
    """
    def fits(w: float) -> list[Placement] | None:
        d = w
        h = w * aspect_h_over_w
        rects = _cut_rects(w, d, h, glass_thickness_um)
        return pack_plates(
            rects, usable_radius_um=usable_radius_um, street_um=street_um
        )

    lo_fit = fits(lo_um)
    if lo_fit is None:
        return None
    hi_um_local = hi_um
    # Ensure hi does NOT fit (so the search brackets the boundary). If it does,
    # hi is already the answer.
    best_w = lo_um
    best_pl = lo_fit
    hi_fit = fits(hi_um_local)
    if hi_fit is not None:
        best_w = hi_um_local
        best_pl = hi_fit
    else:
        lo, hi = lo_um, hi_um_local
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            pl = fits(mid)
            if pl is not None:
                best_w, best_pl = mid, pl
                lo = mid
            else:
                hi = mid
    # Round the winning width DOWN to the nearest 100 um for a clean spec, then
    # re-pack to keep placements consistent with the reported dims.
    w = math.floor(best_w / 100.0) * 100.0
    pl = fits(w)
    if pl is None:  # rounding pushed it over — fall back to the raw winner
        w = best_w
        pl = best_pl
    return MiniBoxResult(
        width_um=w,
        depth_um=w,
        height_um=w * aspect_h_over_w,
        glass_thickness_um=glass_thickness_um,
        tape_width_um=tape_width_um,
        placements=pl,
    )


# --- BoxSpec construction ---------------------------------------------------

# Polygon budget for the whole wafer. gdsfactory/klayout handle far more, but
# a lean layout re-exports/DRC-checks fast and matches the project's hot-path
# frugality rule. The dominant cost is per-face: the decorative frame's
# stroke-buffered vines and the moiré central lattice.
#
# The colonize frame's vine count is strongly SEED-dependent — some seeds grow
# ~20k polys, others ~75k, and the density/band/foliage knobs only partly tame
# this — so a full 6-face artistic frame is ~280-470k and cannot be reliably
# held under budget without touching the (off-limits) frame generator. The
# wafer GDS is a FAB layout, so its default detail keeps the *functional*
# optical moiré at a coarsened period and drops the purely-decorative frame,
# which is (a) deterministic and (b) comfortably under budget (~43k). Two
# richer tiers are available for a single-plate reticle where the budget is
# not a concern. All three tiers use only spec-level knobs the patterns/frames
# already expose — NO generation code is touched, and the box's own default
# spec is unaffected.
WAFER_POLY_BUDGET = 200_000

# Detail tiers (frame band, vine density, moiré period). "optical" drops the
# frame entirely; "framed" adds a lean deterministic-ish frame (may exceed
# budget on dense seeds); "hero" reproduces the box's showpiece frame.
WAFER_MOIRE_PERIOD_UM = 40.0        # coarser than the 20 um UI default
WAFER_FRAMED_BAND_UM = 1500.0
WAFER_FRAMED_DENSITY = 0.3
WAFER_FRAMED_FOLIAGE = 0.15
WAFER_FRAMED_BLOOM = 0.15

DETAIL_TIERS = ("optical", "framed", "hero")


def mini_box_spec(result: MiniBoxResult, *, detail: str = "optical") -> "Any":
    """Build a full ``BoxSpec`` (6 faces, patterns + frames) at the mini dims.

    Takes the REAL six-face plan from ``boxes.default_box_spec`` — per-face
    centerpiece slug AND per-face frame profile — and resizes it to the fitting
    box with the smallest foil tape so the keep-out rim stays inside the
    shrunken plates. (It used to stamp ``DEFAULT_FACE_PATTERN_SLUG`` on all six
    faces, i.e. six copies of the front globe switch, discarding the plan;
    ``export_fine.mini_spec_real_faces`` does the same resize for the fab path.)

    Only :func:`build_wafer_gds_coarse` consumes this now — the fab wafer builds
    its spec in ``export_fine.mini_spec_real_faces``.

    ``detail`` selects the fidelity/polygon-budget tradeoff:
      * ``"optical"`` (default) — no decorative frame, and the coarse
        ``WAFER_MOIRE_PERIOD_UM`` carrier on the faces that actually declare a
        ``period_um`` knob (jamón tray, inscription); the barrier/scanimation
        faces keep their native periods. Leanest tier — but with the real
        six-face plan the total is no longer the ~43k the one-slug version had.
      * ``"framed"`` — lean colonize frame on top. Richer, but the seed-variable
        vine count may push some wafers over ``WAFER_POLY_BUDGET``.
      * ``"hero"`` — every face at its box default (full frame profile, native
        periods). Best for a single-plate reticle; blows the wafer budget.
    """
    if detail not in DETAIL_TIERS:
        raise ValueError(f"detail must be one of {DETAIL_TIERS} (got {detail!r})")

    from .assembly import FACE_IDS, FoilSpec
    from .boxes import default_box_spec
    from .patterns.base import registry
    from .plates import GlassSpec

    spec = default_box_spec()  # real per-face slugs + frame seeds/profiles
    spec.width_um = result.width_um
    spec.depth_um = result.depth_um
    spec.height_um = result.height_um
    spec.glass = GlassSpec(thickness_um=result.glass_thickness_um)
    # Smallest foil preset so the keep-out rim stays inside the shrunk plates.
    spec.foil = FoilSpec(tape_width_um=result.tape_width_um)

    if detail == "optical":
        frame_kw: dict[str, Any] = {"band_um": 0.0, "density": 0.0, "foliage": 0.0, "bloom": 0.0}
        pattern_params: dict[str, Any] = {"period_um": WAFER_MOIRE_PERIOD_UM}
    elif detail == "framed":
        frame_kw = {
            "band_um": WAFER_FRAMED_BAND_UM,
            "density": WAFER_FRAMED_DENSITY,
            "foliage": WAFER_FRAMED_FOLIAGE,
            "bloom": WAFER_FRAMED_BLOOM,
        }
        pattern_params = {"period_um": WAFER_MOIRE_PERIOD_UM}
    else:  # hero — box defaults
        frame_kw = {}
        pattern_params = {}

    for fid in FACE_IDS:
        plate = spec.faces.get(fid)
        if plate is None:
            continue
        if frame_kw:  # tier override on top of the face's own frame profile
            plate.frame = replace(plate.frame, **frame_kw)
        # The six faces run six DIFFERENT generators, most of which have no
        # ``period_um`` knob at all — pass a tier param only to a face whose
        # generator declares it, or ``generate()`` gets an unexpected kwarg.
        declared = registry[plate.pattern_slug].defaults() if plate.pattern_slug in registry else {}
        plate.pattern_params = {k: v for k, v in pattern_params.items() if k in declared}
    spec.normalize_face_dims()
    return spec


# --- GDS builder ------------------------------------------------------------

def _circle_pts(radius_um: float, n: int = 256) -> list[tuple[float, float]]:
    return [
        (radius_um * math.cos(2 * math.pi * i / n), radius_um * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def _add_multipolygon(comp: Any, polys: Any, layer: tuple[int, int], dx: float, dy: float) -> int:
    """Add a shapely MultiPolygon to ``comp`` on ``layer``, translated by
    (dx, dy). Returns the polygon count added.

    Interior rings are SUBTRACTED from their exterior, not drawn: GDSII has no
    hole concept, so a ring emitted on the same layer prints as gold and every
    punched eye / monogram counter / ``raster_to_polygons`` interior fills in.
    The subtraction is per polygon and only for polygons that have interiors —
    see ``export_gds.hole_free_dpolygons`` for why that does not violate the
    no-whole-geometry-boolean rule.
    """
    from shapely.geometry import MultiPolygon, Polygon

    from .export_gds import hole_free_dpolygons

    geoms = polys.geoms if isinstance(polys, MultiPolygon) else [polys]
    count = 0
    for poly in geoms:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        for dpoly in hole_free_dpolygons(poly, dx, dy):
            comp.add_polygon(dpoly, layer=layer)
            count += 1
    return count


def build_wafer_gds(
    out_path: Path,
    *,
    drc_report_path: Path | None = None,
) -> dict[str, Any]:
    """The wafer deliverable — delegates to ``export_fine.build_wafer_fine_gds``.

    This name stays the module's public entry point (it is what the justfile and
    the docs reference) but the geometry now comes from the fine-pitch writer:
    the real six-face plan, the aperture-scaled centerpiece, true 22/23.98/60/4.4
    um periods, DRC healing at the 2 um litho floor and the BSA fiducials.

    The fine builder owns its own packing (``repack_with_keepout``, so no plate
    overlaps a fiducial) and derives the face spec from ``boxes.default_box_spec``
    at the resulting size, so it takes neither a ``spec`` nor ``placements`` —
    the coarse writer's arguments are gone deliberately. The superseded
    ``compose_plate`` layout lives on as :func:`build_wafer_gds_coarse`.
    """
    from .export_fine import build_wafer_fine_gds

    return build_wafer_fine_gds(out_path, drc_report_path=drc_report_path)


def build_wafer_gds_coarse(
    spec: Any,
    placements: list[Placement],
    out_path: Path,
) -> dict[str, Any]:
    """SUPERSEDED coarse layout writer — NOT a fab mask. Layout checks only.

    Composes every plate with ``plates.compose_plate`` and stamps the resulting
    polygons into its footprint on layers (10,0)/(20,0), plate outlines + dicing
    streets on (1,0), the wafer outline on (99,0), face labels on (3,0).

    What comes out is NOT fabbable, by construction: ``compose_plate``
    concatenates the standalone central pattern at the pattern class's own
    ``extent_um`` (a few mm, never scaled to ``_aperture(spec)``) with a SOLID
    frame silhouette, and passes the pattern's own back layer straight through.
    So there is no carrier grating, no barrier comb, no tilt-switch centerpiece
    and no back carrier window — the plate has no optical function, and the solid
    frame areas are a different fab process than the intended 50%-duty lines.
    Use :func:`build_wafer_gds` for anything that will be exposed; this stays
    only because it is the cheapest way to eyeball packing + footprints with
    real polygon content. ``spec`` is a ``BoxSpec``; ``placements`` come from
    :func:`pack_plates` (wafer-centered, un-rotated footprints).

    Returns a summary dict (polygon count, layers, placement report). Requires
    gdsfactory; raises NotImplementedError if it is missing (matching the
    export_gds stub contract).
    """
    try:
        import gdsfactory as gf  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - covered by env
        raise NotImplementedError(
            "Wafer GDS export needs gdsfactory (`uv add gdsfactory`)."
        ) from e

    # Activate the generic PDK so layer tuples resolve; harmless if already on.
    try:
        gf.gpdk.PDK.activate()
    except Exception:  # noqa: BLE001 - already active / benign
        pass

    from shapely import affinity

    from .plates import compose_plate

    comp = gf.Component("wafer_coarse_not_fab")

    # (99,0) wafer usable-region outline.
    comp.add_polygon(_circle_pts(USABLE_RADIUS_UM), layer=LAYER_WAFER)

    by_face = {p.face: p for p in placements}
    total_polys = 0
    report: list[dict[str, Any]] = []

    for fid in _FACE_ORDER:
        p = by_face.get(fid)
        if p is None:
            continue
        plate_spec = spec.faces.get(fid)
        if plate_spec is None:
            continue

        gen = compose_plate(plate_spec)
        front = gen.front
        back = gen.back

        # The plate polygons are generated in plate-centered coords at the
        # plate's own (width_um x height_um). If the packer rotated the plate
        # 90 deg to fit, rotate the geometry to match the footprint.
        if p.rotated:
            front = affinity.rotate(front, 90, origin=(0, 0))
            back = affinity.rotate(back, 90, origin=(0, 0))

        # (1,0) plate outline as a rectangle at the footprint.
        hw, hh = p.width_um / 2.0, p.height_um / 2.0
        comp.add_polygon(
            [
                (p.cx - hw, p.cy - hh),
                (p.cx + hw, p.cy - hh),
                (p.cx + hw, p.cy + hh),
                (p.cx - hw, p.cy + hh),
            ],
            layer=LAYER_OUTLINE,
        )

        # Front + back gold masks translated to the plate center.
        n_front = _add_multipolygon(comp, front, LAYER_FRONT, p.cx, p.cy)
        n_back = _add_multipolygon(comp, back, LAYER_BACK, p.cx, p.cy)
        total_polys += n_front + n_back + 1

        comp.add_label(fid, (p.cx, p.cy), layer=LAYER_LABEL)
        report.append(
            {
                "face": fid,
                "cx_um": round(p.cx, 1),
                "cy_um": round(p.cy, 1),
                "width_um": round(p.width_um, 1),
                "height_um": round(p.height_um, 1),
                "rotated": p.rotated,
                "front_polys": n_front,
                "back_polys": n_back,
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp.write_gds(str(out_path))

    return {
        "gds_path": str(out_path),
        "fab_ready": False,  # no gratings / centerpiece / aperture scaling
        "superseded_by": "export_fine.build_wafer_fine_gds",
        "total_polygons": total_polys,
        "poly_budget": WAFER_POLY_BUDGET,
        "within_budget": total_polys <= WAFER_POLY_BUDGET,
        "layers": {
            "wafer": LAYER_WAFER,
            "outline": LAYER_OUTLINE,
            "front": LAYER_FRONT,
            "back": LAYER_BACK,
            "label": LAYER_LABEL,
        },
        "placements": report,
    }


# --- layout preview SVG -----------------------------------------------------

def write_layout_svg(
    placements: list[Placement],
    out_path: Path,
    *,
    usable_radius_um: float = USABLE_RADIUS_UM,
    wafer_radius_um: float = WAFER_RADIUS_UM,
) -> Path:
    """Write a lightweight layout preview SVG: wafer circles + plate rects +
    dicing streets + face labels. NO pattern polygons — this is the fast
    vision-check companion to the GDS. Coordinates in um, y flipped for SVG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    R = wafer_radius_um
    parts: list[str] = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" '
        f'viewBox="{-R:.1f} {-R:.1f} {2 * R:.1f} {2 * R:.1f}">',
        f'<rect x="{-R:.1f}" y="{-R:.1f}" width="{2 * R:.1f}" height="{2 * R:.1f}" fill="#0e0f13"/>',
        # wafer outline + usable region
        f'<circle cx="0" cy="0" r="{R:.1f}" fill="#1b1d24" stroke="#4a4f5a" stroke-width="300"/>',
        f'<circle cx="0" cy="0" r="{usable_radius_um:.1f}" fill="none" '
        f'stroke="#6b7280" stroke-width="200" stroke-dasharray="1500 900"/>',
    ]
    label_px = max(1500.0, usable_radius_um / 18.0)
    for p in placements:
        # SVG y is down; our y is up -> flip y of the lower-left corner.
        x = p.x0
        y = -(p.y0 + p.height_um)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{p.width_um:.1f}" '
            f'height="{p.height_um:.1f}" fill="#e3b53b" fill-opacity="0.18" '
            f'stroke="#e3b53b" stroke-width="150"/>'
        )
        parts.append(
            f'<text x="{p.cx:.1f}" y="{-p.cy + label_px / 3:.1f}" '
            f'font-family="sans-serif" font-size="{label_px:.0f}" '
            f'fill="#f4f4f5" text-anchor="middle">{p.face}'
            f'{"↻" if p.rotated else ""}</text>'
        )
    parts.append("</svg>")
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


# --- CLI --------------------------------------------------------------------

def _resolve_spec_and_placements(
    force_default: bool, detail: str = "optical"
) -> tuple[Any, list[Placement], dict[str, Any]]:
    """Pick the box for the ``--legacy-coarse`` layout: the largest fitting mini
    box by default, or attempt the (non-fitting) default box under
    --force-default. The fab path does its own solving inside
    :func:`build_wafer_gds`."""
    from .boxes import default_box_spec

    if force_default:
        spec = default_box_spec()
        rects = _cut_rects(
            spec.width_um, spec.depth_um, spec.height_um, spec.glass.thickness_um
        )
        pl = pack_plates(rects)
        info = {"mode": "default", "fits": pl is not None}
        if pl is None:
            raise SystemExit(
                "Default 50x50x40 mm box does NOT fit on a 100 mm wafer "
                "(plate area ~12,700 mm2 > usable ~6,940 mm2). Drop "
                "--force-default to use the fitting mini box."
            )
        return spec, pl, info

    result = solve_max_scale()
    if result is None:
        raise SystemExit("No box size fits the wafer — check constants.")
    spec = mini_box_spec(result, detail=detail)
    info = {"mode": "mini", "fits": True, "detail": detail, **result.dims_mm()}
    return spec, result.placements, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack the 6 box plates onto a 4-inch wafer GDS.")
    ap.add_argument("--out", default="data/wafer/wafer.gds", help="output GDS path")
    ap.add_argument(
        "--mini",
        action="store_true",
        help="use the largest fitting mini box (this is also the default)",
    )
    ap.add_argument(
        "--force-default",
        action="store_true",
        help="[--legacy-coarse only] attempt the default 50x50x40 box "
        "(it does not fit; will error)",
    )
    ap.add_argument(
        "--detail",
        choices=DETAIL_TIERS,
        default=None,
        help="[--legacy-coarse only] fidelity/budget tier: optical (default, "
        "frameless, <budget), framed (lean frame), hero (full showpiece frame, "
        "over budget)",
    )
    ap.add_argument(
        "--pack-only",
        action="store_true",
        help="run only the pure-math packer + SVG preview (no pattern generation)",
    )
    ap.add_argument(
        "--drc-report",
        default=None,
        help="path for the fab writer's per-face DRC report JSON",
    )
    ap.add_argument(
        "--legacy-coarse",
        action="store_true",
        help="write the SUPERSEDED compose_plate layout instead of the fab "
        "wafer: no carrier/barrier gratings, no tilt-switch centerpiece, "
        "central art left at its native few-mm extent. Layout inspection ONLY "
        "— the result is not a fabbable mask.",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    svg_out = out.with_suffix(".svg")

    if args.pack_only:
        result = solve_max_scale()
        if result is None:
            print("no fit")
            return 1
        write_layout_svg(result.placements, svg_out)
        print(f"mini dims (mm): {result.dims_mm()}")
        for p in result.placements:
            print(
                f"  {p.face:7s} x0={p.x0:9.1f} y0={p.y0:9.1f} "
                f"{p.width_um:8.1f}x{p.height_um:8.1f} rot={p.rotated}"
            )
        print(f"svg -> {svg_out}")
        return 0

    if args.legacy_coarse:
        spec, placements, info = _resolve_spec_and_placements(
            args.force_default, args.detail or "optical"
        )
        write_layout_svg(placements, svg_out)
        summary = build_wafer_gds_coarse(spec, placements, out)
        print(f"mode={info}")
        n = summary["total_polygons"]
        over = " OVER BUDGET" if n > WAFER_POLY_BUDGET else ""
        print(f"gds -> {summary['gds_path']} ({n} polygons, budget {WAFER_POLY_BUDGET}{over})")
        print(f"svg -> {svg_out}")
        print("NOT FAB-READY: coarse compose_plate layout (no gratings, no "
              "centerpiece, central art unscaled). Drop --legacy-coarse for the "
              "fab wafer.")
        return 0

    for flag, name in ((args.force_default, "--force-default"), (args.detail, "--detail")):
        if flag:
            raise SystemExit(
                f"{name} applies to --legacy-coarse only; the fab wafer always "
                "uses the largest fitting mini box at native optical periods."
            )

    # Mirror the fine builder's own packing (deterministic: same solver, same
    # fiducial keep-out) so the preview SVG shows the footprints it actually
    # wrote, not the pre-keep-out ones.
    from .export_fine import repack_with_keepout

    result = solve_max_scale()
    if result is None:
        raise SystemExit("No box size fits the wafer — check constants.")
    result, placements = repack_with_keepout(result)
    write_layout_svg(placements, svg_out)

    drc_path = Path(args.drc_report) if args.drc_report else None
    summary = build_wafer_gds(out, drc_report_path=drc_path)
    print(f"mode=mini-fine  dims(mm)={result.dims_mm()}")
    print(f"gds -> {summary['gds_path']} ({summary['total_polygons']} polygons)")
    print(f"svg -> {svg_out}")
    for r in summary["plates"]:
        print(f"  {r['face']:7s} {r['slug']:22s} front={r['front']:6d} back={r['back']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
