"""What a plate IS: the specs, the face classification, the cache id.

Split out of the old 2,800-line ``plates.py`` (2026-09-17). This is the base
module of the package — everything else imports it and it imports nothing from
its siblings, so the dependency order is

    spec -> recipe -> photo -> compose/literal -> svg

and ``__init__`` re-exports the lot. See the package docstring.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any

from .. import region_art as _RA
from ..patterns.frames.api import FrameParams
from ..service import DATA_ROOT

_log = logging.getLogger("optics.plates")

PLATES_ROOT = DATA_ROOT / "plates"

@dataclass
class FrameSpec:
    """Frame generation knobs, serializable across the API.

    The four band-composition dials (``edge_gradient``, ``understory``,
    ``border_vine``, ``corner_fans``) pass straight through to the colonize
    band grower (see frames/api.py::FrameParams). They're surfaced here so a
    box can hand each face its OWN frame recipe — a distinct edge gradient +
    understory density per face makes every side read as a deliberately
    different engraved border even before theme assignment.
    """

    algorithm: str = "wreath"
    theme: str = "esmeralda"
    density: float = 1.0
    bloom: float = 0.6
    foliage: float = 0.6
    seed: int = 1
    band_um: float | None = None  # default: 12% of shorter plate side
    edge_gradient: float = 0.8   # outer-edge density bias (0 flat .. 1 strong)
    understory: float = 0.85     # density of the outer-band small-leaf infill
    border_vine: float = 1.15    # continuous running-ornament border line
    corner_fans: float = 1.0     # size/reach of the corner fan compositions
    # Wreath composition preset (wreath algorithm only): the ordered garland
    # generator's character. "garland2" = lush mixed-tropical garland — the
    # chosen default: keeps the laurel vine's ordered outward flow + corner
    # medallions but restores species variety, full band depth, and the
    # multi-direction moiré shimmer a single-species rank flattened. "laurel" =
    # dense single-species shingled rank (austere classic); "garland" = looser
    # mixed foliage; "clusters" = spaced rosette swag. Ignored by colonize.
    # See frames/algorithms/wreath.py::STYLES (DEFAULT_STYLE == garland2).
    wreath_style: str = "garland2"
    # MOTIF size dial (wreath only; colonize ignores it). Shrinks the leaves,
    # blooms, understory sprigs and corner sprig — and their spacing along the
    # vine with them, so finer motifs simply come more often and the band stays
    # filled. The band WIDTH and the vine gauge are untouched: use this to make
    # the foliage read finer/lacier without narrowing the frame. 1.0 = the
    # tuned production look (bit-identical to before this knob existed).
    motif_scale: float = 1.0

    def to_frame_params(self) -> FrameParams:
        return FrameParams(
            algorithm=self.algorithm,
            theme_slug=self.theme,
            density=self.density,
            bloom=self.bloom,
            foliage=self.foliage,
            seed=self.seed,
            frame_width_um=self.band_um,
            edge_gradient=self.edge_gradient,
            understory=self.understory,
            border_vine=self.border_vine,
            corner_fans=self.corner_fans,
            wreath_style=self.wreath_style,
            motif_scale=self.motif_scale,
        )


@dataclass
class GlassSpec:
    thickness_um: float = 500.0
    material: str = "fused silica"
    n: float = 1.46


class FaceKind(Enum):
    """WHAT a face is, decided once (``PlateSpec.kind``) and read everywhere.

    Until 2026-09-16 each writer re-derived this from the slug: nine
    ``slug == PHOTO_SLUG`` / ``slug in (BLANK_SLUG, SOLID_SLUG)`` /
    ``single_layer_centerpiece(spec)`` tests across ``plates`` (six),
    ``export_fine`` (two) and ``witness_dies`` (one). Two of them drifted in the
    same week — the fab SVG drew a carrier the fine bake did not — which is the
    failure mode a recomputed classification has. The enum is computed from the
    same three inputs those tests read (slug, ``single_ply``, whether the slug
    registered a ``region_art`` map), once, and is stamped into the plate
    manifest's ``recipe_data`` so a consumer can see which branch wrote it.

    The five kinds, in the order the classification tests them (the first match
    wins, which is also the order every writer used to test them in):

    ``BLANK``    bare quartz: no frame, no carrier, no centrepiece, either layer.
    ``SOLID``    one unbroken sheet of gold over the whole ply, no rim.
    ``PHOTO``    the centrepiece is a LINE SCREEN (no silhouette, front only).
    ``REGION``   a SINGLE-PLY face whose slug registers a ``region_art`` map:
                 the centrepiece is regions of gratings, colour by period.
    ``TWO_PLY``  the silhouette-and-carrier construction — a centrepiece
                 ``_centerpiece_masks`` fills with the centre carrier, and a
                 back layer to fill. Every PRODUCTION face is one of the first
                 four; this kind is reached only by the two hidden exemplars
                 (``SWITCH_INTERLACE_SLUGS`` the barrier switch,
                 ``SHIMMER_MOIRE_SLUGS`` the shading moiré) and by a
                 hypothetical two-ply composition of a region slug, which the
                 same branch already handled by emitting nothing.

    WHICH two-ply exemplar a ``TWO_PLY`` face is stays a slug-set test
    (``SWITCH_INTERLACE_SLUGS`` vs ``SHIMMER_MOIRE_SLUGS``): that split changes
    a grating's period and phase inside one construction rather than choosing
    the construction, and folding it in here would make the enum's two halves
    mean different things.
    """

    BLANK = "blank"
    SOLID = "solid"
    PHOTO = "photo"
    REGION = "region"
    TWO_PLY = "two_ply"



# BARE GLASS. A face whose slug is this gets NO frame, NO carrier and NO
# centerpiece on either layer — the production box's back and bottom are plain
# quartz on purpose (see patterns/blank.py). Every writer tests the slug once:
# the composed preview (``_raster_compose_plate``), the fab SVG (``_bake_plate_svg``)
# and the fine GDS (``export_fine.build_plate_fine``, via an all-empty ZoneMasks).
BLANK_SLUG = "blank"
# SOLID GOLD. The base plate: one unbroken sheet of gold over the whole outer
# ply, no frame, no centrepiece, no rim (the gold runs under the foil). Every
# writer tests it once, like BLANK_SLUG; on the CLEAR production plate it is a
# die with no openings. See patterns/solid.py.
SOLID_SLUG = "solid-gold"
# PHOTOGRAPH. The centerpiece is not a silhouette but a LINE SCREEN: horizontal
# bands whose height tracks the picture's coverage, front layer only, with the
# coloured bands carrying a diffraction sub-grating. ``_centerpiece_masks``
# returns None for it (there is no silhouette to place); ``_paste_centerpiece``
# stamps the bands, and ``photo_band_rects`` is the exact vector form both fab
# writers bake. See patterns/bitmap/photo.py for the tone model.
PHOTO_SLUG = "photo-halftone"


def face_kind(spec: "PlateSpec") -> FaceKind:
    """Classify a plate spec — the ONE place the five constructions are told
    apart. See :class:`FaceKind`; ``PlateSpec.kind`` caches this per spec."""
    slug = getattr(spec, "pattern_slug", "")
    if slug == BLANK_SLUG:
        return FaceKind.BLANK
    if slug == SOLID_SLUG:
        return FaceKind.SOLID
    if slug == PHOTO_SLUG:
        return FaceKind.PHOTO
    if _RA.single_layer_centerpiece(spec):
        return FaceKind.REGION
    return FaceKind.TWO_PLY


@dataclass
class PlateSpec:
    """User-level recipe for one glass plate."""

    pattern_slug: str
    pattern_params: dict[str, Any] = field(default_factory=dict)
    frame: FrameSpec = field(default_factory=FrameSpec)
    glass: GlassSpec = field(default_factory=GlassSpec)
    width_um: float = 30000.0
    height_um: float = 30000.0
    # Blank rim around the plate edges reserved for assembly welds — no
    # FRONT gold (foliage art) patterned anywhere inside this border. Default
    # 1 mm matches a typical UV-cure / solder joint allowance. Equals the box
    # keep-out (foil overlap + safety) when stamped from a box.
    weld_margin_um: float = 1000.0
    # Blank rim for the BACK carrier grating — foil overlap ONLY, so the
    # uniform moiré carrier spans the whole *exposed* face (wider than the
    # front weld margin). Mirrors assembly.back_window_um; stamped from the
    # box. ``None`` (standalone plate) falls back to weld_margin_um.
    back_margin_um: float | None = None
    # User-facing GRATING PITCH (μm) of the REAL fabricated back carrier + leaf
    # louvre family. Drives ``_carrier_recipe_data`` (fab back period = this;
    # front louvre = this × FRONT_GRATING_RATIO) and, through it, both the
    # preview shader uniforms and the fab SVG/GDS bake. Default 22 µm (the
    # historical BACK_CARRIER_PERIOD_UM). Litho floor 4 µm (2 µm line + 2 µm
    # gap at 50% duty). The switch/comb barrier faces keep their own 60 µm
    # architecture — this field does NOT touch them. Stamped from the box level
    # (see boxes.normalize_face_dims); part of the plate hash so a pitch change
    # regenerates the cached plate.
    carrier_pitch_um: float = 22.0
    # PER-FACE carrier scaling policy against the plate's real glass:
    #   "gap"   (default) — the fabricated carrier family scales with the
    #           paraxial gap t/n (see parallax_period_scale), preserving the
    #           designed reveal/beat TILT behavior on any stock. A no-op at
    #           the 500 µm fused-silica baseline (scale = 1), so every
    #           existing plate keeps byte-identical geometry.
    #   "fixed" — keep carrier_pitch_um as the literal fabricated pitch. On
    #           thick stock the reveals compress into sub-degree "refraction
    #           shimmer" — a deliberate aesthetic choice per face.
    # The switch/scanimation barrier periods scale with the gap regardless
    # (fab_center_period_um) — their crossing angle is the effect.
    carrier_scale_mode: str = "gap"
    # SINGLE-PLY face: this side of the box is ONE sheet of glass, not a bonded
    # pair, so there is no inner ply to write a back layer on. Everything moves
    # to the FRONT (outer) layer — the frame leaves AND the uniform carrier
    # grating, the carrier filling the back window MINUS the centerpiece art box
    # — and the BACK layer is left empty. The garland's shimmer then comes from
    # each leaf's grating beating against the carrier IN THE PLANE (the two-ply
    # moiré at zero gap: static fringes rather than travelling ones), which is
    # the honest thing a single ply can do. Reference geometry: the single-ply
    # block in ``export_fine.build_plate_fine``, which is what the production
    # witness dies (and every other single-ply face) are actually written from.
    single_ply: bool = False
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_slug": self.pattern_slug,
            "pattern_params": dict(self.pattern_params),
            "frame": asdict(self.frame),
            "glass": asdict(self.glass),
            "width_um": self.width_um,
            "height_um": self.height_um,
            "weld_margin_um": self.weld_margin_um,
            "back_margin_um": self.back_margin_um,
            "carrier_pitch_um": self.carrier_pitch_um,
            "carrier_scale_mode": self.carrier_scale_mode,
            "single_ply": self.single_ply,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlateSpec":
        frame_data = data.get("frame", {}) or {}
        glass_data = data.get("glass", {}) or {}
        return cls(
            pattern_slug=data["pattern_slug"],
            pattern_params=dict(data.get("pattern_params", {})),
            frame=FrameSpec(**frame_data),
            glass=GlassSpec(**glass_data),
            width_um=float(data.get("width_um", 30000.0)),
            height_um=float(data.get("height_um", 30000.0)),
            weld_margin_um=float(data.get("weld_margin_um", 1000.0)),
            back_margin_um=(
                float(data["back_margin_um"])
                if data.get("back_margin_um") is not None
                else None
            ),
            carrier_pitch_um=float(data.get("carrier_pitch_um", 22.0)),
            carrier_scale_mode=str(data.get("carrier_scale_mode", "gap")),
            single_ply=bool(data.get("single_ply", False)),
            label=data.get("label", ""),
        )

    @cached_property
    def kind(self) -> FaceKind:
        """Which of the five constructions this face is (:class:`FaceKind`).

        Cached because every writer asks and the REGION test imports the
        pattern package to see whether the slug registered a region map. Safe
        to cache: it depends only on ``pattern_slug`` and ``single_ply``, and
        nothing mutates those after construction — ``normalize_face_dims``
        stamps dims/glass/pitch, and the one place that swaps a photograph
        (``witness_dies.side_photo_spec``) builds a NEW spec with
        ``dataclasses.replace``. It is deliberately NOT a dataclass field: it
        is derived, so it stays out of ``to_dict`` and therefore out of
        ``plate_hash``.
        """
        return face_kind(self)

    def active_dims(self) -> tuple[float, float]:
        """(W, H) of the FRONT foliage area inside the weld border, in μm."""
        return (
            max(0.0, self.width_um - 2.0 * self.weld_margin_um),
            max(0.0, self.height_um - 2.0 * self.weld_margin_um),
        )

    def back_dims(self) -> tuple[float, float]:
        """(W, H) of the BACK carrier window inside the foil-overlap rim, in μm.

        Wider than ``active_dims`` — the back carrier grating spans the whole
        exposed face (plate minus foil overlap only). Falls back to the weld
        margin for a standalone plate that was never stamped from a box.
        """
        m = self.weld_margin_um if self.back_margin_um is None else self.back_margin_um
        return (
            max(0.0, self.width_um - 2.0 * m),
            max(0.0, self.height_um - 2.0 * m),
        )


def plate_hash(spec: PlateSpec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _band_um(spec: PlateSpec) -> float:
    if spec.frame.band_um is not None:
        return spec.frame.band_um
    aw, ah = spec.active_dims()
    return 0.12 * max(0.0, min(aw, ah))



def _aperture(spec: PlateSpec) -> float:
    """Side length (μm) of the square central aperture inside the frame band.

    Sits inside *both* the weld margin and the frame band, so the central
    optical pattern has clean glass on every side.
    """
    band = _band_um(spec)
    aw, ah = spec.active_dims()
    return max(0.0, min(aw, ah) - 2.0 * band)
