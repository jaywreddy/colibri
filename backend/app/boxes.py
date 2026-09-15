"""Ring box model — six glass plates assembled stained-glass style.

A ``BoxSpec`` (v2) is the user-level recipe for a complete ring box: outer
``(W, D, H)`` dimensions plus box-level glass, copper-foil, and hinge specs.
Each face is its own ``PlateSpec`` carrying the central pattern + frame, but
its cut dimensions, glass, and keep-out rim are *stamped* from the box level
by ``normalize_face_dims`` — they are not independent degrees of freedom.

Construction (see assembly.py for the full math):
  * Body = bottom + 4 walls, copper-foiled and fully soldered (8 seams).
  * Lid = flat top plate on a brass tube-and-rod hinge along the back top
    edge; its rim is tinned foil only, never soldered.
  * The foil fold-over + safety margin becomes each face's ``weld_margin_um``
    (the existing blank-rim mechanism in plates.py), keeping gold clear of
    the copper tape.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .assembly import (
    FACE_IDS,
    FoilSpec,
    HingeSpec,
    assembly_summary,
    back_window_um,
    bonded_assembly_summary,
    bonded_art_keepout_um,
    bonded_back_window_um,
    face_cut_dims,
    keepout_um,
    validate_assembly,
    validate_bonded_assembly,
)
from .plates import FrameSpec, GlassSpec, PlateSpec, materialize_plate
from .service import DATA_ROOT, cache_lock, read_json_cache, write_json_atomic


_log = logging.getLogger("optics.boxes")

BOXES_ROOT = DATA_ROOT / "boxes"

# --- the PRODUCTION box ------------------------------------------------------
# Four written faces and two of bare glass. That is a decision, not an
# omission: the lid's monogram and the front's globe are SINGLE-LAYER
# DIFFRACTION mappings (region_art: colour by region, one written ply), the
# two sides are PHOTOGRAPHS on single plies, the bottom is a SOLID GOLD base
# plate (patterns/solid.py) and the back is left as quartz so the piece has
# somewhere to be quiet. See patterns/blank.py.
#
# 2026-09-15: every face is ONE written ply. The first plate's bonded moiré
# pairs (monogram shading moiré, globe barrier switch) read badly on glass and
# the pair could not be cleaved, so the two-ply effects are gone from the box;
# the inner plies are bare quartz for wall thickness only.
DEFAULT_FACE_PATTERN_SLUG = "globe-atlantic"       # front: the Atlantic globe, colour by region
LID_PATTERN_SLUG = "monogram-jp"                   # top
BOTTOM_PATTERN_SLUG = "solid-gold"                 # bottom: a solid gold base plate (2026-09-15)
BACK_PATTERN_SLUG = "photo-halftone"               # back: a third photograph (2026-09-15)
                                             # (capybara scanimation cut 2026-09: its 15 um
                                             # slots are far below the 1.5 mm near-field limit)
LEFT_PATTERN_SLUG = "photo-halftone"          # left: the beach photograph, faces coloured
RIGHT_PATTERN_SLUG = "photo-halftone"         # right: the sunset photograph, plain gold

# Per-face default centerpiece slug (the production plan).
_FACE_PATTERN_SLUG: dict[str, str] = {
    "front": DEFAULT_FACE_PATTERN_SLUG,
    "back": BACK_PATTERN_SLUG,
    "left": LEFT_PATTERN_SLUG,
    "right": RIGHT_PATTERN_SLUG,
    "top": LID_PATTERN_SLUG,
    "bottom": BOTTOM_PATTERN_SLUG,
}

# Per-face pattern params. Both sides run the same generator on different
# photographs: the beach group takes colour on the FACES (the only saturated
# thing in it, so the sea and sky stay gold), the sunset stays plain because its
# whole subject is one warm gradient and a hue ladder over it would read as
# banding rather than as colour.
_FACE_PATTERN_PARAMS: dict[str, dict[str, Any]] = {
    "left": {"image": "beach", "colour_mode": "authored"},
    "right": {"image": "sunset", "colour_mode": "authored"},
    "back": {"image": "paris", "colour_mode": "authored"},
}

# Faces built from ONE ply instead of a bonded pair. A photograph is a
# single-layer effect — its tone is the height of its bands — so a second ply
# under it would earn nothing and its gold would show through the gaps and lift
# every shadow. The garland on these faces therefore carries BOTH gratings on
# the outer ply (leaves + carrier), and its shimmer is the in-plane beat rather
# than a parallax one. See PlateSpec.single_ply.
_SINGLE_PLY_FACES: frozenset[str] = frozenset(FACE_IDS)

# Motif size dial for every face's frame. 0.68 makes the foliage read finer and
# lacier without narrowing the band — the leaves simply come more often — and
# keeps a flower under a millimetre inside the 2.4 mm band. witness_dies reads
# this, so the production plate is written at the same value.
PRODUCTION_MOTIF_SCALE = 0.68

# One garland band width for all six faces (µm). The default (12% of the
# shorter side) gave the lid a 3.5 mm band and the walls 2.95 mm; with the art
# rim now starting at the inner ply's window (assembly.bonded_art_keepout_um)
# a fixed 2.4 mm keeps the lid's art box at 17 mm and the photo walls' at
# 12.5 mm.
PRODUCTION_BAND_UM = 2400.0

# 3/8" copper foil: the only stock preset that wraps the 6.75 mm stepped edge of
# a 2.25 mm bonded stack with a fold left over (1.39 mm per face). 1/4" tape is
# 0.4 mm short of the edge and validate_bonded_assembly rejects it.
PRODUCTION_TAPE_UM = 9525.0

# The moire carrier every face's garland and the lid's monogram beat against,
# in micrometres AS FABRICATED: witness_geom sizes it so the period subtends
# 0.75 arcmin at 300 mm (65.5 um) — the lines are invisible in hand, only the
# beat shows. Faces run carrier_scale_mode "fixed" so this is the literal pitch
# (the "gap" mode multiplied the 22 um design pitch by the glass's parallax
# ratio and landed at 99 um, a hatch the eye resolves).
from .witness_geom import BOX_CARRIER_UM as PRODUCTION_CARRIER_UM  # noqa: E402

# The production stock: 2.25 mm fused quartz plies, bonded face-to-face, so the
# wall is 4.5 mm and the optical parallax gap is t/n = 1543 um.
PRODUCTION_PLY_UM = 2250.0
PRODUCTION_GLASS_MATERIAL = "fused quartz"
PRODUCTION_GLASS_N = 1.4585
# The RING sizes the box. Assumed envelope (Jay to confirm against the ring):
# a US 6-7 band, outer diameter up to 21 mm, standing 24 mm tall with its head
# up, in a slot in a 1 mm liner. The clear interior between the 4.5 mm bonded
# walls must hold OD + two liners across and the standing height + a 2 mm
# base pad; the plate solve used to size the box instead (the largest box
# whose twelve plies fit one blank came out 29.1 mm, a 20.1 mm interior that
# no adult ring stands in).
RING_OD_UM = 21000.0
RING_STANDING_UM = 24000.0
RING_LINER_UM = 1000.0
RING_BASE_PAD_UM = 2000.0


def ring_interior_um() -> tuple[float, float, float]:
    """(width, depth, height) of clear interior the ring envelope needs."""
    span = RING_OD_UM + 2.0 * RING_LINER_UM
    return span, span, RING_STANDING_UM + RING_BASE_PAD_UM


def ring_fit(width_um: float, depth_um: float, height_um: float, ply_um: float) -> dict[str, float | bool]:
    """Clearances (um) of the ring envelope inside the bonded box: positive is
    room to spare, negative is a ring that does not go in."""
    wall = 2.0 * ply_um
    iw, id_, ih = width_um - 2 * wall, depth_um - 2 * wall, height_um - 2 * wall
    nw, nd, nh = ring_interior_um()
    return {
        "interior_um": [iw, id_, ih],
        "needed_um": [nw, nd, nh],
        "clearance_um": [iw - nw, id_ - nd, ih - nh],
        "fits": bool(iw >= nw and id_ >= nd and ih >= nh),
    }


# Outer box dimensions (um), rounded up to 0.5 mm from the ring envelope:
# 23 x 23 x 26 mm inside 4.5 mm walls -> 32 x 32 x 35 mm. Square in plan; the
# height carries the lid. The frontend's defaultBoxSpec mirrors these numbers.
PRODUCTION_WIDTH_UM = 32000.0
PRODUCTION_DEPTH_UM = 32000.0
PRODUCTION_HEIGHT_UM = 35000.0

# Per-face frame recipe. Each face gets a distinct seed (which the plate
# compositor turns into a distinct moiré CARRIER ANGLE via (seed*17)%180) plus
# a distinct band-composition profile, so every side reads as its own
# deliberately engraved border even before the user assigns per-face themes.
# (fid -> dict of FrameSpec overrides). Seeds 100..105 kept from the original
# default so cached faces that only differ in seed still line up.
_FACE_FRAME_PROFILE: dict[str, dict[str, float]] = {
    #        seed  edge_gradient understory border_vine corner_fans
    "front":  dict(seed=100, edge_gradient=0.75, understory=0.9,  border_vine=1.2,  corner_fans=1.1),
    "back":   dict(seed=101, edge_gradient=1.0,  understory=0.6,  border_vine=0.9,  corner_fans=0.8),
    "top":    dict(seed=102, edge_gradient=0.55, understory=1.05, border_vine=1.35, corner_fans=1.25),
    "bottom": dict(seed=103, edge_gradient=0.9,  understory=0.75, border_vine=1.0,  corner_fans=0.9),
    "left":   dict(seed=104, edge_gradient=0.65, understory=1.0,  border_vine=1.25, corner_fans=1.15),
    "right":  dict(seed=105, edge_gradient=0.85, understory=0.8,  border_vine=1.05, corner_fans=0.95),
}

# Anonymous (live-preview) generates all land in this single scratch slot so
# debounced edits never pile up saved boxes.
SCRATCH_BOX_ID = "__scratch"

# Filesystem-safe box ids: 1-64 chars, letters/digits/._- with a first char
# that can't start a traversal ('..' and absolute/illegal paths are rejected)
# and can't be '_' — that namespace is RESERVED for internal slots. Without
# that last rule a client could POST box_id="__scratch" and land a named
# preset in the slot the next live-preview regen overwrites, silently losing
# it from the dropdown; nothing but UI convention kept it out of reach.
_BOX_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# Internal slots: same charset behind the reserved leading underscore. Only
# READ paths accept these — the scratch box has to stay addressable because the
# live-preview export link is /export/box/<manifest id>/fab.zip.
_INTERNAL_BOX_ID_RE = re.compile(r"_[A-Za-z0-9_][A-Za-z0-9._-]{0,62}")


def _box_dir(box_id: str, *, internal: bool = False) -> Path | None:
    """``BOXES_ROOT/<box_id>`` if the id is filesystem-safe, else ``None``.

    ``box_id`` arrives verbatim from the API; without this check a name like
    ``'../../evil'`` would read/write outside ``data/boxes``. ``internal=True``
    additionally admits the reserved ``_``-prefixed slots (see
    ``SCRATCH_BOX_ID``) — read-only callers pass it, write/delete callers must
    not.
    """
    if _BOX_ID_RE.fullmatch(box_id):
        return BOXES_ROOT / box_id
    if internal and _INTERNAL_BOX_ID_RE.fullmatch(box_id):
        return BOXES_ROOT / box_id
    return None


@dataclass
class BoxSpec:
    """A full ring box: 6 plate specs + box-level glass/foil/hinge.

    Glass, foil keep-out, and per-face cut dims are box-level constraints —
    every face shares them so the assembly registers correctly. The keep-out
    (foil overlap + safety) collapses onto each face's ``weld_margin_um``
    in ``normalize_face_dims`` before materialize, so the per-face cache key
    reflects the actual blank rim.
    """

    width_um: float = 50000.0   # outer X — default 50 mm ring box
    depth_um: float = 50000.0   # outer Z
    height_um: float = 40000.0  # outer Y
    glass: GlassSpec = field(default_factory=GlassSpec)
    foil: FoilSpec = field(default_factory=FoilSpec)
    hinge: HingeSpec = field(default_factory=HingeSpec)
    faces: dict[str, PlateSpec] = field(default_factory=dict)
    # Box-level GRATING PITCH (μm): the fabricated back-carrier + leaf-louvre
    # period, stamped onto every face in ``normalize_face_dims`` (like glass —
    # not an independent per-face degree of freedom). Default 22 µm; litho floor
    # 4 µm. Drives the real fringe spacing + tilt sensitivity of the leaf/back
    # carrier family (the 60 µm switch/comb barrier faces are NOT coupled to it).
    carrier_pitch_um: float = 22.0
    # BONDED (two-ply) construction: each face is TWO single-side plates glued
    # face-to-face — ``glass.thickness_um`` is then the PLY thickness (also the
    # optical parallax gap), the wall is 2x, and cut dims / foil margins follow
    # the nested-shell math (assembly.bonded_*). Default False = the classic
    # single double-side plate.
    bonded: bool = False
    # Litho metal the masks will be written in — a PREVIEW material choice
    # ("gold" | "chrome" | "chrome-ar"): the mask geometry is identical, only
    # the renderer's conductor response (albedo/F0) follows it. "chrome" is
    # standard bright chrome (platinum-line read), "chrome-ar" the
    # low-reflective AR-coated mask grade (ink-black linework).
    metal: str = "gold"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_um": self.width_um,
            "depth_um": self.depth_um,
            "height_um": self.height_um,
            "glass": {
                "thickness_um": self.glass.thickness_um,
                "material": self.glass.material,
                "n": self.glass.n,
            },
            "foil": self.foil.to_dict(),
            "hinge": self.hinge.to_dict(),
            "faces": {fid: p.to_dict() for fid, p in self.faces.items()},
            "carrier_pitch_um": self.carrier_pitch_um,
            "bonded": self.bonded,
            "metal": self.metal,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoxSpec":
        """Tolerant parse: any missing v2 field falls back to its default."""
        glass_data = data.get("glass") or {}
        return cls(
            width_um=float(data.get("width_um", 50000.0)),
            depth_um=float(data.get("depth_um", 50000.0)),
            height_um=float(data.get("height_um", 40000.0)),
            glass=GlassSpec(
                thickness_um=float(glass_data.get("thickness_um", 500.0)),
                material=str(glass_data.get("material", "fused silica")),
                n=float(glass_data.get("n", 1.46)),
            ),
            foil=FoilSpec.from_dict(data.get("foil")),
            hinge=HingeSpec.from_dict(data.get("hinge")),
            faces={fid: PlateSpec.from_dict(p) for fid, p in (data.get("faces") or {}).items()},
            carrier_pitch_um=float(data.get("carrier_pitch_um", 22.0)),
            bonded=bool(data.get("bonded", False)),
            metal=str(data.get("metal", "gold")),
            label=str(data.get("label", "")),
        )

    def normalize_face_dims(self) -> None:
        """Stamp derived cut dims + foil keep-out + box glass onto every face.

        Per-face dims follow the stained-glass cut list (walls sit on the
        bottom; left/right walls fit between front/back), the keep-out rim
        lands in ``weld_margin_um``, and the box glass spec replaces whatever
        the face carried — none of these are face-level choices in a box.
        """
        t = self.glass.thickness_um
        if self.bonded:
            # Two-ply construction: masks are composed in the OUTER ply frame
            # (its cut dims come from the nested outer shell at the PLY
            # thickness). Front art insets by the bonded keep-out from the
            # outer edge; back art lives on the INNER ply, whose edge is
            # already one ply in — so, measured in the shared outer frame, its
            # window insets by ply + the interior foil fold. That guarantees
            # composed back geometry never overhangs the smaller inner plate.
            # Front art starts at the LARGER of the foil keep-out and that
            # back window, so every front feature has the inner ply's carrier
            # behind it (see assembly.bonded_art_keepout_um).
            ko = bonded_art_keepout_um(self.foil, t)
            bw = t + bonded_back_window_um(self.foil, t)
        else:
            ko = keepout_um(self.foil, t)
            bw = back_window_um(self.foil, t)
        for fid in FACE_IDS:
            plate = self.faces.get(fid)
            if plate is None:
                continue
            w, h = face_cut_dims(
                fid, self.width_um, self.depth_um, self.height_um, t
            )
            plate.width_um = w
            plate.height_um = h
            plate.weld_margin_um = ko
            # Back carrier grating covers the whole exposed face (foil overlap
            # only) — wider window than the front weld margin.
            plate.back_margin_um = bw
            plate.glass = replace(self.glass)
            # Grating pitch is a box-level choice (like glass): stamp it onto
            # every face so the per-face plate hash + recipe_data reflect it.
            plate.carrier_pitch_um = self.carrier_pitch_um


def default_box_spec() -> BoxSpec:
    """The PRODUCTION ring box — MUST match the frontend's ``defaultBoxSpec()``.

    32 x 32 x 35 mm (sized by the ring, see RING_*), BONDED from 2.25 mm fused-quartz plies, 3/8" foil,
    5-segment tube hinge. Every written face gets a perimeter foliage FRAME with
    its own seed + band-composition profile (so each side is a visibly distinct
    engraved border) at ``motif_scale`` 0.68 in a 2.4 mm band, around a centerpiece:

      * TOP (lid) → the interlocked cursive J+P monogram (``monogram-jp``) as a
        single-layer diffraction mapping: each letter its own grating period,
        so the two flash different colours — the engagement engraving.
      * FRONT → the Atlantic globe (``globe-atlantic``, real Natural Earth
        geography): one orthographic view holding the US with California,
        Colombia and Europe, land/countries coloured by region.
      * LEFT → the beach photograph as a gold line screen (``photo-halftone``)
        with its authored colour plan.
      * RIGHT → the sunset photograph, same screen, its own colour plan.
      * BACK → the Paris photograph, same screen, its own colour plan.
      * BOTTOM → ``solid-gold``: an unbroken gold base plate under the ring.

    EVERY face is single-ply (``_SINGLE_PLY_FACES``): the box is still built
    from bonded 2.25 mm pairs for its wall thickness, but only the OUTER ply of
    each pair is written; the inner one is bare quartz. A single ply carries
    no carrier and no moiré — its effects are the photograph's tone and the
    spectral colour of fine gratings (``region_art``, ``leaf_fills``).
    """
    spec = BoxSpec(
        width_um=PRODUCTION_WIDTH_UM,
        depth_um=PRODUCTION_DEPTH_UM,
        height_um=PRODUCTION_HEIGHT_UM,
        glass=GlassSpec(
            thickness_um=PRODUCTION_PLY_UM,
            material=PRODUCTION_GLASS_MATERIAL,
            n=PRODUCTION_GLASS_N,
        ),
        foil=FoilSpec(tape_width_um=PRODUCTION_TAPE_UM),
        bonded=True,
    )
    for fid in FACE_IDS:
        profile = _FACE_FRAME_PROFILE.get(fid, {})
        slug = _FACE_PATTERN_SLUG.get(fid, DEFAULT_FACE_PATTERN_SLUG)
        spec.faces[fid] = PlateSpec(
            pattern_slug=slug,
            pattern_params=dict(_FACE_PATTERN_PARAMS.get(fid, {})),
            frame=FrameSpec(**profile, motif_scale=PRODUCTION_MOTIF_SCALE,
                            band_um=PRODUCTION_BAND_UM),
            single_ply=fid in _SINGLE_PLY_FACES,
            carrier_scale_mode="fixed",
        )
    spec.carrier_pitch_um = PRODUCTION_CARRIER_UM
    spec.normalize_face_dims()
    fit = ring_fit(spec.width_um, spec.depth_um, spec.height_um, PRODUCTION_PLY_UM)
    if not fit["fits"]:
        raise ValueError(f"production box does not hold the ring envelope: {fit}")
    return spec


def box_hash(spec: BoxSpec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _lean_face_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy a plate manifest and drop ``recipe_data['frame_scene']``.

    The frame scene is megabytes of segment dicts the box-level consumers
    never need. New plate manifests keep it in a ``scene.json`` sidecar
    (see plates.py), but old cached manifests may still embed it — strip
    defensively. A shallow copy suffices: ``materialize_plate`` returns a
    fresh ``json.loads`` per call and ``materialize_box`` only serializes the
    result (the old ``json.loads(json.dumps(...))`` round-trip was ~60% of a
    warm 6-face box regen).
    """
    lean = dict(manifest)
    rd = dict(manifest.get("recipe_data") or {})
    rd.pop("frame_scene", None)
    lean["recipe_data"] = rd
    return lean


def materialize_box(spec: BoxSpec, *, box_id: str | None = None, force: bool = False) -> dict[str, Any]:
    """Validate, fan out per face, and assemble the v2 box manifest.

    ``box_id`` is the persistent identifier for the *saved preset* (user-named
    box). Anonymous generates (live-preview regens with no id) all reuse the
    single ``SCRATCH_BOX_ID`` slot — they overwrite each other, never appear
    in ``list_boxes``, and can't grow ``data/boxes`` without bound. A caller
    may NOT name that slot (or any other reserved ``_`` id) explicitly: the
    next anonymous regen would overwrite the preset with ``saved: false``.
    Per-face plates are cached by their content hash under ``data/plates/`` and
    shared across boxes — so swapping one face's frame seed re-renders only
    that face.
    """
    box_id = box_id or None  # treat "" like absent
    saved = box_id is not None
    if box_id is not None and _box_dir(box_id) is None:
        if box_id.startswith("_"):
            raise ValueError(
                f"Reserved box_id {box_id!r}: ids starting with '_' belong to internal "
                "slots (the live-preview scratch box, which every anonymous regenerate "
                "overwrites). Save the preset under another name."
            )
        raise ValueError(
            f"Invalid box_id {box_id!r}: use 1-64 letters, digits, '.', '_' or '-' "
            "(must not start with '.', '-' or '_')."
        )

    spec.normalize_face_dims()
    if spec.bonded:
        validate_bonded_assembly(
            spec.width_um, spec.depth_um, spec.height_um,
            spec.glass.thickness_um, spec.foil, spec.hinge,
        )
    else:
        validate_assembly(
            spec.width_um, spec.depth_um, spec.height_um,
            spec.glass.thickness_um, spec.foil, spec.hinge,
        )

    BOXES_ROOT.mkdir(parents=True, exist_ok=True)
    bid = box_id or SCRATCH_BOX_ID
    # Serialize per box id (CLAUDE.md cache contract): two overlapping
    # live-preview regens of the scratch slot must not interleave their writes.
    with cache_lock(f"box:{bid}"):
        return _materialize_box_locked(spec, bid, force, saved)


def _materialize_box_locked(spec: BoxSpec, bid: str, force: bool, saved: bool) -> dict[str, Any]:
    out = BOXES_ROOT / bid
    out.mkdir(parents=True, exist_ok=True)
    box_path = out / "box.json"

    t0 = time.perf_counter()
    face_manifests: dict[str, dict[str, Any]] = {}
    for fid in FACE_IDS:
        plate_spec = spec.faces.get(fid)
        if plate_spec is None:
            continue
        face_manifests[fid] = _lean_face_manifest(materialize_plate(plate_spec, force=force))

    box_manifest = {
        "kind": "box",
        "id": bid,
        # Anonymous live-preview boxes are not presets — list_boxes skips them.
        "saved": saved,
        "spec": spec.to_dict(),
        "name": spec.label or f"Box {bid}",
        "faces": face_manifests,
        "dimensions_um": {
            "width": spec.width_um,
            "height": spec.height_um,
            "depth": spec.depth_um,
        },
        "assembly": (
            bonded_assembly_summary(
                spec.width_um, spec.depth_um, spec.height_um,
                spec.glass.thickness_um, spec.foil, spec.hinge,
            )
            if spec.bonded
            else assembly_summary(
                spec.width_um, spec.depth_um, spec.height_um,
                spec.glass.thickness_um, spec.foil, spec.hinge,
            )
        ),
        "content_hash": box_hash(spec),
    }
    # payload files (the faces) are already published; the manifest lands last,
    # atomically, so a reader never sees a half-written box.json
    write_json_atomic(box_path, box_manifest)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _log.info("materialize_box id=%s faces=%d %dms", bid, len(face_manifests), dt_ms)
    return box_manifest


def list_boxes() -> list[dict[str, Any]]:
    """All saved boxes, skipping corrupt or old-format (pre-v2) box.json files.

    Anonymous live-preview manifests (``saved: false``, the scratch slot) are
    excluded — only explicitly saved presets belong in the dropdown.
    """
    if not BOXES_ROOT.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(BOXES_ROOT.iterdir()):
        m = d / "box.json"
        if m.exists():
            try:
                manifest = json.loads(m.read_text())
                # v2 manifests always carry the derived assembly block —
                # anything without it predates the ring-box redesign.
                if manifest.get("kind") != "box" or "assembly" not in manifest:
                    _log.warning("list_boxes skipping old-format box.json: %s", m)
                    continue
                if manifest.get("saved") is False:
                    continue  # scratch / anonymous regen — not a preset
                out.append(manifest)
            except Exception:  # noqa: BLE001 — skip corrupt
                continue
    return out


def get_box(box_id: str) -> dict[str, Any] | None:
    """Read one box manifest. Reserved internal ids resolve — the unsaved
    live-preview box is exported by its ``SCRATCH_BOX_ID`` manifest id."""
    d = _box_dir(box_id, internal=True)
    if d is None:
        return None
    m = d / "box.json"
    if not m.exists():
        return None
    # a truncated or corrupt manifest is a MISS (None), not a 500
    return read_json_cache(m)


def delete_box(box_id: str) -> bool:
    """Remove a saved preset. Reserved internal slots are not deletable (the
    scratch manifest belongs to the live preview, not to the preset list)."""
    d = _box_dir(box_id)
    if d is None:
        return False
    box_path = d / "box.json"
    if not box_path.exists():
        return False
    box_path.unlink()
    # Leave the data dir + per-face plate caches; just remove the saved preset.
    try:
        d.rmdir()
    except OSError:
        pass
    return True
