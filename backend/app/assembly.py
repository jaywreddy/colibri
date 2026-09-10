"""Assembly math for the stained-glass ring box — single source of truth.

Construction technique (copper-foil / "Tiffany" method, adapted to fused
silica): each of the 6 plates gets adhesive copper foil tape wrapped around
its edges. The tape is wider than the glass is thick, so the excess folds
over onto both faces of the plate — that fold-over is the *overlap*:

    overlap_um = max(0, (tape_width_um - glass_thickness_um) / 2)

The folded foil hides a rim of glass on every plate face, so the gold
lithography must keep clear of it. We add a safety margin on top (foil is
hand-applied; alignment is ±a few hundred µm) and stamp the total into each
face's ``weld_margin_um`` — the existing blank-rim mechanism in plates.py:

    keepout_um = overlap_um + safety_um

Body assembly: walls sit ON the bottom plate, the lid rests on the wall rim.
The bottom + 4 walls are tacked then fully soldered along 8 seams (4 around
the bottom perimeter, 4 vertical corners). Solder only wets the copper foil,
so the bead geometry follows the foiled joint lines exactly.

The lid is NOT soldered. Its rim (and the top rim of the walls) is tinned —
a thin wipe of solder over the foil for finish and corrosion resistance.
The lid attaches with a brass tube-and-rod hinge along the BACK top edge,
cigar-box style: the tube is cut into an odd number of segments alternating
body, lid, body, …(both ends body), threaded on a brass rod.

Everything here is pure geometry/validation — no I/O, no rendering. The
frontend mirrors these formulas in ``src/assembly.ts``; keep them in sync.
All values are micrometers unless suffixed otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


FACE_IDS = ("front", "back", "top", "bottom", "left", "right")

# Gap between adjacent hinge tube segments so they can rotate freely.
HINGE_SEGMENT_GAP_UM = 400.0

# Minimum patternable aperture (per side) a plate must retain after the
# foil keep-out rim is subtracted — below this there is nothing left to
# lithograph and the box is rejected at validation.
MIN_APERTURE_UM = 3000.0

# Solder finish -> preview render color (hex). "bright" is freshly flowed
# tin-lead/lead-free, "copper" leaves the foil bare, "patina" is the black
# sulfide treatment common on stained-glass work.
FINISH_COLORS = {
    "bright": "#c9ced6",
    "copper": "#b87333",
    "patina": "#34343a",
    "gold": "#e3b53b",
    "rose": "#c98a86",
    "gunmetal": "#3a3f47",
}

# Common copper foil tape widths (µm): 3/16", 7/32", 1/4", 5/16", 3/8".
# A BONDED stack wraps a 3-ply stepped edge, so 2.25 mm quartz plies (6.75 mm
# of edge) need the 3/8" tape before any fold-over remains.
FOIL_TAPE_PRESETS_UM = (4763.0, 5556.0, 6350.0, 7938.0, 9525.0)


@dataclass
class FoilSpec:
    """Copper foil tape parameters for the stained-glass seams."""

    tape_width_um: float = 6350.0  # 1/4" tape
    safety_um: float = 500.0       # extra keep-out beyond the foil overlap
    bead_um: float = 2000.0        # solder bead DIAMETER for the preview render
    finish: str = "bright"         # "bright" | "copper" | "patina"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tape_width_um": self.tape_width_um,
            "safety_um": self.safety_um,
            "bead_um": self.bead_um,
            "finish": self.finish,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FoilSpec":
        data = data or {}
        return cls(
            tape_width_um=float(data.get("tape_width_um", 6350.0)),
            safety_um=float(data.get("safety_um", 500.0)),
            bead_um=float(data.get("bead_um", 2000.0)),
            finish=str(data.get("finish", "bright")),
        )


@dataclass
class HingeSpec:
    """Brass tube-and-rod hinge along the back top edge (cigar-box style)."""

    style: str = "tube"
    tube_od_um: float = 2400.0  # 3/32" brass tube outer diameter
    rod_od_um: float = 1600.0   # 1/16" rod
    segments: int = 5           # odd, >= 3; alternating body,lid,body,...
    coverage: float = 0.8       # fraction of box width spanned by the tube run

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "tube_od_um": self.tube_od_um,
            "rod_od_um": self.rod_od_um,
            "segments": self.segments,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HingeSpec":
        data = data or {}
        return cls(
            style=str(data.get("style", "tube")),
            tube_od_um=float(data.get("tube_od_um", 2400.0)),
            rod_od_um=float(data.get("rod_od_um", 1600.0)),
            segments=int(data.get("segments", 5)),
            coverage=float(data.get("coverage", 0.8)),
        )


def round_mm3(value_um: float) -> float:
    """Micrometers -> millimeters rounded to 3 decimals, HALF AWAY FROM ZERO.

    Shared contract helper: ``frontend/src/assembly.ts::roundMm3`` mirrors this
    formula term for term. Neither side may use its language's built-in
    rounding — Python's ``round()`` is half-to-EVEN while JavaScript's
    ``Math.round()`` is half-UP (and asymmetric for negatives), so the two
    disagree on every dimension whose mm value lands exactly on a
    half-thousandth: 24062.5 um is 24.062 mm to Python and 24.063 mm to JS,
    and the manifest cut list then contradicts the BuildPanel readout.

    Because 1 um is exactly 0.001 mm we round the MICROMETER value to an
    integer and divide once. That is deliberate: dividing first and
    re-multiplying (``(um / 1000) * 1000``) reintroduces the quotient's
    representation error and can floor a whole micrometer away (17.4 mm ->
    17399.999999999998 -> 17.399). One exact-integer division also lands on
    the nearest double to n/1000 in both languages, so the fixture's
    ``toBe(width_mm)`` equality holds bit for bit.
    """
    if value_um < 0.0:
        return -(math.floor(-value_um + 0.5) / 1000.0)
    return math.floor(value_um + 0.5) / 1000.0


def overlap_um(foil: FoilSpec, glass_thickness_um: float) -> float:
    """Foil fold-over width onto each plate face.

    The tape wraps the plate edge; whatever width isn't consumed by the
    glass thickness splits evenly between the two faces.
    """
    return max(0.0, (foil.tape_width_um - glass_thickness_um) / 2.0)


def keepout_um(foil: FoilSpec, glass_thickness_um: float) -> float:
    """Pattern keep-out rim per plate edge: foil overlap + safety margin.

    This is stamped into every face's ``weld_margin_um`` so the gold mask
    never extends under (or hazardously close to) the copper foil. It bounds
    the FRONT artwork (foliage silhouette) — decorative gold must not creep
    under or hazardously near the hand-applied copper tape.
    """
    return overlap_um(foil, glass_thickness_um) + foil.safety_um


def back_window_um(foil: FoilSpec, glass_thickness_um: float) -> float:
    """Back-carrier keep-out rim per plate edge: foil overlap ONLY.

    The back layer is a uniform moiré-carrier grating, not decorative art. It
    should cover the whole *exposed* face — everything the folded copper foil
    does not physically hide — so the moiré shimmer reads edge-to-edge behind
    the front foliage. The only hard constraint is that the grating must not
    run under the foil (where it would be invisible and could interfere with
    solder wetting), so we inset by the foil overlap alone and DROP the safety
    margin the front art carries. The result is a wider window than the front
    keep-out (``back_window_um <= keepout_um`` always, since safety >= 0).
    """
    return overlap_um(foil, glass_thickness_um)


def face_cut_dims(
    face_id: str,
    width_um: float,
    depth_um: float,
    height_um: float,
    glass_thickness_um: float,
) -> tuple[float, float]:
    """Cut dimensions (local width x height, µm) for one face.

    Convention: walls sit ON the bottom plate, the lid rests on the wall
    rim — so walls lose a glass thickness top AND bottom, and the left/right
    walls fit BETWEEN the front/back walls (losing a thickness each side).

        bottom : W x D          top/lid: W x D
        front  : W x (H - 2t)   back   : W x (H - 2t)
        left   : (D - 2t) x (H - 2t)   right: same
    """
    t = glass_thickness_um
    if face_id in ("bottom", "top"):
        return width_um, depth_um
    if face_id in ("front", "back"):
        return width_um, height_um - 2.0 * t
    if face_id in ("left", "right"):
        return depth_um - 2.0 * t, height_um - 2.0 * t
    raise ValueError(f"Unknown face id: {face_id!r} (expected one of {FACE_IDS})")


def cut_list(
    width_um: float,
    depth_um: float,
    height_um: float,
    glass_thickness_um: float,
) -> list[dict[str, Any]]:
    """Full 6-plate cut list, ordered bottom/top/front/back/left/right."""
    out: list[dict[str, Any]] = []
    for fid in ("bottom", "top", "front", "back", "left", "right"):
        w, h = face_cut_dims(fid, width_um, depth_um, height_um, glass_thickness_um)
        out.append(
            {
                "face": fid,
                "width_um": w,
                "height_um": h,
                # Shared rounding rule — see round_mm3 / assembly.ts::roundMm3.
                "width_mm": round_mm3(w),
                "height_mm": round_mm3(h),
            }
        )
    return out


# -----------------------------------------------------------------------------
# BONDED (two-ply) construction — thick-stock variant of the box.
#
# With thick sheet stock (e.g. 1.5 mm soda lime) each face is TWO single-side
# chrome plates glued face-to-face (UV optical adhesive): the OUTER ply carries
# the front layer at the bond line, the INNER ply carries the back layer toward
# the box interior, and the optical parallax gap is ONE ply of glass. The wall
# the box is built from is the bonded stack, 2 x ply.
#
# Geometrically the box is TWO NESTED TIFFANY BOXES, each cut at the PLY
# thickness: the outer shell at (W, D, H) and the inner shell at
# (W - 2p, D - 2p, H - 2p), shifted inward by p on every axis. That single rule
# yields, per face, an inner ply inset exactly p on all four edges (centered on
# its outer ply — the bond alignment is trivially "centered"), corners that
# interleave as a two-step staircase approximating a 45 deg miter with no ply
# collisions, and a lid whose inner ply nests INSIDE the wall rim — a
# registration lip that keys the closed lid in place.
#
# Foil: the tape wraps a STEPPED edge — across the outer ply edge (p), the
# exposed step ledge (p), and the inner ply edge (p) — consuming 3p of width
# before any fold-over remains:
#
#     bonded_overlap_um = max(0, (tape_width_um - 3p) / 2)
#
# The fold lands on the outer face measured from the OUTER ply edge, and on
# the interior face measured from the INNER ply edge.
# -----------------------------------------------------------------------------


def bonded_wall_um(ply_um: float) -> float:
    """Effective wall thickness of a bonded two-ply face."""
    return 2.0 * ply_um


def bonded_overlap_um(foil: FoilSpec, ply_um: float) -> float:
    """Foil fold-over per face of the bonded stack (stepped-edge wrap)."""
    return max(0.0, (foil.tape_width_um - 3.0 * ply_um) / 2.0)


def bonded_keepout_um(foil: FoilSpec, ply_um: float) -> float:
    """Front-art keep-out rim, measured from the OUTER ply edge."""
    return bonded_overlap_um(foil, ply_um) + foil.safety_um


def bonded_back_window_um(foil: FoilSpec, ply_um: float) -> float:
    """Back-layer keep-out rim, measured from the INNER ply edge.

    The interior foil fold starts at the inner ply edge (the fold conforms over
    the staircase), so — like the single-ply ``back_window_um`` — the back
    carrier insets by the fold alone, no safety margin.
    """
    return bonded_overlap_um(foil, ply_um)


def bonded_art_keepout_um(foil: FoilSpec, ply_um: float) -> float:
    """FRONT-art rim of a bonded face, measured from the OUTER ply edge.

    The larger of two rims: the foil keep-out (tape fold + safety) and the
    inner ply's carrier window seen in the outer frame (one ply of ledge plus
    the interior fold). Front art drawn between those two — over the ledge
    and the interior fold — has copper tape behind it and no back grating to
    beat against, so a garland there prints as leaves on tape rather than as a
    moiré. Starting the art at the back window keeps every front feature over
    the inner ply's carrier. Single-ply faces use the same rim so all six
    borders of a box start on one line.
    """
    return max(bonded_keepout_um(foil, ply_um),
               ply_um + bonded_back_window_um(foil, ply_um))


def bonded_face_cut_dims(
    face_id: str,
    width_um: float,
    depth_um: float,
    height_um: float,
    ply_um: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """((outer_w, outer_h), (inner_w, inner_h)) for one bonded face.

    Outer ply = the outer nested shell cut at the PLY thickness; inner ply =
    the shell nested one ply in on every axis. The inner ply always comes out
    exactly ``2 * ply`` smaller on both sides — inset one ply per edge.
    """
    p = ply_um
    outer = face_cut_dims(face_id, width_um, depth_um, height_um, p)
    inner = face_cut_dims(
        face_id, width_um - 2.0 * p, depth_um - 2.0 * p, height_um - 2.0 * p, p
    )
    return outer, inner


def bonded_cut_list(
    width_um: float,
    depth_um: float,
    height_um: float,
    ply_um: float,
) -> list[dict[str, Any]]:
    """Full 12-plate bonded cut list: per face, outer ply then inner ply.

    ``inset_um`` on the inner entry is the per-edge inset relative to its outer
    ply (always one ply) — the bench bond alignment target is "centered".
    """
    out: list[dict[str, Any]] = []
    for fid in ("bottom", "top", "front", "back", "left", "right"):
        (ow, oh), (iw, ih) = bonded_face_cut_dims(
            fid, width_um, depth_um, height_um, ply_um
        )
        out.append(
            {
                "face": fid,
                "ply": "outer",
                "width_um": ow,
                "height_um": oh,
                "width_mm": round_mm3(ow),
                "height_mm": round_mm3(oh),
            }
        )
        out.append(
            {
                "face": fid,
                "ply": "inner",
                "width_um": iw,
                "height_um": ih,
                "width_mm": round_mm3(iw),
                "height_mm": round_mm3(ih),
                "inset_um": ply_um,
            }
        )
    return out


def validate_bonded_assembly(
    width_um: float,
    depth_um: float,
    height_um: float,
    ply_um: float,
    foil: FoilSpec,
    hinge: HingeSpec,
) -> None:
    """Bonded-construction validation — the two-ply sibling of
    ``validate_assembly``. Raises ``ValueError`` with actionable messages.

    Checks the NESTED geometry (inner shell must exist: H > 4p, D > 4p), the
    foil sanity shared with the single-ply path, the patternable aperture on
    BOTH plies (front art on the outer ply behind ``bonded_keepout_um``, back
    art on the inner ply behind ``bonded_back_window_um``), and the hinge.
    Predicates keep the exact arrangement of ``validate_assembly`` (see the
    ULP note there) — the frontend mirrors them term for term.
    """
    p = ply_um
    if p <= 0:
        raise ValueError(f"Ply thickness must be positive (got {p} um).")
    if width_um <= 0 or depth_um <= 0 or height_um <= 0:
        raise ValueError(
            f"Box dimensions must be positive (got W={width_um}, D={depth_um}, "
            f"H={height_um} um)."
        )
    if height_um <= 4.0 * p:
        raise ValueError(
            f"Box height {height_um / 1000:.1f} mm leaves no room for the bonded "
            f"walls: the INNER wall ply is H - 4p = "
            f"{(height_um - 4 * p) / 1000:.1f} mm tall with {p / 1000:.1f} mm "
            "plies. Increase height or use thinner stock."
        )
    if depth_um <= 4.0 * p:
        raise ValueError(
            f"Box depth {depth_um / 1000:.1f} mm leaves no room for the bonded "
            f"left/right walls: the INNER ply is D - 4p = "
            f"{(depth_um - 4 * p) / 1000:.1f} mm wide with {p / 1000:.1f} mm "
            "plies. Increase depth or use thinner stock."
        )
    if foil.tape_width_um <= 0:
        raise ValueError(f"Foil tape width must be positive (got {foil.tape_width_um} um).")
    if foil.safety_um < 0:
        raise ValueError(f"Foil safety margin cannot be negative (got {foil.safety_um} um).")
    if foil.tape_width_um < 3.0 * p:
        raise ValueError(
            f"Foil tape {foil.tape_width_um / 1000:.2f} mm is narrower than the "
            f"{3.0 * p / 1000:.2f} mm stepped edge it must wrap (three plies of "
            f"{p / 1000:.2f} mm): nothing folds onto either face, so the seam has "
            "no lap and the hinge nothing to solder to. Use wider tape "
            "(3/8\" for 2.25 mm plies) or thinner stock."
        )

    ko = bonded_art_keepout_um(foil, p)
    bw = bonded_back_window_um(foil, p)
    for entry in bonded_cut_list(width_um, depth_um, height_um, p):
        rim = ko if entry["ply"] == "outer" else bw
        min_side = min(entry["width_um"], entry["height_um"])
        if min_side <= 2.0 * rim + MIN_APERTURE_UM:
            raise ValueError(
                f"Foil keep-out swallows the {entry['face']} {entry['ply']} ply: "
                f"the ply is {entry['width_mm']} x {entry['height_mm']} mm but its "
                f"keep-out rim is {rim / 1000:.2f} mm per edge, leaving under "
                f"{MIN_APERTURE_UM / 1000:.0f} mm of patternable aperture. Use "
                "narrower foil tape, thinner stock, or enlarge the box."
            )

    if hinge.segments < 3 or hinge.segments % 2 == 0:
        raise ValueError(
            f"Hinge segments must be an odd number >= 3 (got {hinge.segments}): "
            "the tube alternates body,lid,body,... with body at both ends."
        )
    if not (0.0 < hinge.coverage <= 1.0):
        raise ValueError(
            f"Hinge coverage must be in (0, 1] of the box width (got {hinge.coverage})."
        )
    layout = hinge_layout(hinge, width_um)
    if layout["segment_length_um"] <= hinge.tube_od_um:
        raise ValueError(
            f"Hinge tube segments come out {layout['segment_length_um'] / 1000:.2f} mm "
            f"long — shorter than the tube OD ({hinge.tube_od_um / 1000:.2f} mm) and "
            "uncuttable. Reduce the segment count or increase hinge coverage."
        )
    if hinge.rod_od_um >= hinge.tube_od_um:
        raise ValueError(
            f"Hinge rod OD ({hinge.rod_od_um} um) must be smaller than the tube OD "
            f"({hinge.tube_od_um} um) so the rod can pass through the tube."
        )


def bonded_assembly_summary(
    width_um: float,
    depth_um: float,
    height_um: float,
    ply_um: float,
    foil: FoilSpec,
    hinge: HingeSpec,
) -> dict[str, Any]:
    """The ``assembly`` block of a bonded box manifest.

    Seams follow the OUTER shell at the ply thickness: the visible bottom
    joint line sits one ply up, and the corner seams span the outer wall ply
    (H - 2p) — the inner plies are hidden behind them.
    """
    return {
        "construction": "bonded",
        "ply_um": ply_um,
        "wall_um": bonded_wall_um(ply_um),
        "keepout_um": bonded_keepout_um(foil, ply_um),
        "art_keepout_um": bonded_art_keepout_um(foil, ply_um),
        "back_window_um": bonded_back_window_um(foil, ply_um),
        "overlap_um": bonded_overlap_um(foil, ply_um),
        "cut_list": bonded_cut_list(width_um, depth_um, height_um, ply_um),
        "seams": seam_list(width_um, depth_um, height_um, ply_um),
        "hinge": {
            **hinge.to_dict(),
            **hinge_layout(hinge, width_um),
        },
    }


def hinge_layout(hinge: HingeSpec, width_um: float) -> dict[str, Any]:
    """Tube run + segment lengths for the back-edge hinge.

    The tube run spans ``coverage * W`` centered on the box, cut into
    ``segments`` pieces separated by ``HINGE_SEGMENT_GAP_UM`` clearance gaps.
    Segments alternate body, lid, body, … with body at both ends (hence odd
    count). The rod is cut a little long so the ends can be peened/capped.
    """
    run_length = hinge.coverage * width_um
    gaps_total = (hinge.segments - 1) * HINGE_SEGMENT_GAP_UM
    segment_length = (run_length - gaps_total) / hinge.segments
    return {
        "run_length_um": run_length,
        "segment_length_um": segment_length,
        "gap_um": HINGE_SEGMENT_GAP_UM,
        "rod_length_um": run_length + 2.0 * hinge.rod_od_um,
        "body_segments": (hinge.segments + 1) // 2,
        "lid_segments": hinge.segments // 2,
    }


def seam_list(
    width_um: float,
    depth_um: float,
    height_um: float,
    glass_thickness_um: float,
) -> list[dict[str, Any]]:
    """The 8 soldered seams of the body (the lid is never soldered).

    4 run around the bottom perimeter where the walls land on the bottom
    plate (y = -H/2 + t), 4 run up the vertical corners where adjacent
    walls meet. Lengths are the joint lines the solder bead must cover.
    """
    t = glass_thickness_um
    corner_len = height_um - 2.0 * t
    # The left/right walls sit BETWEEN the front and back walls (face_cut_dims:
    # D - 2t wide), so their bottom seams are D - 2t long; the front/back walls
    # span the full width. The corners at the ends belong to the corner seams.
    side_len = depth_um - 2.0 * t
    return [
        {"id": "bottom-front", "kind": "bottom", "length_um": width_um},
        {"id": "bottom-back", "kind": "bottom", "length_um": width_um},
        {"id": "bottom-left", "kind": "bottom", "length_um": side_len},
        {"id": "bottom-right", "kind": "bottom", "length_um": side_len},
        {"id": "corner-front-left", "kind": "corner", "length_um": corner_len},
        {"id": "corner-front-right", "kind": "corner", "length_um": corner_len},
        {"id": "corner-back-left", "kind": "corner", "length_um": corner_len},
        {"id": "corner-back-right", "kind": "corner", "length_um": corner_len},
    ]


def validate_assembly(
    width_um: float,
    depth_um: float,
    height_um: float,
    glass_thickness_um: float,
    foil: FoilSpec,
    hinge: HingeSpec,
) -> None:
    """Reject geometrically impossible / unfabbable boxes with clear messages.

    Raises ``ValueError`` describing exactly which constraint failed and what
    to change — these strings surface verbatim in the API 400 response.
    """
    t = glass_thickness_um
    if t <= 0:
        raise ValueError(f"Glass thickness must be positive (got {t} um).")
    if width_um <= 0 or depth_um <= 0 or height_um <= 0:
        raise ValueError(
            f"Box dimensions must be positive (got W={width_um}, D={depth_um}, "
            f"H={height_um} um)."
        )
    if height_um <= 2.0 * t:
        raise ValueError(
            f"Box height {height_um / 1000:.1f} mm leaves no room for walls: walls "
            f"are H - 2t = {(height_um - 2 * t) / 1000:.1f} mm tall with "
            f"{t / 1000:.1f} mm glass. Increase height or use thinner glass."
        )
    if depth_um <= 2.0 * t:
        raise ValueError(
            f"Box depth {depth_um / 1000:.1f} mm leaves no room for the left/right "
            f"walls between front and back ({(depth_um - 2 * t) / 1000:.1f} mm). "
            "Increase depth or use thinner glass."
        )
    if foil.tape_width_um <= 0:
        raise ValueError(f"Foil tape width must be positive (got {foil.tape_width_um} um).")
    if foil.safety_um < 0:
        raise ValueError(f"Foil safety margin cannot be negative (got {foil.safety_um} um).")

    # Keep-out must leave a patternable aperture on every plate.
    #
    # The predicate below is arranged EXACTLY as assembly.ts::validateBox
    # writes it (`min_side <= 2.0 * ko + MIN_APERTURE_UM`, never the
    # algebraically equal `min_side - 2.0 * ko <= MIN_APERTURE_UM`). With a
    # keep-out that is not binary-representable — any fractional tape width or
    # glass thickness — the two arrangements differ by an ULP right at the
    # boundary, which is exactly where a spec the frontend accepts would 400
    # here. The mm figure quoted in the frontend's message is a separate
    # display-only computation for that reason.
    ko = keepout_um(foil, t)
    for entry in cut_list(width_um, depth_um, height_um, t):
        min_side = min(entry["width_um"], entry["height_um"])
        if min_side <= 2.0 * ko + MIN_APERTURE_UM:
            raise ValueError(
                f"Foil keep-out swallows the {entry['face']} plate: the plate is "
                f"{entry['width_mm']} x {entry['height_mm']} mm but the keep-out rim is "
                f"{ko / 1000:.2f} mm per edge ({2 * ko / 1000:.2f} mm total), leaving "
                f"under {MIN_APERTURE_UM / 1000:.0f} mm of patternable aperture. Use "
                "narrower foil tape, reduce the safety margin, or enlarge the box."
            )

    # Hinge layout must be physically cuttable.
    if hinge.segments < 3 or hinge.segments % 2 == 0:
        raise ValueError(
            f"Hinge segments must be an odd number >= 3 (got {hinge.segments}): "
            "the tube alternates body,lid,body,... with body at both ends."
        )
    if not (0.0 < hinge.coverage <= 1.0):
        raise ValueError(
            f"Hinge coverage must be in (0, 1] of the box width (got {hinge.coverage})."
        )
    layout = hinge_layout(hinge, width_um)
    if layout["segment_length_um"] <= hinge.tube_od_um:
        raise ValueError(
            f"Hinge tube segments come out {layout['segment_length_um'] / 1000:.2f} mm "
            f"long — shorter than the tube OD ({hinge.tube_od_um / 1000:.2f} mm) and "
            "uncuttable. Reduce the segment count or increase hinge coverage."
        )
    if hinge.rod_od_um >= hinge.tube_od_um:
        raise ValueError(
            f"Hinge rod OD ({hinge.rod_od_um} um) must be smaller than the tube OD "
            f"({hinge.tube_od_um} um) so the rod can pass through the tube."
        )


def assembly_summary(
    width_um: float,
    depth_um: float,
    height_um: float,
    glass_thickness_um: float,
    foil: FoilSpec,
    hinge: HingeSpec,
) -> dict[str, Any]:
    """The ``assembly`` block of the box manifest — all derived geometry."""
    return {
        "keepout_um": keepout_um(foil, glass_thickness_um),
        "back_window_um": back_window_um(foil, glass_thickness_um),
        "overlap_um": overlap_um(foil, glass_thickness_um),
        "glass_thickness_um": glass_thickness_um,
        "cut_list": cut_list(width_um, depth_um, height_um, glass_thickness_um),
        "seams": seam_list(width_um, depth_um, height_um, glass_thickness_um),
        "hinge": {
            **hinge.to_dict(),
            **hinge_layout(hinge, width_um),
        },
    }
