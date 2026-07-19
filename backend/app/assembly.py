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

# Common copper foil tape widths (µm): 3/16", 7/32", 1/4".
FOIL_TAPE_PRESETS_UM = (4763.0, 5556.0, 6350.0)


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
                "width_mm": round(w / 1000.0, 3),
                "height_mm": round(h / 1000.0, 3),
            }
        )
    return out


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
    return [
        {"id": "bottom-front", "kind": "bottom", "length_um": width_um},
        {"id": "bottom-back", "kind": "bottom", "length_um": width_um},
        {"id": "bottom-left", "kind": "bottom", "length_um": depth_um},
        {"id": "bottom-right", "kind": "bottom", "length_um": depth_um},
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
