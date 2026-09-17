"""Regenerate tools/fixtures/assembly_golden.json from the BACKEND assembly
math — the source of truth for the backend<->frontend contract.

The fixture is consumed by BOTH test suites:
  - backend/tests/test_assembly.py::test_golden_fixture_matches_backend
  - frontend/tests/unit/assemblyGolden.test.ts

so any formula drift between app/assembly.py and src/assembly.ts fails one
side's tests instead of silently diverging. Regenerate ONLY when the contract
itself intentionally changes, and re-run both suites afterwards:

    uv run --directory backend python ../tools/dev/gen_assembly_golden.py

CASE-SELECTION RULES (read before adding or editing a case)

The fixture is only as strong as the parameter region it samples, and a case
that comes out INVALID contributes nothing but a validity bit — none of its
numbers are ever compared. So:

  * Keep at least one VALID case per numeric region you care about. When you
    change a case's hinge/foil numbers, re-check that it is still valid; a
    case named for the region it samples (e.g. "small-box-narrow-tape") that
    silently flips to invalid stops testing that region.
  * Ride the boundaries in PAIRS — one case just inside, one exactly on the
    ``<=`` edge — for every accept/reject threshold: the aperture floor, the
    hinge segment vs tube OD, height/depth vs 2t. Exact-equality cases are the
    only thing that pins the two sides' comparison operators and expression
    arrangement together.
  * Cover every validation branch at least once, including the degenerate ones
    (non-positive glass/dims/tape, negative safety, coverage outside (0, 1],
    rod OD >= tube OD) and the ``max(0, ...)`` overlap clamp (tape narrower
    than the glass).
  * The BONDED (two-ply) cases went with the construction on 2026-09-16 — the
    box is six single butt-jointed plies and ``assembly.py`` has no
    ``bonded_*`` math left to pin. What the bonded build decided survives as
    pinned constants (``production.ART_RIM_UM``, ``ID_TICK_OFFSET_UM``), not
    as formulas.
  * Include at least one case whose mm value lands exactly on a
    half-thousandth (a cut side ending in .5 um) — that is the only thing
    pinning ``assembly.py::round_mm3`` to ``assembly.ts::roundMm3``, and the
    frontend compares mm with exact equality (``toBe``).
  * Include at least one case whose keep-out is NOT binary-representable
    (fractional tape width) near the aperture boundary, so the two sides'
    floating-point arrangements are exercised rather than assumed.

Keep the list reviewable: the invalid cases cost ~15 lines of JSON each, the
valid ones ~70, so add valid cases deliberately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.assembly import (  # noqa: E402
    FoilSpec,
    HingeSpec,
    back_window_um,
    cut_list,
    hinge_layout,
    keepout_um,
    overlap_um,
    seam_list,
    validate_assembly,
)

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "assembly_golden.json"


def case(
    name: str,
    width_um: float,
    depth_um: float,
    height_um: float,
    glass_thickness_um: float,
    foil: FoilSpec,
    hinge: HingeSpec,
) -> dict:
    try:
        validate_assembly(width_um, depth_um, height_um, glass_thickness_um, foil, hinge)
        valid = True
        error = None
    except ValueError as e:
        valid = False
        error = str(e)

    expected: dict = {"valid": valid}
    if error is not None:
        expected["backend_error"] = error
    if valid:
        expected.update(
            {
                "overlap_um": overlap_um(foil, glass_thickness_um),
                "keepout_um": keepout_um(foil, glass_thickness_um),
                # Back-carrier keep-out rim (foil overlap only, no safety) —
                # the newest cross-suite formula; pinned by both
                # backend/tests/test_assembly.py and
                # frontend/tests/unit/assemblyGolden.test.ts.
                "back_window_um": back_window_um(foil, glass_thickness_um),
                "cut_list": cut_list(width_um, depth_um, height_um, glass_thickness_um),
                "seams": {
                    s["id"]: s["length_um"]
                    for s in seam_list(width_um, depth_um, height_um, glass_thickness_um)
                },
                "hinge": hinge_layout(hinge, width_um),
            }
        )
    return {
        "name": name,
        "spec": {
            "width_um": width_um,
            "depth_um": depth_um,
            "height_um": height_um,
            "glass_thickness_um": glass_thickness_um,
            "foil": foil.to_dict(),
            "hinge": hinge.to_dict(),
        },
        "expected": expected,
    }


def main() -> None:
    cases = [
        # -- VALID: numeric regions ------------------------------------------
        # The shipping default. overlap 2925, keep-out 3425, walls 39.0 mm.
        case(
            "default-ring-box",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(),
        ),
        # Narrow (3/16") tape + reduced safety: the ONLY case carrying
        # overlap 2131.5 / keep-out 2431.5. segments=3 (not 7 — at 7 the
        # 5.73 mm segments dropped to 2.23 mm, under the 2.4 mm tube OD, and
        # this case silently stopped comparing its foil numbers at all).
        case(
            "small-box-narrow-tape",
            30000.0, 20000.0, 25000.0, 500.0,
            FoilSpec(tape_width_um=4763.0, safety_um=300.0),
            HingeSpec(segments=3, coverage=0.6),
        ),
        # 7/32" tape on thin glass; non-round mm cut sides (11.4 / 17.4).
        case(
            "thin-glass-mid-tape",
            24000.0, 18000.0, 12000.0, 300.0,
            FoilSpec(tape_width_um=5556.0, safety_um=400.0, bead_um=1500.0, finish="patina"),
            HingeSpec(tube_od_um=2000.0, rod_od_um=1200.0, segments=3, coverage=0.7),
        ),
        # Overlap clamp: tape (1.0 mm) NARROWER than the glass (2.0 mm), so
        # (tape - t) / 2 goes negative and max(0, ...) must floor it — overlap
        # 0, back_window 0, keep-out = safety alone.
        case(
            "tape-narrower-than-glass-clamped",
            40000.0, 30000.0, 20000.0, 2000.0,
            FoilSpec(tape_width_um=1000.0, safety_um=500.0),
            HingeSpec(),
        ),
        # mm rounding tie: t = 468.75 um puts the walls at 24062.5 um and the
        # left/right width at 49062.5 um — exactly on a half-thousandth of a
        # millimetre. Python's round() would emit 24.062 / 49.062 (half-even)
        # and JS's Math.round 24.063 / 49.063 (half-up); the shared
        # round_mm3 / roundMm3 rule makes both sides say 24.063 / 49.063.
        case(
            "mm-tie-half-thousandth",
            50000.0, 50000.0, 25000.0, 468.75,
            FoilSpec(),
            HingeSpec(),
        ),
        # Aperture boundary, INSIDE by 0.9 um, with a keep-out that is NOT
        # binary-representable (tape 6350.1 -> ko 3425.0499999999997). The
        # left/right plates are 9851 um wide against a 9850.0999... um
        # threshold, so the two sides' floating-point arrangement of
        # `min_side <= 2 * ko + MIN_APERTURE` is genuinely exercised.
        case(
            "aperture-just-above-minimum-fractional-keepout",
            50000.0, 10851.0, 21000.0, 500.0,
            FoilSpec(tape_width_um=6350.1, safety_um=500.0),
            HingeSpec(),
        ),
        # Hinge segment boundary, INSIDE by 0.5 um: coverage 0.25 of 50 mm
        # gives 12.5 mm of run and 2180 um segments against a 2179.5 um tube
        # OD. Pairs with invalid-hinge-segment-exactly-tube-od below.
        case(
            "hinge-segment-just-above-tube-od",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(tube_od_um=2179.5, rod_od_um=1600.0, segments=5, coverage=0.25),
        ),
        # -- INVALID: one per validation branch ------------------------------
        # Both validators must reject every case below. Only the validity bit
        # is compared cross-side (the message strings are backend-only), so
        # these are cheap — add one for every branch.
        #
        # Aperture floor hit EXACTLY: left/right width = D - 2t = 9850 um and
        # 2 * ko + MIN_APERTURE_UM = 2 * 3425 + 3000 = 9850, so the `<=` edge
        # itself must reject on both sides (a `<` on either side fails here).
        case(
            "invalid-aperture-exactly-at-minimum",
            50000.0, 10850.0, 21000.0, 500.0,
            FoilSpec(),
            HingeSpec(),
        ),
        # Hinge segment EXACTLY the tube OD: 2180 um segments, 2180 um tube.
        case(
            "invalid-hinge-segment-exactly-tube-od",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(tube_od_um=2180.0, rod_od_um=1600.0, segments=5, coverage=0.25),
        ),
        # Height EXACTLY 2t — zero-height walls (H - 2t = 0).
        case(
            "invalid-height-equals-2t",
            50000.0, 50000.0, 1000.0, 500.0,
            FoilSpec(),
            HingeSpec(),
        ),
        # Depth EXACTLY 2t — zero-width left/right walls (D - 2t = 0).
        case(
            "invalid-depth-equals-2t",
            50000.0, 1000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(),
        ),
        # Aperture swallowed with room to spare (9 mm cube).
        case(
            "invalid-tiny-box",
            9000.0, 9000.0, 9000.0, 500.0,
            FoilSpec(),
            HingeSpec(),
        ),
        case(
            "invalid-even-hinge-segments",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(segments=4),
        ),
        # Coverage outside (0, 1] — both ends of the interval.
        case(
            "invalid-hinge-coverage-zero",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(coverage=0.0),
        ),
        case(
            "invalid-hinge-coverage-above-one",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(coverage=1.5),
        ),
        # Rod EXACTLY the tube OD — the rod cannot pass through.
        case(
            "invalid-rod-od-equals-tube-od",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(),
            HingeSpec(tube_od_um=2400.0, rod_od_um=2400.0),
        ),
        case(
            "invalid-zero-glass",
            50000.0, 50000.0, 40000.0, 0.0,
            FoilSpec(),
            HingeSpec(),
        ),
        case(
            "invalid-zero-tape-width",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(tape_width_um=0.0),
            HingeSpec(),
        ),
        case(
            "invalid-negative-safety",
            50000.0, 50000.0, 40000.0, 500.0,
            FoilSpec(safety_um=-100.0),
            HingeSpec(),
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "comment": (
                    "GENERATED by tools/dev/gen_assembly_golden.py from "
                    "backend/app/assembly.py. Consumed by backend AND frontend "
                    "tests to pin the assembly contract. Do not hand-edit."
                ),
                "cases": cases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
