from __future__ import annotations

import math

import numpy as np

from .._helpers import check_lattice_budget, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..effects.gratings import beat_delta_um, shimmer_moire_layers
from ..motifs import monogram


def _resolve_grid(period_um: float, extent_um: float) -> tuple[int, float]:
    """``(n_grid, cell_um)`` for the carrier raster.

    ≥8 samples/period so the carrier's half-period phase offset is clean.
    Module-level because both ``generate`` and ``pixel_pitch_um`` need it and the
    plate compositor reads the pitch WITHOUT generating — one expression, so they
    cannot disagree.
    """
    n_grid = max(384, int(extent_um / max(1.0, period_um / 10)))
    return n_grid, extent_um / n_grid


@register
class MonogramJP(Pattern):
    """Interlocked cursive J+P wedding monogram, SHADING MOIRE against the carrier.

    The FRONT face carries the entwined J+P silhouette carved into a stripe
    carrier; the BACK face carries a uniform carrier of its own at a slightly
    COARSER pitch, running parallel. Outside the letters you see the back
    carrier alone (a fine even haze of gold); inside them the two gratings drift
    in and out of phase, so broad bands sweep through the monogram as the lid
    tilts — the letterform goes solid gold, then sinks back into the surrounding
    haze. This is the lid centerpiece: a classic engagement engraving that
    breathes instead of sitting still.

    The back layer used to be empty here, which made the second plate pointless
    on this face. It was not, in fact, missing a grating — inside a box the back
    plate already carries its carrier across the whole exposed face. What was
    missing was any reason for the two to BEAT: the fill ran at a fixed 0 deg
    against a per-face carrier angle, a crossing so steep that the beat came out
    FINER than either grating (18 um, 0.21 arcmin, against the ~2 arcmin the eye
    needs). Running the two parallel and mismatching the PITCH is what makes the
    moire, and taking the beat from pitch rather than from a crossing angle is
    what makes it survive a build with no backside alignment — see
    ``shimmer_moire_layers`` for that argument in full.

    In a box the plate compositor dispatches the silhouette through
    ``plates._centerpiece_masks`` and fills it procedurally at
    ``recipe_data['fab_center_period_um']`` / ``['switch_axis_deg']``, which
    ``plates._carrier_recipe_data`` sets to this same parallel-and-mismatched
    geometry. This standalone ``generate`` bakes it directly so the catalog, the
    tilt collage and the standalone fab path all show the same part.
    """

    slug = "monogram-jp"
    name = "J + P monogram (engagement engraving)"
    description = (
        "An interlocked cursive J and P — the couple's initials entwined the "
        "way an engraver would set them on a signet or a wedding invitation "
        "(Great Vibes copperplate swashes). Carved into a fine gold stripe "
        "carrier so the letters shimmer with a moiré highlight as the lid tilts, "
        "the back face left empty (plain glass; inside a box the plate "
        "compositor puts the uniform back carrier behind it) so the monogram "
        "reads as a front-only glimmer rather than a two-image switch."
    )
    tags = ["monogram", "engagement", "cursive", "tilt-shimmer", "lid"]
    tier = 1
    theme = "Colombia"
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_um", "Carrier period", "float", 22.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        ParamSpec("overlap", "Glyph interlock", "float", 0.68, 0.4, 0.85, 0.01),
        # Band spacing — the ONE knob that sets how bold the fringes are. The
        # pitch mismatch that produces it is solved from this (beat_delta_um),
        # not the other way round, so the look survives a change of carrier
        # pitch. 1.64 mm is about a dozen bands across a 20 mm lid. Coarser is
        # bolder AND more alignment-sensitive; see shimmer_moire_layers.
        ParamSpec("beat_um", "Moiré band spacing", "float", 1635.0, 200.0, 8000.0, 5.0, "μm"),
    ]

    # --- metadata (no geometry) — see Pattern.metadata ----------------------

    @classmethod
    def pixel_pitch_um(
        cls,
        period_um: float = 22.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
        beat_um: float = 1635.0,
    ) -> float:
        return _resolve_grid(period_um, extent_um)[1]

    @classmethod
    def min_feature_um(
        cls,
        period_um: float = 22.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
        beat_um: float = 1635.0,
    ) -> float:
        # Both gratings are 50% duty; the FINER of the two sets the limit.
        return min(period_um, period_um + beat_delta_um(period_um, beat_um)) * 0.5

    @classmethod
    def extra_metadata(
        cls,
        period_um: float = 22.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
        beat_um: float = 1635.0,
    ) -> tuple[dict, dict, tuple[str, ...]]:
        # Now that the back layer carries a real carrier, there IS a mask-level
        # tilt effect, so the periods go into recipe_data as well as extra: the
        # readability gates and the tilt collage both read the fabricated pair
        # from here, and the Pattern Lab tilt quick-sets have something to aim at.
        front_period = period_um + beat_delta_um(period_um, beat_um)
        return (
            {
                "carrier_period_um": period_um,
                "switch_axis_deg": 0.0,
                "beat_period_um": beat_um,
            },
            {
                "fab_back_period_um": period_um,
                "fab_front_period_um": front_period,
                "carrier_angle_deg": 0.0,
                "slit_axis_deg": 0.0,
                "grating_duty": 0.5,
                "beat_period_um": beat_um,
            },
            (),
        )

    @classmethod
    def generate(
        cls,
        period_um: float = 22.0,
        extent_um: float = 2000.0,
        overlap: float = 0.68,
        beat_um: float = 1635.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        n_grid, cell_um = _resolve_grid(period_um, extent_um)

        # Rect-count estimate: raster_to_polygons emits one rect per horizontal
        # run, and the carrier chops every row into ~extent/period runs. BOTH
        # layers are gratings now — the back one spans the full field, so it is
        # the worst case, not the glyph-limited front. Count the pair.
        delta = beat_delta_um(period_um, beat_um)
        n_stripes = int(math.ceil(extent_um / min(period_um, period_um + delta)))
        check_lattice_budget(
            2 * n_grid * n_stripes,
            "monogram-jp carrier pair",
            period_um=period_um,
            beat_um=beat_um,
            extent_um=extent_um,
        )

        mono = monogram.monogram_silhouette(extent, n_grid=n_grid, overlap=overlap)

        # FRONT: the glyph filled at period + delta. BACK: the uniform carrier
        # at period, spanning the whole field exactly as the back plate does
        # inside a box. Parallel, so the beat is pure pitch mismatch.
        front_mask, back_mask = shimmer_moire_layers(
            mono,
            back_period_um=period_um,
            delta_um=delta,
            cell_um=cell_um,
        )

        front_poly = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back_poly = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        # Metadata comes from the accessors above so the numbers a plate reads
        # without generating are the ones a full generate publishes.
        kw = dict(
            period_um=period_um,
            extent_um=extent_um,
            overlap=overlap,
            beat_um=beat_um,
        )
        extra, recipe_data, _layer_names = cls.extra_metadata(**kw)
        return GeneratedPattern(
            front=front_poly,
            back=back_poly,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=cls.min_feature_um(**kw),
            extra=extra,
            recipe_data=recipe_data,
        )
