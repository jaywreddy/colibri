"""J+P monogram carrier reveal — the honest single-image phase effect.

This is the physically-correct replacement for what the retired
phase_shift_overlay recipe pretended to do, with one crucial difference:
there is only ONE image, and it lives entirely in the FRONT layer against a
uniform, image-free back carrier — so nothing here relies on a front-face
image "vanishing" (which parallax can never do for two-image switches).

Construction (taxonomy type T5, carrier-phase-reveal, anti-phase variant):
FRONT = monogram ∩ vertical carrier at phase 0; BACK = full-field uniform
carrier at the SAME period in exact anti-phase (complementary columns).
Head-on the two carriers interlock inside the figure — zero transmission —
so the monogram reads as solid dark on the ~50% grey ground. Tilting slides
the back carrier under the front; at a half-period shift the back stripes
hide exactly behind the front stripes, the figure transmits like the ground,
and the monogram dissolves. Contrast oscillates with period p of shift and
is symmetric in tilt sign (magnitude-only effect). Verified numbers from the
2D study: contrast |T_in − T_out| = 0.50 at s = 0 → 0.00 at s = p/2.

Fully honest under the moire_interactive recipe — pure front × parallax-
shifted back sampling reproduces it with no view-sign bias.
"""
from __future__ import annotations

import math

import numpy as np

from .._helpers import check_lattice_budget, exterior_tilt_deg, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register
from ..motifs import monogram


@register
class MonogramCarrierReveal(Pattern):
    slug = "monogram-carrier-reveal"
    name = "J+P carrier reveal"
    description = (
        "The J + P monogram halftoned onto a fine vertical carrier on the "
        "front layer; the back layer is a full-field uniform carrier at the "
        "same period in exact anti-phase. Head-on, the two carriers "
        "interlock inside the glyphs — the monogram reads solid dark on a "
        "grey ground. Snell-refracted parallax slides the back carrier into "
        "registration beneath the front stripes, so the monogram dissolves "
        "into the ground at a half-period shift and re-darkens with every "
        "full period — a breathing appear/disappear that is honest 2D "
        "physics (no view-sign tricks), symmetric in tilt direction."
    )
    tags = ["moire", "carrier", "tilt-reveal", "monogram", "Global Travel"]
    tier = 1
    theme = "Global Travel"
    render_recipe = "moire_interactive"
    params = [
        ParamSpec("period_um", "Carrier period", "float", 40.0, 6.0, 80.0, 0.5, "μm"),
        ParamSpec("duty", "Carrier duty", "float", 0.5, 0.2, 0.8, 0.05),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
    ]

    @classmethod
    def generate(
        cls,
        period_um: float = 40.0,
        duty: float = 0.5,
        extent_um: float = 2000.0,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Resolve the carrier with ≥8 samples per period, capped at 1200 so
        # the raster grid itself stays cheap (1.4M bool cells max).
        n_grid = min(1200, max(384, int(extent_um / max(1.0, period_um / 10))))

        # Rect-count estimate: the full-field back carrier emits one rectangle
        # per row per stripe (n_grid × extent/period), which dominates the
        # figure-gated front. Gate it before building anything.
        n_stripes = int(math.ceil(extent_um / period_um))
        check_lattice_budget(
            n_grid * n_stripes,
            "monogram-carrier-reveal back carrier",
            period_um=period_um,
            extent_um=extent_um,
        )

        cell_um = extent_um / n_grid
        mono = monogram.jp_monogram_silhouette(extent, n_grid=n_grid)

        # Front and back carriers are built on the SAME column grid, and the
        # back phase mask is the exact boolean complement of the front's —
        # this is what makes the anti-phase relation exact regardless of
        # whether extent/period is commensurate (an analytic half-period
        # translate would drift out of phase against the quantized front).
        # The reveal contrast depends on this interlock being gap-free.
        period_pix = period_um / cell_um
        cols = np.arange(n_grid, dtype=np.float32)
        phase_front = (cols % period_pix) < (duty * period_pix)
        phase_back = ~phase_front

        front_mask = mono & phase_front[None, :]
        back_mask = np.broadcast_to(phase_back[None, :], (n_grid, n_grid))

        # Raster-space intersection only (silhouette grid & per-column phase
        # mask) — never a GEOS boolean. raster_to_polygons output is fill-only
        # concatenation; do not feed it to GEOS set-ops downstream.
        front = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back = raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)

        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=period_um * min(duty, 1 - duty),
            extra={
                "carrier_period_um": period_um,
                # Monogram fully dissolved into the ground at θ(p/2); dark
                # again (re-interlocked) at θ(p). Sign-symmetric.
                "vanish_angle_deg": exterior_tilt_deg(period_um / 2),
                "cycle_angle_deg": exterior_tilt_deg(period_um),
            },
            # The period rides in recipe_data (not just extra): the Pattern
            # Lab's zone quick-sets and the catalog contract read it there.
            recipe_data={
                "carrier_period_um": period_um,
                "switch_axis_deg": 0.0,
            },
        )
